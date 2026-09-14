from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

import numpy as np
from tqdm import tqdm


def _fingerprint(sample_ids: List[str], backbone, batch_size: int) -> str:
    payload = {
        "sample_ids": sample_ids,
        "backbone": backbone.name,
        "image_size": backbone.image_size,
        "model_path": str(backbone.model_path),
        "backbone_class": backbone.__class__.__name__,
        "pe_repo": str(getattr(backbone, "pe_repo", "")),
        "feature_batch_size": int(batch_size),
    }
    if backbone.__class__.__name__ == "OpenClipImageBackbone":
        payload["cache_signature"] = getattr(
            backbone, "cache_signature", "unspecified"
        )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_or_extract(
    backbone,
    samples,
    cache_directory: Path,
    cache_key: str,
    batch_size: int,
    decode_workers: int = 4,
) -> np.ndarray:
    """Extract embeddings with a resumable prefix cache.

    The metadata fingerprint includes the ordered sample identifiers, backbone,
    resolution, and checkpoint path. A partial cache is resumed only when every
    fingerprint field matches.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if decode_workers < 1:
        raise ValueError("decode_workers must be positive")
    cache_directory.mkdir(parents=True, exist_ok=True)
    array_path = cache_directory / f"{cache_key}.npy"
    metadata_path = cache_directory / f"{cache_key}.json"
    sample_ids = [sample.key for sample in samples]
    fingerprint = _fingerprint(sample_ids, backbone, batch_size)
    completed = 0
    cached = None

    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            candidate = np.load(array_path, mmap_mode="r+")
            completed_candidate = int(metadata.get("completed", 0))
            if (
                candidate.shape[0] == len(samples)
                and 0 <= completed_candidate <= len(samples)
            ):
                cached = candidate
                completed = completed_candidate
                if completed == len(samples):
                    print(f"Reuse complete feature cache: {array_path}", flush=True)
                    return cached
            else:
                del candidate

    if getattr(backbone, "model", None) is None:
        backbone.load()
    if cached is None:
        first_end = min(batch_size, len(samples))
        first = backbone.embed([sample.open_image() for sample in samples[:first_end]])
        cached = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), first.shape[1]),
        )
        cached[:first_end] = first.astype(np.float32)
        cached.flush()
        completed = first_end
        _write_metadata(
            metadata_path,
            {
                "fingerprint": fingerprint,
                "shape": list(cached.shape),
                "completed": completed,
            },
        )

    starts = list(range(completed, len(samples), batch_size))
    if starts:
        with ThreadPoolExecutor(max_workers=decode_workers) as executor:
            pending = _submit_images(executor, samples, starts[0], batch_size)
            for position, start in enumerate(
                tqdm(starts, desc=f"{cache_key} features")
            ):
                images = [future.result() for future in pending]
                if position + 1 < len(starts):
                    pending = _submit_images(
                        executor,
                        samples,
                        starts[position + 1],
                        batch_size,
                    )
                end = min(start + batch_size, len(samples))
                cached[start:end] = backbone.embed(images).astype(np.float32)
                cached.flush()
                _write_metadata(
                    metadata_path,
                    {
                        "fingerprint": fingerprint,
                        "shape": list(cached.shape),
                        "completed": end,
                    },
                )
    return cached


def _submit_images(
    executor: ThreadPoolExecutor,
    samples: Sequence,
    start: int,
    batch_size: int,
) -> List[Future]:
    end = min(start + batch_size, len(samples))
    return [executor.submit(samples[index].open_image) for index in range(start, end)]
