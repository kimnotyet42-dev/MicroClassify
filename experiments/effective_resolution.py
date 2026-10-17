"""EMDS-7 effective-resolution accuracy and grayscale-entropy experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataio.datasets import Sample, load_emds7
from dataio.protocols import emds_allway_splits
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import whitened_graph_z_accuracy_full
from methods import backbones
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.whitening import normalize_rows


RESOLUTIONS = (224, 128, 96, 64, 48, 32)


class EffectiveResolutionSample:
    def __init__(self, sample: Sample, resolution: int):
        self.source = sample
        self.resolution = resolution
        self.key = "{0}@effective-{1}".format(sample.key, resolution)
        self.label = sample.label
        self.path = sample.path

    def open_image(self) -> Image.Image:
        image = self.source.open_image()
        if self.resolution < 224:
            image = image.resize(
                (self.resolution, self.resolution),
                Image.BILINEAR,
            )
        return image


def grayscale_entropy(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"))
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entropy_values_all_resolutions(
    samples: Sequence[Sample],
    resolutions: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Compute every entropy setting from the same complete image collection."""
    normalized_resolutions = [int(value) for value in resolutions]
    values = {resolution: [] for resolution in normalized_resolutions}
    for sample in tqdm(samples, desc="effective-resolution entropy"):
        image = sample.open_image()
        for resolution in normalized_resolutions:
            transformed = image
            if resolution < 224:
                transformed = image.resize(
                    (resolution, resolution),
                    Image.BILINEAR,
                )
            values[resolution].append(grayscale_entropy(transformed))
    return {
        resolution: np.asarray(entropies, dtype=np.float64)
        for resolution, entropies in values.items()
    }


def update_existing_entropy_rows(
    rows: List[dict],
    entropy_by_resolution: Dict[int, np.ndarray],
) -> None:
    """Replace entropy fields without changing stored accuracy results."""
    rows_by_resolution = {
        int(row["effective_resolution"]): row
        for row in rows
    }
    missing = sorted(set(entropy_by_resolution) - set(rows_by_resolution))
    if missing:
        raise ValueError(
            "Cannot update entropy because result rows are missing for: {0}".format(
                ", ".join(str(value) for value in missing)
            )
        )
    for resolution, entropies in entropy_by_resolution.items():
        row = rows_by_resolution[resolution]
        row["entropy_samples"] = len(entropies)
        row["entropy_mean"] = float(np.mean(entropies))
        row["entropy_std"] = float(np.std(entropies))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--entropy-only",
        action="store_true",
        help="Recompute full-dataset entropy while preserving stored accuracies.",
    )
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "effective_resolution.csv")
    samples = load_emds7(paths.dataset("emds7"))
    splits = emds_allway_splits(samples, shot=4, seeds=5)
    rows = read_rows(output) if args.resume else []
    completed = {int(row["effective_resolution"]) for row in rows}

    entropy_by_resolution = entropy_values_all_resolutions(
        samples,
        args.resolutions,
    )
    if args.entropy_only:
        if not rows:
            raise ValueError(
                "--entropy-only requires an existing result file; "
                "also pass --resume to load it"
            )
        update_existing_entropy_rows(rows, entropy_by_resolution)
        write_rows_atomic(rows, output)
        for resolution in args.resolutions:
            entropies = entropy_by_resolution[resolution]
            print(
                "Updated effective resolution {0}px: samples={1}, entropy={2:.4f}".format(
                    resolution,
                    len(entropies),
                    float(np.mean(entropies)),
                ),
                flush=True,
            )
        return

    if rows:
        update_existing_entropy_rows(
            rows,
            {
                resolution: entropies
                for resolution, entropies in entropy_by_resolution.items()
                if resolution in completed
            },
        )
        write_rows_atomic(rows, output)

    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
        image_size=224,
    )
    for resolution in args.resolutions:
        if resolution in completed:
            continue
        transformed = (
            samples
            if resolution == 224
            else [EffectiveResolutionSample(sample, resolution) for sample in samples]
        )
        cache_key = (
            "emds7_dinov3_224"
            if resolution == 224
            else "emds7_dinov3_effective_{0}".format(resolution)
        )
        features = normalize_rows(
            load_or_extract(
                backbone,
                transformed,
                paths.output("cache"),
                cache_key,
                args.batch,
            )
        )
        accuracies = []
        for split in splits:
            accuracies.append(
                100.0 * whitened_graph_z_accuracy_full(
                    features, split, DEFAULT_GRAPH_SETTINGS
                )
            )
        entropies = entropy_by_resolution[resolution]
        rows.append(
            {
                "dataset": "emds7",
                "effective_resolution": resolution,
                "input_resolution": 224,
                "shot": 4,
                "seeds": len(accuracies),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "entropy_samples": len(entropies),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies)),
            }
        )
        write_rows_atomic(rows, output)
        print(
            "Completed effective resolution {0}px: accuracy={1:.2f}, entropy={2:.2f}".format(
                resolution,
                float(np.mean(accuracies)),
                float(np.mean(entropies)),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
