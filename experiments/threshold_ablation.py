"""Threshold variants for the two self-collected binary-recognition tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from dataio.datasets import load_self_collected
from dataio.protocols import MICROSCOPY_TARGETS, microscopy_split
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import binary_target_scores
from methods import backbones
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.thresholds import ThresholdResult, evaluate_thresholds


METHODS = ("direct_otsu", "equal_kde_otsu", "weighted_kde_otsu", "oracle")
SHOTS = (2, 3, 5)
SEEDS = tuple(range(10))
PROTOCOL = "single-positive-2-3-5-shot-v1"
SummaryKey = Tuple[str, str, str, str, str]


def result_rows(
    results: List[ThresholdResult],
    **metadata,
) -> List[dict]:
    return [
        {
            **metadata,
            "method": result.method,
            "threshold": result.threshold,
            "gamma": "" if result.gamma is None else result.gamma,
            "f1": 100.0 * result.f1,
        }
        for result in results
    ]


def dataset_fingerprint(samples) -> str:
    """Separate caches/checkpoints when image paths or file metadata change."""
    payload = [
        (sample.key, sample.label, str(sample.path.resolve()),
         sample.path.stat().st_size, sample.path.stat().st_mtime_ns)
        for sample in samples
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def complete_self_units(rows: List[dict], tasks: Sequence[str]) -> Set[Tuple[str, int]]:
    completed = set()
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    for task in tasks:
        for shot in SHOTS:
            unit = [row for row in rows if row.get("task") == task and int(row["shot"]) == shot]
            keys = {(int(row["seed"]), row["method"]) for row in unit}
            if len(unit) == len(expected) and keys == expected:
                completed.add((task, shot))
            elif unit:
                rows[:] = [row for row in rows
                           if not (row.get("task") == task and int(row["shot"]) == shot)]
                print(f"Recompute incomplete checkpoint: {task}, {shot}-shot", flush=True)
    return completed


def run_self_collected(
    rows: List[dict],
    paths,
    cache_root: Path,
    device: str,
    batch_size: int,
    output: Path,
    tasks: Sequence[str] = ("golden_mussel", "Daphnia"),
) -> None:
    samples = load_self_collected(paths.dataset("self_collected"))
    fingerprint = dataset_fingerprint(samples)
    model_path = paths.model("dinov3")
    run_signature = hashlib.sha256(json.dumps({
        "protocol": PROTOCOL, "dataset": fingerprint,
        "model": str(model_path.resolve()), "batch": batch_size,
        "graph": asdict(DEFAULT_GRAPH_SETTINGS), "targets": MICROSCOPY_TARGETS,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    for row in rows:
        if row.get("protocol") != PROTOCOL or row.get("run_signature") != run_signature:
            raise ValueError("Checkpoint belongs to a different dataset/protocol; use a new --output or omit --resume")
        if row.get("suite") != "self_collected" or row.get("task") not in tasks or int(row["shot"]) not in SHOTS:
            raise ValueError("Checkpoint contains rows outside the requested tasks/shot settings")
    # Validate every requested task before loading model weights.
    for task in tasks:
        microscopy_split(samples, task, SEEDS[0], shot=max(SHOTS))
    completed = complete_self_units(rows, tasks)
    if all((task, shot) in completed for task in tasks for shot in SHOTS):
        print("All requested task/shot combinations are complete.", flush=True)
        return
    backbone = backbones.build_backbone(
        "dinov3",
        device,
        model_path=model_path,
        image_size=512,
    )
    features = load_or_extract(
        backbone,
        samples,
        cache_root / ("self_collected_" + fingerprint[:16]),
        "self_collected_dinov3_512",
        batch_size,
    )
    if features.shape[0] != len(samples) or not np.isfinite(features).all():
        raise ValueError("Invalid features for the selected dataset")
    for task in tasks:
        for shot in SHOTS:
            if (task, shot) in completed:
                print(f"Skip completed {task}, {shot}-shot", flush=True)
                continue
            started = time.perf_counter()
            for seed in SEEDS:
                split = microscopy_split(samples, task, seed, shot=shot)
                scores = binary_target_scores(features, split, DEFAULT_GRAPH_SETTINGS)
                rows.extend(
                    result_rows(
                        evaluate_thresholds(scores, split.query_labels),
                        suite="self_collected", task=task,
                        class_pool_size="", repeat="",
                        target_class=MICROSCOPY_TARGETS[task][0],
                        seed=seed, shot=shot,
                        query_count=len(split.query_indices),
                        target_fraction=float(split.query_labels.mean()),
                        support_keys="; ".join(samples[i].key for i in split.support_groups[0]),
                        protocol=PROTOCOL, dataset_fingerprint=fingerprint,
                        run_signature=run_signature,
                    )
                )
                write_rows_atomic(rows, output)
                print(f"  completed {task} shot={shot} seed={seed}", flush=True)
            print(
                f"Completed {task}, {shot}-shot in {time.perf_counter() - started:.1f}s",
                flush=True,
            )


def write_summary(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[SummaryKey, List[float]]
    for row in rows:
        grouped[
            (
                str(row["suite"]),
                str(row["task"]),
                str(row["class_pool_size"]),
                str(row["shot"]),
                str(row["method"]),
            )
        ].append(float(row["f1"]))
    summary = []
    for (suite, task, class_pool_size, shot, method), values in grouped.items():
        array = np.asarray(values)
        summary.append(
            {
                "suite": suite,
                "task": task,
                "class_pool_size": class_pool_size,
                "shot": shot,
                "method": method,
                "aggregation": "mean",
                "f1": float(array.mean()),
                "std": float(array.std()),
                "q25": float(np.quantile(array, 0.25)),
                "q75": float(np.quantile(array, 0.75)),
                "count": len(array),
            }
        )
    write_rows_atomic(summary, output)


def main(task: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    add_config_argument(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    paths = runtime_paths(args.config)
    cache_root = paths.output("cache")
    device = device_name()
    filename = "threshold_ablation.csv" if task is None else task.lower() + ".csv"
    output = output_path(args.output, paths, filename)
    rows = (
        [
            row
            for row in read_rows(output)
            if row.get("suite") == "self_collected"
        ]
        if args.resume
        else []
    )

    run_self_collected(
        rows,
        paths,
        cache_root,
        device,
        args.batch,
        output,
        tasks=(task,) if task is not None else ("golden_mussel", "Daphnia"),
    )

    write_rows_atomic(rows, output)
    write_summary(rows, output.with_name(output.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
