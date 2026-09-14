"""Threshold-only ablation on self-collected data and EMDS-7."""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import sparse

from dataio.datasets import load_emds7, load_self_collected
from dataio.protocols import emds_class_pool, microscopy_split
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import binary_stage_scores
from methods import backbones, graph
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.thresholds import ThresholdResult, evaluate_thresholds
from methods.whitening import normalize_rows


CLASS_POOL_SIZES = (5, 10, 15, 20, 25, 30, 35)
EMDS_SEEDS = (0, 1, 2)
EMDS_SHOTS = (4, 8, 12)
METHODS = ("direct_otsu", "equal_kde_otsu", "weighted_kde_otsu", "oracle")


def propagate_columns(transition, seeds: np.ndarray) -> np.ndarray:
    settings = DEFAULT_GRAPH_SETTINGS
    propagated, _, _ = graph.propagate_from_transition(
        transition,
        seeds,
        graph_alpha=settings.alpha,
        graph_max_iter=settings.max_iterations,
        graph_tol=settings.tolerance,
    )
    return propagated


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


def complete_self_tasks(rows: List[dict]) -> Set[str]:
    completed = set()
    for task in ("golden_mussel", "Daphnia"):
        count = sum(
            row.get("suite") == "self_collected" and row.get("task") == task
            for row in rows
        )
        if count == 10 * len(METHODS):
            completed.add(task)
        elif count:
            rows[:] = [
                row
                for row in rows
                if not (row.get("suite") == "self_collected" and row.get("task") == task)
            ]
            print(f"Discard incomplete checkpoint for self-collected task {task}", flush=True)
    return completed


def run_self_collected(
    rows: List[dict],
    paths,
    cache_root: Path,
    device: str,
    batch_size: int,
    output: Path,
) -> None:
    completed = complete_self_tasks(rows)
    samples = load_self_collected(paths.dataset("self_collected"))
    backbone = backbones.build_backbone(
        "dinov3",
        device,
        model_path=paths.model("dinov3"),
        image_size=512,
    )
    features = load_or_extract(
        backbone,
        samples,
        cache_root,
        "self_collected_dinov3_512",
        batch_size,
    )
    for task in ("golden_mussel", "Daphnia"):
        if task in completed:
            print(f"Skip completed self-collected task {task}", flush=True)
            continue
        started = time.perf_counter()
        task_rows = []
        for seed in range(10):
            split = microscopy_split(samples, task, seed, shot=3)
            scores = binary_stage_scores(features, split, DEFAULT_GRAPH_SETTINGS)
            task_rows.extend(
                result_rows(
                    evaluate_thresholds(scores, split.query_labels),
                    suite="self_collected",
                    task=task,
                    class_pool_size="",
                    repeat="",
                    target_class="",
                    seed=seed,
                    shot=3,
                    query_count=len(split.query_indices),
                    target_fraction=float(split.query_labels.mean()),
                )
            )
            print(f"  completed {task} seed={seed}", flush=True)
        rows.extend(task_rows)
        write_rows_atomic(rows, output)
        print(
            f"Completed {task} in {time.perf_counter() - started:.1f}s",
            flush=True,
        )


def evaluate_payload(payload) -> List[ThresholdResult]:
    return evaluate_thresholds(payload[5], payload[6])


def run_emds7(
    rows: List[dict],
    paths,
    cache_root: Path,
    device: str,
    batch_size: int,
    workers: int,
    output: Path,
) -> None:
    samples = load_emds7(paths.dataset("emds7"))
    backbone = backbones.build_backbone(
        "dinov3",
        device,
        model_path=paths.model("dinov3"),
        image_size=224,
    )
    features = normalize_rows(
        load_or_extract(
            backbone,
            samples,
            cache_root,
            "emds7_dinov3_224",
            batch_size,
        )
    )
    all_labels = np.asarray([sample.label for sample in samples])

    for class_pool_size in CLASS_POOL_SIZES:
        started = time.perf_counter()
        repeats = 1 if class_pool_size == max(CLASS_POOL_SIZES) else 3
        print(
            f"Starting EMDS N={class_pool_size}: repeats={repeats}, workers={workers}",
            flush=True,
        )
        for repeat in range(repeats):
            pool_indices, selected_classes = emds_class_pool(
                samples,
                class_pool_size,
                repeat,
            )
            pool_features = features[pool_indices]
            pool_labels = all_labels[pool_indices]

            combinations_by_shot = {}
            for shot in EMDS_SHOTS:
                combinations = []
                for class_name in selected_classes:
                    positive_indices = np.where(pool_labels == class_name)[0]
                    if len(positive_indices) < shot + 3:
                        continue
                    for seed in EMDS_SEEDS:
                        shuffled = list(positive_indices)
                        np.random.RandomState(seed).shuffle(shuffled)
                        combinations.append(
                            (shot, class_name, seed, np.asarray(shuffled[:shot]))
                        )
                combinations_by_shot[shot] = combinations
            expected = (
                sum(len(values) for values in combinations_by_shot.values())
                * len(METHODS)
            )
            existing = [
                row for row in rows
                if row.get("suite") == "emds7"
                and int(row["class_pool_size"]) == class_pool_size
                and int(row["repeat"]) == repeat
            ]
            if len(existing) == expected:
                print(
                    "  skip completed N={0} repeat={1}/{2}".format(
                        class_pool_size, repeat + 1, repeats
                    ),
                    flush=True,
                )
                continue
            if existing:
                rows[:] = [
                    row for row in rows
                    if not (
                        row.get("suite") == "emds7"
                        and int(row["class_pool_size"]) == class_pool_size
                        and int(row["repeat"]) == repeat
                    )
                ]

            distance = graph.pairwise_distance_matrix(
                pool_features,
                "cos",
                DEFAULT_GRAPH_SETTINGS.epsilon,
            )
            transition = sparse.csr_matrix(
                graph.build_knn_transition(
                    distance,
                    DEFAULT_GRAPH_SETTINGS.neighbors,
                    DEFAULT_GRAPH_SETTINGS.sigma_neighbors,
                    DEFAULT_GRAPH_SETTINGS.sigma_method,
                    DEFAULT_GRAPH_SETTINGS.epsilon,
                    False,
                )
            )

            repeat_rows = []
            for shot in EMDS_SHOTS:
                combinations = combinations_by_shot[shot]
                seed_matrix = np.zeros(
                    (len(pool_indices), len(combinations)),
                    dtype=np.float32,
                )
                for column, (_, _, _, support_indices) in enumerate(combinations):
                    seed_matrix[support_indices, column] = 1.0
                propagated = propagate_columns(transition, seed_matrix)

                payloads = []
                for column, (_, class_name, seed, support_indices) in enumerate(combinations):
                    query_mask = np.ones(len(pool_indices), dtype=bool)
                    query_mask[support_indices] = False
                    query_indices = np.where(query_mask)[0]
                    raw_scores = propagated[query_indices, column]
                    scores = (raw_scores - raw_scores.mean()) / (
                        raw_scores.std() + DEFAULT_GRAPH_SETTINGS.epsilon
                    )
                    labels = (pool_labels[query_indices] == class_name).astype(np.int64)
                    payloads.append(
                        (
                            shot,
                            class_name,
                            seed,
                            len(query_indices),
                            float(labels.mean()),
                            scores,
                            labels,
                        )
                    )

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    evaluated = list(executor.map(evaluate_payload, payloads))
                for payload, threshold_results in zip(payloads, evaluated):
                    shot_value, class_name, seed, query_count, target_fraction, _, _ = payload
                    repeat_rows.extend(
                        result_rows(
                            threshold_results,
                            suite="emds7",
                            task="one_vs_rest",
                            class_pool_size=class_pool_size,
                            repeat=repeat,
                            target_class=class_name,
                            seed=seed,
                            shot=shot_value,
                            query_count=query_count,
                            target_fraction=target_fraction,
                        )
                    )
            rows.extend(repeat_rows)
            write_rows_atomic(rows, output)
            print(
                f"  completed N={class_pool_size} repeat={repeat + 1}/{repeats}",
                flush=True,
            )
        print(
            f"Completed EMDS N={class_pool_size} in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        for method in METHODS:
            values = np.asarray(
                [
                    float(row["f1"])
                    for row in rows
                    if row.get("suite") == "emds7"
                    and int(row["class_pool_size"]) == class_pool_size
                    and int(row["shot"]) == 4
                    and row["method"] == method
                ]
            )
            print(f"  {method:<22} median={np.median(values):6.2f}", flush=True)


def write_summary(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, str, str, str, str], List[float]]
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
        aggregation = "mean" if suite == "self_collected" else "median"
        summary.append(
            {
                "suite": suite,
                "task": task,
                "class_pool_size": class_pool_size,
                "shot": shot,
                "method": method,
                "aggregation": aggregation,
                "f1": float(array.mean() if aggregation == "mean" else np.median(array)),
                "std": float(array.std()),
                "q25": float(np.quantile(array, 0.25)),
                "q75": float(np.quantile(array, 0.75)),
                "count": len(array),
            }
        )
    write_rows_atomic(summary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("self_collected", "emds7", "all"),
        default="all",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    add_config_argument(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = runtime_paths(args.config)
    cache_root = paths.output("cache")
    device = device_name()
    output = output_path(args.output, paths, "threshold_ablation.csv")
    rows = read_rows(output) if args.resume else []

    if args.suite in {"self_collected", "all"}:
        run_self_collected(
            rows,
            paths,
            cache_root,
            device,
            args.batch,
            output,
        )
    if args.suite in {"emds7", "all"}:
        run_emds7(
            rows,
            paths,
            cache_root,
            device,
            args.batch,
            args.workers,
            output,
        )

    write_rows_atomic(rows, output)
    write_summary(rows, output.with_name(output.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
