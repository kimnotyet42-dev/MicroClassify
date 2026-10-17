"""MicroClassify on the same EMDS-7 query sets as all-class baselines.

For each class pool and seed, four images are reserved from every class and
removed from the common query set.  MicroClassify receives only the four
support images of the current target class; support images reserved for the
other classes are neither graph nodes nor labeled inputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

from dataio.datasets import load_emds7
from dataio.protocols import emds_class_pool, group_indices, support_for_classes
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from methods import backbones, graph
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.thresholds import (
    binary_f1,
    kde_density,
    otsu_from_density,
    score_grid,
    select_gamma,
)
from methods.whitening import normalize_rows


CLASS_POOL_SIZES = (5, 10, 15, 20, 25, 30, 35)
SEEDS = (0, 1, 2)


def summarize(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[int, List[float]]
    for row in rows:
        grouped[int(row["class_pool_size"])].append(float(row["f1"]))
    summary = []
    for class_pool_size in sorted(grouped):
        values = np.asarray(grouped[class_pool_size], dtype=np.float64)
        summary.append(
            {
                "method": "microclassify_common_query",
                "class_pool_size": class_pool_size,
                "aggregation": "median",
                "f1": float(np.median(values)),
                "count": len(values),
            }
        )
    if summary:
        write_rows_atomic(summary, output)


def evaluate_target(payload) -> dict:
    (
        pool_distance,
        support_local,
        common_query_local,
        binary_labels,
        metadata,
    ) = payload
    settings = DEFAULT_GRAPH_SETTINGS
    nodes = list(support_local) + list(common_query_local)
    distance = pool_distance[np.ix_(nodes, nodes)]
    transition = graph.build_knn_transition(
        distance,
        settings.neighbors,
        min(settings.sigma_neighbors, settings.neighbors),
        settings.sigma_method,
        settings.epsilon,
        False,
    )
    seeds = np.zeros(len(nodes), dtype=np.float32)
    seeds[: len(support_local)] = 1.0
    propagated, _, _ = graph.propagate_from_transition(
        transition,
        seeds,
        graph_alpha=settings.alpha,
        graph_max_iter=settings.max_iterations,
        graph_tol=settings.tolerance,
    )
    raw_scores = propagated[len(support_local) :]
    scores = (raw_scores - raw_scores.mean()) / (
        raw_scores.std() + settings.epsilon
    )
    grid = score_grid(scores)
    gamma = select_gamma(scores, grid)
    threshold = otsu_from_density(grid, kde_density(scores, grid, gamma))
    result = dict(metadata)
    result.update(
        {
            "target_fraction": float(binary_labels.mean()),
            "gamma": gamma,
            "f1": 100.0 * binary_f1(binary_labels, scores >= threshold),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument(
        "--class-pool-sizes",
        type=int,
        nargs="+",
        default=CLASS_POOL_SIZES,
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = runtime_paths(args.config)
    output = output_path(
        args.output,
        paths,
        "binary_common_query_microclassify.csv",
    )
    samples = load_emds7(paths.dataset("emds7"))
    grouped = group_indices(samples)
    all_labels = np.asarray([sample.label for sample in samples])
    rows = read_rows(output) if args.resume else []
    completed = {
        (
            int(row["class_pool_size"]),
            int(row["repeat"]),
            int(row["seed"]),
            str(row["target_class"]),
        )
        for row in rows
    }

    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
    )
    features = load_or_extract(
        backbone,
        samples,
        paths.output("cache"),
        "emds7_dinov3_{0}".format(backbone.image_size),
        args.batch
        if args.batch is not None
        else backbones.recommended_batch_size("dinov3"),
    )
    features = normalize_rows(features)

    for class_pool_size in args.class_pool_sizes:
        if class_pool_size not in CLASS_POOL_SIZES:
            parser.error("Unsupported class-pool size: {0}".format(class_pool_size))
        repeats = 1 if class_pool_size == max(CLASS_POOL_SIZES) else 3
        for repeat in range(repeats):
            pool_indices, selected = emds_class_pool(samples, class_pool_size, repeat)
            expected_keys = {
                (class_pool_size, repeat, seed, class_name)
                for seed in SEEDS
                for class_name in selected
            }
            if expected_keys.issubset(completed):
                print(
                    "Skip completed N={0}, repeat={1}/{2}".format(
                        class_pool_size, repeat + 1, repeats
                    ),
                    flush=True,
                )
                continue
            pool_set = set(pool_indices)
            global_to_pool = {
                global_index: pool_index
                for pool_index, global_index in enumerate(pool_indices)
            }
            pool_distance = graph.pairwise_distance_matrix(
                features[pool_indices],
                "cos",
                DEFAULT_GRAPH_SETTINGS.epsilon,
            )
            label_to_index = {
                label: class_index for class_index, label in enumerate(selected)
            }
            for seed in SEEDS:
                support_groups = support_for_classes(grouped, selected, args.shot, seed)
                reserved_support = {
                    index for group in support_groups for index in group
                }
                common_query = sorted(pool_set - reserved_support)
                query_class_labels = np.asarray(
                    [label_to_index[all_labels[index]] for index in common_query],
                    dtype=np.int64,
                )
                common_query_local = [global_to_pool[index] for index in common_query]
                payloads = []
                for class_index, class_name in enumerate(selected):
                    key = (class_pool_size, repeat, seed, class_name)
                    if key in completed:
                        continue
                    binary_labels = (query_class_labels == class_index).astype(np.int64)
                    support_local = [
                        global_to_pool[index]
                        for index in support_groups[class_index]
                    ]
                    payloads.append(
                        (
                            pool_distance,
                            support_local,
                            common_query_local,
                            binary_labels,
                            {
                                "dataset": "emds7",
                                "method": "microclassify_common_query",
                                "class_pool_size": class_pool_size,
                                "repeat": repeat,
                                "seed": seed,
                                "target_class": class_name,
                                "shot": args.shot,
                                "support_count_visible_to_method": args.shot,
                                "reserved_support_count": len(reserved_support),
                                "query_count": len(common_query),
                            },
                        )
                    )
                if args.workers == 1:
                    seed_rows = [evaluate_target(payload) for payload in payloads]
                else:
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        seed_rows = list(executor.map(evaluate_target, payloads))
                rows.extend(seed_rows)
                for row in seed_rows:
                    key = (
                        int(row["class_pool_size"]),
                        int(row["repeat"]),
                        int(row["seed"]),
                        str(row["target_class"]),
                    )
                    completed.add(key)
                write_rows_atomic(rows, output)
                summarize(rows, output.with_name(output.stem + "_summary.csv"))
                print(
                    "Completed N={0}, repeat={1}/{2}, seed={3}; query={4}".format(
                        class_pool_size,
                        repeat + 1,
                        repeats,
                        seed,
                        len(common_query),
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
