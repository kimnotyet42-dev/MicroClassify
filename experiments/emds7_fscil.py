"""Whole-field EMDS-7 few-shot class-incremental experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataio.datasets import load_emds7
from dataio.protocols import FewShotSplit, emds_fscil_splits
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import centroid_scores, graph_scores, localize_split, zscore_columns
from methods import backbones
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.whitening import (
    normalize_rows,
    support_residuals,
    transform_with_support_residuals_full,
)


METHODS = ("nc", "graph_z", "w_graph_z")


def evaluate_partition(
    features: np.ndarray,
    partition,
    partition_mode: str,
    base_shot: int,
    seed: int,
) -> List[dict]:
    base = partition.sessions[0]
    base_support = [partition.support_by_class[label] for label in base]
    residuals = support_residuals(features, base_support)
    whitened = transform_with_support_residuals_full(features, residuals)

    rows = []
    seen = []  # type: List[str]
    for session_index, new_classes in enumerate(partition.sessions):
        seen.extend(new_classes)
        support_groups = [partition.support_by_class[label] for label in seen]
        query_indices = [
            index for label in seen for index in partition.test_by_class[label]
        ]
        query_labels = np.asarray(
            [class_index for class_index, label in enumerate(seen)
             for _ in partition.test_by_class[label]],
            dtype=np.int64,
        )
        split = FewShotSplit(support_groups, query_indices, query_labels)
        nodes, support_local, query_local = localize_split(split)
        episode_features = features[nodes]
        episode_whitened = whitened[nodes]
        nearest = centroid_scores(episode_features, support_local, query_local)
        propagated = graph_scores(
            episode_features,
            support_local,
            query_local,
            DEFAULT_GRAPH_SETTINGS,
        )
        whitened_propagated = graph_scores(
            episode_whitened,
            support_local,
            query_local,
            DEFAULT_GRAPH_SETTINGS,
        )
        predictions = {
            "nc": nearest.argmax(axis=1),
            "graph_z": zscore_columns(propagated).argmax(axis=1),
            "w_graph_z": zscore_columns(whitened_propagated).argmax(axis=1),
        }
        for method in METHODS:
            rows.append(
                {
                    "dataset": "emds7",
                    "protocol": "fscil_whole_field",
                    "partition": partition_mode,
                    "base_shot": base_shot,
                    "seed": seed,
                    "session": session_index,
                    "seen_classes": len(seen),
                    "support_count": sum(len(group) for group in support_groups),
                    "query_count": len(query_indices),
                    "method": method,
                    "accuracy": 100.0 * float(
                        np.mean(predictions[method] == split.query_labels)
                    ),
                }
            )
    return rows


def summarize(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, int, str, int], List[float]]
    for row in rows:
        grouped[
            (
                str(row["partition"]),
                int(row["base_shot"]),
                str(row["method"]),
                int(row["session"]),
            )
        ].append(float(row["accuracy"]))
    summary = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        summary.append(
            {
                "partition": key[0],
                "base_shot": key[1],
                "method": key[2],
                "session": key[3],
                "seeds": len(values),
                "accuracy_mean": float(values.mean()),
                "accuracy_std": float(values.std()),
            }
        )
    if summary:
        write_rows_atomic(summary, output)


def split_statistics(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(dict)  # type: Dict[Tuple[str, int, str, int], Dict[int, float]]
    for row in rows:
        key = (
            str(row["partition"]),
            int(row["base_shot"]),
            str(row["method"]),
            int(row["seed"]),
        )
        grouped[key][int(row["session"])] = float(row["accuracy"])
    result = []
    for key in sorted(grouped):
        sessions = grouped[key]
        ordered = np.asarray([sessions[index] for index in sorted(sessions)])
        result.append(
            {
                "partition": key[0],
                "base_shot": key[1],
                "method": key[2],
                "seed": key[3],
                "sessions": len(ordered),
                "average_accuracy": float(ordered.mean()),
                "performance_drop": float(ordered[0] - ordered[-1]),
            }
        )
    if result:
        write_rows_atomic(result, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-shots", type=int, nargs="+", default=(4, 10, 25))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--partitions",
        choices=("random", "fixed", "both"),
        default="both",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "emds7_fscil.csv")
    samples = load_emds7(paths.dataset("emds7"))
    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
        image_size=args.size,
    )
    features = normalize_rows(
        load_or_extract(
            backbone,
            samples,
            paths.output("cache"),
            "emds7_dinov3_{0}".format(args.size),
            args.batch,
        )
    )
    rows = read_rows(output) if args.resume else []
    completed = {
        (str(row["partition"]), int(row["base_shot"]), int(row["seed"]))
        for row in rows
    }

    partition_modes = (
        ("random", "fixed") if args.partitions == "both" else (args.partitions,)
    )
    for partition_mode in partition_modes:
        for base_shot in args.base_shots:
            partitions = emds_fscil_splits(
                samples,
                base_shot=base_shot,
                seeds=args.seeds,
                random_partition=partition_mode == "random",
            )
            for seed, partition in enumerate(partitions):
                if (partition_mode, base_shot, seed) in completed:
                    continue
                partition_rows = evaluate_partition(
                    features, partition, partition_mode, base_shot, seed
                )
                rows.extend(partition_rows)
                write_rows_atomic(rows, output)
                summarize(rows, output.with_name(output.stem + "_sessions.csv"))
                split_statistics(rows, output.with_name(output.stem + "_splits.csv"))
                print(
                    "Completed partition={0}, base_shot={1}, seed={2}/{3}".format(
                        partition_mode, base_shot, seed + 1, args.seeds
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
