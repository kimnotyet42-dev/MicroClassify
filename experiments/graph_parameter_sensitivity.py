"""ZooLake2 sensitivity to local bandwidth and kNN neighborhood size."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataio.datasets import load_zoolake2
from dataio.protocols import zoolake_episodes
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import (
    graph_z_accuracy_from_distance,
    localize_split,
    settings_with,
)
from methods import backbones, graph
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.whitening import normalize_rows


NEIGHBORS = (1, 3, 5, 10, 15, 20, 30, 50)
BANDWIDTHS = ("log_mean", "k7", "local_median", "global_median")


def requested_settings(analysis: str) -> List[Tuple[str, str]]:
    values = []  # type: List[Tuple[str, str]]
    if analysis in ("neighbors", "all"):
        values.extend(("neighbors", str(value)) for value in NEIGHBORS)
    if analysis in ("bandwidth", "all"):
        values.extend(("bandwidth", value) for value in BANDWIDTHS)
    return values


def summarize(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, str], List[float]]
    for row in rows:
        grouped[(str(row["analysis"]), str(row["setting"]))].append(
            float(row["accuracy"])
        )
    summary = []
    for key in requested_settings("all"):
        if key not in grouped:
            continue
        values = np.asarray(grouped[key], dtype=np.float64)
        summary.append(
            {
                "dataset": "zoolake2",
                "analysis": key[0],
                "setting": key[1],
                "episodes": len(values),
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std()),
            }
        )
    if summary:
        write_rows_atomic(summary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        choices=("neighbors", "bandwidth", "all"),
        default="all",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "zoolake2_graph_sensitivity.csv")
    summary_output = output.with_name(output.stem + "_summary.csv")
    samples = load_zoolake2(paths.dataset("zoolake2"))
    episodes = zoolake_episodes(samples, way=32, shot=5, query=15, episodes=200)
    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
    )
    features = normalize_rows(
        load_or_extract(
            backbone,
            samples,
            paths.output("cache"),
            "zoolake2_dinov3_224",
            args.batch,
        )
    )

    settings = requested_settings(args.analysis)
    rows = read_rows(output) if args.resume else []
    completed = {
        (str(row["analysis"]), str(row["setting"]), int(row["episode"]))
        for row in rows
    }
    pending_since_checkpoint = 0

    for episode_index, split in enumerate(episodes):
        needed = [
            key for key in settings if (key[0], key[1], episode_index) not in completed
        ]
        if not needed:
            continue
        nodes, support_local, query_local = localize_split(split)
        episode_features = features[nodes]
        distance = graph.pairwise_distance_matrix(
            episode_features,
            "cos",
            DEFAULT_GRAPH_SETTINGS.epsilon,
        )
        for analysis, value in needed:
            graph_settings = (
                settings_with(DEFAULT_GRAPH_SETTINGS, neighbors=int(value))
                if analysis == "neighbors"
                else settings_with(DEFAULT_GRAPH_SETTINGS, sigma_method=value)
            )
            accuracy = graph_z_accuracy_from_distance(
                distance,
                support_local,
                query_local,
                split.query_labels,
                graph_settings,
            )
            rows.append(
                {
                    "dataset": "zoolake2",
                    "analysis": analysis,
                    "setting": value,
                    "episode": episode_index,
                    "accuracy": 100.0 * accuracy,
                }
            )
        pending_since_checkpoint += 1
        if pending_since_checkpoint >= args.checkpoint_every:
            write_rows_atomic(rows, output)
            summarize(rows, summary_output)
            pending_since_checkpoint = 0
            print(
                "Completed episode {0}/{1}".format(episode_index + 1, len(episodes)),
                flush=True,
            )

    if rows:
        write_rows_atomic(rows, output)
        summarize(rows, summary_output)
    for row in read_rows(summary_output):
        if (row["analysis"], row["setting"]) in settings:
            print(
                "{0}={1}: {2:.2f} +/- {3:.2f}".format(
                    row["analysis"],
                    row["setting"],
                    float(row["mean_accuracy"]),
                    float(row["std_accuracy"]),
                )
            )


if __name__ == "__main__":
    main()
