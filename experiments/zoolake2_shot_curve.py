"""ZooLake2 fixed-class accuracy as labeled shots per class increase."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataio.datasets import load_zoolake2
from dataio.protocols import zoolake_shot_splits
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import evaluate_multiclass_split
from methods import backbones
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.whitening import normalize_rows


SHOTS = (4, 8, 12, 16, 20, 24)
METHODS = ("nc", "graph_z", "w_graph_z")


def summarize(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[Tuple[int, str], List[float]]
    for row in rows:
        grouped[(int(row["shot"]), str(row["method"]))].append(
            float(row["accuracy"])
        )
    summary = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        summary.append(
            {
                "shot": key[0],
                "method": key[1],
                "episodes": len(values),
                "accuracy_mean": float(values.mean()),
                "accuracy_std": float(values.std()),
            }
        )
    if summary:
        write_rows_atomic(summary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "zoolake2_shot_curve.csv")
    samples = load_zoolake2(paths.dataset("zoolake2"))
    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
        image_size=224,
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
    rows = read_rows(output) if args.resume else []
    completed_shots = set()
    for shot in args.shots:
        count = sum(int(row["shot"]) == shot for row in rows)
        expected = args.episodes * len(METHODS)
        if count == expected:
            completed_shots.add(shot)
        elif count:
            rows = [row for row in rows if int(row["shot"]) != shot]

    maximum_shot = max(args.shots)
    for shot in args.shots:
        if shot in completed_shots:
            continue
        classes, splits = zoolake_shot_splits(
            samples,
            shot=shot,
            maximum_shot=maximum_shot,
            query=args.query,
            episodes=args.episodes,
            seed=0,
        )
        unit_rows = []
        for episode, split in enumerate(splits):
            result = evaluate_multiclass_split(
                features, split, DEFAULT_GRAPH_SETTINGS
            )
            for method in METHODS:
                unit_rows.append(
                    {
                        "dataset": "zoolake2",
                        "shot": shot,
                        "query_per_class": args.query,
                        "classes": len(classes),
                        "episode": episode,
                        "method": method,
                        "accuracy": 100.0 * result[method],
                    }
                )
        rows.extend(unit_rows)
        write_rows_atomic(rows, output)
        summarize(rows, output.with_name(output.stem + "_summary.csv"))
        print(
            "Completed shot={0}: {1} classes, {2} episodes".format(
                shot, len(classes), len(splits)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
