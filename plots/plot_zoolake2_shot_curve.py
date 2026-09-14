"""Render the ZooLake2 shot curve from generated CSV results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from runtime_config import load_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = load_paths(args.config)
    input_path = args.input or paths.output("results") / "zoolake2_shot_curve_summary.csv"
    output_path = args.output or paths.output("figures") / "zoolake_shot.pdf"
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["method"] == "w_graph_z"
        ]
    rows.sort(key=lambda row: int(row["shot"]))
    shots = [int(row["shot"]) for row in rows]
    accuracy = [float(row["accuracy_mean"]) for row in rows]
    deviation = [float(row["accuracy_std"]) for row in rows]

    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    axis.errorbar(shots, accuracy, yerr=deviation, fmt="-o", capsize=3)
    axis.axhline(92.0, linestyle=":", label="Fully supervised MobileNet")
    axis.set_xlabel("Labeled shots per class")
    axis.set_ylabel("All-way accuracy (%)")
    axis.set_xticks(shots)
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), bbox_inches="tight")


if __name__ == "__main__":
    main()
