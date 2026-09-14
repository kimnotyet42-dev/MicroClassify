"""Render the EMDS-7 effective-resolution figure from generated CSV results."""
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
    input_path = args.input or paths.output("results") / "effective_resolution.csv"
    output_path = args.output or paths.output("figures") / "quality.pdf"
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: -int(row["effective_resolution"]))
    labels = [int(row["effective_resolution"]) for row in rows]
    accuracy = [float(row["accuracy_mean"]) for row in rows]
    entropy = [float(row["entropy_mean"]) for row in rows]
    positions = list(range(len(rows)))

    figure, axis_accuracy = plt.subplots(figsize=(5.2, 3.4))
    axis_accuracy.plot(positions, accuracy, "-o", label="Accuracy")
    axis_accuracy.set_xlabel("Effective resolution (px)")
    axis_accuracy.set_ylabel("All-way 4-shot accuracy (%)")
    axis_accuracy.set_xticks(positions)
    axis_accuracy.set_xticklabels([str(value) for value in labels])
    axis_accuracy.grid(True, linestyle=":", alpha=0.5)
    axis_entropy = axis_accuracy.twinx()
    axis_entropy.plot(positions, entropy, "--s", label="Image entropy")
    axis_entropy.set_ylabel("Image entropy (bits)")
    lines_a, labels_a = axis_accuracy.get_legend_handles_labels()
    lines_b, labels_b = axis_entropy.get_legend_handles_labels()
    axis_accuracy.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8.5)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), bbox_inches="tight")


if __name__ == "__main__":
    main()
