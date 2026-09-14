"""Render the EMDS-7 FSCIL session curve from generated CSV results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from runtime_config import load_paths


PUBLISHED_BASELINES = {
    "SAVC": [92.9, 89.0, 84.7, 82.7, 81.7, 73.3, 73.7, 66.2],
    "FACT": [91.6, 89.0, 83.4, 81.3, 80.5, 73.8, 73.3, 66.6],
    "TEEN": [88.3, 87.1, 82.4, 80.1, 75.3, 72.0, 69.2, 62.1],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = load_paths(args.config)
    input_path = args.input or paths.output("results") / "emds7_fscil_sessions.csv"
    output_path = args.output or paths.output("figures") / "sessions.pdf"
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ours = sorted(
        [row for row in rows
         if row["partition"] == "random"
         and int(row["base_shot"]) == 25
         and row["method"] == "w_graph_z"],
        key=lambda row: int(row["session"]),
    )
    if len(ours) != 8:
        raise ValueError("Expected eight FSCIL session summaries")
    sessions = list(range(8))
    values = [float(row["accuracy_mean"]) for row in ours]

    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    axis.plot(sessions, values, "-o", linewidth=2.6, label="MicroClassify")
    for name, baseline in PUBLISHED_BASELINES.items():
        axis.plot(sessions, baseline, "--o", linewidth=1.5, markersize=4, label=name)
    axis.set_xlabel("Incremental session")
    axis.set_ylabel("Cumulative accuracy (%)")
    axis.set_xticks(sessions)
    axis.set_xticklabels(["S{0}".format(index) for index in sessions])
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(fontsize=8.5, loc="lower left")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), bbox_inches="tight")


if __name__ == "__main__":
    main()
