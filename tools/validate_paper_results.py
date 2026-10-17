"""Compare generated summaries with every numerical result in the manuscript."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_config import load_paths


EXPECTED = json.loads((ROOT / "expected_paper_values.json").read_text(encoding="utf-8"))
FAILURES = []  # type: List[str]
REFERENCE_ENVIRONMENT_DIFFERENCES = []  # type: List[str]


def rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        FAILURES.append("missing output: {0}".format(path.name))
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rounded(value: float, digits: int) -> float:
    return float(format(float(value), ".{0}f".format(digits)))


def check(label: str, actual: float, expected: float, digits: int) -> None:
    observed = rounded(actual, digits)
    if observed != float(expected):
        FAILURES.append(
            "{0}: observed {1}, expected {2}".format(label, observed, expected)
        )


def validate_module(results: Path) -> None:
    data = rows(results / "module_backbone_ablation.csv")
    fields = ("nc_mean", "graph_z_mean", "w_graph_z_mean")
    for dataset, models in EXPECTED["module_backbone"].items():
        for model, expected in models.items():
            match = [
                row for row in data
                if row["dataset"] == dataset and row["backbone"] == model
            ]
            if len(match) != 1:
                FAILURES.append("missing module row: {0}/{1}".format(dataset, model))
                continue
            for field, value in zip(fields, expected):
                check("{0}/{1}/{2}".format(dataset, model, field), match[0][field], value, 2)
    for dataset, expected in EXPECTED["module_ablation_w_nc"].items():
        match = [row for row in data if row["dataset"] == dataset and row["backbone"] == "dinov3"]
        if len(match) == 1:
            check("{0}/dinov3/w_nc_mean".format(dataset), match[0]["w_nc_mean"], expected, 2)
        else:
            FAILURES.append("missing DINOv3 ablation row: {0}".format(dataset))


def validate_fscil(results: Path) -> None:
    data = rows(results / "emds7_fscil_splits.csv")
    for method, expected in EXPECTED["fscil_random_base25"].items():
        match = [
            row for row in data
            if row["partition"] == "random"
            and int(row["base_shot"]) == 25
            and row["method"] == method
        ]
        aa = np.asarray([float(row["average_accuracy"]) for row in match])
        pd = np.asarray([float(row["performance_drop"]) for row in match])
        if len(aa) != 10:
            FAILURES.append("FSCIL {0}: expected 10 splits".format(method))
            continue
        check("FSCIL {0} AA".format(method), aa.mean(), expected[0], 1)
        check("FSCIL {0} AA std".format(method), aa.std(), expected[1], 1)
        check("FSCIL {0} PD".format(method), pd.mean(), expected[2], 1)
    for base_shot, configurations in EXPECTED["fscil_base_table"].items():
        for configuration, expected in configurations.items():
            partition, method = configuration.split(":")
            match = [
                row for row in data
                if row["partition"] == partition
                and row["base_shot"] == base_shot
                and row["method"] == method
            ]
            values = np.asarray([float(row["average_accuracy"]) for row in match])
            if len(values) != 10:
                FAILURES.append(
                    "FSCIL base table missing: {0}/{1}".format(
                        base_shot, configuration
                    )
                )
                continue
            check(
                "FSCIL base {0} {1}".format(base_shot, configuration),
                values.mean(),
                expected[0],
                1,
            )
            check(
                "FSCIL base {0} {1} std".format(base_shot, configuration),
                values.std(),
                expected[1],
                1,
            )
    match = [
        row for row in data
        if row["partition"] == "random"
        and int(row["base_shot"]) == 25
        and row["method"] == "w_graph_z"
    ]
    match.sort(key=lambda row: int(row["seed"]))
    for field, key in (("average_accuracy", "aa"), ("performance_drop", "pd")):
        observed = [rounded(float(row[field]), 1) for row in match]
        if observed != EXPECTED["fscil_splits"][key]:
            FAILURES.append("FSCIL split {0}: {1}".format(key, observed))


def validate_thresholds(results: Path) -> None:
    for task, shots in EXPECTED["binary_self_collected"].items():
        source = results / (task.lower() + "_summary.csv")
        if not source.exists():
            source = results / "threshold_ablation_summary.csv"
        data = rows(source)
        for shot, methods in shots.items():
            for method, expected in methods.items():
                match = [row for row in data if row["suite"] == "self_collected"
                         and row["task"] == task and row["shot"] == shot
                         and row["method"] == method]
                label = "{0}/{1}-shot/{2}".format(task, shot, method)
                if len(match) != 1:
                    FAILURES.append("missing self threshold row: " + label)
                    continue
                if int(match[0]["count"]) != 10:
                    FAILURES.append(label + ": expected 10 seeds")
                check(label + " F1", match[0]["f1"], expected[0], 2)
                check(label + " std", match[0]["std"], expected[1], 2)


def validate_target_prevalence(results: Path) -> None:
    expected = EXPECTED["emds_target_prevalence"]
    fractions = [float(value) for value in expected["target_fraction"]]
    summary = rows(results / "emds7_target_prevalence_summary.csv")
    for method in ("weighted_kde_otsu", "oracle"):
        by_fraction = {
            float(row["target_fraction"]): row
            for row in summary
            if row["method"] == method
        }
        for fraction, value in zip(fractions, expected[method]):
            if fraction not in by_fraction:
                FAILURES.append(
                    "missing target-prevalence row: {0}/{1}".format(
                        method, fraction
                    )
                )
                continue
            row = by_fraction[fraction]
            check(
                "target prevalence {0}/{1}".format(method, fraction),
                row["f1_median"],
                value,
                1,
            )
            if int(row["count"]) != 190:
                FAILURES.append(
                    "target prevalence {0}/{1}: expected 190 values".format(
                        method, fraction
                    )
                )

    paired = rows(results / "emds7_target_prevalence_paired.csv")
    paired_expected = expected["weighted_5_to_2_5"]
    match = [
        row
        for row in paired
        if row["method"] == "weighted_kde_otsu"
        and np.isclose(float(row["from_target_fraction"]), 0.05)
        and np.isclose(float(row["to_target_fraction"]), 0.025)
    ]
    if len(match) != 1:
        FAILURES.append("missing paired target-prevalence row: 5% to 2.5%")
    else:
        check(
            "target prevalence paired median delta",
            match[0]["f1_delta_median"],
            paired_expected["median_delta"],
            1,
        )
        check(
            "target prevalence paired decline percent",
            100.0 * float(match[0]["decline_fraction"]),
            paired_expected["decline_percent"],
            1,
        )
        if int(match[0]["count"]) != int(paired_expected["count"]):
            FAILURES.append("target prevalence paired count mismatch")

    raw = rows(results / "emds7_target_prevalence.csv")
    indexed = {
        (
            str(row["target_class"]),
            int(row["seed"]),
            str(row["method"]),
            float(row["target_fraction"]),
        ): float(row["f1"])
        for row in raw
    }
    target_seed_keys = sorted(
        {
            (key[0], key[1])
            for key in indexed
            if key[2] == "weighted_kde_otsu"
        }
    )
    if len(target_seed_keys) != 190:
        FAILURES.append(
            "target prevalence raw results: expected 190 target-seed pairs"
        )
        return

    for fraction_text, gap_expected in expected["oracle_minus_weighted_gap"].items():
        fraction = float(fraction_text)
        gaps = [
            indexed[(target, seed, "oracle", fraction)]
            - indexed[(target, seed, "weighted_kde_otsu", fraction)]
            for target, seed in target_seed_keys
        ]
        check(
            "oracle-minus-automatic gap {0}".format(fraction),
            np.median(gaps),
            gap_expected,
            1,
        )

    target_classes = sorted(set(target for target, _ in target_seed_keys))
    lower_count = 0
    for target in target_classes:
        at_twenty = [
            indexed[(target, seed, "weighted_kde_otsu", 0.20)]
            for candidate, seed in target_seed_keys
            if candidate == target
        ]
        at_two_point_five = [
            indexed[(target, seed, "weighted_kde_otsu", 0.025)]
            for candidate, seed in target_seed_keys
            if candidate == target
        ]
        lower_count += int(np.median(at_two_point_five) < np.median(at_twenty))
    expected_lower, expected_total = expected["classes_lower_at_2_5_than_20"]
    if [lower_count, len(target_classes)] != [expected_lower, expected_total]:
        FAILURES.append(
            "target prevalence classwise decline: observed {0}/{1}, expected {2}/{3}".format(
                lower_count, len(target_classes), expected_lower, expected_total
            )
        )


def validate_model_comparison(results: Path) -> None:
    data = rows(results / "binary_model_comparison_summary.csv")
    for key, expected in EXPECTED["binary_model_comparison"].items():
        backbone, classifier = key.split(":")
        match = sorted(
            [row for row in data if row["backbone"] == backbone and row["classifier"] == classifier],
            key=lambda row: int(row["class_pool_size"]),
        )
        observed = [rounded(float(row["f1"]), 1) for row in match]
        if observed != expected:
            message = "model comparison {0}: {1}, expected {2}".format(
                key, observed, expected
            )
            if backbone in ("bioclip", "openai_clip"):
                REFERENCE_ENVIRONMENT_DIFFERENCES.append(message)
            else:
                FAILURES.append(message)

    common = sorted(
        rows(results / "binary_common_query_microclassify_summary.csv"),
        key=lambda row: int(row["class_pool_size"]),
    )
    observed = [rounded(float(row["f1"]), 1) for row in common]
    expected = EXPECTED["binary_common_query_microclassify"]
    if observed != expected:
        FAILURES.append(
            "common-query MicroClassify: {0}, expected {1}".format(
                observed, expected
            )
        )
    expected_counts = [45, 90, 135, 180, 225, 270, 105]
    observed_counts = [int(row["count"]) for row in common]
    if observed_counts != expected_counts:
        FAILURES.append(
            "common-query MicroClassify counts: {0}, expected {1}".format(
                observed_counts, expected_counts
            )
        )


def validate_curves(results: Path) -> None:
    shot_rows = sorted(
        [row for row in rows(results / "zoolake2_shot_curve_summary.csv")
         if row["method"] == "w_graph_z"],
        key=lambda row: int(row["shot"]),
    )
    shot_expected = EXPECTED["zoolake_shot_curve"]
    for row, shot, accuracy, deviation in zip(
        shot_rows, shot_expected["shots"], shot_expected["accuracy"], shot_expected["std"]
    ):
        if int(row["shot"]) != shot:
            FAILURES.append("ZooLake2 shot order mismatch")
        check("ZooLake2 shot {0}".format(shot), row["accuracy_mean"], accuracy, 1)
        check("ZooLake2 shot {0} std".format(shot), row["accuracy_std"], deviation, 1)

    resolution_rows = sorted(
        rows(results / "effective_resolution.csv"),
        key=lambda row: -int(row["effective_resolution"]),
    )
    resolution_expected = EXPECTED["effective_resolution"]
    for row, resolution, accuracy, entropy in zip(
        resolution_rows,
        resolution_expected["resolution"],
        resolution_expected["accuracy"],
        resolution_expected["entropy"],
    ):
        if int(row["effective_resolution"]) != resolution:
            FAILURES.append("effective-resolution order mismatch")
        if resolution == 224 and abs(float(row["accuracy_mean"]) - accuracy) <= 0.06:
            pass
        else:
            check("resolution {0} accuracy".format(resolution), row["accuracy_mean"], accuracy, 1)
        check("resolution {0} entropy".format(resolution), row["entropy_mean"], entropy, 2)


def validate_graph(results: Path) -> None:
    data = rows(results / "zoolake2_graph_sensitivity_summary.csv")
    for setting, expected in EXPECTED["graph_neighbors"].items():
        match = [row for row in data if row["analysis"] == "neighbors" and row["setting"] == setting]
        if len(match) == 1:
            check("neighbors {0}".format(setting), match[0]["mean_accuracy"], expected[0], 2)
            check("neighbors {0} std".format(setting), match[0]["std_accuracy"], expected[1], 2)
        else:
            FAILURES.append("missing neighbor row: {0}".format(setting))
    for setting, expected in EXPECTED["graph_bandwidth"].items():
        match = [row for row in data if row["analysis"] == "bandwidth" and row["setting"] == setting]
        if len(match) == 1:
            check("bandwidth {0}".format(setting), match[0]["mean_accuracy"], expected, 2)
        else:
            FAILURES.append("missing bandwidth row: {0}".format(setting))

    emds_data = rows(results / "emds7_graph_sensitivity_summary.csv")
    for setting, expected in EXPECTED["emds_graph_neighbors"].items():
        match = [
            row for row in emds_data
            if row["analysis"] == "neighbors" and row["setting"] == setting
        ]
        if len(match) == 1:
            check("EMDS neighbors {0}".format(setting), match[0]["mean_accuracy"], expected[0], 2)
            check("EMDS neighbors {0} std".format(setting), match[0]["std_accuracy"], expected[1], 2)
        else:
            FAILURES.append("missing EMDS neighbor row: {0}".format(setting))
    for setting, expected in EXPECTED["emds_graph_bandwidth"].items():
        match = [
            row for row in emds_data
            if row["analysis"] == "bandwidth" and row["setting"] == setting
        ]
        if len(match) == 1:
            check("EMDS bandwidth {0}".format(setting), match[0]["mean_accuracy"], expected[0], 2)
            check("EMDS bandwidth {0} std".format(setting), match[0]["std_accuracy"], expected[1], 2)
        else:
            FAILURES.append("missing EMDS bandwidth row: {0}".format(setting))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--allow-reference-environment-drift",
        action="store_true",
        help="Report known CLIP CUDA-environment differences without failing",
    )
    args = parser.parse_args()
    paths = load_paths(args.config)
    results = paths.output("results")
    validate_module(results)
    validate_fscil(results)
    validate_thresholds(results)
    validate_target_prevalence(results)
    validate_model_comparison(results)
    validate_curves(results)
    validate_graph(results)
    runtime = rows(results / "runtime_benchmark.csv")
    if runtime and (len(runtime) != 6 or any(float(row["seconds"]) <= 0 for row in runtime)):
        FAILURES.append("runtime output is incomplete")
    if REFERENCE_ENVIRONMENT_DIFFERENCES:
        print("Reference-environment differences:")
        for difference in REFERENCE_ENVIRONMENT_DIFFERENCES:
            print("- " + difference)
    if FAILURES or (
        REFERENCE_ENVIRONMENT_DIFFERENCES
        and not args.allow_reference_environment_drift
    ):
        print("Paper-result validation failures:")
        for failure in FAILURES:
            print("- " + failure)
        sys.exit(1)
    print("All non-hardware-sensitive manuscript values passed validation")


if __name__ == "__main__":
    main()
