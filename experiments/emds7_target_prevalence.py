"""Controlled EMDS-7 target-prevalence experiment.

The query batch always contains 200 images.  For each target class and seed,
the same four support images are used at target prevalences of 20%, 10%, 5%,
and 2.5%.  Positive and negative query sets are nested across prevalence levels,
and every eligible negative class is represented in every query batch.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dataio.datasets import load_emds7
from dataio.protocols import group_indices
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from methods import backbones, graph
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.thresholds import evaluate_thresholds
from methods.whitening import normalize_rows


DEFAULT_PREVALENCES = (0.20, 0.10, 0.05, 0.025)
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_SHOT = 4
DEFAULT_QUERY_SIZE = 200
MINIMUM_POOL_CLASS_SIZE = 12
MAXIMUM_TARGET_SHARE = 0.05
METHODS = ("direct_otsu", "equal_kde_otsu", "weighted_kde_otsu", "oracle")


def _label_number(label: str) -> int:
    digits = "".join(character for character in label if character.isdigit())
    if not digits:
        raise ValueError("EMDS-7 label has no numeric component: {0}".format(label))
    return int(digits)


def select_target_classes(
    grouped: Dict[str, List[int]],
    total_images: int,
    shot: int = DEFAULT_SHOT,
    query_size: int = DEFAULT_QUERY_SIZE,
    prevalences: Sequence[float] = DEFAULT_PREVALENCES,
    maximum_target_share: float = MAXIMUM_TARGET_SHARE,
) -> List[str]:
    """Select non-dominant classes that support every prevalence setting."""
    maximum_positive_queries = max(int(round(query_size * value)) for value in prevalences)
    minimum_images = shot + maximum_positive_queries
    targets = [
        label
        for label, indices in grouped.items()
        if len(indices) >= minimum_images
        and float(len(indices)) / float(total_images) <= maximum_target_share
    ]
    return sorted(targets, key=lambda label: (-len(grouped[label]), label))


def _balanced_negative_order(
    grouped: Dict[str, List[int]],
    negative_classes: Sequence[str],
    required: int,
    seed: int,
) -> List[int]:
    """Return a nested, approximately class-balanced negative sample order."""
    rng = np.random.RandomState(seed)
    shuffled = {}
    positions = {}
    for label in sorted(negative_classes):
        values = np.asarray(grouped[label], dtype=np.int64).copy()
        rng.shuffle(values)
        shuffled[label] = values
        positions[label] = 0

    output = []  # type: List[int]
    while len(output) < required:
        round_classes = list(sorted(negative_classes))
        rng.shuffle(round_classes)
        advanced = False
        for label in round_classes:
            position = positions[label]
            values = shuffled[label]
            if position >= len(values):
                continue
            output.append(int(values[position]))
            positions[label] = position + 1
            advanced = True
            if len(output) == required:
                break
        if not advanced:
            raise ValueError(
                "Negative pool contains fewer than {0} available images".format(required)
            )
    return output


def build_paired_splits(
    grouped: Dict[str, List[int]],
    target_class: str,
    negative_classes: Sequence[str],
    seed: int,
    shot: int = DEFAULT_SHOT,
    query_size: int = DEFAULT_QUERY_SIZE,
    prevalences: Sequence[float] = DEFAULT_PREVALENCES,
) -> Dict[float, Tuple[List[int], List[int], np.ndarray]]:
    """Construct exact-prevalence splits with shared support and nested queries."""
    positive_counts = {}
    for prevalence in prevalences:
        count = int(round(query_size * prevalence))
        if not np.isclose(float(count) / float(query_size), prevalence):
            raise ValueError(
                "query_size={0} cannot represent prevalence={1} exactly".format(
                    query_size, prevalence
                )
            )
        positive_counts[float(prevalence)] = count

    target_values = np.asarray(grouped[target_class], dtype=np.int64).copy()
    target_rng = np.random.RandomState(100000 * seed + _label_number(target_class))
    target_rng.shuffle(target_values)
    maximum_positive = max(positive_counts.values())
    if len(target_values) < shot + maximum_positive:
        raise ValueError(
            "Target {0} has {1} images; required {2}".format(
                target_class, len(target_values), shot + maximum_positive
            )
        )
    support = target_values[:shot].tolist()
    positive_order = target_values[shot : shot + maximum_positive].tolist()

    maximum_negative = query_size - min(positive_counts.values())
    negative_order = _balanced_negative_order(
        grouped,
        negative_classes,
        maximum_negative,
        seed=200000 * seed + _label_number(target_class) + 7919,
    )

    splits = {}
    for prevalence in prevalences:
        prevalence = float(prevalence)
        positive_count = positive_counts[prevalence]
        negative_count = query_size - positive_count
        query = positive_order[:positive_count] + negative_order[:negative_count]
        labels = np.asarray(
            [1] * positive_count + [0] * negative_count,
            dtype=np.int64,
        )
        order_rng = np.random.RandomState(
            300000 * seed + _label_number(target_class) + int(round(prevalence * 10000))
        )
        order = order_rng.permutation(query_size)
        splits[prevalence] = (
            list(support),
            np.asarray(query, dtype=np.int64)[order].tolist(),
            labels[order],
        )
    return splits


def _precision_recall(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    false_negative = int(np.sum(~predictions & (labels == 1)))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return precision, recall


def _evaluate_split(payload) -> List[dict]:
    features, support, query, labels, metadata = payload
    settings = DEFAULT_GRAPH_SETTINGS
    nodes = list(support) + list(query)
    node_features = features[nodes]
    propagated = graph.graph_propagate_score(
        node_features,
        "cos",
        list(range(len(support))),
        settings.neighbors,
        settings.sigma_neighbors,
        settings.sigma_method,
        settings.alpha,
        settings.max_iterations,
        settings.tolerance,
        settings.epsilon,
    )
    raw_scores = propagated[len(support) :]
    scores = (raw_scores - raw_scores.mean()) / (raw_scores.std() + settings.epsilon)
    threshold_results = evaluate_thresholds(scores, labels)
    rows = []
    for result in threshold_results:
        precision, recall = _precision_recall(labels, scores >= result.threshold)
        row = dict(metadata)
        row.update(
            {
                "method": result.method,
                "threshold": result.threshold,
                "gamma": "" if result.gamma is None else result.gamma,
                "precision": 100.0 * precision,
                "recall": 100.0 * recall,
                "f1": 100.0 * result.f1,
            }
        )
        rows.append(row)
    return rows


def _summary_rows(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_fraction"]), str(row["method"]))].append(row)
    output = []
    for (prevalence, method), values in sorted(
        grouped.items(), key=lambda item: (-item[0][0], METHODS.index(item[0][1]))
    ):
        f1 = np.asarray([float(value["f1"]) for value in values], dtype=np.float64)
        precision = np.asarray(
            [float(value["precision"]) for value in values], dtype=np.float64
        )
        recall = np.asarray([float(value["recall"]) for value in values], dtype=np.float64)
        output.append(
            {
                "target_fraction": prevalence,
                "method": method,
                "aggregation": "median_across_target_class_and_seed",
                "f1_median": float(np.median(f1)),
                "f1_q25": float(np.quantile(f1, 0.25)),
                "f1_q75": float(np.quantile(f1, 0.75)),
                "f1_mean": float(np.mean(f1)),
                "f1_std": float(np.std(f1)),
                "precision_median": float(np.median(precision)),
                "recall_median": float(np.median(recall)),
                "count": len(values),
            }
        )
    return output


def _paired_rows(rows: Sequence[dict]) -> List[dict]:
    indexed = {
        (
            str(row["target_class"]),
            int(row["seed"]),
            str(row["method"]),
            float(row["target_fraction"]),
        ): float(row["f1"])
        for row in rows
    }
    methods = sorted(set(str(row["method"]) for row in rows), key=METHODS.index)
    fractions = sorted(
        set(float(row["target_fraction"]) for row in rows), reverse=True
    )
    output = []
    for method in methods:
        for from_index, from_fraction in enumerate(fractions[:-1]):
            for to_fraction in fractions[from_index + 1 :]:
                differences = []
                for key, current in indexed.items():
                    target_class, seed, key_method, key_fraction = key
                    if key_method != method or not np.isclose(
                        key_fraction, to_fraction
                    ):
                        continue
                    baseline_key = (target_class, seed, method, from_fraction)
                    if baseline_key in indexed:
                        differences.append(current - indexed[baseline_key])
                values = np.asarray(differences, dtype=np.float64)
                if values.size == 0:
                    continue
                output.append(
                    {
                        "method": method,
                        "from_target_fraction": from_fraction,
                        "to_target_fraction": to_fraction,
                        "f1_delta_median": float(np.median(values)),
                        "f1_delta_q25": float(np.quantile(values, 0.25)),
                        "f1_delta_q75": float(np.quantile(values, 0.75)),
                        "f1_delta_mean": float(np.mean(values)),
                        "decline_fraction": float(np.mean(values < 0.0)),
                        "count": int(values.size),
                    }
                )
    return output


def _write_derived(rows: Sequence[dict], output: Path) -> None:
    if not rows:
        return
    write_rows_atomic(_summary_rows(rows), output.with_name(output.stem + "_summary.csv"))
    paired = _paired_rows(rows)
    if paired:
        write_rows_atomic(paired, output.with_name(output.stem + "_paired.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--query-size", type=int, default=DEFAULT_QUERY_SIZE)
    parser.add_argument(
        "--prevalences",
        type=float,
        nargs="+",
        default=DEFAULT_PREVALENCES,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--target-classes", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "emds7_target_prevalence.csv")
    samples = load_emds7(paths.dataset("emds7"))
    grouped = group_indices(samples)
    eligible_pool = sorted(
        label
        for label, indices in grouped.items()
        if len(indices) >= MINIMUM_POOL_CLASS_SIZE
    )
    targets = select_target_classes(
        grouped,
        len(samples),
        args.shot,
        args.query_size,
        args.prevalences,
    )
    if args.target_classes is not None:
        invalid = [label for label in args.target_classes if label not in targets]
        if invalid:
            parser.error("Ineligible target classes: {0}".format(invalid))
        targets = list(args.target_classes)
    if not targets:
        parser.error("No target classes satisfy the protocol")

    print(
        "EMDS-7 target prevalence: {0} targets, {1} seeds, fractions={2}".format(
            len(targets), len(args.seeds), ",".join(str(value) for value in args.prevalences)
        ),
        flush=True,
    )
    print(
        "Targets: {0}".format(
            ", ".join(
                "{0}({1})".format(label, len(grouped[label])) for label in targets
            )
        ),
        flush=True,
    )

    backbone = backbones.build_backbone(
        "dinov3",
        device_name(),
        model_path=paths.model("dinov3"),
        image_size=224,
    )
    batch_size = (
        args.batch
        if args.batch is not None
        else backbones.recommended_batch_size("dinov3")
    )
    features = normalize_rows(
        load_or_extract(
            backbone,
            samples,
            paths.output("cache"),
            "emds7_dinov3_224",
            batch_size,
        )
    )

    rows = read_rows(output) if args.resume else []
    completed_counts = defaultdict(int)
    for row in rows:
        completed_counts[
            (
                str(row["target_class"]),
                int(row["seed"]),
                float(row["target_fraction"]),
            )
        ] += 1

    incomplete_units = set()
    for target in targets:
        for seed in args.seeds:
            keys = [
                (target, seed, float(prevalence)) for prevalence in args.prevalences
            ]
            counts = [completed_counts.get(key, 0) for key in keys]
            if any(counts) and not all(count == len(METHODS) for count in counts):
                incomplete_units.add((target, seed))
    if incomplete_units:
        rows = [
            row
            for row in rows
            if (str(row["target_class"]), int(row["seed"])) not in incomplete_units
        ]
        completed_counts = defaultdict(int)
        for row in rows:
            completed_counts[
                (
                    str(row["target_class"]),
                    int(row["seed"]),
                    float(row["target_fraction"]),
                )
            ] += 1
        print(
            "Discarded {0} incomplete target-seed units".format(
                len(incomplete_units)
            ),
            flush=True,
        )

    total_units = len(targets) * len(args.seeds)
    completed_units = 0
    experiment_started = time.perf_counter()
    for target in targets:
        negative_classes = [label for label in eligible_pool if label != target]
        for seed in args.seeds:
            keys = [
                (target, seed, float(prevalence)) for prevalence in args.prevalences
            ]
            if all(completed_counts.get(key, 0) == len(METHODS) for key in keys):
                completed_units += 1
                print(
                    "Skip {0} seed={1} ({2}/{3})".format(
                        target, seed, completed_units, total_units
                    ),
                    flush=True,
                )
                continue

            unit_started = time.perf_counter()
            splits = build_paired_splits(
                grouped,
                target,
                negative_classes,
                seed,
                args.shot,
                args.query_size,
                args.prevalences,
            )
            payloads = []
            for prevalence in args.prevalences:
                prevalence = float(prevalence)
                support, query, labels = splits[prevalence]
                payloads.append(
                    (
                        features,
                        support,
                        query,
                        labels,
                        {
                            "target_class": target,
                            "target_total_images": len(grouped[target]),
                            "target_global_fraction": float(len(grouped[target]))
                            / float(len(samples)),
                            "seed": seed,
                            "shot": args.shot,
                            "query_count": args.query_size,
                            "positive_query_count": int(labels.sum()),
                            "negative_query_count": int(len(labels) - labels.sum()),
                            "target_fraction": prevalence,
                            "negative_class_count": len(negative_classes),
                        },
                    )
                )
            with ThreadPoolExecutor(max_workers=min(args.workers, len(payloads))) as executor:
                evaluated = list(executor.map(_evaluate_split, payloads))
            for key, result in zip(keys, evaluated):
                rows.extend(result)
                completed_counts[key] = len(result)
            write_rows_atomic(rows, output)
            _write_derived(rows, output)
            completed_units += 1
            elapsed = time.perf_counter() - experiment_started
            eta = elapsed / completed_units * (total_units - completed_units)
            print(
                "Completed {0} seed={1} ({2}/{3}) in {4:.2f}s; ETA {5:.1f}s".format(
                    target,
                    seed,
                    completed_units,
                    total_units,
                    time.perf_counter() - unit_started,
                    eta,
                ),
                flush=True,
            )

    _write_derived(rows, output)
    print("Completed target-prevalence experiment: {0}".format(output), flush=True)


if __name__ == "__main__":
    main()
