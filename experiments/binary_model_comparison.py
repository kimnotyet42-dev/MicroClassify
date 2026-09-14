"""All-class-support classifiers converted to EMDS-7 one-vs-rest decisions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dataio.datasets import load_emds7
from dataio.protocols import emds_class_pool, group_indices
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from methods import backbones
from methods.baselines import logistic_predict, one_vs_rest_f1, simpleshot_predict
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic


CLASS_POOL_SIZES = (5, 10, 15, 20, 25, 30, 35)
SEEDS = (0, 1, 2)
PAPER_METHODS = {
    "dinov3_logistic": ("dinov3", "logistic"),
    "dinov3_simpleshot": ("dinov3", "simpleshot"),
    "bioclip_logistic": ("bioclip", "logistic"),
    "openai_clip_logistic": ("openai_clip", "logistic"),
}


def support_for_classes(
    grouped: Dict[str, List[int]],
    selected: Sequence[str],
    shot: int,
    seed: int,
) -> List[List[int]]:
    output = []
    for label in selected:
        indices = np.asarray(grouped[label], dtype=np.int64).copy()
        np.random.RandomState(seed).shuffle(indices)
        output.append(indices[:shot].tolist())
    return output


def summarize(rows: List[dict], output: Path) -> None:
    grouped = defaultdict(list)  # type: Dict[Tuple[str, str, int], List[float]]
    for row in rows:
        grouped[
            (str(row["backbone"]), str(row["classifier"]), int(row["class_pool_size"]))
        ].append(float(row["f1"]))
    summary = []
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=np.float64)
        summary.append(
            {
                "backbone": key[0],
                "classifier": key[1],
                "class_pool_size": key[2],
                "aggregation": "median",
                "f1": float(np.median(values)),
                "count": len(values),
            }
        )
    if summary:
        write_rows_atomic(summary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument(
        "--class-pool-sizes",
        type=int,
        nargs="+",
        default=CLASS_POOL_SIZES,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(PAPER_METHODS),
        default=tuple(PAPER_METHODS),
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--cv-jobs", type=int, default=1)
    parser.add_argument(
        "--no-support-projection",
        action="store_true",
        help="Disable the mathematically equivalent support-span speedup",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "binary_model_comparison.csv")
    samples = load_emds7(paths.dataset("emds7"))
    grouped = group_indices(samples)
    all_labels = np.asarray([sample.label for sample in samples])
    rows = read_rows(output) if args.resume else []

    for method_id in args.methods:
        backbone_name, classifier_name = PAPER_METHODS[method_id]
        backbone = backbones.build_backbone(
            backbone_name,
            device_name(),
            model_path=paths.model(backbone_name),
        )
        features = load_or_extract(
            backbone,
            samples,
            paths.output("cache"),
            "emds7_{0}_{1}".format(backbone_name, backbone.image_size),
            args.batch if args.batch is not None else backbones.recommended_batch_size(backbone_name),
        )
        for class_pool_size in args.class_pool_sizes:
            if class_pool_size not in CLASS_POOL_SIZES:
                parser.error(
                    "Unsupported class-pool size: {0}".format(class_pool_size)
                )
            expected = (
                (1 if class_pool_size == max(CLASS_POOL_SIZES) else 3)
                * class_pool_size
                * len(SEEDS)
            )
            existing = [
                row for row in rows
                if row["backbone"] == backbone_name
                and row["classifier"] == classifier_name
                and int(row["class_pool_size"]) == class_pool_size
            ]
            if len(existing) == expected:
                continue
            if existing:
                rows = [
                    row for row in rows
                    if not (
                        row["backbone"] == backbone_name
                        and row["classifier"] == classifier_name
                        and int(row["class_pool_size"]) == class_pool_size
                    )
                ]

            repeats = 1 if class_pool_size == max(CLASS_POOL_SIZES) else 3
            unit_rows = []
            for repeat in range(repeats):
                pool_indices, selected = emds_class_pool(
                    samples, class_pool_size, repeat
                )
                pool_set = set(pool_indices)
                for seed in SEEDS:
                    support_groups = support_for_classes(
                        grouped, selected, args.shot, seed
                    )
                    support_set = {
                        index for group in support_groups for index in group
                    }
                    query_indices = sorted(pool_set - support_set)
                    label_to_index = {
                        label: class_index
                        for class_index, label in enumerate(selected)
                    }
                    query_labels = np.asarray(
                        [label_to_index[all_labels[index]] for index in query_indices],
                        dtype=np.int64,
                    )
                    if classifier_name == "simpleshot":
                        predictions = simpleshot_predict(
                            features, support_groups, query_indices
                        )
                        selected_c = ""
                    else:
                        predictions, selected_c = logistic_predict(
                            features,
                            support_groups,
                            query_indices,
                            cv_jobs=args.cv_jobs,
                            project_support_span=not args.no_support_projection,
                        )
                    for class_index, class_name in enumerate(selected):
                        unit_rows.append(
                            {
                                "dataset": "emds7",
                                "backbone": backbone_name,
                                "classifier": classifier_name,
                                "class_pool_size": class_pool_size,
                                "repeat": repeat,
                                "seed": seed,
                                "target_class": class_name,
                                "shot": args.shot,
                                "support_count": sum(len(group) for group in support_groups),
                                "query_count": len(query_indices),
                                "selected_c": selected_c,
                                "f1": 100.0 * one_vs_rest_f1(
                                    predictions, query_labels, class_index
                                ),
                            }
                        )
            rows.extend(unit_rows)
            write_rows_atomic(rows, output)
            summarize(rows, output.with_name(output.stem + "_summary.csv"))
            print(
                "Completed {0}/{1}, N={2}".format(
                    backbone_name, classifier_name, class_pool_size
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
