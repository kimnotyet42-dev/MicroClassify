"""Module and frozen-backbone ablation on EMDS-7 and ZooLake2."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from dataio.datasets import load_emds7, load_zoolake2
from dataio.protocols import emds_allway_splits, zoolake_episodes
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from experiments.common import evaluate_multiclass_split
from methods import backbones
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.feature_cache import load_or_extract
from methods.results import read_rows, write_rows_atomic
from methods.whitening import normalize_rows


def dataset_and_splits(name: str, paths):
    if name == "emds7":
        samples = load_emds7(paths.dataset("emds7"))
        return samples, emds_allway_splits(samples, shot=4, seeds=5)
    if name == "zoolake2":
        samples = load_zoolake2(paths.dataset("zoolake2"))
        return samples, zoolake_episodes(
            samples,
            way=32,
            shot=5,
            query=15,
            episodes=200,
        )
    raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("emds7", "zoolake2"),
        default=("emds7", "zoolake2"),
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("dinov2", "dinov3", "pecore"),
        default=("dinov2", "dinov3", "pecore"),
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    add_config_argument(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    cache_root = paths.output("cache")
    device = device_name()
    output = output_path(args.output, paths, "module_backbone_ablation.csv")
    rows = read_rows(output) if args.resume else []
    completed = {(row["dataset"], row["backbone"]) for row in rows}

    for dataset_name in args.datasets:
        samples, splits = dataset_and_splits(dataset_name, paths)
        print(
            f"{dataset_name}: images={len(samples)}, splits={len(splits)}",
            flush=True,
        )
        for backbone_name in args.backbones:
            if (dataset_name, backbone_name) in completed:
                print(f"Skip completed {dataset_name}/{backbone_name}", flush=True)
                continue
            backbone = backbones.build_backbone(
                backbone_name,
                device,
                model_path=paths.model(backbone_name),
                pe_repo=(
                    paths.optional_model("pe_repo")
                    if backbone_name == "pecore"
                    else None
                ),
            )
            batch_size = (
                args.batch
                if args.batch is not None
                else backbones.recommended_batch_size(backbone_name)
            )
            features = normalize_rows(
                load_or_extract(
                    backbone,
                    samples,
                    cache_root,
                    f"{dataset_name}_{backbone_name}_{backbone.image_size}",
                    batch_size,
                )
            )
            metrics = defaultdict(list)  # type: Dict[str, List[float]]
            for split in tqdm(
                splits,
                desc=f"{dataset_name}/{backbone_name} evaluation",
            ):
                result = evaluate_multiclass_split(
                    features,
                    split,
                    DEFAULT_GRAPH_SETTINGS,
                )
                for method, value in result.items():
                    metrics[method].append(100.0 * value)
            row = {
                "dataset": dataset_name,
                "backbone": backbone_name,
                "resolution": backbone.image_size,
                "splits": len(splits),
            }
            for method in ("nc", "w_nc", "graph_z", "w_graph_z"):
                values = np.asarray(metrics[method])
                row[f"{method}_mean"] = float(values.mean())
                row[f"{method}_std"] = float(values.std())
            rows.append(row)
            write_rows_atomic(rows, output)
            print(
                f"Completed {dataset_name}/{backbone_name}: "
                f"NC={row['nc_mean']:.2f}, W+NC={row['w_nc_mean']:.2f}, "
                f"Graph-z={row['graph_z_mean']:.2f}, "
                f"W+Graph-z={row['w_graph_z_mean']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
