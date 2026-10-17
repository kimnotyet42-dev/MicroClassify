"""Measure feature extraction and post-processing runtime."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from dataio.datasets import Sample, load_emds7, load_self_collected
from dataio.protocols import microscopy_split
from experiments.cli import add_config_argument, device_name, output_path, runtime_paths
from methods import backbones, graph
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.results import read_rows, write_rows_atomic
from methods.whitening import full_whitening_matrix, normalize_rows


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def extract_timed(
    backbone,
    samples: Sequence[Sample],
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    backbone.load()
    vectors = []
    synchronize(backbone.device)
    started = time.perf_counter()
    for offset in tqdm(
        range(0, len(samples), batch_size),
        desc="runtime feature extraction",
    ):
        batch = samples[offset : offset + batch_size]
        vectors.append(backbone.embed([sample.open_image() for sample in batch]))
    synchronize(backbone.device)
    elapsed = time.perf_counter() - started
    return np.concatenate(vectors).astype(np.float32), elapsed


def propagation_seconds(
    features: np.ndarray,
    support_count: int,
    repeats: int,
) -> float:
    features = normalize_rows(features)
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        scores = graph.graph_propagate_score(
            features,
            metric="cos",
            seed_idx=list(range(support_count)),
            knn_k=DEFAULT_GRAPH_SETTINGS.neighbors,
            k_sigma=DEFAULT_GRAPH_SETTINGS.sigma_neighbors,
            sigma_method=DEFAULT_GRAPH_SETTINGS.sigma_method,
            graph_alpha=DEFAULT_GRAPH_SETTINGS.alpha,
            graph_max_iter=DEFAULT_GRAPH_SETTINGS.max_iterations,
            graph_tol=DEFAULT_GRAPH_SETTINGS.tolerance,
            eps=DEFAULT_GRAPH_SETTINGS.epsilon,
        )
        graph.zscore_subset(
            scores,
            list(range(support_count, len(features))),
            DEFAULT_GRAPH_SETTINGS.epsilon,
        )
        values.append(time.perf_counter() - started)
    return float(np.mean(values))


def whitening_seconds(
    features: np.ndarray,
    labels: np.ndarray,
    repeats: int,
) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        residual_parts = []
        for label in np.unique(labels):
            group = features[labels == label]
            residual_parts.append(group - group.mean(axis=0, keepdims=True))
        residuals = np.concatenate(residual_parts)
        full_whitening_matrix(residuals)
        values.append(time.perf_counter() - started)
    return float(np.mean(values))


def benchmark(
    name: str,
    samples: Sequence[Sample],
    limit: int,
    support_count: int,
    model_path: Path,
    image_size: int,
    batch_size: int,
    repeats: int,
) -> List[dict]:
    selected = list(samples[:limit])
    if len(selected) != limit:
        raise ValueError(
            "{0} needs {1} images; found {2}".format(name, limit, len(selected))
        )
    device = device_name()
    backbone = backbones.build_backbone(
        "dinov3",
        device,
        model_path=model_path,
        image_size=image_size,
    )
    features, feature_seconds = extract_timed(backbone, selected, batch_size)
    labels = np.asarray([sample.label for sample in selected])
    graph_features = features
    if name == "self_collected":
        split = microscopy_split(selected, "golden_mussel", seed=0, shot=support_count)
        order = split.support_groups[0] + split.query_indices
        graph_features = features[order]
    common = {
        "dataset": name,
        "image_count": len(selected),
        "support_count": support_count,
        "repeats": repeats,
    }
    return [
        {
            **common,
            "component": "feature_extraction",
            "seconds": feature_seconds,
        },
        {
            **common,
            "component": "graph_propagation_zscore",
            "seconds": propagation_seconds(graph_features, support_count, repeats),
        },
        {
            **common,
            "component": "intra_class_whitening",
            "seconds": whitening_seconds(features, labels, repeats),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("self_collected", "emds7"),
        default=("self_collected", "emds7"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    add_config_argument(parser)
    args = parser.parse_args()

    paths = runtime_paths(args.config)
    output = output_path(args.output, paths, "runtime_benchmark.csv")
    rows = read_rows(output) if args.resume else []
    completed = {row["dataset"] for row in rows}
    specifications = {
        "self_collected": (
            load_self_collected,
            390,
            5,
        ),
        "emds7": (
            load_emds7,
            1600,
            500,
        ),
    }
    for name in args.datasets:
        if name in completed:
            print("Skipping completed runtime dataset: {0}".format(name), flush=True)
            continue
        loader, limit, support_count = specifications[name]
        dataset_rows = benchmark(
            name,
            loader(paths.dataset(name)),
            limit,
            support_count,
            paths.model("dinov3"),
            args.image_size,
            args.batch,
            args.repeats,
        )
        rows.extend(dataset_rows)
        write_rows_atomic(rows, output)
        print("Completed runtime dataset: {0}".format(name), flush=True)
        for row in dataset_rows:
            print(row, flush=True)


if __name__ == "__main__":
    main()
