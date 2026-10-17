from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataio.protocols import FewShotSplit
from methods import graph
from methods.config import GraphSettings
from methods.whitening import (
    normalize_rows,
    support_residuals,
    transform_with_support_residuals,
    transform_with_support_residuals_full,
)


def localize_split(split: FewShotSplit) -> Tuple[List[int], List[List[int]], List[int]]:
    flat_support = [index for group in split.support_groups for index in group]
    nodes = flat_support + list(split.query_indices)
    support_local = []
    cursor = 0
    for group in split.support_groups:
        support_local.append(list(range(cursor, cursor + len(group))))
        cursor += len(group)
    query_local = list(range(cursor, len(nodes)))
    return nodes, support_local, query_local


def zscore_columns(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return (scores - scores.mean(axis=0, keepdims=True)) / (
        scores.std(axis=0, keepdims=True) + epsilon
    )


def centroid_scores(
    features: np.ndarray,
    support_groups: List[List[int]],
    query_indices: List[int],
) -> np.ndarray:
    prototypes = normalize_rows(
        np.stack([features[indices].mean(axis=0) for indices in support_groups])
    )
    return normalize_rows(features[query_indices]) @ prototypes.T


def graph_scores(
    features: np.ndarray,
    support_groups: List[List[int]],
    query_indices: List[int],
    settings: GraphSettings,
) -> np.ndarray:
    settings.validate()
    scores = graph.graph_propagate_nway(
        x=np.asarray(features, dtype=np.float32),
        metric="cos",
        class_seed_indices=support_groups,
        knn_k=settings.neighbors,
        k_sigma=min(settings.sigma_neighbors, settings.neighbors),
        sigma_method=settings.sigma_method,
        graph_alpha=settings.alpha,
        graph_max_iter=settings.max_iterations,
        graph_tol=settings.tolerance,
        eps=settings.epsilon,
    )
    return scores[query_indices]


def evaluate_multiclass_split(
    all_features: np.ndarray,
    split: FewShotSplit,
    settings: GraphSettings,
) -> Dict[str, float]:
    all_features = np.asarray(all_features, dtype=np.float32)
    nodes, support_local, query_local = localize_split(split)
    episode_features = all_features[nodes]
    labels = split.query_labels

    nearest_centroid = centroid_scores(episode_features, support_local, query_local)
    propagated = graph_scores(episode_features, support_local, query_local, settings)

    residuals = support_residuals(all_features, split.support_groups)
    whitened = transform_with_support_residuals(episode_features, residuals)
    whitened_centroid = centroid_scores(whitened, support_local, query_local)
    whitened_propagated = graph_scores(whitened, support_local, query_local, settings)

    return {
        "nc": float(np.mean(nearest_centroid.argmax(axis=1) == labels)),
        "w_nc": float(np.mean(whitened_centroid.argmax(axis=1) == labels)),
        "graph_z": float(np.mean(zscore_columns(propagated).argmax(axis=1) == labels)),
        "w_graph_z": float(
            np.mean(zscore_columns(whitened_propagated).argmax(axis=1) == labels)
        ),
    }


def graph_z_accuracy(
    all_features: np.ndarray,
    split: FewShotSplit,
    settings: GraphSettings,
) -> float:
    all_features = np.asarray(all_features, dtype=np.float32)
    nodes, support_local, query_local = localize_split(split)
    scores = graph_scores(all_features[nodes], support_local, query_local, settings)
    return float(np.mean(zscore_columns(scores).argmax(axis=1) == split.query_labels))


def whitened_graph_z_accuracy(
    all_features: np.ndarray,
    split: FewShotSplit,
    settings: GraphSettings,
) -> float:
    """Evaluate only the complete whitening + Graph-z pipeline."""
    all_features = np.asarray(all_features, dtype=np.float32)
    nodes, support_local, query_local = localize_split(split)
    residuals = support_residuals(all_features, split.support_groups)
    whitened = transform_with_support_residuals(all_features[nodes], residuals)
    scores = graph_scores(whitened, support_local, query_local, settings)
    return float(np.mean(zscore_columns(scores).argmax(axis=1) == split.query_labels))


def whitened_graph_z_accuracy_full(
    all_features: np.ndarray,
    split: FewShotSplit,
    settings: GraphSettings,
) -> float:
    """Evaluate the original full-covariance whitening + Graph-z path."""
    all_features = np.asarray(all_features, dtype=np.float32)
    nodes, support_local, query_local = localize_split(split)
    residuals = support_residuals(all_features, split.support_groups)
    whitened = transform_with_support_residuals_full(all_features[nodes], residuals)
    scores = graph_scores(whitened, support_local, query_local, settings)
    return float(np.mean(zscore_columns(scores).argmax(axis=1) == split.query_labels))


def graph_z_accuracy_from_distance(
    distance: np.ndarray,
    support_local: List[List[int]],
    query_local: List[int],
    query_labels: np.ndarray,
    settings: GraphSettings,
) -> float:
    """Evaluate Graph-z while reusing an episode's pairwise distance matrix."""
    scores = graph.graph_propagate_nway_from_dist(
        distance,
        support_local,
        knn_k=settings.neighbors,
        k_sigma=min(settings.sigma_neighbors, settings.neighbors),
        sigma_method=settings.sigma_method,
        graph_alpha=settings.alpha,
        graph_max_iter=settings.max_iterations,
        graph_tol=settings.tolerance,
        eps=settings.epsilon,
    )
    query_scores = scores[query_local]
    return float(np.mean(zscore_columns(query_scores).argmax(axis=1) == query_labels))


def binary_target_scores(
    all_features: np.ndarray,
    split: FewShotSplit,
    settings: GraphSettings,
) -> np.ndarray:
    """Propagate one target class and calibrate scores over its query set."""
    if len(split.support_groups) != 1:
        raise ValueError("Binary target recognition requires one positive support group")
    all_features = normalize_rows(all_features)
    nodes, support_local, query_local = localize_split(split)
    episode_features = all_features[nodes]
    transition = graph.transition_from_features(
        episode_features,
        metric="cos",
        knn_k=settings.neighbors,
        k_sigma=min(settings.sigma_neighbors, settings.neighbors),
        sigma_method=settings.sigma_method,
        eps=settings.epsilon,
    )
    seeds = np.zeros(len(episode_features), dtype=np.float32)
    seeds[support_local[0]] = 1.0
    raw, _, _ = graph.propagate_from_transition(
        transition,
        seeds,
        graph_alpha=settings.alpha,
        graph_max_iter=settings.max_iterations,
        graph_tol=settings.tolerance,
    )
    query_scores = raw[query_local]
    return (query_scores - query_scores.mean()) / (query_scores.std() + settings.epsilon)


def settings_with(
    settings: GraphSettings,
    *,
    neighbors: Optional[int] = None,
    sigma_method: Optional[str] = None,
) -> GraphSettings:
    return replace(
        settings,
        neighbors=settings.neighbors if neighbors is None else neighbors,
        sigma_method=settings.sigma_method if sigma_method is None else sigma_method,
    )
