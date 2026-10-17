from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def stats_dict(prefix: str, x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        result = {
            "{0}_{1}".format(prefix, key): None
            for key in ["min", "p25", "median", "mean", "p75", "max", "std"]
        }
        result["{0}_count".format(prefix)] = 0
        return result
    return {
        f"{prefix}_count": int(x.size),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def pairwise_distance_matrix(x: np.ndarray, metric: str, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix: {0}".format(x.shape))
    if metric == "cos":
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
        sim = np.clip(x @ x.T, -1.0, 1.0)
        d = 1.0 - sim
        np.fill_diagonal(d, 0.0)
        return d.astype(np.float32)
    if metric == "l2":
        sq = np.sum(x * x, axis=1, keepdims=True)
        d2 = np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)
        d = np.sqrt(d2)
        np.fill_diagonal(d, 0.0)
        return d.astype(np.float32)
    raise ValueError("Unknown distance metric: {0}".format(metric))


def sigma_from_knn_dist(
    neighbor_dist: np.ndarray,
    order: np.ndarray,
    dist: np.ndarray,
    sigma_method: str,
    k_sigma: int,
    eps: float = 1e-8,
) -> np.ndarray:
    n = dist.shape[0]
    if sigma_method == "k7":
        sigma_idx = order[:, int(k_sigma)]
        sigma = dist[np.arange(n), sigma_idx].astype(np.float32)
        return np.maximum(sigma, eps).astype(np.float32)
    if sigma_method == "log_mean":
        sigma = np.exp(np.mean(np.log(neighbor_dist.astype(np.float64) + eps), axis=1))
        return np.maximum(sigma, eps).astype(np.float32)
    if sigma_method == "local_median":
        sigma = np.median(neighbor_dist.astype(np.float64), axis=1)
        return np.maximum(sigma, eps).astype(np.float32)
    if sigma_method == "global_median":
        upper = dist[np.triu_indices(n, k=1)]
        upper = upper[np.isfinite(upper) & (upper > eps)]
        if upper.size == 0:
            raise RuntimeError("Cannot estimate global-median bandwidth")
        sigma = np.full(n, np.median(upper), dtype=np.float32)
        return np.maximum(sigma, eps).astype(np.float32)
    raise ValueError("Unknown bandwidth method: {0}".format(sigma_method))


def build_knn_transition(
    dist: np.ndarray,
    knn_k: int,
    k_sigma: int,
    sigma_method: str = "k7",
    eps: float = 1e-8,
    return_info: bool = False,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], np.ndarray]:
    n = dist.shape[0]
    if dist.shape != (n, n):
        raise ValueError("Distance matrix must be square: {0}".format(dist.shape))
    if not (1 <= int(knn_k) < n):
        raise ValueError("knn_k={0} is invalid for n={1}".format(knn_k, n))
    if not (1 <= int(k_sigma) < n):
        raise ValueError("k_sigma={0} is invalid for n={1}".format(k_sigma, n))
    if int(k_sigma) > int(knn_k):
        raise ValueError("k_sigma={0} exceeds knn_k={1}".format(k_sigma, knn_k))

    k = int(knn_k)
    # Retrieve the k neighbors plus self without fully sorting every row.
    rows = np.arange(n)[:, None]
    part = np.argpartition(dist, kth=k, axis=1)[:, :k + 1]
    order = part[rows, np.argsort(dist[rows, part], axis=1)]
    knn = order[:, 1:k + 1]
    neighbor_dist = dist[rows, knn].astype(np.float32)
    sigma = sigma_from_knn_dist(neighbor_dist, order, dist, sigma_method, int(k_sigma), eps)

    # Vectorized equivalent of assigning Gaussian weights node by node.
    denom = 2.0 * sigma[:, None].astype(np.float64) * sigma[knn].astype(np.float64) + eps
    exponent = -(neighbor_dist.astype(np.float64) ** 2) / denom
    wij = np.exp(exponent)
    # A selected kNN edge must not disappear solely because its Gaussian
    # weight underflows when converted to float32 (most visible for k=1).
    wij = np.maximum(wij, eps).astype(np.float32)
    w = np.zeros((n, n), dtype=np.float32)
    w[rows, knn] = wij
    w = np.maximum(w, w.T)
    row_sum = w.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 1e-12):
        bad = np.where(row_sum.ravel() <= 1e-12)[0]
        raise RuntimeError("kNN graph has isolated nodes: {0}".format(bad[:10].tolist()))
    trans = (w / row_sum).astype(np.float32)

    info = {}  # type: Dict[str, Any]
    if return_info:
        d_sigma = dist[np.arange(n), order[:, int(k_sigma)]].astype(np.float32)
        degree = np.sum(w > 0, axis=1).astype(np.float64)
        weights = w[w > 0].astype(np.float64)
        sigma_log = np.exp(np.mean(np.log(neighbor_dist.astype(np.float64) + eps), axis=1)).astype(np.float32)
        info = {
            "sigma_method": sigma_method,
            "n": int(n),
            "knn_k": int(knn_k),
            "k_sigma": int(k_sigma),
            **stats_dict("sigma", sigma),
            **stats_dict("d_sigma", d_sigma),
            **stats_dict("sigma_log_mean", sigma_log),
            **stats_dict("sigma_over_d_sigma", sigma / (d_sigma + eps)),
            **stats_dict("degree_after_sym", degree),
            **stats_dict("edge_weight", weights),
        }
    return (trans, info) if return_info else trans


def graph_propagate_score(
    x: np.ndarray,
    metric: str,
    seed_idx: List[int],
    knn_k: int,
    k_sigma: int,
    sigma_method: str = "k7",
    graph_alpha: float = 0.99,
    graph_max_iter: int = 200,
    graph_tol: float = 1e-7,
    eps: float = 1e-8,
    return_info: bool = False,
) -> Union[Tuple[np.ndarray, Dict[str, Any]], np.ndarray]:
    if not seed_idx:
        raise RuntimeError("seed_idx is empty")
    n = np.asarray(x).shape[0]
    for i in seed_idx:
        if i < 0 or i >= n:
            raise IndexError("seed_idx out of range: {0}, n={1}".format(i, n))
    dist = pairwise_distance_matrix(x, metric, eps)
    trans, info = build_knn_transition(
        dist, knn_k, k_sigma, sigma_method, eps, return_info=True
    )
    y = np.zeros(n, dtype=np.float32)
    y[seed_idx] = 1.0
    f, n_iter, last_delta = propagate_from_transition(
        trans,
        y,
        graph_alpha=graph_alpha,
        graph_max_iter=graph_max_iter,
        graph_tol=graph_tol,
    )
    info.update({
        "metric": metric,
        "alpha": float(graph_alpha),
        "graph_max_iter": int(graph_max_iter),
        "graph_tol": float(graph_tol),
        "n_iter": int(n_iter),
        "last_delta": last_delta,
        "seed_count": int(len(seed_idx)),
    })
    return (f.astype(np.float32), info) if return_info else f.astype(np.float32)


def graph_propagate_nway(
    x: np.ndarray,
    metric: str,
    class_seed_indices: List[List[int]],
    knn_k: int,
    k_sigma: int,
    sigma_method: str = "log_mean",
    graph_alpha: float = 0.99,
    graph_max_iter: int = 200,
    graph_tol: float = 1e-7,
    eps: float = 1e-8,
) -> np.ndarray:
    """N-way label propagation. Returns score matrix (n_nodes, n_classes).

    Builds the transition matrix once and propagates all class labels
    simultaneously — N times faster than running N binary propagations.
    """
    n = x.shape[0]
    n_classes = len(class_seed_indices)
    dist = pairwise_distance_matrix(x, metric, eps)
    trans = build_knn_transition(dist, knn_k, k_sigma, sigma_method, eps, return_info=False)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    for c, seeds in enumerate(class_seed_indices):
        for i in seeds:
            Y[i, c] = 1.0
    F, _, _ = propagate_from_transition(
        trans,
        Y,
        graph_alpha=graph_alpha,
        graph_max_iter=graph_max_iter,
        graph_tol=graph_tol,
    )
    return F  # (n_nodes, n_classes)


def graph_propagate_nway_from_dist(
    dist: np.ndarray,
    class_seed_indices: List[List[int]],
    knn_k: int,
    k_sigma: int,
    sigma_method: str = "log_mean",
    graph_alpha: float = 0.99,
    graph_max_iter: int = 200,
    graph_tol: float = 1e-7,
    eps: float = 1e-8,
) -> np.ndarray:
    """Same as graph_propagate_nway but accepts a precomputed distance matrix.
    Skips the internal pairwise_distance_matrix step entirely.
    """
    n = dist.shape[0]
    n_classes = len(class_seed_indices)
    trans = build_knn_transition(dist, knn_k, k_sigma, sigma_method, eps, return_info=False)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    for c, seeds in enumerate(class_seed_indices):
        for i in seeds:
            Y[i, c] = 1.0
    F, _, _ = propagate_from_transition(
        trans,
        Y,
        graph_alpha=graph_alpha,
        graph_max_iter=graph_max_iter,
        graph_tol=graph_tol,
    )
    return F


def propagate_from_transition(
    transition: np.ndarray,
    seeds: np.ndarray,
    graph_alpha: float = 0.99,
    graph_max_iter: int = 200,
    graph_tol: float = 1e-7,
) -> Tuple[np.ndarray, int, Optional[float]]:
    """Propagate one or more seed columns on a precomputed transition matrix."""
    seeds = np.asarray(seeds, dtype=np.float32)
    propagated = seeds.copy()
    last_delta = None  # type: Optional[float]
    iterations = 0
    for iteration in range(int(graph_max_iter)):
        updated = graph_alpha * (transition @ propagated) + (1.0 - graph_alpha) * seeds
        last_delta = float(np.max(np.abs(updated - propagated)))
        propagated = np.asarray(updated, dtype=np.float32)
        iterations = iteration + 1
        if last_delta < graph_tol:
            break
    return propagated, iterations, last_delta


def transition_from_features(
    features: np.ndarray,
    metric: str,
    knn_k: int,
    k_sigma: int,
    sigma_method: str = "log_mean",
    eps: float = 1e-8,
) -> np.ndarray:
    """Build a transition matrix once for reuse by several seed configurations."""
    distance = pairwise_distance_matrix(features, metric, eps)
    return build_knn_transition(
        distance,
        knn_k,
        k_sigma,
        sigma_method,
        eps,
        return_info=False,
    )


def zscore_subset(scores: np.ndarray, subset_idx: List[int], eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(scores, dtype=np.float64)[subset_idx]
    mean = float(v.mean())
    std = float(v.std())
    return ((scores - mean) / (std + eps)).astype(np.float32)
