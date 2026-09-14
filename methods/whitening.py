from __future__ import annotations

import numpy as np
from typing import List


EPSILON = 1e-8


def normalize_rows(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + epsilon)


def transform_with_support_residuals(
    values: np.ndarray,
    residuals: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Apply analytical Ledoit--Wolf intra-class whitening.

    The eigendecomposition is evaluated in the support-residual subspace. This
    is mathematically equivalent to constructing the full feature covariance,
    but avoids a dense feature-dimension eigendecomposition for every episode.
    """

    values = np.asarray(values, dtype=np.float32)
    residuals = np.asarray(residuals, dtype=np.float32)
    sample_count, dimension = residuals.shape
    if sample_count < 1 or dimension != values.shape[1]:
        raise ValueError("residual and feature dimensions are inconsistent")

    gram = residuals @ residuals.T
    trace_covariance = float(np.trace(gram) / sample_count)
    mean_eigenvalue = trace_covariance / dimension
    covariance_frobenius = float(
        np.sum(gram * gram) / (sample_count * sample_count)
    )
    signal_variance = max(
        covariance_frobenius - trace_covariance * trace_covariance / dimension,
        0.0,
    )
    residual_norms = np.diag(gram)
    sampling_variance = float(
        (
            np.sum(residual_norms * residual_norms)
            - np.sum(gram * gram) / sample_count
        )
        / (sample_count * sample_count)
    )
    sampling_variance = min(max(sampling_variance, 0.0), signal_variance)
    shrinkage = (
        sampling_variance / signal_variance
        if signal_variance > epsilon
        else 1.0
    )

    _, singular_values, right_vectors = np.linalg.svd(
        residuals / np.sqrt(sample_count),
        full_matrices=False,
    )
    null_eigenvalue = max(shrinkage * mean_eigenvalue, epsilon)
    null_scale = null_eigenvalue ** -0.5
    active_eigenvalues = np.maximum(
        (1.0 - shrinkage) * singular_values * singular_values
        + shrinkage * mean_eigenvalue,
        epsilon,
    )
    corrections = active_eigenvalues ** -0.5 - null_scale
    transformed = (
        null_scale * values
        + ((values @ right_vectors.T) * corrections) @ right_vectors
    )
    return normalize_rows(transformed, epsilon)


def transform_with_support_residuals_full(
    values: np.ndarray,
    residuals: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Reference full-covariance whitening used by the FSCIL paper runs.

    The low-rank function above is mathematically equivalent and is retained
    for the episodic experiments.  FSCIL keeps this original floating-point
    path because its published per-split table contains a boundary prediction
    that changes under the low-rank rearrangement.
    """
    values = np.asarray(values, dtype=np.float32)
    transform = full_whitening_matrix(residuals, epsilon)
    return normalize_rows(values.dot(transform), epsilon)


def full_whitening_matrix(
    residuals: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    """Return the original full Ledoit--Wolf inverse square-root matrix."""
    residuals = np.asarray(residuals, dtype=np.float32)
    sample_count, dimension = residuals.shape
    covariance = residuals.T.dot(residuals) / sample_count
    mean_eigenvalue = np.trace(covariance) / dimension
    signal_variance = np.sum(
        (covariance - mean_eigenvalue * np.eye(dimension)) ** 2
    )
    residual_norms = np.sum(residuals * residuals, axis=1)
    sampling_variance = (
        np.sum(residual_norms ** 2)
        - 2.0 * np.sum((residuals.dot(covariance)) * residuals)
        + sample_count * np.sum(covariance ** 2)
    ) / (sample_count * sample_count)
    sampling_variance = min(float(sampling_variance), float(signal_variance))
    shrinkage = (
        float(sampling_variance / signal_variance)
        if signal_variance > epsilon
        else 1.0
    )
    shrunk = (
        (1.0 - shrinkage) * covariance
        + shrinkage * (np.trace(covariance) / dimension) * np.eye(dimension)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    eigenvalues = np.clip(eigenvalues, epsilon, None)
    return eigenvectors.dot(np.diag(eigenvalues ** -0.5)).dot(eigenvectors.T)


def support_residuals(
    features: np.ndarray,
    support_groups: List[List[int]],
) -> np.ndarray:
    residuals = []
    for indices in support_groups:
        class_features = features[indices]
        residuals.append(class_features - class_features.mean(axis=0))
    return np.concatenate(residuals, axis=0)
