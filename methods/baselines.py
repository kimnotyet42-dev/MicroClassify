"""Shared frozen-feature classifiers used by the comparison experiments."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import warnings
from typing import List, Sequence, Tuple

import numpy as np


LOGISTIC_C_VALUES = 10.0 ** np.linspace(-6.0, 5.0, 45)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A feature has zero L2 norm")
    return values / norms


def simpleshot_predict(
    features: np.ndarray,
    support_groups: Sequence[Sequence[int]],
    query_indices: Sequence[int],
) -> np.ndarray:
    """BioCLIP's official support-centering SimpleShot classifier."""
    support_indices = np.asarray(
        [index for group in support_groups for index in group], dtype=np.int64
    )
    support_labels = np.concatenate(
        [np.full(len(group), class_index, dtype=np.int64)
         for class_index, group in enumerate(support_groups)]
    )
    support = np.asarray(features[support_indices], dtype=np.float64)
    query = np.asarray(features[list(query_indices)], dtype=np.float64)
    support_mean = support.mean(axis=0)
    support = normalize_rows(support - support_mean)
    query = normalize_rows(query - support_mean)
    centroids = np.stack(
        [support[support_labels == class_index].mean(axis=0)
         for class_index in range(len(support_groups))]
    )
    squared_distance = np.sum(
        (centroids[:, None, :] - query[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distance, axis=0).astype(np.int64)


def _make_logistic_classifier(c_value: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(c_value),
        max_iter=1000,
        tol=1e-12,
        n_jobs=1,
    )


def _cross_validated_accuracy(
    c_value: float,
    support: np.ndarray,
    labels: np.ndarray,
    support_groups: Sequence[Sequence[int]],
) -> float:
    fold_count = min(len(group) for group in support_groups)
    scores = []
    group_offsets = []
    cursor = 0
    for group in support_groups:
        group_offsets.append(np.arange(cursor, cursor + len(group)))
        cursor += len(group)
    for fold in range(fold_count):
        validation = np.asarray([indices[fold] for indices in group_offsets])
        training = np.concatenate(
            [np.delete(indices, fold) for indices in group_offsets]
        )
        classifier = _make_logistic_classifier(c_value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(support[training], labels[training])
        scores.append(
            float(np.mean(classifier.predict(support[validation]) == labels[validation]))
        )
    return float(np.mean(scores))


def logistic_predict(
    features: np.ndarray,
    support_groups: Sequence[Sequence[int]],
    query_indices: Sequence[int],
    cv_jobs: int = 1,
    project_support_span: bool = True,
) -> Tuple[np.ndarray, float]:
    """Support-only cross-validated logistic regression used in the paper."""
    if min(len(group) for group in support_groups) < 2:
        raise ValueError("Logistic support CV needs at least two samples per class")
    support_indices = np.asarray(
        [index for group in support_groups for index in group], dtype=np.int64
    )
    support_labels = np.concatenate(
        [np.full(len(group), class_index, dtype=np.int64)
         for class_index, group in enumerate(support_groups)]
    )
    support = np.asarray(features[support_indices], dtype=np.float64)
    query = np.asarray(features[list(query_indices)], dtype=np.float64)
    if project_support_span:
        _, _, basis = np.linalg.svd(support, full_matrices=False)
        support = support.dot(basis.T)
        query = query.dot(basis.T)
    if os.name == "nt" and cv_jobs > 1:
        print(
            "Windows logistic-regression validation uses one worker to avoid "
            "concurrent LBFGS runtime failures.",
            flush=True,
        )
        cv_jobs = 1

    def evaluate_c(c_value: float) -> float:
        return _cross_validated_accuracy(
            float(c_value), support, support_labels, support_groups
        )

    if cv_jobs == 1:
        scores = [evaluate_c(float(value)) for value in LOGISTIC_C_VALUES]
    else:
        with ThreadPoolExecutor(max_workers=cv_jobs) as executor:
            scores = list(executor.map(evaluate_c, LOGISTIC_C_VALUES))
    best_c = float(LOGISTIC_C_VALUES[int(np.argmax(scores))])
    classifier = _make_logistic_classifier(best_c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(support, support_labels)
    return classifier.predict(query).astype(np.int64), best_c


def one_vs_rest_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    positive_class: int,
) -> float:
    predicted_positive = predictions == positive_class
    actual_positive = labels == positive_class
    true_positive = int(np.sum(predicted_positive & actual_positive))
    false_positive = int(np.sum(predicted_positive & ~actual_positive))
    false_negative = int(np.sum(~predicted_positive & actual_positive))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0
