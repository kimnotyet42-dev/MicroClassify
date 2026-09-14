from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dataio.datasets import Sample
from dataio.protocols import FewShotSplit, emds_fscil_splits
from experiments.common import (
    binary_stage_scores,
    graph_z_accuracy,
    graph_z_accuracy_from_distance,
    localize_split,
)
from methods import graph
from methods.baselines import simpleshot_predict
from methods.config import DEFAULT_GRAPH_SETTINGS
from methods.whitening import normalize_rows, transform_with_support_residuals


class CoreEquivalenceTests(unittest.TestCase):
    def test_reused_binary_graph_matches_repeated_graph_build(self):
        rng = np.random.RandomState(7)
        features = rng.normal(size=(18, 12)).astype(np.float32)
        split = FewShotSplit(
            support_groups=[[0, 1], [2, 3]],
            query_indices=list(range(4, 18)),
            query_labels=np.zeros(14, dtype=np.int64),
        )
        optimized = binary_stage_scores(features, split, DEFAULT_GRAPH_SETTINGS)
        normalized = normalize_rows(features)
        distance = graph.pairwise_distance_matrix(normalized, "cos")
        transition = graph.build_knn_transition(
            distance,
            DEFAULT_GRAPH_SETTINGS.neighbors,
            DEFAULT_GRAPH_SETTINGS.sigma_neighbors,
            DEFAULT_GRAPH_SETTINGS.sigma_method,
        )
        reference_columns = []
        for support in split.support_groups:
            seeds = np.zeros(len(normalized), dtype=np.float32)
            seeds[support] = 1.0
            propagated = seeds.copy()
            for _ in range(DEFAULT_GRAPH_SETTINGS.max_iterations):
                updated = (
                    DEFAULT_GRAPH_SETTINGS.alpha * transition.dot(propagated)
                    + (1.0 - DEFAULT_GRAPH_SETTINGS.alpha) * seeds
                )
                delta = float(np.max(np.abs(updated - propagated)))
                propagated = updated.astype(np.float32)
                if delta < DEFAULT_GRAPH_SETTINGS.tolerance:
                    break
            raw = propagated[split.query_indices]
            reference_columns.append(
                (raw - raw.mean()) / (raw.std() + DEFAULT_GRAPH_SETTINGS.epsilon)
            )
        reference = np.max(np.stack(reference_columns, axis=1), axis=1)
        np.testing.assert_allclose(optimized, reference, rtol=0.0, atol=1e-6)

    def test_reused_distance_matches_direct_graph_evaluation(self):
        rng = np.random.RandomState(9)
        features = normalize_rows(rng.normal(size=(20, 10)).astype(np.float32))
        split = FewShotSplit(
            support_groups=[[0, 1], [2, 3]],
            query_indices=list(range(4, 20)),
            query_labels=np.asarray([0] * 8 + [1] * 8, dtype=np.int64),
        )
        direct = graph_z_accuracy(features, split, DEFAULT_GRAPH_SETTINGS)
        nodes, support_local, query_local = localize_split(split)
        distance = graph.pairwise_distance_matrix(features[nodes], "cos")
        reused = graph_z_accuracy_from_distance(
            distance,
            support_local,
            query_local,
            split.query_labels,
            DEFAULT_GRAPH_SETTINGS,
        )
        self.assertEqual(direct, reused)

    def test_low_rank_whitening_matches_full_covariance(self):
        rng = np.random.RandomState(11)
        values = rng.normal(size=(17, 9)).astype(np.float32)
        residuals = rng.normal(size=(7, 9)).astype(np.float32)
        optimized = transform_with_support_residuals(values, residuals)

        count, dimension = residuals.shape
        covariance = residuals.T.dot(residuals) / count
        mu = np.trace(covariance) / dimension
        signal = np.sum((covariance - mu * np.eye(dimension)) ** 2)
        norms = np.sum(residuals * residuals, axis=1)
        sampling = (
            np.sum(norms ** 2)
            - 2.0 * np.sum((residuals.dot(covariance)) * residuals)
            + count * np.sum(covariance ** 2)
        ) / (count * count)
        sampling = min(max(float(sampling), 0.0), float(signal))
        shrinkage = sampling / signal if signal > 1e-8 else 1.0
        shrunk = (1.0 - shrinkage) * covariance + shrinkage * mu * np.eye(dimension)
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
        whitening = eigenvectors.dot(
            np.diag(np.maximum(eigenvalues, 1e-8) ** -0.5)
        ).dot(eigenvectors.T)
        reference = normalize_rows(values.dot(whitening))
        np.testing.assert_allclose(optimized, reference, rtol=2e-5, atol=2e-5)

    def test_simpleshot_small_case(self):
        features = np.asarray(
            [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0],
             [3.0, 0.1], [0.1, 3.0]],
            dtype=np.float64,
        )
        predictions = simpleshot_predict(features, [[0, 1], [2, 3]], [4, 5])
        self.assertEqual(predictions.tolist(), [0, 1])

    def test_fscil_protocol_has_no_support_test_overlap(self):
        samples = []
        for class_index in range(6):
            for image_index in range(12):
                samples.append(
                    Sample(
                        "c{0}/{1}".format(class_index, image_index),
                        Path("unused"),
                        "c{0}".format(class_index),
                    )
                )
        splits = emds_fscil_splits(
            samples,
            base_shot=4,
            seeds=2,
            base_classes=3,
            incremental_way=1,
            incremental_shot=2,
        )
        self.assertEqual(len(splits), 2)
        for split in splits:
            for label in split.support_by_class:
                self.assertFalse(
                    set(split.support_by_class[label])
                    & set(split.test_by_class[label])
                )


if __name__ == "__main__":
    unittest.main()
