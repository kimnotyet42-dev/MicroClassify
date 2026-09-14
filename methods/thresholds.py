from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde
from skimage.filters import threshold_otsu


@dataclass(frozen=True)
class ThresholdResult:
    method: str
    threshold: float
    f1: float
    gamma: Optional[float] = None


def binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    false_negative = int(np.sum(~predictions & (labels == 1)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def oracle_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positives = int(sorted_labels.sum())
    true_positive = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, len(scores) + 1)
    f1 = 2.0 * true_positive / (predicted_positive + positives)
    boundary = np.empty(len(scores), dtype=bool)
    boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    boundary[-1] = True
    best = int(np.argmax(np.where(boundary, f1, -1.0)))
    count = best + 1
    threshold = (
        0.5 * (sorted_scores[count - 1] + sorted_scores[count])
        if count < len(scores)
        else sorted_scores[-1] - 1e-6
    )
    return float(threshold), float(f1[best])


def score_grid(scores: np.ndarray, points: int = 2048, padding: float = 0.08) -> np.ndarray:
    minimum = float(scores.min())
    maximum = float(scores.max())
    span = maximum - minimum
    if span <= 0:
        raise ValueError("scores must not be constant")
    return np.linspace(minimum - padding * span, maximum + padding * span, points)


def kde_density(
    scores: np.ndarray,
    grid: np.ndarray,
    gamma: Optional[float],
) -> np.ndarray:
    weights = None
    if gamma is not None:
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        weights = normalized ** gamma
    density = np.asarray(
        gaussian_kde(scores, weights=weights, bw_method="scott")(grid),
        dtype=np.float64,
    )
    density[~np.isfinite(density)] = 0.0
    density[density < 0] = 0.0
    area = _trapezoid(density, grid)
    return density / area if area > 0 else density


def otsu_from_density(grid: np.ndarray, density: np.ndarray) -> float:
    probability = density / density.sum()
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * grid)
    total_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    numerator = (total_mean * cumulative_probability - cumulative_mean) ** 2
    variance = np.zeros_like(denominator)
    np.divide(numerator, denominator, out=variance, where=denominator > 1e-12)
    return float(grid[int(np.argmax(variance))])


def right_area_fraction(
    scores: np.ndarray,
    grid: np.ndarray,
    gamma: float,
) -> float:
    midpoint = 0.5 * (float(scores.min()) + float(scores.max()))
    split = max(1, min(int(np.searchsorted(grid, midpoint)), len(grid) - 1))
    density = kde_density(scores, grid, gamma)
    left = _trapezoid(density[:split], grid[:split])
    right = _trapezoid(density[split:], grid[split:])
    return right / (left + right) if left + right > 0 else 0.5


def select_gamma(
    scores: np.ndarray,
    grid: np.ndarray,
    maximum: float = 10.0,
    iterations: int = 40,
    tolerance: float = 1e-4,
) -> float:
    """Find the first right-dominant gamma to the requested bracket precision."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if right_area_fraction(scores, grid, 1.0) > 0.5:
        return 1.0
    if right_area_fraction(scores, grid, maximum) <= 0.5:
        return 1.0
    lower, upper = 1.0, maximum
    for _ in range(iterations):
        if upper - lower <= tolerance:
            break
        middle = 0.5 * (lower + upper)
        if right_area_fraction(scores, grid, middle) > 0.5:
            upper = middle
        else:
            lower = middle
    return float(upper)


def evaluate_thresholds(scores: np.ndarray, labels: np.ndarray) -> List[ThresholdResult]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    if len(scores) < 3 or np.ptp(scores) <= 0:
        raise ValueError("at least three non-constant finite scores are required")

    direct = float(threshold_otsu(scores, nbins=256))
    grid = score_grid(scores)
    equal = otsu_from_density(grid, kde_density(scores, grid, None))
    gamma = select_gamma(scores, grid)
    weighted = otsu_from_density(grid, kde_density(scores, grid, gamma))
    oracle, oracle_f1 = oracle_threshold(scores, labels)

    return [
        ThresholdResult("direct_otsu", direct, binary_f1(labels, scores >= direct)),
        ThresholdResult("equal_kde_otsu", equal, binary_f1(labels, scores >= equal), 0.0),
        ThresholdResult(
            "weighted_kde_otsu",
            weighted,
            binary_f1(labels, scores >= weighted),
            gamma,
        ),
        ThresholdResult("oracle", oracle, oracle_f1),
    ]


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """NumPy 1.x/2.x compatible trapezoidal integration."""
    function = getattr(np, "trapezoid", None)
    if function is None:
        function = np.trapz
    return float(function(values, coordinates))
