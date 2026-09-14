from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphSettings:
    """Parameters shared by all graph-propagation experiments."""

    neighbors: int = 10
    sigma_neighbors: int = 7
    sigma_method: str = "log_mean"
    alpha: float = 0.99
    max_iterations: int = 200
    tolerance: float = 1e-7
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if self.sigma_neighbors < 1:
            raise ValueError("sigma_neighbors must be positive")
        if self.sigma_method not in {
            "log_mean",
            "k7",
            "local_median",
            "global_median",
        }:
            raise ValueError(f"unsupported sigma_method: {self.sigma_method}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")


DEFAULT_GRAPH_SETTINGS = GraphSettings()
