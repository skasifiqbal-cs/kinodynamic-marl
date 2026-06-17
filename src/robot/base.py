"""BaseRobot: common interface for all kinodynamic models."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from src.collision.shapes import Shape


class BaseRobot(ABC):
    """All robots: state=[x, y, θ], obs=[x, y, sin θ, cos θ].
    Subclasses set self.shape (CircleShape or BoxShape) in __init__."""

    shape: Shape  # set by concrete subclass

    # ── state / action metadata ───────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return 3  # [x, y, theta]

    @property
    def obs_feature_dim(self) -> int:
        return 4  # [x, y, sin θ, cos θ]

    def obs_features(self, state: np.ndarray) -> np.ndarray:
        x, y, theta = state
        return np.array([x, y, np.sin(theta), np.cos(theta)], dtype=np.float32)

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_low(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def action_high(self) -> np.ndarray: ...

    # ── dynamics ──────────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """RK4-integrate state one step. Must normalise theta to [-π, π]."""
        ...
