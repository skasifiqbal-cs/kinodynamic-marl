"""Unicycle kinematic model with RK4 integration."""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from src.robot.base import BaseRobot
from src.collision.shapes import Shape


class UnicycleModel(BaseRobot):
    """State: [x, y, θ].  Action: [v, ω]."""

    def __init__(self, v_max: float, v_min: float, omega_max: float,
                 omega_min: float, shape: Shape) -> None:
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.omega_min = omega_min
        self.shape = shape

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_low(self) -> np.ndarray:
        return np.array([self.v_min, self.omega_min], dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([self.v_max, self.omega_max], dtype=np.float32)

    def _f(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        _, _, theta = state
        v, omega = action
        return np.array([v * np.cos(theta), v * np.sin(theta), omega], dtype=np.float64)

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        v = np.clip(action[0], self.v_min, self.v_max)
        omega = np.clip(action[1], self.omega_min, self.omega_max)
        a = np.array([v, omega], dtype=np.float64)

        k1 = self._f(state, a)
        k2 = self._f(state + 0.5 * dt * k1, a)
        k3 = self._f(state + 0.5 * dt * k2, a)
        k4 = self._f(state + dt * k3, a)

        next_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        return next_state
