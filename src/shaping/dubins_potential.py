from __future__ import annotations

import numpy as np

from .base import BasePotential


class DubinsPotential(BasePotential):
    """phi = -dubins_path_length(state, goal) / v_max  (time units)."""

    def __init__(self, min_turning_radius: float, v_max: float) -> None:
        import dubins  # lazy import — optional dep
        self._dubins = dubins
        self.min_turning_radius = min_turning_radius
        self.v_max = v_max

    def phi(self, state: np.ndarray, goal: np.ndarray) -> float:
        q0 = (float(state[0]), float(state[1]), float(state[2]))
        q1 = (float(goal[0]), float(goal[1]), float(goal[2]))
        path = self._dubins.shortest_path(q0, q1, self.min_turning_radius)
        return -path.path_length() / self.v_max
