"""Full-state observer: own pose + goal + relative other agents + obstacle repr."""
from __future__ import annotations

from typing import List

import numpy as np

from src.obs.base import BaseObsBuilder
from src.robot.base import BaseRobot
from src.collision.shapes import Obstacle

# Per-obstacle encoding: [rel_x, rel_y, hw, hl, sin(a), cos(a)] — 6 values.
# Circles map to [rel_x, rel_y, r, r, 0, 1]; keeps obs_dim uniform across shape types.
OBS_PER_OBSTACLE = 6


class FullStateObsBuilder(BaseObsBuilder):
    """obs = [own(4), goal(4), others(2×n_other), obstacles(6×n_obs)]"""

    def __init__(self, n_agents: int, n_obstacles: int) -> None:
        self._n_other = n_agents - 1
        self._n_obs = n_obstacles
        self._dim = 4 + 4 + 2 * self._n_other + OBS_PER_OBSTACLE * self._n_obs

    def obs_dim(self) -> int:
        return self._dim

    def build(
        self,
        own_state: np.ndarray,
        own_robot: BaseRobot,
        goal: np.ndarray,
        other_states: List[np.ndarray],
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        own = own_robot.obs_features(own_state).astype(np.float32)
        goal_feat = np.array(
            [goal[0], goal[1], np.sin(goal[2]), np.cos(goal[2])], dtype=np.float32
        )
        others = np.concatenate(
            [np.array(s[:2] - own_state[:2], dtype=np.float32) for s in other_states]
        ) if other_states else np.array([], dtype=np.float32)

        obs_feats = []
        for obs in obstacles:
            feat = obs.obs_repr.copy()         # 6-D template
            feat[0] = obs.x - own_state[0]    # fill relative position
            feat[1] = obs.y - own_state[1]
            obs_feats.append(feat)

        parts = [own, goal_feat, others] + obs_feats
        return np.concatenate(parts).astype(np.float32)
