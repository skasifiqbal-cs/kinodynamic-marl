"""Lidar observer: N range rays + own pose + goal."""
from __future__ import annotations

from typing import List

import numpy as np

from src.obs.base import BaseObsBuilder
from src.robot.base import BaseRobot
from src.collision.shapes import Obstacle, ray_distance, CircleShape


class LidarObsBuilder(BaseObsBuilder):
    """obs = [own(4), goal(4), lidar_ranges(num_rays)]

    Rays are cast at equally-spaced angles in the world frame (not robot frame),
    detecting both static obstacles and other agents.
    Range is normalised to [0, 1] by dividing by max_range.
    """

    def __init__(self, n_agents: int, n_obstacles: int,
                 num_rays: int, max_range: float) -> None:
        self._num_rays = num_rays
        self._max_range = max_range
        self._dim = 4 + 4 + num_rays
        self._angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

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

        origin = np.array(own_state[:2], dtype=np.float64)
        ranges = np.full(self._num_rays, self._max_range)

        for i, angle in enumerate(self._angles):
            direction = np.array([np.cos(angle), np.sin(angle)])

            # Static obstacles
            for obs in obstacles:
                d = ray_distance(origin, direction, obs.shape, obs.pose, self._max_range)
                if d < ranges[i]:
                    ranges[i] = d

            # Other agents (treated as circles using bounding radius)
            for s in other_states:
                agent_shape = CircleShape(radius=own_robot.shape.bounding_radius)
                agent_pose = (float(s[0]), float(s[1]), float(s[2]))
                d = ray_distance(origin, direction, agent_shape, agent_pose, self._max_range)
                if d < ranges[i]:
                    ranges[i] = d

        normalised = (ranges / self._max_range).astype(np.float32)
        return np.concatenate([own, goal_feat, normalised])
