"""N-agent kinodynamic navigation — PettingZoo Parallel API.

Fully modular: robot type, obs builder, initializer, and shaping potential
are all injected, not hardcoded.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from omegaconf import DictConfig

from src.robot import BaseRobot, build_robot, load_robot_cfg
from src.collision.shapes import Obstacle, build_obstacle, collides
from src.obs.base import BaseObsBuilder
from src.init.initializer import BaseInitializer
from src.shaping.base import BasePotential


class MultiAgentNav(ParallelEnv):
    """Kinodynamic N-agent navigation.

    Each agent has its own robot (dynamics + shape), goal, and shaping potential.
    Collision detection is shape-aware (circle or OBB).
    """

    metadata = {"render_modes": [], "name": "multiagent_nav_v0"}

    def __init__(
        self,
        cfg: DictConfig,
        robots: List[BaseRobot],
        potentials: List[BasePotential],
        obs_builder: BaseObsBuilder,
        initializer: BaseInitializer,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.robots = robots
        self.potentials = potentials
        self.obs_builder = obs_builder
        self.initializer = initializer
        self.gamma = cfg.shaping.gamma

        # Scenario params
        self.dt = cfg.env.dt
        self.max_steps = cfg.env.max_steps
        self.goal_radius = cfg.env.goal_radius
        self.collision_penalty = cfg.env.reward.collision
        self.reach_reward = cfg.env.reward.reach
        self.step_penalty = cfg.env.reward.step_penalty

        # Obstacles parsed from config
        self._obstacles: List[Obstacle] = [
            build_obstacle(o) for o in cfg.env.obstacles
        ]

        # Agent identifiers and static config
        self._agent_cfgs = list(cfg.env.agents)
        self.possible_agents = [a.id for a in self._agent_cfgs]
        self._n = len(self.possible_agents)

        # Goals are set by initializer on reset; pre-allocate
        self._goals: List[np.ndarray] = [np.zeros(3) for _ in range(self._n)]

        # Spaces (per-agent — may differ for heterogeneous robots)
        obs_d = obs_builder.obs_dim()
        self._obs_spaces = {
            a: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_d,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: gym.spaces.Box(
                low=robots[i].action_low,
                high=robots[i].action_high,
                dtype=np.float32,
            )
            for i, a in enumerate(self.possible_agents)
        }

        # Runtime state (initialised on reset)
        self.agents: List[str] = []
        self._states: List[np.ndarray] = [np.zeros(3) for _ in range(self._n)]
        self._step_count: int = 0
        self._reached: List[bool] = [False] * self._n
        self._rng = np.random.default_rng()

    # ── PettingZoo API ────────────────────────────────────────────────────────

    def observation_space(self, agent: str) -> gym.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self._act_spaces[agent]

    def reset(self, seed: int | None = None, options: Any = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._reached = [False] * self._n

        self._states, self._goals = self.initializer.reset(
            self._agent_cfgs, self.cfg.env, self._obstacles, self._rng
        )
        return self._build_obs(), {a: {} for a in self.agents}

    def step(self, actions: dict):
        prev_states = [s.copy() for s in self._states]
        rewards = {a: 0.0 for a in self.possible_agents}

        # Dynamics
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue
            act = np.asarray(actions[agent], dtype=np.float64)
            self._states[i] = self.robots[i].step(self._states[i], act, self.dt)

        self._step_count += 1

        # Rewards
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.agents:
                continue

            # Step penalty — encourages speed
            rewards[agent] += self.step_penalty

            # Potential-based shaping
            rewards[agent] += self.potentials[i].shaping(
                prev_states[i], self._states[i], self._goals[i], self.gamma
            )

            # Goal reach (one-shot)
            if not self._reached[i]:
                if np.linalg.norm(self._states[i][:2] - self._goals[i][:2]) < self.goal_radius:
                    rewards[agent] += self.reach_reward
                    self._reached[i] = True

            # Agent–agent collision
            for j in range(self._n):
                if j == i:
                    continue
                pose_i = (self._states[i][0], self._states[i][1], self._states[i][2])
                pose_j = (self._states[j][0], self._states[j][1], self._states[j][2])
                if collides(self.robots[i].shape, pose_i, self.robots[j].shape, pose_j):
                    rewards[agent] += self.collision_penalty

            # Agent–obstacle collision
            pose_i = (self._states[i][0], self._states[i][1], self._states[i][2])
            for obs in self._obstacles:
                if collides(self.robots[i].shape, pose_i, obs.shape, obs.pose):
                    rewards[agent] += self.collision_penalty

        # Termination
        all_reached = all(self._reached)
        timeout = self._step_count >= self.max_steps
        done = all_reached or timeout

        terminations = {a: all_reached for a in self.possible_agents}
        truncations  = {a: timeout and not all_reached for a in self.possible_agents}

        if done:
            self.agents = []

        infos = {
            a: {"reached": self._reached[i], "step": self._step_count}
            for i, a in enumerate(self.possible_agents)
        }
        return self._build_obs(), rewards, terminations, truncations, infos

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_obs(self) -> dict:
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            others = [self._states[j] for j in range(self._n) if j != i]
            obs[agent] = self.obs_builder.build(
                own_state=self._states[i],
                own_robot=self.robots[i],
                goal=self._goals[i],
                other_states=others,
                obstacles=self._obstacles,
            )
        return obs
