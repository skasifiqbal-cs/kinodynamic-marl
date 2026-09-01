"""Prioritised multi-robot trajectory optimisation (K-ARC's first ladder rung).

Plans every agent with the CasADi minimum-time program in
:mod:`src.approach.planning.trajopt`, in priority order: agent *i* treats the
already-committed trajectories of agents ``0..i-1`` as moving obstacles. This is the
rung K-ARC tries first (arXiv:2501.01559, Alg. 2) before falling back to decoupled
and then composite kinodynamic RRT.

Execution detail: the trajectories are solved on the **env's** time grid
(``dt_fixed=env.dt``) so the resulting controls can be replayed step-for-step by
``act``. ``solve_trajectory`` also supports a free ``dt`` (true minimum-time, K-ARC's
actual objective) -- use that for planning-quality comparisons, not for execution.

Params (``cfg.approach.optimization``): ``horizon``, ``effort_weight``,
``goal_tol``, ``max_iters``.
"""
from __future__ import annotations

import numpy as np

from src.approach.planning.base import BasePlanner
from src.approach.planning.trajopt import solve_trajectory
from src.shaping.braking_potential import bangbang_time


class OptimizationPlanner(BasePlanner):
    method = "optimization"

    def reset(self, env) -> None:
        p = self.params
        horizon = p.get("horizon", None)
        if horizon is None:
            horizon = self._auto_horizon(env, float(p.get("slack", 1.5)))
        horizon = int(horizon)
        committed: list[np.ndarray] = []
        radii: list[float] = []
        self._controls, self._solved = {}, {}

        for i, agent in enumerate(env.possible_agents):
            X, U, _, ok = solve_trajectory(
                env.robots[i], env._states[i], env._goals[i], env._obstacles,
                env._world_size,
                horizon=horizon,
                effort_weight=float(p.get("effort_weight", 0.01)),
                dt_fixed=env.dt,
                avoid=tuple(committed), avoid_radii=tuple(radii),
                goal_tol=float(p.get("goal_tol", env.goal_radius)),
                clearance=float(p.get("clearance", 0.05)),
                max_iter=int(p.get("max_iters", 500)),
            )
            self._controls[agent] = list(U)
            self._solved[agent] = ok
            # Commit even an unsolved trajectory: later robots should still avoid where
            # this one intends to go. The flag is what reports the failure.
            committed.append(X)
            radii.append(env.robots[i].shape.bounding_radius)

        self._plan = self._controls

    @staticmethod
    def _auto_horizon(env, slack: float) -> int:
        """Smallest common horizon whose time budget covers every agent's traverse.

        ``horizon * env.dt`` is the trajectory duration the program is allowed, so a
        hand-picked horizon silently makes the problem infeasible whenever the robot
        or the scenario changes. Derive it instead from the straight-line bang-bang
        time (the same primitive the braking potential uses), inflated by ``slack``
        to leave room for detours around obstacles and other robots.
        """
        worst = 0.0
        for i in range(env._n):
            d = float(np.linalg.norm(env._goals[i][:2] - env._states[i][:2]))
            r = env.robots[i]
            worst = max(worst, bangbang_time(d, 0.0, r.v_max, r.a_max))
        return min(env.max_steps, max(10, int(np.ceil(slack * worst / env.dt))))

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            out[agent] = seq.pop(0) if seq else np.zeros(env.robots[i].action_dim)
        return out
