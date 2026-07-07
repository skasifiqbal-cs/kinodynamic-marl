"""Kinodynamic RRT planner — STUB for the intern to implement.

Unlike geometric RRT, plan directly in the robot's state space by sampling
*controls* and propagating the true dynamics, so the resulting trajectory is
dynamically feasible (respects v/ω/accel limits and non-holonomy).

TODO (see BasePlanner docstring + docs/INTERN.md):
  reset(env):
    - tree nodes are full states env._states[i]-like; root = start state
    - to extend: pick a target, find nearest node, sample n_control_samples
      controls u ~ Uniform(robot.action_low, robot.action_high), propagate each
      for propagation_steps via robot.step(state, u, propagation_dt), keep the
      endpoint closest to target that stays collision-free
    - stop when a node is within goal_radius of env._goals[i][:2]
    - store the winning control sequence per agent on self
  act(obs_dict, env):
    - pop the next control from the stored sequence -> {agent: control}
      (hold last / zero control once the sequence is exhausted; clip to bounds)
Params: n_control_samples, propagation_dt, propagation_steps, max_iters.
"""
from __future__ import annotations

from src.approach.planning.base import BasePlanner


class KinodynamicRRTPlanner(BasePlanner):
    method = "kinodynamic_rrt"

    def reset(self, env) -> None:
        raise self._todo("kinodynamic RRT control-space planning")

    def act(self, obs_dict: dict, env) -> dict:
        raise self._todo("kinodynamic RRT control playback")
