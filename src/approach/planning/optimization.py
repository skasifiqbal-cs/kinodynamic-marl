"""Trajectory-optimization planner (nonlinear solver) — STUB for the intern.

Formulate navigation as a nonlinear optimal-control problem and solve it with a
nonlinear solver (e.g. CasADi + IPOPT, or scipy.optimize.minimize with SLSQP):

  minimize   sum_t cost(state_t, u_t)              # e.g. control effort + time
  subject to state_{t+1} = f(state_t, u_t)         # robot dynamics (env.robots[i])
             u_t in [action_low, action_high]      # control bounds
             collision-free(state_t)               # obstacle/wall clearance
             state_0 = start, state_H ≈ goal

TODO (see BasePlanner docstring + docs/INTERN.md):
  reset(env):
    - build the NLP over a horizon H (params['horizon']); dynamics from
      robot.step or a symbolic model; solve; store the optimal control sequence
  act(obs_dict, env):
    - pop the next optimal control -> {agent: control} (clip to bounds)
Params: solver (e.g. 'ipopt'), horizon, max_iters. Add the solver dependency to
the environment before implementing.
"""
from __future__ import annotations

from src.approach.planning.base import BasePlanner


class OptimizationPlanner(BasePlanner):
    method = "optimization"

    def reset(self, env) -> None:
        raise self._todo("trajectory-optimization solve")

    def act(self, obs_dict: dict, env) -> dict:
        raise self._todo("optimization control playback")
