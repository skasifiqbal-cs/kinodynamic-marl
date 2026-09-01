"""Minimum-time kinodynamic trajectory optimisation for one robot (CasADi + IPOPT).

This is the low-level primitive K-ARC calls (its ``SolveMRKP``): a single-robot
nonlinear program over the second-order unicycle, solved to a dynamically feasible
minimum-time trajectory that clears the static obstacles and, optionally, a set of
already-committed trajectories belonging to higher-priority robots.

Formulation (K-ARC Eq. 4-6, arXiv:2501.01559)::

    minimise    beta1 * sum_k ||u_k||^2  +  T          with T = horizon * dt
    subject to  x_0     = start
                x_{k+1} = RK4(x_k, u_k, dt)            second-order unicycle
                u_k     in [action_low, action_high]
                v, omega within the robot's state box
                ||p_k - c||     >= r_obs + r_robot     static obstacles
                ||p_k - q_k^j|| >= r_robot + r_j       higher-priority robots
                p_N within goal_tol of the goal, v_N = omega_N = 0

**dt is a decision variable**, which is what makes this minimum-time rather than a
tracking MPC: the solver buys a shorter horizon by spending control effort, traded
off by ``effort_weight``. Fix it with ``dt_fixed`` when several robots must share a
common time grid -- the inter-robot constraints below are only meaningful when
index k denotes the same instant for every robot, which is exactly why K-ARC
segments its plans and synchronises milestones.

Note on the avoidance constraint: it is a pure geometric separation test on
positions, with no velocity term. That is deliberate -- it reproduces what K-ARC
(Eq. 6) and every planner in that family actually does, so this stays a faithful
baseline. See ``scripts/ics_diag.py`` for the braking-margin alternative.

casadi is an optional dependency (``pip install -e ".[planning]"``); it is imported
lazily so the rest of the package still imports without it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.collision.shapes import Obstacle


def solve_trajectory(
    robot,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle],
    world_size: float,
    horizon: int = 40,
    effort_weight: float = 0.01,
    dt_bounds: tuple[float, float] = (0.02, 0.5),
    dt_fixed: float | None = None,
    avoid: Sequence[np.ndarray] = (),
    avoid_radii: Sequence[float] = (),
    goal_tol: float = 0.1,
    clearance: float = 0.05,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Solve one robot's minimum-time trajectory.

    ``avoid`` holds already-committed trajectories as ``(>=horizon+1, >=2)`` arrays
    on the *same* time grid; ``avoid_radii`` their bounding radii.

    ``clearance`` inflates every separation constraint. It is not cosmetic: minimising
    time drives the solution hard onto the constraint boundary, and a trajectory that
    clears an obstacle by millimetres at the knots overlaps it once execution drifts
    off the planned states. Returns
    ``(states (H+1, 5), controls (H, 2), dt, solved)``. On solver failure the last
    iterate is returned with ``solved=False`` -- it may violate the constraints, so
    callers must check the flag before trusting it.
    """
    import casadi as ca

    N = int(horizon)
    r_self = float(robot.shape.bounding_radius)
    lo, hi = np.asarray(robot.action_low, float), np.asarray(robot.action_high, float)

    opti = ca.Opti()
    X = opti.variable(5, N + 1)
    U = opti.variable(2, N)
    if dt_fixed is None:
        dt = opti.variable()
        opti.subject_to(opti.bounded(dt_bounds[0], dt, dt_bounds[1]))
        opti.set_initial(dt, 0.5 * sum(dt_bounds))
    else:
        dt = float(dt_fixed)

    def f(x, u):
        return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), x[4], u[0], u[1])

    # Minimum time, regularised by control effort. N*dt is the trajectory duration.
    obj = N * dt + effort_weight * sum(ca.sumsqr(U[:, k]) for k in range(N))
    opti.minimize(obj)

    opti.subject_to(X[:, 0] == ca.DM(np.asarray(start, float).reshape(5)))

    for k in range(N):
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + 0.5 * dt * k1, U[:, k])
        k3 = f(X[:, k] + 0.5 * dt * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        opti.subject_to(X[:, k + 1] == X[:, k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        opti.subject_to(opti.bounded(lo[0], U[0, k], hi[0]))
        opti.subject_to(opti.bounded(lo[1], U[1, k], hi[1]))

    for k in range(N + 1):
        opti.subject_to(opti.bounded(robot.v_min, X[3, k], robot.v_max))
        opti.subject_to(opti.bounded(robot.omega_min, X[4, k], robot.omega_max))
        opti.subject_to(opti.bounded(r_self, X[0, k], world_size - r_self))
        opti.subject_to(opti.bounded(r_self, X[1, k], world_size - r_self))

        # ponytail: obstacles inflated to their bounding circle -- sound (conservative)
        # but loose for long boxes. Swap in a superquadric if clutter gets tight.
        for obs in obstacles:
            clear = float(obs.shape.bounding_radius) + r_self + clearance
            opti.subject_to((X[0, k] - obs.x) ** 2 + (X[1, k] - obs.y) ** 2 >= clear**2)

        for traj, r_other in zip(avoid, avoid_radii):
            kk = min(k, len(traj) - 1)
            clear = r_self + float(r_other) + clearance
            opti.subject_to(
                (X[0, k] - float(traj[kk][0])) ** 2 + (X[1, k] - float(traj[kk][1])) ** 2
                >= clear**2
            )

    g = np.asarray(goal, float)
    opti.subject_to(ca.sumsqr(X[0:2, N] - ca.DM(g[:2].reshape(2))) <= goal_tol**2)
    opti.subject_to(X[3, N] == 0.0)   # arrive at rest -- the env's stop-at-goal gate
    opti.subject_to(X[4, N] == 0.0)

    # Warm start: straight line from start to goal, at rest.
    s = np.asarray(start, float)
    for k in range(N + 1):
        t = k / N
        opti.set_initial(X[0, k], (1 - t) * s[0] + t * g[0])
        opti.set_initial(X[1, k], (1 - t) * s[1] + t * g[1])
        opti.set_initial(X[2, k], np.arctan2(g[1] - s[1], g[0] - s[0]))

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "max_iter": max_iter})
    try:
        sol = opti.solve()
        dt_val = float(sol.value(dt)) if dt_fixed is None else float(dt_fixed)
        return sol.value(X).T, sol.value(U).T, dt_val, True
    except RuntimeError:
        d = opti.debug
        dt_val = float(d.value(dt)) if dt_fixed is None else float(dt_fixed)
        return d.value(X).T, d.value(U).T, dt_val, False
