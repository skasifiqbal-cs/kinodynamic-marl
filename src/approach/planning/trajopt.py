"""Minimum-time kinodynamic trajectory optimisation (CasADi + IPOPT).

This is the low-level primitive K-ARC calls (its ``SolveMRKP``): a nonlinear program
over the second-order unicycle, solved to a dynamically feasible minimum-time
trajectory that clears the static obstacles and, optionally, a set of already-committed
trajectories belonging to higher-priority robots.

Formulation (K-ARC Eq. 4-6, arXiv:2501.01559)::

    minimise    beta1 * sum_k ||u_k||^2  +  T          with T = horizon * dt
    subject to  x_0     = start
                x_{k+1} = RK4(x_k, u_k, dt)            second-order unicycle
                u_k     in [action_low, action_high]
                v, omega within the robot's state box
                ||p_k - c||     >= r_obs + r_robot     static obstacles
                ||p_k - q_k^j|| >= r_robot + r_j       higher-priority robots
                p_N within goal_tol of the goal, v_N = omega_N = 0

``solve_group`` puts several robots in ONE program and adds the pairwise separation
``||p_k^i - p_k^j|| >= r_i + r_j`` between them. That is K-ARC's ``AdaptSubProblem``:
when resolving a conflict by ordering fails, the conflicting robots are re-solved
together instead of one-after-another. It is strictly more capable than the prioritised
rung and strictly more expensive -- the program grows with the number of robots in the
group, which is why it is the last rung and not the first. Symmetric head-on swaps need
it: whichever robot goes second in a priority order has nowhere to yield to, so no
ordering of single-robot solves is feasible while a joint solve is.

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


def solve_group(
    robots: Sequence,
    starts: Sequence[np.ndarray],
    goals: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle],
    world_size: float,
    horizon: int = 40,
    effort_weight: float = 0.01,
    dt_bounds: tuple[float, float] = (0.02, 0.5),
    dt_fixed: float | None = None,
    avoid: Sequence[np.ndarray] = (),
    avoid_radii: Sequence[float] = (),
    goal_tol: float | Sequence[float] = 0.1,
    clearance: float = 0.05,
    terminal_stop: bool = True,
    max_iter: int = 500,
) -> tuple[list[np.ndarray], list[np.ndarray], float, bool]:
    """Solve one minimum-time program covering ``len(robots)`` robots jointly.

    All robots share the time grid, so index k is the same instant for each of them and
    the pairwise separation constraints between them are meaningful. ``avoid`` holds
    already-committed trajectories *outside* this group as ``(>=horizon+1, >=2)`` arrays
    on that same grid; ``avoid_radii`` their bounding radii.

    ``goal_tol`` may be a scalar or one tolerance per robot -- per-robot values are what
    let a caller relax only the lower-priority robots' terminal constraints.

    ``clearance`` inflates every separation constraint. It is not cosmetic: minimising
    time drives the solution hard onto the constraint boundary, and a trajectory that
    clears an obstacle by millimetres at the knots overlaps it once execution drifts
    off the planned states.

    Returns ``(states_per_robot, controls_per_robot, dt, solved)`` with states
    ``(H+1, 5)`` and controls ``(H, 2)``. On solver failure the last iterate is
    returned with ``solved=False`` -- it may violate the constraints, so callers must
    check the flag before trusting it.
    """
    import casadi as ca

    N = int(horizon)
    n = len(robots)
    radii = [float(r.shape.bounding_radius) for r in robots]
    tols = [float(goal_tol)] * n if np.isscalar(goal_tol) else [float(t) for t in goal_tol]

    opti = ca.Opti()
    X = [opti.variable(5, N + 1) for _ in range(n)]
    U = [opti.variable(2, N) for _ in range(n)]
    if dt_fixed is None:
        dt = opti.variable()
        opti.subject_to(opti.bounded(dt_bounds[0], dt, dt_bounds[1]))
        opti.set_initial(dt, 0.5 * sum(dt_bounds))
    else:
        dt = float(dt_fixed)

    def f(x, u):
        return ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), x[4], u[0], u[1])

    # Minimum time, regularised by control effort. N*dt is the trajectory duration --
    # shared by the whole group, so the group finishes when its slowest member does.
    obj = N * dt + effort_weight * sum(
        ca.sumsqr(U[i][:, k]) for i in range(n) for k in range(N)
    )
    opti.minimize(obj)

    for i, robot in enumerate(robots):
        lo = np.asarray(robot.action_low, float)
        hi = np.asarray(robot.action_high, float)
        r_self = radii[i]
        s = np.asarray(starts[i], float)
        g = np.asarray(goals[i], float)

        opti.subject_to(X[i][:, 0] == ca.DM(s.reshape(5)))

        for k in range(N):
            k1 = f(X[i][:, k], U[i][:, k])
            k2 = f(X[i][:, k] + 0.5 * dt * k1, U[i][:, k])
            k3 = f(X[i][:, k] + 0.5 * dt * k2, U[i][:, k])
            k4 = f(X[i][:, k] + dt * k3, U[i][:, k])
            opti.subject_to(
                X[i][:, k + 1] == X[i][:, k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            )
            opti.subject_to(opti.bounded(lo[0], U[i][0, k], hi[0]))
            opti.subject_to(opti.bounded(lo[1], U[i][1, k], hi[1]))

        for k in range(N + 1):
            opti.subject_to(opti.bounded(robot.v_min, X[i][3, k], robot.v_max))
            opti.subject_to(opti.bounded(robot.omega_min, X[i][4, k], robot.omega_max))
            opti.subject_to(opti.bounded(r_self, X[i][0, k], world_size - r_self))
            opti.subject_to(opti.bounded(r_self, X[i][1, k], world_size - r_self))

            # ponytail: obstacles inflated to their bounding circle -- sound
            # (conservative) but loose for long boxes. Swap in a superquadric if
            # clutter gets tight.
            for obs in obstacles:
                clear = float(obs.shape.bounding_radius) + r_self + clearance
                opti.subject_to(
                    (X[i][0, k] - obs.x) ** 2 + (X[i][1, k] - obs.y) ** 2 >= clear**2
                )

            for traj, r_other in zip(avoid, avoid_radii):
                kk = min(k, len(traj) - 1)
                clear = r_self + float(r_other) + clearance
                opti.subject_to(
                    (X[i][0, k] - float(traj[kk][0])) ** 2
                    + (X[i][1, k] - float(traj[kk][1])) ** 2
                    >= clear**2
                )

        opti.subject_to(ca.sumsqr(X[i][0:2, N] - ca.DM(g[:2].reshape(2))) <= tols[i] ** 2)
        if terminal_stop:
            # Arrive at rest -- the env's stop-at-goal gate. MUST be False for the
            # intermediate milestones of a segmented plan: forcing a full stop at every
            # waypoint turns one trajectory into m stop-start hops.
            opti.subject_to(X[i][3, N] == 0.0)
            opti.subject_to(X[i][4, N] == 0.0)

        # Warm start: straight line from start to goal, at rest.
        for k in range(N + 1):
            t = k / N
            opti.set_initial(X[i][0, k], (1 - t) * s[0] + t * g[0])
            opti.set_initial(X[i][1, k], (1 - t) * s[1] + t * g[1])
            opti.set_initial(X[i][2, k], np.arctan2(g[1] - s[1], g[0] - s[0]))

    # Pairwise separation WITHIN the group. This is the whole point of a joint solve:
    # both robots' trajectories are free, so the solver can make them yield to each
    # other symmetrically instead of one deferring to an already-fixed path.
    for i in range(n):
        for j in range(i + 1, n):
            clear = radii[i] + radii[j] + clearance
            for k in range(N + 1):
                opti.subject_to(
                    (X[i][0, k] - X[j][0, k]) ** 2 + (X[i][1, k] - X[j][1, k]) ** 2
                    >= clear**2
                )

    opti.solver("ipopt", {"print_time": 0}, {"print_level": 0, "max_iter": max_iter})
    try:
        sol = opti.solve()
        dt_val = float(sol.value(dt)) if dt_fixed is None else float(dt_fixed)
        ok = True
    except RuntimeError:
        sol = opti.debug
        dt_val = float(sol.value(dt)) if dt_fixed is None else float(dt_fixed)
        ok = False
    return [sol.value(x).T for x in X], [np.atleast_2d(sol.value(u).T) for u in U], dt_val, ok


def solve_trajectory(
    robot,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle],
    world_size: float,
    **kw,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Single-robot case of :func:`solve_group`, with unwrapped return values."""
    xs, us, dt, ok = solve_group(
        [robot], [start], [goal], obstacles, world_size, **kw
    )
    return xs[0], us[0], dt, ok
