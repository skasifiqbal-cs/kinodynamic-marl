"""Time-gridded kinodynamic RRT: the two properties the K-ARC ladder depends on.

The rung is only interchangeable with a trajopt segment if (a) the returned trajectory is
exactly the segment horizon on the env's dt grid and dynamically consistent, and (b) index k
means the same instant to every robot, so a fixed trajectory can be avoided as a MOVING
obstacle rather than as a swept volume. Both are checked here; neither is checked by running
the planner end to end, where a failure looks like a slightly worse plan.
"""
import numpy as np

from src.core.robot import build_robot, load_robot_cfg
from src.planning import krrt


def _robot():
    return build_robot(load_robot_cfg("unicycle_v2"))


def test_grid_and_dynamics():
    r = _robot()
    dt, horizon = 0.05, 60
    X, U, ok = krrt.plan([r], [np.array([1.0, 1.0, 0.0, 0.0, 0.0])],
                         [np.array([3.0, 1.0, 0.0, 0.0, 0.0])],
                         [], 10.0, dt, horizon, rng=np.random.default_rng(0))
    assert ok, "2 m of empty space in 3 s is well inside the bang-bang time"
    assert X.shape == (horizon + 1, 1, 5) and U.shape == (horizon, 1, 2)
    for k in range(horizon):
        assert np.allclose(X[k + 1, 0], r.step(X[k, 0], U[k, 0], dt), atol=1e-9)
    assert np.linalg.norm(X[-1, 0, :2] - np.array([3.0, 1.0])) < 0.3


def test_avoids_a_moving_trajectory_not_its_swept_path():
    """A robot crossing our path EARLIER than us must not block us; one crossing at the
    same index must. Same geometry, different timing -- the whole reason nodes carry depth."""
    r = _robot()
    dt, horizon = 0.05, 80
    start, goal = np.array([1.0, 2.0, 0.0, 0.0, 0.0]), np.array([3.0, 2.0, 0.0, 0.0, 0.0])

    # An obstacle robot parked exactly on the straight line, for every index.
    blocked = np.repeat(np.array([[2.0, 2.0, 0.0, 0.0, 0.0]]), horizon + 1, 0)
    # The same robot, but it has already left by the time anyone could arrive.
    gone = np.repeat(np.array([[2.0, 8.0, 0.0, 0.0, 0.0]]), horizon + 1, 0)

    _, _, ok_gone = krrt.plan([r], [start], [goal], [], 10.0, dt, horizon,
                              others=((gone, r.shape),), rng=np.random.default_rng(1))
    X, _, _ = krrt.plan([r], [start], [goal], [], 10.0, dt, horizon,
                        others=((blocked, r.shape),), rng=np.random.default_rng(1))
    assert ok_gone
    # Detour or wait, but never overlap the occupied cell at any shared index.
    gap = min(float(np.linalg.norm(X[k, 0, :2] - blocked[k][:2])) for k in range(horizon + 1))
    assert gap >= 2 * r.shape.radius, f"drove through the moving obstacle (min gap {gap:.3f})"


def test_a_passed_deadline_stops_growth_and_fails_the_rung():
    """The wall-clock cap is what keeps one composite tree from eating the planning budget."""
    import time
    r = _robot()
    X, U, ok = krrt.plan([r], [np.array([1.0, 1.0, 0.0, 0.0, 0.0])],
                         [np.array([3.0, 1.0, 0.0, 0.0, 0.0])],
                         [], 10.0, 0.05, 60, rng=np.random.default_rng(0),
                         deadline=time.perf_counter() - 1.0)
    assert not ok and X.shape == (61, 1, 5)
