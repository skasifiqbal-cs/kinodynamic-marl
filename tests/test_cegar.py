"""The two pure pieces the CEGAR loop's correctness rests on.

`_split` decides where a robot stops when it waits en route, and `_hits` is the refutation
oracle -- every clause the solver learns comes from it, so a false "clear" is a collision
the solver will happily schedule, and a false "touch" forbids a pass that was fine.
"""
import numpy as np

from src.approach.planning.cegar import _hits, _split
from src.collision.shapes import BoxShape


class _Env:
    """Only what `_hits` reads."""

    def __init__(self, shape):
        self.robots = [_Rb(shape), _Rb(shape)]


class _Rb:
    def __init__(self, shape):
        self.shape = shape


def _track(x, y, th):
    x = np.asarray(x, float)
    return np.column_stack([x, np.full(len(x), y), np.full(len(x), th),
                            np.zeros(len(x)), np.zeros(len(x))])


def test_split_halves_arclength():
    head, tail = _split(np.array([[0.0, 0.0], [4.0, 0.0]]), 0.5)
    assert np.allclose(head[-1], [2.0, 0.0])
    assert np.allclose(tail[0], [2.0, 0.0])
    assert np.allclose(head[0], [0.0, 0.0]) and np.allclose(tail[-1], [4.0, 0.0])


def test_hits_catches_a_head_on_pass_and_clears_a_parallel_one():
    env = _Env(BoxShape(width=0.5, length=0.25))
    x = np.linspace(1.0, 9.0, 60)
    east = _track(x, 5.0, 0.0)
    assert _hits(env, 0, 1, east, _track(x[::-1], 5.0, np.pi), 0.05) is not None
    # Same motion two metres to the side: nothing can bring these together.
    assert _hits(env, 0, 1, east, _track(x[::-1], 7.0, np.pi), 0.05) is None


def test_hits_holds_the_shorter_candidate_at_its_last_state():
    """A robot that arrives early sits on its goal, and sitting there can still block."""
    env = _Env(BoxShape(width=0.5, length=0.25))
    parked = _track([5.0], 5.0, 0.0)
    through = _track(np.linspace(1.0, 9.0, 60), 5.0, 0.0)
    assert _hits(env, 0, 1, through, parked, 0.05) is not None


def test_contact_step_is_when_first_contact_happens():
    """The trajopt repair keeps a candidate up to shortly before this step, so it must be the
    step of the SAME contact `first_contact` reports, not merely a close one."""
    from src.conflict.pairwise import contact_step, first_contact

    shape = BoxShape(width=0.5, length=0.25)
    x = np.linspace(1.0, 9.0, 81)
    east, west = _track(x, 5.0, 0.0), _track(x[::-1], 5.0, np.pi)
    k = contact_step(shape, shape, east, west, 0.05)
    assert k is not None and 0 < k < 40
    assert contact_step(shape, shape, east[:k], west[:k], 0.05) is None
    assert np.allclose(first_contact(shape, shape, east, west, 0.05),
                       0.5 * (east[k, :2] + west[k, :2]))


def test_trajopt_first_order_plan_is_what_the_robot_executes():
    from omegaconf import OmegaConf

    from src.approach.planning.trajopt import solve_trajectory
    from src.robot import build_robot

    robot = build_robot(OmegaConf.load("conf/robot/unicycle1_db.yaml"))
    start, goal = np.array([1.0, 1.0, 0.0]), np.array([4.0, 2.0, 0.0])
    xs, us, _, ok = solve_trajectory(robot, start, goal, [], 17.0, horizon=90, dt_fixed=0.1,
                                     goal_tol=0.15, body_discs=3)
    assert ok and xs.shape == (91, 3) and us.shape == (90, 2)
    st = start.copy()
    for u in np.clip(us, robot.action_low, robot.action_high):
        st = robot.step(st, u, 0.1)
    assert np.linalg.norm(st[:2] - goal[:2]) <= 0.16
    assert np.allclose(us[-1], 0.0, atol=1e-6)
