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
