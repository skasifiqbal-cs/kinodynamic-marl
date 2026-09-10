"""The delay table the complete scheduler is built on.

`_forbidden` is the load-bearing piece: it decides, for one pair of traversals, exactly
which delay differences collide. If it under-reports, the solver hands back a plan that
`_verify` then rejects; if it over-reports, it can call an instance unschedulable that is
not. Both directions are pinned here on a case whose answer is known by hand.
"""
import numpy as np

from src.approach.planning.constructive import _forbidden, _runs
from src.collision.shapes import CircleShape


class _Env:
    """Just the two attributes `_forbidden` reads."""

    def __init__(self, r):
        self.robots = [_Rb(r), _Rb(r)]


class _Rb:
    def __init__(self, r):
        self.shape = CircleShape(r)


def _walk(x0, y0, n, dx=0.05):
    """n poses marching east at dx per step: (x, y, theta)."""
    return np.column_stack([x0 + dx * np.arange(n), np.full(n, y0), np.zeros(n)])


def test_two_robots_on_one_track_collide_only_at_small_delay_differences():
    """Same lane, same direction: they clash unless one is far enough behind the other.

    Both walk the identical 100-step track. At difference 0 they are superimposed the
    whole way, so it must be forbidden. Push one far enough back and the leader has left
    -- except it has NOT, because a robot holds its goal after arriving, which is the
    clamping `_forbidden` models. So the far tail must stay forbidden too, and the answer
    is one contiguous band rather than a small window near zero.
    """
    env, track = _Env(0.25), _walk(0.0, 0.0, 100)
    bad = _forbidden(env, 0, 1, track, track.copy(), step=5, kmax=40, clearance=0.05)
    assert 0 in bad
    assert _runs(bad) == [(min(bad), max(bad))], "expected one contiguous band"


def test_tracks_a_body_apart_never_collide_at_any_delay():
    """Parallel lanes wider than the two bodies plus the clearance: no delay matters."""
    env = _Env(0.25)
    a, b = _walk(0.0, 0.0, 100), _walk(0.0, 0.9, 100)
    assert _forbidden(env, 0, 1, a, b, step=5, kmax=40, clearance=0.05) == []


def test_a_crossing_pair_is_free_once_one_is_let_through():
    """Perpendicular tracks meeting at the origin: forbidden near zero, free far out.

    Neither robot ends up standing where the other must pass, so unlike the shared-lane
    case the clamped tail IS clear -- a large enough difference lets one cross and go.
    """
    env = _Env(0.2)
    a = _walk(-2.5, 0.0, 100)
    b = np.column_stack([np.zeros(100), -2.5 + 0.05 * np.arange(100),
                         np.full(100, np.pi / 2)])
    bad = _forbidden(env, 0, 1, a, b, step=5, kmax=60, clearance=0.05)
    assert bad, "a crossing pair must forbid something"
    assert 60 not in bad and -60 not in bad


def test_runs_compresses_a_gap_into_two_intervals():
    assert _runs([-3, -2, -1, 4, 5]) == [(-3, -1), (4, 5)]
