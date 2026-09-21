"""The two spline properties the repair operator rests on.

If `nudge` were not local, repairing one robot's conflict would silently move its curve
somewhere else -- possibly into an obstacle that was checked before the edit. And if the
fit did not follow the sampled path, the RRT's choice of corridor would mean nothing.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from src.planning.splinecegar import curve, fit, nudge  # noqa: E402


def _s_path(n=40):
    """An S-bend, so the curve has real curvature to get wrong."""
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return np.column_stack([np.linspace(0.0, 12.0, n), 2.0 * np.sin(t / 4.0)])


def _dense(path, n=4000):
    """Deviation must be measured against a DENSE target.

    Nearest-vertex distance to a 40-point polyline reports up to half its vertex spacing
    as "deviation" for a curve running exactly down its middle -- 15 cm here, purely a
    sampling artefact. The same artefact once cost 15 of 16 robots their smooth driving
    in `constructive`, so it is pinned here rather than rediscovered.
    """
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    w = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(w, cum, path[:, 0]), np.interp(w, cum, path[:, 1])])


def test_fit_follows_the_sampled_path_and_pins_its_ends():
    path = _s_path()
    knots, ctrl = fit(path, 1.0)          # one control point per metre
    pts = curve(knots, ctrl)[0]
    target = _dense(path)
    dev = np.linalg.norm(pts[:, None, :] - target[None, :, :], axis=2).min(axis=1).max()
    assert dev < 0.02, f"fit strays {dev:.3f} m from the path it was given"
    assert np.allclose(pts[0], path[0], atol=1e-6)    # clamped: start pose is exact
    assert np.allclose(pts[-1], path[-1], atol=1e-6)  # and so is the goal


def test_nudge_is_local_and_actually_displaces():
    """Local support: the edit moves its own stretch and leaves the rest alone."""
    path = _s_path()
    knots, ctrl = fit(path, 1.0)
    before = curve(knots, ctrl)[0]
    at = before[len(before) // 2]

    moved = nudge(knots, ctrl, at, push=0.6, reach=2.0)
    assert moved is not None
    after = curve(knots, moved)[0]

    shift = np.linalg.norm(after - before, axis=1)
    assert shift.max() > 0.1, "repair did not move the curve at all"

    # Locality is in the PARAMETER, not in metres: each displaced control point perturbs
    # its own k+1 knot spans. So the moved samples must form one contiguous window well
    # short of the whole curve, and the curve must be untouched at both ends -- which is
    # what lets an obstacle check of the repaired stretch stand for the whole curve.
    moved = np.flatnonzero(shift > 1e-9)
    assert moved.max() - moved.min() + 1 == len(moved), "edit is not contiguous"
    assert len(moved) < 0.6 * len(shift), "edit touched most of the curve"
    assert shift[0] < 1e-12 and shift[-1] < 1e-12
    assert np.allclose(after[0], before[0]) and np.allclose(after[-1], before[-1])


def test_curvature_is_finite_and_matches_a_known_circle():
    """A circle of radius R has curvature 1/R everywhere -- the one case with an answer."""
    a = np.linspace(0.0, 2.0 * np.pi * 0.9, 60)
    R = 3.0
    knots, ctrl = fit(np.column_stack([R * np.cos(a), R * np.sin(a)]), 0.5)
    kappa = curve(knots, ctrl)[2]
    mid = kappa[len(kappa) // 4: 3 * len(kappa) // 4]
    assert np.all(np.isfinite(kappa))
    assert abs(abs(np.median(mid)) - 1.0 / R) < 0.02
