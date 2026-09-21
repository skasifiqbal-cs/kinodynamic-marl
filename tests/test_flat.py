"""The smooth (flatness-based) trajectory construction.

Two properties are worth pinning here because both were bugs. Controls must be able to act
TOGETHER -- that is the whole reason this module exists -- and a lane displacement must be
back on the centre line by the time the route ends, or the smooth curve has to reach the
goal by a hairpin and the traverse blows past the horizon.
"""
import numpy as np

from src.core.collision.shapes import CircleShape
from src.core.robot.unicycle import Unicycle2Model
from src.planning import constructive, flat

DT = 0.1


def _robot():
    """The benchmark unicycle2_v0 bounds; the shape is irrelevant to `flat`."""
    return Unicycle2Model(v_max=0.5, v_min=0.0, omega_max=0.5, omega_min=-0.5,
                          a_max=0.25, alpha_max=0.25, shape=CircleShape(0.2795))


def _line(a, b, n=128):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def test_a_straight_drive_costs_what_the_analytic_trapezoid_costs():
    """15 m rest-to-rest is 320 steps: 20 up, 280 cruising at the cap, 20 down."""
    rb = _robot()
    out = flat.trajectory(rb, np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                          _line([0, 0], [15, 0]), DT)
    assert out is not None
    xs, us = out
    assert abs(len(xs) - 320) <= 5
    assert np.linalg.norm(xs[-1][:2] - np.array([15.0, 0.0])) < 0.05
    assert abs(float(xs[-1][3])) < 1e-2                      # ends at rest
    assert np.abs(xs[:, 3]).max() <= rb.v_max + 1e-6


def test_both_controls_fire_together_on_a_curved_route():
    """The point of the module: turning while accelerating, which stop-and-go never does.

    The legs construction produced 0 such steps out of ~25k across every scenario, because
    it drives each leg with one control at a time by design.
    """
    rb = _robot()
    route = np.vstack([_line([0, 0], [8, 0], 64), _line([8, 0], [8, 8], 64)])
    out = flat.trajectory(rb, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), route, DT)
    assert out is not None
    us = out[1]
    both = np.sum((np.abs(us[:, 0]) > 1e-6) & (np.abs(us[:, 1]) > 1e-6))
    assert both > 0


def test_no_bound_is_exceeded_on_a_curved_route():
    rb = _robot()
    route = np.vstack([_line([0, 0], [8, 0], 64), _line([8, 0], [8, 8], 64)])
    xs, us = flat.trajectory(rb, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), route, DT)
    assert np.abs(xs[:, 3]).max() <= rb.v_max + 1e-6
    assert np.abs(xs[:, 4]).max() <= rb.omega_max + 1e-6
    assert np.abs(us[:, 0]).max() <= rb.a_max + 1e-9
    assert np.abs(us[:, 1]).max() <= rb.alpha_max + 1e-9


def test_blur_leaves_the_endpoints_where_they_were():
    """Padding is linear extrapolation, not edge clamping.

    Clamping drags the first and last samples inward, and those two samples are the
    robot's start pose and its goal -- smoothing must not quietly move the goal.
    """
    v = np.linspace(0.0, 10.0, 200)
    out = flat._blur(v, sigma=6.0)
    assert abs(out[0] - v[0]) < 1e-9
    assert abs(out[-1] - v[-1]) < 1e-9


def test_a_lane_displacement_is_back_on_the_centre_line_by_the_goal():
    """Regression: a window padded to [0, 1] used to leave the route half a lane off.

    Stop-and-go hid it -- the leftover jog cost one cheap extra leg -- but a smooth curve
    has to take it as a hairpin, and on cluttered_cross_16 that capped one robot at
    0.03 m/s and stretched its traverse to 1483 steps, past the 1300-step horizon.
    """
    pts = _line([0, 0], [10, 0], 128)
    moved = constructive._offset_path(pts, dy=0.25, lo=0.0, hi=1.0, taper=0.12,
                                      axis=(0.0, 1.0))
    assert abs(float(moved[0][1])) < 1e-9
    assert abs(float(moved[-1][1])) < 1e-9
    assert np.abs(moved[:, 1]).max() > 0.2      # the lane is still actually applied


def test_legs_lands_on_every_waypoint_and_ends_at_rest():
    """The whole point of a rest-to-rest primitive: it arrives, and it stops.

    `profile` is solved on the same semi-implicit Euler the env integrates with, so a leg
    should land ON its target rather than near it -- if that drifts, every candidate the
    `drive=legs` ablation produces misses its goal and the ablation silently measures
    nothing.
    """
    route = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0]])
    got = flat.legs(_robot(), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), route, DT)
    assert got is not None
    states, controls = got
    assert np.linalg.norm(states[-1][:2] - route[-1]) < 0.05
    assert abs(float(states[-1][3])) < 1e-2 and abs(float(states[-1][4])) < 1e-2
    assert len(states) == len(controls)
