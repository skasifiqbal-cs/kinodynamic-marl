"""Renderer checks: greyscale, and legible at any robot count."""
from __future__ import annotations

import numpy as np

from src.collision.shapes import CircleShape
from src.viz.renderer import _forward_reach, _label_fontsize, render_frame_with_shapes


def _frame(n: int, world: float = 6.0, radius: float = 0.2795, fig_px: int = 480):
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    states = [np.array([world / 2 + 2.0 * np.cos(a), world / 2 + 2.0 * np.sin(a),
                        a + np.pi, 0.3, 0.0]) for a in ang]
    goals = [np.array([world / 2 - 2.0 * np.cos(a), world / 2 - 2.0 * np.sin(a), 0.0])
             for a in ang]
    return render_frame_with_shapes(
        states=states, robot_shapes=[CircleShape(radius)] * n, goals=goals,
        obstacles=[], trails=[[s[:2]] for s in states], world_size=world,
        reached=[False] * n, step=0, fig_px=fig_px,
    )


def test_frame_is_greyscale():
    """Robots are told apart by an index, not by hue, so the figure must survive a
    black-and-white print. Greyscale means R == G == B at every pixel."""
    f = _frame(8).astype(int)
    assert f.shape[2] == 3
    assert np.array_equal(f[:, :, 0], f[:, :, 1])
    assert np.array_equal(f[:, :, 1], f[:, :, 2])


def test_robot_count_is_not_capped_by_a_palette():
    """The old renderer cycled a 4-colour palette, so robots 0 and 4 drew identically.
    Indices have no such limit: 12 robots must render without raising."""
    assert _frame(12).shape[2] == 3


def test_label_fontsize_scales_with_the_figure():
    """Labels sit outside the bodies, so they track figure size, not robot radius."""
    assert _label_fontsize(960) > _label_fontsize(480)
    assert 6.0 <= _label_fontsize(64) and _label_fontsize(4000) <= 14.0


def test_forward_reach_uses_the_box_axis_the_robot_actually_points_along():
    """conf/robot/unicycle_db.yaml sets width as the extent along local x (forward), to
    match dynobench's size[0], and _obb_corners lays the box out on that axis. Taking
    `length` instead draws the heading line out through the robot's side."""
    from src.collision.shapes import BoxShape

    assert _forward_reach(BoxShape(width=0.5, length=0.25)) == 0.25   # width/2
    assert _forward_reach(CircleShape(0.13)) == 0.13


def test_robot_and_goal_labels_go_to_opposite_corners():
    """A robot parked on its own goal is the normal end state. Same-side labels would
    put a0 on top of g0 exactly then."""
    from src.viz.renderer import _label_pos

    ax, ay = _label_pos(2.5, 2.5, 0.13, 5.0, side=1)     # robot
    gx, gy = _label_pos(2.5, 2.5, 0.20, 5.0, side=-1)    # its goal, same spot
    assert ax > 2.5 and ay > 2.5
    assert gx < 2.5 and gy < 2.5
    assert np.hypot(ax - gx, ay - gy) > 0.3
