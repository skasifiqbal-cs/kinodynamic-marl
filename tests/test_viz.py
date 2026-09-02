"""Renderer checks: greyscale, and legible at any robot count."""
from __future__ import annotations

import numpy as np

from src.collision.shapes import CircleShape
from src.viz.renderer import _index_fontsize, render_frame_with_shapes


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


def test_index_fontsize_tracks_body_size_on_screen():
    """Radii differ by >2x across configs (0.13 circle vs 0.2795 dynobench box), so a
    fixed point size either overflows small bodies or vanishes inside large ones."""
    small = _index_fontsize(0.13, world_size=5.0, fig_px=480)
    large = _index_fontsize(0.2795, world_size=5.0, fig_px=480)
    assert large > small
    # A big robot in a big world is no larger on screen than a small one up close.
    assert _index_fontsize(1.0, 40.0, 480) == _index_fontsize(0.25, 10.0, 480)
    # The ceiling must not bind at radii the repo actually uses, or the scaling is
    # decorative and every robot gets the same size anyway.
    assert small < 40.0 and large < 40.0
    # Clamped both ways for degenerate inputs.
    assert _index_fontsize(0.001, 100.0, 480) >= 5.0
    assert _index_fontsize(50.0, 1.0, 480) <= 40.0
