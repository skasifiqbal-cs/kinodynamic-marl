"""Stateless matplotlib renderer — supports circle and box shapes.

Greyscale on purpose. Robots and their goals are told apart by an index printed on
them, not by hue: figures survive a black-and-white print, and the count of robots is
no longer capped by the length of a colour palette.
"""
from __future__ import annotations

import io
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from src.collision.shapes import BoxShape, CircleShape, Obstacle

# Neutral greys only: R == G == B. The blue-tinted "greys" this replaced (#F1F3F4,
# #6C757D) survive a mono print but are not actually greyscale, and a figure that is
# 87% off-neutral pixels is a colour figure as far as a journal is concerned.
ROBOT_COLOR = "#2B2B2B"   # body: near-black, so a white index reads on top of it
GOAL_COLOR  = "#8A8A8A"   # goal ring and its index
TRAIL_COLOR = "#B0B0B0"
OBS_COLOR   = "#777777"
BG_COLOR    = "#F2F2F2"


def _obb_corners(x: float, y: float, theta: float, w: float, l: float) -> np.ndarray:
    hw, hl = w / 2, l / 2
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([x, y])


def _index_fontsize(radius: float, world_size: float, fig_px: int) -> float:
    """Size the index to the body drawn on screen, not to a fixed point size.

    Robot radii differ by more than 2x across configs (0.13 circle vs 0.2795 for the
    dynobench box), so one hard-coded size either overflows the small bodies or gets
    lost inside the large ones.
    """
    px = 2.0 * radius / max(world_size, 1e-9) * fig_px
    # 0.55 puts the digit's cap height at roughly half the body it sits in. The bounds
    # are there for degenerate cases only -- a ceiling low enough to bind at ordinary
    # radii silently turns this back into the fixed size it exists to avoid.
    return float(np.clip(0.55 * px, 5.0, 40.0))


def _draw_heading(ax, x, y, theta, reach, alpha):
    """Heading nub OUTSIDE the body, so it never collides with the index on top of it."""
    ax.plot(
        [x + reach * np.cos(theta), x + 1.55 * reach * np.cos(theta)],
        [y + reach * np.sin(theta), y + 1.55 * reach * np.sin(theta)],
        color=ROBOT_COLOR, lw=1.4, alpha=alpha, solid_capstyle="round", zorder=6,
    )


def _draw_shape(ax, x, y, theta, shape, color, alpha=1.0, zorder=2):
    if isinstance(shape, CircleShape):
        ax.add_patch(mpatches.Circle((x, y), shape.radius,
                                     color=color, alpha=alpha, zorder=zorder))
    elif isinstance(shape, BoxShape):
        corners = _obb_corners(x, y, theta, shape.width, shape.length)
        ax.add_patch(Polygon(corners, closed=True,
                             color=color, alpha=alpha, zorder=zorder))


def _add_reward_bar(fig, ax, step_rewards: Dict[str, float]) -> None:
    """Add per-agent step reward text below the axes."""
    parts = [f"{i}: {r:+.2f}" for i, (_a, r) in enumerate(sorted(step_rewards.items()))]
    ax.set_xlabel("    ".join(parts), fontsize=8, labelpad=4, family="monospace")


def _finish_frame(fig, ax, dpi: int) -> np.ndarray:
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))


def render_frame_with_shapes(
    states: List[np.ndarray],
    robot_shapes,
    goals: List[np.ndarray],
    obstacles: List[Obstacle],
    trails: List[List[np.ndarray]],
    world_size: float,
    reached: List[bool],
    step: int,
    fig_px: int = 480,
    step_rewards: Optional[Dict[str, float]] = None,
    goal_radius: float = 0.2,
) -> np.ndarray:
    """Render one frame with correct robot shapes (circle or OBB).

    Robot ``i`` is drawn with ``i`` on its body; its goal is the dashed ring labelled
    ``i``. Indices match the agent order in the env, so they line up with ``agent_0``,
    ``agent_1``, ... and with the reward bar underneath.
    """
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"step {step}", fontsize=9, pad=3)

    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape,
                    color=OBS_COLOR, alpha=0.75, zorder=2)

    for i, goal in enumerate(goals):
        ax.add_patch(mpatches.Circle(
            (goal[0], goal[1]), goal_radius, facecolor="none", edgecolor=GOAL_COLOR,
            linestyle="--", linewidth=1.2, zorder=3,
        ))
        ax.text(goal[0], goal[1], str(i), fontsize=_index_fontsize(goal_radius, world_size, fig_px),
                color=GOAL_COLOR, fontweight="bold", ha="center", va="center", zorder=4)

    for trail in trails:
        if len(trail) > 1:
            t = np.array(trail)
            ax.plot(t[:, 0], t[:, 1], color=TRAIL_COLOR, linewidth=1.2, zorder=3)

    for i, state in enumerate(states):
        x, y, theta = state[0], state[1], state[2]
        alpha = 0.45 if reached[i] else 1.0
        shape = robot_shapes[i]
        _draw_shape(ax, x, y, theta, shape, ROBOT_COLOR, alpha=alpha, zorder=5)
        # Half-width for a box: the index has to fit across the NARROW axis.
        reach = shape.radius if isinstance(shape, CircleShape) else shape.width / 2
        _draw_heading(ax, x, y, theta, reach, alpha)
        ax.text(x, y, str(i), fontsize=_index_fontsize(reach, world_size, fig_px),
                color="white", fontweight="bold", ha="center", va="center", zorder=7)

    if step_rewards is not None:
        _add_reward_bar(fig, ax, step_rewards)

    return _finish_frame(fig, ax, dpi)
