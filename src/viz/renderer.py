"""Stateless matplotlib renderer — supports circle and box shapes."""
from __future__ import annotations

import io
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

from src.collision.shapes import Obstacle, CircleShape, BoxShape

AGENT_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
GOAL_COLORS  = ["#FF6B6B", "#74B9FF", "#52D9CC", "#F4D35E"]
OBS_COLOR    = "#6C757D"
BG_COLOR     = "#F1F3F4"
GRID_COLOR   = "#DADCE0"


def _obb_corners(x: float, y: float, theta: float, w: float, l: float) -> np.ndarray:
    hw, hl = w / 2, l / 2
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([x, y])


def _draw_shape(ax, x, y, theta, shape, color, alpha=1.0, zorder=2):
    if isinstance(shape, CircleShape):
        ax.add_patch(mpatches.Circle((x, y), shape.radius,
                                     color=color, alpha=alpha, zorder=zorder))
    elif isinstance(shape, BoxShape):
        corners = _obb_corners(x, y, theta, shape.width, shape.length)
        ax.add_patch(Polygon(corners, closed=True,
                             color=color, alpha=alpha, zorder=zorder))


def render_frame(
    states: List[np.ndarray],
    goals: List[np.ndarray],
    obstacles: List[Obstacle],
    trails: List[List[np.ndarray]],
    world_size: float,
    reached: List[bool],
    step: int,
    fig_px: int = 480,
) -> np.ndarray:
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.5, world_size + 0.5)
    ax.set_ylim(-0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_title(f"step {step}", fontsize=9, pad=3)

    # Obstacles
    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape,
                    color=OBS_COLOR, alpha=0.75, zorder=2)

    # Goal regions (circle regardless of robot shape — goal is a point)
    for i, goal in enumerate(goals):
        c = GOAL_COLORS[i % len(GOAL_COLORS)]
        ax.add_patch(mpatches.Circle((goal[0], goal[1]), 0.2,
                                     color=c, alpha=0.25, zorder=1))
        ax.plot(goal[0], goal[1], "*", color=c, markersize=11, zorder=3)

    # Trails
    for i, trail in enumerate(trails):
        if len(trail) > 1:
            t = np.array(trail)
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(t[:, 0], t[:, 1], color=c, alpha=0.25, linewidth=1.2, zorder=3)

    # Agents
    for i, (state, shape_obj) in enumerate(
        zip(states, [None] * len(states))  # shape fetched via obstacle list not passed here
    ):
        x, y, theta = state
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        alpha = 0.5 if reached[i] else 1.0
        # Default to small circle if no shape info — env passes shapes separately
        ax.add_patch(mpatches.Circle((x, y), 0.13, color=c, alpha=alpha, zorder=5))
        dx, dy = 0.22 * np.cos(theta), 0.22 * np.sin(theta)
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=10),
                    zorder=6)
        label = f"A{i}✓" if reached[i] else f"A{i}"
        ax.text(x + 0.17, y + 0.17, label, fontsize=7, color=c,
                fontweight="bold", zorder=7)

    handles = [mpatches.Patch(color=AGENT_COLORS[i % len(AGENT_COLORS)], label=f"Agent {i}")
               for i in range(len(states))]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7, borderpad=0.4)

    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))


def render_frame_with_shapes(
    states: List[np.ndarray],
    robot_shapes,       # list of Shape objects
    goals: List[np.ndarray],
    obstacles: List[Obstacle],
    trails: List[List[np.ndarray]],
    world_size: float,
    reached: List[bool],
    step: int,
    fig_px: int = 480,
) -> np.ndarray:
    """Full render with correct robot shapes (circle or OBB)."""
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-0.5, world_size + 0.5)
    ax.set_ylim(-0.5, world_size + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_title(f"step {step}", fontsize=9, pad=3)

    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape,
                    color=OBS_COLOR, alpha=0.75, zorder=2)

    for i, goal in enumerate(goals):
        c = GOAL_COLORS[i % len(GOAL_COLORS)]
        ax.add_patch(mpatches.Circle((goal[0], goal[1]), 0.2, color=c, alpha=0.25, zorder=1))
        ax.plot(goal[0], goal[1], "*", color=c, markersize=11, zorder=3)

    for i, trail in enumerate(trails):
        if len(trail) > 1:
            t = np.array(trail)
            ax.plot(t[:, 0], t[:, 1], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    alpha=0.25, linewidth=1.2, zorder=3)

    for i, state in enumerate(states):
        x, y, theta = state
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        alpha = 0.5 if reached[i] else 1.0
        _draw_shape(ax, x, y, theta, robot_shapes[i], color=c, alpha=alpha, zorder=5)
        dx, dy = 0.25 * np.cos(theta), 0.25 * np.sin(theta)
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0, mutation_scale=10),
                    zorder=6)
        label = f"A{i}✓" if reached[i] else f"A{i}"
        ax.text(x + 0.17, y + 0.17, label, fontsize=7, color=c,
                fontweight="bold", zorder=7)

    handles = [mpatches.Patch(color=AGENT_COLORS[i % len(AGENT_COLORS)], label=f"Agent {i}")
               for i in range(len(states))]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7, borderpad=0.4)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))
