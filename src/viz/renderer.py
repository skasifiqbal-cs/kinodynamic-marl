"""Stateless matplotlib renderer — supports circle and box shapes."""
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


def _draw_circle_robot(ax, x, y, theta, r, color, alpha):
    """Draw circular robot body with heading wedge."""
    ax.add_patch(mpatches.Circle((x, y), r, color=color, alpha=alpha, zorder=5))
    angle_deg = np.degrees(theta)
    ax.add_patch(mpatches.Wedge(
        (x, y), r * 0.88, angle_deg - 30, angle_deg + 30,
        color="white", alpha=0.85, zorder=6,
    ))


def _draw_box_robot(ax, x, y, theta, shape: BoxShape, color, alpha):
    """Draw OBB robot body with heading line."""
    corners = _obb_corners(x, y, theta, shape.width, shape.length)
    ax.add_patch(Polygon(corners, closed=True, color=color, alpha=alpha, zorder=5))
    front_x = x + (shape.length / 2) * np.cos(theta)
    front_y = y + (shape.length / 2) * np.sin(theta)
    ax.plot([x, front_x], [y, front_y], color="white", lw=2.0, zorder=6)


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
    parts = []
    for i, (agent, r) in enumerate(sorted(step_rewards.items())):
        label = f"A{i}"
        parts.append(f"{label}: {r:+.2f}")
    text = "    ".join(parts)
    ax.set_xlabel(text, fontsize=8, labelpad=4, family="monospace")


def _finish_frame(fig, ax, dpi: int) -> np.ndarray:
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return np.array(Image.open(buf).convert("RGB"))


def render_frame(
    states: List[np.ndarray],
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
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_title(f"step {step}", fontsize=9, pad=3)

    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape,
                    color=OBS_COLOR, alpha=0.75, zorder=2)

    for i, goal in enumerate(goals):
        c = GOAL_COLORS[i % len(GOAL_COLORS)]
        ax.add_patch(mpatches.Circle((goal[0], goal[1]), goal_radius,
                                     color=c, alpha=0.25, zorder=1))
        ax.plot(goal[0], goal[1], "*", color=c, markersize=11, zorder=3)

    for i, trail in enumerate(trails):
        if len(trail) > 1:
            t = np.array(trail)
            c = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(t[:, 0], t[:, 1], color=c, alpha=0.25, linewidth=1.2, zorder=3)

    for i, state in enumerate(states):
        x, y, theta = state[0], state[1], state[2]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        alpha = 0.5 if reached[i] else 1.0
        _draw_circle_robot(ax, x, y, theta, 0.13, c, alpha)
        label = f"A{i}✓" if reached[i] else f"A{i}"
        ax.text(x + 0.17, y + 0.17, label, fontsize=7, color=c,
                fontweight="bold", zorder=8)

    handles = [mpatches.Patch(color=AGENT_COLORS[i % len(AGENT_COLORS)], label=f"Agent {i}")
               for i in range(len(states))]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7, borderpad=0.4)

    if step_rewards is not None:
        _add_reward_bar(fig, ax, step_rewards)

    return _finish_frame(fig, ax, dpi)


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
    """Full render with correct robot shapes (circle or OBB) and heading wedge."""
    dpi = 96
    fig_in = fig_px / dpi
    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_title(f"step {step}", fontsize=9, pad=3)

    for obs in obstacles:
        _draw_shape(ax, obs.x, obs.y, obs.angle, obs.shape,
                    color=OBS_COLOR, alpha=0.75, zorder=2)

    for i, goal in enumerate(goals):
        c = GOAL_COLORS[i % len(GOAL_COLORS)]
        ax.add_patch(mpatches.Circle((goal[0], goal[1]), goal_radius, color=c, alpha=0.25, zorder=1))
        ax.plot(goal[0], goal[1], "*", color=c, markersize=11, zorder=3)

    for i, trail in enumerate(trails):
        if len(trail) > 1:
            t = np.array(trail)
            ax.plot(t[:, 0], t[:, 1], color=AGENT_COLORS[i % len(AGENT_COLORS)],
                    alpha=0.25, linewidth=1.2, zorder=3)

    for i, state in enumerate(states):
        x, y, theta = state[0], state[1], state[2]
        c = AGENT_COLORS[i % len(AGENT_COLORS)]
        alpha = 0.5 if reached[i] else 1.0
        shape = robot_shapes[i]
        if isinstance(shape, CircleShape):
            _draw_circle_robot(ax, x, y, theta, shape.radius, c, alpha)
        else:
            _draw_box_robot(ax, x, y, theta, shape, c, alpha)
        label = f"A{i}✓" if reached[i] else f"A{i}"
        ax.text(x + 0.17, y + 0.17, label, fontsize=7, color=c,
                fontweight="bold", zorder=8)

    handles = [mpatches.Patch(color=AGENT_COLORS[i % len(AGENT_COLORS)], label=f"Agent {i}")
               for i in range(len(states))]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7, borderpad=0.4)

    if step_rewards is not None:
        _add_reward_bar(fig, ax, step_rewards)

    return _finish_frame(fig, ax, dpi)
