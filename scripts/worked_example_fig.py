"""The figure and the numbers for the worked instance in `paper/core_guided.tex`.

Runs `splinecegar` on `open_cross_4_unicycle2` at seed 0 and draws the four stages of one
round of the loop on the lower row of that scenario: the seeded B-spline, the refuted
proposal, the core-guided repair, and the verified plan. The upper row is the mirror image
and is resolved identically one round later, so drawing it twice would say nothing new.

Every quantity quoted in the paper's Sec. "A Fully Worked Instance" is printed here, so a
claim in the text can be checked against a rerun rather than against a note.

    python scripts/worked_example_fig.py
"""
import sys

sys.path.insert(0, ".")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from hydra import compose  # noqa: E402
from hydra.initialize import initialize_config_dir  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.core.collision.shapes import shape_distance  # noqa: E402
from src.core.env.factory import build_env  # noqa: E402
from src.planning import splinecegar as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs" / "worked_example.pdf"
BLUE, RED = "#1f77b4", "#d62728"
COL = [BLUE, RED]
CLEAR = 0.05


def _env_and_params():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=["approach=planning", "approach.method=splinecegar",
                                 "env=open_cross_4_unicycle2"])
    OmegaConf.set_struct(cfg, False)
    env = build_env(cfg)
    env.reset(seed=0)
    return env, OmegaConf.to_container(cfg.approach.splinecegar, resolve=True)


def _box(ax, env, state, i):
    """The robot's actual body at a pose -- not its bounding disc, which is 2.2x wider."""
    w, ln = env.robots[i].shape.width, env.robots[i].shape.length
    patch = Rectangle((-w / 2, -ln / 2), w, ln, facecolor=COL[i], edgecolor="k",
                      lw=0.5, zorder=6)
    patch.set_transform(matplotlib.transforms.Affine2D()
                        .rotate(state[2]).translate(state[0], state[1]) + ax.transData)
    ax.add_patch(patch)


def _frame(ax, title, xlim=(0.0, 17.0), ylim=(-0.05, 2.05)):
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.15, lw=0.4)
    ax.set_ylabel("y [m]", fontsize=7)


def main():
    env, params = _env_and_params()
    body = 2.0 * env.robots[0].shape.bounding_radius + CLEAR

    rng = np.random.default_rng(0)
    fits = [S.sample(env, i, CLEAR, rng, params) for i in range(env._n)]
    fastest = [S.variants(env, i, *fits[i], params, np.random.default_rng(0))[0]
               for i in range(env._n)]
    out = S.plan(env, dict(params), CLEAR)
    if out is None:
        raise SystemExit("splinecegar failed on the worked instance -- nothing to draw")
    tracks, _, info = out
    print("INFO", info)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.2))

    # (a) the representation: one clamped cubic B-spline per robot, control points shown.
    ax = axes[0]
    for i in (0, 1):
        pts = S.curve(*fits[i])[0]
        ax.plot(pts[:, 0], pts[:, 1], color=COL[i], lw=1.5,
                ls="-" if i == 0 else (0, (5, 4)), zorder=3)
        ax.plot(fits[i][1][0], fits[i][1][1], "o", color=COL[i], ms=3.0, alpha=0.75, zorder=4)
        _box(ax, env, env._states[i], i)
        goal = np.asarray(env._goals[i], float)
        ax.plot(goal[0], goal[1], "*", color=COL[i], ms=8, zorder=5)
    n_ctrl = fits[0][1].shape[1]
    span = float(S.curve(*fits[0])[3][-1]) / (n_ctrl - 3)
    ax.annotate(f"{n_ctrl} control points, one interior knot span = {span:.2f} m of arclength",
                (8.5, 1.55), ha="center", fontsize=7)
    _frame(ax, r"(a) each robot's candidate set starts from one cubic B-spline "
               r"$C_i(u)=\sum_m P_{i,m}B_{m,3}(u)$")

    # (b) the first proposal, and the refutation that kills it.
    ax = axes[1]
    ta, tb = fastest[0][0], fastest[1][0]
    at = S.first_contact(env.robots[0].shape, env.robots[1].shape, ta, tb, CLEAR)
    shared = min(len(ta), len(tb))
    k = int(np.argmin([np.linalg.norm(ta[q][:2] - tb[q][:2]) for q in range(shared)]))
    for i, track in ((0, ta), (1, tb)):
        ax.plot(track[:, 0], track[:, 1], color=COL[i], lw=1.5,
                ls="-" if i == 0 else (0, (5, 4)), zorder=3)
        _box(ax, env, track[k], i)
    ax.plot(at[0], at[1], "x", color="k", ms=10, mew=2.0, zorder=8)
    ax.annotate(rf"first contact $\xi=({at[0]:.2f},\,{at[1]:.2f})$ at step {k}",
                (at[0], 1.5), ha="center", fontsize=7)
    _frame(ax, r"(b) z3 proposes $(c_0,c_1)=(0,0)$; the oracle refutes it and the clause "
               r"$\neg x_{0,0}\vee\neg x_{1,0}$ is learned")

    # (c) the repair, zoomed. The shaded band is the measured support of the edit: this is
    # the locality claim, drawn rather than asserted.
    ax = axes[2]
    knots, ctrl = fits[0]
    before = S.curve(knots, ctrl)[0]
    after = S.curve(knots, S.nudge(knots, ctrl, np.asarray(at), body, 2.0 * body))[0]
    moved = np.flatnonzero(np.linalg.norm(after - before, axis=1) > 1e-3)
    lo, hi = before[moved[0], 0], before[moved[-1], 0]
    ax.axvspan(lo, hi, color="0.88", zorder=0)
    ax.plot(before[:, 0], before[:, 1], color=BLUE, lw=1.2, ls=":", zorder=2, label="original")
    ax.plot(ctrl[0], ctrl[1], "o", color=BLUE, ms=3.5, alpha=0.45, zorder=3)
    ax.plot(after[:, 0], after[:, 1], color=BLUE, lw=1.8, zorder=4, label="repaired")
    nudged = S.nudge(knots, ctrl, np.asarray(at), body, 2.0 * body)
    ax.plot(nudged[0], nudged[1], "s--", color="k", ms=3.5, lw=0.7, alpha=0.85, zorder=5,
            label="control polygon")
    ax.plot(at[0], at[1], "x", color="k", ms=10, mew=2.0, zorder=8)
    total = float(S.curve(knots, ctrl)[3][-1])
    ax.annotate(f"support of the edit: {hi - lo:.2f} m of a {total:.2f} m curve",
                (0.5 * (lo + hi), 1.72), ha="center", fontsize=7)
    ax.legend(fontsize=6.5, loc="lower left", ncol=3, framealpha=0.9)
    _frame(ax, r"(c) the core blames $\{0,1\}$ at $\xi$; 3 of 19 control points move by "
               r"$\eta\,\mathbf{n}$ — the rest of the curve is bit-identical",
           xlim=(4.0, 13.0))
    ax.set_aspect("auto")            # a true aspect here would make the edit invisible

    # (d) the verified plan, with the bodies drawn at their closest approach.
    ax = axes[3]
    gaps = [shape_distance(env.robots[0].shape, tuple(tracks[0][q][:3]),
                           env.robots[1].shape, tuple(tracks[1][q][:3]))
            for q in range(len(tracks[0]))]
    k = int(np.argmin(gaps))
    for i in (0, 1):
        ax.plot(tracks[i][:, 0], tracks[i][:, 1], color=COL[i], lw=1.6, zorder=3)
        _box(ax, env, tracks[i][k], i)
        goal = np.asarray(env._goals[i], float)
        ax.plot(goal[0], goal[1], "*", color=COL[i], ms=8, zorder=5)
    ax.annotate(f"min surface gap {min(gaps):.2f} m", (tracks[0][k][0], 1.55),
                ha="center", fontsize=7)
    _frame(ax, f"(d) verified plan: both robots swing, {info['steps']} steps = "
               f"{info['steps'] * env.dt:.1f} s, {info['resampled']} resamples")
    axes[-1].set_xlabel("x [m]", fontsize=7)

    plt.tight_layout(h_pad=0.9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")

    print(f"pair (0,1) min surface gap {min(gaps):.4f} m at step {k}")
    print(f"edit span {hi - lo:.3f} m, {len(moved)} of {len(before)} curve samples changed")
    for i in range(env._n):
        track = tracks[i]
        lat = float(np.abs(track[:, 1] - env._states[i][1]).max())
        reached = np.linalg.norm(track[:, :2] - np.asarray(env._goals[i], float)[:2], axis=1)
        arrive = int(np.flatnonzero(reached < env.goal_radius)[0])
        length = float(np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1).sum())
        print(f"robot {i}: excursion {lat:.3f} m, arrives {arrive * env.dt:.1f} s, "
              f"path {length:.3f} m, max|v| {np.abs(track[:, 3]).max():.3f}, "
              f"max|w| {np.abs(track[:, 4]).max():.3f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
