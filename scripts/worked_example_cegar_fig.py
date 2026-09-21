"""The figure and the numbers for the worked instance in `paper/cegar_paper.tex`.

Runs `cegar` ONCE on `open_cross_4_unicycle2` at seed 0 and draws the loop on the lower row:
the sampled guide and its smoothed curve, the refuted first proposal, the core-guided
resampling around discs dropped on the contested points, and the verified plan.

Run it in a FRESH process and run the planner once. z3's choice among satisfying models
depends on the order its terms were created, which is process-global state: a second
`plan()` call in the same interpreter can pick a different first proposal and end one step
different. Every number the paper quotes is printed here from that single run.

    python scripts/worked_example_cegar_fig.py
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
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.core.collision.shapes import shape_distance  # noqa: E402
from src.core.env.factory import build_env  # noqa: E402
from src.planning import cegar as C  # noqa: E402
from src.planning import flat, geometric_rrt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs" / "worked_example_cegar.pdf"
COL = ["#1f77b4", "#d62728"]
DASH = ["-", (0, (5, 4))]
CLEAR = 0.05


def _env_and_params():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=["approach=planning", "approach.method=cegar",
                                 "env=open_cross_4_unicycle2"])
    OmegaConf.set_struct(cfg, False)
    env = build_env(cfg)
    env.reset(seed=0)
    return env, OmegaConf.to_container(cfg.approach.cegar, resolve=True)


def _box(ax, env, state, i):
    w, ln = env.robots[i].shape.width, env.robots[i].shape.length
    patch = Rectangle((-w / 2, -ln / 2), w, ln, facecolor=COL[i], edgecolor="k", lw=0.5,
                      zorder=6)
    patch.set_transform(matplotlib.transforms.Affine2D()
                        .rotate(state[2]).translate(state[0], state[1]) + ax.transData)
    ax.add_patch(patch)


def _smoothed(env, i, path, params):
    """The curve the planner actually drives for a guide: same blur search, same cap."""
    robot = env.robots[i]
    base = float(params.get("smooth", 0.12))
    cap = float(params.get("blur_cap", 0.5)) * (2.0 * robot.shape.bounding_radius + CLEAR)
    blur = flat.fit_blur(robot.shape, env._obstacles, path, path, base, cap)
    got = flat._curve(path, base if blur is None else blur)
    return (None if got is None else got[0]), (base if blur is None else blur)


def main():
    env, params = _env_and_params()
    n = env._n
    trace = []
    log = {"samples": [], "variants": [], "checks": 0, "refuted": 0}

    real_sample, real_hits, real_variants = C._sample, C._hits, C._variants

    def sample(env_, i, cl, rng, p, blocks=()):
        path = real_sample(env_, i, cl, rng, p, blocks=blocks)
        rnd = sum("unsat core blames" in t["label"] for t in trace)
        log["samples"].append({"round": rnd, "robot": i, "path": np.asarray(path, float),
                               "blocks": [(b.pose[0], b.pose[1], b.shape.radius)
                                          for b in blocks],
                               "checks": log["checks"], "refuted": log["refuted"]})
        return path

    def hits(*a, **k):
        log["checks"] += 1
        got = real_hits(*a, **k)
        log["refuted"] += got is not None
        return got

    def variants(env_, i, path, p, rng, cl=CLEAR):
        got = real_variants(env_, i, path, p, rng, cl)
        log["variants"].append((i, [len(c[0]) for c in got]))
        return got

    C._sample, C._hits, C._variants = sample, hits, variants
    try:
        out = C.plan(env, dict(params), CLEAR, trace=trace)
    finally:
        C._sample, C._hits, C._variants = real_sample, real_hits, real_variants
    if out is None:
        raise SystemExit("cegar failed on the worked instance -- nothing to draw")
    tracks, _, info = out

    # ── numbers ────────────────────────────────────────────────────────────────────
    print("INFO", info)
    print("\nTRACE")
    for t in trace:
        print("  ", t["label"], [(round(a, 2), round(b, 2)) for a, b in t["markers"]][:4])
    print("\nSAMPLES (round 0 = seeding)")
    for s in log["samples"]:
        L = float(np.linalg.norm(np.diff(s["path"], axis=0), axis=1).sum())
        print(f"   round {s['round']} robot {s['robot']}: {len(s['path'])} vertices, "
              f"{L:.3f} m, blocks {[(round(x, 2), round(y, 2)) for x, y, _ in s['blocks']]}"
              f"{' r=%.3f' % s['blocks'][0][2] if s['blocks'] else ''}, "
              f"checks so far {s['checks']}, refuted so far {s['refuted']}")
    print("\nVARIANT LENGTHS (steps), in call order:")
    for i, lens in log["variants"]:
        print(f"   robot {i}: {lens}")
    for i in range(n):
        _, blur = _smoothed(env, i, log["samples"][i]["path"], params)
        print(f"   robot {i} seeded guide blur = {blur:.3f} m")
    print("\nFINAL PLAN")
    for i, track in enumerate(tracks):
        lat = float(np.abs(track[:, 1] - env._states[i][1]).max())
        reach = np.linalg.norm(track[:, :2] - np.asarray(env._goals[i], float)[:2], axis=1)
        arrive = int(np.flatnonzero(reach < env.goal_radius)[0])
        L = float(np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1).sum())
        print(f"   robot {i}: excursion {lat:.3f} m, arrives {arrive * env.dt:.1f} s, "
              f"path {L:.3f} m, max|v| {np.abs(track[:, 3]).max():.3f}, "
              f"max|w| {np.abs(track[:, 4]).max():.3f}")
    gaps = {}
    for i, j in ((0, 1), (2, 3)):
        g = [shape_distance(env.robots[i].shape, tuple(tracks[i][k][:3]),
                            env.robots[j].shape, tuple(tracks[j][k][:3]))
             for k in range(len(tracks[0]))]
        gaps[(i, j)] = (min(g), int(np.argmin(g)))
        print(f"   pair ({i},{j}) min surface gap {min(g):.4f} m at step {int(np.argmin(g))}")

    # ── figure: lower row only ─────────────────────────────────────────────────────
    s0 = np.asarray(env._states[0], float)[:2]
    g0 = np.asarray(env._goals[0], float)[:2]
    radius = env.robots[0].shape.bounding_radius + CLEAR
    kw = dict(max_iters=int(params["guide_rrt_iters"]), step=float(params["guide_rrt_step"]),
              goal_bias=float(params["guide_rrt_goal_bias"]))
    raw = geometric_rrt.plan_path(s0, g0, env._obstacles, env._world_size, radius=radius,
                                  shortcut=False, rng=np.random.default_rng(0), **kw)
    first = next(t for t in trace if "proposal refuted" in t["label"])
    repair = [s for s in log["samples"] if s["round"] == 1 and s["robot"] in (0, 1)]
    smooth_rep = {s["robot"]: _smoothed(env, s["robot"], s["path"], params)[0] for s in repair}
    ya = (min(0.0, float(raw[:, 1].min()) - 0.2), float(raw[:, 1].max()) + 0.4)
    yb = (-0.05, 2.3)
    yc_pts = [1.0] + [float(v) for c in smooth_rep.values() if c is not None
                      for v in (c[:, 1].min(), c[:, 1].max())]
    yc = (min(yc_pts) - 0.5, max(yc_pts) + 0.5)
    yd_pts = [float(v) for i in (0, 1) for v in (tracks[i][:, 1].min(), tracks[i][:, 1].max())]
    yd = (min(yd_pts) - 0.4, max(yd_pts) + 0.55)
    # Equal aspect in every panel, heights proportional to each panel's y-span: the discs
    # in (c) must stay circles, because their radius is the quantity the text quotes.
    spans = [ya[1] - ya[0], yb[1] - yb[0], yc[1] - yc[0], yd[1] - yd[0]]
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 0.41 * sum(spans) + 1.6),
                             gridspec_kw={"height_ratios": spans})

    def frame(ax, title, ylim):
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_xlim(0.0, 17.0)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.15, lw=0.4)
        ax.set_ylabel("y [m]", fontsize=7)

    # (a) the guide: raw RRT, shortcut, smoothed.
    ax = axes[0]
    short = log["samples"][0]["path"]
    curve, blur0 = _smoothed(env, 0, short, params)
    ax.plot(raw[:, 0], raw[:, 1], "-o", color="0.6", lw=0.7, ms=1.6, zorder=2,
            label=f"RRT path ({len(raw)} vertices)")
    ax.plot(short[:, 0], short[:, 1], "s", color=COL[0], ms=4, zorder=4,
            label=f"after shortcut ({len(short)} vertices)")
    ax.plot(curve[:, 0], curve[:, 1], color=COL[0], lw=1.6, zorder=3, label="smoothed curve")
    _box(ax, env, env._states[0], 0)
    ax.plot(g0[0], g0[1], "*", color=COL[0], ms=8, zorder=5)
    ax.legend(fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 0.22), ncol=3,
              framealpha=0.95)
    frame(ax, "(a) robot 0's guide: a geometric RRT path, shortcut, then smoothed "
              f"(blur {blur0:.2f} m)", ya)

    # (b) the first proposal the oracle refutes.
    ax = axes[1]
    row = [p for p in first["markers"] if p[1] < 8.5]
    for i in (0, 1):
        track = np.asarray(first["anim"][i], float)
        ax.plot(track[:, 0], track[:, 1], color=COL[i], ls=DASH[i], lw=1.5, zorder=3)
    if row:
        xi = row[0]
        ta, tb = np.asarray(first["anim"][0]), np.asarray(first["anim"][1])
        shared = min(len(ta), len(tb))
        k = int(np.argmin([np.hypot(*(ta[q][:2] - tb[q][:2])) for q in range(shared)]))
        _box(ax, env, ta[k], 0)
        _box(ax, env, tb[k], 1)
        ax.plot(xi[0], xi[1], "x", color="k", ms=10, mew=2.0, zorder=8)
        ax.annotate(rf"first contact $\xi=({xi[0]:.2f},\,{xi[1]:.2f})$", (xi[0], 1.75),
                    ha="center", fontsize=7)
    frame(ax, r"(b) the first proposal is refuted; the clause "
              r"$\neg x_{0,c_0}\vee\neg x_{1,c_1}$ is learned", yb)

    # (c) round-1 repair: discs on the contested points, resampled guides around them.
    ax = axes[2]
    for s in repair:
        i = s["robot"]
        for x, y, r in s["blocks"]:
            ax.add_patch(Circle((x, y), r, facecolor="0.85", edgecolor="0.4", lw=0.6,
                                zorder=1))
        ax.plot(s["path"][:, 0], s["path"][:, 1], "o", color=COL[i], ms=3, zorder=4)
        c = smooth_rep[i]
        if c is not None:
            L = float(np.linalg.norm(np.diff(s["path"], axis=0), axis=1).sum())
            ax.plot(c[:, 0], c[:, 1], color=COL[i], ls=DASH[i], lw=1.5, zorder=3,
                    label=f"robot {i} resampled ({L:.1f} m)")
    ax.plot([1.0, 16.0], [1.0, 1.0], ":", color="0.35", lw=1.0, zorder=2,
            label="seeded guides (15.0 m)")
    ax.legend(fontsize=6.3, loc="upper right", ncol=1, framealpha=0.95)
    frame(ax, "(c) the core blames {0,1}: discs on the contested points, both guides "
              "resampled around them", yc)

    # (d) the verified plan.
    ax = axes[3]
    gap, k = gaps[(0, 1)]
    ys = []
    for i in (0, 1):
        ax.plot(tracks[i][:, 0], tracks[i][:, 1], color=COL[i], ls=DASH[i], lw=1.6, zorder=3)
        _box(ax, env, tracks[i][k], i)
        goal = np.asarray(env._goals[i], float)
        ax.plot(goal[0], goal[1], "*", color=COL[i], ms=8, zorder=5)
        ys += [float(tracks[i][:, 1].min()), float(tracks[i][:, 1].max())]
    ax.annotate(f"closest approach: {gap:.2f} m surface gap",
                (tracks[0][k][0], max(ys) + 0.15), ha="center", fontsize=7)
    frame(ax, f"(d) verified plan: {info['steps']} steps = {info['steps'] * env.dt:.1f} s "
              f"after bisection", yd)
    axes[-1].set_xlabel("x [m]", fontsize=7)

    plt.tight_layout(h_pad=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
