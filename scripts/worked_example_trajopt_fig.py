"""The figure and the numbers for the worked instance in `paper/cegar_trajopt.tex`.

Runs `cegar` with `drive=trajopt` ONCE on `open_cross_4_unicycle2` at seed 0 and draws the
loop on the lower row: the guide and the two optimised candidates it yields, the refuted
first proposal, the core-guided yield re-solves, and the verified plan.

Run it in a FRESH process and run the planner once. z3's choice among satisfying models
depends on the order its terms were created, which is process-global state. Every number
the paper quotes is printed here from that single run.

    python scripts/worked_example_trajopt_fig.py
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

from src.approach.planning import cegar as C  # noqa: E402
from src.approach.planning import geometric_rrt  # noqa: E402
from src.collision.shapes import shape_distance  # noqa: E402
from src.conflict.pairwise import contact_step  # noqa: E402
from src.env.factory import build_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs" / "worked_example_trajopt.pdf"
COL = ["#1f77b4", "#d62728"]
DASH = ["-", (0, (5, 4))]
CLEAR = 0.05


def _env_and_params():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="config",
                      overrides=["approach=planning", "approach.method=cegar",
                                 "env=open_cross_4_unicycle2", "approach.cegar.drive=trajopt"])
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


def main():
    env, params = _env_and_params()
    n = env._n
    trace = []
    log = {"runs": [], "yields": [], "checks": 0, "refuted": 0}

    real_run, real_yield, real_hits = C._topt_run, C._topt_yield_jobs, C._hits

    def rnd():
        return sum("unsat core blames" in t["label"] for t in trace)

    def run(env_, pool, jobs, info):
        got = real_run(env_, pool, jobs, info)
        ok = [i for i, _ in got]
        log["runs"].append({
            "round": rnd(), "checks": log["checks"],
            "jobs": [{"robot": i, "prefix": len(pre), "steps": spec[5]["horizon"],
                      "avoid": "avoid" in spec[5], "seed": spec[5]["guides"][0],
                      "x0": spec[1]} for i, pre, spec in jobs],
            "solved": [(i, c) for i, c in got], "n_ok": len(ok)})
        return got

    def yields(env_, i, j, ci, cj, clearance, params_, back):
        fi = np.vstack([env_._states[i], ci[0]])
        fj = np.vstack([env_._states[j], cj[0]])
        k = contact_step(env_.robots[i].shape, env_.robots[j].shape, fi, fj, clearance)
        log["yields"].append({"round": rnd(), "i": i, "j": j, "k": k, "back": back,
                              "cj": cj[0], "len_ci": len(ci[0])})
        return real_yield(env_, i, j, ci, cj, clearance, params_, back)

    def hits(*a, **k):
        log["checks"] += 1
        got = real_hits(*a, **k)
        log["refuted"] += got is not None
        return got

    C._topt_run, C._topt_yield_jobs, C._hits = run, yields, hits
    try:
        out = C.plan(env, dict(params), CLEAR, trace=trace)
    finally:
        C._topt_run, C._topt_yield_jobs, C._hits = real_run, real_yield, real_hits
    if out is None:
        raise SystemExit("cegar(trajopt) failed on the worked instance -- nothing to draw")
    tracks, _, info = out

    # ── numbers ────────────────────────────────────────────────────────────────────
    print("INFO", info)
    print("\nTRACE")
    for t in trace:
        print("  ", t["label"], [(round(a, 2), round(b, 2)) for a, b in t["markers"]][:4])
    print("\nSOLVE BATCHES (round 0 = seeding)")
    cands = [[] for _ in range(n)]
    for b in log["runs"]:
        print(f"   round {b['round']}: {len(b['jobs'])} jobs, {b['n_ok']} converged, "
              f"checks so far {b['checks']}")
        for jb in b["jobs"]:
            print(f"      robot {jb['robot']}: prefix {jb['prefix']} steps, horizon "
                  f"{jb['steps']}, avoid={jb['avoid']}, start {np.round(jb['x0'][:2], 2)}, "
                  f"seed max |y-start| {np.abs(jb['seed'][:, 1] - jb['seed'][0, 1]).max():.2f}")
        for i, c in b["solved"]:
            cands[i].append(c)
            L = float(np.linalg.norm(np.diff(c[0][:, :2], axis=0), axis=1).sum())
            print(f"      -> robot {i} candidate {len(cands[i]) - 1}: {len(c[0])} steps, "
                  f"path {L:.2f} m, max excursion "
                  f"{np.abs(c[0][:, 1] - env._states[i][1]).max():.3f} m, "
                  f"max|v| {np.abs(c[0][:, 3]).max():.3f}")
    print("\nYIELD REQUESTS")
    for y in log["yields"]:
        print(f"   round {y['round']}: robot {y['i']} yields to robot {y['j']}, first contact step "
              f"{y['k']}, back {y['back']}, re-solve from step {max((y['k'] or 0) - y['back'], 0)}")
    print("\nFINAL PLAN")
    for i, track in enumerate(tracks):
        pick = next((c for c, cc in enumerate(cands[i])
                     if len(cc[0]) <= len(track) and np.allclose(cc[0], track[:len(cc[0])])),
                    None)
        lat = float(np.abs(track[:, 1] - env._states[i][1]).max())
        reach = np.linalg.norm(track[:, :2] - np.asarray(env._goals[i], float)[:2], axis=1)
        arrive = int(np.flatnonzero(reach < env.goal_radius)[0])
        L = float(np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1).sum())
        print(f"   robot {i}: candidate {pick}, excursion {lat:.3f} m, arrives "
              f"{arrive * env.dt:.1f} s, path {L:.3f} m, max|v| {np.abs(track[:, 3]).max():.3f}, "
              f"max|w| {np.abs(track[:, 4]).max():.3f}")
    gaps = {}
    for i, j in ((0, 1), (2, 3)):
        g = [shape_distance(env.robots[i].shape, tuple(tracks[i][k][:3]),
                            env.robots[j].shape, tuple(tracks[j][k][:3]))
             for k in range(len(tracks[0]))]
        gaps[(i, j)] = (min(g), int(np.argmin(g)))
        print(f"   pair ({i},{j}) min surface gap {min(g):.4f} m at step {int(np.argmin(g)) + 1} "
              "(start = step 0)")
    print(f"   candidate lengths: {[[len(c[0]) for c in cc] for cc in cands]}")

    # ── figure: lower row only ─────────────────────────────────────────────────────
    s0 = np.asarray(env._states[0], float)[:2]
    g0 = np.asarray(env._goals[0], float)[:2]
    raw = geometric_rrt.plan_path(s0, g0, env._obstacles, env._world_size,
                                  radius=env.robots[0].shape.bounding_radius + CLEAR,
                                  max_iters=int(params["guide_rrt_iters"]),
                                  step=float(params["guide_rrt_step"]),
                                  goal_bias=float(params["guide_rrt_goal_bias"]),
                                  shortcut=False, rng=np.random.default_rng(0))
    seed_batch = log["runs"][0]
    guide0 = next(jb["seed"] for jb in seed_batch["jobs"] if jb["robot"] == 0)
    first = next(t for t in trace if "proposal refuted" in t["label"])
    rep = [b for b in log["runs"] if b["round"] >= 1 and any(jb["avoid"] for jb in b["jobs"])]
    rep_low = [(b, jb) for b in rep for jb in b["jobs"] if jb["robot"] in (0, 1)]
    rep_solved = [(i, c) for b in rep for i, c in b["solved"] if i in (0, 1)]

    ys_c = [1.0] + [float(v) for _, jb in rep_low for v in (jb["seed"][:, 1].min(),
                                                           jb["seed"][:, 1].max())]
    ys_c += [float(v) for _, c in rep_solved for v in (c[0][:, 1].min(), c[0][:, 1].max())]
    ya = (min(0.0, float(raw[:, 1].min()) - 0.2), float(raw[:, 1].max()) + 0.4)
    yb = (-0.05, 2.3)
    yc = (max(min(ys_c) - 0.4, -0.05), max(ys_c) + 0.6)
    yd_pts = [float(v) for i in (0, 1) for v in (tracks[i][:, 1].min(), tracks[i][:, 1].max())]
    yd = (min(yd_pts) - 0.4, max(yd_pts) + 0.55)
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

    # (a) guide and the fast candidate it yields.
    ax = axes[0]
    fast0 = cands[0][0][0]
    ax.plot(raw[:, 0], raw[:, 1], "-o", color="0.6", lw=0.7, ms=1.6, zorder=2,
            label=f"RRT path ({len(raw)} vertices)")
    ax.plot(guide0[:, 0], guide0[:, 1], "s", color=COL[0], ms=4, zorder=4,
            label=f"after shortcut ({len(guide0)} vertices)")
    ax.plot(fast0[:, 0], fast0[:, 1], color=COL[0], lw=1.6, zorder=3,
            label=f"optimised candidate ({len(fast0)} steps)")
    _box(ax, env, env._states[0], 0)
    ax.plot(g0[0], g0[1], "*", color=COL[0], ms=8, zorder=5)
    ax.legend(fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 0.22), ncol=3,
              framealpha=0.95)
    frame(ax, "(a) robot 0: RRT path, shortcut, and the trajectory the optimiser returns "
              "from it", ya)

    # (b) first refuted proposal.
    ax = axes[1]
    row = [p for p in first["markers"] if p[1] < 8.5]
    ta, tb = np.asarray(first["anim"][0], float), np.asarray(first["anim"][1], float)
    for i, tr in ((0, ta), (1, tb)):
        ax.plot(tr[:, 0], tr[:, 1], color=COL[i], ls=DASH[i], lw=1.5, zorder=3)
    if row:
        xi = row[0]
        k = contact_step(env.robots[0].shape, env.robots[1].shape, ta, tb, CLEAR)
        _box(ax, env, ta[min(k, len(ta) - 1)], 0)
        _box(ax, env, tb[min(k, len(tb) - 1)], 1)
        ax.plot(xi[0], xi[1], "x", color="k", ms=10, mew=2.0, zorder=8)
        # +1: `anim` starts at the first executed state, the paper counts the start as step 0.
        ax.annotate(rf"first contact $\xi=({xi[0]:.2f},\,{xi[1]:.2f})$ at step {k + 1}",
                    (xi[0], 1.75), ha="center", fontsize=7)
    frame(ax, r"(b) the first proposal is refuted; the clause "
              r"$\neg x_{0,c_0}\vee\neg x_{1,c_1}$ is learned", yb)

    # (c) yield re-solves in the lower row.
    ax = axes[2]
    for b, jb in rep_low:
        i = jb["robot"]
        ax.plot(jb["seed"][:, 0], jb["seed"][:, 1], ":", color=COL[i], lw=0.9, zorder=2)
        ax.plot(jb["x0"][0], jb["x0"][1], "o", color=COL[i], ms=4, zorder=5)
    for i, c in rep_solved:
        ax.plot(c[0][:, 0], c[0][:, 1], color=COL[i], ls=DASH[i], lw=1.5, zorder=3,
                label=f"robot {i} yield candidate ({len(c[0])} steps)")
    ax.plot([], [], ":", color="0.3", lw=0.9, label="seeds, pushed to both sides")
    ax.plot([], [], "o", color="0.3", ms=4, label="re-solve starts here")
    ax.legend(fontsize=6.0, loc="upper left", ncol=1, framealpha=0.95)
    frame(ax, "(c) the core blames {0,1}: each is re-solved from before the contact, "
              "keeping clear of the other", yc)

    # (d) verified plan.
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
              "after bisection", yd)
    axes[-1].set_xlabel("x [m]", fontsize=7)

    plt.tight_layout(h_pad=0.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
