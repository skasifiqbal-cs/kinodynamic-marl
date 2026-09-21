"""Is Open Cross hard at N=32 because the corridor is FULL, or because K-ARC SPENDS it badly?

Row spacing s = 15.0/(N/2-1), and a pair needs centre separation D to pass. Two robots of
one row veering apart, plus the neighbouring row doing the same, needs s >= 2D. That is the
budget. This plots what the planner actually withdraws from it: the max lateral excursion
|y - y_row| of every robot in the plan K-ARC produced, against D/2 (the share a fair
allocation would take) and s/2 (the point at which a robot is standing in its neighbour's row).

If excursions cluster near D/2 the corridor is genuinely full and no allocator saves us.
If early-resolved pairs take far more than D/2 and late ones are starved, the corridor is
NOT full -- it is badly allocated, and a global assignment is worth building.

    python scripts/lateral_budget.py approach=planning env=open_cross_32_unicycle2 \
        init=fixed approach.karc.timeout=420 approach.trajopt.body_discs=2
"""
import sys

sys.path.insert(0, ".")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import hydra  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.core.env.factory import build_env  # noqa: E402
from src.planning import build_planner  # noqa: E402


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.approach.get("type") != "planning" or "karc" not in cfg.approach:
        raise SystemExit("pass approach=planning")
    OmegaConf.set_struct(cfg, False)
    cfg.approach.method = "karc"

    env = build_env(cfg)
    env.reset(seed=0)
    n = env._n
    y0 = np.asarray([env._states[i][1] for i in range(n)], float)

    cfg.approach.karc.trace = True
    planner = build_planner(cfg.approach)
    planner.reset(env)

    # `_traj` starts as ONE state per robot and only grows as segments commit, so on a
    # failed plan it is length 1 and every excursion is trivially 0.0 -- an empty
    # measurement that looks exactly like "nobody needed to move". Fall back to the trace,
    # which records the trajectories the planner PROPOSED, which is the quantity of
    # interest anyway: how much room does a resolution try to take?
    traj = [np.asarray(t, float) for t in planner._traj]
    committed = min(len(t) for t in traj)
    src = "committed plan"
    if committed <= 1:
        src = "proposed trajectories (trace) -- NO PLAN COMMITTED"
        dev = np.zeros(n)
        for stage in (planner.trace or []):
            for i, a in enumerate(stage.get("anim", [])):
                a = np.asarray(a, float)
                if i < n and len(a):
                    dev[i] = max(dev[i], float(np.abs(a[:, 1] - y0[i]).max()))
    else:
        dev = np.asarray([float(np.abs(t[:, 1] - y0[i]).max()) for i, t in enumerate(traj)])
    print(f"source: {src}  (shortest committed trajectory = {committed} states, "
          f"{len(planner.trace or [])} trace stages)")
    if float(dev.max()) == 0.0:
        raise SystemExit("no motion recorded at all -- measurement is empty, not a result")

    rows = sorted(set(np.round(y0, 3)))
    s = float(np.diff(rows).mean()) if len(rows) > 1 else float("inf")
    # Same cover the trajectory optimiser constrains with: k discs over the body.
    sh = env.robots[0].shape
    k = int(cfg.approach.trajopt.get("body_discs", 1))
    fwd, lat = float(sh.width), float(sh.length)
    r_k = float(np.hypot(fwd / (2 * k), lat / 2))
    clearance = float(cfg.approach.karc.get("clearance", 0.05))
    D = 2 * r_k + clearance

    print(f"N={n}  rows={len(rows)}  spacing s={s:.3f}  discs k={k}  "
          f"disc radius={r_k:.4f}  required separation D={D:.3f}  "
          f"budget s>=2D -> {s:.3f} >= {2*D:.3f} : {'OK' if s >= 2*D else 'INFEASIBLE'}")
    print(f"spare room per row gap = {s - 2*D:+.3f} m "
          f"({100*(s-2*D)/s:+.0f}% of spacing)")
    print(f"excursion |y-y_row|: mean={dev.mean():.3f} max={dev.max():.3f} "
          f"fair share D/2={D/2:.3f}  neighbour's row at s/2={s/2:.3f}")
    over = int((dev > D / 2).sum())
    print(f"robots exceeding their fair share: {over}/{n}")
    print(f"robots past the row midline (in a neighbour's lane): {int((dev > s/2).sum())}/{n}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = np.argsort(y0)
    ax.bar(range(n), dev[order], color=["#c0392b" if d > s / 2 else
                                        "#e08a1e" if d > D / 2 else "#2b7a78"
                                        for d in dev[order]])
    ax.axhline(D / 2, ls="--", c="#2b7a78", label=f"fair share D/2 = {D/2:.3f} m")
    ax.axhline(s / 2, ls="--", c="#c0392b", label=f"neighbour's lane s/2 = {s/2:.3f} m")
    ax.set_xlabel("robot (ordered by row)")
    ax.set_ylabel("max |y - y_row|  (m)")
    ax.set_title(f"Lateral room withdrawn per robot — N={n}, k={k} discs, "
                 f"spacing {s:.3f} m, spare {s-2*D:+.3f} m\n{src}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = cfg.get("plot_path", None) or f"experiments/viz/lateral_budget_n{n}_k{k}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
