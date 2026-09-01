"""Stage-A falsifier for the braking-band hypothesis (no RL, CPU, minutes).

Hypothesis S1: for a second-order robot no *position-only* potential can track the
optimal cost-to-go, and the irreducible error is the VELOCITY variation of V*, which
grows with the braking distance d_brake = v_max^2 / (2 a_max).

This measures that error directly. It solves min-time-to-goal by value iteration on a
(x, y, theta, v, omega) lattice -- using this project's own Unicycle2 dynamics -- and
reports

    gap(p, theta, omega) = 1/2 * [ max_v V*(...) - min_v V*(...) ]

which is exactly the best achievable worst-case error of a position-only potential:
for a fixed cell the optimal constant phi is the midpoint of the velocity range, and
its worst-case error is half the range.

Falsification: sweep a_max. If gap does NOT grow with d_brake, S1 is dead and the
~15 h RL sample-complexity sweep is not worth running.

Usage:
    .venv/bin/python scripts/shaping_gap.py --env swap2_unicycle2 --robot unicycle_db
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, ".")
from src.collision.shapes import build_obstacle  # noqa: E402
from src.robot import build_robot, load_robot_cfg  # noqa: E402


def action_levels(lo, hi, step):
    """Accelerations on a FIXED absolute grid, truncated to [lo, hi].

    Bang-off-bang (+-a_max only) makes the velocity resolution of the action set scale
    with a_max itself, so every a_max solves a differently-resolved problem and V* stops
    being monotone in a_max. A fixed step makes a larger a_max a strict SUPERSET of a
    smaller one, which is both physically right (a strong robot can command weak accel)
    and gives the monotonicity self-test its meaning.
    """
    k = int(np.floor(max(-lo, hi) / step + 1e-9))
    lv = np.arange(-k, k + 1) * step
    return lv[(lv >= lo - 1e-9) & (lv <= hi + 1e-9)]


def rk4(robot, X, Y, TH, V, W, a, al, dt):
    """Vectorised copy of Unicycle2Model.step (asserted equal in self_check)."""
    def f(x, y, th, v, w):
        return v * np.cos(th), v * np.sin(th), w, np.full_like(v, a), np.full_like(w, al)

    k1 = f(X, Y, TH, V, W)
    k2 = f(X + .5 * dt * k1[0], Y + .5 * dt * k1[1], TH + .5 * dt * k1[2],
           V + .5 * dt * k1[3], W + .5 * dt * k1[4])
    k3 = f(X + .5 * dt * k2[0], Y + .5 * dt * k2[1], TH + .5 * dt * k2[2],
           V + .5 * dt * k2[3], W + .5 * dt * k2[4])
    k4 = f(X + dt * k3[0], Y + dt * k3[1], TH + dt * k3[2],
           V + dt * k3[3], W + dt * k3[4])
    out = []
    for i, s in enumerate((X, Y, TH, V, W)):
        out.append(s + (dt / 6.) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
    nx, ny, nth, nv, nw = out
    nth = (nth + np.pi) % (2 * np.pi) - np.pi
    nv = np.clip(nv, robot.v_min, robot.v_max)
    nw = np.clip(nw, robot.omega_min, robot.omega_max)
    return nx, ny, nth, nv, nw


def self_check(robot, dt, n=200, seed=0):
    """The vectorised integrator must agree with the one the env actually runs."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s = np.array([rng.uniform(0, 5), rng.uniform(0, 5), rng.uniform(-np.pi, np.pi),
                      rng.uniform(robot.v_min, robot.v_max),
                      rng.uniform(robot.omega_min, robot.omega_max)])
        u = np.array([rng.uniform(robot.a_min, robot.a_max),
                      rng.uniform(robot.alpha_min, robot.alpha_max)])
        ref = robot.step(s.copy(), u, dt)
        got = rk4(robot, *[np.array([x]) for x in s], u[0], u[1], dt)
        assert np.allclose(ref, [g[0] for g in got], atol=1e-12), (ref, got)


def solve(robot, world, goal_xy, goal_r, stop_speed, dt, obstacles,
          nxy, nth, nv, nw, max_sweeps, hold, args_cell_v, args_cell_w):
    """Value iteration for min time-to-goal. Returns V* of shape (nxy,nxy,nth,nv,nw)."""
    xs = np.linspace(0, world, nxy)
    ys = np.linspace(0, world, nxy)
    ths = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    vs = np.linspace(robot.v_min, robot.v_max, nv)
    ws = np.linspace(robot.omega_min, robot.omega_max, nw)
    shape = (nxy, nxy, nth, nv, nw)
    N = int(np.prod(shape))

    X, Y, TH, V, W = (g.ravel() for g in np.meshgrid(xs, ys, ths, vs, ws, indexing="ij"))

    # ponytail: free-space test uses the robot's bounding circle, not its true OBB.
    # Conservative (shrinks free space), keeps the lattice check vectorised. Swap for
    # src.collision.shapes.collides if a scenario ever needs the exact footprint.
    rad = robot.shape.bounding_radius
    free = (X > rad) & (X < world - rad) & (Y > rad) & (Y < world - rad)
    for ob in obstacles:
        free &= (np.hypot(X - ob.x, Y - ob.y) > rad + ob.shape.bounding_radius)

    def to_idx(x, y, th, v, w):
        ix = np.clip(np.rint(x / world * (nxy - 1)), 0, nxy - 1).astype(np.int64)
        iy = np.clip(np.rint(y / world * (nxy - 1)), 0, nxy - 1).astype(np.int64)
        it = np.rint((th + np.pi) / (2 * np.pi) * nth).astype(np.int64) % nth
        iv = np.clip(np.rint((v - robot.v_min) / (robot.v_max - robot.v_min) * (nv - 1)),
                     0, nv - 1).astype(np.int64)
        iw = np.clip(np.rint((w - robot.omega_min) / (robot.omega_max - robot.omega_min)
                             * (nw - 1)), 0, nw - 1).astype(np.int64)
        return (((ix * nxy + iy) * nth + it) * nv + iv) * nw + iw

    # One env step moves at most v_max*dt. If that is far below the cell size every
    # transition snaps back into its own cell and value iteration cannot propagate, so
    # each action is HELD for `hold` sub-steps -- a motion primitive, which is also how
    # db-A* / db-CBS discretise. Edge cost is hold*dt.
    a_lv = action_levels(robot.a_min, robot.a_max, args_cell_v / (hold * dt))
    al_lv = action_levels(robot.alpha_min, robot.alpha_max, args_cell_w / (hold * dt))
    n_act = len(a_lv) * len(al_lv)
    succ = np.empty((n_act, N), dtype=np.int32)  # int32: halves memory at fine lattices
    ok = np.empty((n_act, N), dtype=bool)
    k = 0
    for a in a_lv:
        for al in al_lv:
            nx, ny, nt, nvv, nww = X, Y, TH, V, W
            valid = np.ones(N, dtype=bool)
            for _ in range(hold):
                nx, ny, nt, nvv, nww = rk4(robot, nx, ny, nt, nvv, nww, a, al, dt)
                valid &= (nx > rad) & (nx < world - rad) & (ny > rad) & (ny < world - rad)
                for ob in obstacles:
                    valid &= np.hypot(nx - ob.x, ny - ob.y) > rad + ob.shape.bounding_radius
            j = to_idx(nx, ny, nt, nvv, nww)
            succ[k] = j
            ok[k] = valid & free[j]
            k += 1

    goal = (np.hypot(X - goal_xy[0], Y - goal_xy[1]) < goal_r) & (np.abs(V) <= stop_speed) & free
    INF = np.inf
    Vv = np.where(goal, 0.0, INF)
    for it in range(max_sweeps):
        cand = np.full(N, INF)
        for k in range(succ.shape[0]):
            nxt = np.where(ok[k], Vv[succ[k]], INF)
            cand = np.minimum(cand, nxt)
        new = np.where(goal, 0.0, np.where(free, hold * dt + cand, INF))
        if np.array_equal(np.isfinite(new), np.isfinite(Vv)) and \
           np.allclose(new[np.isfinite(new)], Vv[np.isfinite(Vv)], atol=1e-12):
            break
        Vv = new
    return Vv.reshape(shape), free.reshape(shape), it + 1, n_act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="swap2_unicycle2")
    ap.add_argument("--robot", default="unicycle_db")
    ap.add_argument("--v-max", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0],
                    help="swept variable; d_brake = v_max^2/(2 a_max) grows quadratically")
    ap.add_argument("--nxy", type=int, default=32)
    ap.add_argument("--nth", type=int, default=8)
    ap.add_argument("--cell-v", type=float, default=0.05, help="velocity lattice spacing")
    ap.add_argument("--cell-w", type=float, default=0.125, help="omega lattice spacing")
    ap.add_argument("--hold", type=int, default=0)
    ap.add_argument("--a-max", type=float, nargs="+", default=None,
                    help="CLEAN MODE: fix v_max at the robot's value and sweep a_max "
                         "instead. Keeps the state space, lattice, cell population and "
                         "V* quantisation identical across rows, so the only thing that "
                         "varies is braking capability. Sweeping v_max (the default) "
                         "confounds braking distance with the width of the velocity range.")
    ap.add_argument("--stop-frac", type=float, default=None,
                    help="override env.stop_speed as a fraction of v_max, so the goal "
                         "semantics stay constant while v_max is swept")
    ap.add_argument("--max-sweeps", type=int, default=400)
    args = ap.parse_args()

    env = OmegaConf.load(f"conf/env/{args.env}.yaml")
    rc = load_robot_cfg(args.robot)
    base = build_robot(rc)
    self_check(base, float(env.dt))

    obstacles = [build_obstacle(o) for o in env.obstacles]
    goal_xy = [float(env.agents[0].goal[0]), float(env.agents[0].goal[1])]
    world, dt, goal_r = float(env.world_size), float(env.dt), float(env.goal_radius)
    cell = world / (args.nxy - 1)
    rho_turn = base.v_max / base.omega_max  # keep the turning radius fixed across the sweep

    # We sweep v_max, NOT a_max. Sweeping a_max changes how far one motion primitive
    # moves v, so the velocity lattice resolves differently at each point and the rows
    # stop being comparable (that confound produced a spurious verdict). With a_max and
    # the cell sizes fixed, the primitive's dv is constant and only d_brake varies.
    print(f"env={args.env} robot={args.robot} goal={goal_xy} goal_r={goal_r} dt={dt}")
    print(f"stop gate: {'%.2f*v_max' % args.stop_frac if args.stop_frac is not None else '%.3f abs' % float(env.stop_speed)}")
    print(f"a_max={base.a_max} (fixed)  turning radius={rho_turn:.2f} m (fixed)  "
          f"xy cell={cell:.3f} m  v cell={args.cell_v}  omega cell={args.cell_w}")
    print(f"{'swept':>7} {'d_brake':>8} {'rho':>6} {'nv':>4} {'nw':>4} {'hold':>5} "
          f"{'gap_max':>8} {'gap_med':>8} {'gap_mean':>9} {'V_med':>7} {'gap/V':>7} {'swp':>4}")

    # (v_max, a_max) pairs. --a-max => clean mode: v_max pinned, lattice identical.
    pairs = ([(base.v_max, a) for a in args.a_max] if args.a_max
             else [(v, base.a_max) for v in args.v_max])
    results = []
    for vm, am in pairs:
        om = vm / rho_turn
        # The lattice spacing must be EXACTLY cell_v / cell_w at every sweep point,
        # otherwise the rows are quantised differently and stop being comparable; and
        # the interval count must be even so v=0 / omega=0 sit on the lattice (the goal
        # test needs them). Sweep values that violate this are rejected, not rounded.
        iv, iw = 2 * vm / args.cell_v, 2 * om / args.cell_w
        for name, n, cw in (("v_max", iv, args.cell_v), ("omega_max", iw, args.cell_w)):
            if abs(n - round(n)) > 1e-9 or round(n) % 2:
                raise SystemExit(f"{name}={vm if name=='v_max' else om}: 2*{name}/{cw} = "
                                 f"{n:.4f} must be an even integer; adjust --v-max/--cell-v")
        nv, nw = int(round(iv)) + 1, int(round(iw)) + 1
        r = build_robot(OmegaConf.merge(rc, {"v_max": float(vm), "v_min": -float(vm),
                                             "omega_max": float(om), "omega_min": -float(om),
                                             "a_max": float(am)}))
        hold = args.hold or max(1, int(np.ceil(cell / (vm * dt))),
                                int(np.ceil(args.cell_v / (r.a_max * dt))))
        # One primitive changes v by a_max*hold*dt. That must be a WHOLE number of v
        # cells: otherwise the reachable velocities form an irregular sublattice, which
        # bins are reachable changes non-monotonically with a_max, and V* stops being
        # monotone in a_max -- physically impossible (less accel cannot be faster) and a
        # sure sign discretisation is dominating. <1 cell freezes v entirely.
        dv_cells = r.a_max * hold * dt / args.cell_v
        if dv_cells < 1.0 - 1e-6 or abs(dv_cells - round(dv_cells)) > 1e-6:
            valid = [k * args.cell_v / (hold * dt) for k in range(1, 9)]
            raise SystemExit(
                f"a_max={am}: one primitive moves v by {dv_cells:.3f} cells; must be a "
                f"whole number >= 1. Valid a_max at hold={hold}, cell_v={args.cell_v}: "
                + ", ".join(f"{v:.6f}" for v in valid))
        stop = args.stop_frac * vm if args.stop_frac is not None else float(env.stop_speed)
        Vg, free, sw, n_act = solve(r, world, goal_xy, goal_r, stop, dt, obstacles,
                                    args.nxy, args.nth, nv, nw, args.max_sweeps, hold,
                                    args.cell_v, args.cell_w)
        iw0 = nw // 2                                   # evaluate at omega = 0
        Vs = Vg[:, :, :, :, iw0]                        # (nxy, nxy, nth, nv)
        ok = np.isfinite(Vs).all(axis=3)                # cells finite for EVERY v
        results.append(dict(vm=vm, am=am, om=om, nv=nv, nw=nw, hold=hold, sw=sw, V=Vs,
                            ok=ok, n_act=n_act, d_brake=vm ** 2 / (2 * r.a_max)))
        print(f"  v_max={vm:.4f} a_max={am:.4f} dv/prim={dv_cells:.1f} cells nv={nv} nw={nw} hold={hold} "
              f"(travel/prim={vm*hold*dt:.3f}m vs cell={cell:.3f}m) "
              f"solved cells={ok.sum()}/{ok.size} ({ok.sum()/ok.size:.1%}) sweeps={sw}")

    # Compare on the cells solved at EVERY sweep point, so coverage differences between
    # rows cannot masquerade as a change in the gap.
    common = np.logical_and.reduce([q["ok"] for q in results])
    print(f"common cell set: {common.sum()} of {common.size} (x,y,theta) cells "
          f"({common.sum()/common.size:.1%})")
    if common.sum() == 0:
        print("no cells solved at every sweep point -- widen the lattice or the sweep")
        return
    rows = []
    for q in results:
        V = q["V"]
        Vc = V[common]                      # mask first: inf - inf elsewhere is noise
        gap = 0.5 * (Vc.max(axis=1) - Vc.min(axis=1))
        vmed = float(np.median(Vc[:, q["nv"] // 2]))               # V* at v = 0
        rows.append((q["am"], q["d_brake"], q["d_brake"] / goal_r, q["nv"], q["nw"],
                     q["hold"], gap.max(), np.median(gap), gap.mean(),
                     vmed, gap.mean() / vmed, q["sw"]))
        print(f"{rows[-1][0]:7.4f} {rows[-1][1]:8.3f} {rows[-1][2]:6.2f} {rows[-1][3]:4d} "
              f"{rows[-1][4]:4d} {rows[-1][5]:5d} {rows[-1][6]:8.3f} {rows[-1][7]:8.3f} "
              f"{rows[-1][8]:9.3f} {rows[-1][9]:7.2f} {rows[-1][10]:7.3f} {rows[-1][11]:4d}")

    if args.a_max:
        order = np.argsort([-q["am"] for q in results])   # strongest accel first
        bad = 0
        for i in range(len(order) - 1):
            hi, lo = results[order[i]], results[order[i + 1]]
            viol = (hi["V"][common] - lo["V"][common] > 1e-9).sum()
            bad += viol
            print(f"  self-test a_max {hi['am']:.4f} vs {lo['am']:.4f}: "
                  f"{viol} cells where MORE accel costs MORE time"
                  + ("  <-- IMPOSSIBLE" if viol else "  ok"))
        if bad:
            print("SOLVER INCONSISTENT: V* is not monotone in a_max. Discretisation is "
                  "dominating the physics; the gap numbers below mean nothing.")
            return

    holds = {r[5] for r in rows}
    d = np.array([r[1] for r in rows]); g = np.array([r[8] for r in rows])  # mean, unquantised
    o = np.argsort(d)
    mono = bool(np.all(np.diff(g[o]) >= -1e-9))
    print(f"\nprimitive length constant across rows: {len(holds) == 1} (hold={sorted(holds)})")
    print(f"mean gap monotone increasing in d_brake: {mono}   "
          f"(median is quantised in units of hold*dt/2 = {rows[0][5]*dt/2:.3f}s -- "
          f"do not test monotonicity on it)")
    print(f"corr(d_brake, gap) = {np.corrcoef(d, g)[0, 1]:.3f}")
    if len(holds) != 1:
        print("VERDICT: INCONCLUSIVE -- rows use different primitive lengths, so V* is "
              "quantised differently per row. Re-run with --hold fixed.")
    else:
        print(f"VERDICT: {'S1 survives Stage A' if mono else 'S1 FALSIFIED at Stage A'}")


if __name__ == "__main__":
    main()
