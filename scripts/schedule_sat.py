#!/usr/bin/env python3
"""Is the constructive planner's SCHEDULE what fails on cluttered_cross_32, or its ROUTES?

    .venv/bin/python scripts/schedule_sat.py env=cluttered_cross_32_unicycle2

Greedy insertion seats 29 of 32 robots there. Two explanations fit that observation equally
well and want opposite fixes:

  A. a valid assignment EXISTS over the candidates the planner already generates, and
     greedy insertion is too weak to find it        -> fix the scheduler;
  B. no assignment exists over those candidates at all, so no scheduler however clever
     could succeed                                  -> fix ROUTE GENERATION.

A greedy failure cannot tell them apart; a complete solver can. This poses the planner's
OWN candidate set -- its lane offsets, its speed variants, its delay grid, its collision
code, read out of `_build` through the `_capture` hook rather than re-derived -- to z3.

WHAT EACH ANSWER MEANS. UNSAT is UNSAT *over this candidate set*, never over the instance:
some other route entirely might still work. That is exactly the useful reading, because it
is the claim that separates B from A. SAT is the stronger verdict: the witness is re-checked
with the environment's own `_verify` before it is believed, so a reported witness is a plan.

The delay space collapses because `_clash` holds a robot at its start before it departs and
at its goal after it arrives. Shifting BOTH robots equally changes nothing, so a pair
clashes as a function of the DIFFERENCE of its two delays alone -- a 1-D question per pair,
answerable by reading diagonals off one distance matrix per trajectory pair.
"""
import pathlib
import sys
import time

sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from hydra import compose, initialize  # noqa: E402

from src.approach.planning import constructive as C  # noqa: E402
from src.approach.planning import flat, geometric_rrt  # noqa: E402
from src.approach.planning.constructive import _forbidden, _runs  # noqa: E402


def _guides(env, clearance):
    rng = np.random.default_rng(0)
    out = []
    for i in range(env._n):
        s0 = np.asarray(env._states[i], float)
        g = np.asarray(env._goals[i], float)
        p = geometric_rrt.plan_path(
            s0[:2], g[:2], env._obstacles, env._world_size,
            radius=env.robots[i].shape.bounding_radius + clearance,
            max_iters=5000, step=0.6, goal_bias=0.1, shortcut=True, rng=rng)
        out.append(np.asarray(p, float) if p is not None else np.vstack([s0[:2], g[:2]]))
    return out


def _candidates(env, cap, params, clearance, dt, smooth, speeds):
    """Every (lane, speed) trajectory the planner could have chosen, per robot.

    A superset of what `_build` picks: it commits to one lane per robot and falls back to
    the centre line when that lane is blocked, whereas every unblocked lane is offered here.
    Enlarging the set can only make UNSAT harder to reach, so it does not weaken verdict B.
    """
    refs, windows = cap["refs"], cap["windows"]
    lanes, pitch, taper = cap["lanes"], cap["pitch"], cap["taper"]
    orbit_of, axis_of, lane_w = cap["orbit_of"], cap["axis_of"], cap["lane_w"]
    lat, blur_cap = cap["lat"], float(params.get("blur_cap", 0.5))
    pad = float(params.get("window_pad", 0.35))
    per = []
    for i in range(env._n):
        win = windows.get(i)
        if win:
            win = (max(0.0, win[0] - pad), min(1.0, win[1] + pad))
        dense = C._resample(refs[i], 128)
        if i in orbit_of:
            offs, win = [orbit_of[i][0]], orbit_of[i][1]
        elif win and lanes > 1:
            offs = [0.0] + [(c - (lanes - 1) / 2.0) * pitch for c in range(lanes)]
        else:
            offs = [0.0]
        # Same budget `_build` uses: what the lane pitch leaves over after the bodies.
        room = (blur_cap * lane_w if (i in orbit_of or pitch <= 1e-9)
                else blur_cap * max(0.0, pitch - (lat + clearance)))
        start, goal = np.asarray(env._states[i], float), np.asarray(env._goals[i], float)
        got = []
        for dy in offs:
            route = dense
            if win and abs(dy) > 1e-9:
                route = C._offset_path(dense, dy, win[0], win[1], taper,
                                       axis_of.get(i, (0.0, 1.0)))
                if C._hits_obstacle(env, i, route):
                    continue
            full = np.vstack([start[:2], route, goal[:2]])
            blur = C._fit_blur(env, i, full, route, smooth, room)
            if blur is None:
                r2 = C._simplify(route, float(params.get("simplify_tol", 0.05)))
                built = C._legs(env, i, np.vstack([start[:2], r2, goal[:2]]), dt)
                if built is not None:
                    got.append(built[0])
                continue
            for sp in speeds:
                built = flat.trajectory(env.robots[i], env._states[i], full, dt,
                                        smooth=blur, slow=sp)
                if built is not None:
                    got.append(built[0])
        if not got:
            return None, i
        per.append(got)
    return per, None


def main(argv):
    import z3

    from src.env.factory import build_env

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose("config", overrides=["approach=planning", "approach.method=karc",
                                           "init=fixed"] + argv)
    env = build_env(cfg)
    env.reset()
    clearance, dt = 0.05, float(env.dt)
    params = dict(cfg.approach.karc)
    params["constructive"] = True
    smooth = float(params.get("smooth", 0.12))
    speeds = [1.0] + [float(x) for x in params.get("speeds", (1.4, 2.0))]

    cap: dict = {}
    params["_capture"] = cap
    C._build(env, params, clearance, guides=_guides(env, clearance))
    if not cap:
        print("planner derived no geometry (no conflicts); nothing to pose")
        return 1

    per, bad = _candidates(env, cap, params, clearance, dt, smooth, speeds)
    if per is None:
        print(f"robot {bad} has no buildable candidate at all -- verdict B without a solve")
        return 0

    n = env._n
    step = max(1, int(params.get("delay_step", 5)))
    horizon = int(getattr(env, "max_steps", 0) or 0) or 10 ** 6
    kmax = horizon // step
    print(f"n={n} candidates/robot={[len(p) for p in per]} "
          f"lanes={cap['lanes']} pitch={cap['pitch']:.3f}")
    print(f"delay grid: step={step} kmax={kmax} horizon={horizon}", flush=True)

    tag = next((a.split("=", 1)[1] for a in argv if a.startswith("env=")), "scenario")
    sel = [z3.Int(f"c{i}") for i in range(n)]
    kd = [z3.Int(f"k{i}") for i in range(n)]
    s = z3.Solver()
    for i in range(n):
        s.add(sel[i] >= 0, sel[i] < len(per[i]), kd[i] >= 0)
        for a, t in enumerate(per[i]):
            s.add(z3.Implies(sel[i] == a, kd[i] * step + len(t) <= horizon))

    # Pair prune: if no candidate of i passes within a body-and-clearance of any candidate
    # of j ANYWHERE in space, no relative timing can bring them together.
    thr = 2 * max(env.robots[i].shape.bounding_radius for i in range(n)) + clearance
    box = [(np.min([t[:, :2].min(axis=0) for t in per[i]], axis=0) - thr,
            np.max([t[:, :2].max(axis=0) for t in per[i]], axis=0) + thr) for i in range(n)]

    t0, combos, constrained = time.time(), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if (box[i][0] > box[j][1]).any() or (box[j][0] > box[i][1]).any():
                continue
            for a, ta in enumerate(per[i]):
                for b, tb in enumerate(per[j]):
                    combos += 1
                    bad_d = _forbidden(env, i, j, ta, tb, step, kmax, clearance)
                    if not bad_d:
                        continue
                    constrained += 1
                    guard = z3.And(sel[i] == a, sel[j] == b)
                    for lo, hi in _runs(bad_d):
                        s.add(z3.Implies(guard, z3.Not(z3.And(kd[i] - kd[j] >= lo,
                                                              kd[i] - kd[j] <= hi))))
        print(f"  robot {i}: {combos} combos tested, {constrained} constrained, "
              f"{time.time() - t0:.0f}s", flush=True)

    # A wall clock on the solver, because `unknown` and `unsat` are opposite verdicts and
    # a hang would silently look like neither.
    # Write the formula out in SMT-LIB2 so it can be read, diffed, or handed to another
    # solver. The encoding is otherwise only ever a Python object graph, which makes the
    # one claim worth checking -- which theory fragment this actually lands in -- something
    # you have to take on trust rather than read. QF_LIA rather than QF_IDL because of the
    # horizon atoms: `k_i * step + len <= cap` carries a coefficient, and a coefficient is
    # what puts it outside difference logic. Dividing through by `step` would bring it back.
    smt = f"experiments/schedule_sat_{tag}.smt2"
    pathlib.Path(smt).write_text(
        f"; {tag}: {n} robots, delay step {step}, horizon {horizon}\n"
        f"; candidates per robot: {[len(c) for c in per]}\n"
        "(set-logic QF_LIA)\n" + s.sexpr() + "(check-sat)\n(get-model)\n")
    print(f"SMT-LIB2 -> {smt}")

    s.set("timeout", 1000 * int(params.get("sat_timeout", 3600)))
    print(f"clash tables built in {time.time() - t0:.0f}s; solving ...", flush=True)
    t1 = time.time()
    res = s.check()
    print(f"z3: {res}  ({time.time() - t1:.0f}s)")
    if res == z3.unknown:
        print("\nz3 gave up inside the time limit. Neither verdict: raise sat_timeout.")
        return 1
    if res != z3.sat:
        print("\nUNSAT over the planner's own candidate set: no assignment of lane, speed")
        print("and delay seats every robot. No scheduler can succeed on these routes --")
        print("the fix belongs in ROUTE GENERATION, not in the scheduler.")
        return 0

    m = s.model()
    pick = [m[sel[i]].as_long() for i in range(n)]
    delay = [m[kd[i]].as_long() * step for i in range(n)]
    tracks = [per[i][pick[i]] for i in range(n)]
    starts = [np.asarray(x, float) for x in env._states]
    held = [np.vstack([np.repeat(starts[i][None, :], delay[i], axis=0), tracks[i]])
            if delay[i] else tracks[i] for i in range(n)]
    T = max(len(h) for h in held)
    full = [np.vstack([h, np.repeat(h[-1][None, :], T - len(h), axis=0)])
            if len(h) < T else h for h in held]
    rep: dict = {}
    gap = C._verify(env, full, clearance, rep)
    print(f"\nWITNESS steps={T} delays={delay} candidates={pick}")
    if gap is None:
        print(f"verify FAILED: {rep} -- the encoding does not match the checker")
        return 1
    print(f"verify PASS, min surface gap {gap:.4f}")
    print("\nA valid assignment EXISTS over the planner's own candidates. Greedy insertion")
    print("is what fails, not the routes. Study this witness and fix the SCHEDULER.")


    np.savez(f"experiments/schedule_sat_{tag}.npz", delay=np.array(delay),
             pick=np.array(pick), tracks=np.stack(full))
    _gif(env, full, f"experiments/schedule_sat_{tag}.gif")
    return 0


def _gif(env, tracks, path, skip=4, fps=15):
    """The witness, driven. A schedule that only exists as a delay vector is unreadable."""
    from src.approach.rollout import save_gif
    from src.viz import render_frame_with_shapes
    shapes = [r.shape for r in env.robots]
    frames = []
    for t in range(0, len(tracks[0]), skip):
        frames.append(render_frame_with_shapes(
            states=[a[t] for a in tracks], robot_shapes=shapes, goals=env._goals,
            obstacles=env._obstacles,
            trails=[[p for p in a[:t + 1, :2]] for a in tracks],
            world_size=env._world_size, reached=[False] * env._n, step=t,
            title="SAT witness schedule", goal_radius=env.goal_radius))
    save_gif(frames, path, fps=fps)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
