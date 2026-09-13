"""Core-guided B-spline coordination (OURS).

A candidate motion is a **cubic B-spline**, and a B-spline is its control points. That one
representation choice replaces most of what `cegar.py` hand-builds:

  * smoothness is the basis, not a Gaussian blur -- a cubic B-spline is C^2 by
    construction, so curvature (and with it the cornering cap v <= w_max/|kappa|) is
    continuous without anything being smoothed;
  * curvature is ANALYTIC, from the spline's own derivatives, rather than two finite
    differences of a blurred polyline followed by a second blur to hide the noise the
    first one made;
  * repair is LOCAL, because a B-spline basis function is nonzero over only k+1 knot
    spans: displacing the two or three control points nearest a contested point changes
    that stretch of the curve and provably nothing else. `cegar` had to throw the whole
    path away and resample it.

The coordination loop is the same idea and deliberately so -- this module exists to change
the geometry layer, not the reasoning:

  1. Each robot owns a growing set of candidate splines. A candidate is driven as one
     trajectory (TOPP speed profile + differential flatness), plus slowed variants and
     variants that stop EN ROUTE for a while. Nothing waits on its start line.
  2. z3 picks one candidate per robot. Collision clauses are added LAZILY, only for pairs
     the solver actually proposes, refuted by `conflict.pairwise.first_contact`.
  3. UNSAT -> take the unsat core. It names the robots whose candidate sets are jointly
     impossible and the points where their candidates met. Those robots get NEW candidates
     by pushing control points sideways off the contested point -- the sampler is only
     consulted when local repair cannot clear it.
  4. Once a plan verifies, its horizon is bisected down over the same candidates and the
     same learned clauses.

References for the representation: de Boor's basis, and its use in planning by Usenko et
al. (IROS 2017), Zhou et al. (RA-L 2019, Fast-Planner) and Zhou et al. (RA-L 2021,
EGO-Planner), whose "repair a colliding spline against a collision-free guide" is the same
move as step 3 here, driven by a gradient instead of a proof.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import splev, splprep

from src.approach.planning import flat, geometric_rrt
from src.approach.planning.base import BasePlanner
from src.approach.planning.schedule import verify
from src.conflict.pairwise import first_contact

try:                                                        # pragma: no cover - optional
    import z3
except ImportError:                                         # pragma: no cover
    z3 = None

SAMPLES = 600


# ── the spline layer ────────────────────────────────────────────────────────────────

def fit(path, spacing):
    """A clamped cubic B-spline through a sampled path. Returns (knots, control points).

    Knots are placed by ARCLENGTH -- one control point per `spacing` metres -- and the
    curve is the least-squares fit over those knots. Two measured reasons for fixing the
    knots instead of letting scipy choose them from a smoothing factor `s`:

      * accuracy. On a 12 m S-bend the smoothing fit kept 5 control points and strayed
        0.107 m; the same path with a control point per metre strays 0.002 m.
      * locality, which is the whole reason for using a spline here. Support is k+1 knot
        spans, so with 5 control points an edit covers most of the curve and "local
        repair" means nothing. Spacing sets the size of the window a repair can touch.

    A least-squares fit does not interpolate its endpoints, and those two points are the
    robot's start pose and its goal, so the first and last control points are overwritten
    with them afterwards -- which the clamped basis turns into exact endpoints.
    """
    pts = np.asarray(path, float)[:, :2]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    if length < 1e-9:
        return None
    if len(pts) < 8:            # splprep needs points to fit; densify a short polyline
        cum = np.concatenate([[0.0], np.cumsum(seg)]) / length
        u = np.linspace(0.0, 1.0, 64)
        pts = np.column_stack([np.interp(u, cum, pts[:, 0]), np.interp(u, cum, pts[:, 1])])
    n_ctrl = int(np.clip(round(length / max(float(spacing), 1e-3)) + 4, 8, 60))
    interior = np.linspace(0.0, 1.0, n_ctrl - 4 + 2)[1:-1]
    # splprep wants the FULL knot vector here, boundary multiplicities included. Handing it
    # the interior knots alone is accepted silently and mis-read: it takes the first and
    # last four AS the boundary knots, so 17 control points came back as 9 with the knots
    # bunched in the middle -- and "local repair" then moved most of the curve.
    knots = np.concatenate([np.zeros(4), interior, np.ones(4)])
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], t=knots, task=-1, k=3)
    except (TypeError, ValueError):
        try:                                     # degenerate knots: let scipy choose
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * 0.05 ** 2, k=3)
        except (TypeError, ValueError):          # pragma: no cover - scipy
            return None
    ctrl = np.asarray(tck[1], float).copy()
    ctrl[:, 0] = pts[0]
    ctrl[:, -1] = pts[-1]
    return tck[0], ctrl


def curve(knots, ctrl, samples=SAMPLES):
    """Points, heading, curvature and arclength of a spline, sampled uniformly in u.

    Every derivative here is the spline's own: `splev(..., der=1)` is exactly the degree-2
    B-spline whose control points are the scaled differences of these, so kappa is the
    analytic curvature of the curve that will actually be driven -- not a finite difference
    of something that approximates it.
    """
    tck = (knots, list(ctrl), 3)
    u = np.linspace(0.0, 1.0, samples)
    try:
        p = np.column_stack(splev(u, tck))
        d1 = np.column_stack(splev(u, tck, der=1))
        d2 = np.column_stack(splev(u, tck, der=2))
    except (TypeError, ValueError):                          # pragma: no cover - scipy
        return None
    sp = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp ** 3
    th = np.unwrap(np.arctan2(d1[:, 1], d1[:, 0]))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return p, th, kappa, s


def nudge(knots, ctrl, at, push, reach):
    """Displace the control points near `at`, perpendicular to the curve. New control points.

    This is the repair operator, and it is one line of geometry only because of local
    support: a cubic basis function is nonzero over four knot spans, so moving the control
    points within `reach` of the contested point changes the curve there and leaves the
    rest bit-identical. The first and last control points are never touched -- they are the
    start pose and the goal.
    """
    ctrl = np.asarray(ctrl, float).copy()
    pts = ctrl.T                                   # (m, 2): scipy stores x and y rows
    at = np.asarray(at, float)[:2]
    d = np.linalg.norm(pts - at, axis=1)
    hit = np.flatnonzero(d < reach)
    hit = hit[(hit > 0) & (hit < len(pts) - 1)]
    if len(hit) == 0:
        return None
    # Perpendicular to the control polygon, which is the curve's own direction to within
    # the hull it lies in.
    tan = pts[min(hit[-1] + 1, len(pts) - 1)] - pts[max(hit[0] - 1, 0)]
    if np.linalg.norm(tan) < 1e-9:
        return None
    normal = np.array([-tan[1], tan[0]]) / np.linalg.norm(tan)
    # Taper the displacement so the repaired stretch rejoins the original smoothly rather
    # than acquiring a corner at each end of the edit.
    w = 0.5 * (1.0 + np.cos(np.pi * np.clip(d[hit] / reach, 0.0, 1.0)))
    pts[hit] = pts[hit] + push * w[:, None] * normal
    return pts.T


# ── candidates ──────────────────────────────────────────────────────────────────────

def _reference(robot, knots, ctrl, dt, slow):
    """TOPP speed profile along a spline, sampled at the env's timestep."""
    got = curve(knots, ctrl)
    if got is None:
        return None
    p, th, kappa, s = got
    if s[-1] < 1e-6:
        return None
    # `_topp` wants a uniform arclength grid; the spline is uniform in its parameter, not
    # in arclength, so re-space before profiling.
    even = np.linspace(0.0, s[-1], len(s))
    p = np.column_stack([np.interp(even, s, p[:, 0]), np.interp(even, s, p[:, 1])])
    th = np.interp(even, s, th)
    kappa = np.interp(even, s, kappa)
    ds = float(even[1] - even[0])
    v = flat._topp(kappa, ds, robot.v_max, robot.omega_max, robot.a_max)
    v = v / max(float(slow), 1.0)
    for _ in range(6):
        ref = flat._sample(p, th, kappa, even, v, dt)
        if ref is None:
            return None
        a = np.diff(ref["v"]) / dt
        al = np.diff(ref["w"]) / dt
        if len(a) == 0:
            return ref
        ratio = max(float(np.max(np.abs(a))) / robot.a_max,
                    float(np.max(np.abs(al))) / robot.alpha_max, 1e-9)
        if ratio <= 1.0 + 1e-6:
            return ref
        v = v / np.sqrt(min(ratio, 4.0))    # t -> L*t: both accelerations fall as L^2
    return ref


def drive(env, i, knots, ctrl, params, slow=1.0, cut=None, wait=0):
    """One candidate motion: (states, controls, knots, ctrl), or None.

    `cut` stops the robot at that fraction of the spline for `wait` steps and then carries
    on, so a wait is a motion with a real deceleration and a real restart -- not a frozen
    frame, and never taken on the start line.
    """
    dt, robot = float(env.dt), env.robots[i]
    state = np.asarray(env._states[i], float)
    if flat.hits_obstacle(robot.shape, env._obstacles, curve(knots, ctrl)[0]):
        return None
    if cut is None:
        ref = _reference(robot, knots, ctrl, dt, slow)
        if ref is None:
            return None
        got = flat.execute(robot, state, ref, dt)
        return None if got is None else (got[0], got[1], knots, ctrl)

    ref = _reference(robot, knots, ctrl, dt, slow)
    if ref is None or len(ref["x"]) < 4:
        return None
    k = int(np.clip(int(len(ref["x"]) * float(cut)), 1, len(ref["x"]) - 2))
    head = {key: val[:k + 1] for key, val in ref.items()}
    tail = {key: val[k:] for key, val in ref.items()}
    for part in (head, tail):                      # each half starts and ends at rest
        part["v"] = part["v"].copy()
        part["v"][0] = part["v"][-1] = 0.0
        part["w"] = part["w"].copy()
        part["w"][0] = part["w"][-1] = 0.0
    a = flat.execute(robot, state, head, dt)
    if a is None:
        return None
    b = flat.execute(robot, a[0][-1], tail, dt)
    if b is None:
        return None
    hold = np.repeat(a[0][-1][None, :], int(wait), axis=0)
    return (np.vstack([a[0], hold, b[0]]),
            np.vstack([a[1], np.zeros((int(wait), 2)), b[1]]), knots, ctrl)


def variants(env, i, knots, ctrl, params, rng):
    """What one spline offers: as fast as it goes, slower, and with a stop part-way."""
    out = []
    for slow in (1.0, float(params.get("slow", 1.6))):
        got = drive(env, i, knots, ctrl, params, slow=slow)
        if got is not None:
            out.append(got)
    for w in [int(x) for x in params.get("waits", [30, 80])]:
        got = drive(env, i, knots, ctrl, params, cut=float(rng.uniform(0.25, 0.7)), wait=w)
        if got is not None:
            out.append(got)
    return out


def sample(env, i, clearance, rng, params):
    """A fresh sampled guide for robot i, fitted as a spline. Returns (knots, ctrl)."""
    s0 = np.asarray(env._states[i], float)
    g = np.asarray(env._goals[i], float)
    path = geometric_rrt.plan_path(
        s0[:2], g[:2], env._obstacles, env._world_size,
        radius=env.robots[i].shape.bounding_radius + clearance,
        max_iters=int(params.get("guide_rrt_iters", 5000)),
        step=float(params.get("guide_rrt_step", 0.6)),
        goal_bias=float(params.get("guide_rrt_goal_bias", 0.1)),
        shortcut=bool(params.get("guide_rrt_shortcut", True)), rng=rng)
    if path is None:
        path = np.vstack([s0[:2], g[:2]])
    return fit(path, float(params.get("ctrl_spacing", 1.0)))


# ── the loop ────────────────────────────────────────────────────────────────────────

def plan(env, params, clearance=0.05, trace=None):
    """A verified plan, or None. Returns (tracks, controls, info)."""
    if z3 is None:
        if isinstance(params, dict):
            params.setdefault("_reject", {})["z3"] = "missing"
        return None

    n = env._n
    rng = np.random.default_rng(int(params.get("seed", 0)))
    deadline = time.perf_counter() + float(params.get("timeout", 600.0))
    body = 2.0 * max(r.shape.bounding_radius for r in env.robots) + clearance
    shapes = [r.shape for r in env.robots]

    fits = [sample(env, i, clearance, rng, params) for i in range(n)]
    if any(f is None for f in fits):
        if isinstance(params, dict):
            params.setdefault("_reject", {})["unfittable_guide"] = 1
        return None
    cand = [variants(env, i, *fits[i], params, rng) for i in range(n)]
    if any(not c for c in cand):
        if isinstance(params, dict):
            params.setdefault("_reject", {})["undrivable_robot"] = 1
        return None

    touch: dict = {}        # (i, a, j, b) -> contested point: every refutation ever found
    info = {"rounds": 0, "refutations": 0, "repaired": 0, "resampled": 0, "checks": 0}

    def _snap(label, pick=None, spots=()):
        if trace is None:
            return
        static = [np.asarray(curve(*cand[i][0][2:])[0], float) for i in range(n)]
        anim = ([np.asarray(cand[i][pick[i]][0], float)[:, :3] for i in range(n)]
                if pick is not None else [])
        trace.append({"label": label, "static": static, "anim": anim,
                      "markers": [(float(p[0]), float(p[1])) for p in spots],
                      "waypoints": []})

    def _search(cap, tag=""):
        """The lazy loop under a horizon cap. Returns (plan | None, verdict, blame)."""
        x = [[z3.Bool(f"x_{i}_{c}") for c in range(len(cand[i]))] for i in range(n)]
        shown: list = []
        s = z3.Solver()
        for i in range(n):
            ok = [x[i][c] for c in range(len(cand[i]))
                  if cap is None or len(cand[i][c][0]) <= cap]
            if not ok:
                return None, "unsat", {}
            s.add(z3.PbEq([(v, 1) for v in ok], 1))
            for c in range(len(cand[i])):
                if cap is not None and len(cand[i][c][0]) > cap:
                    s.add(z3.Not(x[i][c]))
        for (i, a_, j, b_) in touch:
            s.assert_and_track(z3.Or(z3.Not(x[i][a_]), z3.Not(x[j][b_])),
                               z3.Bool(f"r_{i}_{a_}_{j}_{b_}"))

        while True:
            left = deadline - time.perf_counter()
            if left <= 0.0:
                return None, "timeout", {}
            # Bound the solver call itself. Checking the clock between calls is not enough:
            # with thousands of learned clauses one `check()` can run for minutes, which is
            # how circular_cross_32 sailed past a 300 s budget without ever looking up.
            s.set("timeout", max(1, int(left * 1000)))
            got = s.check()
            if got == z3.unknown:
                return None, "timeout", {}
            if got != z3.sat:
                break
            m = s.model()
            pick = [next(c for c in range(len(cand[i])) if z3.is_true(m[x[i][c]]))
                    for i in range(n)]
            fresh = []
            for i in range(n):
                if time.perf_counter() > deadline:
                    return None, "timeout", {}
                for j in range(i + 1, n):
                    key = (i, pick[i], j, pick[j])
                    if key in touch:
                        continue
                    info["checks"] += 1
                    at = first_contact(shapes[i], shapes[j], cand[i][pick[i]][0],
                                       cand[j][pick[j]][0], clearance)
                    if at is not None:
                        touch[key] = at
                        fresh.append(key)
            if not fresh:
                out = _assemble(env, cand, pick, clearance, info)
                if out is not None:
                    _snap(f"{tag}solved, {out[2]['steps']} steps", pick)
                    return out, "sat", {}
                s.add(z3.Or([z3.Not(x[i][pick[i]]) for i in range(n)]))
                continue
            info["refutations"] += len(fresh)
            if not shown:
                shown.append(1)
                _snap(f"{tag}round {info['rounds']}: proposal refuted, {len(fresh)} pairs",
                      pick, [touch[k] for k in fresh])
            for (i, a_, j, b_) in fresh:
                s.assert_and_track(z3.Or(z3.Not(x[i][a_]), z3.Not(x[j][b_])),
                                   z3.Bool(f"r_{i}_{a_}_{j}_{b_}"))

        blame: dict = {}
        for name in (str(c) for c in s.unsat_core()):
            if not name.startswith("r_"):
                continue
            i, a_, j, b_ = (int(v) for v in name[2:].split("_"))
            at = touch.get((i, a_, j, b_))
            if at is None:
                continue
            blame.setdefault(i, []).append((a_, at))
            blame.setdefault(j, []).append((b_, at))
        return None, "unsat", blame

    for rnd in range(int(params.get("rounds", 60))):
        info["rounds"] = rnd + 1
        # The budget has to be checked HERE as well as inside the solver loop: repairing
        # and re-driving a blamed robot's candidates is the expensive half of a round, and
        # a deadline the refinement never looks at is not a deadline. Measured on
        # circular_cross_32, where 32 robots meeting at one hub put 20 robots in a core
        # and the round outran a 600 s budget without a single solver call being made.
        if time.perf_counter() > deadline:
            if isinstance(params, dict):
                params.setdefault("_reject", {}).update({"timed_out": 1, **info})
            return None
        out, verdict, blame = _search(None)
        if verdict == "timeout":
            if isinstance(params, dict):
                params.setdefault("_reject", {}).update({"timed_out": 1, **info})
            return None
        if out is not None:
            lo = max(min(len(c[0]) for c in cand[i]) for i in range(n))
            for _ in range(int(params.get("bisect", 12))):
                T = out[2]["steps"]
                if lo >= T:
                    break
                mid = (lo + T - 1) // 2
                got, got_verdict, _ = _search(mid, tag=f"bisect <= {mid}: ")
                if got is not None:
                    out = got
                elif got_verdict == "timeout":
                    break
                else:
                    lo = mid + 1
            return out

        if not blame:
            blame = {i: [(0, None)] for i in range(n)}
        _snap(f"round {rnd + 1}: unsat core blames {len(blame)} robots, repairing",
              spots=[at for hits in blame.values() for _, at in hits[:1] if at is not None])
        for i, hits in blame.items():
            if time.perf_counter() > deadline:
                break
            cand[i].extend(_repair(env, i, cand, hits, params, rng, clearance, body, info))

    if isinstance(params, dict):
        params.setdefault("_reject", {}).update({"exhausted": 1, **info})
    return None


def _repair(env, i, cand, hits, params, rng, clearance, body, info):
    """New candidates for a robot the core blamed.

    Local first: push the control points near the contested point off it, both ways, over
    a couple of distances. Only if every one of those is undrivable or hits an obstacle is
    the sampler asked for a whole new guide -- resampling is the fallback, not the method.
    """
    grew = []
    reach = float(params.get("repair_reach", 2.0)) * body
    for c, at in hits[:2]:
        if at is None:
            continue
        knots, ctrl = cand[i][c][2], cand[i][c][3]
        for scale in (1.0, 2.0):
            for sign in (1.0, -1.0):
                moved = nudge(knots, ctrl, at, sign * scale * body, reach)
                if moved is None:
                    continue
                got = drive(env, i, knots, moved, params)
                if got is not None:
                    grew.append(got)
                    info["repaired"] += 1
    if grew:
        # One timing alternative alongside the geometric ones: the core says WHEN as much
        # as where, and a robot that cannot go round may still be able to go later.
        held = drive(env, i, cand[i][0][2], cand[i][0][3], params,
                     cut=float(rng.uniform(0.2, 0.75)),
                     wait=int(params.get("long_wait", 200)))
        if held is not None:
            grew.append(held)
        return grew

    fresh = sample(env, i, clearance, rng, params)
    info["resampled"] += 1
    return [] if fresh is None else variants(env, i, *fresh, params, rng)


def _assemble(env, cand, pick, clearance, info):
    """Pad the chosen candidates to one horizon and verify with the env's own checker."""
    n = env._n
    tracks = [cand[i][pick[i]][0] for i in range(n)]
    ctrls = [cand[i][pick[i]][1] for i in range(n)]
    T = max(len(t) for t in tracks)
    tracks = [np.vstack([t, np.repeat(t[-1][None, :], T - len(t), axis=0)])
              if len(t) < T else t for t in tracks]
    ctrls = [np.vstack([c, np.zeros((T - len(c), 2))]) if len(c) < T else c for c in ctrls]
    rep: dict = {}
    gap = verify(env, tracks, clearance, rep)
    if gap is None:
        return None
    return tracks, ctrls, {**info, "steps": T, "min_surface_gap": round(float(gap), 4),
                           "candidates": sum(len(c) for c in cand)}


class SplineCEGARPlanner(BasePlanner):
    """`approach.method=splinecegar` -- B-spline candidates, lazy SMT, core-guided repair."""

    method = "splinecegar"

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        params = dict(self.params)
        clearance = float(self.approach_cfg.get("trajopt", {}).get("clearance", 0.05))
        agents = list(env.possible_agents)
        self._controls = {a: [] for a in agents}
        self.trace = [] if params.get("trace", False) else None
        self.stats = {"method": "splinecegar", "solved": 0}
        built = plan(env, params, clearance, trace=self.trace)
        if built is not None:
            tracks, ctrls, got = built
            self.stats["solved"] = 1
            self.stats.update(got)
            self.stats["makespan"] = round(len(tracks[0]) * env.dt, 4)
            for i, a in enumerate(agents):
                self._controls[a] = [np.asarray(u, float) for u in ctrls[i]]
        else:
            self.stats.update(params.get("_reject", {}))
        self.stats["wall_time"] = round(time.perf_counter() - t0, 3)
        self._plan = self._controls

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            out[agent] = seq.pop(0) if seq else np.zeros(env.robots[i].action_dim)
        return out
