"""Conflict-guided resampling with a lazy SMT scheduler (OURS).

The difference from `constructive.py` is what decides the geometry. There, lanes,
roundabouts and passing sides come from a rulebook written against the benchmarks -- it
works, but every device in it encodes something the author knew about the scene. Here
NOTHING is designed: each robot owns a growing set of sampled candidate motions, a solver
picks one per robot, and when no pick works the solver's own explanation says which robots
to resample and where. The rulebook is replaced by the refinement loop.

One round:

  1. Each robot has candidates -- a sampled path, driven as one smooth flat trajectory
     (`flat.trajectory`), plus slowed variants and variants that stop EN ROUTE for a while.
     Waiting is a motion like any other, sampled along the path; nothing is held at its
     start line, because a robot idling on its start is a device only a planner that owns
     the whole world can use.
  2. A SAT problem: one candidate per robot, and no pair of picks may collide. Collision
     clauses are added LAZILY -- the solver proposes, the pair check refutes, the refutation
     comes back as a clause. Most pairs never get checked, which is what makes it cheap.
  3. SAT with no violated pair -> verify against the environment's own collision checker
     and return. UNSAT -> ask z3 for the unsat core. The core is a set of pairwise
     refutations: exactly the robots and the places where the current candidate sets are
     not enough. Those robots get resampled, with a disc dropped on the contested point so
     the sampler is pushed out of that corridor rather than back into it.

That last step is the point of the method. An UNSAT answer over a candidate set is a proof
that no schedule exists over those candidates, and its core localises the proof: instead of
"try again with another seed", the failure names the robots to re-plan and the region to
avoid. The loop then repeats with a strictly larger candidate set, so the same refutation
can never be produced twice.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from src.approach.planning import flat, geometric_rrt, schedule
from src.approach.planning.base import BasePlanner
from src.collision.shapes import CircleShape, shape_distance
from src.conflict.margin import inscribed_radius

try:                                                        # pragma: no cover - optional
    import z3
except ImportError:                                         # pragma: no cover
    z3 = None


# ── candidates ──────────────────────────────────────────────────────────────────────

def _split(path, frac):
    """Cut a polyline at `frac` of its arclength. Returns (head, tail), both polylines."""
    pts = np.asarray(path, float)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return None
    cut = float(frac) * s[-1]
    k = int(np.searchsorted(s, cut))
    k = min(max(k, 1), len(pts) - 1)
    t = (cut - s[k - 1]) / max(s[k] - s[k - 1], 1e-9)
    mid = pts[k - 1] + t * (pts[k] - pts[k - 1])
    return np.vstack([pts[:k], mid]), np.vstack([mid, pts[k:]])


def _drive(env, i, path, params, clearance=0.05, slow=1.0, cut=None, wait=0):
    """One candidate motion. Returns (states, controls) or None.

    `cut` stops the robot at that fraction of the path for `wait` steps and then carries
    on. Both halves are flown as their own flat trajectory, so the stop is a real
    deceleration to rest and the restart is a real acceleration -- not a frozen frame.
    """
    dt, robot = float(env.dt), env.robots[i]
    smooth = float(params.get("smooth", 0.12))
    state = np.asarray(env._states[i], float)
    # A sampled path is a polyline, and a kink in it caps the cornering speed for the whole
    # traverse (v <= w_max/kappa) even where the geometry is otherwise open. Round the
    # corners as hard as the corridor allows -- `fit_blur` returns the largest blur whose
    # curve stays within `cap` of the sampled path and clear of the obstacles, so the
    # rounding can never cut a corner into a pillar.
    if smooth > 0.0:
        cap = float(params.get("blur_cap", 0.5)) * (
            2.0 * robot.shape.bounding_radius + clearance)
        blur = flat.fit_blur(robot.shape, env._obstacles, path, path, smooth, cap)
        smooth = smooth if blur is None else blur
    if cut is None:
        got = flat.trajectory(robot, state, path, dt, smooth=smooth, slow=slow)
        return got
    parts = _split(path, cut)
    if parts is None:
        return None
    head = flat.trajectory(robot, state, parts[0], dt, smooth=smooth, slow=slow)
    if head is None:
        return None
    tail = flat.trajectory(robot, head[0][-1], parts[1], dt, smooth=smooth, slow=slow)
    if tail is None:
        return None
    hold = np.repeat(head[0][-1][None, :], int(wait), axis=0)
    return (np.vstack([head[0], hold, tail[0]]),
            np.vstack([head[1], np.zeros((int(wait), 2)), tail[1]]))


def _variants(env, i, path, params, rng, clearance=0.05):
    """The candidate motions a path offers: as fast as it goes, slower, and with a wait."""
    out = []
    for slow in (1.0, float(params.get("slow", 1.6))):
        got = _drive(env, i, path, params, clearance, slow=slow)
        if got is not None:
            out.append(got)
    waits = [int(w) for w in params.get("waits", [30, 80])]
    for w in waits:
        cut = float(rng.uniform(0.25, 0.7))
        got = _drive(env, i, path, params, clearance, cut=cut, wait=w)
        if got is not None:
            out.append(got)
    return out


def _sample(env, i, clearance, rng, params, blocks=()):
    """A fresh guide for robot i, pushed away from `blocks` when that is still possible."""
    s0 = np.asarray(env._states[i], float)
    g = np.asarray(env._goals[i], float)
    radius = env.robots[i].shape.bounding_radius + clearance
    for obstacles in ([*env._obstacles, *blocks], list(env._obstacles)):
        path = geometric_rrt.plan_path(
            s0[:2], g[:2], obstacles, env._world_size, radius=radius,
            max_iters=int(params.get("guide_rrt_iters", 5000)),
            step=float(params.get("guide_rrt_step", 0.6)),
            goal_bias=float(params.get("guide_rrt_goal_bias", 0.1)),
            shortcut=bool(params.get("guide_rrt_shortcut", True)), rng=rng)
        if path is not None:
            return np.asarray(path, float)
        # A block can seal the only corridor. Falling back to the real obstacles keeps the
        # robot planning; the refinement gets its diversity from the sampler's own seed.
    return np.vstack([s0[:2], g[:2]])


def _block(point, radius):
    """A disc the sampler must route around. Same duck type as an env obstacle."""
    return SimpleNamespace(shape=CircleShape(radius=float(radius)),
                           pose=(float(point[0]), float(point[1]), 0.0))


# ── pairwise conflict test ──────────────────────────────────────────────────────────

def _hits(env, i, j, ta, tb, clearance):
    """Do two candidates touch when driven from the same instant? Returns the point or None.

    Both are held at their last state once they finish, which is what the executed plan
    does, so the comparison runs to the longer horizon. Three bands, as in
    `schedule.forbidden`: outside the sum of bounding radii is provably clear, inside the
    sum of inscribed radii is provably touching, and only the annulus pays for an exact
    box-to-box distance.
    """
    La, Lb = len(ta), len(tb)
    T = max(La, Lb)
    ia = np.clip(np.arange(T), 0, La - 1)
    ib = np.clip(np.arange(T), 0, Lb - 1)
    pa, pb = ta[ia], tb[ib]
    d = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
    ri = env.robots[i].shape.bounding_radius + env.robots[j].shape.bounding_radius
    qi = inscribed_radius(env.robots[i].shape) + inscribed_radius(env.robots[j].shape)
    near = np.flatnonzero(d < ri + clearance)
    if len(near) == 0:
        return None
    for k in near:
        if d[k] < qi + clearance:
            return 0.5 * (pa[k][:2] + pb[k][:2])
        gap = shape_distance(env.robots[i].shape,
                             (float(pa[k][0]), float(pa[k][1]), float(pa[k][2])),
                             env.robots[j].shape,
                             (float(pb[k][0]), float(pb[k][1]), float(pb[k][2])))
        if gap < clearance:
            return 0.5 * (pa[k][:2] + pb[k][:2])
    return None


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

    paths = [[_sample(env, i, clearance, rng, params)] for i in range(n)]
    cand = [_variants(env, i, paths[i][0], params, rng, clearance)
            for i in range(n)]
    if any(not c for c in cand):
        if isinstance(params, dict):
            params.setdefault("_reject", {})["undrivable_robot"] = 1
        return None

    touch: dict = {}          # (i, a, j, b) -> contested point; the refutations found so far
    info = {"rounds": 0, "refutations": 0, "resampled": 0, "checks": 0}

    def _snap(label, pick=None, spots=()):
        """One stage in the shape `scripts/karc_trace_gif.py` draws.

        The sampled candidate paths are the dim context, what the solver picked is what
        moves against them, and the contested points the core named are the markers -- so
        a trace of a failed round shows exactly the places the next round is sampling away
        from.
        """
        if trace is None:
            return
        static = [np.asarray(paths[i][-1], float)[:, :2] for i in range(n)]
        anim = ([np.asarray(cand[i][pick[i]][0], float)[:, :3] for i in range(n)]
                if pick is not None else [])
        trace.append({"label": label, "static": static, "anim": anim,
                      "markers": [(float(p[0]), float(p[1])) for p in spots],
                      "waypoints": []})

    def _search(cap, tag=""):
        """The lazy loop under a horizon cap. Returns (plan | None, verdict, blame).

        `blame` is read off the unsat core: the robots whose candidate sets are jointly
        impossible, and the points where their candidates met. An UNSAT under a cap is not
        the same failure -- it only says the cap is too tight -- so the caller decides
        whether to refine or to widen.
        """
        x = [[z3.Bool(f"x_{i}_{c}") for c in range(len(cand[i]))] for i in range(n)]
        shown: list = []
        s = z3.Solver()
        for i in range(n):
            fits = [x[i][c] for c in range(len(cand[i]))
                    if cap is None or len(cand[i][c][0]) <= cap]
            if not fits:
                return None, "unsat", {}
            s.add(z3.PbEq([(v, 1) for v in fits], 1))
            for c in range(len(cand[i])):
                if cap is not None and len(cand[i][c][0]) > cap:
                    s.add(z3.Not(x[i][c]))
        # Every refutation ever found is re-asserted, so nothing is re-derived: the
        # pairwise checks are the expensive part and they are never repeated.
        for (i, a_, j, b_) in touch:
            s.assert_and_track(z3.Or(z3.Not(x[i][a_]), z3.Not(x[j][b_])),
                               z3.Bool(f"r_{i}_{a_}_{j}_{b_}"))

        while True:
            if time.perf_counter() > deadline:
                return None, "timeout", {}
            if s.check() != z3.sat:
                break
            m = s.model()
            pick = [next(c for c in range(len(cand[i])) if z3.is_true(m[x[i][c]]))
                    for i in range(n)]
            fresh = []
            for i in range(n):
                for j in range(i + 1, n):
                    key = (i, pick[i], j, pick[j])
                    if key in touch:
                        continue
                    info["checks"] += 1
                    at = _hits(env, i, j, cand[i][pick[i]][0], cand[j][pick[j]][0],
                               clearance)
                    if at is not None:
                        touch[key] = at
                        fresh.append(key)
            if not fresh:
                out = _assemble(env, cand, pick, clearance, info)
                if out is not None:
                    _snap(f"{tag}solved, {out[2]['steps']} steps", pick)
                    return out, "sat", {}
                # The environment's own checker is the authority. If it disagrees with the
                # pairwise test, forbid this whole assignment rather than trusting either.
                s.add(z3.Or([z3.Not(x[i][pick[i]]) for i in range(n)]))
                continue
            info["refutations"] += len(fresh)
            if not shown:
                # The first proposal a round gets refuted on: what the solver thought would
                # work, driven, with the contested points marked. A still of the paths
                # cannot show a timing conflict -- only driving them can.
                shown.append(1)
                _snap(f"{tag}round {info['rounds']}: "
                      f"proposal refuted, {len(fresh)} pairs",
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
            blame.setdefault(i, []).append(at)
            blame.setdefault(j, []).append(at)
        return None, "unsat", blame

    for rnd in range(int(params.get("rounds", 8))):
        info["rounds"] = rnd + 1
        out, verdict, blame = _search(None)
        if verdict == "timeout":
            break
        if out is not None:
            # Any satisfying pick is collision-free but says nothing about how long it
            # takes, and the cheapest way out of a conflict -- wait longer -- is exactly
            # the one that inflates the horizon. Bisect it back down over the SAME
            # candidates and the same refutations, so the search costs solver time only.
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
            info["makespan_bisected"] = 1
            return out

        if not blame:                     # no usable explanation -- resample everyone
            blame = {i: [] for i in range(n)}
        _snap(f"round {rnd + 1}: unsat core blames {len(blame)} robots, resampling",
              spots=[p for spots in blame.values() for p in spots[:1]])
        for i, spots in blame.items():
            blocks = [_block(p, body) for p in spots[:4]]
            path = _sample(env, i, clearance, rng, params, blocks=blocks)
            paths[i].append(path)
            grew = _variants(env, i, path, params, rng, clearance)
            # Waiting longer on a path the robot already has is the other way out of the
            # same core, and it costs no sampling: the core says WHEN as much as where.
            cut = float(rng.uniform(0.2, 0.75))
            more = _drive(env, i, paths[i][0], params, clearance,
                          cut=cut, wait=int(params.get("long_wait", 200)))
            if more is not None:
                grew.append(more)
            cand[i].extend(grew)
            info["resampled"] += 1

    if isinstance(params, dict):
        params.setdefault("_reject", {}).update({"exhausted": 1, **info})
    return None


def _assemble(env, cand, pick, clearance, info):
    """Pad the chosen candidates to one horizon and verify them with the env's checker."""
    n = env._n
    tracks = [cand[i][pick[i]][0] for i in range(n)]
    ctrls = [cand[i][pick[i]][1] for i in range(n)]
    T = max(len(t) for t in tracks)
    tracks = [np.vstack([t, np.repeat(t[-1][None, :], T - len(t), axis=0)])
              if len(t) < T else t for t in tracks]
    ctrls = [np.vstack([c, np.zeros((T - len(c), 2))]) if len(c) < T else c for c in ctrls]
    rep: dict = {}
    gap = schedule.verify(env, tracks, clearance, rep)
    if gap is None:
        return None
    return tracks, ctrls, {**info, "steps": T, "min_surface_gap": round(float(gap), 4),
                           "candidates": sum(len(c) for c in cand)}


class CEGARPlanner(BasePlanner):
    """`approach.method=cegar` -- sampled candidates, lazy SMT, core-guided resampling."""

    method = "cegar"

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        params = dict(self.params)
        clearance = float(self.approach_cfg.get("trajopt", {}).get("clearance", 0.05))
        agents = list(env.possible_agents)
        self._controls = {a: [] for a in agents}
        self.trace = [] if params.get("trace", False) else None
        self.stats = {"method": "cegar", "solved": 0}
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
