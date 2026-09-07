"""K-ARC — Kinodynamic Adaptive Robot Coordination (arXiv:2501.01559).

Reimplemented from the paper against this repo's env; the authors publish no code.
Structure follows the paper's Algorithm 1 and 2:

1. A kinematic reference path per robot, split into ``m_segments`` equal segments.
   Segmenting is what synchronises the robots onto a shared time grid — the
   inter-robot constraints only mean anything when index ``k`` denotes the same
   instant for everyone.
2. Per segment, every robot first solves its own trajectory **uncoordinated**
   (Alg. 1 lines 17-18) with the minimum-time program in :mod:`.trajopt`.
3. Conflicts between those local trajectories are detected, the conflicting robots
   become a subproblem, and the subproblem is handed to a **ladder** of resolution
   strategies (Alg. 2) — the next rung is tried only when the previous one fails.
4. The segment is committed and the next one starts from its terminal states.

Faithfulness notes, all deliberate:

* The conflict predicate is ``||p_i(k) - p_j(k)|| < d_min`` — purely geometric, no
  velocity term, matching K-ARC Eq. 6. Every planner in this family detects
  conflicts this way. That is precisely the property this baseline exists to expose,
  so it is reproduced rather than improved. ``scripts/ics_diag.py`` holds the
  braking-margin alternative.
* K-ARC's objective is ``beta1*||u||^2 + dt`` with ``dt`` free. For *execution* we
  must land on the env's grid, so segments are solved at ``dt_fixed=env.dt``; the
  free-dt mode stays available in :func:`.trajopt.solve_trajectory` for
  planning-quality comparisons.
* The paper publishes neither ``m``, ``d_min``, the timestep, nor the robot
  dimensions. Every one of those is a config knob here, defaulted from our own
  geometry, and none of it should be compared against their published runtimes.

Everything is configured from ``conf/approach/planning.yaml`` under ``approach.karc``
(coordination) and ``approach.trajopt`` (the solver). ``self.stats`` records what the
paper reports — conflicts found, resolution rounds, solver calls and wall time.
"""
from __future__ import annotations

import time

import numpy as np

from src.approach.planning import geometric_rrt, krrt
from src.approach.planning.base import BasePlanner
from src.approach.planning.trajopt import solve_group, solve_trajectory
from src.shaping.dijkstra_potential import DijkstraPotential


class KARCPlanner(BasePlanner):
    method = "karc"

    # ── planning ──────────────────────────────────────────────────────────────

    def reset(self, env) -> None:
        t0 = time.perf_counter()
        t_cfg = self.approach_cfg.get("trajopt", {})
        k_cfg = self.params or {}

        m = max(1, int(k_cfg.get("m_segments", 4)))
        ladder = list(k_cfg.get("ladder", ["prioritized"]))
        max_rounds = int(k_cfg.get("max_rounds", 3))
        adapt_max = int(k_cfg.get("adapt_max", 1))
        d_min = k_cfg.get("d_min", None)
        clearance = float(t_cfg.get("clearance", 0.05))
        on_unsolved = str(k_cfg.get("on_unsolved", "return_empty"))
        budget = k_cfg.get("timeout", 600.0)
        # A plan is only a plan if it arrives in time. K-ARC's experimental setup gives every
        # method 600 s per instance, so a run that keeps solving past it has not produced a
        # slow success -- it has produced a failure that nobody stopped. Without this the
        # hierarchy is unbounded: rounds x subproblems x rungs, and the composite RRT alone
        # can spend minutes on one subproblem.
        self._deadline = None if not budget else t0 + float(budget)

        self.stats = {
            "conflicts": 0, "rounds": 0, "subproblems": 0,
            "solver_calls": 0, "unsolved_segments": 0, "braked_segments": 0,
            "joint_solves": 0,
            "decoupled_rrt_solves": 0,
            "composite_rrt_solves": 0,
            "merges": 0,
            "adaptations": 0,
            "timed_out": 0,
            "plan_failed": 0,
            "initial_path_fallbacks": 0,
            "subproblem_sizes": [],
            "rungs": {},
        }

        radii = [float(r.shape.bounding_radius) for r in env.robots]
        agents = list(env.possible_agents)
        # Alg. 1 lines 2-5, then the horizon: both sized from the reference paths, since with
        # obstacles the journey and the chord are different lengths.
        milestones, ref_paths = self._milestones(env, m, clearance)
        total_h = self._total_horizon(env, t_cfg, ref_paths)

        self._controls = {a: [] for a in agents}
        self._solved = {a: True for a in agents}
        state = [env._states[i].copy() for i in range(env._n)]

        # Every intermediate stage of Alg. 1/2 is computed below and then overwritten.
        # With trace on they are kept, so the planning PROCESS can be drawn rather than
        # only its outcome: reference paths, the uncoordinated solve, the conflicts it
        # produced, and what each ladder rung did about them.
        self.trace = [] if k_cfg.get("trace", False) else None
        self._committed = [np.asarray(state[i][:3], float).reshape(1, 3)
                           for i in range(env._n)]
        # Dotted circles at every segment boundary, on every stage.
        self._waypoints = [np.asarray(ms, float)[:2]
                           for chain in milestones for ms in chain]
        # The reference is a PATH, not a trajectory: it has no dynamics and no timing. But
        # walking the robots along it at a common arclength fraction is exactly the
        # uncoordinated motion K-ARC starts from, and the collisions it produces are the
        # reason the rest of the algorithm exists -- so it is driven, not drawn.
        self._snap("kinematic reference paths (Alg. 1 line 3) -- uncoordinated",
                   [], anim=[self._walk(r) for r in ref_paths],
                   static=[np.zeros((0, 2)) for _ in agents])

        # Checkpoints, one per committed window, so AdaptSubProblem can re-open the
        # previous one. A window is normally a segment; after an adaptation it spans
        # several, and the checkpoints it consumed are popped with it.
        prev_starts: list[dict] = []

        conflicts: list = []   # survives a timeout before the first segment is planned

        # Steps to bring the fastest robot from v_max to rest, for the timeout fallback.
        brake_h = max(2, int(max(r.v_max / max(r.a_max, 1e-6) for r in env.robots)
                             / env.dt) + 2)

        for j in range(m):
            if self._over_budget():
                # Out of time with segments left. Every robot brakes to rest from wherever
                # it stands: `act` pads an exhausted control sequence with ZERO acceleration,
                # which for a second-order robot means coasting at its current velocity into
                # whatever is ahead. A timeout must fail safely, not fail moving.
                for i, a in enumerate(agents):
                    self._solved[a] = False
                    us, state[i], braked = self._brake(env, i, state[i], brake_h)
                    self._controls[a].extend(us)
                    if self.trace is not None:
                        self._committed[i] = np.vstack([self._committed[i], braked[:, :3]])
                self.stats["unsolved_segments"] += m - j
                if on_unsolved == "return_empty":
                    self._abandon(agents, env)
                break

            goals = [milestones[i][j] for i in range(env._n)]
            last = (j == m - 1)   # only the final milestone requires a full stop
            start_ck = self._checkpoint(agents, state)

            # Alg. 2's outer `while P' == ∅`: run the whole solver hierarchy, and only if
            # ALL of it fails widen the subproblem and run it again. K-ARC §III-C states the
            # widening exactly: "we adapt the subproblem by setting the start query to the
            # robot's previous segment start and the goal query to its next segment goal ...
            # as opposed to in ARC where the queries are obtained by small incremental
            # expansions". Re-opening committed motion is the point -- a segment can be
            # unsolvable purely because the one before it arrived badly placed, and no
            # amount of re-solving inside its own window can fix that.
            adapt = 0
            while True:
                # Sized from the window's own geometry, so rolling the start back widens
                # the budget by itself -- the leg is longer, the bang-bang time is longer.
                # Alg. 1 line 17 hands the optimizer this window's slice of the kinematic
                # path. After an adaptation the window reaches back `adapt` segments, so
                # the guide does too -- and the horizon is measured along it.
                lo, hi = max(0, j - adapt) / m, (j + 1) / m
                guides = [self._guide(ref_paths[i], state[i], lo, hi)
                          for i in range(env._n)]
                seg_h = self._segment_horizon(env, t_cfg, state, goals, total_h, m, guides)
                segs, ctrls, oks, conflicts, rounds = self._plan_segment(
                    env, state, goals, seg_h, last, t_cfg, radii, d_min, clearance,
                    ladder, max_rounds, j, m, adapt, guides,
                )
                self.stats["rounds"] += rounds
                if ((not conflicts and all(oks)) or adapt >= adapt_max
                        or not prev_starts or self._over_budget()):
                    break
                adapt += 1
                self.stats["adaptations"] += 1
                start_ck = prev_starts.pop()
                state = self._restore(agents, start_ck)

            prev_starts.append(start_ck)
            unsolved = bool(conflicts) or not all(oks)
            if unsolved:
                self.stats["unsolved_segments"] += 1
            if unsolved and on_unsolved == "return_empty":
                self._abandon(agents, env)
                break

            # Commit the segment and advance. An UNSOLVED segment is never
            # committed: solve_trajectory returns IPOPT's last iterate on failure,
            # which can violate every constraint, and executing it produces exactly
            # the collisions the planner is supposed to prevent. Brake to rest
            # instead and report the failure through `_solved`.
            executed = []
            for i, a in enumerate(agents):
                if oks[i]:
                    self._controls[a].extend(np.atleast_2d(ctrls[i]))
                    state[i] = np.asarray(segs[i][-1], dtype=np.float64)
                    executed.append(np.asarray(segs[i], dtype=np.float64))
                else:
                    self._solved[a] = False
                    self.stats["braked_segments"] += 1
                    us, state[i], braked = self._brake(
                        env, i, state[i], len(np.atleast_2d(ctrls[i])))
                    self._controls[a].extend(us)
                    executed.append(braked)
            if self.trace is not None:
                self._committed = [
                    np.vstack([self._committed[i], executed[i][:, :3]])
                    for i in range(env._n)
                ]

        # Drive the whole committed plan end to end: the payoff shot.
        self._snap("final plan", [], static=[np.zeros((0, 2)) for _ in agents],
                   anim=self._committed if self.trace is not None else [])
        sizes = self.stats.pop("subproblem_sizes")
        # |R'| is the whole point of the pair-vs-merged question: a "local" subproblem that
        # contains every robot is a coupled solve. Max and mean say which one ran.
        self.stats["subproblem_max"] = max(sizes) if sizes else 0
        self.stats["subproblem_mean"] = round(sum(sizes) / len(sizes), 2) if sizes else 0.0
        self.stats["conflicts_remaining"] = len(conflicts)
        self.stats["wall_time"] = time.perf_counter() - t0
        self.stats["path_cost"] = sum(len(v) for v in self._controls.values()) * env.dt
        self._plan = self._controls

    def act(self, obs_dict: dict, env) -> dict:
        out = {}
        for i, agent in enumerate(env.agents):
            seq = self._controls.get(agent, [])
            out[agent] = seq.pop(0) if seq else np.zeros(env.robots[i].action_dim)
        return out

    @staticmethod
    def _walk(path: np.ndarray, steps: int = 160) -> np.ndarray:
        """A geometric path -> (steps, 3) poses, sampled at equal fractions of arclength.

        Every robot gets the same number of samples, so index k is the same fraction of
        the way along for all of them. That is the synchronisation K-ARC's segmentation
        imposes, applied to the reference itself, and it is what makes the resulting
        overlaps meaningful rather than an artefact of unequal path lengths. Heading comes
        from the path tangent -- the reference is kinematic, so there is no other source.
        """
        pts = np.asarray(path, dtype=float)[:, :2]
        if len(pts) < 2:
            return np.zeros((0, 3))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] <= 0:
            return np.zeros((0, 3))
        want = np.linspace(0.0, arc[-1], steps)
        xy = np.column_stack([np.interp(want, arc, pts[:, 0]),
                              np.interp(want, arc, pts[:, 1])])
        d = np.gradient(xy, axis=0)
        theta = np.arctan2(d[:, 1], d[:, 0])
        return np.column_stack([xy, theta])

    def _snap(self, label, segs, conflicts=(), static=None, anim=None) -> None:
        """Keep one stage of the plan for rendering. No-op unless karc.trace is set.

        Two path sets per stage, because they are drawn differently. ``static`` is context
        that is already settled -- the committed segments, or the reference path -- and is
        rendered as a dim trail. ``anim`` is the trajectory under consideration, with
        headings, and the robots are DRIVEN along it: that is what makes a candidate
        trajectory legible as motion rather than as a line on a picture.

        A conflict (i, j, k) is the first index where two trajectories violate separation;
        the marker goes at the pair's midpoint.
        """
        if self.trace is None:
            return
        segs = [np.asarray(sg, dtype=float) for sg in segs]
        self.trace.append({
            "label": label,
            "static": [np.asarray(p, float)[:, :2] for p in
                       (static if static is not None else self._committed)],
            "anim": [np.asarray(p, float)[:, :3] for p in
                     (anim if anim is not None else segs)],
            "markers": [0.5 * (segs[i][k][:2] + segs[j][k][:2]) for i, j, k in conflicts],
            "waypoints": list(self._waypoints),
        })

    def _abandon(self, agents, env) -> None:
        """Alg. 1 lines 26-27: `return ∅` -- no plan at all, not a partial one.

        K-ARC fails the WHOLE instance when a segment's subproblem is unresolved; it never
        emits a plan with a conflict left in it. Executing the segments that did solve, as
        `on_unsolved: brake` does, measures something the algorithm would not have returned.
        No plan means no motion, so every control sequence is discarded and the robots stand
        where they started.
        """
        self.stats["plan_failed"] = 1
        for i, a in enumerate(agents):
            self._controls[a] = []
            self._solved[a] = False
        if self.trace is not None:
            self._committed = [np.asarray(env._states[i], float)[None, :3].copy()
                               for i in range(env._n)]

    def _over_budget(self) -> bool:
        """True once the planning time budget is spent.

        Checked at loop boundaries only -- between rungs, rounds and segments -- so the
        overshoot is bounded by ONE solver call rather than by the ladder. Both solvers cap
        their own work (IPOPT `max_iters`, RRT `rrt_iters`), so that call terminates.
        """
        if self._deadline is None:
            return False
        if time.perf_counter() < self._deadline:
            return False
        self.stats["timed_out"] = 1
        return True

    @staticmethod
    def _brake(env, i, state, n_steps):
        """Decelerate to rest and hold — the safe fallback for an unsolved segment.

        Returns the controls, the final state, and the states passed through. The last is
        for the trace: an unsolved segment is NOT executed, so committing its trajectory
        would animate a plan the robot never follows -- and the unsolved segments are
        exactly the ones worth watching.
        """
        r = env.robots[i]
        st = np.asarray(state, dtype=np.float64).copy()
        us, path = [], [st.copy()]
        for _ in range(max(0, int(n_steps))):
            a = float(np.clip(-st[3] / env.dt, r.a_min, r.a_max))
            al = float(np.clip(-st[4] / env.dt, r.alpha_min, r.alpha_max))
            u = np.array([a, al], dtype=np.float64)
            us.append(u)
            st = r.step(st, u, env.dt)
            path.append(st.copy())
        return us, st, np.asarray(path, dtype=np.float64)

    # ── pieces ────────────────────────────────────────────────────────────────

    @staticmethod
    def _segment_horizon(env, t_cfg, state, goals, total_h, m, guides=None) -> int:
        """Steps allotted to one segment: the slowest robot's bang-bang time over its
        own leg. Capped by the whole-plan budget so a pathological leg cannot eat it.

        The leg is measured along the KINEMATIC GUIDE, not start-to-goal. With obstacles the
        two diverge without limit -- a robot rounding a pillar covers far more ground than
        the chord -- and a horizon sized from the chord makes the segment infeasible on time
        alone. The optimizer then reports failure for a segment that has a perfectly good
        solution, and the ladder burns every rung rediscovering that. In an empty world the
        guide IS the chord and nothing changes.
        """
        h = t_cfg.get("horizon", None)
        if h is not None:
            return max(2, int(h) // m)
        slack = float(t_cfg.get("slack", 1.5))
        from src.shaping.braking_potential import bangbang_time
        legs = [
            KARCPlanner._path_len(guides[i]) if guides is not None
            else float(np.linalg.norm(np.asarray(goals[i])[:2] - np.asarray(state[i])[:2]))
            for i in range(env._n)
        ]
        worst = max(
            bangbang_time(legs[i], 0.0, env.robots[i].v_max, env.robots[i].a_max)
            for i in range(env._n)
        )
        return int(np.clip(np.ceil(slack * worst / env.dt), 2, total_h))

    @staticmethod
    def _path_len(path: np.ndarray) -> float:
        pts = np.asarray(path, dtype=float)[:, :2]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0

    @staticmethod
    def _total_horizon(env, t_cfg, refs=None) -> int:
        """Time budget in env steps. See OptimizationPlanner._auto_horizon.

        Measured along the reference paths when they are available, for the same reason
        `_segment_horizon` is: with obstacles the chord understates the journey.
        """
        h = t_cfg.get("horizon", None)
        if h is not None:
            return int(h)
        slack = float(t_cfg.get("slack", 1.5))
        from src.shaping.braking_potential import bangbang_time
        legs = [
            KARCPlanner._path_len(refs[i]) if refs is not None
            else float(np.linalg.norm(env._goals[i][:2] - env._states[i][:2]))
            for i in range(env._n)
        ]
        worst = max(
            bangbang_time(legs[i], 0.0, env.robots[i].v_max, env.robots[i].a_max)
            for i in range(env._n)
        )
        return min(env.max_steps, max(10, int(np.ceil(slack * worst / env.dt))))

    def _milestones(self, env, m: int, clearance: float):
        """Milestones spaced evenly along an obstacle-aware reference path.

        K-ARC seeds its optimiser from a *kinematic planner*, and that matters more
        than it looks: with a straight-line seed, a milestone on the far side of an
        obstacle forces the segment to detour around it and return to the line inside
        one segment's time budget, which is often infeasible -- and an infeasible
        segment gets committed and executed as a collision.

        The reference comes from the clearance-inflated Dijkstra cost-to-go field
        already used by the shaping potentials (``src/shaping/dijkstra_potential.py``),
        walked greedily downhill from start to goal. Milestones are then placed at
        equal arclength along it, and finally pulled apart where two of them coincide
        (see ``_separate``).
        """
        grid = DijkstraPotential(
            env.cfg.env.obstacles, env._world_size, v_max=1.0,
            clearance=clearance + max(r.shape.bounding_radius for r in env.robots),
        )
        self._grid = grid
        source = str(self.params.get("initial_paths", "rrt"))
        rng = np.random.default_rng(int(self.params.get("rrt_seed", 0)))
        out, refs = [], []
        for i in range(env._n):
            s0 = np.asarray(env._states[i], dtype=np.float64)
            g = np.asarray(env._goals[i], dtype=np.float64)
            path = None
            if source == "rrt":
                path = geometric_rrt.plan_path(
                    s0[:2], g[:2], env._obstacles, env._world_size,
                    radius=env.robots[i].shape.bounding_radius + clearance,
                    max_iters=int(self.params.get("initial_rrt_iters", 5000)),
                    step=float(self.params.get("initial_rrt_step", 0.6)),
                    goal_bias=float(self.params.get("initial_rrt_goal_bias", 0.1)),
                    shortcut=bool(self.params.get("initial_rrt_shortcut", False)),
                    rng=rng,
                )
                if path is None:
                    # A guide is required, and a straight line through a pillar is worse
                    # than a grid path. Falling back is reported, never silent.
                    self.stats["initial_path_fallbacks"] += 1
            if path is None:
                path = KARCPlanner._descend(grid, s0[:2], g[:2])
            refs.append(np.asarray(path, dtype=np.float64))
            out.append(KARCPlanner._resample(path, g, m))
        radii = [float(r.shape.bounding_radius) for r in env.robots]
        return KARCPlanner._separate(out, radii, clearance, env._world_size), refs

    @staticmethod
    def _separate(ms, radii, clearance, world_size):
        """Pull coinciding intermediate milestones apart.

        Equal-arclength milestones are computed per robot, independently. In a
        symmetric head-on swap both robots descend the SAME reference path, so their
        k-th milestones land on the same point — and since a milestone is the terminal
        constraint of segment k while the robots must stay ``r_i + r_j + clearance``
        apart at every index, that segment is infeasible *by construction*. No amount
        of re-solving fixes it: not prioritised, not relaxed, not joint. The milestones
        themselves have to move.

        Offset is LATERAL — perpendicular to the robot's own direction of travel, in
        opposite directions for the pair. That axis is not a detail: pushing two
        head-on robots apart *along* their shared path just re-orders their milestones
        and still requires them to pass through each other on the same line. Only a
        sideways offset lets them go around. The final milestone is the true goal and
        is never moved.
        """
        m = len(ms[0])
        for k in range(m - 1):                      # goal (k = m-1) is fixed
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    pi, pj = ms[i][k][:2], ms[j][k][:2]
                    need = radii[i] + radii[j] + clearance
                    delta = pj - pi
                    if float(np.linalg.norm(delta)) >= need:
                        continue
                    # Unit normal to robot i's travel direction (prev -> next).
                    prev = ms[i][k - 1][:2] if k else pi
                    t = ms[i][k + 1][:2] - prev
                    nt = float(np.linalg.norm(t))
                    u = np.array([-t[1], t[0]]) / nt if nt > 1e-9 else np.array([0.0, 1.0])
                    lat = float(delta @ u)
                    if lat < 0.0:
                        u, lat = -u, -lat           # keep whatever lateral bias exists
                    # Separation splits into a longitudinal part the offset cannot change
                    # and a lateral part it can, so only the lateral shortfall is closed.
                    par = float(np.linalg.norm(delta - lat * u))
                    push = 0.5 * (np.sqrt(max(need**2 - par**2, 0.0)) - lat) + 1e-3
                    if push <= 0.0:
                        continue
                    lo = max(radii[i], radii[j])
                    hi = world_size - lo
                    ms[i][k][:2] = np.clip(pi - push * u, lo, hi)
                    ms[j][k][:2] = np.clip(pj + push * u, lo, hi)
        return ms

    @staticmethod
    def _descend(grid: DijkstraPotential, start, goal, max_steps: int = 4000):
        """Greedy descent on the cost-to-go field: the obstacle-aware reference."""
        field = grid._dist_field(np.asarray(goal, dtype=np.float64))
        i, j = grid._nearest_free(*grid._to_cell(float(start[0]), float(start[1])))
        pts = [np.array([start[0], start[1]], dtype=np.float64)]
        n = grid.n
        for _ in range(max_steps):
            if not np.isfinite(field[i, j]) or field[i, j] <= 0.0:
                break
            best, bi, bj = field[i, j], i, j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and field[ni, nj] < best:
                        best, bi, bj = field[ni, nj], ni, nj
            if (bi, bj) == (i, j):
                break                      # local minimum: fall back to the goal
            i, j = bi, bj
            pts.append(np.array([(i + 0.5) * grid.cell, (j + 0.5) * grid.cell]))
        pts.append(np.asarray(goal, dtype=np.float64)[:2])
        return np.asarray(pts)

    @staticmethod
    def _guide(path: np.ndarray, start, lo: float, hi: float) -> np.ndarray:
        """The piece of a reference path between two arclength fractions, from `start`.

        This is Alg. 1 line 17's ``Pri[j]`` -- the kinematic segment handed to the optimizer
        as its reference. The robot is rarely standing exactly on the reference when the
        segment begins (the previous segment ended wherever the dynamics allowed, which
        §III-A calls out as the whole reason the construction is sequential), so the guide
        starts from where the robot actually is and joins the path from there.
        """
        pts = np.asarray(path, dtype=float)[:, :2]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-9:
            return np.asarray([start[:2], pts[-1]], dtype=float)
        a, b = total * lo, total * hi
        # Interpolate AT the fractions rather than keeping whichever vertices fall between
        # them. A shortcut-smoothed path can be two vertices long, and vertex-membership
        # slicing then returns a degenerate guide -- which silently sizes the segment horizon
        # to nothing, since the horizon is measured along the guide.
        ends = [np.array([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])])
                for t in (a, b)]
        mid = [q for q, c in zip(pts, cum) if a < c < b]
        out = [np.asarray(start, dtype=float)[:2], ends[0], *mid, ends[1]]
        # Drop points that repeat: a zero-length step contributes no arclength and no heading.
        keep = [out[0]]
        for q in out[1:]:
            if float(np.linalg.norm(q - keep[-1])) > 1e-9:
                keep.append(q)
        return np.asarray(keep if len(keep) >= 2 else [out[0], pts[-1]], dtype=float)

    @staticmethod
    def _resample(path: np.ndarray, goal: np.ndarray, m: int) -> list[np.ndarray]:
        """m waypoints at equal arclength; the last is the true goal."""
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        pts = []
        for k in range(1, m + 1):
            p = goal.copy()
            if k < m and total > 1e-9:
                target = total * k / m
                idx = int(np.searchsorted(cum, target))
                idx = min(max(idx, 1), len(path) - 1)
                span = cum[idx] - cum[idx - 1]
                frac = 0.0 if span < 1e-12 else (target - cum[idx - 1]) / span
                p[:2] = path[idx - 1] + frac * (path[idx] - path[idx - 1])
            pts.append(p)
        return pts

    # The env's goal test is a STRICT inequality (multiagent_nav.py: dist < goal_radius),
    # and the objective is minimum time, so the solver parks the terminal state exactly on
    # whatever tolerance it is given -- a plan that "reaches" the goal at exactly
    # goal_radius scores as a failure. Keeping the terminal tolerance strictly inside the
    # test is the difference between reporting the plan we made and reporting float noise.
    GOAL_INSET = 0.9

    def _terminal_tol(self, env, t_cfg, goal_scale, terminal_stop) -> float:
        tol = float(t_cfg.get("goal_tol", env.goal_radius)) * goal_scale
        # Intermediate milestones are never tested by the env, so only the final one needs
        # the inset -- and relaxing it there would mean a robot that stops short "succeeds".
        return min(tol, self.GOAL_INSET * env.goal_radius) if terminal_stop else tol

    def _solve(self, env, i, start, goal, seg_h, t_cfg, avoid, goal_scale=1.0,
               terminal_stop=True, guide=None):
        avoid_trajs = tuple(a[0] for a in avoid)
        avoid_radii = tuple(a[1] for a in avoid)
        self.stats["solver_calls"] += 1
        X, _U, _dt, ok = solve_trajectory(
            env.robots[i], start, goal, env._obstacles, env._world_size,
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=env.dt,
            avoid=avoid_trajs, avoid_radii=avoid_radii,
            goal_tol=self._terminal_tol(env, t_cfg, goal_scale, terminal_stop),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=terminal_stop,
            max_iter=int(t_cfg.get("max_iters", 500)),
            guides=None if guide is None else [guide],
            obstacle_margin=t_cfg.get("obstacle_margin", None),
        )
        return X, _U, ok

    @staticmethod
    def _find_conflicts(segs, radii, d_min, clearance):
        """K-ARC Eq. 6: geometric separation at matching time indices.

        No velocity term — see the module docstring.
        """
        out = []
        n = len(segs)
        for i in range(n):
            for j in range(i + 1, n):
                thresh = (
                    float(d_min) if d_min is not None
                    else radii[i] + radii[j] + clearance
                )
                horizon = min(len(segs[i]), len(segs[j]))
                for k in range(horizon):
                    d = float(np.linalg.norm(segs[i][k][:2] - segs[j][k][:2]))
                    if d < thresh:
                        out.append((i, j, k))
                        break
        return out

    def _plan_segment(self, env, state, goals, seg_h, last, t_cfg, radii, d_min,
                      clearance, ladder, max_rounds, j, m, adapt, guides):
        """Solve one window: uncoordinated first (Alg. 1 lines 17-18), then the hierarchy."""
        span = f"segment {j + 1}/{m}" + (f" (+{adapt} back)" if adapt else "")
        segs, ctrls, oks = [], [], []
        for i in range(env._n):
            X, U, ok = self._solve(env, i, state[i], goals[i], seg_h, t_cfg, (),
                                   terminal_stop=last, guide=guides[i])
            segs.append(X)
            ctrls.append(U)
            oks.append(ok)

        conflicts = self._find_conflicts(segs, radii, d_min, clearance)
        self._snap(f"{span}: uncoordinated solve "
                   f"({len(conflicts)} conflict{'' if len(conflicts) == 1 else 's'})",
                   segs, conflicts)

        # The hierarchy handles two failure kinds, not one. A segment can be in conflict,
        # but it can also just be INFEASIBLE on its own: segmentation constrains
        # intermediate milestones by position only, so the previous segment is free to
        # arrive pointing the wrong way, and the next one then cannot turn around and reach
        # its milestone in the time it has. Gating on conflicts alone sends those straight
        # to the braking fallback without ever trying a rung.
        rounds = 0
        while (conflicts or not all(oks)) and rounds < max_rounds \
                and not self._over_budget():
            self.stats["conflicts"] += len(conflicts)
            segs, ctrls, oks, conflicts = self._resolve_segment(
                conflicts, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                radii, last, ladder, d_min, clearance, span, guides,
            )
            rounds += 1
        return segs, ctrls, oks, conflicts, rounds

    def _checkpoint(self, agents, state) -> dict:
        """Everything AdaptSubProblem has to be able to undo."""
        return {
            "state": [np.asarray(s, float).copy() for s in state],
            "ctrl_len": {a: len(self._controls[a]) for a in agents},
            "solved": dict(self._solved),
            "committed_len": ([len(c) for c in self._committed]
                              if self.trace is not None else None),
        }

    def _restore(self, agents, ck: dict) -> list:
        """Undo committed motion back to a checkpoint. Effort counters are NOT rolled
        back: those solver calls happened and cost wall time, and reporting otherwise would
        understate what adaptation costs. Outcome counters are, since the segments they
        described are being re-planned."""
        for a in agents:
            del self._controls[a][ck["ctrl_len"][a]:]
        self._solved = dict(ck["solved"])
        if self.trace is not None and ck["committed_len"] is not None:
            self._committed = [c[:n] for c, n in zip(self._committed, ck["committed_len"])]
        return [np.asarray(s, float).copy() for s in ck["state"]]

    def _resolve_segment(self, conflicts, segs, ctrls, oks, env, state, goals, seg_h,
                         t_cfg, radii, last, ladder, d_min, clearance, span, guides):
        """One pass of Alg. 2 over a segment: a SUBPROBLEM PER CONFLICTING PAIR.

        ARC (arXiv:2312.08554 SS IV-B) is explicit that a subproblem is built around one
        conflict -- ``R' = R_i u R_j merges the involved robots`` -- and that R' grows only
        reactively: *"there are instances where resolving one conflict invalidates a prior
        conflict resolution. ARC can identify such occurrences and adapt R' to account for
        all the involved robots."* K-ARC inherits this; it says nothing about changing it.

        The point is locality. Merging every conflicting robot in the segment into one
        problem, as this used to do, makes a 32-robot "local" subproblem out of 16
        independent head-on pairs, and then asks the last robot to thread 31 frozen
        trajectories. That is a coupled solve wearing a subproblem's name, and its cost and
        failure rate are ours, not the algorithm's. ``karc.subproblem=merged`` restores it
        for the ablation.

        A robot whose own segment came back infeasible has no partner to pair with, so it
        forms a singleton subproblem. ARC does not discuss this case -- its subproblems
        exist only for conflicts -- but segmentation here constrains milestones by position
        only, so a segment can be individually infeasible with no conflict at all, and that
        is a failure the hierarchy can repair.
        """
        pairs = sorted({frozenset(c[:2]) for c in conflicts}, key=sorted)
        groups = [set(pr) for pr in pairs]
        if self.params.get("subproblem", "pair") == "merged":
            groups = [set().union(*groups)] if groups else []
        stranded = {i for i, ok in enumerate(oks) if not ok} - set().union(*groups, set())
        groups += [{i} for i in sorted(stranded)]

        settled: list[set] = []
        for group in groups:
            group = set(group)
            while True:
                self.stats["subproblems"] += 1
                self.stats["subproblem_sizes"].append(len(group))
                for rung in ladder:
                    if self._over_budget():
                        return segs, ctrls, oks, self._find_conflicts(
                            segs, radii, d_min, clearance)
                    segs, ctrls, oks = self._resolve(
                        rung, group, segs, ctrls, oks, env, state, goals, seg_h,
                        t_cfg, radii, last, guides,
                    )
                    self.stats["rungs"][rung] = self.stats["rungs"].get(rung, 0) + 1
                    conflicts = self._find_conflicts(segs, radii, d_min, clearance)
                    self._snap(f"{span}: R'={sorted(group)} {rung} -> "
                               f"{len(conflicts)} conflicts remaining", segs, conflicts)
                    if self._clear(group, conflicts, oks):
                        break

                # Did resolving this subproblem invalidate an earlier one? A conflict that
                # straddles the boundary, with a robot on the settled side, means it did.
                conflicts = self._find_conflicts(segs, radii, d_min, clearance)
                spoiled = {r for a, b, _ in conflicts for r in (a, b)
                           if (a in group) != (b in group)
                           and any(r in prev for prev in settled)}
                merged = set(group).union(*[prev for prev in settled if prev & spoiled],
                                          set())
                if merged == group:
                    break
                self.stats["merges"] += 1
                group = merged
            settled = [prev for prev in settled if not (prev & group)] + [group]

        return segs, ctrls, oks, self._find_conflicts(segs, radii, d_min, clearance)

    @staticmethod
    def _clear(group, conflicts, oks) -> bool:
        """This subproblem is done: none of its robots is in a conflict or infeasible."""
        return (all(oks[i] for i in group)
                and not any(a in group or b in group for a, b, _ in conflicts))

    def _group_paths(self, env, involved, state, goals):
        """Kinematic paths for R' from a GROUP planner -- §IV-C's first step, before any
        optimisation happens.

        The paper is explicit about the order: *"For robots {r1, r2, ..., rk}, we first find
        the kinematic paths through a group planner. The paths are then sequentially
        optimized."* Skipping the group planner and re-optimising from the previous seed, as
        this used to do, leaves every robot in the homotopy class its solo path picked -- so
        the only concession the rung can express is slowing down, and it can never route a
        robot the other way round an obstacle. That is the rung doing half its job.

        Prioritised planning on the shared grid: each robot descends the cost-to-go field
        with the earlier robots' paths masked out as obstacles. Cells near a robot's own
        start and goal are never masked -- blocking them would make its own query
        unsolvable rather than route it elsewhere.
        """
        grid = self._grid
        free0 = grid._free
        blocked = free0.copy()
        radii = [float(r.shape.bounding_radius) for r in env.robots]
        paths = {}
        try:
            for i in involved:
                grid._free = blocked
                start, goal = np.asarray(state[i], float), np.asarray(goals[i], float)
                path = self._descend(grid, start[:2], goal[:2])
                if len(path) < 2:                      # masked into a dead end
                    grid._free = free0
                    path = self._descend(grid, start[:2], goal[:2])
                paths[i] = np.asarray(path, dtype=np.float64)
                # Mask this path for whoever comes next.
                r = radii[i] + max(radii) + float(
                    self.approach_cfg.get("trajopt", {}).get("clearance", 0.05))
                cells = int(np.ceil(r / grid.cell))
                keep = (start[:2], goal[:2])
                for pt in paths[i]:
                    if min(float(np.linalg.norm(pt[:2] - k)) for k in keep) < r:
                        continue
                    ci, cj = grid._to_cell(float(pt[0]), float(pt[1]))
                    lo_i, hi_i = max(0, ci - cells), min(grid.n, ci + cells + 1)
                    lo_j, hi_j = max(0, cj - cells), min(grid.n, cj + cells + 1)
                    blocked[lo_i:hi_i, lo_j:hi_j] = False
        finally:
            grid._free = free0
        return paths

    def _resolve(self, rung, involved, segs, ctrls, oks, env, state, goals, seg_h,
                 t_cfg, radii, last=True, guides=None):
        """One rung of the solver hierarchy S, applied to ONE subproblem.

        `involved` is the subproblem's robot set R', chosen by the caller. Its members are
        re-solved in priority order, each avoiding every trajectory outside the subproblem
        plus those already fixed within it.
        """
        involved = sorted(involved)
        if self.params.get("priority", "index") == "distance":
            involved.sort(key=lambda i: float(np.linalg.norm(goals[i][:2] - state[i][:2])))

        relax = float(self.params.get("relax_per_level", 1.0))
        segs = list(segs)
        ctrls = list(ctrls)
        oks = list(oks)
        # Non-involved robots keep their trajectories and must still be avoided.
        avoid = [(segs[i], radii[i]) for i in range(env._n) if i not in involved]

        if rung == "joint":
            return self._solve_joint(
                involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                tuple(avoid), last, guides,
            )
        if rung in ("decoupled_rrt", "composite_rrt"):
            return self._solve_rrt(
                rung, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                last,
            )

        # §IV-C: group planner first, then sequential optimisation against it.
        group_paths = self._group_paths(env, involved, state, goals)
        for level, i in enumerate(involved):
            scale = 1.0 if rung == "prioritized" else relax ** level
            X, U, ok = self._solve(
                env, i, state[i], goals[i], seg_h, t_cfg, tuple(avoid), goal_scale=scale,
                terminal_stop=last, guide=group_paths.get(i),
            )
            segs[i], ctrls[i], oks[i] = X, U, ok
            avoid.append((X, radii[i]))
        return segs, ctrls, oks

    def _solve_rrt(self, rung, involved, segs, ctrls, oks, env, state, goals, seg_h,
                   t_cfg, last):
        """K-ARC's sampling rungs: Decoupled and Composite Kinodynamic RRT.

        Why the ladder has them at all (ARC arXiv:2312.08554 SS IV-C): the prioritised
        rungs re-solve one robot at a time inside the SAME homotopy the reference path
        picked, so the only concession a lower-priority robot can make is to slow down or
        stop. When the resolution requires leaving the path -- backing into free space,
        going around the far side of an obstacle -- trajopt cannot find it, because a
        nonlinear program started from an infeasible seed does not change homotopy class.
        Sampling does: it "adds additional configurations that robots can use to move out
        of the way".

        decoupled_rrt  - one tree per robot, in priority order, each avoiding the
                         trajectories already fixed (inside and outside R'). Cheap;
                         inherits the incompleteness of prioritised planning.
        composite_rrt  - ONE tree over the joint state of R'. Complete for the subproblem
                         given enough samples, exponential in |R'|, hence last.

        Both use the same time-gridded planner (`src.approach.planning.krrt`), so their
        output is index-comparable with a trajopt segment and can be committed the same way.
        """
        rng = np.random.default_rng(
            int(self.params.get("rrt_seed", 0)) + 1000 * self.stats["rounds"]
            + len(self.stats["subproblem_sizes"]))
        kw = dict(
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            terminal_stop=last,
            max_iters=int(self.params.get("rrt_iters", 3000)),
            n_controls=int(self.params.get("rrt_controls", 10)),
            steps=int(self.params.get("rrt_steps", 5)),
            goal_bias=float(self.params.get("rrt_goal_bias", 0.15)),
            rng=rng,
        )
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        outside = [(segs[i], env.robots[i].shape)
                   for i in range(env._n) if i not in involved]

        if rung == "composite_rrt":
            self.stats["composite_rrt_solves"] += 1
            X, U, ok = krrt.plan(
                [env.robots[i] for i in involved], [state[i] for i in involved],
                [goals[i] for i in involved], env._obstacles, env._world_size,
                env.dt, seg_h, others=tuple(outside), **kw)
            # One tree, one verdict -- as with the joint program.
            for slot, i in enumerate(involved):
                segs[i], ctrls[i], oks[i] = X[:, slot], U[:, slot], ok
            return segs, ctrls, oks

        self.stats["decoupled_rrt_solves"] += 1
        avoid = list(outside)
        for i in involved:
            X, U, ok = krrt.plan(
                [env.robots[i]], [state[i]], [goals[i]], env._obstacles,
                env._world_size, env.dt, seg_h, others=tuple(avoid), **kw)
            segs[i], ctrls[i], oks[i] = X[:, 0], U[:, 0], ok
            avoid.append((segs[i], env.robots[i].shape))
        return segs, ctrls, oks

    def _solve_joint(self, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                     avoid, last, guides=None):
        """Re-solve the conflicting robots TOGETHER in one nonlinear program.

        NOT a K-ARC rung -- its hierarchy goes prioritized -> decoupled RRT -> composite RRT
        (`_solve_rrt`), and AdaptSubProblem is the window-widening loop in `reset`, not this.
        This is the optimisation-side analogue of composite_rrt, kept because it is cheaper
        than sampling when the resolution stays in one homotopy class.

        The prioritised rungs fix one robot's trajectory and ask the next to work around
        it. That cannot solve a symmetric head-on swap in a corridor — whichever robot is
        ordered second has nowhere to yield to, and no permutation of single-robot solves
        changes that. Here every robot in the subproblem is a free variable in one
        program, so the solver can move both aside at once.

        Costlier than the prioritised rungs (the program grows with the group), which is
        why it belongs at the END of the ladder: only the conflicts that ordering cannot
        fix pay for it.
        """
        self.stats["joint_solves"] += 1
        self.stats["solver_calls"] += 1
        Xs, Us, _dt, ok = solve_group(
            [env.robots[i] for i in involved],
            [state[i] for i in involved],
            [goals[i] for i in involved],
            env._obstacles, env._world_size,
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=env.dt,
            avoid=tuple(a[0] for a in avoid), avoid_radii=tuple(a[1] for a in avoid),
            goal_tol=self._terminal_tol(env, t_cfg, 1.0, last),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=last,
            max_iter=int(t_cfg.get("max_iters", 500)),
            guides=None if guides is None else [guides[i] for i in involved],
            obstacle_margin=t_cfg.get("obstacle_margin", None),
        )
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        # One program, one verdict: the group is feasible together or not at all.
        for slot, i in enumerate(involved):
            segs[i], ctrls[i], oks[i] = Xs[slot], Us[slot], ok
        return segs, ctrls, oks
