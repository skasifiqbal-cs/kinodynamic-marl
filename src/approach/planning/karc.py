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
        d_min = k_cfg.get("d_min", None)
        clearance = float(t_cfg.get("clearance", 0.05))

        total_h = self._total_horizon(env, t_cfg)

        radii = [float(r.shape.bounding_radius) for r in env.robots]
        agents = list(env.possible_agents)
        milestones = self._milestones(env, m, clearance)

        self.stats = {
            "conflicts": 0, "rounds": 0, "subproblems": 0,
            "solver_calls": 0, "unsolved_segments": 0, "braked_segments": 0,
            "joint_solves": 0,
            "rungs": {},
        }
        self._controls = {a: [] for a in agents}
        self._solved = {a: True for a in agents}
        state = [env._states[i].copy() for i in range(env._n)]

        for j in range(m):
            goals = [milestones[i][j] for i in range(env._n)]
            last = (j == m - 1)   # only the final milestone requires a full stop
            # Per-segment time budget, sized from THIS segment's own geometry rather
            # than as total_h/m. dt is fixed here (robots must share a time grid), so
            # the segment lasts exactly seg_h*dt -- the horizon is a deadline, not a
            # cap, and a robot that cannot reach its milestone in it fails outright.
            # Segments are not equally hard: `_separate` lengthens exactly the ones
            # where robots have to go around each other, and a uniform split hands
            # those the same budget as a straight run.
            seg_h = self._segment_horizon(env, t_cfg, state, goals, total_h, m)

            # Alg. 1 lines 17-18: every robot solves its own segment, uncoordinated.
            segs, ctrls, oks = [], [], []
            for i in range(env._n):
                X, U, ok = self._solve(env, i, state[i], goals[i], seg_h, t_cfg, (),
                                       terminal_stop=last)
                segs.append(X)
                ctrls.append(U)
                oks.append(ok)

            conflicts = self._find_conflicts(segs, radii, d_min, clearance)
            # The ladder handles two failure kinds, not one. A segment can be in
            # conflict, but it can also just be INFEASIBLE on its own: segmentation
            # constrains intermediate milestones by position only, so the previous
            # segment is free to arrive pointing the wrong way, and the next one then
            # cannot turn around and reach its milestone in the time it has. Gating the
            # loop on conflicts alone sends those straight to the braking fallback
            # without ever trying a rung.
            rounds = 0
            while (conflicts or not all(oks)) and rounds < max_rounds:
                self.stats["conflicts"] += len(conflicts)
                self.stats["subproblems"] += 1
                for rung in ladder:
                    segs, ctrls, oks = self._resolve(
                        rung, conflicts, segs, ctrls, oks, env, state, goals, seg_h,
                        t_cfg, radii, last
                    )
                    self.stats["rungs"][rung] = self.stats["rungs"].get(rung, 0) + 1
                    conflicts = self._find_conflicts(segs, radii, d_min, clearance)
                    if not conflicts and all(oks):
                        break
                rounds += 1
            self.stats["rounds"] += rounds
            if conflicts or not all(oks):
                self.stats["unsolved_segments"] += 1

            # Commit the segment and advance. An UNSOLVED segment is never
            # committed: solve_trajectory returns IPOPT's last iterate on failure,
            # which can violate every constraint, and executing it produces exactly
            # the collisions the planner is supposed to prevent. Brake to rest
            # instead and report the failure through `_solved`.
            for i, a in enumerate(agents):
                if oks[i]:
                    self._controls[a].extend(np.atleast_2d(ctrls[i]))
                    state[i] = np.asarray(segs[i][-1], dtype=np.float64)
                else:
                    self._solved[a] = False
                    self.stats["braked_segments"] += 1
                    us, state[i] = self._brake(env, i, state[i], len(np.atleast_2d(ctrls[i])))
                    self._controls[a].extend(us)

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
    def _brake(env, i, state, n_steps):
        """Decelerate to rest and hold — the safe fallback for an unsolved segment."""
        r = env.robots[i]
        st = np.asarray(state, dtype=np.float64).copy()
        us = []
        for _ in range(max(0, int(n_steps))):
            a = float(np.clip(-st[3] / env.dt, r.a_min, r.a_max))
            al = float(np.clip(-st[4] / env.dt, r.alpha_min, r.alpha_max))
            u = np.array([a, al], dtype=np.float64)
            us.append(u)
            st = r.step(st, u, env.dt)
        return us, st

    # ── pieces ────────────────────────────────────────────────────────────────

    @staticmethod
    def _segment_horizon(env, t_cfg, state, goals, total_h, m) -> int:
        """Steps allotted to one segment: the slowest robot's bang-bang time over its
        own leg. Capped by the whole-plan budget so a pathological leg cannot eat it."""
        h = t_cfg.get("horizon", None)
        if h is not None:
            return max(2, int(h) // m)
        slack = float(t_cfg.get("slack", 1.5))
        from src.shaping.braking_potential import bangbang_time
        worst = max(
            bangbang_time(
                float(np.linalg.norm(np.asarray(goals[i])[:2] - np.asarray(state[i])[:2])),
                0.0, env.robots[i].v_max, env.robots[i].a_max,
            )
            for i in range(env._n)
        )
        return int(np.clip(np.ceil(slack * worst / env.dt), 2, total_h))

    @staticmethod
    def _total_horizon(env, t_cfg) -> int:
        """Time budget in env steps. See OptimizationPlanner._auto_horizon."""
        h = t_cfg.get("horizon", None)
        if h is not None:
            return int(h)
        slack = float(t_cfg.get("slack", 1.5))
        from src.shaping.braking_potential import bangbang_time
        worst = max(
            bangbang_time(
                float(np.linalg.norm(env._goals[i][:2] - env._states[i][:2])),
                0.0, env.robots[i].v_max, env.robots[i].a_max,
            )
            for i in range(env._n)
        )
        return min(env.max_steps, max(10, int(np.ceil(slack * worst / env.dt))))

    @staticmethod
    def _milestones(env, m: int, clearance: float) -> list[list[np.ndarray]]:
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
        out = []
        for i in range(env._n):
            s0 = np.asarray(env._states[i], dtype=np.float64)
            g = np.asarray(env._goals[i], dtype=np.float64)
            path = KARCPlanner._descend(grid, s0[:2], g[:2])
            out.append(KARCPlanner._resample(path, g, m))
        radii = [float(r.shape.bounding_radius) for r in env.robots]
        return KARCPlanner._separate(out, radii, clearance, env._world_size)

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

    def _solve(self, env, i, start, goal, seg_h, t_cfg, avoid, goal_scale=1.0,
               terminal_stop=True):
        avoid_trajs = tuple(a[0] for a in avoid)
        avoid_radii = tuple(a[1] for a in avoid)
        self.stats["solver_calls"] += 1
        X, _U, _dt, ok = solve_trajectory(
            env.robots[i], start, goal, env._obstacles, env._world_size,
            horizon=seg_h,
            effort_weight=float(t_cfg.get("effort_weight", 0.01)),
            dt_fixed=env.dt,
            avoid=avoid_trajs, avoid_radii=avoid_radii,
            goal_tol=float(t_cfg.get("goal_tol", env.goal_radius)) * goal_scale,
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=terminal_stop,
            max_iter=int(t_cfg.get("max_iters", 500)),
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

    def _resolve(self, rung, conflicts, segs, ctrls, oks, env, state, goals, seg_h,
                 t_cfg, radii, last=True):
        """One rung of Alg. 2. Robots in the subproblem are re-solved in priority
        order, each avoiding everything already committed this round."""
        # A robot joins the subproblem if it is in a conflict OR its own segment came
        # back infeasible -- both are failures the ladder exists to repair.
        involved = sorted(
            {i for c in conflicts for i in c[:2]}
            | {i for i, ok in enumerate(oks) if not ok}
        )
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
                tuple(avoid), last,
            )

        for level, i in enumerate(involved):
            scale = 1.0 if rung == "prioritized" else relax ** level
            X, U, ok = self._solve(
                env, i, state[i], goals[i], seg_h, t_cfg, tuple(avoid), goal_scale=scale,
                terminal_stop=last,
            )
            segs[i], ctrls[i], oks[i] = X, U, ok
            avoid.append((X, radii[i]))
        return segs, ctrls, oks

    def _solve_joint(self, involved, segs, ctrls, oks, env, state, goals, seg_h, t_cfg,
                     avoid, last):
        """K-ARC's AdaptSubProblem: re-solve the conflicting robots TOGETHER.

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
            goal_tol=float(t_cfg.get("goal_tol", env.goal_radius)),
            clearance=float(t_cfg.get("clearance", 0.05)),
            terminal_stop=last,
            max_iter=int(t_cfg.get("max_iters", 500)),
        )
        segs, ctrls, oks = list(segs), list(ctrls), list(oks)
        # One program, one verdict: the group is feasible together or not at all.
        for slot, i in enumerate(involved):
            segs[i], ctrls[i], oks[i] = Xs[slot], Us[slot], ok
        return segs, ctrls, oks
