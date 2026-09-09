"""Construct a coordinated plan instead of searching for one.

K-ARC's hierarchy is search all the way down: prioritised NLP, then decoupled kinodynamic
RRT, then a tree over the joint state. Every rung answers "find me a trajectory". At
density that is the wrong question -- ``scripts/feasibility_certificate.py`` shows
open_cross_32 is solvable inside K-ARC's own horizon by a plan with no search in it at
all, built from an analytic accelerate/decelerate profile and a two-group schedule.

What that witness could not claim was generality: it read the coordination straight off the
robot indices (``row = i // 2``, ``dy = ... if i % 2 == 0``, ``hold = ... if row % 2``),
which only works because we already knew the benchmark's layout. This module derives the
same three decisions from the conflict graph:

  pairing      which robots contend            -> proximity on the straight-line references
  side         who goes which way round        -> 2-colouring WITHIN each conflict
  time group   who manoeuvres when             -> colouring OVER the conflicts

Then it builds the trajectories analytically and verifies them against the environment's
own collision checker. Nothing is committed that does not verify, so a wrong assignment
costs one construction attempt and the caller's ladder still runs.

The geometry (how far to veer, where along the path) is still shaped by hand; only the
combinatorics are derived. See `veer_margin` and `lead`/`span` below.
"""
from __future__ import annotations

import numpy as np

from src.collision.shapes import collides, shape_distance


def _profile(delta, dt, acc_max, vel_max):
    """Accelerate/cruise/decelerate covering exactly `delta`, ending at rest.

    Solved on the semi-implicit Euler the env actually integrates with, so a leg LANDS on
    its target rather than near it. n steps at +a then n at -a advance a*dt^2*n^2; adding
    m cruise steps at the peak makes it a*dt^2*n*(n+m).

    The cruise phase is not a refinement, it is most of the horizon. Without it a long leg
    has to keep accelerating to its midpoint, so the velocity cap alone forces
    n = d/(dt*v_max) steps each way -- 600 steps for the 15 m traverse here against 320
    for the same motion at the same limits. Triangular profiles were costing 1.9x the
    budget on every robot, which is what put the cluttered plans over the horizon.
    """
    d = abs(float(delta))
    if d < 1e-12:
        return []
    sign = float(np.sign(delta))

    # Triangular: fastest when the leg is too short to reach the speed cap.
    n = max(1, int(np.ceil(np.sqrt(d / (dt * dt * acc_max)))))
    while True:
        a = d / (dt * dt * n * n)
        if a <= acc_max + 1e-12 and n * a * dt <= vel_max + 1e-12:
            break
        n += 1
    best = [a] * n + [-a] * n

    # Trapezoidal: reach the cap in `nu` steps, then hold it for the rest of the distance.
    nu = max(1, int(np.ceil(vel_max / (acc_max * dt))))
    if dt * dt * acc_max * nu * nu < d:
        m = int(np.ceil(d / (dt * dt * acc_max * nu) - nu))
        if m > 0:
            a2 = d / (dt * dt * nu * (nu + m))
            if a2 <= acc_max + 1e-12 and nu * a2 * dt <= vel_max + 1e-12:
                trap = [a2] * nu + [0.0] * m + [-a2] * nu
                if len(trap) < len(best):
                    best = trap
    return [x * sign for x in best]


def _turn_then_go(robot, state, target, dt):
    """One stop-and-go leg: turn in place to face `target`, drive to it, end at rest.

    Both phases start and end at rest, so the leg is realisable by any second-order
    unicycle with the bounds this one reports. Returns (states, controls).
    """
    st = np.asarray(state, dtype=np.float64).copy()
    xs, us = [], []
    want = float(np.arctan2(target[1] - st[1], target[0] - st[0]))
    turn = float(np.arctan2(np.sin(want - st[2]), np.cos(want - st[2])))
    dist = float(np.linalg.norm(np.asarray(target)[:2] - st[:2]))

    seq = [(0.0, al) for al in _profile(turn, dt, robot.alpha_max, robot.omega_max)]
    seq += [(a, 0.0) for a in _profile(dist, dt, robot.a_max, robot.v_max)]
    for a, al in seq:
        u = np.array([float(np.clip(a, robot.a_min, robot.a_max)),
                      float(np.clip(al, robot.alpha_min, robot.alpha_max))])
        st = robot.step(st, u, dt)
        xs.append(st.copy())
        us.append(u)
    return (np.asarray(xs), np.asarray(us)) if xs else None


def _resample(path, steps):
    """`steps` points spaced evenly by ARCLENGTH along a polyline."""
    pts = np.asarray(path, float)[:, :2]
    if len(pts) == 1:
        return np.repeat(pts, steps, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-12:
        return np.repeat(pts[:1], steps, axis=0)
    want = np.linspace(0.0, cum[-1], steps)
    return np.column_stack([np.interp(want, cum, pts[:, 0]),
                            np.interp(want, cum, pts[:, 1])])


def _references(starts, goals, guides=None, steps=64):
    """Each robot's intended route on a shared normalised progress grid.

    With `guides` this follows the OBSTACLE-FREE kinematic path the planner already
    computed; without them it degrades to the straight start->goal line. Sampling by
    arclength rather than by time is what makes index k comparable across robots whose
    routes differ in length -- everyone is k/steps of the way through their own journey.

    This is not a plan. It is only enough to say which robots contend, and WHERE along
    each one's route the contention happens.
    """
    if guides is not None:
        return [_resample(g, steps) for g in guides]
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return [np.asarray(s, float)[:2] * (1 - t) + np.asarray(g, float)[:2] * t
            for s, g in zip(starts, goals)]


def _window(refs, i, partners, sep):
    """Fractional span of i's route over which it is close to anything it contends with.

    The manoeuvre belongs where the conflict is, not at the middle of the map. On a row
    swap that recovers the midpoint; on a circular layout it lands at the hub; in clutter
    it lands wherever the routes actually pinch.
    """
    near = np.zeros(len(refs[i]), dtype=bool)
    for j in partners:
        near |= np.linalg.norm(refs[i] - refs[j], axis=1) < sep
    idx = np.flatnonzero(near)
    if len(idx) == 0:
        return None
    n = len(refs[i]) - 1
    return float(idx[0]) / n, float(idx[-1]) / n


def _lane_axis(refs, members, windows):
    """A single world-frame direction along which one conflict's robots are spread apart.

    It cannot be each robot's own normal. A normal flips with heading, so two robots meeting
    head-on have opposite normals, and giving them opposite lane signs displaces them to the
    SAME side -- they never separate. The axis therefore belongs to the CONFLICT, not to the
    robot: take the dominant direction of travel across everyone involved (via the tangent
    covariance, which is blind to v vs -v, exactly as a head-on pair requires) and spread
    them perpendicular to it.
    """
    tang = []
    for i in members:
        pts = refs[i]
        m = len(pts) - 1
        if m < 1 or windows.get(i) is None:
            continue
        k = min(max(int(0.5 * (windows[i][0] + windows[i][1]) * m), 0), m - 1)
        t = pts[k + 1] - pts[k]
        ln = float(np.linalg.norm(t))
        if ln > 1e-9:
            tang.append(t / ln)
    if not tang:
        return np.array([0.0, 1.0])
    T = np.asarray(tang)
    w, V = np.linalg.eigh(T.T @ T)
    d = V[:, int(np.argmax(w))]
    return np.array([-d[1], d[0]])


def _offset_path(pts, dy, lo, hi, taper, axis):
    """Displace a route sideways by `dy` along `axis` across [lo, hi], ramped in and out.

    `axis` is a fixed world-frame direction shared by everyone in the conflict, so opposite
    lane signs put robots on genuinely opposite sides whatever their headings. Outside the
    ramp the route is untouched, which keeps start and goal exact.
    """
    pts = np.asarray(pts, float)
    m = len(pts)
    f = np.linspace(0.0, 1.0, m)
    ramp = np.clip(np.minimum((f - (lo - taper)) / taper,
                              ((hi + taper) - f) / taper), 0.0, 1.0)
    axis = np.asarray(axis, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return pts + axis[None, :] * (dy * ramp)[:, None]


def _conflict_pairs(refs, sep):
    """Robots whose routes come within `sep` at the same normalised progress."""
    out = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            if float(np.linalg.norm(refs[i] - refs[j], axis=1).min()) < sep:
                out.append((i, j))
    return out


def _colour(nodes, adj):
    """Greedy colouring, highest degree first. Returns {node: colour}."""
    colour: dict = {}
    for u in sorted(nodes, key=lambda x: -len(adj.get(x, ()))):
        taken = {colour[v] for v in adj.get(u, ()) if v in colour}
        c = 0
        while c in taken:
            c += 1
        colour[u] = c
    return colour


def _sides(n, pairs):
    """Which lane each robot takes: partners in a conflict get different colours."""
    adj: dict = {i: set() for i in range(n)}
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    return _colour(range(n), adj)


def _verify(env, tracks, clearance, report=None):
    """Every pair at every shared index, with the env's own checker.

    Returns the tightest surface gap, or None if anything touches, a robot misses its
    goal, or the tightest pass is under `clearance`. A rejected construction is the normal
    case on a scenario the geometry does not suit, so `report` is filled with WHY -- a
    silent None is the difference between "does not generalise" and "generalises but the
    lane is 2 cm too wide", and those need opposite responses.
    """
    n, T = env._n, len(tracks[0])
    goals = [np.asarray(g, float) for g in env._goals]
    rep = {"obstacle_hits": 0, "robot_hits": 0, "missed_goals": 0,
           "min_gap": float("inf"), "steps": T}
    worst = float("inf")
    for k in range(T):
        for i in range(n):
            pi = (float(tracks[i][k][0]), float(tracks[i][k][1]), float(tracks[i][k][2]))
            for ob in env._obstacles:
                if collides(env.robots[i].shape, pi, ob.shape, ob.pose):
                    rep["obstacle_hits"] += 1
            for j in range(i + 1, n):
                pj = (float(tracks[j][k][0]), float(tracks[j][k][1]),
                      float(tracks[j][k][2]))
                if collides(env.robots[i].shape, pi, env.robots[j].shape, pj):
                    rep["robot_hits"] += 1
                worst = min(worst, shape_distance(env.robots[i].shape, pi,
                                                 env.robots[j].shape, pj))
    rep["missed_goals"] = sum(
        float(np.linalg.norm(tracks[i][-1][:2] - goals[i][:2])) >= env.goal_radius
        for i in range(n))
    rep["min_gap"] = round(float(worst), 4)
    # A plan longer than the horizon is not a plan: the episode ends before the robots
    # arrive, so "collision-free and reaches every goal" is true only of a trajectory that
    # never finishes being executed. Serialising groups trades collisions for length, so
    # this is the constraint that keeps that trade honest.
    budget = int(getattr(env, "max_steps", 0) or 0)
    rep["over_budget"] = int(bool(budget and T > budget))
    if report is not None:
        report.update(rep)
    if (rep["obstacle_hits"] or rep["robot_hits"] or rep["missed_goals"]
            or rep["over_budget"]):
        return None
    return worst if worst >= clearance else None


def _simplify(pts, tol=0.05):
    """Drop vertices that the route does not need, by DEVIATION from the chord.

    Every waypoint costs a full stop -- `_turn_then_go` ends each leg at rest -- so a route
    resampled to a fixed count pays for stops on straight ground. Douglas-Peucker, not a
    turn-angle test: the lane displacement is a gradual bulge over ~128 samples whose
    per-step turn is far below any sensible angle threshold, so an angle test deletes the
    entire manoeuvre and puts a head-on pair back on one line. Deviation keeps the bulge
    (it is exactly the thing that departs from the chord) and still collapses straight runs.
    """
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        chord = pts[b] - pts[a]
        ln = float(np.linalg.norm(chord))
        seg = pts[a + 1:b] - pts[a]
        if ln < 1e-12:
            d = np.linalg.norm(seg, axis=1)
        else:
            d = np.abs(np.cross(np.repeat((chord / ln)[None, :], len(seg), axis=0), seg))
        m = int(np.argmax(d))
        if float(d[m]) > tol:
            keep[a + 1 + m] = True
            stack += [(a, a + 1 + m), (a + 1 + m, b)]
    return pts[keep]


def _hits_obstacle(env, i, pts):
    """Does this route put robot i's body through an obstacle at any sample?"""
    sh = env.robots[i].shape
    pts = np.asarray(pts, float)
    for k in range(len(pts)):
        th = 0.0
        if k + 1 < len(pts):
            d = pts[k + 1] - pts[k]
            if np.linalg.norm(d) > 1e-9:
                th = float(np.arctan2(d[1], d[0]))
        pose = (float(pts[k][0]), float(pts[k][1]), th)
        if any(collides(sh, pose, ob.shape, ob.pose) for ob in env._obstacles):
            return True
    return False


def _traj_conflicts(tracks, radii, sep):
    """Pairs that are too close on the CONSTRUCTED trajectories, with the last such index.

    The lane assignment is spatial and can be decided from the routes alone, but WHEN two
    robots are in the same place depends on how long their legs actually took -- which is
    known only after construction. Scheduling on reference progress instead assumes every
    robot advances at the same rate, which is true on the open cross (identical robots,
    identical routes) and false everywhere else.
    """
    n = len(tracks)
    out: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            T = min(len(tracks[i]), len(tracks[j]))
            d = np.linalg.norm(np.asarray(tracks[i])[:T, :2]
                               - np.asarray(tracks[j])[:T, :2], axis=1)
            bad = np.flatnonzero(d < radii[i] + radii[j] + sep)
            if len(bad):
                out[(i, j)] = int(bad[-1])
    return out


def _legs(env, i, route, dt):
    """Drive a route as a sequence of rest-to-rest legs. Returns (states, controls)."""
    st = np.asarray(env._states[i], float).copy()
    xs, us = [], []
    for wp in np.asarray(route, float)[1:]:
        piece = _turn_then_go(env.robots[i], st, wp, dt)
        if piece is None:
            continue
        xs.append(piece[0])
        us.append(piece[1])
        st = piece[0][-1].copy()
    if not xs:
        return None
    return np.vstack(xs), np.vstack(us)


def plan(env, params, clearance=0.05, guides=None):
    """A verified coordinated plan, or None. Returns (tracks, controls, info).

    ``tracks[i]`` is (T, 5) states and ``controls[i]`` is (T, 2), padded to a common
    horizon with a rest hold so index k is the same instant for every robot.

    Nothing here knows what a row is. The two assignments are split by what each actually
    depends on:

      LANE  (which side of a contested stretch a robot takes) is geometry. Which routes
            cross is a property of the routes, so it is decided from them, before any
            trajectory exists, by colouring the route-proximity graph.
      GROUP (when a robot takes its manoeuvre) is dynamics. Two robots share a place only
            if they are there at the same TIME, and the time a leg takes is known only
            after the leg is built -- so this is decided on the constructed trajectories.

    Deciding both from reference progress is what limits the construction to scenarios
    where every robot advances at the same rate.
    """
    dt = float(env.dt)
    n = env._n
    starts = [np.asarray(s, float) for s in env._states]
    goals = [np.asarray(g, float) for g in env._goals]
    radii = [float(r.shape.bounding_radius) for r in env.robots]

    sh = env.robots[0].shape
    lat = float(getattr(sh, "length", 2 * radii[0]))
    lane_w = lat + clearance + float(params.get("veer_margin", 0.2))
    taper = float(params.get("taper", 0.12))

    # --- LANES: spatial, from the routes ------------------------------------------------
    refs = _references(starts, goals, guides)
    pairs = _conflict_pairs(refs, lat + clearance)
    if not pairs:
        return None
    side = _sides(n, pairs)
    lanes = max(side.values()) + 1 if side else 1
    partners: dict = {i: set() for i in range(n)}
    for i, j in pairs:
        partners[i].add(j)
        partners[j].add(i)

    # One lane axis per CONFLICT CLUSTER, in the world frame. Everyone involved in the
    # same encounter is spread along the same direction, which is what makes opposite lane
    # signs mean opposite sides regardless of who is heading which way.
    windows = {i: _window(refs, i, partners[i], lane_w) for i in range(n)}
    axis_of: dict = {}
    seen: set = set()
    for i in range(n):
        if i in seen or not partners[i]:
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack += [v for v in partners[u] if v not in seen]
        ax = _lane_axis(refs, comp, windows)
        for u in comp:
            axis_of[u] = ax

    routes, base = [], []
    for i in range(n):
        win = windows[i]
        # Hold the lane across the whole stretch the encounter COULD occupy, not just the
        # slice where the reference paths happen to be closest. The two robots traverse
        # their routes at different rates -- a robot that has to turn around first loses
        # ~200 steps to the turn -- so a narrow bulge is entered at different times and both
        # are back on the centre line when they actually meet. Measured on open_cross_32:
        # window (0.492, 0.508), lanes correctly +/-0.25, and still a dead-centre collision.
        if win:
            pad = float(params.get("window_pad", 0.35))
            win = (max(0.0, win[0] - pad), min(1.0, win[1] + pad))
        dy = (side.get(i, 0) - (lanes - 1) / 2.0) * lane_w if win else 0.0
        dense = _resample(refs[i], 128)
        route = dense
        if win and abs(dy) > 1e-9:
            # A lane is only available if it is free. Try the assigned side, then its
            # mirror, then give up on displacement and let the schedule do the work --
            # rather than pushing the body through a pillar to honour a colouring.
            for cand in (dy, -dy):
                trial = _offset_path(dense, cand, win[0], win[1], taper,
                                     axis_of.get(i, (0.0, 1.0)))
                if not _hits_obstacle(env, i, trial):
                    route = trial
                    break
        route = _simplify(route, float(params.get("simplify_tol", 0.05)))
        route = np.vstack([route, goals[i][:2]])
        built = _legs(env, i, np.vstack([starts[i][:2], route]), dt)
        if built is None:
            return None
        routes.append(route)
        base.append(built)

    tracks = [b[0] for b in base]
    ctrls = [b[1] for b in base]

    # --- GROUPS: temporal, and iterated to a fixed point -------------------------------
    # Scheduling is over CONFLICTS, not robots: the two robots of a swap must move at the
    # same time, since one's goal is the other's start and holding one parks it exactly
    # where its partner is heading. Lanes separate partners; groups separate one conflict
    # from another.
    #
    # The subtlety is that shifting a group CHANGES which robots meet. Detecting conflicts
    # once on the unheld trajectories and then applying holds invalidates the very
    # detection the holds were derived from -- measured on cluttered_cross_16 as 280
    # collisions between robots that are both MOVING (none parked), surviving even the
    # largest offset. So re-detect at the shifted timing and accumulate: the conflict graph
    # only grows, so this terminates.
    pair_of: dict = {}
    for pi, (i, j) in enumerate(pairs):
        pair_of.setdefault(i, pi)
        pair_of.setdefault(j, pi)

    def _hold(off, gof):
        out_t, out_u = [], []
        for r in range(n):
            h = off * int(gof.get(r, 0))
            if h:
                out_t.append(np.vstack([np.repeat(starts[r][None, :], h, axis=0),
                                        tracks[r]]))
                out_u.append(np.vstack([np.zeros((h, 2)), ctrls[r]]))
            else:
                out_t.append(tracks[r])
                out_u.append(ctrls[r])
        H = max(len(t) for t in out_t)
        return ([np.vstack([t, np.repeat(t[-1][None, :], H - len(t), axis=0)])
                 if len(t) < H else t for t in out_t],
                [np.vstack([c, np.zeros((H - len(c), 2))]) if len(c) < H else c
                 for c in out_u], H)

    edges: set = set()
    intra = 0
    best, last = None, {}
    forced = int(params.get("group_offset", 0))
    for _ in range(int(params.get("schedule_iters", 5))):
        adj: dict = {pi: set() for pi in range(len(pairs))}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        gcol = _colour(range(len(pairs)), adj)
        group_of = {r: gcol.get(pair_of.get(r, -1), 0) for r in range(n)}
        ngroups = max(gcol.values()) + 1 if gcol else 1

        probe = _traj_conflicts(tracks, radii, clearance)
        cross = [k for (a, b), k in probe.items()
                 if pair_of.get(a) is not None and pair_of.get(a) != pair_of.get(b)]
        hi = forced or (max(cross) + 2 if cross else 0)
        cands = [hi] if (forced or hi == 0) else sorted(
            {max(1, hi // 4), max(1, hi // 2), max(1, 3 * hi // 4), hi})

        found = None
        for off in cands:
            ct, cu, H = _hold(off, group_of)
            rep: dict = {}
            gap = _verify(env, ct, clearance, rep)
            last = rep
            if gap is not None:
                found = (ct, cu, off, gap, H, ngroups)
                break
            # What is still touching AT THIS TIMING is what the next round must schedule.
            for (a, b), _k in _traj_conflicts(ct, radii, clearance).items():
                pa, pb = pair_of.get(a), pair_of.get(b)
                if pa is None or pb is None:
                    continue
                if pa == pb:
                    intra += 1
                else:
                    edges.add((pa, pb))
        if found:
            best = found
            break

    if best is None:
        if isinstance(params, dict):
            params.setdefault("_reject", {}).update(last)
        return None
    tracks, ctrls, offset, gap, T, ngroups = best
    info = {"pairs": len(pairs), "lanes": lanes, "groups": ngroups, "lane_failures": intra,
            "offset": offset, "steps": T, "min_surface_gap": round(float(gap), 4)}
    return tracks, ctrls, info
