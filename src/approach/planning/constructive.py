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

from src.approach.planning import flat
from src.approach.planning.flat import profile as _profile
from src.collision.shapes import collides, shape_distance


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


def _axis_groups(refs, comp, windows, tol=np.pi / 4):
    """Split a conflict cluster into sub-groups whose routes actually share a direction.

    One axis per connected cluster is right when the cluster IS one encounter -- and every
    cluster in `open_cross_32` and `cluttered_cross_16` is (tangent eigenvalue ratio 0.000
    to 0.038). It stops being right when a component grows by transitivity: the 20-robot
    cluster in `cluttered_cross_32` has a ratio of 0.159, meaning its routes point in
    unrelated directions and the dominant eigenvector is close to arbitrary. Displacing 20
    robots along one arbitrary axis separates none of them.

    Orientation is taken mod pi, so a head-on pair -- whose tangents differ by exactly pi --
    stays in one group and keeps the shared axis that makes its opposite lane signs mean
    opposite sides. Only genuinely differently-oriented encounters are split apart.
    """
    ang = {}
    for i in comp:
        pts = refs[i]
        m = len(pts) - 1
        if m < 1 or windows.get(i) is None:
            continue
        k = min(max(int(0.5 * (windows[i][0] + windows[i][1]) * m), 0), m - 1)
        t = pts[k + 1] - pts[k]
        if float(np.linalg.norm(t)) < 1e-9:
            continue
        ang[i] = float(np.arctan2(t[1], t[0]) % np.pi)
    if len(ang) < 2:
        return [comp]
    order = sorted(ang, key=lambda i: ang[i])
    groups, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if ang[b] - ang[a] > tol:
            groups.append(cur)
            cur = [b]
        else:
            cur.append(b)
    # Orientation wraps at pi, so the last group may be the same direction as the first.
    if groups and (ang[order[0]] + np.pi) - ang[order[-1]] <= tol:
        groups[0] = cur + groups[0]
    else:
        groups.append(cur)
    rest = [i for i in comp if i not in ang]
    if rest:
        groups.append(rest)
    return groups


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
    ramp the route is untouched, which keeps start and goal exact -- but only if the ramp
    has room to close, so the window is held back from both ends by one taper. Without that
    clamp a window padded out to [0, 1] leaves the displacement at FULL value on the last
    sample, so the route ends half a lane to the side of the goal and the goal has to be
    reached by an extra jog. Stop-and-go absorbs that jog as one cheap leg; a smooth curve
    cannot, because the jog is a hairpin -- measured on cluttered_cross_16, |kappa| 17.3
    (a 5.8 cm turning radius) on the last sample of an 18 m route, which caps that robot at
    0.03 m/s and stretched its traverse to 1483 steps, past the horizon.
    """
    pts = np.asarray(pts, float)
    m = len(pts)
    lo, hi = max(float(lo), taper), min(float(hi), 1.0 - taper)
    if hi < lo:
        lo = hi = 0.5 * (lo + hi)
    f = np.linspace(0.0, 1.0, m)
    ramp = np.clip(np.minimum((f - (lo - taper)) / taper,
                              ((hi + taper) - f) / taper), 0.0, 1.0)
    axis = np.asarray(axis, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return pts + axis[None, :] * (dy * ramp)[:, None]


def _hub(refs, comp, sep):
    """A conflict cluster whose encounters all pile onto ONE point, or None.

    Lanes are a two-sided device: they separate the two robots of a crossing by putting
    one on each side of it. That is the right tool when a cluster is a handful of distinct
    encounters spread through space, and the wrong one when every route in the cluster
    passes through the same place -- no number of sides separates n routes at a point.
    Measured: open_cross_32 and cluttered_cross_16 have encounter midpoints spread over
    ~4.6 m of y (independent pairwise crossings), while circular_cross_16/32 have a
    midpoint spread of exactly 0.000 -- all 32 routes are diameters of one circle.

    Returns (centre, members) when the pairwise closest-approach points of the cluster sit
    inside a body-sized ball, which is what "one point" means for a robot of this size.
    """
    if len(comp) < 3:
        return None
    hits = []
    for a in range(len(comp)):
        for b in range(a + 1, len(comp)):
            i, j = comp[a], comp[b]
            k = int(np.argmin(np.linalg.norm(refs[i] - refs[j], axis=1)))
            hits.append(0.5 * (refs[i][k] + refs[j][k]))
    if not hits:
        return None
    hits = np.asarray(hits)
    centre = hits.mean(0)
    if float(np.linalg.norm(hits - centre, axis=1).max()) > sep:
        return None
    return centre


def _orbit(ref, centre, radius):
    """Right-hand offset that carries `ref` around `centre` instead of through it.

    Every member of a hub is displaced to its OWN right, so they all circulate the hub the
    same way -- a robot heading +x passes below it, one heading +y passes to its right, and
    those two senses agree. That is what makes the displacement collective: a shared axis
    cannot do it, because "the same side" of a hub is a different direction for each
    approach. Returns (axis, dy, window) for `_offset_path`, or None if `ref` misses the
    hub anyway.
    """
    d = np.linalg.norm(ref - centre[None, :], axis=1)
    k = int(np.argmin(d))
    m = len(ref) - 1
    if m < 1 or d[k] > radius + 1e-9 and k in (0, m):
        return None
    a = max(k - 1, 0)
    b = min(k + 1, m)
    t = ref[b] - ref[a]
    ln = float(np.linalg.norm(t))
    if ln < 1e-9:
        return None
    t = t / ln
    axis = np.array([t[1], -t[0]])          # right-hand normal
    # Already past the hub on the correct side? Then only top it up to the ring.
    dy = radius - float(np.dot(ref[k] - centre, axis))
    f = k / m
    return axis, dy, (max(0.0, f - 0.25), min(1.0, f + 0.25))


def _timed(env, refs, dt, smooth):
    """Where each robot IS at each instant, on its own undisplaced route.

    This exists to replace a proxy. Comparing routes at equal NORMALISED PROGRESS asks
    "are these two robots the same fraction of the way through their own journeys, and
    close when they are?" -- which is a statement about progress, not about time, and is a
    good stand-in for a conflict only while every robot advances at a comparable rate. It
    stops being one as soon as the rates differ: a robot that must turn around first, or
    one throttled to 0.15 m/s through a bend while its partner cruises at 0.5, is nowhere
    near the same clock time as its equal-progress partner. The grid then both invents
    conflicts that never happen and misses ones that do.

    Once a route can be given a time parameterisation (`flat.reference`) that proxy is not
    needed. Index k here is a real instant -- k*dt seconds after a common start -- so the
    conflict test below is an ordinary space-time test and the normalisation disappears.
    Robots that finish early are held at their goal, which is where they actually are.

    Returns (positions padded to a common horizon, per-robot cumulative arclength).
    """
    pos, arc = [], []
    for i, r in enumerate(refs):
        ref = flat.reference(r, env.robots[i], dt, smooth=smooth)
        if ref is None:
            return None
        p = np.column_stack([ref["x"], ref["y"]])
        if len(p) < 2:
            return None
        pos.append(p)
        arc.append(np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]))
    T = max(len(p) for p in pos)
    held = [np.vstack([p, np.repeat(p[-1:], T - len(p), axis=0)]) if len(p) < T else p
            for p in pos]
    return held, arc


def _window_timed(pos, arc, i, partners, sep):
    """Where along i's route it is close IN TIME to something it contends with.

    The answer has to come back as a fraction of ARCLENGTH, not of time, because that is
    what the lane displacement is applied over -- the manoeuvre is a piece of road, not a
    piece of clock. So the contested instants are found in time and then converted through
    the route's own cumulative arclength.
    """
    near = np.zeros(len(pos[i]), dtype=bool)
    for j in partners:
        near |= np.linalg.norm(pos[i] - pos[j], axis=1) < sep
    idx = np.flatnonzero(near)
    total = float(arc[i][-1])
    if len(idx) == 0 or total < 1e-9:
        return None
    m = len(arc[i]) - 1
    k0, k1 = int(min(idx[0], m)), int(min(idx[-1], m))
    return float(arc[i][k0] / total), float(arc[i][k1] / total)


def _conflict_pairs(refs, sep):
    """Robots whose routes come within `sep` at the same normalised progress."""
    out = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            if float(np.linalg.norm(refs[i] - refs[j], axis=1).min()) < sep:
                out.append((i, j))
    return out


def _goal_order(refs, starts, goals, sep):
    """An insertion order that respects where robots come to REST.

    A robot that has arrived never moves again. If q's goal sits on r's route, then q
    parked there blocks r at every delay -- and delaying r, the only lever insertion has,
    makes the block more certain rather than less. Measured on cluttered_cross_32: the
    robot that could not be seated had exactly one blocker, parked on its goal at every
    one of its candidate delays, and 250 random orders never cleared it.

    So the constraint is read off the geometry instead of searched for: edge r -> q when
    q's goal lies within `sep` of r's route means "r goes first". Swaps make 2-cycles (each
    one's goal is the other's start), which is not a contradiction -- both simply leave
    early -- so cycles are broken by longest trajectory first and the rest of the order
    still holds. Returns a topological order over as much of the graph as is acyclic.
    """
    n = len(refs)
    goals = [np.asarray(g, float)[:2] for g in goals]
    starts = [np.asarray(s, float)[:2] for s in starts]
    after: dict = {r: set() for r in range(n)}     # r must precede everything in after[r]

    def _on(route, pt):
        return float(np.linalg.norm(route - pt[None, :], axis=1).min()) < sep

    for r in range(n):
        for q in range(n):
            if q == r:
                continue
            # q parks on r's route, so r must be through before q arrives.
            # r waits on q's route, so r must leave before q gets there -- and delaying r
            # only keeps it sitting there longer, which is the same trap seen from the
            # other end. Both say the same thing: r goes first.
            if _on(refs[r], goals[q]) or _on(refs[q], starts[r]):
                after[r].add(q)
    indeg = {q: 0 for q in range(n)}
    for r in range(n):
        for q in after[r]:
            indeg[q] += 1
    # A swap makes the constraint MUTUAL: each partner's goal lies exactly on the other's
    # route, so whichever arrives first parks in the other's way. That is harmless when the
    # two go at similar times -- they cross in the middle and trade places -- and fatal when
    # they do not. On cluttered_cross_32 robot 29 was seated 22nd, by which point every
    # small delay was taken and its partner had long since parked, so no delay existed.
    # Partners are therefore kept ADJACENT in the order rather than merely ranked.
    mate = {}
    for r in range(n):
        for q in after[r]:
            if r in after[q] and r not in mate and q not in mate:
                mate[r] = q
                mate[q] = r

    order, left = [], set(range(n))
    while left:
        ready = [q for q in left if indeg[q] == 0]
        if not ready:                               # a cycle: release the longest route
            ready = [max(left, key=lambda q: len(refs[q]))]
        ready.sort(key=lambda q: -len(after[q]))
        for q in ready:
            if q not in left:
                continue
            for t in (q, mate.get(q)):
                if t is None or t not in left:
                    continue
                order.append(t)
                left.discard(t)
                for u in after[t]:
                    if u in left:
                        indeg[u] -= 1
                indeg[t] = 0
    return order


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


def _fit_blur(env, i, route, target, base, cap):
    """The largest blur this route can take while still being the route that was planned.

    Blur is not a cosmetic setting, it is the speed knob. Curvature caps speed through
    w = kappa*v, so a sharp corner inherited from the guide throttles the whole bend --
    measured on cluttered_cross_16, blur 0.12 leaves a robot crawling at 0.245 m/s and its
    traverse 952 steps long, over the horizon once a delay is added, while blur 0.60 gets
    the same robot to 0.401 m/s and 513 steps. Rounding the corner IS the speed-up.

    What stops it being free is that a blurred curve cuts the corner, and a route that has
    been displaced into a lane must stay in that lane or the displacement meant nothing.
    So take the largest blur whose curve stays within `cap` of the intended route and
    clear of the obstacles, and only then hand it to the profiler.
    """
    for mult in (8.0, 4.0, 2.0, 1.0):
        got = flat._curve(route, base * mult)
        if got is None:
            continue
        pts = got[0]
        dev = float(np.linalg.norm(pts[:, None, :] - target[None, :, :],
                                   axis=2).min(axis=1).max())
        if dev <= cap and not _hits_obstacle(env, i, pts):
            return base * mult
    return base


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

    Smooth first, stop-and-go as the fallback. The two constructions fail on DIFFERENT
    scenarios, and nothing is committed that has not passed `_verify`, so trying the fast
    one and keeping the reliable one costs a rejected construction and buys the union of
    what either can do. Measured: smooth cuts open_cross_32 from 633 to 332 steps,
    circular_cross_16 from 836 to 454 and circular_cross_32 from 1212 to 730, while
    cluttered_cross_16 seats only 14 of 16 robots under smooth and 16 of 16 under legs.
    """
    if float(params.get("smooth", 0.0)) > 0.0:
        out = _build(env, params, clearance, guides)
        if out is not None:
            return out
        fallback = dict(params)
        fallback["smooth"] = 0.0
        out = _build(env, fallback, clearance, guides)
        if out is not None:
            out[2]["drive"] = "legs"
        elif isinstance(params, dict):
            params.setdefault("_reject", {}).update(fallback.get("_reject", {}))
        return out
    return _build(env, params, clearance, guides)


def _build(env, params, clearance=0.05, guides=None):
    """One construction at the drive mode `params` asks for. Returns (tracks, controls, info).

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

    refs = _references(starts, goals, guides)

    # --- LANES: spatial, from the routes ------------------------------------------------
    # Two BOXES can touch with their centres a body diagonal apart, not a lateral extent
    # apart, and a route pair missed here is a pair that never gets a lane.
    diag = float(np.hypot(float(getattr(sh, "width", 2 * radii[0])), lat))
    # WHEN conflicts are measured. The progress grid is the one discrete approximation
    # left in an otherwise continuous pipeline; `spacetime` replaces it with the real
    # thing, at the cost of needing a time parameterisation to exist -- so it rides with
    # smooth mode, which is where one does.
    smooth = float(params.get("smooth", 0.0))
    timed = None
    if smooth > 0.0 and bool(params.get("spacetime", False)):
        timed = _timed(env, refs, dt, smooth)
    field = timed[0] if timed is not None else refs

    pairs = _conflict_pairs(field, diag + clearance)
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
    if timed is not None:
        windows = {i: _window_timed(timed[0], timed[1], i, partners[i], lane_w)
                   for i in range(n)}
    else:
        windows = {i: _window(refs, i, partners[i], lane_w) for i in range(n)}
    axis_of: dict = {}
    orbit_of: dict = {}
    hubs: list = []
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
        # Which device this cluster gets is read off the cluster's own geometry: a handful
        # of separate crossings gets lanes, a pile-up on one point gets a roundabout.
        centre = _hub(refs, comp, diag + clearance)
        if centre is not None:
            # Size the ring so the whole cluster would fit around it even if every member
            # arrived at once: |comp| bodies at (diag + clearance) of arc.
            # pack=1.0 (the geometric minimum) is also the best measured setting: on
            # circular_cross_32 it seats all 32, while 1.3 seats 21 and 2.0 seats 18. A
            # wider ring is a longer detour through the same congested annulus, so the
            # extra room costs more than it buys.
            pack = float(params.get("hub_pack", 1.0))
            R = max(lane_w, pack * len(comp) * (diag + clearance) / (2.0 * np.pi))
            for u in comp:
                orb = _orbit(refs[u], centre, R)
                if orb is not None:
                    axis_of[u], orbit_of[u] = orb[0], (orb[1], orb[2])
            hubs.append((centre, R, len(comp)))
            continue
        for grp in _axis_groups(refs, comp, windows):
            ax = _lane_axis(refs, grp, windows)
            for u in grp:
                axis_of[u] = ax

    # A lane has to fit the corridor the robot actually has. The colouring can ask for more
    # lanes than the map holds: cluttered_cross_32 wants 4, which at lane_w 0.5 spans 1.5 m
    # between rows 1.0 m apart, so the outer lanes reach into the NEIGHBOURING row and the
    # device manufactures conflicts instead of resolving them. Room is the closest a robot
    # comes to a route it does NOT contend with, less the body it has to keep clear.
    room = float("inf")
    for i in range(n):
        for j in range(n):
            if j != i and j not in partners[i]:
                room = min(room, float(np.linalg.norm(refs[i] - refs[j], axis=1).min()))
    room = max(0.0, room - (lat + clearance)) if np.isfinite(room) else 0.0
    fits = int(room // (lat + clearance)) + 1 if room > 0 else 1
    lanes = max(1, min(lanes, fits))
    # Keep the tuned pitch where it fits and only compress when it does not, so scenarios
    # that already work are untouched (open_cross_32: 2 lanes, 0.70 m of room, unchanged).
    pitch = min(lane_w, room / (lanes - 1)) if lanes > 1 else 0.0

    routes, base, dropped = [], [], 0
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
        if i in orbit_of:
            dy, win = orbit_of[i]
        else:
            dy = ((side.get(i, 0) % lanes) - (lanes - 1) / 2.0) * pitch if win else 0.0
        dense = _resample(refs[i], 128)
        route = dense
        if win and abs(dy) > 1e-9:
            # A lane is only available if it is free. Try the assigned side, then its
            # mirror, then give up on displacement and let the schedule do the work --
            # rather than pushing the body through a pillar to honour a colouring.
            # Mirroring is a lane's fallback, not a roundabout's: sending one member the
            # other way round the hub puts it head-on into the whole circulation.
            # A blocked lane is not the same as no lane. The colouring assigns a lane; if
            # a pillar sits in it, the robot should CHANGE LANE, not give up and sit on the
            # centre line where its head-on partner already is -- that leaves the encounter
            # to be resolved in time, which on cluttered_cross_32 there is none of (1.0 m
            # rows, 1300 steps). Measured there: all 5 dropped robots were in the outermost
            # lanes of a 4-lane split, so the free lane is an inner one and widening the
            # blocked offset -- the obvious ladder -- searches away from it.
            if i in orbit_of:
                cands = [dy]
            else:
                lane_set = [(c - (lanes - 1) / 2.0) * pitch for c in range(lanes)]
                cands = sorted(lane_set, key=lambda x: (abs(x - dy), -abs(x)))
            for cand in cands:
                if abs(cand) < 1e-9:
                    continue
                trial = _offset_path(dense, cand, win[0], win[1], taper,
                                     axis_of.get(i, (0.0, 1.0)))
                if not _hits_obstacle(env, i, trial):
                    route = trial
                    break
            else:
                # No side of this encounter is free of the obstacles. The robot keeps its
                # centre line and the conflict is left entirely to the schedule -- which is
                # the expensive resolution, so it is worth counting rather than silent.
                dropped += 1
        # How the route is DRIVEN. Stop-and-go turns the route into a polyline and takes
        # it one rest-to-rest leg at a time, which never fires both controls at once and
        # pays a full stop per waypoint. Smooth mode flies the whole route as one
        # time-parameterised curve (`flat`), so linear and angular acceleration act
        # together and the intermediate waypoints stop being events at all -- which is
        # also why `_simplify` is skipped there: Douglas-Peucker exists to delete stops,
        # and in smooth mode there are none to delete.
        if smooth > 0.0:
            full = np.vstack([starts[i][:2], route, goals[i][:2]])
            # Room to round corners in. The budget is a fraction of the LANE PITCH
            # because that is what the deviation spends: two robots in adjacent lanes
            # each straying `cap` toward the other close 2*cap of the pitch that was
            # separating them, so the fraction has to leave the lane meaning something.
            cap = float(params.get("blur_cap", 0.5)) * (pitch if pitch > 1e-9 else lane_w)
            built = flat.trajectory(env.robots[i], env._states[i], full, dt,
                                    smooth=_fit_blur(env, i, full, route, smooth, cap))
        else:
            route = _simplify(route, float(params.get("simplify_tol", 0.05)))
            route = np.vstack([route, goals[i][:2]])
            built = _legs(env, i, np.vstack([starts[i][:2], route]), dt)
        if built is None:
            return None
        routes.append(route)
        base.append(built)

    tracks = [b[0] for b in base]
    ctrls = [b[1] for b in base]

    # --- SCHEDULE: per-robot delays, by insertion ---------------------------------------
    # Colouring conflicts and staggering colour c by c*offset is uniform and blunt: every
    # robot in a colour waits the same amount whether it needed 10 steps or 400, and the
    # horizon is (groups-1)*offset + max_traj, which explodes exactly when a scenario needs
    # many colours. Measured on cluttered_cross_16: 3 groups at offset 407 gave 1668 steps
    # and 280 surviving collisions, and finer colouring made it 2804 steps and 2923.
    #
    # Insert robots one at a time instead, each delayed the LEAST that clears everyone
    # already placed. A placement is only accepted against trajectories already committed,
    # so the schedule is correct by construction rather than by re-detection; the robots
    # that need no delay get none. Longest trajectory first, because the hardest robot to
    # fit should choose while the space is still empty.
    step = max(1, int(params.get("delay_step", 5)))
    horizon_cap = int(getattr(env, "max_steps", 0) or 0) or 10 ** 6

    def _clash(a, da, b, db):
        """Do a (delayed da) and b (delayed db) ever come within `clearance`?

        Compared over the FULL span, clamped at both ends: before its delay a robot sits
        at its start, and after arrival it sits at its goal. Both are real occupancy --
        skipping the tail let a robot park on a spot a later robot drives through (79 hits
        on circular_cross_32). Cheap centre-distance pre-filter, exact shape test only on
        the steps that survive it.
        """
        ta, tb = tracks[a], tracks[b]
        span = max(da + len(ta), db + len(tb))
        ia = np.clip(np.arange(span) - da, 0, len(ta) - 1)
        ib = np.clip(np.arange(span) - db, 0, len(tb) - 1)
        pa, pb = ta[ia], tb[ib]
        near = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1) < radii[a] + radii[b] + clearance
        if not near.any():
            return False
        for k in np.flatnonzero(near):
            qa, qb = pa[k], pb[k]
            if shape_distance(env.robots[a].shape, (float(qa[0]), float(qa[1]), float(qa[2])),
                              env.robots[b].shape, (float(qb[0]), float(qb[1]), float(qb[2]))
                              ) < clearance:
                return True
        return False

    def _blockers(r, got):
        """Who stopped `r` from fitting, and WHERE were they standing when they did?

        Three failures look identical from a count and want opposite fixes. A blocker that
        has ARRIVED never moves again, so no delay for `r` can help. A blocker still parked
        at its START has not left yet, so delaying `r` is the fix and the horizon is the
        limit. A blocker that is DRIVING is an ordinary congested crossing. Report which,
        at the delay where `r` is least obstructed.
        """
        best_at, fewest = None, None
        d = 0
        while d + len(tracks[r]) <= horizon_cap:
            hit = [q for q in got if _clash(r, d, q, got[q])]
            if fewest is None or len(hit) < len(fewest):
                best_at, fewest = d, hit
                if not hit:
                    break
            d += step
        where = {"at_goal": 0, "at_start": 0, "driving": 0}
        for q in (fewest or []):
            ta, tb = tracks[r], tracks[q]
            span = max(best_at + len(ta), got[q] + len(tb))
            ia = np.clip(np.arange(span) - best_at, 0, len(ta) - 1)
            ib = np.clip(np.arange(span) - got[q], 0, len(tb) - 1)
            pa, pb = ta[ia], tb[ib]
            near = np.flatnonzero(np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
                                  < radii[r] + radii[q] + clearance)
            k = int(near[0]) if len(near) else 0
            if k >= got[q] + len(tb):
                where["at_goal"] += 1
            elif k < got[q]:
                where["at_start"] += 1
            else:
                where["driving"] += 1
        return {"blocked_robot": int(r), "blockers": len(fewest or []), **where}

    def _fit(r, got, skip=None):
        """Least delay at which r clears everything in `got`, or None."""
        d = 0
        while d + len(tracks[r]) <= horizon_cap:
            if not any(_clash(r, d, q, got[q]) for q in got if q != skip):
                return d
            d += step
        return None

    def _place(order, why=None, budget=None):
        """Insert in this order, each robot delayed the least that clears those before it.

        Insertion is greedy and has no backtracking, so one badly-seated robot can leave a
        later one with no delay at all. Measured on cluttered_cross_32, that later robot had
        exactly ONE blocker -- so allow a bounded repair: evict the single blocker, seat the
        robot that could not fit, and re-seat the blocker afterwards. Each robot may be
        evicted `evictions` times, which bounds the work and keeps the loop finite. Measured
        on cluttered_cross_32: no repair seats 21 of 32, one eviction each seats 28.
        """
        got: dict = {}
        evicted: dict = {}
        if budget is None:
            budget = int(params.get("evictions", 3))
        queue = list(order)
        while queue:
            r = queue.pop(0)
            d = _fit(r, got)
            if d is not None:
                got[r] = d
                continue
            # Who blocks r at every delay? Take the delay where r is least obstructed: if
            # exactly one robot stands in the way there, that one is the repairable blocker.
            # (Scanning delays once beats testing every placed robot for removability --
            # the latter is O(placed^2 * delays) and does not finish.)
            best_at, fewest = None, None
            d = 0
            while d + len(tracks[r]) <= horizon_cap:
                hit = [q for q in got if _clash(r, d, q, got[q])]
                if fewest is None or len(hit) < len(fewest):
                    best_at, fewest = d, hit
                    if len(hit) == 1:
                        break
                d += step
            if fewest is not None and len(fewest) == 1 and evicted.get(fewest[0], 0) < budget:
                q = fewest[0]
                evicted[q] = evicted.get(q, 0) + 1
                del got[q]
                got[r] = best_at
                queue.append(q)
                continue
            if why is not None:
                why.update(_blockers(r, got))
            return None, len(got)
        return got, len(got)

    # Insertion can only ever DELAY, and delaying is the wrong move when a robot has to go
    # EARLY: in a swap, A's goal is B's start, so B must be gone before A arrives. Placed in
    # an unlucky order, B's only lever makes things worse and the greedy run dead-ends --
    # circular_cross_32 placed 1 of 32. Order is a heuristic, not a commitment, so try
    # several and keep the first that seats everyone. Longest-first and shortest-first are
    # the two structured guesses; the rest are seeded shuffles, so this is reproducible.
    rng = np.random.default_rng(int(params.get("order_seed", 0)))
    by_len = sorted(range(n), key=lambda r: -len(tracks[r]))
    # Longest-first stays the FIRST order tried: the goal-precedence order is a repair for
    # scenarios greed strands, and leading with it costs makespan where greed already works
    # -- circular_cross_32 went 1212 -> 1300 steps (its whole budget) when it led.
    orders = [by_len, _goal_order(refs, starts, goals, radii[0] * 2 + clearance),
              by_len[::-1]]
    orders += [list(rng.permutation(n)) for _ in range(int(params.get("order_tries", 6)))]

    # Eviction rescues orders that greed alone cannot seat, but the schedule it rescues is
    # not necessarily good: on circular_cross_32 it let the FIRST order succeed at 1300
    # steps -- the entire budget -- where an order further down the list had been winning at
    # 1212. So sweep every order with no evictions first, and only fall back to the repair
    # for scenarios where nothing seats without it.
    delay, best, why = None, 0, {}
    for allowance in (0, int(params.get("evictions", 3))):
        for od in orders:
            probe: dict = {}
            delay, got = _place([int(r) for r in od], probe, budget=allowance)
            if got > best:
                best, why = got, probe
            if delay is not None:
                break
        if delay is not None:
            break
    if delay is None:
        if isinstance(params, dict):
            params.setdefault("_reject", {}).update(
                {"unschedulable_robot": 1, "orders_tried": len(orders),
                 "best_placed": best, "n": n, "offset_dropped": dropped, **why})
        return None

    T = max(delay[r] + len(tracks[r]) for r in range(n))
    held = [np.vstack([np.repeat(starts[r][None, :], delay[r], axis=0), tracks[r]])
            if delay[r] else tracks[r] for r in range(n)]
    heldu = [np.vstack([np.zeros((delay[r], 2)), ctrls[r]]) if delay[r] else ctrls[r]
             for r in range(n)]
    tracks = [np.vstack([t, np.repeat(t[-1][None, :], T - len(t), axis=0)])
              if len(t) < T else t for t in held]
    ctrls = [np.vstack([c, np.zeros((T - len(c), 2))]) if len(c) < T else c for c in heldu]

    rep: dict = {}
    gap = _verify(env, tracks, clearance, rep)
    if gap is None:
        if isinstance(params, dict):
            params.setdefault("_reject", {}).update(rep)
        return None
    info = {"pairs": len(pairs), "lanes": lanes, "lane_pitch": round(pitch, 3),
            "hubs": len(hubs), "offset_dropped": dropped,
            "orbiting": len(orbit_of),
            "hub_radius": round(max([h[1] for h in hubs], default=0.0), 3),
            "drive": "smooth" if smooth > 0.0 else "legs",
            "delayed": sum(1 for d in delay.values() if d),
            "max_delay": max(delay.values()), "steps": T,
            "min_surface_gap": round(float(gap), 4)}
    return tracks, ctrls, info
