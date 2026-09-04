#!/usr/bin/env python3
"""Gate: is a CONVEX inner approximation of unicycle2_v0 tight enough to be useful?

Decoupled convex planners for multi-robot teams -- RLSS (arXiv:2302.12863), LSC
(arXiv:2109.09041) -- get their speed and their lack of a coupled fallback from one
property: every subproblem is a QP. They reach that by planning in differentially-flat
output space and bounding DERIVATIVE MAGNITUDES. RLSS says so directly (Sec. 3):

    "Note that we do not model robot orientation. If a robot can rotate, the collision
     shape should contain the union of spaces occupied by the robot for each possible
     orientation at a given position; which is minimally a hypersphere ..."

Bounding ||p'|| and ||p''|| is NOT the nonholonomic constraint. The turn rate is

    omega = (x' y'' - y' x'') / ||p'||^2                                        (exact)

a nonconvex function of the flat outputs, not a derivative magnitude. Quadrotors have no
curvature limit and differential-drive robots turn in place, so it never bit those papers.
dynobench unicycle2_v0 has v_max 0.5 and omega_max 0.5 -- a hard turning radius of 1.0 m.

So before building anything multi-robot, answer one question with one robot:

    Can a convex program produce a LATERAL DETOUR -- the shape a separating hyperplane
    against another robot would force -- that the real RK4 unicycle executes within its
    own omega_max, and how much detour does convexity cost versus the true limit?

Method. M Bezier segments, C2-continuous, over a fixed duration. Bezier control points
give the convex-hull property, so bounding CONTROL POINTS bounds the whole CONTINUOUS
curve, not just sampled knots (already stronger than K-ARC Eq. 6, which constrains at
sampled indices only). Constraints are sound inner approximations, detailed at each
builder below. The solved trajectory is then converted back to (theta, v, omega, a) with
the EXACT formulas above -- no small-angle anything -- and rolled through the real
Unicycle2Model.step. Every number reported is measured on the exact recovery, so the
convexification's error has nowhere to hide.

    python scripts/convex_gate.py [--detours 0 0.2 ... ] [--segments 4] [--degree 5]

CPU only, no new dependencies (CasADi is already required for approach=planning).
"""
from __future__ import annotations

import argparse
import csv
import sys
from math import comb
from pathlib import Path

import casadi as ca
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.robot import build_robot, load_robot_cfg  # noqa: E402

# Sound inner approximation of a disc of radius r by a regular K-gon: halfspaces
# v . n_k <= r*cos(pi/K) give an INSCRIBED polygon, whose circumradius is exactly r.
# So every feasible point satisfies ||v|| <= r (sound), and the worst-case tightness
# is cos(pi/K) -- 1.9% at K=16. Using r instead of r*cos(pi/K) would circumscribe the
# disc and admit ||v|| up to r/cos(pi/K), which is NOT sound.
N_FACETS = 16
# qrqp is an active-set solver and needs a positive-DEFINITE Hessian; a minimum-jerk
# objective is only semi-definite (rigid shifts of the control points cost nothing), so a
# tiny Tikhonov term makes it solvable without meaningfully changing the optimum.
REG = 1e-6


def facets(k: int = N_FACETS):
    ang = np.arange(k) * (2 * np.pi / k)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1), np.cos(np.pi / k)


def bernstein(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a Bezier with control points ``ctrl`` (k+1, dim) at ``u`` in [0, 1]."""
    k = ctrl.shape[0] - 1
    basis = np.stack([comb(k, i) * u**i * (1 - u) ** (k - i) for i in range(k + 1)])
    return basis.T @ ctrl


def deriv_ctrl(ctrl, h: float):
    """Control points of the derivative of a Bezier defined over a span of length ``h``."""
    k = ctrl.shape[0] - 1
    return (k / h) * (ctrl[1:] - ctrl[:-1])


def v_floor(model, dist: float, duration: float, n_seg: int, frac: float) -> np.ndarray:
    """A speed floor the program is REQUIRED to meet, hence guaranteed, hence usable.

    Trapezoid that ramps at a_max to `frac * v_max`, holds, and ramps down; the value
    returned per segment is its slowest end. Zero at both ends, which forces the lateral
    profile straight while starting and stopping -- correct, since a path-based
    formulation cannot express turning in place.

    Why the bang-bang profile cannot be used here. It is not a lower bound on anything:
    with a time budget above the minimum the robot may legitimately travel slower, and
    its peak of v_max also contradicts the polyhedral speed cap (v_max * cos(pi/K) is
    strictly smaller), which makes the program infeasible outright.

    This is where the formulation's conservatism actually lives. The turn-rate bound needs
    a LOWER bound on speed, but the true turn rate divides by the ACTUAL speed, so the
    bound is loose by exactly the ratio between them. Raising `frac` tightens the turn
    bound and buys detour, until the floor itself becomes infeasible against the distance
    the robot has time to cover. That trade is the number this gate reports.
    """
    # Fraction of the AVERAGE speed the traverse allows, not of v_max. The floor forces
    # distance: holding v_lo for the whole duration covers about v_peak * duration, which
    # must stay under `dist` or the program is infeasible no matter what else it asks for.
    # Expressing it against dist/duration makes `frac` mean the same thing at every time
    # budget, so the slack sweep below measures the formulation rather than this scaling.
    v_peak = min(frac * dist / duration, model.v_max)
    t_ramp = v_peak / model.a_max
    edges = np.linspace(0.0, duration, n_seg + 1)
    v = np.minimum(v_peak, model.a_max * np.minimum(edges, duration - edges))
    v = np.maximum(v, 0.0)
    return np.minimum(v[:-1], v[1:]) if t_ramp < duration / 2 else np.zeros(n_seg)


def solve(model, dist: float, detour: float, duration: float, n_seg: int, degree: int,
          window: float = 0.25, cont: int = 3, floor_frac: float = 0.4,
          v_lo_override=None):
    """Minimum-jerk Bezier spline from rest to rest, forced ``detour`` metres sideways.

    Frenet frame: `s` runs start -> goal, `l` is lateral. Returns the control points as
    ``(n_seg, degree+1, 2)`` in (s, l), or None if the QP is infeasible.
    """
    h = duration / n_seg
    d = degree
    nv = n_seg * (d + 1) * 2
    P = ca.SX.sym("P", nv)

    def cp(m):                       # (d+1, 2) symbolic control points of segment m
        blk = P[m * (d + 1) * 2:(m + 1) * (d + 1) * 2]
        return ca.reshape(blk, 2, d + 1).T

    def dctrl(c, order):
        for k in range(order):
            deg = d - k
            c = (deg / h) * (c[1:, :] - c[:-1, :])
        return c

    g, lbg, ubg = [], [], []

    def eq(expr):
        g.append(expr); lbg.append(0.0); ubg.append(0.0)

    def leq(expr, hi):
        g.append(expr); lbg.append(-ca.inf); ubg.append(hi)

    segs = [cp(m) for m in range(n_seg)]

    # ── Boundary: pinned endpoints, at rest, zero acceleration (a clean stop) ────
    eq(segs[0][0, 0]); eq(segs[0][0, 1])                  # start at (0, 0)
    eq(segs[-1][d, 0] - dist); eq(segs[-1][d, 1])         # end at (dist, 0)
    for expr in (dctrl(segs[0], 1)[0, :], dctrl(segs[0], 2)[0, :],
                 dctrl(segs[-1], 1)[-1, :], dctrl(segs[-1], 2)[-1, :]):
        eq(expr[0]); eq(expr[1])

    # ── Continuity at the joints; C3 by default, and that is not cosmetic ───────
    # alpha = d(omega)/dt depends on the THIRD derivative of position. A C2 spline has a
    # discontinuous third derivative at every knot, so omega-dot JUMPS there and alpha is
    # impulsive no matter what is imposed inside a segment. Measured on this instance:
    # at C2 the per-segment jerk bound holds and the exact alpha still reaches 1.57 with
    # 16 segments (6.3x the limit), and it gets WORSE with more segments because there are
    # more knots to jump at. C3 removes the jumps and makes the alpha bound meaningful.
    # RLSS and LSC impose C2 -- enough for a quadrotor, not for a bounded-alpha unicycle.
    for m in range(n_seg - 1):
        a_, b_ = segs[m], segs[m + 1]
        eq(a_[d, 0] - b_[0, 0]); eq(a_[d, 1] - b_[0, 1])
        for order in range(1, cont + 1):
            va, vb = dctrl(a_, order)[-1, :], dctrl(b_, order)[0, :]
            eq(va[0] - vb[0]); eq(va[1] - vb[1])

    # ── The separating-hyperplane detour ────────────────────────────────────────
    # A halfspace over the MIDDLE `window` fraction of segments, applied to their control
    # points; the convex hull property then forces the whole continuous arc of those
    # segments to the required side. This is the shape an inter-robot separating hyperplane
    # imposes, and the only thing here that demands curvature. It covers a window rather
    # than every interior segment so the robot has room to ease in and out -- constraining
    # all of them would force the entire lateral offset to be built inside segment 0, which
    # the turn-rate bound forbids, and would measure the window choice instead of the
    # convexification.
    lo = int(round(n_seg * (0.5 - window / 2)))
    hi = int(round(n_seg * (0.5 + window / 2)))
    for m in range(max(lo, 1), min(hi, n_seg - 1)):
        for i in range(d + 1):
            leq(-segs[m][i, 1], -detour)                  # l >= detour

    # ── Speed: sound polyhedral inner approximation of the disc ||v|| <= v_max ──
    nrm, shrink = facets()
    for m in range(n_seg):
        V = dctrl(segs[m], 1)
        for i in range(V.shape[0]):
            for k in range(nrm.shape[0]):
                leq(nrm[k, 0] * V[i, 0] + nrm[k, 1] * V[i, 1], model.v_max * shrink)

    v_lo_all = (v_floor(model, dist, duration, n_seg, floor_frac)
                if v_lo_override is None else np.asarray(v_lo_override, float))
    # Never above what the polyhedral speed cap admits, or the program is infeasible.
    v_lo_all = np.clip(v_lo_all, 0.0, model.v_max * shrink)

    # ── Enforce the speed floor the turn-rate bound relies on ───────────────────
    # Without this the v_lo above would be an assumption about the solution rather than a
    # property of it, and the soundness argument in v_floor would not close.
    for m in range(n_seg):
        V = dctrl(segs[m], 1)
        for i in range(V.shape[0]):
            leq(-V[i, 0], -float(v_lo_all[m]))        # longitudinal speed >= v_lo

    # ── Tangential acceleration and turn rate ───────────────────────────────────
    # |s''| <= a_max stands in for |v'| <= a_max, exact when lateral motion is small.
    # For the turn rate, in the Frenet frame omega = (s' l'' - l' s'')/s'^2 with leading
    # term l''/s', so |l''| <= omega_max * v_ref bounds it. v_ref is a nominal speed
    # profile, floored at v_ref_floor so the constraint stays finite through the rest-to-
    # rest ends. BOTH are approximations -- which is precisely what this gate measures:
    # the reported omega is recomputed from the EXACT formula after solving.
    for m in range(n_seg):
        A = dctrl(segs[m], 2)
        v_ref = float(v_lo_all[m])
        for i in range(A.shape[0]):
            leq(A[i, 0], model.a_max); leq(-A[i, 0], model.a_max)
            leq(A[i, 1], model.omega_max * v_ref); leq(-A[i, 1], model.omega_max * v_ref)

    # ── Angular acceleration: the bound the flat/convex literature omits ─────
    # unicycle2 bounds alpha = d(omega)/dt, not just omega. Differentiating the Frenet
    # relation omega ~ (2nd derivative of l)/(speed) once more gives
    # alpha ~ (3rd derivative of l)/(speed), so bounding the LATERAL JERK control points
    # by alpha_max * v_ref bounds it -- and that is LINEAR here, because the third
    # derivative of a Bezier is again a Bezier in the same variables.
    #
    # RLSS-style "bound the k-th derivative magnitude" cannot express this: it is a bound
    # on a RATIO of derivatives, not on a magnitude. And it is what actually binds for
    # this robot -- measured without it, a 0.9 m detour keeps the turn rate legal
    # (0.52 vs 0.5) while alpha reaches 1.14, which is 4.6x over the limit.
    for m in range(n_seg):
        J = dctrl(segs[m], 3)
        v_ref = float(v_lo_all[m])
        for i in range(J.shape[0]):
            leq(J[i, 1], model.alpha_max * v_ref)
            leq(-J[i, 1], model.alpha_max * v_ref)

    # ── Objective: minimum jerk (a QP; keeps the curve smooth and short) ────────
    # Normalised. A jerk control point carries a factor d(d-1)(d-2)/h^3, so the Hessian
    # in the control points scales as 1/h^6 and its entries grow ~6x each time the segment
    # count doubles. Left unscaled this wrecks the conditioning: qrqp reports "failed to
    # calculate search direction" at 16 segments on a problem proxqp solves fine, which
    # reads as infeasibility and is not. Scaling an objective by a positive constant leaves
    # the optimum untouched, so this only buys conditioning.
    scale = (h ** 3 / (d * (d - 1) * (d - 2))) ** 2
    obj = 0
    for m in range(n_seg):
        J = dctrl(segs[m], 3)
        for i in range(J.shape[0]):
            obj += J[i, 0] ** 2 + J[i, 1] ** 2
    obj *= scale

    qp = {"x": P, "f": obj + REG * ca.sumsqr(P), "g": ca.vertcat(*g)}
    # proxqp, not CasADi's own qrqp. qrqp is a dense-ish active-set method and this
    # Hessian is inherently ill-conditioned (a jerk control point carries d(d-1)(d-2)/h^3,
    # so the Hessian scales as 1/h^6); measured, qrqp fails at 16 segments on a problem
    # proxqp solves, and worse, at some sizes it RETURNS a point that violates the speed
    # constraints without reporting failure. selfcheck() asserts against exactly that.
    S = ca.qpsol("S", "proxqp", qp,
                 {"print_time": False, "error_on_fail": False,
                  "proxqp": {"verbose": False, "eps_abs": 1e-9, "max_iter": 20000}})
    # Warm start along the straight line so the active-set solver has a sane basis.
    x0 = np.zeros(nv)
    for m in range(n_seg):
        for i in range(d + 1):
            frac = (m + i / d) / n_seg
            x0[(m * (d + 1) + i) * 2] = dist * frac
    r = S(x0=x0, lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
    if not S.stats().get("success", False):
        return None
    return np.array(r["x"]).reshape(n_seg, d + 1, 2)


def solve_scp(model, dist, detour, duration, n_seg, degree, window, cont, floor_frac,
              iters: int = 8, damping: float = 0.5, tol: float = 1e-3):
    """Sequential convex programming on the speed the turn-rate bound divides by.

    A single solve has to bound the turn rate using a speed floor guaranteed BEFORE the
    trajectory exists, while the true turn rate divides by the speed the trajectory
    actually reaches. That ratio is the entire conservatism -- measured at roughly 5x on
    the swap2 traverse. Here the floor is re-estimated from the previous iterate instead:
    per segment, the minimum longitudinal speed that iterate attained.

    Soundness still closes AT THE FIXED POINT, and only there. The program enforces
    longitudinal speed >= v_lo, so |omega| ~ |2nd derivative of l| / speed
    <= omega_max * v_lo / speed <= omega_max whenever speed >= v_lo. Away from the fixed
    point v_lo is a guess about a trajectory that does not exist yet and guarantees
    nothing, which is why the caller still checks the EXACT recovery against every
    actuation bound. That check is what decides `executable`, not this loop.

    Damped, because undamped the floor chases its own tail: raising v_lo relaxes the turn
    bound, which buys a wigglier path, which lowers the attained speed, which lowers v_lo.

    Returns (control points, iterations used) or (None, k) if it never solved.
    """
    v_lo = v_floor(model, dist, duration, n_seg, floor_frac)
    best_ctrl = None
    for k in range(1, iters + 1):
        ctrl = solve(model, dist, detour, duration, n_seg, degree, window, cont,
                     floor_frac, v_lo_override=v_lo)
        if ctrl is None:
            return best_ctrl, k
        best_ctrl = ctrl
        rec = recover(ctrl, duration)
        # Per-segment minimum of the longitudinal speed actually attained. The minimum,
        # not the mean: the bound has to hold everywhere inside the segment.
        per = len(rec["t"]) // n_seg
        attained = np.array([rec["v_s"][m * per:(m + 1) * per].min() for m in range(n_seg)])
        attained = np.maximum(attained, 0.0)
        nxt = v_lo + damping * (attained - v_lo)
        if np.max(np.abs(nxt - v_lo)) < tol:
            return ctrl, k
        v_lo = nxt
    return best_ctrl, iters


def recover(ctrl, duration: float, n_per_seg: int = 200):
    """Exact (t, p, theta, v, omega, a) from the spline. No small-angle approximation."""
    n_seg, dp1, _ = ctrl.shape
    h = duration / n_seg
    u = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
    P, V, A = [], [], []
    for m in range(n_seg):
        c = ctrl[m]
        P.append(bernstein(c, u))
        V.append(bernstein(deriv_ctrl(c, h), u))
        A.append(bernstein(deriv_ctrl(deriv_ctrl(c, h), h), u))
    p = np.concatenate(P); v = np.concatenate(V); acc = np.concatenate(A)
    t = np.linspace(0.0, duration, p.shape[0], endpoint=False)

    speed = np.linalg.norm(v, axis=1)
    v_s = v[:, 0]                      # longitudinal component, the one bounded below
    theta = np.arctan2(v[:, 1], v[:, 0])
    safe = np.maximum(speed, 1e-9)
    omega = (v[:, 0] * acc[:, 1] - v[:, 1] * acc[:, 0]) / safe**2
    a_tan = (v[:, 0] * acc[:, 0] + v[:, 1] * acc[:, 1]) / safe
    # Where the robot is essentially stopped, heading and turn rate are undefined by the
    # flat map (the classic v=0 flatness singularity). Report them separately rather than
    # letting a 1e-9 denominator manufacture a huge omega.
    moving = speed > 0.05 * 0.5
    return dict(t=t, p=p, theta=theta, v=speed, v_s=v_s, omega=omega, a=a_tan,
                moving=moving)


def consistency(model, rec, span: float = 1.0) -> float:
    """Largest position error when the REAL RK4 model replays the recovered controls.

    Deliberately over a short window, starting from a sample where the robot is already
    moving. Two reasons. The flat map is singular at v=0 -- theta and omega are undefined
    there, so seeding an integration at t=0 injects an error that has nothing to do with
    the trajectory. And open-loop integration of a nonholonomic system drifts without
    bound, so a whole-traverse replay measures the absence of a tracking controller, not
    whether the plan is executable. Executability is decided by the actuation bounds in
    `recover`; this only verifies the flat recovery agrees with the model's own dynamics.
    """
    mv = np.flatnonzero(rec["moving"])
    if mv.size < 10:
        return float("nan")
    dt = float(rec["t"][1] - rec["t"][0])
    i0 = int(mv[0])
    n = min(int(span / dt), int(mv[-1]) - i0)
    if n < 2:
        return float("nan")
    alpha = np.gradient(rec["omega"], dt)
    state = np.array([rec["p"][i0, 0], rec["p"][i0, 1], rec["theta"][i0],
                      rec["v"][i0], rec["omega"][i0]])
    dev = 0.0
    for i in range(i0, i0 + n):
        state = model.step(state, [rec["a"][i], alpha[i]], dt)
        dev = max(dev, float(np.linalg.norm(state[:2] - rec["p"][i + 1])))
    return dev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="unicycle_db")
    ap.add_argument("--dist", type=float, default=3.0, help="swap2 traverse length")
    ap.add_argument("--detours", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1])
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--degree", type=int, default=7)
    ap.add_argument("--continuity", type=int, default=3,
                    help="spline continuity order; 3 is required to bound alpha")
    ap.add_argument("--duration", type=float, default=None)
    # Bounding CONTROL POINTS is conservative: the hull of the acceleration control points
    # over-bounds the acceleration actually attained, so a convex program cannot run at the
    # bang-bang optimum. Same reason conf/approach/planning.yaml carries `slack: 1.5`.
    ap.add_argument("--slack", type=float, default=1.5,
                    help="duration multiplier over the bang-bang minimum time")
    # Default 1. The loop is a documented no-op on THIS formulation, kept because it is
    # the natural thing to reach for and the reason it fails is worth being able to
    # reproduce: minimum jerk under a speed FLOOR rides the floor exactly, so
    # re-estimating the floor from the previous iterate is a fixed point by construction
    # (measured: converges in 2 iterations, detour unchanged at 0.10 m). Linearising the
    # EXACT turn-rate constraint around the previous iterate is a different loop and is
    # not implemented here.
    ap.add_argument("--scp", type=int, default=1,
                    help="SCP iterations on the speed floor; a no-op on this formulation")
    # 0.9 of the average speed the traverse allows. Swept 0.2-0.9 x slack 1.0-3.0; this is
    # the most permissive cell that stays feasible, so the gate quotes the formulation at
    # its best rather than at an arbitrary setting.
    ap.add_argument("--floor-frac", type=float, default=0.9,
                    help="speed floor as a fraction of v_max; raises the turn budget "
                         "but must stay coverable within the time budget")
    ap.add_argument("--window", type=float, default=0.25,
                    help="fraction of the traverse the separating halfspace covers")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    m = build_robot(load_robot_cfg(args.robot))
    rho = m.v_max / m.omega_max
    # Rest-to-rest over `dist` at v_max needs the cruise time plus both ramps.
    bang = args.dist / m.v_max + m.v_max / m.a_max
    dur = args.duration or args.slack * bang
    print(f"robot={args.robot}  v_max={m.v_max}  omega_max={m.omega_max}  a_max={m.a_max}")
    print(f"turning radius rho = v_max/omega_max = {rho:.3f} m")
    print(f"traverse {args.dist} m: bang-bang minimum {bang:.2f} s, "
          f"budget {dur:.2f} s (slack {dur / bang:.2f}x)")
    print(f"{args.segments} x degree-{args.degree} Bezier, C{args.continuity} continuous\n")

    # An S-curve of lateral offset D over longitudinal length L needs peak curvature
    # ~8D/L^2; at v_max that is feasible only while 8D/L^2 <= 1/rho.
    d_theory = args.dist**2 / (8 * rho)
    print(f"theoretical max detour at full speed ~ L^2/(8*rho) = {d_theory:.3f} m\n")

    hdr = ["detour_m", "solved", "scp_iters", "max_v", "max_|a|", "max_|omega|",
           "max_|alpha|", "executable", "recovery_err_m"]
    print("  ".join(h.rjust(15) for h in hdr))
    rows, best = [], None
    for det in args.detours:
        ctrl, n_it = solve_scp(m, args.dist, det, dur, args.segments, args.degree,
                               args.window, args.continuity, args.floor_frac, args.scp)
        if ctrl is None:
            rows.append(dict.fromkeys(hdr, ""))
            rows[-1].update(detour_m=det, solved=False)
            print("  ".join(str(rows[-1][h]).rjust(15) for h in hdr))
            continue
        rec = recover(ctrl, dur)
        dt = float(rec["t"][1] - rec["t"][0])
        mov = rec["moving"]
        mv = float(rec["v"].max())
        ma = float(np.abs(rec["a"][mov]).max())
        mo = float(np.abs(rec["omega"][mov]).max())
        mal = float(np.abs(np.gradient(rec["omega"], dt)[mov]).max())
        # Executable iff EVERY actuation bound of the real robot holds on the exact
        # recovery. alpha_max matters as much as omega_max: a turn rate inside the limit
        # that has to be reached instantly is just as infeasible as one that is too large.
        ok = (mv <= m.v_max + 1e-6 and ma <= m.a_max + 1e-6
              and mo <= m.omega_max + 1e-6 and mal <= m.alpha_max + 1e-6)
        if ok:
            best = det
        row = {"detour_m": det, "solved": True, "scp_iters": n_it,
               "max_v": round(mv, 4),
               "max_|a|": round(ma, 4), "max_|omega|": round(mo, 4),
               "max_|alpha|": round(mal, 4), "executable": ok,
               "recovery_err_m": round(consistency(m, rec), 5)}
        rows.append(row)
        print("  ".join(str(row[h]).rjust(15) for h in hdr))

    selfcheck(m, args, dur, rows)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "convex_gate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

    # Feasibility must shrink as the detour grows. Where it does not, the solver -- not
    # the geometry -- decided, and the threshold below is approximate. Reported rather
    # than asserted: the verdict here spans an order of magnitude and does not turn on
    # which side of one grid point the boundary sits.
    seq = [bool(r["solved"]) for r in rows]
    flips = [rows[i]["detour_m"] for i in range(1, len(seq)) if seq[i] and not seq[i - 1]]
    if flips:
        print(f"\nWARNING: feasibility is not monotone in the detour (solved again at "
              f"{flips}).")
        print("Residual solver flakiness; treat the threshold below as approximate.")

    need = 2 * m.shape.bounding_radius
    print(f"\nlargest detour executable within ALL actuation bounds: "
          f"{'none' if best is None else f'{best:.2f} m'}")
    print(f"unconstrained manoeuvre range L^2/(8*rho):                {d_theory:.2f} m")
    print(f"separation a real pair needs (2 x bounding radius):       {need:.2f} m")

    if best is None:
        print("\nGATE FAILS: the convex program cannot produce any executable detour.")
    elif best >= need:
        print(f"\nGATE PASSES: {best:.2f} m clears the {need:.2f} m a separating hyperplane "
              "must buy.")
    else:
        print(f"\nGATE FAILS: {best:.2f} m is {100 * best / need:.0f}% of the {need:.2f} m "
              "an inter-robot separating hyperplane has to produce,")
        print(f"and {100 * best / d_theory:.0f}% of what the robot can physically do. The "
              "convexification, not the robot, is the binding constraint.")
    return 0


def selfcheck(model, args, dur, rows) -> None:
    """Refuse to trust the table unless the machinery itself is verified."""
    solved = [r for r in rows if r["solved"]]
    if not solved:
        # A whole sweep coming back infeasible is a legitimate outcome at aggressive
        # settings (a high speed floor forces more distance than the time budget allows),
        # so report it rather than raising -- the remaining checks need a solution to
        # inspect and simply have nothing to say here.
        print("\nnothing solved at these settings; the speed floor most likely forces "
              "more distance than the duration allows.")
        return

    # 1. The straight case must solve and need essentially no turning.
    zero = [r for r in rows if r["solved"] and r["detour_m"] == 0.0]
    if zero:
        assert zero[0]["max_|omega|"] < 1e-3, \
            f"straight traverse needs omega={zero[0]['max_|omega|']} -- frame or recovery is wrong"

    # 2. Bezier derivative control points must agree with finite differences of the curve.
    ref = max(r["detour_m"] for r in solved)
    ctrl, _ = solve_scp(model, args.dist, ref, dur, args.segments, args.degree,
                        args.window, args.continuity, args.floor_frac, args.scp)
    assert ctrl is not None, f"reference instance (detour {ref}) stopped solving"
    rec = recover(ctrl, dur, n_per_seg=400)
    fd = np.gradient(rec["p"], rec["t"], axis=0)
    err = np.abs(np.linalg.norm(fd, axis=1) - rec["v"])[5:-5].max()
    assert err < 1e-3, f"analytic derivative disagrees with finite difference by {err:.2e}"

    # 3. Sound inner approximation: the solved speed must never exceed v_max.
    for r in solved:
        assert r["max_v"] <= model.v_max + 1e-6, \
            f"speed {r['max_v']} exceeds v_max -- the polygon is circumscribed, not inscribed"


if __name__ == "__main__":
    raise SystemExit(main())
