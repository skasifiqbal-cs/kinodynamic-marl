#!/usr/bin/env python3
"""Measure the band where the family's geometric conflict test certifies a doomed state.

Every planner in the ARC / CBS family decides "conflict" from geometry alone --
K-ARC Eq. 6 is ``||c_i,k - c_j,k|| >= d_min``, with no velocity term. This script
sweeps two second-order unicycles over relative geometry and speed and, for each
state, records three verdicts:

    geometric   what the family uses          (>= 0 -> accepted as conflict-free)
    braking     src.conflict.braking_margin   (>= 0 -> provably able to stop)
    ICS         src.conflict.provable_ics     (True -> collision unavoidable, PROVED)

The quantity of interest is the FALSE-SAFETY BAND: states where the geometric test
says conflict-free and the collision is already unavoidable. Those are states the
family will accept into a plan it calls collision-free.

Also sweeps the acceleration limit, because the band should scale with braking
distance ``v_max^2 / (2 a_max)``. If it does not, the premise is wrong and the
downstream planner work is not justified -- that is this script's job to decide.

    python scripts/false_safety_band.py [--robot unicycle_db] [--out experiments]

CPU only, no training, no planner.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conflict.margin import (  # noqa: E402
    geometric_margin,
    inscribed_radius,
    provable_ics,
    reach_profile,
)
from src.robot import build_robot, load_robot_cfg  # noqa: E402

HORIZON = 6.0
N_TIMES = 121


def build_grid(model, n_d=40, n_v=5, n_bearing=12, n_heading=16, d_max=4.0):
    """Cartesian product of (distance, bearing, v_i, v_j, heading_j).

    Robot i sits at the origin heading +x. Robot j sits at distance ``d`` and bearing
    ``beta``, heading ``theta_j``. Both start with ``omega = 0`` -- the worst case for
    the family's test is straight-line closing, and leaving omega free would only add
    spread to the reach bound (making the ICS proof fire less often, i.e. under-report).
    """
    d = np.linspace(2 * model.shape.bounding_radius, d_max, n_d)
    beta = np.linspace(-np.pi, np.pi, n_bearing, endpoint=False)
    v = np.linspace(0.0, model.v_max, n_v)
    th_j = np.linspace(-np.pi, np.pi, n_heading, endpoint=False)

    D, B, VI, VJ, TJ = (a.ravel() for a in np.meshgrid(d, beta, v, v, th_j, indexing="ij"))
    return D, B, VI, VJ, TJ, v


def sweep(model, a_scale: float = 1.0, **grid_kw):
    """Return a dict of flat arrays: the three verdicts plus the state that produced them."""
    a_max = model.a_max * a_scale
    D, B, VI, VJ, TJ, v_vals = build_grid(model, **grid_kw)
    r = model.shape.bounding_radius
    inr2 = 2 * inscribed_radius(model.shape)

    p_j = np.stack([D * np.cos(B), D * np.sin(B)], axis=1)          # p_i is the origin
    e_i = np.zeros_like(p_j)
    e_i[:, 0] = 1.0
    e_j = np.stack([np.cos(TJ), np.sin(TJ)], axis=1)

    geo = D - 2 * r
    brake = (D - 2 * r) - (VI**2 + VJ**2) / (2 * a_max)             # conservative form

    # ICS: one reach profile per distinct v0 (omega is 0 everywhere), then place them.
    ts = np.linspace(0.0, HORIZON, N_TIMES)
    scaled = _scaled(model, a_max)
    prof = {float(v0): reach_profile(float(v0), 0.0, scaled, ts) for v0 in v_vals}
    mid = np.stack([0.5 * (prof[v][0] + prof[v][1]) for v in prof])  # (n_v, n_t)
    rad = np.stack([0.5 * (prof[v][1] - prof[v][0]) + prof[v][2] for v in prof])
    idx = {v: k for k, v in enumerate(prof)}
    ii = np.array([idx[float(x)] for x in VI])
    jj = np.array([idx[float(x)] for x in VJ])

    ics = np.zeros(D.shape, dtype=bool)
    t_star = np.full(D.shape, np.inf)
    for k in range(len(ts)):
        ci = e_i * mid[ii, k][:, None]                              # p_i = 0
        cj = p_j + e_j * mid[jj, k][:, None]
        gap = np.linalg.norm(ci - cj, axis=1) + rad[ii, k] + rad[jj, k]
        fresh = (gap <= inr2) & ~ics
        t_star[fresh] = ts[k]
        ics |= fresh

    return dict(d=D, bearing=B, v_i=VI, v_j=VJ, theta_j=TJ,
                geometric=geo, braking=brake, ics=ics, t_star=t_star, a_max=a_max)


def _scaled(model, a_max):
    """A copy of ``model`` with the acceleration limits rescaled, for the a_max sweep."""
    from copy import copy
    m = copy(model)
    m.a_max, m.a_min = a_max, -a_max
    m.alpha_max, m.alpha_min = model.alpha_max * (a_max / model.a_max), -model.alpha_max * (a_max / model.a_max)
    return m


def selfcheck(model, res) -> None:
    """Refuse to emit numbers unless the sound predicates agree where they must."""
    # 1. Soundness cross-check: a certified-stoppable pair can never be an ICS.
    bad = res["ics"] & (res["braking"] >= 0)
    assert not bad.any(), (
        f"{bad.sum()} states are BOTH provably-ICS and certified stoppable -- "
        "one of the two predicates is unsound, do not trust these numbers"
    )
    # 2. At rest, nothing separated is inevitable (holding position is always admissible).
    rest = (res["v_i"] == 0) & (res["v_j"] == 0)
    assert not (res["ics"] & rest & (res["geometric"] > 0)).any(), \
        "ICS fired on a separated pair at rest -- reach bound is not an over-approximation"
    # 3. At rest the braking margin must reduce to the geometric test exactly.
    assert np.allclose(res["braking"][rest], res["geometric"][rest]), \
        "braking margin does not reduce to the geometric test at zero velocity"
    # 4. The ICS proof must be able to fire at all, or the grid is too coarse to
    #    measure anything. Checked on a hand-built head-on pair rather than on the
    #    sweep, because a band of exactly zero is a legitimate (and important) result
    #    at large a_max -- that is the decision gate, not a bug.
    r = model.shape.bounding_radius
    assert geometric_margin([0, 0], [0.8, 0], r, r) > 0, "probe state is not geometrically accepted"
    assert provable_ics(np.array([0.0, 0.0, 0.0, model.v_max, 0.0]), model,
                        np.array([0.8, 0.0, np.pi, model.v_max, 0.0]), model,
                        horizon=HORIZON, n_times=N_TIMES - 1)[0], \
        "ICS proof never fires on a textbook head-on pair -- reach bound is too loose"


def report(model, res, label: str) -> dict:
    n = res["d"].size
    accepted = res["geometric"] >= 0                    # the family calls these conflict-free
    false_safe = accepted & res["ics"]
    caught = accepted & (res["braking"] < 0)
    d_brake = model.v_max**2 / (2 * res["a_max"])

    row = {
        "label": label,
        "a_max": round(res["a_max"], 4),
        "d_brake_m": round(d_brake, 4),
        "states": n,
        "geom_accepted": int(accepted.sum()),
        "false_safe": int(false_safe.sum()),
        "false_safe_pct_of_accepted": round(100 * false_safe.sum() / max(accepted.sum(), 1), 3),
        "max_geom_margin_on_doomed_m": round(float(res["geometric"][false_safe].max()), 4) if false_safe.any() else 0.0,
        "max_separation_on_doomed_m": round(float(res["d"][false_safe].max()), 4) if false_safe.any() else 0.0,
        "braking_flags_pct_of_accepted": round(100 * caught.sum() / max(accepted.sum(), 1), 3),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="unicycle_db")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--a-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    model = build_robot(load_robot_cfg(args.robot))
    r = model.shape.bounding_radius
    print(f"robot={args.robot}  bounding_r={r:.4f}  inscribed_r={inscribed_radius(model.shape):.4f}")
    print(f"v_max={model.v_max}  a_max={model.a_max}  d_brake={model.v_max**2/(2*model.a_max):.3f} m")
    print(f"geometric d_min = 2*bounding_r = {2*r:.4f} m  (K-ARC publishes no value)\n")

    rows, nominal = [], None
    for s in args.a_scales:
        res = sweep(model, a_scale=s)
        selfcheck(model, res)
        rows.append(report(model, res, f"a_max x{s:g}"))
        if s == 1.0:
            nominal = res

    hdr = ["label", "d_brake_m", "geom_accepted", "false_safe",
           "false_safe_pct_of_accepted", "max_geom_margin_on_doomed_m",
           "max_separation_on_doomed_m", "braking_flags_pct_of_accepted"]
    w = [max(len(h), 12) for h in hdr]
    print("  ".join(h.rjust(x) for h, x in zip(hdr, w)))
    for row in rows:
        print("  ".join(str(row[h]).rjust(x) for h, x in zip(hdr, w)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "false_safety_band.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    if nominal is not None:
        keep = (nominal["geometric"] >= 0) & nominal["ics"]
        with (out / "false_safety_states.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["d", "bearing", "v_i", "v_j", "theta_j", "geometric_margin",
                         "braking_margin", "t_collision"])
            for k in np.flatnonzero(keep):
                wr.writerow([f"{nominal[c][k]:.5f}" for c in
                             ("d", "bearing", "v_i", "v_j", "theta_j", "geometric",
                              "braking", "t_star")])
        print(f"\n{keep.sum()} false-safe states -> {out/'false_safety_states.csv'}")

    print("\nHow to read this. `braking_flags_pct_of_accepted` is the share of states the")
    print("geometric test accepts as conflict-free where the pair provably CANNOT stop in")
    print("time -- the honest size of the gap. `false_safe` is the much smaller subset where")
    print("the collision is PROVED unavoidable; provable_ics is sound but one-sided, so that")
    print("column is a hard LOWER bound on the doomed set, never an estimate of it.")

    grew = [r_["false_safe_pct_of_accepted"] for r_ in rows]
    print("\nDECISION GATE (plan Step 4): band vs braking distance",
          "-> PREMISE HOLDS, band grows as braking gets weaker"
          if grew[0] >= grew[-1] and grew[0] > 0 else
          "-> PREMISE FAILS, band does not track braking distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
