"""The three claims behind the dynamics-blind-conflict-detection thesis.

Each is a runnable check, not prose. Claim 3 comes out NEGATIVE for the dynobench
robot and is asserted in that direction on purpose -- see its docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.conflict import (
    braking_margin,
    geometric_margin,
    inscribed_radius,
    provable_ics,
    reach_disc,
)
from src.robot import build_robot, load_robot_cfg


@pytest.fixture(scope="module")
def robot():
    return build_robot(load_robot_cfg("unicycle_db"))


def _cart_vel(state):
    return state[3] * np.array([np.cos(state[2]), np.sin(state[2])])


# ── Sanity: the braking margin generalises the geometric test ────────────────

def test_reduces_to_geometric_at_rest(robot):
    """With both robots stopped the braking term vanishes and the two agree exactly."""
    r = robot.shape.bounding_radius
    for d in (0.3, 0.6, 1.2, 3.0):
        p_i, p_j = np.zeros(2), np.array([d, 0.0])
        g = geometric_margin(p_i, p_j, r, r)
        for conservative in (True, False):
            b = braking_margin(p_i, np.zeros(2), r, robot.a_max,
                               p_j, np.zeros(2), r, robot.a_max,
                               conservative=conservative)
            assert b == pytest.approx(g, abs=1e-12)


# ── Claim 1: the conservative margin is SOUND ───────────────────────────────

def test_nonnegative_margin_survives_max_braking(robot):
    """`braking_margin >= 0` => braking maximally keeps the pair separated, forever.

    Randomised: sample pairs, keep those the margin certifies, roll both robots
    forward under full braking with the real RK4 integrator and assert the bodies
    never touch. The conservative form must hold for NON-HOLONOMIC robots, so the
    rollout also steers -- worst-case alpha, both signs -- rather than only braking
    in a straight line.
    """
    rng = np.random.default_rng(0)
    r = robot.shape.bounding_radius
    dt, steps = 0.05, 200
    checked = 0

    for _ in range(400):
        p_i = np.zeros(2)
        d = rng.uniform(0.5, 4.0)
        ang = rng.uniform(-np.pi, np.pi)
        p_j = p_i + d * np.array([np.cos(ang), np.sin(ang)])
        s_i = np.array([*p_i, rng.uniform(-np.pi, np.pi), rng.uniform(robot.v_min, robot.v_max),
                        rng.uniform(robot.omega_min, robot.omega_max)])
        s_j = np.array([*p_j, rng.uniform(-np.pi, np.pi), rng.uniform(robot.v_min, robot.v_max),
                        rng.uniform(robot.omega_min, robot.omega_max)])

        if braking_margin(s_i[:2], _cart_vel(s_i), r, robot.a_max,
                          s_j[:2], _cart_vel(s_j), r, robot.a_max) < 0:
            continue
        checked += 1

        for alpha_i, alpha_j in ((0.0, 0.0), (robot.alpha_max, robot.alpha_min),
                                 (robot.alpha_min, robot.alpha_max)):
            a, b = s_i.copy(), s_j.copy()
            for _ in range(steps):
                # Maximal braking: oppose the current sign of v.
                a = robot.step(a, [-np.sign(a[3]) * robot.a_max, alpha_i], dt)
                b = robot.step(b, [-np.sign(b[3]) * robot.a_max, alpha_j], dt)
                assert np.linalg.norm(a[:2] - b[:2]) >= 2 * r - 1e-9, (
                    f"certified pair collided: {s_i} vs {s_j}"
                )

    assert checked > 50, f"sample too thin to mean anything ({checked} certified pairs)"


# ── Claim 2: the geometric test is NOT a safety certificate ─────────────────

def test_geometric_pass_can_be_a_proven_ics(robot):
    """A state K-ARC Eq. 6 accepts, where collision is unavoidable under ANY controls.

    Head-on at v_max. `provable_ics` is one-sided sound -- it fires only when every
    reachable point of one robot is inside the contact disc of every reachable point
    of the other -- so a True here is a proof, not a heuristic.
    """
    r = robot.shape.bounding_radius
    s_i = np.array([0.0, 0.0, 0.0, robot.v_max, 0.0])
    s_j = np.array([0.8, 0.0, np.pi, robot.v_max, 0.0])

    assert geometric_margin(s_i[:2], s_j[:2], r, r) > 0.2, "geometric test must PASS, clearly"
    is_ics, t_star = provable_ics(s_i, robot, s_j, robot)
    assert is_ics and np.isfinite(t_star)

    # ...and the braking margin catches exactly what the geometric test missed.
    assert braking_margin(s_i[:2], _cart_vel(s_i), r, robot.a_max,
                          s_j[:2], _cart_vel(s_j), r, robot.a_max) < 0


def test_provable_ics_is_one_sided(robot):
    """It must never fire on a genuinely escapable state -- at rest, holding still works."""
    for d in (0.6, 1.0, 2.0):
        s_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        s_j = np.array([d, 0.0, np.pi, 0.0, 0.0])
        assert geometric_margin(s_i[:2], s_j[:2], robot.shape.bounding_radius,
                                robot.shape.bounding_radius) > 0
        assert not provable_ics(s_i, robot, s_j, robot)[0]


def test_reach_disc_contains_real_rollouts(robot):
    """The outer bound must actually over-cover, or the ICS proof is worthless."""
    rng = np.random.default_rng(1)
    dt = 0.02
    for _ in range(20):
        s0 = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi),
                       rng.uniform(robot.v_min, robot.v_max),
                       rng.uniform(robot.omega_min, robot.omega_max)])
        for t in (0.2, 0.6, 1.2, 2.0):
            disc = reach_disc(s0, robot, t)
            n = int(round(t / dt))
            for _ in range(15):
                s = s0.copy()
                a = rng.uniform(robot.a_min, robot.a_max)
                al = rng.uniform(robot.alpha_min, robot.alpha_max)
                for _ in range(n):
                    s = robot.step(s, [a, al], dt)
                assert np.linalg.norm(s[:2] - disc.center) <= disc.radius + 1e-6


# ── Claim 3: the inter-sample gap -- NEGATIVE for this robot ────────────────

def test_intersample_tunnelling_is_impossible_at_planner_dt(robot):
    """Sampled-instant tests can miss a between-knot overlap -- but NOT here.

    A pair can slip through a per-timestep geometric test only if it covers more than
    the full contact diameter within one step. With the dynobench limits the largest
    relative speed is `2*v_max`, so tunnelling needs

        dt > 2*(r_i + r_j) / (2*v_max)

    which lands far above `approach.trajopt.dt_bounds` (max 0.5). Asserted in that
    direction on purpose: the inter-sample argument is real against planners that
    sample coarsely, and it is NOT available for this robot at this dt. Do not put
    it in a paper that uses these parameters. (It is also weaker against K-CBS,
    which tests time INTERVALS rather than instants.)
    """
    r = robot.shape.bounding_radius
    dt_critical = 2 * (2 * r) / (2 * robot.v_max)
    assert dt_critical > 0.5, f"tunnelling now possible at planner dt ({dt_critical=:.3f})"

    # And confirm it directly: worst case (head-on, v_max) at the largest allowed dt.
    dt = 0.5
    s_i = np.array([0.0, 0.0, 0.0, robot.v_max, 0.0])
    s_j = np.array([2 * r + 1e-3 + 2 * robot.v_max * dt, 0.0, np.pi, robot.v_max, 0.0])
    sep = [np.linalg.norm(s_i[:2] - s_j[:2])]
    for k in range(1, 21):                       # sub-sample one planner step
        f = k / 20 * dt
        a = s_i[:2] + f * robot.v_max * np.array([1.0, 0.0])
        b = s_j[:2] - f * robot.v_max * np.array([1.0, 0.0])
        sep.append(np.linalg.norm(a - b))
    assert min(sep) >= 2 * r - 1e-9, "unexpected inter-sample overlap"


def test_inscribed_radius_is_sound(robot):
    """Inscribed <= bounding, and the box value is the half-minor-extent."""
    assert inscribed_radius(robot.shape) == pytest.approx(0.125)
    assert inscribed_radius(robot.shape) < robot.shape.bounding_radius
