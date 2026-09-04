"""Trajectory-optimisation planner: dynamic optimality and collision-free execution."""
import os

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

pytest.importorskip("casadi", reason='needs the planning extra: pip install -e ".[planning]"')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(env, **over):
    GlobalHydra.instance().clear()
    ov = ["approach=planning", "approach.method=optimization",
          f"env={env}", "shaping=euclidean", "init=fixed"]
    ov += [f"{k}={v}" for k, v in over.items()]
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        return compose("config", overrides=ov)


def test_min_time_matches_analytic_bangbang():
    """Free-dt solve must find the true minimum-time profile, not just a feasible one.

    swap2 is a straight 3 m traverse from rest to rest with the heading already
    aligned, so the optimum is exactly the bang-bang time the braking potential
    computes. This is what separates a minimum-time program from a tracking MPC.
    """
    from src.approach.planning.trajopt import solve_trajectory
    from src.env.factory import build_env
    from src.shaping.braking_potential import bangbang_time

    env = build_env(_cfg("swap2_unicycle2"))
    env.reset(seed=0)
    r, tol = env.robots[0], 0.1

    X, U, dt, ok = solve_trajectory(
        r, env._states[0], env._goals[0], env._obstacles, env._world_size,
        horizon=40, goal_tol=tol, clearance=0.0,
    )
    assert ok

    # It stops goal_tol short, so compare against the distance actually covered.
    covered = float(np.linalg.norm(X[-1, :2] - env._states[0][:2]))
    assert 40 * dt == pytest.approx(bangbang_time(covered, 0.0, r.v_max, r.a_max), rel=2e-3)

    # Bang-bang saturates both bounds, and the terminal stop constraint holds.
    assert np.abs(X[:, 3]).max() == pytest.approx(r.v_max, rel=1e-3)
    assert np.abs(U[:, 0]).max() == pytest.approx(r.a_max, rel=1e-3)
    assert abs(X[-1, 3]) < 1e-6 and abs(X[-1, 4]) < 1e-6


def test_prioritised_plan_executes_collision_free():
    """The head-on swap IPPO cannot solve. Planned controls must replay cleanly."""
    from src.approach.planning import build_planner
    from src.approach.rollout import run_episode
    from src.env.factory import build_env

    cfg = _cfg("swap2_unicycle2")
    env = build_env(cfg)
    planner = build_planner(cfg.approach)
    stats, _ = run_episode(env, planner, render=False)

    assert all(planner._solved.values()), "IPOPT failed to solve"
    assert stats["success"]
    assert stats["collisions"] == 0.0


def test_horizon_is_derived_not_hardcoded():
    """A fixed horizon silently makes the program infeasible when dt or the robot
    changes: horizon * env.dt is the time budget. It must cover the traverse."""
    from src.approach.planning import build_planner
    from src.env.factory import build_env
    from src.shaping.braking_potential import bangbang_time

    for name in ("swap2_unicycle2", "gap_2agent"):
        env = build_env(_cfg(name))
        env.reset(seed=0)
        h = build_planner(_cfg(name).approach)._auto_horizon(env, 1.5)
        need = max(
            bangbang_time(float(np.linalg.norm(env._goals[i][:2] - env._states[i][:2])),
                          0.0, env.robots[i].v_max, env.robots[i].a_max)
            for i in range(env._n)
        )
        assert h * env.dt >= need, f"{name}: budget {h * env.dt:.2f}s < traverse {need:.2f}s"


def test_planning_config_drops_learning_groups():
    """approach=planning must not carry network/train: they belong to the RL approach,
    so `--cfg job` shows only knobs that actually affect the run."""
    plan, rl = _cfg("swap2_unicycle2"), None
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        rl = compose("config", overrides=["env=swap2_unicycle2"])

    assert "network" not in plan and "train" not in plan
    assert "network" in rl and "train" in rl          # RL keeps them
    # everything a planning run needs is still there, and is configurable
    assert plan.approach.karc.m_segments >= 1
    assert plan.approach.trajopt.clearance > 0


def test_karc_solves_obstacle_scenario_collision_free():
    from src.approach.planning import build_planner
    from src.approach.rollout import run_episode
    from src.env.factory import build_env

    cfg = _cfg("gap_2agent", **{"approach.method": "karc"})
    env = build_env(cfg)
    planner = build_planner(cfg.approach)
    stats, _ = run_episode(env, planner, render=False)

    assert all(planner._solved.values())
    assert stats["success"] and stats["collisions"] == 0.0
    assert planner.stats["braked_segments"] == 0     # no segment was abandoned


def test_separate_pulls_coinciding_milestones_apart_laterally():
    """Two robots swapping head-on descend the SAME reference path, so their k-th
    milestones coincide — and since a milestone is a segment's terminal constraint while
    the robots must stay r_i+r_j+clearance apart at every index, that segment is
    infeasible by construction. No solver rung can repair it; the milestones must move.
    """
    from src.approach.planning.karc import KARCPlanner

    # Both robots travel along y = 2.5 in opposite directions and meet in the middle.
    ms = [
        [np.array([1.8, 2.5, 0.0]), np.array([2.53, 2.5, 0.0]), np.array([4.0, 2.5, 0.0])],
        [np.array([3.2, 2.5, 0.0]), np.array([2.50, 2.5, 0.0]), np.array([1.0, 2.5, 0.0])],
    ]
    radii, clearance = [0.28, 0.28], 0.05
    need = sum(radii) + clearance
    assert np.linalg.norm(ms[0][1][:2] - ms[1][1][:2]) < need   # infeasible before

    out = KARCPlanner._separate(ms, radii, clearance, world_size=5.0)

    assert np.linalg.norm(out[0][1][:2] - out[1][1][:2]) >= need
    # The offset must be LATERAL. Pushing head-on robots apart ALONG their shared path
    # only reorders the milestones and still routes them through each other.
    assert abs(out[0][1][1] - out[1][1][1]) > 0.5 * need, "milestones not split across y"
    # The final milestone is the true goal and must never move.
    assert np.allclose(out[0][-1][:2], [4.0, 2.5])
    assert np.allclose(out[1][-1][:2], [1.0, 2.5])


def test_karc_solves_symmetric_head_on_swap():
    """swap2 is the case prioritised resolution cannot fix: whichever robot is ordered
    second has nowhere to yield to. Needs the joint rung AND separated milestones."""
    from src.approach.planning import build_planner
    from src.approach.rollout import run_episode
    from src.env.factory import build_env

    cfg = _cfg("swap2_unicycle2", **{"approach.method": "karc"})
    env = build_env(cfg)
    planner = build_planner(cfg.approach)
    stats, _ = run_episode(env, planner, render=False)

    assert stats["success"], "symmetric head-on swap not solved"
    assert stats["collisions"] == 0.0
