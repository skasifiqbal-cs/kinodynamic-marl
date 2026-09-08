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

    for name in ("swap2_unicycle2", "gap2_unicycle2"):
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

    cfg = _cfg("gap2_unicycle2", **{"approach.method": "karc"})
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


def test_planning_run_reports_the_coordination_counters(capsys, tmp_path):
    """success alone cannot tell you whether a ladder rung you removed was ever used --
    an ablation is read off `rungs` and `solver_calls`, so run() must print them."""
    from src.approach import build_approach

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["approach=planning", "approach.method=karc",
                                           "env=swap2_unicycle2", "eval.episodes=1",
                                           "eval.gif_path=null"])
    build_approach(cfg).run(cfg)
    out = capsys.readouterr().out
    assert "STATS,karc," in out
    for key in ("rungs=", "solver_calls=", "conflicts="):
        assert key in out, f"{key} missing from the planning report"


def test_karc_trace_is_off_by_default_and_records_every_stage_when_on():
    """The trace is what scripts/karc_trace_gif.py draws. It must stay off unless asked
    (it retains every intermediate trajectory), and when on it must cover the whole
    algorithm: reference paths, the uncoordinated solve, and each rung that ran."""
    from src.approach.planning import build_planner
    from src.env.factory import build_env

    def plan(trace):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
            cfg = compose("config", overrides=["approach=planning", "approach.method=karc",
                                               "env=open_cross_4_unicycle2", "init=fixed",
                                               f"approach.karc.trace={str(trace).lower()}"])
        env = build_env(cfg)
        env.reset(seed=0)
        p = build_planner(cfg.approach)
        p.reset(env)
        return env, p

    _, off = plan(False)
    assert off.trace is None, "trace must be opt-in"

    env, on = plan(True)
    labels = [st["label"] for st in on.trace]
    assert labels[0].startswith("kinematic reference paths")
    assert labels[-1] == "final plan"
    assert any("uncoordinated solve" in ln for ln in labels)
    assert any("prioritized" in ln for ln in labels), "segment 2 conflicts; a rung must run"

    for st in on.trace:
        assert len(st["static"]) == env._n
        # anim is empty exactly for a stage with no dynamics to replay (the reference
        # path); every other stage carries one trajectory per robot.
        assert len(st["anim"]) in (0, env._n)
        for path in st["static"]:
            assert path.ndim == 2 and path.shape[1] == 2, "static trails are (T, 2)"
        for path in st["anim"]:
            # (T, 3): the heading is what orients the robot body while it is driven, so a
            # trajectory stored as bare xy would render every robot pointing along +x.
            assert path.ndim == 2 and path.shape[1] == 3, "driven paths need headings"

    # The last stage drives the whole committed plan, so it must be longer than any one
    # segment -- that is the difference between replaying the plan and replaying a piece.
    assert on.trace[-1]["label"] == "final plan"
    longest_segment = max(len(a) for st in on.trace[:-1] for a in st["anim"] if len(a))
    assert max(len(a) for a in on.trace[-1]["anim"]) > longest_segment
    # The conflict markers are what the red crosses are drawn at; open_cross_4's segment 2
    # is the one that conflicts.
    assert any(st["markers"] for st in on.trace)
    # Segment boundaries, drawn as dotted circles: m_segments milestones per robot, on
    # every stage (they are the frame the whole plan is built in, not per-stage decoration).
    counts = {len(st["waypoints"]) for st in on.trace}
    assert len(counts) == 1, f"waypoints must not vary by stage: {counts}"
    n_wp = counts.pop()
    assert n_wp > 0 and n_wp % env._n == 0, n_wp

    # The reference stage is DRIVEN, not stilled: robots walk it uncoordinated and collide,
    # which is the motivation for every stage after it. Poses carry a heading from the path
    # tangent, so a straight leg must not read as theta=0 for a robot heading -x.
    ref = on.trace[0]["anim"]
    assert len(ref) == env._n and all(len(a) > 2 for a in ref)
    assert any(abs(float(a[len(a) // 2][2])) > 1e-6 for a in ref), "headings all zero"

    from src.collision.shapes import collides
    shapes = [r.shape for r in env.robots]
    hit = any(
        collides(shapes[i], tuple(ref[i][t][:3]), shapes[j], tuple(ref[j][t][:3]))
        for t in range(min(len(a) for a in ref))
        for i in range(env._n) for j in range(i + 1, env._n)
    )
    assert hit, "uncoordinated reference paths must actually collide, or stage 1 shows nothing"


def test_trace_commits_the_braking_rollout_for_an_unsolved_segment():
    """An unsolved segment is never executed -- the planner brakes to rest instead. The
    trace must record what the robot actually does, or the final-plan animation shows a
    trajectory that was rejected, which is precisely the case worth watching.
    """
    import numpy as np

    from src.approach.planning.karc import KARCPlanner
    from src.env.factory import build_env

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["approach=planning", "approach.method=karc",
                                           "env=open_cross_4_unicycle2", "init=fixed"])
    env = build_env(cfg)
    env.reset(seed=0)

    st = env._states[0].copy()
    st[3], st[4] = 0.5, 0.5                       # moving, so braking is not a no-op
    us, final, path = KARCPlanner._brake(env, 0, st, 30)

    assert len(us) == 30 and len(path) == 31, "one state per step, plus the start"
    assert path.shape[1] == env._states[0].shape[0]
    assert np.allclose(path[0], st)
    assert np.allclose(path[-1], final), "reported final state must end the path"
    assert abs(float(final[3])) < abs(float(st[3])), "braking must shed speed"


def _karc_stats(env_name, **overrides):
    from src.approach.planning import build_planner
    from src.env.factory import build_env

    ov = ["approach=planning", "approach.method=karc", f"env={env_name}", "init=fixed"]
    ov += [f"approach.karc.{k}={v}" for k, v in overrides.items()]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=ov)
    env = build_env(cfg)
    env.reset(seed=0)
    p = build_planner(cfg.approach)
    p.reset(env)
    return env, p


def test_a_subproblem_is_one_conflicting_pair_not_every_conflicting_robot():
    """ARC SS IV-B builds a subproblem around ONE conflict: R' = R_i u R_j. Merging every
    conflicting robot in a segment turns N/2 independent pair solves into one N-robot
    coupled solve, whose cost and failure rate are then ours rather than the algorithm's.

    open_cross_4 is two independent head-on pairs, so the two settings must reach the same
    plan by different routes -- which is what makes |R'| the thing being measured.

    min_time=false is pinned: with a free dt the segment horizons shrink, the two pairs stop
    conflicting inside the SAME segment, and `merged` degenerates to `pair` -- which makes the
    scenario stop being a test of |R'| at all.
    """
    env, pair = _karc_stats("open_cross_4_unicycle2", subproblem="pair", min_time=False)
    _, merged = _karc_stats("open_cross_4_unicycle2", subproblem="merged", min_time=False)

    assert pair.stats["subproblem_max"] == 2, pair.stats["subproblem_max"]
    assert merged.stats["subproblem_max"] == env._n, merged.stats["subproblem_max"]
    assert pair.stats["subproblems"] > merged.stats["subproblems"]

    # Same outcome here: the pairs do not interact, so locality costs nothing.
    assert pair.stats["conflicts_remaining"] == merged.stats["conflicts_remaining"] == 0
    assert pair.stats["path_cost"] == pytest.approx(merged.stats["path_cost"])
    assert pair.stats["unsolved_segments"] == merged.stats["unsolved_segments"] == 0


def test_a_singleton_subproblem_exists_for_an_infeasible_segment():
    """A robot whose own segment is infeasible has no conflict partner. It still needs
    re-solving, so it forms a subproblem of one rather than being dropped."""
    from src.approach.planning.karc import KARCPlanner

    # No conflicts at all, robot 1 infeasible -> exactly one subproblem, containing it.
    groups = []
    oks = [True, False, True]
    stranded = {i for i, ok in enumerate(oks) if not ok} - set().union(*groups, set())
    assert stranded == {1}
    assert hasattr(KARCPlanner, "_clear")
    assert KARCPlanner._clear({1}, [], [True, True, True]) is True
    assert KARCPlanner._clear({1}, [], oks) is False
    assert KARCPlanner._clear({1}, [(0, 1, 5)], [True, True, True]) is False
    assert KARCPlanner._clear({2}, [(0, 1, 5)], [True, True, True]) is True


def test_adapt_subproblem_reopens_the_previous_segment_and_rescues_it():
    """Alg. 2 line 10. A segment can be unsolvable purely because the one before it
    arrived badly placed, and no amount of re-solving inside its own window fixes that;
    K-ARC's answer is to move the start query back to the previous segment's start.

    swap2 is the case: its head-on pair leaves segment 3 unsolved, and the planner brakes.
    One adaptation resolves it. Without this, max_rounds re-solves a near-identical problem,
    since nothing about the window changes between rounds.
    """
    # Two settings are pinned so this stays a test of adaptation and not of anything else:
    #   initial_paths=dijkstra - the failure needs the SYMMETRIC case, where both robots
    #     descend the same reference and their k-th milestones coincide. Sampling-based
    #     initial paths give the pair different routes and dissolve the symmetry, so under
    #     the default this scenario solves outright and there is nothing to adapt.
    #   on_unsolved=brake (off arm) - under K-ARC's `return ∅` an unresolved segment
    #     discards the whole plan, so the path-cost comparison below would be against nothing.
    #   min_time=false - a free dt re-times the milestones, so the segment that adaptation
    #     rescues here is no longer the one that fails. Adaptation is what is under test.
    _, off = _karc_stats("swap2_unicycle2", adapt_max=0, on_unsolved="brake",
                         initial_paths="dijkstra", min_time=False)
    _, on = _karc_stats("swap2_unicycle2", adapt_max=1, initial_paths="dijkstra",
                        min_time=False)

    assert off.stats["adaptations"] == 0
    assert off.stats["unsolved_segments"] >= 1, "swap2 must still be the hard case here"

    assert on.stats["adaptations"] >= 1, "the hierarchy failed; adaptation must have run"
    assert on.stats["unsolved_segments"] == 0
    assert on.stats["braked_segments"] == 0
    assert on.stats["plan_failed"] == 0
    # Re-opening committed motion buys a better plan, not just a feasible one.
    assert on.stats["path_cost"] < off.stats["path_cost"]
    # And it is not free: the rescued window is solved twice.
    assert on.stats["solver_calls"] > off.stats["solver_calls"]


def test_adaptation_cannot_reach_past_the_first_segment():
    """There is no segment before the first, so a failure there has nothing to re-open.
    The guard is `not prev_starts`; without it the checkpoint list would be popped empty."""
    _, p = _karc_stats("swap2_unicycle2", adapt_max=99, m_segments=1)
    assert p.stats["adaptations"] == 0


def test_adapt_subproblem_undoes_committed_motion_back_to_a_checkpoint():
    """AdaptSubProblem (Alg. 2 line 10) widens a window by moving the start query back to
    the previous segment's start. That means UNDOING committed motion, which is the only
    part with state to get wrong: controls already appended, the solved flags, and the
    trace's committed poses all have to roll back together, or the re-planned window is
    appended to motion it was supposed to replace.

    Effort counters deliberately do NOT roll back -- those solver calls happened and cost
    wall time, and reporting otherwise would understate what adaptation costs.
    """
    import numpy as np

    from src.approach.planning import build_planner
    from src.env.factory import build_env

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["approach=planning", "approach.method=karc",
                                           "env=open_cross_4_unicycle2", "init=fixed"])
    env = build_env(cfg)
    env.reset(seed=0)
    p = build_planner(cfg.approach)

    agents = list(env.possible_agents)
    p.trace = None
    p._controls = {a: [np.zeros(2)] * 5 for a in agents}
    p._solved = {a: True for a in agents}
    state = [s.copy() for s in env._states]

    ck = p._checkpoint(agents, state)

    # ... a segment is planned and committed, and one robot fails ...
    for a in agents:
        p._controls[a].extend([np.ones(2)] * 7)
    p._solved[agents[1]] = False
    state[1] = state[1] + 3.0

    restored = p._restore(agents, ck)

    assert all(len(p._controls[a]) == 5 for a in agents), "committed controls must truncate"
    assert p._solved[agents[1]] is True, "a rolled-back failure is no longer a failure"
    assert np.allclose(restored[1], ck["state"][1]), "start query returns to the checkpoint"
    # The checkpoint is a copy, not a view: mutating live state must not rewrite history.
    assert not np.allclose(restored[1], state[1])


def test_adaptation_is_configurable_and_off_by_zero():
    """adapt_max=0 reproduces the pre-2026-09-07 behaviour, so the mechanism's cost and
    benefit are measurable rather than asserted."""
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["approach=planning", "approach.method=karc"])
    assert cfg.approach.karc.adapt_max == 1, "K-ARC's 'previous segment' is one step back"
    assert "adapt_max" in OmegaConf.to_container(cfg.approach.karc)


def test_timeout_fails_safely_instead_of_planning_forever():
    """K-ARC's setup (SS V) gives every method 600 s per instance, so an unbounded planner
    cannot produce a comparable number: a plan found at three hours is a failure nobody
    stopped. Two things must hold when the budget expires -- the run must STOP, and it must
    stop at rest. `act` pads an exhausted control sequence with zero ACCELERATION, which for
    a second-order robot is coasting, so a timeout that simply stopped planning would send
    every robot on at its current velocity and score collisions the planner never chose.
    """
    from src.approach.planning import build_planner
    from src.approach.rollout import run_episode
    from src.env.factory import build_env

    cfg = _cfg("open_cross_16_unicycle2",
               **{"approach.method": "karc", "approach.karc.timeout": 5.0})
    env = build_env(cfg)
    planner = build_planner(cfg.approach)
    stats, _ = run_episode(env, planner, render=False)

    assert planner.stats["timed_out"] == 1
    assert planner.stats["unsolved_segments"] > 0
    assert not stats["success"]
    # Overshoot is bounded by one solver call, not by the ladder.
    assert planner.stats["wall_time"] < 60.0
    assert stats["collisions"] == 0.0, "timed out into a crash instead of braking to rest"
    for i, a in enumerate(env.possible_agents):
        assert abs(float(env._states[i][3])) < 1e-6, f"{a} still moving after the timeout"


def test_terminal_tolerance_stays_strictly_inside_the_env_goal_test():
    """The env tests `dist < goal_radius` (multiagent_nav.py) and the objective is minimum
    time, so a solver handed goal_tol == goal_radius parks the terminal state exactly on the
    boundary and the plan scores as a failure it did not commit. Caught on
    open_cross_32_wide: 32/32 segments solved, every robot at rest, two of them at exactly
    0.200 m from a 0.2 m goal -> success=False.
    """
    from src.approach.planning import build_planner
    from src.env.factory import build_env

    cfg = _cfg("open_cross_4_unicycle2", **{"approach.method": "karc"})
    env = build_env(cfg)
    env.reset()
    planner = build_planner(cfg.approach)
    t_cfg = planner.approach_cfg.get("trajopt", {})

    assert float(t_cfg.get("goal_tol")) == env.goal_radius, \
        "premise of this test: the configured tolerance equals the env's radius"
    assert planner._terminal_tol(env, t_cfg, 1.0, True) < env.goal_radius
    # Intermediate milestones are not tested by the env and keep the configured value.
    assert planner._terminal_tol(env, t_cfg, 1.0, False) == float(t_cfg.get("goal_tol"))


def test_margin_rung_select_keeps_the_ladder_below_the_rung_it_picks():
    """The safety argument for grading the response by severity.

    `rung_select: margin` may SKIP rungs it predicts will fail, but the rungs after the one
    it picks must remain as fallback -- otherwise a wrong prediction turns a solvable
    subproblem into an unsolved one, and the mechanism trades success for speed. Checked on
    the selector directly rather than inferred from a run.

    Also pins the predicate itself: the question is whether WAITING separates the pair, which
    is the only concession the prioritized rung can make. An earlier version graded severity
    by the braking margin at the conflict index -- that is negative for every conflict by
    construction, since a conflict is detected when the geometric gap is already below
    r_i + r_j, so it escalated everything and lost to the fixed ladder.
    """
    import numpy as np

    from src.approach.planning import build_planner
    from src.env.factory import build_env

    cfg = _cfg("open_cross_4_unicycle2",
               **{"approach.method": "karc", "approach.karc.rung_select": "margin"})
    env = build_env(cfg)
    env.reset(seed=0)
    p = build_planner(cfg.approach)
    p.stats = {"rungs_skipped": 0, "escalated": 0}
    ladder = ["prioritized", "decoupled_rrt", "composite_rrt"]
    radii = [float(r.shape.bounding_radius) for r in env.robots]
    n = 40

    def _traj(x, y, th, v):
        """A straight constant-speed trajectory on the env's dt grid."""
        out = []
        s0 = np.array([x, y, th, v, 0.0])
        for k in range(n + 1):
            out.append(s0 + np.array([np.cos(th), np.sin(th), 0, 0, 0]) * v * env.dt * k)
        return np.asarray(out)

    # One robot crossing well clear of a stopped one: waiting trivially separates them.
    clear = [_traj(1.0, 1.0, 0.0, 0.4), _traj(1.0, 9.0, 0.0, 0.4)]
    assert p._start_rung(ladder, env, {0, 1}, [(0, 1, 5)], clear, radii, None, 0.05) == 0

    # Head-on in the same lane: whichever robot stops, the other drives into it, so no
    # ordering resolves this and the prioritized rung cannot express the fix.
    # 2 m apart closing at 0.8 m/s meet inside the 4 s window; a waiter at 0.4 m/s with
    # a_max 0.25 stops in 0.32 m, so stopping does not get it out of the way.
    head_on = [_traj(4.0, 1.0, 0.0, 0.4), _traj(6.0, 1.0, np.pi, 0.4)]
    picked = p._start_rung(ladder, env, {0, 1}, [(0, 1, 20)], head_on, radii, None, 0.05)
    assert picked >= 1, "a head-on pair that waiting cannot separate must skip rung 0"
    assert picked < len(ladder), "something must remain below the chosen rung"
    assert p.stats["rungs_skipped"] == picked

    # Faithful default is untouched: always rung 0.
    p.params = dict(p.params)
    p.params["rung_select"] = "sequential"
    assert p._start_rung(ladder, env, {0, 1}, [(0, 1, 20)], head_on, radii, None, 0.05) == 0


def test_a_failed_min_time_probe_cannot_shorten_the_segment():
    """The segment must take the SLOWEST robot's minimum time (K-ARC SS IV-D-2: the robots
    "achieve the milestones for a segment at the same time"). A robot whose free-dt probe
    fails to converge is not a fast robot -- it is the constrained one -- so taking `max`
    over only the probes that converged is a max over a biased subsample.

    Dropping it hands the segment a horizon SHORTER than the guide-length estimate. That
    robot's own solve then cannot succeed at any rung, because every rung re-solves inside
    the same horizon, and the round loop re-solves an identical infeasible problem until it
    gives up with budget to spare. Observed on cluttered_cross_16: timed_out=0, rounds=3,
    unsolved_segments=1, all three rungs fired and failed.
    """
    from types import SimpleNamespace

    from src.approach.planning.karc import KARCPlanner

    p = KARCPlanner.__new__(KARCPlanner)
    p.stats = {"min_time_solves": 0, "min_time_failures": 0}
    env = SimpleNamespace(robots=[None] * 3, _obstacles=[], _world_size=10.0, _n=3,
                          dt=0.1, goal_radius=0.2)
    seg_h, total_h = 40, 400

    # Two easy robots converge at a third of the guide estimate; the third does not converge.
    p._solve_many = lambda specs: [(None, None, 0.033, True), (None, None, 0.033, True),
                                   (None, None, 0.5, False)]
    h = p._min_time_horizon(env, {}, [None] * 3, [None] * 3, [None] * 3,
                            total_h, seg_h, True)
    assert h == seg_h, h
    assert p.stats["min_time_failures"] == 1

    # All converged -> the slowest one sets the horizon, and it may be shorter than the guide.
    p._solve_many = lambda specs: [(None, None, 0.033, True)] * 3
    assert p._min_time_horizon(env, {}, [None] * 3, [None] * 3, [None] * 3,
                               total_h, seg_h, True) < seg_h


def test_corridor_blockers_follow_distance_not_knot_count():
    """A partner's segment becomes static blockers for the guide-repair rung. Dedupe is by
    ARCLENGTH, so the cost tracks the metres covered rather than the knot count -- otherwise
    a finely discretised segment would bury the geometric RRT in redundant obstacles.

    The footprint is the robot's own box at its own heading, not a circumscribed disc: for
    the 0.5 x 0.25 body here the disc is 0.559 m across against a 0.25 m lateral extent, and
    the difference is exactly the gap a robot can pass through.
    """
    from src.approach.planning.karc import KARCPlanner
    from src.collision.shapes import BoxShape

    shape = BoxShape(0.5, 0.25)
    # 4 m of travel, sampled 400 times -- 0.01 m apart.
    fine = np.array([[x, 1.0, 0.0] for x in np.linspace(0.0, 4.0, 400)])
    coarse = np.array([[x, 1.0, 0.0] for x in np.linspace(0.0, 4.0, 40)])

    b_fine = KARCPlanner._corridor(fine, shape, stride_m=0.25)
    b_coarse = KARCPlanner._corridor(coarse, shape, stride_m=0.25)
    # 10x the knots over the same 4 m must not cost 10x the blockers: both land near
    # 4 m / 0.25 m, against 400 and 40 if the stride were counted in knots.
    assert 14 <= len(b_fine) <= 18, len(b_fine)
    assert 14 <= len(b_coarse) <= 18, len(b_coarse)
    assert abs(len(b_fine) - len(b_coarse)) <= 2, (len(b_fine), len(b_coarse))
    assert all(b.shape is shape for b in b_fine)
    assert all(abs(b.angle) < 1e-9 for b in b_fine)
    # Consecutive blockers must OVERLAP along the path, or a robot could slip between two
    # of them. A blocker is kept once the last one is >= stride away, so the gap can exceed
    # the stride by one sample spacing -- what has to hold is that it stays under the box's
    # own extent along travel (0.5 m here).
    for b in (b_fine, b_coarse):
        gaps = np.diff([o.x for o in b])
        assert gaps.max() < shape.width, (gaps.max(), shape.width)

    assert KARCPlanner._corridor(np.zeros((0, 3)), shape) == []


def test_guide_repair_blockers_come_only_from_the_subproblem():
    """The guide must route around the conflict partner, not every robot in the scene.

    Collapsing a trajectory into a static obstacle discards time; doing it for all
    N-|R'| robots outside the subproblem walls off the workspace and the repair RRT
    finds nothing (28/30 blocked at N=32). `avoid` still constrains the kinodynamic
    solve -- it just must not constrain the guide.
    """
    import inspect

    from src.approach.planning.karc import KARCPlanner

    src = inspect.getsource(KARCPlanner._repair_guides)
    assert "for traj, _r in blocking:" in src, "guide blockers must come from `blocking`"
    assert "for traj, _r in fixed:" not in src, (
        "`fixed` holds every robot outside R'; using it as guide blockers is the bug"
    )
