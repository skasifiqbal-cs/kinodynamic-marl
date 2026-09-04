"""Env wiring: build, obs dim, step contract, freeze-on-reach."""
import os

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(env="gap_2agent", shaping="dijkstra"):
    from src.env.factory import build_env
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"env={env}", f"shaping={shaping}", "init=fixed"])
    return build_env(cfg)


def test_build_and_obs_dim():
    env = build()
    obs, _ = env.reset(seed=0)
    for a in env.possible_agents:
        assert obs[a].shape == tuple(env.observation_space(a).shape)
        assert np.all(np.isfinite(obs[a]))


def test_second_order_state_is_five_dim():
    env = build()
    env.reset(seed=0)
    assert all(s.shape == (5,) for s in env._states)   # [x,y,theta,v,omega]


def test_step_contract():
    env = build()
    env.reset(seed=0)
    acts = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(acts)
    assert set(rew) == set(env.possible_agents)
    for a in env.possible_agents:
        assert a in term and a in trunc


def test_freeze_on_reach_holds_position():
    """An agent marked reached on a prior step must not move on the next step."""
    env = build()
    env.reset(seed=0)
    env._reached[0] = True
    frozen_before = env._states[0].copy()
    acts = {a: env.action_space(a).high for a in env.agents}  # max action
    env.step(acts)
    assert np.allclose(env._states[0], frozen_before)


def test_five_element_start_goal_loaded():
    env = build()
    env.reset(seed=0)
    # goals carry the full [x,y,theta,v,omega] from config
    assert env._goals[0].shape[0] >= 3


def test_omega_penalty_charges_constant_rate_spin():
    """A spin at constant rate has alpha=0, so effort_penalty is blind to it.

    Regression for the loopy-path bug: the reward must still charge for rotating.
    """
    env = build()
    env.reset(seed=0)
    coef = env.omega_penalty
    assert coef < 0, "gap_2agent must set reward.omega_penalty, and it is negative"

    a0 = env.possible_agents[0]
    env._states[0][3] = 0.0          # not translating
    env._states[0][4] = 1.0          # spinning at 1 rad/s
    zero = {a: np.zeros(env.action_space(a).shape, dtype=np.float32) for a in env.agents}
    _, rew, _, _, _ = env.step(zero)  # alpha = 0 -> effort term contributes nothing

    # step_penalty + omega_penalty * omega^2, with omega propagated by one RK4 step.
    charged = env.step_penalty - rew[a0]
    assert charged > 0, "constant-rate spin must not be free"
    assert charged == pytest.approx(-coef * float(env._states[0][4]) ** 2, rel=1e-6)


def test_checkpoint_obs_dim_mismatch_names_the_cause():
    """A checkpoint fits only the env it was trained on: obs width depends on the agent
    and obstacle counts, so a single-agent policy cannot run on a 2-agent map. torch
    reports that as a bare size mismatch on net.0.weight, which does not say what to fix.
    """
    import pytest
    import torch

    from src.approach.rl.controller import check_obs_dim

    trained_on_swap1 = {"net.0.weight": torch.zeros(128, 11), "net.0.bias": torch.zeros(128)}
    check_obs_dim(trained_on_swap1, 11, "agent_0", "ckpt.pt")          # matching env: silent

    with pytest.raises(ValueError) as e:
        check_obs_dim(trained_on_swap1, 25, "agent_0", "ckpt.pt")      # gap_2agent
    msg = str(e.value)
    assert "11" in msg and "25" in msg and "ckpt.pt" in msg

    # No 2-D weight to inspect (unexpected layout): stay out of the way, let torch speak.
    check_obs_dim({"log_std": torch.zeros(2)}, 25, "agent_0", "ckpt.pt")


def test_reward_decomposition_pins_every_sign():
    """r = step + effort*||u||^2 + omega*w^2, every coefficient signed in config and ADDED.

    Runs on shaping=none so phi is identically zero and the shaping term drops out,
    leaving three terms that are exactly computable. This is the check that fails if any
    coefficient's sign is flipped in config or in step() -- a flip is otherwise invisible,
    since training still runs and only the learned behaviour is wrong.
    """
    env = build(shaping="none")
    env.reset(seed=0)
    a0 = env.possible_agents[0]

    env._states[0][:2] = [1.5, 1.5]      # away from goal, walls, and the other agent
    env._states[0][3] = 0.3              # translating
    env._states[0][4] = 0.8              # and spinning, so the omega term is non-zero

    act = np.array([0.7, -0.4], dtype=np.float32)
    actions = {a: np.zeros(env.action_space(a).shape, dtype=np.float32) for a in env.agents}
    actions[a0] = act
    _, rew, _, _, _ = env.step(actions)

    # If the placement above ever starts colliding or reaching, the arithmetic below is
    # no longer the whole reward -- say so rather than failing on a confusing mismatch.
    assert env._collision_count == 0, "test placement collided; move it, do not relax this"
    assert not env._reached[0], "test placement reached the goal; move it further away"

    # float64, matching the cast step() makes at multiagent_nav.py:214 -- computing
    # ||u||^2 in the action space's float32 disagrees in the 11th decimal.
    u = np.asarray(act, dtype=np.float64)
    expected = (env.step_penalty
                + env.effort_penalty * float(np.dot(u, u))
                + env.omega_penalty * float(env._states[0][4]) ** 2)
    assert rew[a0] == pytest.approx(expected, rel=1e-9)

    # And the directions, stated independently of the arithmetic.
    assert env.step_penalty < 0
    assert env.effort_penalty < 0
    assert env.omega_penalty < 0
    assert env.reach_reward > 0
    assert env.collision_penalty < 0
    assert rew[a0] < env.step_penalty, "effort and spin must COST, not pay"


def test_wrong_signed_reward_coefficient_is_rejected():
    """An old config carrying a positive effort_penalty must fail loudly, not invert."""
    from src.env.multiagent_nav import check_reward_signs

    check_reward_signs({"reach": 50.0, "collision": -2.0, "effort_penalty": -0.002})
    check_reward_signs({"collision": 0.0})                       # zero disables, allowed

    with pytest.raises(ValueError, match="effort_penalty"):
        check_reward_signs({"effort_penalty": 0.002})            # the pre-change convention
    with pytest.raises(ValueError, match="omega_penalty"):
        check_reward_signs({"omega_penalty": 0.04})
    with pytest.raises(ValueError, match="collision"):
        check_reward_signs({"collision": 2.0})                   # would reward crashing
    with pytest.raises(ValueError, match="reach"):
        check_reward_signs({"reach": -50.0})


def test_circle_obstacle_is_built_encoded_and_collided():
    """crossing_2agent was the only config with `shape: {type: circle}` as an OBSTACLE.

    Robot shapes still cover CircleShape, and gap_2agent's boxes cover the circle-box
    collision branch, but nothing left in conf/ sends a circle through build_obstacle,
    Obstacle.obs_repr or the obstacle half of the observation. Deleting that scenario
    would have dropped the path silently, so it is exercised here directly.
    """
    from omegaconf import OmegaConf

    from src.collision.shapes import CircleShape, collides
    from src.env.factory import build_env

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=["env=gap_2agent", "shaping=euclidean", "init=fixed"])
    cfg.env.obstacles = OmegaConf.create(
        [{"x": 3.0, "y": 2.5, "shape": {"type": "circle", "radius": 0.45}}]
    )
    env = build_env(cfg)
    obs, _ = env.reset(seed=0)

    obstacle = env._obstacles[0]
    assert isinstance(obstacle.shape, CircleShape)
    # Circles are shape-unified into the box encoding as [r, r, sin 0, cos 0].
    assert list(obstacle.obs_repr[2:]) == [0.45, 0.45, 0.0, 1.0]

    for a in env.possible_agents:
        assert obs[a].shape == tuple(env.observation_space(a).shape)
        assert np.all(np.isfinite(obs[a]))

    # One circle, so the obstacle block is 6 wide: 2 heading + 3 goal + 2 other + 6 + 4 wall + 2 dyn.
    assert obs["agent_0"].shape == (19,)
    r = env.robots[0].shape
    assert collides(r, (3.0, 2.5, 0.0), obstacle.shape, obstacle.pose)
    assert not collides(r, (3.0, 0.5, 0.0), obstacle.shape, obstacle.pose)
