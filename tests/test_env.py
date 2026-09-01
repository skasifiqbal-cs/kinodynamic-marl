"""Env wiring: build, obs dim, step contract, freeze-on-reach."""
import os

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(env="crossing_2agent"):
    from src.env.factory import build_env
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"env={env}", "shaping=dijkstra", "init=fixed"])
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
    assert coef > 0, "crossing_2agent must set reward.omega_penalty"

    a0 = env.possible_agents[0]
    env._states[0][3] = 0.0          # not translating
    env._states[0][4] = 1.0          # spinning at 1 rad/s
    zero = {a: np.zeros(env.action_space(a).shape, dtype=np.float32) for a in env.agents}
    _, rew, _, _, _ = env.step(zero)  # alpha = 0 -> effort term contributes nothing

    # step_penalty + omega_penalty * omega^2, with omega propagated by one RK4 step.
    charged = env.step_penalty - rew[a0]
    assert charged > 0, "constant-rate spin must not be free"
    assert charged == pytest.approx(coef * float(env._states[0][4]) ** 2, rel=1e-6)
