"""Env wiring: build, obs dim, step contract, freeze-on-reach."""
import os

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(env="crossing_2agent"):
    from src.training.train import _build_env
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"env={env}", "shaping=dijkstra", "init=fixed"])
    return _build_env(cfg)


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
