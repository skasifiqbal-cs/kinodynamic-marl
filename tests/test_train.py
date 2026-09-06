"""Model construction: parameter sharing across a homogeneous team."""
import os

import pytest
import torch
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.approach.rl.train import build_models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubEnv:
    """Just the three spaces accessors build_models uses. Widths per agent, so a mixed
    team is expressible without a robot config that produces one."""

    def __init__(self, obs_dims):
        self.possible_agents = [f"agent_{i}" for i in range(len(obs_dims))]
        self._obs = dict(zip(self.possible_agents, obs_dims))

    def observation_space(self, agent):
        return spaces.Box(-1.0, 1.0, shape=(self._obs[agent],))

    def action_space(self, agent):
        return spaces.Box(-1.0, 1.0, shape=(2,))


def _cfg(share):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"train.share_policy={str(share).lower()}"])
    return cfg


def test_sharing_gives_every_agent_the_same_module_and_separate_gives_distinct():
    """Distinct-but-equal modules is the silent failure: training runs, weights diverge.
    So this asserts object identity, not equality."""
    env = StubEnv([17] * 4)
    device = torch.device("cpu")

    shared = build_models(env, _cfg(True), device)
    assert len({id(m["policy"]) for m in shared.values()}) == 1
    assert len({id(m["value"]) for m in shared.values()}) == 1

    # And it is real parameter sharing: a write through one agent is visible through another.
    p0 = next(shared["agent_0"]["policy"].parameters())
    with torch.no_grad():
        p0.add_(1.0)
    assert next(shared["agent_3"]["policy"].parameters()) is p0

    separate = build_models(env, _cfg(False), device)
    assert len({id(m["policy"]) for m in separate.values()}) == 4
    assert len({id(m["value"]) for m in separate.values()}) == 4


def test_sharing_a_mixed_team_is_rejected_by_name():
    """Different observation widths cannot share an input layer. Torch would report that as a
    bare shape mismatch inside a forward pass; this names the config key that caused it."""
    env = StubEnv([17, 19])
    with pytest.raises(ValueError, match="share_policy"):
        build_models(env, _cfg(True), torch.device("cpu"))
