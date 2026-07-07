"""RL evaluation controller: a trained IPPO policy as a rollout action source.

Wraps the per-agent policy networks + the observation ``RunningStandardScaler``
used in training, loaded from a checkpoint. Lifts the action logic that used to
live inline in ``evaluate.py`` / ``scripts/fasteval.py`` behind the shared
``Controller`` interface.
"""
from __future__ import annotations

import numpy as np
import torch

from src.approach.base import Controller
from src.networks import build_policy


class RLController(Controller):
    def __init__(self, cfg, env, checkpoint: str | None, device, deterministic: bool = True):
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policies: dict = {}
        self.preprocessors: dict = {}

        for agent_id in env.possible_agents:
            self.policies[agent_id] = build_policy(
                env.observation_space(agent_id), env.action_space(agent_id), device, cfg.network
            ).to(device)
            self.preprocessors[agent_id] = None

        if checkpoint:
            self._load(checkpoint, env)
        else:
            print("[warn] No checkpoint — RL controller uses random weights.")

        for p in self.policies.values():
            p.eval()

    def _load(self, checkpoint: str, env):
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        ckpt = torch.load(checkpoint, map_location=self.device)
        for agent_id, policy in self.policies.items():
            if agent_id in ckpt and "policy" in ckpt[agent_id]:
                policy.load_state_dict(ckpt[agent_id]["policy"], strict=False)
            elif f"{agent_id}/policy" in ckpt:
                policy.load_state_dict(ckpt[f"{agent_id}/policy"], strict=False)
            else:
                print(f"[warn] {agent_id}/policy not in checkpoint — random weights")
            # Restore the observation normaliser (CRITICAL — policy trained on normalised obs).
            if agent_id in ckpt and ckpt[agent_id].get("state_preprocessor"):
                pp = RunningStandardScaler(size=env.observation_space(agent_id), device=self.device)
                pp.load_state_dict(ckpt[agent_id]["state_preprocessor"])
                pp.eval()
                self.preprocessors[agent_id] = pp
        print(f"Loaded checkpoint: {checkpoint}")

    def reset(self, env) -> None:  # policies are stateless across episodes
        return None

    def act(self, obs_dict: dict, env) -> dict:
        actions = {}
        for agent in env.possible_agents:
            if agent not in env.agents:
                continue
            obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.preprocessors.get(agent) is not None:
                obs_t = self.preprocessors[agent](obs_t, train=False)
            with torch.no_grad():
                if self.deterministic:
                    # compute() returns the mean action (deterministic)
                    act_t, _, _ = self.policies[agent].compute({"states": obs_t}, role="policy")
                else:
                    act_t, _, _ = self.policies[agent].act({"states": obs_t}, role="policy")
            act = np.asarray(act_t).squeeze() if isinstance(act_t, np.ndarray) else act_t.squeeze(0).cpu().numpy()
            actions[agent] = act
        return actions
