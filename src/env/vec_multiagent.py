"""Vectorized multi-agent navigation: B independent MultiAgentNav worlds stepped together.

Why: single-env on-policy PPO is sample-starved on continuous-control nav (gradient
variance too high, reach signal too sparse per update). Literature (CADRL, Long 2018) runs
many parallel actors. This holds B independent MultiAgentNav instances and presents batched
(B, ...) tensors to skrl's IPPO, so each PPO update sees rollouts × B transitions.

Design choice: the inner MultiAgentNav is left UNTOUCHED (it is the validated core). We get
parallelism by composition, not by rewriting the env — zero regression risk. Sub-envs
auto-reset internally (gym vector-env semantics) because skrl's multi-agent trainer only
issues a global reset when `env.agents` is empty, which we never allow.
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

import numpy as np
import torch
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper

from src.env.multiagent_nav import MultiAgentNav


class VecMultiAgentNav:
    """Holds B MultiAgentNav worlds. Batched numpy I/O; sub-envs auto-reset on episode end."""

    def __init__(self, make_env, num_envs: int, base_seed: int = 0) -> None:
        self.envs: List[MultiAgentNav] = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        self._base_seed = base_seed
        e0 = self.envs[0]
        self.possible_agents = list(e0.possible_agents)
        self.num_agents = len(self.possible_agents)
        self._obs_dim = e0.observation_space(self.possible_agents[0]).shape[0]
        self._seeded = False

    # ── spaces (identical across sub-envs) ────────────────────────────────────
    def observation_space(self, agent): return self.envs[0].observation_space(agent)
    def action_space(self, agent):      return self.envs[0].action_space(agent)
    @property
    def observation_spaces(self):       return self.envs[0].observation_spaces
    @property
    def action_spaces(self):            return self.envs[0].action_spaces
    @property
    def state_spaces(self):             return self.envs[0].state_spaces
    @property
    def state_space(self):              return next(iter(self.envs[0].state_spaces.values()))

    @property
    def agents(self) -> List[str]:
        # ALWAYS non-empty: we auto-reset internally, so the trainer must never global-reset.
        return self.possible_agents

    def reset(self) -> Tuple[Mapping[str, np.ndarray], dict]:
        outs = []
        for i, e in enumerate(self.envs):
            seed = (self._base_seed + i) if not self._seeded else None
            o, _ = e.reset(seed=seed)
            outs.append(o)
        self._seeded = True
        obs = {a: np.stack([o[a] for o in outs]).astype(np.float32) for a in self.possible_agents}
        return obs, {}

    def step(self, actions: Mapping[str, np.ndarray]):
        obs_b   = {a: np.zeros((self.num_envs, self._obs_dim), np.float32) for a in self.possible_agents}
        rew_b   = {a: np.zeros(self.num_envs, np.float32) for a in self.possible_agents}
        term_b  = {a: np.zeros(self.num_envs, bool) for a in self.possible_agents}
        trunc_b = {a: np.zeros(self.num_envs, bool) for a in self.possible_agents}

        finished = []  # episode stats from sub-envs that ended this tick
        for i, e in enumerate(self.envs):
            act_i = {a: actions[a][i] for a in self.possible_agents}
            o, r, term, trunc, info = e.step(act_i)
            for a in self.possible_agents:
                rew_b[a][i]   = r[a]
                term_b[a][i]  = bool(term[a])
                trunc_b[a][i] = bool(trunc[a])
            if not e.agents:  # episode done -> capture stats, auto-reset
                ep = info[self.possible_agents[0]].get("episode")
                if ep is not None:
                    finished.append(ep)
                o, _ = e.reset(seed=None)
            for a in self.possible_agents:
                obs_b[a][i] = o[a]

        infos: dict = {}
        if finished:
            infos["episode"] = {
                "success":        float(np.mean([f["success"] for f in finished])),
                "episode_length": float(np.mean([f["episode_length"] for f in finished])),
                "collisions":     float(np.mean([f["collisions"] for f in finished])),
            }
        return obs_b, rew_b, term_b, trunc_b, infos

    def state(self) -> np.ndarray:
        return np.stack([e.state() for e in self.envs]).astype(np.float32)

    def render(self, *a, **k): pass
    def close(self): pass


class VecPettingZooWrapper(MultiAgentEnvWrapper):
    """skrl multi-agent wrapper for VecMultiAgentNav: batched numpy -> torch (B, ...)."""

    def __init__(self, env: VecMultiAgentNav) -> None:
        super().__init__(env)

    def reset(self):
        obs, infos = self._env.reset()
        obs = {a: torch.as_tensor(v, dtype=torch.float32, device=self.device) for a, v in obs.items()}
        return obs, infos

    def step(self, actions: Mapping[str, torch.Tensor]):
        np_actions = {a: t.detach().cpu().numpy() for a, t in actions.items()}
        obs, rew, term, trunc, infos = self._env.step(np_actions)
        dev = self.device
        obs   = {a: torch.as_tensor(v, dtype=torch.float32, device=dev) for a, v in obs.items()}
        rew   = {a: torch.as_tensor(v, dtype=torch.float32, device=dev).view(self.num_envs, 1) for a, v in rew.items()}
        term  = {a: torch.as_tensor(v, dtype=torch.bool, device=dev).view(self.num_envs, 1) for a, v in term.items()}
        trunc = {a: torch.as_tensor(v, dtype=torch.bool, device=dev).view(self.num_envs, 1) for a, v in trunc.items()}
        if "episode" in infos:
            infos["episode"] = {k: torch.tensor(v, device=dev) for k, v in infos["episode"].items()}
        return obs, rew, term, trunc, infos

    def state(self) -> torch.Tensor:
        return torch.as_tensor(self._env.state(), dtype=torch.float32, device=self.device)

    def render(self, *args, **kwargs): pass
    def close(self): pass
