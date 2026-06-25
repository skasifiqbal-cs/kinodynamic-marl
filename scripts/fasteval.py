"""Headless success-rate evaluator — no rendering, fast bulk eval for sanity checks.

Reports BOTH deterministic (mean action) and stochastic (sampled action) success,
because entropy_loss_scale keeps action std high at convergence: the deterministic
mean can be timid while the sampled policy is competent. Both numbers matter.

Usage (hydra overrides identical to training):
    PYTHONPATH=. python scripts/fasteval.py \
        eval.checkpoint=runs/.../best_agent.pt \
        env=single_agent shaping=dubins init=random \
        eval.episodes=100
"""
from __future__ import annotations

import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.networks import build_policy
from src.training.train import _build_env


def _run(env, policies, device, preprocessors, n_episodes, deterministic):
    succ, steps_on_success, crashes, coll_counts = [], [], [], []
    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        step = 0
        last_info = {}
        while env.agents:
            actions = {}
            for agent in env.possible_agents:
                if agent not in env.agents:
                    continue
                obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
                if preprocessors.get(agent) is not None:
                    obs_t = preprocessors[agent](obs_t, train=False)
                with torch.no_grad():
                    if deterministic:
                        act_t, _, _ = policies[agent].compute({"states": obs_t}, role="policy")
                    else:
                        act_t, _, _ = policies[agent].act({"states": obs_t}, role="policy")
                act = np.asarray(act_t).squeeze() if isinstance(act_t, np.ndarray) else act_t.squeeze(0).cpu().numpy()
                act = np.clip(act, env.action_space(agent).low, env.action_space(agent).high)
                actions[agent] = act
            obs_dict, _, _, _, info = env.step(actions)
            last_info = info
            step += 1
        ep = last_info.get(env.possible_agents[0], {}).get("episode", {})
        ok = bool(ep.get("success", float(all(env._reached))))
        succ.append(float(ok))
        crashes.append(float(ep.get("crashed", 0.0)))
        coll_counts.append(float(ep.get("collisions", env._collision_count)))
        if ok:
            steps_on_success.append(step)
    return (
        float(np.mean(succ)),
        float(np.mean(steps_on_success)) if steps_on_success else float("nan"),
        float(np.mean(crashes)),
        float(np.mean(coll_counts)),
    )


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, os.getcwd())
    eval_cfg = cfg.get("eval", OmegaConf.create({}))
    checkpoint = eval_cfg.get("checkpoint", None)
    n_episodes = int(eval_cfg.get("episodes", 100))
    seed = int(eval_cfg.get("seed", 12345))

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _build_env(cfg)
    env.reset(seed=seed)

    from skrl.resources.preprocessors.torch import RunningStandardScaler

    policies, preprocessors = {}, {}
    for agent_id in env.possible_agents:
        policies[agent_id] = build_policy(
            env.observation_space(agent_id), env.action_space(agent_id), device, cfg.network
        ).to(device)
        preprocessors[agent_id] = None

    if not checkpoint:
        print("[fatal] eval.checkpoint=... required")
        return

    ckpt = torch.load(checkpoint, map_location=device)
    for agent_id, policy in policies.items():
        if agent_id in ckpt and "policy" in ckpt[agent_id]:
            policy.load_state_dict(ckpt[agent_id]["policy"], strict=False)
        if agent_id in ckpt and ckpt[agent_id].get("state_preprocessor"):
            pp = RunningStandardScaler(size=env.observation_space(agent_id), device=device)
            pp.load_state_dict(ckpt[agent_id]["state_preprocessor"])
            pp.eval()
            preprocessors[agent_id] = pp
        policy.eval()

    print(f"checkpoint: {checkpoint}")
    print(f"env={cfg.env.get('_name_', 'custom')} shaping={cfg.shaping.type} init={cfg.init.type} "
          f"episodes={n_episodes}")

    for label, det in (("deterministic", True), ("stochastic", False)):
        sr, avg_steps, crash, coll = _run(env, policies, device, preprocessors, n_episodes, det)
        print(f"  {label:13s}: success={sr:6.1%}  crash_rate={crash:6.1%}  "
              f"avg_collisions={coll:5.1f}  avg_steps_on_success={avg_steps:6.1f}")


if __name__ == "__main__":
    main()
