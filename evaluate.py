"""Load a checkpoint and render evaluation episodes to GIF + wandb.

Usage:
    python evaluate.py eval.checkpoint=runs/exp/checkpoints/agent_1000000.pt
    python evaluate.py eval.checkpoint=... eval.episodes=5 wandb.enabled=true
"""
from __future__ import annotations

import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.networks import build_policy

# Single source of truth for env construction (egocentric obs, omega override,
# auto-derived Dubins rho) — do not duplicate the build logic here.
from src.training.train import _build_env
from src.viz import render_frame_with_shapes


def run_episode(env, policies, device, render=True, frame_skip=2, preprocessors=None):
    obs_dict, _ = env.reset()
    trails = [[] for _ in range(env._n)]
    frames = []
    total_rewards = {a: 0.0 for a in env.possible_agents}
    step = 0

    while env.agents:
        actions = {}
        for i, agent in enumerate(env.possible_agents):
            if agent not in env.agents:
                continue
            obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32, device=device).unsqueeze(0)
            # Apply the SAME observation normaliser used in training
            if preprocessors is not None and preprocessors.get(agent) is not None:
                obs_t = preprocessors[agent](obs_t, train=False)
            with torch.no_grad():
                # Use compute() directly for deterministic mean action
                mean_act, _, _ = policies[agent].compute({"states": obs_t}, role="policy")
            act = mean_act.squeeze(0).cpu().numpy()
            act = np.clip(act, env.action_space(agent).low, env.action_space(agent).high)
            actions[agent] = act

        obs_dict, rewards, _, _, _ = env.step(actions)
        for a, r in rewards.items():
            total_rewards[a] += r
        for i in range(env._n):
            trails[i].append(env._states[i][:2].copy())

        if render and step % frame_skip == 0:
            frames.append(render_frame_with_shapes(
                states=[s.copy() for s in env._states],
                robot_shapes=[r.shape for r in env.robots],
                goals=[g.copy() for g in env._goals],
                obstacles=env._obstacles,
                trails=[list(t) for t in trails],
                world_size=env.cfg.env.world_size,
                reached=list(env._reached),
                step=step,
                step_rewards=rewards,
                goal_radius=env.goal_radius,
            ))
        step += 1

    stats = {
        "total_reward": {a: total_rewards[a] for a in env.possible_agents},
        "steps": step,
        "both_reached": all(env._reached),
    }
    return stats, frames


def save_gif(frames, path, fps=15):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"GIF → {path}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, os.getcwd())
    eval_cfg   = cfg.get("eval", OmegaConf.create({}))
    checkpoint = eval_cfg.get("checkpoint", None)
    n_episodes = int(eval_cfg.get("episodes", 3))
    gif_path   = eval_cfg.get("gif_path", "episode.gif")
    fps        = int(eval_cfg.get("fps", 15))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _build_env(cfg)

    from skrl.resources.preprocessors.torch import RunningStandardScaler

    # Rebuild policies (same architecture as training)
    policies = {}
    preprocessors = {}
    for i, agent_id in enumerate(env.possible_agents):
        obs_sp = env.observation_space(agent_id)
        act_sp = env.action_space(agent_id)
        policies[agent_id] = build_policy(obs_sp, act_sp, device, cfg.network).to(device)
        preprocessors[agent_id] = None

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        for agent_id, policy in policies.items():
            if agent_id in ckpt and "policy" in ckpt[agent_id]:
                policy.load_state_dict(ckpt[agent_id]["policy"], strict=False)
                print(f"Loaded {agent_id}/policy")
            elif f"{agent_id}/policy" in ckpt:
                policy.load_state_dict(ckpt[f"{agent_id}/policy"], strict=False)
                print(f"Loaded {agent_id}/policy")
            else:
                print(f"[warn] {agent_id}/policy not in checkpoint — random weights")
            # Restore observation normaliser (CRITICAL — policy trained on normalised obs)
            if agent_id in ckpt and ckpt[agent_id].get("state_preprocessor"):
                pp = RunningStandardScaler(
                    size=env.observation_space(agent_id), device=device
                )
                pp.load_state_dict(ckpt[agent_id]["state_preprocessor"])
                pp.eval()
                preprocessors[agent_id] = pp
                print(f"Loaded {agent_id}/state_preprocessor")
    else:
        print("[warn] No checkpoint — using random weights")

    for p in policies.values():
        p.eval()

    use_wandb = cfg.get("wandb", OmegaConf.create({})).get("enabled", False)
    wb_run = None
    if use_wandb:
        import wandb
        wb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=f"eval_{cfg.shaping.type}_{cfg.network.type}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.shaping.type, cfg.network.type, cfg.obs.type, "eval"],
        )

    all_stats, all_frames = [], []
    for ep in range(n_episodes):
        print(f"Episode {ep+1}/{n_episodes} ...", end=" ", flush=True)
        stats, frames = run_episode(env, policies, device, preprocessors=preprocessors)
        all_stats.append(stats)
        all_frames.extend(frames)
        if frames and ep < n_episodes - 1:
            all_frames.extend([frames[-1]] * 10)
        print(f"steps={stats['steps']}  reached={stats['both_reached']}  "
              f"rewards={stats['total_reward']}")

    success_rate = np.mean([s["both_reached"] for s in all_stats])
    print(f"\nSuccess rate: {success_rate:.0%}")

    if all_frames:
        save_gif(all_frames, gif_path, fps)

    if wb_run:
        wb_run.log({
            "eval/success_rate": success_rate,
            "eval/episode_gif": wandb.Video(gif_path, fps=fps, format="gif"),
        })
        wb_run.finish()
        print(f"wandb: {wb_run.url}")


if __name__ == "__main__":
    main()
