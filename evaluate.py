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

from src.robot import build_robot, load_robot_cfg
from src.shaping import build_potential
from src.obs import build_obs_builder
from src.init import build_initializer
from src.env.multiagent_nav import MultiAgentNav
from src.networks import build_policy
from src.viz import render_frame


def _build_env(cfg: DictConfig) -> MultiAgentNav:
    agent_cfgs = list(cfg.env.agents)
    robots = [build_robot(load_robot_cfg(a.robot)) for a in agent_cfgs]
    potentials = [build_potential(cfg, v_max=getattr(r, "v_max", 1.0)) for r in robots]
    obs_builder = build_obs_builder(cfg.obs, n_agents=len(robots), n_obstacles=len(cfg.env.obstacles))
    initializer = build_initializer(cfg.init)
    return MultiAgentNav(cfg, robots, potentials, obs_builder, initializer)


def run_episode(env, policies, device, render=True, frame_skip=2):
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
            frames.append(render_frame(
                states=[s.copy() for s in env._states],
                goals=[g.copy() for g in env._goals],
                obstacles=env._obstacles,
                trails=[list(t) for t in trails],
                world_size=env.cfg.env.world_size,
                reached=list(env._reached),
                step=step,
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

    # Rebuild policies (same architecture as training)
    policies = {}
    for i, agent_id in enumerate(env.possible_agents):
        obs_sp = env.observation_space(agent_id)
        act_sp = env.action_space(agent_id)
        policies[agent_id] = build_policy(obs_sp, act_sp, device, cfg.network).to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        for agent_id, policy in policies.items():
            key = f"{agent_id}/policy"
            if key in ckpt:
                policy.load_state_dict(ckpt[key], strict=False)
                print(f"Loaded {key}")
            else:
                print(f"[warn] {key} not in checkpoint — random weights")
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
        stats, frames = run_episode(env, policies, device)
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
