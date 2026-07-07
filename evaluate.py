"""Evaluate an approach and render episodes to GIF (+ optional wandb).

    # RL: load a trained checkpoint
    python evaluate.py eval.checkpoint=runs/exp/checkpoints/agent_1000000.pt
    python evaluate.py eval.checkpoint=... eval.episodes=5 wandb.enabled=true

    # Planning: no checkpoint — the planner computes controls online
    python evaluate.py approach=planning approach.method=rrt

Thin shim over ``src.approach``: builds the env, asks the approach for an
evaluation ``Controller`` (RL loads the checkpoint; a planner returns itself),
and rolls episodes out through the shared loop in ``src.approach.rollout``.
``run_episode`` / ``save_gif`` are re-exported here for backwards compatibility.
"""
from __future__ import annotations

import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.approach import build_approach
from src.approach.rollout import run_episode, save_gif  # re-exported (back-compat)
from src.env.factory import build_env

__all__ = ["main", "run_episode", "save_gif"]


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, os.getcwd())
    eval_cfg   = cfg.get("eval", OmegaConf.create({}))
    n_episodes = int(eval_cfg.get("episodes", 3))
    gif_path   = eval_cfg.get("gif_path", "episode.gif")
    fps        = int(eval_cfg.get("fps", 15))

    env = build_env(cfg)
    controller = build_approach(cfg).build_controller(env)

    all_stats, all_frames = [], []
    for ep in range(n_episodes):
        print(f"Episode {ep+1}/{n_episodes} ...", end=" ", flush=True)
        stats, frames = run_episode(env, controller, render=True)
        all_stats.append(stats)
        all_frames.extend(frames)
        if frames and ep < n_episodes - 1:
            all_frames.extend([frames[-1]] * 10)  # brief hold between episodes
        print(f"steps={stats['steps']}  reached={stats['both_reached']}  "
              f"rewards={stats['total_reward']}")

    success_rate = float(np.mean([s["both_reached"] for s in all_stats]))
    print(f"\nSuccess rate: {success_rate:.0%}")

    if all_frames:
        save_gif(all_frames, gif_path, fps)

    use_wandb = cfg.get("wandb", OmegaConf.create({})).get("enabled", False)
    if use_wandb:
        import wandb
        wb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=f"eval_{cfg.approach.type}_{cfg.network.type}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.approach.type, cfg.network.type, cfg.obs.type, "eval"],
        )
        wb_run.log({
            "eval/success_rate": success_rate,
            "eval/episode_gif": wandb.Video(gif_path, fps=fps, format="gif"),
        })
        wb_run.finish()
        print(f"wandb: {wb_run.url}")


if __name__ == "__main__":
    main()
