"""Evaluate an approach and render episodes to GIF (+ optional wandb).

    # RL: a checkpoint is enough. env/shaping/obs/init/network come from the run.
    python evaluate.py eval.checkpoint=runs/exp/2026-08-24_19-36/checkpoints/agent_400000.pt

    # Anything you type still wins over what the run recorded.
    python evaluate.py eval.checkpoint=... env=gap_2agent eval.episodes=5

    # Planning: no checkpoint — the planner computes controls online
    python evaluate.py approach=planning approach.method=rrt

Thin shim over ``src.approach``: builds the env, asks the approach for an
evaluation ``Controller`` (RL loads the checkpoint; a planner returns itself),
and rolls episodes out through the shared loop in ``src.approach.rollout``.
``run_episode`` / ``save_gif`` are re-exported here for backwards compatibility.
"""
from __future__ import annotations

import os
import pathlib
import sys

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.approach import build_approach
from src.approach.rollout import run_episode, save_gif  # re-exported (back-compat)
from src.env.factory import build_env

__all__ = ["main", "run_episode", "save_gif"]

# Config groups: `env=gap_2agent` names a FILE to compose, so it cannot be applied as a
# plain key=value the way `eval.episodes=5` can. Hydra has already composed these into
# cfg, so for a group the user typed we take cfg's version wholesale.
_GROUPS = frozenset({"env", "shaping", "obs", "init", "network", "train", "approach"})


def merge_saved(cfg: DictConfig, saved: DictConfig, typed: list[str]) -> DictConfig:
    """Saved run config as the base; anything typed on the command line wins.

    ``typed`` is Hydra's raw override list (``["env=gap_2agent", "eval.episodes=5"]``).
    Without it there is no way to tell an override the user asked for from a default
    Hydra filled in, and the defaults would silently overwrite the run's own settings —
    which is the whole thing this exists to prevent.
    """
    groups, dotlist = OmegaConf.create({}), []
    for ov in typed:
        key = ov.split("=", 1)[0].lstrip("+~")
        if key in _GROUPS:
            groups[key] = cfg[key]
        else:
            dotlist.append(ov.lstrip("+~"))
    return OmegaConf.merge(saved, groups, OmegaConf.from_dotlist(dotlist))


def restore_from_checkpoint(cfg: DictConfig) -> DictConfig:
    """Recover the training config sitting next to ``eval.checkpoint``.

    Layout is ``<run>/config.yaml`` alongside ``<run>/checkpoints/agent_N.pt``, written
    by src/approach/rl/train.py. Runs from before that are left alone — better to
    evaluate with what the user asked for than to guess the env from a timestamp.
    """
    ckpt = cfg.get("eval", OmegaConf.create({})).get("checkpoint", None)
    if not ckpt:
        return cfg
    saved_path = pathlib.Path(ckpt).parent.parent / "config.yaml"
    if not saved_path.is_file():
        print(f"[warn] no config.yaml in {saved_path.parent} — this run predates config "
              f"saving, so env/shaping/obs must be passed by hand.")
        return cfg
    merged = merge_saved(cfg, OmegaConf.load(saved_path), HydraConfig.get().overrides.task)
    print(f"Restored run config: {saved_path}")
    return merged


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, os.getcwd())
    cfg = restore_from_checkpoint(cfg)
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
        # cfg.network exists only for the RL approach (planning drops the group).
        net = cfg.network.type if "network" in cfg else cfg.approach.get("method", "n/a")
        wb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", None),
            name=f"eval_{cfg.approach.type}_{net}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.approach.type, net, cfg.obs.type, "eval"],
        )
        wb_run.log({
            "eval/success_rate": success_rate,
            "eval/episode_gif": wandb.Video(gif_path, fps=fps, format="gif"),
        })
        wb_run.finish()
        print(f"wandb: {wb_run.url}")


if __name__ == "__main__":
    main()
