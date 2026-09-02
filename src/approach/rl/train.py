"""SKRL IPPO training — relocated under the `reinforcement_learning` approach.

Logic is unchanged from the original `src/training/train.py`; only the env
builder moved to `src.env.factory.build_env` (paradigm-neutral). Checkpoints,
IPPO config, and run-directory layout are byte-for-byte identical.
"""
from __future__ import annotations

import pathlib
from datetime import datetime

import torch
from omegaconf import DictConfig, OmegaConf
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer

from src.env.factory import build_env
from src.networks import build_policy, build_value


def run_training(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = int(cfg.train.get("num_envs", 1))
    if num_envs > 1:
        # Parallel actors: B independent worlds -> each PPO update sees rollouts x B
        # transitions (lower-variance gradient, denser reach signal). Literature standard.
        from src.env.vec_multiagent import VecMultiAgentNav, VecPettingZooWrapper
        vec = VecMultiAgentNav(lambda: build_env(cfg), num_envs=num_envs, base_seed=cfg.train.seed)
        env = VecPettingZooWrapper(vec)
    else:
        raw_env = build_env(cfg)
        env = wrap_env(raw_env, wrapper="pettingzoo")

    rollouts = cfg.train.rollouts

    models: dict[str, dict] = {}
    memories: dict[str, object] = {}

    for agent_id in env.possible_agents:
        obs_sp = env.observation_space(agent_id)
        act_sp = env.action_space(agent_id)
        models[agent_id] = {
            "policy": build_policy(obs_sp, act_sp, device, cfg.network).to(device),
            "value":  build_value(obs_sp, act_sp, device, cfg.network).to(device),
        }
        memories[agent_id] = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))

    ippo_cfg = IPPO_DEFAULT_CONFIG.copy()
    ippo_cfg["rollouts"]              = rollouts
    ippo_cfg["learning_rate"]         = cfg.train.learning_rate
    ippo_cfg["discount_factor"]       = cfg.train.discount
    ippo_cfg["lambda"]                = cfg.train.lambda_
    ippo_cfg["ratio_clip"]            = cfg.train.clip_ratio
    ippo_cfg["learning_epochs"]       = cfg.train.epochs
    ippo_cfg["mini_batches"]          = cfg.train.mini_batches
    ippo_cfg["entropy_loss_scale"]       = cfg.train.entropy_loss_scale
    ippo_cfg["kl_threshold"]             = cfg.train.kl_threshold
    ippo_cfg["clip_predicted_values"]    = True
    # KL-adaptive LR: drops lr when an update spikes KL.
    # CRITICAL: cap max_lr. The skrl default max_lr=0.01 let lr climb to 200x base
    # during stable periods, detonating the policy (the reach-then-diverge oscillation).
    ippo_cfg["learning_rate_scheduler"]        = KLAdaptiveLR
    ippo_cfg["learning_rate_scheduler_kwargs"] = {
        "kl_threshold": cfg.train.kl_threshold,
        "min_lr": float(cfg.train.get("min_lr", 1.0e-5)),
        # ceiling on the adaptive LR. Lower = less post-convergence detonation (policy
        # diverging after it already reached 100%). Configurable per run.
        "max_lr": float(cfg.train.get("max_lr", 1.0e-4)),
    }
    ippo_cfg["state_preprocessor"]       = RunningStandardScaler
    ippo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space(env.possible_agents[0]), "device": device}
    if cfg.train.get("value_preprocessor", True):
        ippo_cfg["value_preprocessor"]       = RunningStandardScaler
        ippo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    else:
        ippo_cfg["value_preprocessor"]       = None
    run_dir = f"{cfg.shaping.type}_{cfg.network.type}_{cfg.obs.type}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Save the config NEXT TO the checkpoints. Hydra already writes it, but into its own
    # outputs/<date>/<time>/ tree with no link back here, and the two timestamps do not
    # even agree — so given a checkpoint there was no way to tell which env trained it.
    # evaluate.py reads this back, which is what makes a checkpoint path self-sufficient.
    exp_dir = pathlib.Path("runs") / run_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")
    ippo_cfg["experiment"] = {
        "directory": f"runs/{run_dir}",
        "experiment_name": timestamp,
        "write_interval": 1000,
        # A short run used to finish with an empty checkpoints/ dir, so there was
        # nothing to evaluate or render until 50k steps had gone by.
        "checkpoint_interval": int(cfg.train.get("checkpoint_interval", 50_000)),
        "wandb": use_wandb,
        "wandb_kwargs": {
            "project": wandb_cfg.get("project", "kinodynamic-rl"),
            "entity": wandb_cfg.get("entity", None),
            "tags": [cfg.shaping.type, cfg.network.type, cfg.obs.type],
            "config": {
                "shaping":  cfg.shaping.type,
                "network":  cfg.network.type,
                "obs":      cfg.obs.type,
                "init":     cfg.init.type,
                "env":      cfg.env.get("_name_", "custom"),
            },
        } if use_wandb else {},
    }

    agent = IPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces={a: env.observation_space(a) for a in env.possible_agents},
        action_spaces={a: env.action_space(a) for a in env.possible_agents},
        device=device,
        cfg=ippo_cfg,
    )

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": cfg.train.timesteps, "headless": True},
    )
    trainer.train()
