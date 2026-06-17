"""SKRL IPPO training — fully modular."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory, SequenceMemory
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from src.env.multiagent_nav import MultiAgentNav
from src.robot import build_robot, load_robot_cfg
from src.shaping import build_potential
from src.obs import build_obs_builder
from src.init import build_initializer
from src.networks import build_policy, build_value


def _build_env(cfg: DictConfig) -> MultiAgentNav:
    agent_cfgs = list(cfg.env.agents)
    robots = [build_robot(load_robot_cfg(a.robot)) for a in agent_cfgs]
    potentials = [build_potential(cfg, v_max=getattr(r, "v_max", 1.0)) for r in robots]
    obs_builder = build_obs_builder(cfg.obs, n_agents=len(robots), n_obstacles=len(cfg.env.obstacles))
    initializer = build_initializer(cfg.init)
    return MultiAgentNav(cfg, robots, potentials, obs_builder, initializer)


def run_training(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_env = _build_env(cfg)
    env = wrap_env(raw_env, wrapper="pettingzoo")

    rollouts = cfg.train.rollouts
    use_gru = cfg.network.type == "gru"

    models: dict[str, dict] = {}
    memories: dict[str, object] = {}

    for agent_id in env.possible_agents:
        obs_sp = env.observation_space(agent_id)
        act_sp = env.action_space(agent_id)
        models[agent_id] = {
            "policy": build_policy(obs_sp, act_sp, device, cfg.network).to(device),
            "value":  build_value(obs_sp, act_sp, device, cfg.network).to(device),
        }
        if use_gru:
            memories[agent_id] = SequenceMemory(
                memory_size=rollouts,
                sequence_length=int(cfg.network.get("sequence_length", 16)),
                num_envs=1,
                device=device,
            )
        else:
            memories[agent_id] = RandomMemory(memory_size=rollouts, num_envs=1, device=device)

    wandb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wandb_cfg.get("enabled", False))

    ippo_cfg = IPPO_DEFAULT_CONFIG.copy()
    ippo_cfg["rollouts"]         = rollouts
    ippo_cfg["learning_rate"]    = cfg.train.learning_rate
    ippo_cfg["discount_factor"]  = cfg.train.discount
    ippo_cfg["lambda"]           = cfg.train.lambda_
    ippo_cfg["ratio_clip"]       = cfg.train.clip_ratio
    ippo_cfg["learning_epochs"]  = cfg.train.epochs
    ippo_cfg["mini_batches"]     = cfg.train.mini_batches
    ippo_cfg["experiment"] = {
        "directory": "runs",
        "experiment_name": f"{cfg.shaping.type}_{cfg.network.type}_{cfg.obs.type}",
        "write_interval": 1000,
        "checkpoint_interval": 50_000,
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
