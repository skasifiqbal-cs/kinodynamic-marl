"""SKRL IPPO training — fully modular."""
from __future__ import annotations

from datetime import datetime

import torch
from omegaconf import DictConfig, OmegaConf
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer

from src.env.multiagent_nav import MultiAgentNav
from src.init import build_initializer
from src.networks import build_policy, build_value
from src.obs import build_obs_builder
from src.robot import build_robot, load_robot_cfg
from src.shaping import build_potential


def _build_env(cfg: DictConfig) -> MultiAgentNav:
    agent_cfgs = list(cfg.env.agents)
    # env.omega_max_override (null by default) lets the E2 sweep vary the turning
    # constraint without a per-radius robot yaml. Robot configs are loaded outside
    # hydra (by name), so a `robot.omega_max=` CLI override cannot reach them.
    omega_override = cfg.env.get("omega_max_override", None)
    robots = []
    for a in agent_cfgs:
        rc = load_robot_cfg(a.robot)
        if omega_override is not None:
            rc = OmegaConf.merge(
                rc, {"omega_max": float(omega_override), "omega_min": -float(omega_override)}
            )
        robots.append(build_robot(rc))
    # Obstacle-aware potentials (e.g. dijkstra) need world geometry; pass it through.
    obstacles = list(cfg.env.obstacles)
    world_size = float(cfg.env.world_size)
    potentials = [
        build_potential(
            cfg,
            v_max=getattr(r, "v_max", 1.0),
            omega_max=getattr(r, "omega_max", None),
            obstacles=obstacles,
            world_size=world_size,
            robot=r,
        )
        for r in robots
    ]
    # n_dyn: second-order robots expose internal velocity [v, ω] in the obs (else 0).
    n_dyn = robots[0].n_dynamic_features if robots else 0
    obs_builder = build_obs_builder(
        cfg.obs, n_agents=len(robots), n_obstacles=len(obstacles),
        world_size=world_size, n_dyn=n_dyn,
    )
    initializer = build_initializer(cfg.init)
    return MultiAgentNav(cfg, robots, potentials, obs_builder, initializer)


def run_training(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = int(cfg.train.get("num_envs", 1))
    if num_envs > 1:
        # Parallel actors: B independent worlds -> each PPO update sees rollouts x B
        # transitions (lower-variance gradient, denser reach signal). Literature standard.
        from src.env.vec_multiagent import VecMultiAgentNav, VecPettingZooWrapper
        vec = VecMultiAgentNav(lambda: _build_env(cfg), num_envs=num_envs, base_seed=cfg.train.seed)
        env = VecPettingZooWrapper(vec)
    else:
        raw_env = _build_env(cfg)
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
    ippo_cfg["experiment"] = {
        "directory": f"runs/{run_dir}",
        "experiment_name": timestamp,
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
