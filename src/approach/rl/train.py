"""SKRL IPPO training, under the `reinforcement_learning` approach.

Env construction lives in `src.env.factory.build_env`, which is paradigm-neutral so
the planning approach builds the same env from the same config.
"""
from __future__ import annotations

import os
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


def build_models(env, cfg, device) -> dict[str, dict]:
    """One policy+value pair per agent, or ONE pair shared by all of them.

    Parameter sharing (``train.share_policy``) exists because the teams here are homogeneous:
    with separate nets, open_cross at N=32 trains 64 networks on 1/32 of the experience each.
    Sharing pools it into one network, which is what pays for running fewer parallel envs.
    skrl keeps one optimizer per agent, so shared weights get N sequential updates per cycle
    and still see N x num_envs x rollouts transitions.
    """
    share = bool(cfg.train.get("share_policy", False))
    first = env.possible_agents[0]
    models: dict[str, dict] = {}
    shared: dict | None = None

    for agent_id in env.possible_agents:
        obs_sp = env.observation_space(agent_id)
        act_sp = env.action_space(agent_id)
        if share and agent_id != first:
            # A mixed team does not even fit the same input layer. Torch would report that as
            # a shape mismatch deep in a forward pass, which does not name the cause.
            ref_obs, ref_act = env.observation_space(first), env.action_space(first)
            if obs_sp.shape != ref_obs.shape or act_sp.shape != ref_act.shape:
                raise ValueError(
                    f"train.share_policy=true needs a homogeneous team, but {agent_id} has "
                    f"obs{obs_sp.shape}/act{act_sp.shape} against {first}'s "
                    f"obs{ref_obs.shape}/act{ref_act.shape}. "
                    "Set train.share_policy=false for a mixed-robot scenario."
                )
        if share:
            if shared is None:
                shared = {"policy": build_policy(obs_sp, act_sp, device, cfg.network).to(device),
                          "value":  build_value(obs_sp, act_sp, device, cfg.network).to(device)}
            models[agent_id] = shared
        else:
            models[agent_id] = {
                "policy": build_policy(obs_sp, act_sp, device, cfg.network).to(device),
                "value":  build_value(obs_sp, act_sp, device, cfg.network).to(device),
            }
    return models


def resolve_num_envs(cfg) -> int:
    """Parallel worlds to run: ``train.num_envs`` verbatim, or derived from
    ``train.envs_x_agents`` so that a sweep over N is a controlled comparison.

    With ``share_policy`` every agent writes into one network, so a PPO update sees
    ``num_agents x num_envs x rollouts`` transitions. Holding num_envs fixed across a sweep
    therefore does NOT hold the training signal fixed -- open_cross at N=32 would get 8x the
    transitions per update that N=4 does, and any difference in the results could be read
    either as the scenario being harder or as N=4 being starved. Fixing the product is what
    makes N the only variable, and it is also what makes large N affordable: cost per timestep
    is O(num_envs x N^2), so the derived value cancels one factor of N.
    """
    target = cfg.train.get("envs_x_agents", None)
    if target is None:
        return int(cfg.train.get("num_envs", 1))
    n_agents = len(cfg.env.agents)
    if int(target) < n_agents:
        raise ValueError(
            f"train.envs_x_agents={target} is below this scenario's {n_agents} agents, so the "
            "derived num_envs would round to 0. Raise it to a multiple of the largest N in "
            "the sweep, or set it to null and give train.num_envs directly."
        )
    return int(target) // n_agents


def run_dir_name(cfg) -> str:
    """Directory for this run, under ``runs/``.

    The env goes FIRST and is not optional: the name used to be
    ``<shaping>_<network>_<obs>``, which is identical for every scenario, so launching
    open_cross at N=4/8/16/32 in the same minute put four jobs in one directory, where they
    overwrote each other's config.yaml and checkpoints. Nothing failed -- the runs simply
    produced one corrupt result instead of four.

    Hydra's chosen config-group name is the scenario's real identity (``open_cross_8_unicycle2``).
    It is only available under @hydra.main, so callers outside it (tests, notebooks) fall back
    to the env's own ``_name_``.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        env_name = HydraConfig.get().runtime.choices["env"]
    except Exception:
        env_name = cfg.env.get("_name_", "custom")
    return f"{env_name}_{cfg.shaping.type}_{cfg.network.type}_{cfg.obs.type}"


def run_training(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    # A 128x128 MLP on a 17-dim observation does not fill a thread pool. Torch defaults to
    # one thread per core and then spends more on synchronising them than on the matmul --
    # measured 16% faster wall-clock at one thread on a 32-core box. It also stops several
    # concurrent runs from fighting over the same cores, which is the actual way to use a
    # many-core machine here: rollout is ~92% env stepping, which is single-threaded Python.
    # An explicit OMP_NUM_THREADS wins, so a larger network can opt out.
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(1)
    # GPU is used automatically when present. It is not the lever: the networks are ~7% of
    # per-step cost here, so a GPU cannot touch the rest and its per-step launch latency can
    # make the rollout slower. Worth re-checking only if the networks grow a lot.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = resolve_num_envs(cfg)
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

    models = build_models(env, cfg, device)
    memories = {a: RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)
                for a in env.possible_agents}

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
    run_dir = run_dir_name(cfg)
    # Seconds, not minutes: two runs of the SAME scenario started together would otherwise
    # still collide, which is what a sweep launched from one loop does.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

    if use_wandb:
        upload_run_artifact(exp_dir)


def upload_run_artifact(exp_dir: pathlib.Path) -> None:
    """Publish the run's config and best policy to W&B so someone else can render it.

    skrl's W&B integration logs scalars and syncs tensorboard; it never uploads the
    checkpoints (no wandb.save, no artifact, verified in skrl/multi_agents/torch/base.py).
    Without this the reviewer gets curves and has to take the rest on trust, because the
    weights exist only on the machine that trained them.

    The artifact preserves the run's own layout — ``config.yaml`` beside
    ``checkpoints/`` — because that is what evaluate.py looks for. After

        wandb artifact get <project>/<name>:latest --root /tmp/r
        python evaluate.py eval.checkpoint=/tmp/r/checkpoints/best_agent.pt

    the env, shaping, obs and network come from the downloaded config, so the reviewer
    renders the same policy without being told a single flag.
    """
    import wandb

    if wandb.run is None:                       # training ran with wandb off, or it died
        return
    best = exp_dir / "checkpoints" / "best_agent.pt"
    cfg_file = exp_dir / "config.yaml"
    if not best.is_file():
        print(f"[warn] no best_agent.pt in {exp_dir} — nothing to publish.")
        return

    # "/" is not legal in an artifact name, and run_dir/timestamp both contain one.
    name = f"{exp_dir.parent.name}_{exp_dir.name}".replace("/", "_")
    art = wandb.Artifact(name, type="model")
    art.add_file(str(best), name="checkpoints/best_agent.pt")
    if cfg_file.is_file():
        art.add_file(str(cfg_file), name="config.yaml")
    wandb.log_artifact(art)
    print(f"Published W&B artifact: {name} (config.yaml + best_agent.pt)")
