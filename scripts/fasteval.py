"""Headless success-rate evaluator — no rendering, fast bulk eval for sanity checks.

For RL, reports BOTH deterministic (mean action) and stochastic (sampled action)
success, because entropy_loss_scale keeps action std high at convergence: the
deterministic mean can be timid while the sampled policy is competent. For
planning there is no such split — a single line labelled by the method is emitted.

Usage (hydra overrides identical to training):
    # RL — a checkpoint from a run trained after config saving needs nothing else
    python scripts/fasteval.py eval.checkpoint=runs/.../checkpoints/agent_400000.pt \
        eval.episodes=100
    # Planning
    python scripts/fasteval.py approach=planning approach.method=karc \
        env=crossing_2agent eval.episodes=100

Use evaluate.py instead when you want to watch an episode rather than count them.

Four entry points, one job each:
    main.py              dispatch on `approach`: train (RL) or plan+evaluate (planning)
    train.py             train an RL policy
    evaluate.py          render an episode to GIF and report success
    scripts/fasteval.py  the same scoring, headless and in bulk — no rendering

Thin shim over ``src.approach``: builds the approach's evaluation controller and
scores it with the shared rollout loop.
"""
from __future__ import annotations

import os
import pathlib
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Python puts THIS file's directory (scripts/) on sys.path, not the repo root, so `src`
# is not importable from here the way it is from evaluate.py at the top level. Resolved
# from __file__ rather than cwd because hydra chdir's into its output dir before main().
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.approach import build_approach  # noqa: E402
from src.approach.rollout import run_episode, summarize  # noqa: E402
from src.env.factory import build_env  # noqa: E402


def _score(env, controller, n_episodes: int) -> dict:
    return summarize([run_episode(env, controller)[0] for _ in range(n_episodes)])


def _print_result(label: str, m: dict) -> None:
    print(f"  {label:13s}: success={m['success_rate']:6.1%}  crash_rate={m['crash_rate']:6.1%}  "
          f"avg_collisions={m['avg_collisions']:5.1f}  "
          f"avg_steps_on_success={m['avg_steps_on_success']:6.1f}")
    # Machine-readable line for sweep scripts (stable, easy to grep/parse):
    # RESULT,<mode>,<success>,<crash_rate>,<avg_collisions>,<avg_steps_on_success>
    print(f"RESULT,{label},{m['success_rate']:.4f},{m['crash_rate']:.4f},"
          f"{m['avg_collisions']:.2f},{m['avg_steps_on_success']:.1f}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, os.getcwd())
    eval_cfg = cfg.get("eval", OmegaConf.create({}))
    n_episodes = int(eval_cfg.get("episodes", 100))
    seed = int(eval_cfg.get("seed", 12345))

    torch.manual_seed(seed)
    env = build_env(cfg)
    env.reset(seed=seed)

    approach = build_approach(cfg)
    approach_type = cfg.approach.type

    if approach_type == "reinforcement_learning":
        if not eval_cfg.get("checkpoint", None):
            print("[fatal] eval.checkpoint=... required for approach=reinforcement_learning")
            return
        print(f"checkpoint: {eval_cfg.get('checkpoint')}")
        print(f"env={cfg.env.get('_name_', 'custom')} approach=rl shaping={cfg.shaping.type} "
              f"init={cfg.init.type} episodes={n_episodes}")
        for label, det in (("deterministic", True), ("stochastic", False)):
            controller = approach.build_controller(env, deterministic=det)
            _print_result(label, _score(env, controller, n_episodes))
    else:  # planning (or any non-RL approach): single deterministic pass
        print(f"env={cfg.env.get('_name_', 'custom')} approach={approach_type} "
              f"method={cfg.approach.get('method', '-')} episodes={n_episodes}")
        controller = approach.build_controller(env)
        _print_result(cfg.approach.get("method", approach_type), _score(env, controller, n_episodes))


if __name__ == "__main__":
    main()
