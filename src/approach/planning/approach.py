"""Planning approach: roll a classical/kinodynamic planner out over the env.

No training and no checkpoint — the planner computes controls online. ``run``
evaluates the selected method over ``eval.episodes`` episodes and prints the
same ``RESULT,...`` line the RL path emits, so sweep scripts parse both.
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.approach.base import BaseApproach
from src.approach.planning import build_planner
from src.approach.rollout import run_episode, save_gif, summarize
from src.env.factory import build_env


class PlanningApproach(BaseApproach):
    """``approach=planning`` — dispatches on ``approach.method``."""

    def build_controller(self, env):
        return build_planner(self.cfg.approach)

    def run(self, cfg: DictConfig) -> None:
        eval_cfg = cfg.get("eval", None)
        n_episodes = int(eval_cfg.get("episodes", 20)) if eval_cfg else 20
        gif_path = eval_cfg.get("gif_path", None) if eval_cfg else None
        render = bool(gif_path)

        env = build_env(cfg)
        planner = self.build_controller(env)

        method = cfg.approach.method
        print(f"approach=planning method={method}  env={cfg.env.get('_name_', 'custom')}  "
              f"episodes={n_episodes}")

        stats_list, frames = [], []
        for ep in range(n_episodes):
            stats, fr = run_episode(env, planner, render=render)
            stats_list.append(stats)
            frames.extend(fr)

        m = summarize(stats_list)
        print(f"  success={m['success_rate']:6.1%}  crash_rate={m['crash_rate']:6.1%}  "
              f"avg_collisions={m['avg_collisions']:5.1f}  "
              f"avg_steps_on_success={m['avg_steps_on_success']:6.1f}")
        # Machine-readable line, identical format to scripts/fasteval.py:
        # RESULT,<mode>,<success>,<crash_rate>,<avg_collisions>,<avg_steps_on_success>
        print(f"RESULT,{method},{m['success_rate']:.4f},{m['crash_rate']:.4f},"
              f"{m['avg_collisions']:.2f},{m['avg_steps_on_success']:.1f}")

        if render and frames and gif_path:
            save_gif(frames, gif_path, int(eval_cfg.get("fps", 15)))
