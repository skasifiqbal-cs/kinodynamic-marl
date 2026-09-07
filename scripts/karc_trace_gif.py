"""Render the K-ARC planning PROCESS as a GIF, one frame per algorithm stage.

The episode GIFs show the plan being executed. This shows how it was arrived at:

    1. kinematic reference paths          (Alg. 1 line 3, Dijkstra descent)
    2. per-segment uncoordinated solve    (Alg. 1 lines 17-18) + the conflicts it produced
    3. each resolution rung in turn       (Alg. 2) + the conflicts left after it
    4. the committed, conflict-free plan

Nothing is recomputed: the planner already builds every one of these and overwrites them,
so `approach.karc.trace=true` keeps them and this walks the list.

    python scripts/karc_trace_gif.py approach=planning env=open_cross_4_unicycle2
    python scripts/karc_trace_gif.py approach=planning env=swap2_unicycle2 \
        eval.gif_path=experiments/swap2_trace.gif
"""
import sys

sys.path.insert(0, ".")

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.approach.planning import build_planner  # noqa: E402
from src.approach.rollout import save_gif  # noqa: E402
from src.env.factory import build_env  # noqa: E402
from src.viz import render_frame_with_shapes  # noqa: E402


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # `approach=planning` is what loads the karc block; the default config selects RL and
    # has no planner settings at all, so say that rather than failing on a missing key.
    if cfg.approach.get("type") != "planning" or "karc" not in cfg.approach:
        raise SystemExit("pass approach=planning (and approach.method=karc, the default)")
    OmegaConf.set_struct(cfg, False)
    cfg.approach.method = "karc"
    cfg.approach.karc.trace = True

    out = cfg.eval.get("gif_path", None) or "karc_trace.gif"
    fps = int(cfg.eval.get("fps", 15))
    hold = fps                            # one second per stage

    env = build_env(cfg)
    env.reset(seed=0)
    planner = build_planner(cfg.approach)
    planner.reset(env)

    if not planner.trace:
        raise RuntimeError("planner recorded no stages — is approach.method=karc?")

    states = [s.copy() for s in env._states]      # robots stay at their starts
    frames = []
    for stage in planner.trace:
        frame = render_frame_with_shapes(
            states=states,
            robot_shapes=[r.shape for r in env.robots],
            goals=env._goals,
            obstacles=env._obstacles,
            trails=[[p for p in path] for path in stage["paths"]],
            world_size=env._world_size,
            reached=[False] * env._n,
            step=0,
            title=stage["label"],
            markers=stage["markers"],
            goal_radius=env.goal_radius,
        )
        frames.extend([frame] * hold)
        print(f"  {stage['label']}")

    save_gif(frames, out, fps)
    print(f"{len(planner.trace)} stages → {out}")


if __name__ == "__main__":
    main()
