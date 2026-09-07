"""Render the K-ARC planning PROCESS as a GIF, one frame per algorithm stage.

The episode GIFs show the finished plan being executed. This shows how it was arrived
at, with the robots DRIVEN along each candidate trajectory as it is proposed:

    1. kinematic reference paths          (Alg. 1 line 3, Dijkstra descent)
    2. per-segment uncoordinated solve    (Alg. 1 lines 17-18) + the conflicts it produced
    3. each resolution rung in turn       (Alg. 2) + the conflicts left after it
    4. the committed, conflict-free plan

Nothing is recomputed: the planner already builds every one of these and overwrites them,
so `approach.karc.trace=true` keeps them and this walks the list.

    python scripts/karc_trace_gif.py approach=planning env=open_cross_4_unicycle2
    # every 2nd planned step instead of every 3rd (slower, smoother)
    python scripts/karc_trace_gif.py approach=planning env=swap2_unicycle2 +frame_skip=2
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

    env = build_env(cfg)
    env.reset(seed=0)
    planner = build_planner(cfg.approach)
    planner.reset(env)

    if not planner.trace:
        raise RuntimeError("planner recorded no stages — is approach.method=karc?")

    shapes = [r.shape for r in env.robots]
    start = [s.copy() for s in env._states]
    skip = int(cfg.get("frame_skip", 3))

    def draw(states, trails, label, markers):
        return render_frame_with_shapes(
            states=states, robot_shapes=shapes, goals=env._goals,
            obstacles=env._obstacles, trails=trails, world_size=env._world_size,
            reached=[False] * env._n, step=0, title=label, markers=markers,
            goal_radius=env.goal_radius,
        )

    frames = []
    for stage in planner.trace:
        static = [list(p) for p in stage["static"]]
        anim = stage["anim"]
        label, markers = stage["label"], stage["markers"]

        if not anim or max(len(a) for a in anim) < 2:
            # A path with no dynamics (the kinematic reference): nothing to drive.
            frames.extend([draw(start, static, label, markers)] * fps)
            print(f"  {label}  [still]")
            continue

        # Drive every robot along its candidate trajectory, holding the short ones at
        # their last state so the frame count is the longest robot's, not the shortest.
        horizon = max(len(a) for a in anim)
        for t in range(0, horizon, skip):
            states = [a[min(t, len(a) - 1)] for a in anim]
            trails = [static[i] + [a[k, :2] for k in range(min(t, len(a) - 1) + 1)]
                      for i, a in enumerate(anim)]
            frames.append(draw(states, trails, label, markers))
        # Hold on the finished trajectory, with the conflicts it produced still marked.
        end = [a[-1] for a in anim]
        end_trails = [static[i] + [p for p in a[:, :2]] for i, a in enumerate(anim)]
        frames.extend([draw(end, end_trails, label, markers)] * (fps // 2))
        print(f"  {label}  [{horizon} steps]")

    save_gif(frames, out, fps)
    print(f"{len(planner.trace)} stages, {len(frames)} frames → {out}")


if __name__ == "__main__":
    main()
