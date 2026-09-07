"""Render the K-ARC planning PROCESS as a GIF, one frame per algorithm stage.

The episode GIFs show the finished plan being executed. This shows how it was arrived
at, with the robots DRIVEN along each candidate trajectory as it is proposed:

    1. kinematic reference paths          (Alg. 1 line 3, Dijkstra descent) -- driven
       uncoordinated, so the robots collide; that is what the rest of the algorithm is for
    2. per-segment uncoordinated solve    (Alg. 1 lines 17-18) + the conflicts it produced
    3. each resolution rung in turn       (Alg. 2) + the conflicts left after it
    4. the committed, conflict-free plan

Nothing is recomputed: the planner already builds every one of these and overwrites them,
so `approach.karc.trace=true` keeps them and this walks the list.

    python scripts/karc_trace_gif.py approach=planning env=open_cross_4_unicycle2
    # every 2nd planned step instead of every 3rd (slower, smoother)
    python scripts/karc_trace_gif.py approach=planning env=open_cross_8_unicycle2 +frame_skip=2
    python scripts/karc_trace_gif.py approach=planning env=swap2_unicycle2 \
        eval.gif_path=experiments/swap2_trace.gif
"""
import sys

sys.path.insert(0, ".")

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.approach.planning import build_planner  # noqa: E402
from src.approach.rollout import save_gif  # noqa: E402
from src.collision.shapes import collides  # noqa: E402
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

    def touching(states):
        """Which robots are in body-on-body contact right now.

        The same predicate the env scores collisions with, so what the animation marks red
        is a real overlap of the actual shapes -- not a centre-distance proxy, and not the
        planner's conflict test, which is a different (purely geometric) thing.
        """
        hit = [False] * len(states)
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                pi = (float(states[i][0]), float(states[i][1]), float(states[i][2]))
                pj = (float(states[j][0]), float(states[j][1]), float(states[j][2]))
                if collides(shapes[i], pi, shapes[j], pj):
                    hit[i] = hit[j] = True
        return hit

    def draw(states, trails, label, markers, waypoints):
        return render_frame_with_shapes(
            states=states, robot_shapes=shapes, goals=env._goals,
            obstacles=env._obstacles, trails=trails, world_size=env._world_size,
            reached=[False] * env._n, step=0, title=label, markers=markers,
            goal_radius=env.goal_radius, waypoints=waypoints,
            highlight=touching(states),
        )

    frames = []
    for stage in planner.trace:
        static = [list(p) for p in stage["static"]]
        anim = stage["anim"]
        label, markers, wp = stage["label"], stage["markers"], stage["waypoints"]

        if not anim or max(len(a) for a in anim) < 2:
            # A path with no dynamics (the kinematic reference): nothing to drive.
            frames.extend([draw(start, static, label, markers, wp)] * fps)
            print(f"  {label}  [still]")
            continue

        # Drive every robot along its candidate trajectory, holding the short ones at
        # their last state so the frame count is the longest robot's, not the shortest.
        horizon = max(len(a) for a in anim)
        at = lambda t: [a[min(t, len(a) - 1)] for a in anim]   # noqa: E731

        # Contact is often only a handful of steps long -- 5 of 160 on the reference paths
        # -- so plain subsampling can step clean over the collision this is meant to show.
        # Every contact step is kept regardless of frame_skip, and the first one is held.
        contact = [t for t in range(horizon) if any(touching(at(t)))]
        shown = sorted(set(range(0, horizon, skip)) | set(contact))

        for t in shown:
            states = at(t)
            trails = [static[i] + [a[k, :2] for k in range(min(t, len(a) - 1) + 1)]
                      for i, a in enumerate(anim)]
            frame = draw(states, trails, label, markers, wp)
            frames.append(frame)
            if contact and t == contact[0]:
                frames.extend([frame] * fps)
        # Hold on the finished trajectory, with the conflicts it produced still marked.
        end = [a[-1] for a in anim]
        end_trails = [static[i] + [p for p in a[:, :2]] for i, a in enumerate(anim)]
        frames.extend([draw(end, end_trails, label, markers, wp)] * (fps // 2))
        print(f"  {label}  [{horizon} steps"
              + (f", {len(contact)} in contact]" if contact else "]"))

    save_gif(frames, out, fps)
    print(f"{len(planner.trace)} stages, {len(frames)} frames → {out}")


if __name__ == "__main__":
    main()
