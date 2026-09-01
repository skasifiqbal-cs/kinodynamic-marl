# Diagnostic: distinguishes "policy is cautious but progressing" from "policy froze".
# Joint success (all_reached) reports 0%% even when one agent is 0.04 m from its goal,
# so per-agent closest-approach and path length are what actually tell you which.
"""Did the policy make progress, or just stop moving? Fewer collisions is only a
virtue if the robots are still travelling toward their goals."""
import sys; sys.path.insert(0, ".")
import hydra
import numpy as np
from omegaconf import DictConfig

from src.approach import build_approach
from src.env.factory import build_env


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    env = build_env(cfg)
    ctrl = build_approach(cfg).build_controller(env)
    for ep in range(3):
        obs, _ = env.reset()
        ctrl.reset(env)
        starts = [np.linalg.norm(env._states[i][:2] - env._goals[i][:2]) for i in range(len(env.robots))]
        mind = list(starts); path = [0.0] * len(env.robots); prev = [s[:2].copy() for s in env._states]
        vs = [[] for _ in env.robots]
        for t in range(env.max_steps):
            act = ctrl.act(obs, env)
            obs, _, term, trunc, _ = env.step(act)
            for i in range(len(env.robots)):
                d = np.linalg.norm(env._states[i][:2] - env._goals[i][:2])
                mind[i] = min(mind[i], d)
                path[i] += np.linalg.norm(env._states[i][:2] - prev[i]); prev[i] = env._states[i][:2].copy()
                vs[i].append(env._states[i][3])
            if all(term.values()) or all(trunc.values()): break
        print(f"ep{ep}: " + "  ".join(
            f"agent{i}: closest={mind[i]:.2f} path={path[i]:.2f} "
            f"mean_v={np.mean(vs[i]):+.3f} mean|v|={np.mean(np.abs(vs[i])):.3f} "
            f"frac_reverse={np.mean(np.array(vs[i])<-0.02):.0%}"
            for i in range(len(env.robots))))
main()
