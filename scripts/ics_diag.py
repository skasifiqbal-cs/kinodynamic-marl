# Diagnostic for the "commitment horizon" hypothesis: does the reciprocal braking
# margin go negative BEFORE contact, and by how much? Answers whether swap2 fails
# from late credit (timing) or from something else.
"""Is swap2 failing from DELAYED credit, or from IRREVERSIBILITY + exploration?
Measures when the reciprocal braking margin goes negative vs when contact happens."""
import sys; sys.path.insert(0, ".")
import hydra
import numpy as np
from omegaconf import DictConfig

from src.approach import build_approach
from src.conflict import braking_margin
from src.env.factory import build_env


def margins(env):
    """Reciprocal braking margin for every pair, via the shared predicate."""
    out = []
    for i in range(len(env.robots)):
        for j in range(i + 1, len(env.robots)):
            si, sj = env._states[i], env._states[j]
            out.append(braking_margin(
                si[:2], si[3] * np.array([np.cos(si[2]), np.sin(si[2])]),
                env.robots[i].shape.bounding_radius, env.robots[i].a_max,
                sj[:2], sj[3] * np.array([np.cos(sj[2]), np.sin(sj[2])]),
                env.robots[j].shape.bounding_radius, env.robots[j].a_max,
                conservative=False,   # closing-speed form: what this diagnostic always used
            ))
    return out


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    env = build_env(cfg); ctrl = build_approach(cfg).build_controller(env)
    first_ics, first_coll, n_ep = [], [], 5
    for ep in range(n_ep):
        obs, _ = env.reset(); ctrl.reset(env)
        ics_t, coll_t, prev_cc = None, None, env._collision_count
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(ctrl.act(obs, env))
            if ics_t is None and min(margins(env)) <= 0: ics_t = t
            if coll_t is None and env._collision_count > prev_cc: coll_t = t
            prev_cc = env._collision_count
            if all(term.values()) or all(trunc.values()): break
        first_ics.append(ics_t); first_coll.append(coll_t)
    f = lambda L: [("-" if x is None else str(x)) for x in L]
    print(f"  first ICS entry  : {f(first_ics)}")
    print(f"  first collision  : {f(first_coll)}")
main()
