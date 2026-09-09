#!/usr/bin/env python3
"""Is an Open/Cluttered Cross instance actually SOLVABLE, or is K-ARC failing on a
solvable one?

    .venv/bin/python scripts/feasibility_certificate.py env=open_cross_32_unicycle2

The question matters because "no method scales past 8 robots" (K-ARC SS V-D) and "0.22 m of
lateral room at N=32" are both statements about instances being HARD. Neither proves one is
impossible, and a planner that fails on a solvable instance is a planner with a gap.

What this produces is a witness, not an opinion. The abstraction is a RESTRICTION of the
real problem, never a relaxation:

  * one robot moves at a time, so every other robot is a static obstacle and the multi-robot
    coupling disappears by construction;
  * robots go start -> STAGING -> goal, because in Open Cross every goal is occupied by
    another robot's start (the rows are swaps), so plain sequential motion deadlocks on the
    first robot. Staging cells sit in the empty interior, clear of both columns, and are
    vacated in the same order they were filled;
  * each move is STOP-AND-GO -- turn in place to face the next waypoint, accelerate, brake
    back to rest -- which a second-order unicycle can always execute inside its own
    a_max/alpha_max, so a discrete plan lifts to a real trajectory rather than assuming one;
  * the lifted trajectory is then checked with the SAME collision code and the SAME
    integrator the env uses, not with the grid it was planned on. A certificate that trusts
    its own abstraction certifies nothing.

Therefore: witness found and verified  =>  the instance is SOLVABLE, and any planner that
fails it fails for algorithmic reasons.

The converse does NOT hold. Sequential search is incomplete, so no witness means UNKNOWN,
never UNSAT. Proving unsolvability needs the other direction -- an over-approximation where
every robot is strictly MORE capable than the real one -- and a complete solver over it.

The witness is deliberately slow (robots move one at a time), so its makespan is an upper
bound on what is achievable, never an estimate of it. `steps_vs_budget` reports whether it
also fits the horizon K-ARC is given, which is the weaker, separate question.
"""
import sys

sys.path.insert(0, ".")
import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.approach.planning import geometric_rrt  # noqa: E402
from src.collision.shapes import Obstacle, collides  # noqa: E402
from src.env.factory import build_env  # noqa: E402


def _park(env, poses, skip):
    """Every robot except `skip`, as a static obstacle at its own footprint and heading."""
    return [Obstacle(float(p[0]), float(p[1]), env.robots[i].shape, float(p[2]))
            for i, p in enumerate(poses) if i != skip]


def _staging(env, starts, goals, n, sep):
    """N mutually clear holding cells in the empty interior.

    They must be clear of every start AND every goal, or a robot parked on one would block
    the column it later has to enter. The interior of a cross scenario is empty by
    construction, which is exactly the room a swap needs and the row itself does not have.
    """
    w = env._world_size
    lo, hi = 0.22 * w, 0.78 * w             # inside the columns at either margin
    keep = [np.asarray(p, float)[:2] for p in list(starts) + list(goals)]
    grid = np.arange(lo, hi, max(sep * 1.6, 1.2))
    out = []
    for x in grid:
        for y in grid:
            c = np.array([x, y])
            if any(float(np.linalg.norm(c - k)) < sep * 1.6 for k in keep):
                continue
            if any(collides(env.robots[0].shape, (float(x), float(y), 0.0),
                            ob.shape, ob.pose) for ob in env._obstacles):
                continue
            if any(float(np.linalg.norm(c - o)) < sep * 1.6 for o in out):
                continue
            out.append(c)
    return None if len(out) < n else [np.array([p[0], p[1], 0.0]) for p in out[:n]]


def _profile(delta, dt, acc_max, vel_max):
    """Symmetric accelerate/decelerate profile that covers exactly `delta` and ends at rest.

    Solved on the INTEGRATOR the env actually uses (semi-implicit Euler), not in continuous
    time, so the leg lands on its target instead of near it. n steps up then n steps down
    advance `a*dt^2*n^2`, so n fixes the acceleration: pick the smallest n whose implied
    a and peak velocity both sit inside the robot's bounds. Feedback bang-bang was tried
    first and chatters -- at dt=0.1 it never simultaneously satisfies |err| and |v| small.
    """
    d = abs(float(delta))
    if d < 1e-12:
        return []
    n = max(1, int(np.ceil(np.sqrt(d / (dt * dt * acc_max)))))
    while True:
        a = d / (dt * dt * n * n)
        if a <= acc_max + 1e-12 and n * a * dt <= vel_max + 1e-12:
            break
        n += 1
    a *= float(np.sign(delta))
    return [a] * n + [-a] * n


def _turn_then_go(robot, state, target, dt):
    """One stop-and-go leg: turn in place to face `target`, drive to it, end at rest.

    Both phases start and end at rest, so the leg is realisable by ANY second-order unicycle
    that has the bounds this one reports -- which is what lets a discrete plan stand as a
    statement about the real robot.
    """
    st = np.asarray(state, dtype=np.float64).copy()
    out = []
    want = float(np.arctan2(target[1] - st[1], target[0] - st[0]))
    turn = float(np.arctan2(np.sin(want - st[2]), np.cos(want - st[2])))
    dist = float(np.linalg.norm(np.asarray(target)[:2] - st[:2]))

    seq = [(0.0, al) for al in _profile(turn, dt, robot.alpha_max, robot.omega_max)]
    seq += [(a, 0.0) for a in _profile(dist, dt, robot.a_max, robot.v_max)]
    for a, al in seq:
        u = np.array([float(np.clip(a, robot.a_min, robot.a_max)),
                      float(np.clip(al, robot.alpha_min, robot.alpha_max))])
        st = robot.step(st, u, dt)
        out.append(st.copy())
    return np.asarray(out) if out else None


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    env = build_env(cfg)
    env.reset()
    n = env._n
    dt = env.dt
    clearance = float(cfg.approach.get("trajopt", {}).get("clearance", 0.05))

    poses = [np.asarray(s, float)[:3].copy() for s in env._states]
    goals = [np.asarray(g, float)[:3].copy() for g in env._goals]
    states = [np.asarray(s, float).copy() for s in env._states]
    sep = 2.0 * max(r.shape.bounding_radius for r in env.robots) + clearance

    stage = _staging(env, poses, goals, n, sep)
    if stage is None:
        print("RESULT,UNKNOWN,staging,no_free_cells")
        return

    tracks, total = [], 0

    def move(i, target):
        """Walk robot i to `target` around everything currently parked. Returns False if
        no path exists -- which is UNKNOWN, not proof of anything."""
        nonlocal total
        r = env.robots[i]
        path = geometric_rrt.plan_path(
            poses[i][:2], np.asarray(target)[:2], env._obstacles + _park(env, poses, i),
            env._world_size, radius=r.shape.bounding_radius + clearance,
            max_iters=20000, step=0.6, goal_bias=0.15, shortcut=True,
            rng=np.random.default_rng(i))
        if path is None:
            return False
        leg = []
        for wp in np.asarray(path)[1:]:
            piece = _turn_then_go(r, states[i], wp, dt)
            if piece is None:
                return False
            leg.append(piece)
            states[i] = piece[-1].copy()
        if leg:
            traj = np.vstack(leg)
            tracks.append((i, traj))
            total += len(traj)
            poses[i] = states[i][:3].copy()
        return True

    for phase, targets in (("stage", stage), ("goal", goals)):
        for i in range(n):
            if not move(i, targets[i]):
                print(f"RESULT,UNKNOWN,robot_{i},no_path_to_{phase}")
                return

    # ---- verify the WITNESS, with the env's own collision code -----------------
    bad_obs = bad_bot = 0
    scene = [np.asarray(s, float)[:3].copy() for s in env._states]
    for i, traj in tracks:
        for st in traj:
            scene[i] = st[:3].copy()
            for ob in env._obstacles:
                if collides(env.robots[i].shape, tuple(scene[i]), ob.shape, ob.pose):
                    bad_obs += 1
            for j in range(n):
                if j != i and collides(env.robots[i].shape, tuple(scene[i]),
                                       env.robots[j].shape, tuple(scene[j])):
                    bad_bot += 1
    reached = sum(float(np.linalg.norm(states[i][:2] - goals[i][:2])) < env.goal_radius
                  for i in range(n))
    budget = int(getattr(env, "max_steps", 0))
    ok = bad_obs == 0 and bad_bot == 0 and reached == n
    print(f"RESULT,{'SOLVABLE' if ok else 'UNKNOWN'},{env._n},"
          f"obstacle_hits={bad_obs},robot_hits={bad_bot},goals={reached}/{n},"
          f"witness_steps={total},budget={budget},"
          f"fits_budget={'yes' if budget and total <= budget else 'no'}")


if __name__ == "__main__":
    main()
