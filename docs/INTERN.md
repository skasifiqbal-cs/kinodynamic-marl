# Adding a planning method (intern guide)

This repo can solve the multi-robot navigation problem two ways, chosen from
config by the **`approach`** dimension:

```bash
python main.py                                          # approach=reinforcement_learning (train IPPO)
python main.py approach=planning approach.method=rrt    # approach=planning (plan + evaluate)
```

Switch experiments by editing `conf/experiment/local.yaml` (copy it from
`conf/experiment/local.yaml.example`), not `conf/config.yaml`. That file is gitignored,
so your experiment settings never reach a commit and never conflict with anyone else's.
See the "Choosing an experiment" section of the README.

Both share the **same robots, agents, and environment**. Reinforcement learning
trains a neural policy; planning computes controls online with a classical or
kinodynamic planner. Your job: implement the planning methods.

## Where things live

```
src/approach/
  base.py              BaseApproach + Controller (the interfaces)
  rollout.py           the shared episode loop (run_episode) + metrics + save_gif
  rl/                  the RL approach (already done — reference only)
  planning/
    base.py            BasePlanner  (read this — it has the env cheat-sheet)
    __init__.py        build_planner  (the dispatch table — register here)
    rrt.py             <- implement
    kinodynamic_rrt.py <- implement
    optimization.py    done — prioritised minimum-time NLP
    karc.py            done — K-ARC (segmentation + conflict ladder)
    trajopt.py         the CasADi program both of those call
conf/approach/planning.yaml   parameters for each method
```

A planner **is a `Controller`**: implement two methods.

- `reset(env)` — called once per episode, env is live. **Do your planning here**
  and stash the result on `self` (e.g. a per-agent control sequence).
- `act(obs_dict, env)` — called every step. Return `{agent: control}`; usually
  pop the next control from your plan. Controls are clipped to bounds for you.

## The recipe (add a new method, e.g. `prm`)

1. **Create** `src/approach/planning/prm.py`:
   ```python
   from src.approach.planning.base import BasePlanner

   class PRMPlanner(BasePlanner):
       method = "prm"
       def reset(self, env):
           # plan here using env geometry (see cheat-sheet below)
           ...
       def act(self, obs_dict, env):
           # return {agent: control} for this step
           ...
   ```
2. **Register** it in `src/approach/planning/__init__.py` — add to `_PLANNERS`:
   ```python
   from src.approach.planning.prm import PRMPlanner
   _PLANNERS = { ..., "prm": PRMPlanner }
   ```
3. **Add params** in `conf/approach/planning.yaml`:
   ```yaml
   prm:
     n_samples: 500
     k_neighbors: 10
   ```
   They arrive as `self.params` (already resolved to the `prm` block).
4. **Run** it:
   ```bash
   python evaluate.py approach=planning approach.method=prm   # renders a GIF
   ```
5. **Score** it (success / crash / collisions / steps):
   ```bash
   python scripts/fasteval.py approach=planning approach.method=prm eval.episodes=100
   ```

`rrt` and `kinodynamic_rrt` are still stubs and carry a step-by-step TODO in their
docstrings — start with `rrt.py`. `optimization` and `karc` are implemented; read
`trajopt.py` first if you want to see how they talk to the solver. Those two need
CasADi: `pip install -e ".[dev,planning]"`.

## Env cheat-sheet (everything a planner can query)

All are live attributes on the env passed to `reset(env)` / `act(obs, env)`:

| what | attribute |
|---|---|
| obstacles | `env._obstacles` (list of `Obstacle`, `src/collision/shapes.py`) |
| goal pose of agent i | `env._goals[i]` → `[x, y, θ]` |
| current state of agent i | `env._states[i]` |
| robot model of agent i | `env.robots[i]` |
| propagate a control | `env.robots[i].step(state, u, dt)` (RK4) |
| control bounds | `env.robots[i].action_low` / `.action_high` |
| body shape | `env.robots[i].shape` |
| world size | `env._world_size` — the world is `[0, world_size]` on BOTH axes, not a half-extent |
| success threshold | `env.goal_radius` |
| timestep | `env.dt` |

Control vector `u`: `[v, ω]` for the kinematic unicycle, `[a, α]`
(linear/angular **acceleration**) for the second-order `unicycle2`.

## Reuse — don't reinvent

- **Collision checks**: `from src.collision.shapes import collides, collides_wall`.
  Validate a candidate state against `env._obstacles` (+ `robot.shape`) and the
  world walls (`env._world_size`).
- **Occupancy grid + cost-to-go**: `src/shaping/dijkstra_potential.py` already
  builds a clearance-inflated free-space grid (`DijkstraPotential._free`) and an
  8-connected shortest-path field (`._dist_field(goal)`). Reuse it as a sampling
  domain / goal-bias heuristic for grid-based or informed planners.

## What "done" looks like

`python evaluate.py approach=planning approach.method=<yours>` runs an episode
end-to-end and writes a GIF; `scripts/fasteval.py` prints a `RESULT,...` line you
can compare against the RL numbers on the same env.
