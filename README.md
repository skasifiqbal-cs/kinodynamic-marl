# Kinodynamic RL — Second-Order Multi-Robot Navigation

Multi-robot navigation for **second-order (kinodynamic)** non-holonomic robots:
acceleration-controlled unicycles that carry momentum and must brake to stop. Robots
reach goals while routing around obstacles and avoiding each other.

The same problem is solved two ways, chosen by one config field:

| `approach=` | What it does | Entry |
|---|---|---|
| `reinforcement_learning` (default) | trains decentralized IPPO policies with obstacle-aware potential-based shaping | `train.py` |
| `planning` | computes controls online — sampling-based or minimum-time trajectory optimisation, including a K-ARC reimplementation | `evaluate.py` |

Both build the **same env from the same config**, so a scenario is described once and
either approach can be pointed at it. Every component (robot, observation, initializer,
shaping potential, planner) is swappable via a single config field.

See `paper/` and `notes/` for write-ups and results.

## Stack

| Component | Choice |
|---|---|
| Config | Hydra 1.3 |
| Tensors / nets | PyTorch |
| Multi-agent RL | skrl (IPPO) |
| Env contract | PettingZoo Parallel API |
| Trajectory optimisation | CasADi + IPOPT |
| Robots | RK4 unicycle (1st order) / second-order unicycle (accel control) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # runtime + pytest/ruff; exposes the `src` package
pip install -e ".[dev,planning]"        # add CasADi for approach=planning
# other extras: pip install -e ".[wandb,viewer,dubins]"
```

`dubins` is **optional** — `DubinsPotential` falls back to a bundled pure-Python
implementation (`src/shaping/_dubins_py.py`). Without `planning`, the planner tests skip.

## Entry points

Four, one job each. `main.py` dispatches on the approach; the rest are the focused paths.

```bash
python main.py                       # whichever approach the config selects
python train.py                      # train an RL policy
python evaluate.py                   # render one episode to GIF + report success
python scripts/fasteval.py           # same scoring, headless and in bulk, no rendering
```

## Choosing an experiment

Edit the `env:` and `shaping:` lines in `conf/config.yaml`. That is the whole mechanism —
there is no per-person override file.

It does mean `conf/config.yaml` conflicts whenever two people run different things. When
it does, take the **incoming** `defaults:` block whole (it carries structure, not
experiment choice) and re-apply your own `env:`/`shaping:` on top. Keeping both sides is
what produces `network appears more than once in the final defaults list`.

Command-line overrides win over the file and are the right tool for a **sweep**, where
several scenarios run at once and the file can only name one. That is what the examples
below use, so they stay copy-pasteable.

## Running

```bash
# ── planning: nothing to train ──────────────────────────────────────────────
python evaluate.py approach=planning approach.method=karc env=swap2_unicycle2

# ── RL: train, then render ──────────────────────────────────────────────────
python train.py env=gap2_unicycle2 shaping=dijkstra train.timesteps=400000

# a checkpoint is enough — env/shaping/obs/init/network come from the run
python evaluate.py eval.checkpoint=runs/gap2_unicycle2_dijkstra_mlp_full_state/<ts>/checkpoints/agent_400000.pt

# anything typed still wins over what the run recorded
python evaluate.py eval.checkpoint=<...>.pt env=gap2_unicycle2 eval.episodes=5

# bulk metrics (emits a RESULT,<mode>,... line for scripts)
python scripts/fasteval.py eval.checkpoint=<...>.pt eval.episodes=50
```

`train.py` writes checkpoints, TensorBoard logs and `config.yaml` to
`runs/{env}_{shaping}_{network}_{obs}/<timestamp>/`. That saved config is what makes the
one-argument `evaluate.py` above work; runs from before it was added still need
`env=`/`shaping=` by hand.

### Sweeping over N

The teams are homogeneous, so `share_policy` puts every robot on one network and
`envs_x_agents` then sets `num_envs = envs_x_agents / N`. That keeps the transitions per
PPO update identical at every N — with a fixed `num_envs` the shared net would get eight
times more data at N=32 than at N=4, and N would stop being the only variable. It also
cancels one factor of N from the `O(num_envs · N²)` timestep cost.

```bash
for n in 4 8 16 32; do
  python train.py env=open_cross_${n}_unicycle2 \
    train.share_policy=true train.envs_x_agents=32
done
```

### K-ARC and its resolution ladder

The default ladder is K-ARC's own (§III-C): **prioritized trajectory optimization →
Decoupled Kinodynamic RRT → Composite Kinodynamic RRT**. The two sampling rungs exist
because a prioritized re-solve cannot change homotopy class — it can only make a robot slow
down or stop, never route it the other way round an obstacle — so a ladder of trajopt-only
rungs is not K-ARC's ladder. They share one time-gridded planner,
`src/approach/planning/krrt.py`: extensions advance a whole number of `env.dt` steps and a
result is padded to exactly the segment horizon, so an RRT trajectory is index-comparable
with a trajopt one and fixed trajectories are avoided as *moving* obstacles.

Two extra rungs are ours, not K-ARC's, and are available by naming them: `relaxed_goal`
(lower-priority robots get a looser terminal tolerance) and `joint` (the conflicting robots
in one nonlinear program — the optimization-side counterpart of `composite_rrt`).

`ladder` is a list, so dropping rungs from it *is* the ablation — no code change:

```bash
# full ladder (the default)
python main.py approach=planning approach.method=karc env=open_cross_4_unicycle2

# prioritized rung only. Quote it: bash eats the brackets.
python main.py approach=planning approach.method=karc \
  env=open_cross_4_unicycle2 'approach.karc.ladder=[prioritized]'

# sampling rungs only — the way to exercise them on a scenario the first rung can solve
python main.py approach=planning approach.method=karc \
  env=open_cross_4_unicycle2 'approach.karc.ladder=[decoupled_rrt,composite_rrt]'

# ... and render it
python main.py approach=planning approach.method=karc \
  env=open_cross_4_unicycle2 'approach.karc.ladder=[prioritized]' \
  eval.gif_path=experiments/open_cross_karc.gif
```

`scripts/karc_trace_gif.py` renders the planning *process* instead of its execution. The
robots are driven along each candidate trajectory as it is proposed: the kinematic
reference paths, each segment's uncoordinated solve with its conflicts marked, what every
ladder rung changed, and finally the whole committed plan. `+frame_skip=2` for smoother.

```bash
python scripts/karc_trace_gif.py approach=planning env=open_cross_4_unicycle2 \
  'approach.karc.ladder=[prioritized]' eval.gif_path=experiments/oc4_trace.gif
```

**Time budget.** `approach.karc.timeout` (default **600 s**) matches K-ARC's setup — §V-A:
*"Each method is given 20 trials for each scenario, with a timeout of 600 seconds."* It is
checked between rungs, rounds and segments, so overshoot is one solver call. On expiry every
robot brakes to rest, `timed_out=1` is reported, and the remaining segments count as
unsolved: a timeout is a failure, not a slow success. `null` disables it, which is only
useful for debugging — never for a number that goes in a table.

**Baseline** (`experiments/karc_ladder_baseline.txt`, default ladder, 600 s budget, n=1):

| scenario | success | collisions | steps | rungs fired | wall |
|---|---|---|---|---|---|
| `open_cross_4` | 100% | 0 | 571 | `prioritized` ×2 | 12 s |
| `open_cross_8` | 100% | 0 | 571 | `prioritized` ×4 | 26 s |
| `open_cross_16` | 100% | 0 | 572 | `prioritized` ×8 | 60 s |
| `open_cross_32` | **0%** | 0 | — | all three, 12/12/11 | timeout (635 s) |
| `open_cross_32_wide` | 100% | 0 | 573 | `prioritized` ×16 | 163 s |

`open_cross_32` is the only failure, and it is a *congestion* failure, not a robot-count
one: `open_cross_32_wide` is the same 32 robots over the same 15 m traverse with rows at
`open_cross_16`'s 2.14 m spacing instead of 1.0 m, and it solves with the first rung alone.
Our fixed 17×17 world at every N is a design choice of this repo (`docs/dhrubo_open_cross.md`)
— K-ARC never states its world or robot size, and calls open cross *"an easy scenario"* —
so the paper-comparable 32-robot row is the wide one.

Every planning run prints a `STATS,<method>,...` line beside `RESULT`: conflicts,
subproblems, rounds, `rungs` (which ones actually fired), `solver_calls`, `joint_solves`,
`wall_time`. Read the ablation off `rungs` — success rate alone cannot tell you whether
the rung you removed was ever reached. On `open_cross_4` both commands above give 100%
success, 0 collisions, 578 steps and `rungs={'prioritized': 2}`, i.e. the lower rungs
never fire at N=4. Forcing the sampling rungs (third command) also succeeds, in 544 steps
and ~8x the wall time — the price of the completeness they buy.

> **Reproducibility.** The sampling rungs are randomised, so a run is only reproducible
> for a given `approach.karc.rrt_seed`. Runs whose `rungs` shows only `prioritized` are
> unaffected and stay deterministic — which is every open-cross result at N ≤ 16.

## Robots

| Config | Type | State | Action | Shape | Notes |
|---|---|---|---|---|---|
| `unicycle_v1` | `unicycle` | `[x, y, θ]` | `[v, ω]` | disc r=0.13 | first-order / kinematic |
| `unicycle_v2` | `unicycle2` | `[x, y, θ, v, ω]` | `[a, α]` | disc r=0.13 | **second-order / kinodynamic** |
| `unicycle_db` | `unicycle2` | `[x, y, θ, v, ω]` | `[a, α]` | box 0.5×0.25 | dynobench `unicycle2_v0`, for the db-CBS ports |
| `car_kinematic` | `car` | `[x, y, θ]` | `[v, δ]` | box 0.3×0.5 | steering-angle car |

The second-order robot exposes its velocity `[v, ω]` in the observation (else the system
is a POMDP) and has a braking distance `≈ v²/(2·a_max)`, so it must decelerate before the
goal. `unicycle_db` is a **box**, which is why `swap1`/`swap2` render as rectangles.

## Environments

| Env | Agents | Robot | Description |
|---|---|---|---|
| `gap2_unicycle2` | 2 | `unicycle_v2` | head-on through one shared narrow gap |
| `swap1_unicycle2` | 1 | `unicycle_db` | db-CBS port; single robot, empty world — for testing a *potential* |
| `swap2_unicycle2` | 2 | `unicycle_db` | db-CBS port; symmetric head-on swap — a *coordination* problem |
| `open_cross_{4,8,16,32}_unicycle2` | 4–32 | `unicycle_db` | K-ARC Open Cross port; N/2 symmetric head-on rows, empty world. Generated — run `python scripts/gen_open_cross.py`, don't hand-edit |

## Shaping potentials (`shaping=…`)

`none` (φ=0) · `euclidean` (φ=−‖p−g‖) · `dubins` (φ=−L_dubins/v_max, heading-aware) ·
`dijkstra` (obstacle-aware grid cost-to-go — routes *around* obstacles) ·
`braking` (dijkstra plus the velocity-dependent stopping cost a position-only potential
cannot express).

## Planning methods (`approach=planning approach.method=…`)

`rrt` · `kinodynamic_rrt` (both stubs — intern exercises, see `docs/INTERN.md`) ·
`optimization` (prioritised minimum-time NLP) ·
`karc` (K-ARC, arXiv:2501.01559 — segmented plans, geometric conflict detection, and a
configurable resolution ladder). Everything is set from `conf/approach/planning.yaml`;
`approach=planning` also drops the `network`/`train` groups, so `--cfg job` shows only
knobs that affect the run.

## Layout

```
conf/            Hydra configs (approach/ env/ robot/ shaping/ obs/ init/ network/ train/)
src/
  approach/      the RL-vs-planning split
    rl/            IPPO training + the eval controller that loads checkpoints
    planning/      rrt, kinodynamic_rrt, optimization, karc, and the CasADi NLP
    rollout.py     the episode loop both approaches score with
  robot/         UnicycleModel, Unicycle2Model, CarModel (RK4)
  env/           MultiAgentNav (PettingZoo), vectorized wrapper, factory.build_env
  obs/           egocentric full-state / lidar observation builders
  shaping/       potentials (BasePotential)
  collision/     circle + OBB shapes and the overlap tests
  init/          start/goal initializers (fixed, random, random_heading)
  networks/      policy and value nets (mlp, gru)
  viz/           greyscale matplotlib renderer
scripts/         fasteval.py (bulk metrics), viewer.py (streamlit), numerical diagnostics
tests/           pytest: robot dynamics, shaping, env contract, planners, renderer, eval
docs/            task notes for collaborators (INTERN.md, results.md, ...)
paper/ notes/    write-ups and results
main.py train.py evaluate.py   Hydra entry points
```

## Tests

```bash
ruff check . && pytest        # CI runs both on every push (.github/workflows/ci.yml)
```

## Extending

- **New potential**: subclass `BasePotential` (`phi(state, goal)->float`), register in
  `src/shaping/__init__.py:build_potential`, add `conf/shaping/<name>.yaml`.
- **New robot**: subclass `BaseRobot` (`step`, `reset_state`, action/obs metadata),
  register in `src/robot/__init__.py:build_robot`, add `conf/robot/<name>.yaml`.
- **New planner**: subclass `BasePlanner`, register in
  `src/approach/planning/__init__.py:_PLANNERS`, add a same-named block to
  `conf/approach/planning.yaml`. See `docs/INTERN.md`.
