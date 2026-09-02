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

## Running

```bash
# ── planning: nothing to train ──────────────────────────────────────────────
python evaluate.py approach=planning approach.method=karc env=swap2_unicycle2

# ── RL: train, then render ──────────────────────────────────────────────────
python train.py env=gap_2agent shaping=dijkstra train.timesteps=400000

# a checkpoint is enough — env/shaping/obs/init/network come from the run
python evaluate.py eval.checkpoint=runs/dijkstra_mlp_full_state/<ts>/checkpoints/agent_400000.pt

# anything typed still wins over what the run recorded
python evaluate.py eval.checkpoint=<...>.pt env=gap_2agent eval.episodes=5

# bulk metrics (emits a RESULT,<mode>,... line for scripts)
python scripts/fasteval.py eval.checkpoint=<...>.pt eval.episodes=50
```

`train.py` writes checkpoints, TensorBoard logs and `config.yaml` to
`runs/{shaping}_{network}_{obs}/<timestamp>/`. That saved config is what makes the
one-argument `evaluate.py` above work; runs from before it was added still need
`env=`/`shaping=` by hand.

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
| `crossing_2agent` | 2 | `unicycle_v2` | open workspace, 2 staggered obstacles, robots swap sides |
| `gap_2agent` | 2 | `unicycle_v2` | head-on through one shared narrow gap |
| `swap1_unicycle2` | 1 | `unicycle_db` | db-CBS port; single robot, empty world — for testing a *potential* |
| `swap2_unicycle2` | 2 | `unicycle_db` | db-CBS port; symmetric head-on swap — a *coordination* problem |

## Shaping potentials (`shaping=…`)

`none` (φ=0) · `euclidean` (φ=−‖p−g‖) · `dubins` (φ=−L_dubins/v_max, heading-aware) ·
`dijkstra` (obstacle-aware grid cost-to-go — routes *around* obstacles) ·
`braking` (dijkstra plus the velocity-dependent stopping cost a position-only potential
cannot express).

## Planning methods (`approach=planning approach.method=…`)

`rrt` · `kinodynamic_rrt` · `optimization` (prioritised minimum-time NLP) ·
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
docs/            task notes for collaborators (INTERN.md, dhrubo_wandb.md, ...)
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
