# Kinodynamic RL — Second-Order Multi-Robot Navigation

Decentralized reinforcement learning for **second-order (kinodynamic)** non-holonomic
robots: acceleration-controlled unicycles that carry momentum and must brake to stop.
Robots learn to reach goals while routing around obstacles and avoiding each other,
trained with IPPO + obstacle-aware potential-based reward shaping. Every component
(robot, observation, initializer, shaping potential) is swappable via a single config field.

See `paper/` and `notes/` for the write-ups and results.

## Stack

| Component | Choice |
|---|---|
| Config | Hydra 1.3 |
| Tensors / nets | PyTorch |
| Multi-agent RL | skrl (IPPO) |
| Env contract | PettingZoo Parallel API |
| Robots | RK4 unicycle (1st order) / second-order unicycle (accel control) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # runtime + pytest/ruff; exposes the `src` package
# optional extras: pip install -e ".[wandb,viewer,dubins]"
```

`dubins` is **optional** — `DubinsPotential` falls back to a bundled pure-Python
implementation (`src/shaping/_dubins_py.py`).

## Robots

| Config | Type | State | Action | Notes |
|---|---|---|---|---|
| `unicycle_v1` | `unicycle` | `[x, y, θ]` | `[v, ω]` | first-order / kinematic |
| `unicycle_v2` | `unicycle2` | `[x, y, θ, v, ω]` | `[a, α]` | **second-order / kinodynamic** (accel) |

The second-order robot exposes its velocity `[v, ω]` in the observation (else the system
is a POMDP) and has a braking distance `≈ v²/(2·a_max)`, so the policy must decelerate
before the goal.

## Environments

| Env | Agents | Description |
|---|---|---|
| `crossing_2agent` | 2 | open workspace, 2 staggered obstacles, robots swap sides |
| `gap_2agent` | 2 | head-on through one shared narrow gap (opposite-side lanes) |

## Shaping potentials (`shaping=…`)

`none` (φ=0) · `euclidean` (φ=−‖p−g‖) · `dubins` (φ=−L_dubins/v_max, heading-aware) ·
`dijkstra` (obstacle-aware grid cost-to-go — routes *around* obstacles).

## Running

```bash
# train the default demo (crossing_2agent + dijkstra)
python train.py train.timesteps=400000

# pick env / shaping / robot via the env config
python train.py env=gap_2agent shaping=dijkstra

# headless metrics (success / collisions; emits a RESULT,<mode>,... line for scripts)
PYTHONPATH=. python scripts/fasteval.py \
  eval.checkpoint=runs/dijkstra_mlp_full_state/<ts>/checkpoints/agent_400000.pt \
  env=gap_2agent shaping=dijkstra init=fixed eval.episodes=50

# render a GIF
python evaluate.py eval.checkpoint=runs/.../agent_400000.pt env=crossing_2agent \
  shaping=dijkstra init=fixed eval.episodes=1 eval.gif_path=crossing_2agent.gif
```

Checkpoints + TensorBoard logs: `runs/{shaping}_{network}_{obs}/<timestamp>/`.

## Layout

```
conf/            Hydra configs (env/ robot/ shaping/ obs/ init/ network/ train/)
src/
  robot/         UnicycleModel, Unicycle2Model (RK4)
  shaping/       none | euclidean | dubins | dijkstra potentials (BasePotential)
  obs/           egocentric full-state / lidar observation builders
  env/           MultiAgentNav (PettingZoo) + vectorized wrapper
  training/      IPPO setup (_build_env is the single source of truth for construction)
  viz/           matplotlib renderer
scripts/         fasteval.py (metrics), viewer.py (streamlit checkpoint viewer)
tests/           pytest: robot dynamics, shaping, env contract
paper/ notes/    write-ups and results
train.py evaluate.py   Hydra entry points
```

## Tests

```bash
ruff check . && pytest        # CI runs both (.github/workflows/ci.yml)
```

## Extending

- **New potential**: subclass `BasePotential` (`phi(state, goal)->float`), register in
  `src/shaping/__init__.py:build_potential`, add `conf/shaping/<name>.yaml`.
- **New robot**: subclass `BaseRobot` (`step`, `reset_state`, action/obs metadata),
  register in `src/robot/__init__.py:build_robot`, add `conf/robot/<name>.yaml`.
