# Kinodynamic RL — Second-Order Multi-Robot Navigation

Multi-robot navigation for **second-order (kinodynamic)** non-holonomic robots:
acceleration-controlled unicycles that carry momentum and must brake to stop. Robots
reach goals while routing around obstacles and avoiding each other.

The same problem is solved two ways, chosen by one config field:

| `approach=` | What it does | Entry |
|---|---|---|
| `reinforcement_learning` (default) | trains decentralized IPPO policies with obstacle-aware potential-based shaping | `train.py` |
| `planning` | computes controls online — sampling-based, minimum-time trajectory optimisation, the K-ARC baseline, or our constructive coordinator | `evaluate.py` |

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
pip install -e ".[dev,planning]"        # add CasADi + z3 + scipy for approach=planning
# other extras: pip install -e ".[wandb,viewer,dubins]"
```

`dubins` is **optional** — `DubinsPotential` falls back to a bundled pure-Python
implementation (`src/core/shaping/_dubins_py.py`). Without `planning`, the planner tests skip.

## Entry points

Four, one job each. `main.py` dispatches on the approach; the rest are the focused paths.

```bash
python main.py                       # whichever approach the config selects
python train.py                      # train an RL policy
python evaluate.py                   # render one episode to GIF + report success
python scripts/fasteval.py           # same scoring, headless and in bulk, no rendering
```

## Who owns what

The three packages under `src/` are layered, and the layering is the rule that keeps two
people out of each other's way:

| Package | Depends on | Touched by |
|---|---|---|
| `src/core/` | nothing else in `src/` | either side, by agreement — a change here moves both |
| `src/planning/` | `src/core/`, `src/approach/` | the planner work (CEGAR, K-ARC, K-CBS) |
| `src/rl/` | `src/core/`, `src/approach/` | the RL work |

`src/planning/` and `src/rl/` never import each other. If a change seems to need that, it
belongs in `src/core/` instead.

The guided-RL work lives in `src/rl/` and reads three things out of the other packages,
none of which it should modify: `src/planning/geometric_rrt.py` (the guide paths),
`src/core/conflict/margin.py` (braking margins) and `src/core/shaping/braking_potential.py`.

Before pushing: `ruff check . && pytest` — both must pass.
`tests/test_planning.py::test_adapt_subproblem_reopens_the_previous_segment_and_rescues_it`
is a known failure on this branch and is not yours.

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

### Rendering an episode

Same command for both approaches — `evaluate.py` only differs in what it is pointed at:

```bash
# planner
python evaluate.py approach=planning approach.method=cegar env=open_cross_8_unicycle2

# RL policy
python evaluate.py eval.checkpoint=<...>.pt

# same three knobs either way
python evaluate.py <...> eval.episodes=3 eval.gif_path=out.gif eval.fps=15
```

| key | default | effect |
|---|---|---|
| `eval.gif_path` | `episode.gif` | where the GIF goes; set it to `null` to score without rendering |
| `eval.episodes` | `1` | episodes per GIF, concatenated with a short freeze between them |
| `eval.fps` | `15` | playback rate |

Both paths render through `run_episodes` (`src/approach/rollout.py`), so a planner GIF
and an RL GIF of the same scenario are produced identically — same frame skip, same
freeze between episodes, same writer. `main.py approach=planning` renders through the
same function, so it agrees with `evaluate.py` too. Nothing renders when `gif_path` is
unset, which is how `scripts/fasteval.py` stays free of matplotlib.

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

## Methods

Four planners, documented in **[docs/methods.md](docs/methods.md)** — the ladders, the
ablation switches and the commands that exercise each one.

| `approach.method=` | what it is |
|---|---|
| `karc` | K-ARC (arXiv:2501.01559) reimplementation: prioritized trajopt → decoupled RRT → composite RRT. The baseline. |
| `constructive` | ours: reference paths → lanes → one scheduling program. No conflict loop. |
| `cegar` | ours: sampling-based, each failed round resamples the robots its unsat core blamed. |
| `splinecegar` | ours, current: B-spline trajectories, core-guided repair, makespan certificate. |
| `optimization` | prioritized minimum-time NLP, one robot at a time. |
| `kcbs` | K-CBS baseline. |
| `rrt`, `kinodynamic_rrt` | stubs — see [docs/INTERN.md](docs/INTERN.md). |

Everything is set from `conf/approach/planning.yaml`. `approach=planning` also drops the
`network`/`train` groups, so `--cfg job` shows only knobs that affect the run.

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
| `cluttered_cross_{4,8,16,32}_unicycle2` | 4–32 | `unicycle_db` | K-ARC Cluttered Cross port; same rows, with pillars in the way. Generated by `scripts/gen_open_cross.py` |
| `circular_cross_{4,8,16,32}_unicycle2` | 4–32 | `unicycle_db` | **not** a K-ARC benchmark — antipodal swaps on a circle, so there are no rows to index and every route meets at one hub. Generated by `scripts/gen_circular_cross.py` |

## Shaping potentials (`shaping=…`)

`none` (φ=0) · `euclidean` (φ=−‖p−g‖) · `dubins` (φ=−L_dubins/v_max, heading-aware) ·
`dijkstra` (obstacle-aware grid cost-to-go — routes *around* obstacles) ·
`braking` (dijkstra plus the velocity-dependent stopping cost a position-only potential
cannot express).

## Layout

```
conf/            Hydra configs (approach/ env/ robot/ shaping/ obs/ init/ network/ train/)
src/
  approach/      the shared contract: BaseApproach/Controller, build_approach (the
                 factory that dispatches to planning or rl), rollout.py (the episode
                 loop both approaches score with)
  core/          everything both sides need; imports from neither of them
    robot/         UnicycleModel, Unicycle2Model, CarModel (RK4)
    env/           MultiAgentNav (PettingZoo), vectorized wrapper, factory.build_env
    obs/           egocentric full-state / lidar observation builders
    shaping/       potentials (BasePotential)
    collision/     circle + OBB shapes and the overlap tests
    conflict/      pairwise contact + the dynamics-aware braking/reachability margins
    init/          start/goal initializers (fixed, random, random_heading)
    networks/      policy and value nets (mlp, gru)
    viz/           greyscale matplotlib renderer
  planning/      geometric_rrt + krrt (samplers), rrt, kinodynamic_rrt, optimization,
                 the CasADi NLP, karc (baseline), kcbs, and constructive + cegar +
                 splinecegar + flat + schedule (ours)
  rl/            IPPO training + the eval controller that loads checkpoints
scripts/         fasteval.py (bulk metrics), viewer.py (streamlit), karc_trace_gif.py
                 (planning process as a GIF), schedule_sat.py (SMT-LIB2 dump),
                 gen_*_cross.py (scenario generators), numerical diagnostics
tests/           pytest: robot dynamics, shaping, env contract, planners, renderer, eval
docs/            CONTRIBUTING.md (conventions — read first), methods.md (what each
                 planner does), INTERN.md (planner walkthrough), results.md
runs/            training output, written by train.py only        (git-ignored)
experiments/     results from scripts/: CSVs, figures, tables     (git-ignored)
outputs/         Hydra's per-invocation working dir, disposable   (git-ignored)
notes/           results write-ups (paper/ and overleaf_cegar_trajopt/ are on
                 disk but git-ignored: paper sources go to Overleaf, not GitHub)
main.py train.py evaluate.py   Hydra entry points
```

## `runs/` vs `experiments/`

Two output directories, different owners. Neither is tracked by git.

| | `runs/` | `experiments/` |
|---|---|---|
| written by | `train.py`, nothing else | `scripts/*.py` |
| path | `runs/{env}_{shaping}_{network}_{obs}/{timestamp}/` — chosen for you | whatever the script's `--out` says, default `experiments/` |
| holds | `checkpoints/*.pt`, TensorBoard events, `config.yaml` | CSVs, `.png` figures, `.tex` tables, `.smt2` dumps, GIFs |
| read back by | `eval.checkpoint=<...>.pt` | `scripts/plot_baselines.py`, and you, by hand |
| regenerating it | costs a training run | rerun the script |

So: **training writes `runs/`, analysis writes `experiments/`.** You never name a
`runs/` path yourself — you point `eval.checkpoint` at one. Planning has no `runs/`
entry at all, because there is nothing to train; a planner's numbers go straight to
`experiments/` via the benchmark scripts.

One exception to the ignore rules: `runs/*/*/config.yaml` is deliberately *not* ignored
(`.gitignore:29`). It is 4 KB and it is the record of what was run, so committing one
makes a result reviewable in the diff. Checkpoints and event files stay out — W&B is
where a full run is published (`train.py ... wandb.enabled=true`).

`experiments/` and `outputs/` are safe to delete when they get large; nothing reads them
that cannot regenerate them.

## Tests

```bash
ruff check . && pytest        # both must pass before you push
```

## Extending

Conventions, and the reasoning behind them, are in
**[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)**. Read it before the first change.

The three things you are most likely to add, all the same shape — subclass, register in a
factory, add a YAML with the same name:

| add | subclass | register in | config |
|---|---|---|---|
| potential | `BasePotential` (`phi(state, goal)->float`) | `src/core/shaping/__init__.py:build_potential` | `conf/shaping/<name>.yaml` |
| robot | `BaseRobot` (`step`, `reset_state`, action/obs metadata) | `src/core/robot/__init__.py:build_robot` | `conf/robot/<name>.yaml` |
| planner | `BasePlanner` (`reset`, `act`) | `src/planning/__init__.py:_PLANNERS` | a block in `conf/approach/planning.yaml` |

A planner walkthrough with the env cheat-sheet is in [docs/INTERN.md](docs/INTERN.md).
