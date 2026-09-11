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
pip install -e ".[dev,planning]"        # add CasADi + z3 for approach=planning
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

## K-ARC baseline (reimplementation)

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

## Constructive coordination (ours)

`approach.method=constructive` is **not** a rung, a flag or a variant of the baseline above.
It is a separate planner in `src/approach/planning/constructive.py` with its own config
block, its own guide generator and its own scheduler; the two share no module, no config
key and no code path, so neither can be quietly turned into the other by a switch.

Where K-ARC plans segments and then repairs whichever conflicts it finds, this one builds a
plan that is collision-free by construction and then asks a solver whether the whole team
can be seated in time at once:

1. **Guides.** One geometric RRT path per robot (`src/approach/planning/geometric_rrt.py`),
   sampled in continuous space and shortcut — obstacle-free, but ignorant of other robots.
2. **Lanes and roundabouts.** Robots whose guides run together are offset onto parallel
   lanes; robots whose guides meet at a shared hub are routed around it in one consistent
   sense. Lane pitch is bounded by the corridor each robot actually has.
3. **Smooth driving.** Each route is fitted with a blurred, arclength-uniform curve
   (`smooth` is the blur length in metres, so curvature — and with it the cornering cap
   `v <= ω_max/κ` — stays usable) and realised through differential flatness
   (`src/approach/planning/flat.py`): the flat outputs (x, y) give θ, v, ω, a, α exactly,
   so the trajectory is dynamically feasible by construction rather than feasible up to a
   bounded discontinuity. A TOPP-style forward/backward sweep sets the speed profile
   under `v_max`, `ω_max`, `a_max`, `α_max`.
4. **Timing.** Each robot gets a small set of candidate trajectories (lane variants,
   speeds). For every pair and every candidate pair, the *differences* of departure time
   at which the two bodies would touch are tabulated exactly — bounding-radius and
   inscribed-radius bands decide most cells, and only the annulus needs an OBB distance.
5. **Schedule.** The tables go to z3 as a QF_LIA/difference-logic problem: pick one
   candidate per robot and one departure time per robot such that no pair lands on a
   forbidden difference. Makespan is then bisected down using the same tables. A
   satisfying model is a plan; **UNSAT is a proof** that no schedule exists over that
   candidate set, which is what makes the candidate set (not the scheduler) the thing to
   fix when it fails.

```bash
# plan, execute and report the coordination counters (`STATS,constructive,...`)
python main.py approach=planning approach.method=constructive \
  env=cluttered_cross_16_unicycle2

# the planning *process* as a GIF (reference paths -> lanes -> schedule)
python scripts/karc_trace_gif.py approach=planning approach.method=constructive \
  env=cluttered_cross_16_unicycle2 eval.gif_path=experiments/cc16_trace.gif

# greedy insertion instead of the solver — the scheduler ablation
python main.py approach=planning approach.method=constructive \
  env=cluttered_cross_16_unicycle2 approach.constructive.schedule=greedy

# render the execution to a GIF
python evaluate.py approach=planning approach.method=constructive \
  env=cluttered_cross_16_unicycle2 eval.gif_path=experiments/cc16.gif
```

Knobs live under `constructive:` in `conf/approach/planning.yaml`: `guide_rrt_*` (the
sampler), `smooth` / `blur_cap` (how far the fitted curve may stray), `veer_margin` and
`hub_pack` (lane and roundabout geometry), `spacetime` (conflict test), `schedule`
(`sat` | `greedy`), `sat_timeout`, `delay_step`, and `trace`.

**Looking at the encoding.** `scripts/schedule_sat.py` runs the timing stage standalone
and dumps the problem it posed, so the constraints can be read or re-solved outside the
planner:

```bash
python scripts/schedule_sat.py env=cluttered_cross_8_unicycle2
# -> experiments/schedule_sat_cluttered_cross_8_unicycle2.smt2   (SMT-LIB2, QF_LIA)
```

**Where it stands** (`experiments/ours_baseline_compare.txt`, one run each, every plan
verified collision-free by the environment's own checker; steps at `dt=0.1`):

| scenario | `constructive` | `cegar` |
|---|---|---|
| `open_cross_32` | **332** steps / 92 s | 516 / 51 s |
| `circular_cross_16` | **416** / 21 s | 556 / 32 s |
| `cluttered_cross_16` | **675** / 33 s | 711 / **11 s** |
| `cluttered_cross_32` | 785 / 166 s | **619** / **119 s** |

The split is the interesting part and it is not noise: the rulebook wins where its devices
are exactly right — parallel rows want lanes, a circle wants a roundabout — and loses in
clutter, where lane pitch is squeezed by the pillars and the sampled alternative simply
goes round. Single seeds, so read the pattern, not the third digit.


## Conflict-guided resampling (ours, the sampling-based one)

`approach.method=cegar` is the same problem attacked without a rulebook. The method above
decides lanes, roundabouts and passing sides from geometry the author picked out of the
benchmarks; this one decides nothing in advance. Each robot owns a growing set of **sampled
candidate motions**, a solver picks one per robot, and when no pick works the solver's own
explanation says which robots to resample and where.

1. **Candidates.** A sampled path (geometric RRT), its corners rounded as hard as the
   corridor allows (`flat.fit_blur`: a kink in a polyline caps the cornering speed for the
   whole traverse, and open_cross_32 is 701 steps driving the path as sampled against 516
   with it rounded), driven as one smooth flat trajectory — the same `flat` realisation as
   above — plus the same geometry traversed slower and
   variants that stop EN ROUTE for a sampled number of steps. Waiting is a motion like any
   other: nothing is ever held on its start line, because a robot parked on its start is a
   device only a planner that owns the whole world can use.
2. **Lazy SMT.** Pick exactly one candidate per robot. Collision clauses are added only
   when the solver actually proposes a pair — it proposes, the pair test refutes, the
   refutation comes back as a clause `¬(x_ia ∧ x_jb)`. Most pairs are never checked.
3. **Core-guided refinement.** UNSAT means no combination of the current candidates works.
   The **unsat core** is a set of pairwise refutations, so it names the robots that cannot
   be reconciled and the points where their candidates met. Those robots resample with a
   disc dropped on the contested point, which pushes the sampler out of that corridor
   instead of back into it. The candidate set grows strictly, so a refutation is never
   re-derived.
4. **Makespan.** Any satisfying pick is collision-free but says nothing about duration, and
   the cheapest way out of a conflict — wait longer — is the one that inflates it. Once a
   plan verifies, its horizon is bisected down over the same candidates and the same
   learned refutations, so the search costs solver time only.

```bash
python main.py approach=planning approach.method=cegar env=cluttered_cross_16_unicycle2

# the loop itself: each failed round draws the points its core blamed
python scripts/karc_trace_gif.py approach=planning approach.method=cegar \
  env=cluttered_cross_16_unicycle2 eval.gif_path=experiments/cc16_cegar.gif
```

Knobs under `cegar:` in `conf/approach/planning.yaml`: `guide_rrt_*`, `smooth`/`blur_cap`,
`slow`,
`waits` / `long_wait` (how long an en-route stop may be), `rounds`, `timeout`, `trace`.
`timeout` is meant to be the binding budget — a round is cheap, and stopping on a round
count throws away a loop that was still making progress.

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

## Planning methods (`approach=planning approach.method=…`)

`rrt` · `kinodynamic_rrt` (both stubs — intern exercises, see `docs/INTERN.md`) ·
`optimization` (prioritised minimum-time NLP) ·
`karc` (K-ARC, arXiv:2501.01559 — segmented plans, geometric conflict detection, and a
configurable resolution ladder) ·
`constructive` (ours — sampled guides, lanes/roundabouts, flatness-based smooth driving,
and an exact SMT schedule) ·
`cegar` (ours — sampled candidate motions, lazy SMT, and unsat-core-guided resampling).
Both of ours are described above. Everything is set from `conf/approach/planning.yaml`;
`approach=planning` also drops the `network`/`train` groups, so `--cfg job` shows only
knobs that affect the run.

## Layout

```
conf/            Hydra configs (approach/ env/ robot/ shaping/ obs/ init/ network/ train/)
src/
  approach/      the RL-vs-planning split
    rl/            IPPO training + the eval controller that loads checkpoints
    planning/      geometric_rrt + krrt (samplers), rrt, kinodynamic_rrt, optimization,
                   the CasADi NLP, karc (baseline), and constructive + cegar + flat +
                   schedule (ours)
    rollout.py     the episode loop both approaches score with
  robot/         UnicycleModel, Unicycle2Model, CarModel (RK4)
  env/           MultiAgentNav (PettingZoo), vectorized wrapper, factory.build_env
  obs/           egocentric full-state / lidar observation builders
  shaping/       potentials (BasePotential)
  collision/     circle + OBB shapes and the overlap tests
  init/          start/goal initializers (fixed, random, random_heading)
  networks/      policy and value nets (mlp, gru)
  viz/           greyscale matplotlib renderer
scripts/         fasteval.py (bulk metrics), viewer.py (streamlit), karc_trace_gif.py
                 (planning process as a GIF), schedule_sat.py (SMT-LIB2 dump),
                 gen_*_cross.py (scenario generators), numerical diagnostics
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
