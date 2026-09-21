# Methods

What each planner actually does, why its rungs are shaped the way they are, and the
commands that exercise them. Split out of the README, which now carries only the parts
you need before you have chosen a method.

- [K-ARC baseline (reimplementation)](#k-arc-baseline-reimplementation) — the ladder we compare against
- [Constructive coordination (ours)](#constructive-coordination-ours) — reference paths, lanes, schedule
- [Conflict-guided resampling (ours, the sampling-based one)](#conflict-guided-resampling-ours-the-sampling-based-one)
- [Core-guided B-spline coordination (ours, current)](#core-guided-b-spline-coordination-ours-current)

## K-ARC baseline (reimplementation)

The default ladder is K-ARC's own (§III-C): **prioritized trajectory optimization →
Decoupled Kinodynamic RRT → Composite Kinodynamic RRT**. The two sampling rungs exist
because a prioritized re-solve cannot change homotopy class — it can only make a robot slow
down or stop, never route it the other way round an obstacle — so a ladder of trajopt-only
rungs is not K-ARC's ladder. They share one time-gridded planner,
`src/planning/krrt.py`: extensions advance a whole number of `env.dt` steps and a
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
It is a separate planner in `src/planning/constructive.py` with its own config
block, its own guide generator and its own scheduler; the two share no module, no config
key and no code path, so neither can be quietly turned into the other by a switch.

Where K-ARC plans segments and then repairs whichever conflicts it finds, this one builds a
plan that is collision-free by construction and then asks a solver whether the whole team
can be seated in time at once:

1. **Guides.** One geometric RRT path per robot (`src/planning/geometric_rrt.py`),
   sampled in continuous space and shortcut — obstacle-free, but ignorant of other robots.
2. **Lanes and roundabouts.** Robots whose guides run together are offset onto parallel
   lanes; robots whose guides meet at a shared hub are routed around it in one consistent
   sense. Lane pitch is bounded by the corridor each robot actually has.
3. **Smooth driving.** Each route is fitted with a blurred, arclength-uniform curve
   (`smooth` is the blur length in metres, so curvature — and with it the cornering cap
   `v <= ω_max/κ` — stays usable) and realised through differential flatness
   (`src/planning/flat.py`): the flat outputs (x, y) give θ, v, ω, a, α exactly,
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

## Core-guided B-spline coordination (ours, current)

`approach.method=splinecegar` is the `cegar` loop with its geometry layer replaced. A
candidate motion **is a cubic B-spline**, and a B-spline is its control points
(`src/planning/splinecegar.py`), which changes three things:

| | `cegar` | `splinecegar` |
|---|---|---|
| smoothness | Gaussian blur of a polyline, blur length tuned per scenario | the cubic basis — C² by construction |
| curvature | two finite differences of the blurred path, then a second blur to hide the noise | analytic, from the spline's own derivatives |
| repair | drop the path, resample a whole new RRT with a disc on the conflict | displace the 2–3 control points nearest the conflict; local support bounds the edit |

Knots are placed by arclength — one control point per `ctrl_spacing` metres — and the fit
is least-squares over those knots, with the first and last control points overwritten by
the start pose and the goal (a clamped basis makes those exact). Spacing is the one knob
that matters: it sets both how closely the spline follows the sampled path and how local a
repair is.

The loop above it is unchanged: lazy SMT over candidates, clauses only for pairs the
solver proposes, unsat cores to name who is stuck, makespan bisection at the end. The
sampler is now the *fallback* — it is called only when every control-point repair is
undrivable or hits an obstacle.

```bash
python main.py approach=planning approach.method=splinecegar env=cluttered_cross_16_unicycle2

python scripts/karc_trace_gif.py approach=planning approach.method=splinecegar \
  env=open_cross_32_unicycle2 eval.gif_path=experiments/oc32.gif +fig_px=640
```

Two bugs worth knowing about, both caught by `tests/test_splinecegar.py` and both fixed:
`scipy.interpolate.splprep(t=…)` wants the **full** knot vector, and silently mis-reads a
list of interior knots (17 control points came back as 9, which quietly destroyed
locality); and deviation from a sampled path must be measured against a **densified**
target or nearest-vertex distance reports the polyline's own vertex spacing as error.
