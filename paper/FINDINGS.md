# Kinodynamic Multi-Robot Coordination: Reproduction, Diagnosis, and a Feasibility Certificate

**Working findings document — no page limit. Written 2026-09-09.**

Baseline under study: **K-ARC**, "Kinodynamic Adaptive Robot Coordination" (arXiv:2501.01559),
and its predecessor **ARC** (arXiv:2312.08554). No public code exists for either; the authors
did not answer email. Everything below is measured against **our own reimplementation**, whose
fidelity is documented in §III and whose deviations are enumerated in §III.C.

> **Status legend used throughout.**
> **[P]** proven or directly measured, reproducible.
> **[M]** measured but with a stated confound.
> **[N]** negative result — the idea was tested and did not work.
> **[O]** open, not yet tested.
> **[!]** claim I asserted earlier in the work and later **retracted**; kept because the
> retraction is itself informative.

---

## Abstract

We reimplement K-ARC's kinodynamic multi-robot coordination hierarchy and evaluate it on the
Open Cross and Cluttered Cross benchmarks with second-order unicycle robots. Three results
stand. First, the quality of the *initial kinematic guide* dominates every other factor we
varied: shortcutting the sampling-based guides turns an 1800 s failure at N=16 into a 51.9 s
solve, and drives subproblem merges from 10 to 0. Second, we show that **pairwise conflict
resolvability does not imply joint resolvability**: at N=32, 19 of 19 detected conflicts are
individually resolvable by a single robot waiting, yet no plan is found — K-ARC's per-pair
subproblem decomposition (Alg. 2) cannot represent the interaction. Third, and most
consequentially, we construct **verified feasibility certificates** proving that Open Cross at
N=4, 16 and 32 is solvable, and — via a two-time-group construction derived from the
benchmark's own row-adjacency structure — solvable **within K-ARC's own planning horizon**
(886 of 1300 steps, 32 robots moving simultaneously, minimum surface clearance 0.25 m against
a 0.05 m requirement). The failure at N=32 is therefore algorithmic: neither instance
infeasibility nor an insufficient horizon accounts for it. This contradicts a density/geometry
explanation we ourselves advanced earlier. We additionally report four negative results on conflict-resolution
mechanisms, each with a diagnosed cause, because each rules out a plausible line of attack.

---

## I. Introduction

K-ARC coordinates kinodynamic robots by segmenting each robot's path into milestones, solving
each segment per-robot, detecting conflicts, and resolving them with a fixed hierarchy of
increasingly powerful solvers. The reported evaluation (§V) covers Open Cross and Cluttered
Cross at N up to 32, with 20 trials per scenario and a 600 s timeout on a 32-core i9-14900K
with 64 GB RAM, implemented in C++ in the Parasol Planning Library.

Our reimplementation is Python driving IPOPT through CasADi on a 31 GB machine. **Wall-clock
numbers in this document are therefore not comparable to the paper's**, and no claim here
depends on such a comparison. What *is* comparable is structural: which conflicts arise, how
many subproblems merge, which rung of the hierarchy resolves them, and whether a plan exists
at all.

The work proceeded as a reproduction that turned into a diagnosis. We set out to add a
mechanism and instead spent most of the effort establishing what the failure actually *is*.
§V and §VI are the substantive contributions; §VII records what did not work and why, which
we consider equally load-bearing given how much of the design space it eliminates.

---

## II. Related Work

**K-ARC / ARC.** K-ARC (arXiv:2501.01559) segments per-robot kinematic paths into milestones
(Alg. 1), solves segments as trajectory-optimisation problems, and resolves conflicts through
a solver hierarchy (Alg. 2: prioritized trajectory optimisation → decoupled kinodynamic RRT →
composite kinodynamic RRT), with `AdaptSubProblem` re-opening previously committed motion.
ARC (arXiv:2312.08554, §IV-B) establishes the reactive-merge rule `R' = R_i ∪ R_j` that K-ARC
inherits.

**Shortcutting of sampling-based paths.** Verified references, safe to cite:
- R. Geraerts and M. H. Overmars, "Creating high-quality paths for motion planning,"
  *IJRR* 26(8):845–863, 2007. doi:10.1177/0278364907079280
- K. Hauser and V. Ng-Thow-Hing, "Fast smoothing of manipulator trajectories using
  optimal bounded-acceleration shortcuts," *ICRA* 2010, pp. 2493–2498.
  doi:10.1109/ROBOT.2010.5509683
- OMPL `PathSimplifier` (Şucan, Moll, Kavraki) — standard practice, implementation reference.

**Needs verification before entering the bibliography** (do NOT cite from this document):
- Path–velocity decomposition, usually attributed to Kant & Zucker (mid-1980s), and the
  coordination-diagram line that follows it. Relevant to §VII.C.
- Unsolvability certificates for classical planning (Eriksson, Röger, Helmert, ICAPS ~2017).
  Relevant to §VI.
- Pebble-motion feasibility (Kornhauser, Miller, Spirakis) — relevant to §VI.B's staging
  argument.

These three are load-bearing for positioning and each must be checked against the actual
publication before use. Per IEEE-RAS policy, AI cannot be an author and AI-assisted content
must be disclosed; every citation must be verified by a human before submission.

---

## III. Faithful Reimplementation

### A. Scenario construction

`scripts/gen_open_cross.py` generates both benchmarks. Open Cross places N robots in N/2
head-on rows in a fixed 17.0 m square world with a 1.0 m wall margin, so the usable span is
15.0 m on both axes. Robot `2k` starts at the left column facing +x with its goal at the right
column; robot `2k+1` mirrors it. Headings are held at start and goal, making each row a pure
translation conflict. **N is a congestion knob**: the world does not grow, so more robots means
tighter rows.

Cluttered Cross was rebuilt to match the paper's Fig. 2(b): four large rectangular interior
blocks spanning x ∈ [0.19, 0.80] of the world, never reaching the start or goal columns, with
the start/goal rows kept clear. An earlier version of ours used 31 small pillars placed *on*
the travel rows, which is a materially harder and different problem; results predating commit
`c656760` are not comparable.

Robot model `unicycle_db`: state `[x, y, θ, v, ω]`, action `[a, α]`, body 0.5 × 0.25 m
(bounding radius 0.2795 m, diagonal 0.559 m), `v_max` 0.5 m/s, `a_max` 0.25 m/s².
Environment `dt` = 0.1 s.

### B. Geometry of the benchmark [P]

Pair clearance threshold is `r_i + r_j + clearance` = 0.2795 + 0.2795 + 0.05 = **0.61 m**.

| N | rows | row spacing | lateral room per side |
|---|---|---|---|
| 4 | 2 | 15.000 m | 7.220 m |
| 8 | 4 | 5.000 m | 2.220 m |
| 16 | 8 | 2.143 m | **0.792 m** |
| 32 | 16 | 1.000 m | **0.220 m** |

Reproduce: `python -c "from scripts.gen_open_cross import geometry, body_diameter; ..."`
(exact snippet in §IX).

At N=16 a robot can complete a head-on swap *within its own lane* (0.792 m > 0.61 m). At N=32
it cannot (0.220 m < 0.61 m), so any swap must borrow a neighbouring lane. **See §VI.C for why
this does not imply infeasibility** — a claim we made and retracted.

### C. Deviations from the paper, and their resolution

Four deviations were identified by auditing our implementation against the PDF and closed in
commit `cc4c807`:

1. **Two-stage Δt.** K-ARC treats Δt as a decision variable (Eq. 2, §IV-B) and does not
   execute its plans. We had been re-solving onto the env's fixed 0.1 s grid. Resolved: the
   planner now reports plan metrics with a shared free timestep, and `execute: false` is the
   default. The dt floor is applied only on the execute path.
2. **Robot–robot distance.** Eq. 6 compares `c_i,k`, the geometric pose, and §V-A states
   "simple polyhedrons for both our obstacle and robot models... shortest distances between any
   two objects." We had compared centres against a sum of bounding radii — strictly stronger.
   For a 0.5 × 0.25 box the circumscribed disc is 0.559 m across against a 0.25 m half-width,
   so the old test reported conflicts at up to 0.3 m of genuine clearance per pair. Resolved
   with exact polyhedral distance (`src/collision/shapes.py::shape_distance`), validated at
   0 disagreements with the boolean `collides` over 4000 random pairs, and a radius prefilter
   that preserves the O(n²T) cost.
3. **Singleton subproblems.** Alg. 1 line 21 builds subproblems from the conflict set alone,
   so a robot that merely misses its milestone is not a subproblem in K-ARC. Now off by default.
4. **`_separate` milestone pulling-apart.** Entirely our invention. Now off by default.

Two parameters the PDF forces were added: `milestone_region` (Alg. 1 line 16 makes an
intermediate goal "a region centered around G_ri[j]", not a state) and `min_time_slack`
(sizing a segment at the unconstrained per-robot minimum time leaves no room for coordination).

### D. Baseline integrity [P]

After every mechanism added during this work, the default configuration reproduces the prior
baseline **bit-identically**: open_cross_16 at defaults solves in **51.9 s**, 16/16 goals,
`composite_rrt_solves=0`, `merges=0`, `rungs={'prioritized': 9}` — identical before and after.
Every mechanism in §VII is off by default. This is the control that makes the rest meaningful.

---

## IV. Experimental Setup

All runs: `init=fixed`, `approach.karc.workers=8`, single trial (**n=1**), 600 s or 1200 s
solver budget as noted. CasADi/IPOPT holds the GIL, so parallelism is by process
(measured 4.11× on 8 processes vs 0.65× on 8 threads).

**Standing caveats that belong on any table built from this document:**
- **n = 1.** K-ARC reports 20 trials per scenario (§V-A). A single deterministic trial is one
  sample, not a success *rate*. Nothing here should be reported as a rate.
- **Wall times are not comparable** to the paper's (Python+CasADi vs C++/Parasol).
- Concurrency was capped after an early batch caused swapping (17 GB used, 1.4 GB swap) on a
  31 GB machine; affected numbers were discarded and rerun sequentially.

---

## V. Finding 1: Guide Quality Dominates [P]

### A. Result

K-ARC obtains initial per-robot paths from "a kinematic sampling-based method" (§IV-A) and
frames the choice as a static pre-planning trade-off (§IV-D-1). It reports no measurement of
this choice. We find it dominates everything else we varied.

Isolation experiment (`experiments/karc_guide_isolation.txt`), open_cross_16:

| initial guide | outcome |
|---|---|
| Dijkstra descent | solved, 85 s |
| RRT + shortcutting | solved, 84 s |
| **raw RRT (no shortcut)** | **failed, 654 s** |

Confirmed at a larger budget:

| open_cross_16 | raw RRT guides | shortcut guides |
|---|---|---|
| result | **FAIL** (1800 s) | **SOLVE, 51.9 s** |
| merges | 10 | **0** |
| subproblem_max | 10 | 2 |
| conflicts_remaining | 9 | 0 |
| rungs fired | all three | `{'prioritized': 9}` |
| goals reached | — | 16/16 |

### B. Mechanism

Raw RRT guides measured 1.25× chord length with 33 vertices and excursions of 3.16 m off the
row, against a row spacing of 2.14 m. A guide that wanders out of its own lane **manufactures
conflicts that do not exist in the problem**. Those conflicts then drive subproblem merges, and
merging is the exponential rung. Shortcutting removes the excursions, the manufactured
conflicts disappear, and the cheapest rung suffices for every remaining conflict.

### C. Threats

Shortcutting is not homotopy-preserving in general. We stated twice that it "cannot leave the
homotopy class" **[!]** and retract that: vertex shortcutting preserves collision-freeness but
can change homotopy class. The empirical claim is unaffected; the theoretical gloss was wrong.

`initial_rrt_shortcut: true` became the default in commit `7a0ec45`.

---

## VI. Finding 2: Feasibility Certificates [P]

### A. Motivation

We had explained the N=32 failure by the geometry of §III.B — 0.220 m of lateral room against
0.61 m needed. That argument only covers **lane-preserving** solutions. It says nothing about
a robot routing through another lane entirely, and therefore does not establish infeasibility.
We built a certificate to settle it.

### B. Method (`scripts/feasibility_certificate.py`)

The abstraction is a **restriction** of the real problem, never a relaxation, so that a witness
in the abstraction is a witness for the real problem:

1. **One robot moves at a time**, so every other robot is a static obstacle and multi-robot
   coupling disappears by construction.
2. **start → staging → goal.** In Open Cross every goal is occupied by another robot's start
   (the rows are swaps), so plain sequential motion deadlocks on the first robot — this is
   pebble-motion structure. Staging cells are chosen in the empty interior, mutually clear and
   clear of both columns.
3. **Stop-and-go legs.** Each leg turns in place to face the next waypoint, drives to it, and
   ends at rest. Both phases use a symmetric accelerate/decelerate profile solved on the
   *integrator the env actually uses* (semi-implicit Euler): n steps up and n steps down
   advance `a·dt²·n²`, so n fixes the acceleration; the smallest n whose implied acceleration
   and peak velocity are both inside the robot's bounds is chosen. Legs land exactly on target
   at exactly zero velocity (measured error 0.0, |v|=0.0), so nothing is "snapped".
4. **Verification uses the env's own collision code and integrator**, not the abstraction the
   plan was built on.

### C. Result [P] — and a retraction [!]

```
open_cross_4    SOLVABLE   4/4   goals   0 obstacle hits  0 robot hits   witness  3768 steps
open_cross_16   SOLVABLE   16/16 goals   0 obstacle hits  0 robot hits   witness 14292 steps
open_cross_32   SOLVABLE   32/32 goals   0 obstacle hits  0 robot hits   witness 30568 steps
```

**Open Cross at N=32 is solvable. K-ARC fails on a provably solvable instance, so the failure
is algorithmic.** This retracts the density/geometry explanation of §III.B as an account of
*infeasibility*; §III.B remains valid as an account of why *lane-preserving* resolution is
impossible at N=32, which is a narrower and still useful statement.

### D. Solvable *within K-ARC's own horizon* [P]

The sequential witness needs 30568 steps against a 1300-step budget (23×), leaving open the
sharper question: is the instance solvable in the time K-ARC is actually given? It is, and the
structure of the benchmark supplies the construction with no search and no solver.

Each row is a head-on swap with no lateral room at N=32, so its two robots must separate to
pass: one veers +0.5 m, the other −0.5 m, giving 1.0 m of centre separation. A veering robot
sits at `y_k ± 0.5`, exactly where the **neighbouring** row's veering robot goes — and since
only adjacent rows interact (the `n ± 2` coupling of §VII.B), the conflict graph over rows is a
path, so **two time groups suffice**: even rows manoeuvre while odd rows hold, then swap.
Veers are shallow diagonals rather than right angles, because `alpha_max` is 0.25 rad/s² with
`omega_max` 0.5 and four 90° turns per robot would spend the budget on rotation alone.

```
open_cross_8    SOLVABLE_IN_BUDGET   8/8   goals  0 hits  witness 886 steps / budget 1300
open_cross_16   SOLVABLE_IN_BUDGET   16/16 goals  0 hits  witness 886 steps / budget 1300
open_cross_32   SOLVABLE_IN_BUDGET   32/32 goals  0 hits  witness 886 steps / budget 1300
                min surface gap, all pairs, all times: 0.2500 m  (5x the 0.05 m required)
```

All robots move **simultaneously**, so verification checks every pair at every one of the 886
time indices with the env's own collision code, and every trajectory is produced by the env's
integrator under the real acceleration bounds. Reproduce with `+witness=coordinated`.

**Consequence.** K-ARC's failure at N=32 is algorithmic: neither instance infeasibility nor an
insufficient horizon can account for it. Both escape hatches are closed by construction.

### E. Soundness limits

Sequential search is **incomplete**. A missing witness prints `UNKNOWN` and never `UNSAT`.
Proving unsolvability requires the opposite abstraction — an over-approximation where every
robot is strictly *more* capable than the real one — plus a complete solver over it. See §VIII.

---

## VII. Finding 3: Pairwise ≠ Joint Resolvability [P]

### A. Result

We instrumented the conflict loop to ask, for every detected conflict and without solving
anything, whether the pair is resolvable by one robot simply waiting (`_waiting_resolves`:
hold one robot's trajectory, replace the other's with a brake-to-rest-and-hold rollout, check
every shared time index, try both assignments).

| | open_cross_16 | open_cross_32 |
|---|---|---|
| `pairs_seen` | 9 | 19 |
| `pairs_wait_resolvable` | **9** | **19** |
| `pairs_ics` (provably unavoidable) | 0 | 0 |
| plan found | yes, 51.9 s | **no** (600 s) |
| `conflicts_remaining` | 0 | 4 |
| `merges` | 0 | 3 |
| `composite_rrt_solves` | 0 | 14 |

**Every conflict at N=32 is individually resolvable by waiting, and no plan is found.**

### B. Mechanism

`_waiting_resolves` is pairwise-blind: it certifies pair (a, b) while ignoring that b's
resolution — a lateral dodge — lands on top of robot c. At N=32 there is no lateral room within
a lane (§III.B), so every resolution must borrow a neighbouring lane, and every subproblem
independently chooses the same locally-cheapest place to borrow. The merge cascade is
**structural, not stochastic**. K-ARC builds one subproblem per conflicting pair and merges only
*reactively* (ARC §IV-B), which is sound exactly when conflicts are independent — and at density
they are not.

**Scenario structure.** With robot `2k` at the left of row k and `2k+1` at the right, the
same-side neighbours in adjacent rows are `2k ± 2`. Conflicts couple each row to its `n ± 2`
neighbours, so the conflict graph is a ladder. *(This structural observation is due to the
project lead, not to the analysis above; it preceded and predicted the measurement.)*

### C. This is the strongest available direction for a mechanism [O]

K-ARC's three rungs escalate **solver power** while leaving the **intervention** identical —
every rung may rewrite the entire trajectory. Nothing in the hierarchy coordinates *between*
subproblems. That is where the defect lives, and it remains untested (§VII.D explains why every
attempt so far was inconclusive).

---

## VIII. Negative and Inconclusive Results

Reported in full because each eliminates a plausible line of attack.

### A. `guide_repair` rung — DEAD [N]

**Idea.** Insert a rung between `prioritized` and the RRT rungs that re-plans the 2-D geometric
*guide* of each lower-priority robot around the partner's swept corridor, then re-solves
kinodynamically along it — changing homotopy class at 2-D cost instead of joint-state cost.

**Result**, open_cross_32, 1200 s: `guide_repair_attempts=30`, `blocked=28`, `found=2`,
`solved=2`, `composite_rrt_solves` 25 → 26 (**up**), `conflicts_remaining=15`, `plan_valid=0`.

**Diagnosed cause 1 (my bug).** Blockers were built from the entire `avoid` set — 30 other
robots' whole trajectories flattened into static walls, discarding the time dimension 30 times
over. Fixed to use only the subproblem's own robots. **Counters came back byte-identical**,
disproving this as the cause.

**Diagnosed cause 2 (the real one).** A conflict *means* robot i stands where robot j's
trajectory passes. Collapsing j's trajectory into a static corridor therefore places i's own
start — and usually its milestone — inside an obstacle, so the RRT fails at seeding before
sampling anything. Verified directly: `start blocked: True, goal blocked: True, path: None`.

**Fatal objection.** `_group_paths` (`src/approach/planning/karc.py:1103`, called
unconditionally by the `prioritized` rung at `:1185`) *already performs* masked prioritized
guide re-planning, and its docstring already names the exact trap: "Cells near a robot's own
start and goal are never masked — blocking them would make its own query unsolvable rather than
route it elsewhere." The rung duplicated existing baseline behaviour. **Do not revisit.**

### B. Random detuning (symmetry breaking) — NOT SUPPORTED [N]

**Idea.** Open Cross is exactly symmetric (identical robots, identical evenly-spaced rows,
simultaneous starts), so all subproblems compute the same resolution and collide over the same
space. Real robots are never identical; scale each robot's acceleration bounds by a factor drawn
once from [1−ε, 1] (downward only, so plans stay executable).

**Result** at ε = 0.05: N=32 still fails (`conflicts_remaining` 4 → 9, `merges` 3 → 4);
N=16 still solves but slightly slower (51.9 s → 56.1 s, conflicts 9 → 10).

**Diagnosed cause — the experiment was under-powered by ~10×.** These robots are *speed*-limited,
not acceleration-limited: the acceleration phase lasts `v_max/a_max` = 2 s while the traverse is
15 m at 0.5 m/s = 30 s. A 5% acceleration change shifts arrival by ≈0.1 s ≈ 1 timestep, whereas
clearing 0.61 m at 0.5 m/s needs ≈1.2 s ≈ 12 timesteps. **The test could not have detected the
effect either way.** This is a null result about the experiment, not about the hypothesis.

### C. Directed speed assignment — INCONCLUSIVE [M]

**Idea.** Random jitter makes robots *differ*; it does not make the *right* robot faster. A
conflict needs an **orientation**. Orient every conflict, rank the robots, turn rank into a
speed or a delay.

**Why the optimizer will not do this itself** [P]: the objective is
`obj = N·dt + effort_weight·Σ‖u‖²` (`src/approach/planning/trajopt.py:209`), **per robot**,
matching K-ARC Eq. 2's subscript i. Both terms improve monotonically by going faster and
arriving earlier. A robot *can* stagger and never *will*, because staggering is strictly worse
for its own cost. **Nothing in the formulation asks for it.** Bounds were never the constraint.

Two design axes, both implemented, both retained as ablation arms:

- **`speed_mode: offset | clamp`.** Clamping bounds does not add the missing incentive; it
  removes authority, *including the authority to dodge*. `offset` leaves every bound alone and
  buys a later start, which §IV-D-2 explicitly permits ("segments can be of different timesteps
  between different robots", with waiting states for whoever arrives first).
- **`speed_rank: colour | chain`.** A topological *precedence* rank was the wrong structure.
  Open Cross pairs are exactly symmetric about the mid-line, so every tie breaks by robot index,
  the order runs deep, and (measured on the N=32 ladder) **25 of 32 robots hit the speed floor**,
  collapsing ranks and handing **38 conflicting pairs the same rank** — destroying the very
  separation the mechanism depends on. Proper *colouring* separates neighbours in **3 ranks with
  0 shared**, because conflicting robots need to be **separated, not ordered**. Randomised greedy
  colouring with 8 restarts finds 3 ranks where a single sweep found 4.

**Results and why they are inconclusive** [M]:

| N=32 arm | conflicts remaining | composite | rounds |
|---|---|---|---|
| baseline | 4 | 14 | **1** |
| random detune (clamp) | 9 | 9 | 1 |
| directed clamp + chain | 10 | 12 | 1 |
| offset + colour | 12 | 10 | 1 |

**[!] I initially read this as a monotone worsening trend and retract that reading.** Every arm
shows `rounds=1`: at N=32 with a 600 s budget **none of them completes a single resolution
round**. `conflicts_remaining` is therefore measuring *how far each got before the clock stopped*,
not resolution quality — and the offset arm spends 32 extra solves on its global pass up front,
so it gets less far. **N=32 at 600 s cannot discriminate between these arms at all.** Any
future comparison must use N=16 (which completes in ~52 s) or N=32 with a much larger budget.

At N=16 every arm solves: baseline 51.9 s, detune 56.1 s, clamp+chain 52.1 s, offset+colour
51.7 s — i.e. uninformative in the other direction, because the baseline already succeeds.

### D. The `wait` rung — STILL UNTESTED [O]

`wait_attempts=0` in every run reported here; the rung has never fired at N=32, because all
runs used the default ladder. Testing it requires `ladder=[wait,prioritized,decoupled_rrt,
composite_rrt]` **and** a budget large enough to complete a round (§VIII.C).

---

## IX. What Should Be Done Next

Ordered by information gained per unit of compute.

1. **Bounded-horizon unsolvability — ANSWERED, and not via SAT [P].** The question was
   whether open_cross_32 is unsolvable inside the budget. It is not (§VI.D): a verified witness
   fits in 886 of 1300 steps with 5× the required clearance. A SAT UNSAT proof was scoped first
   and abandoned on its own arithmetic, recorded here because the obstruction is structural
   rather than a matter of effort. Two real boxes can sit 0.25 m apart centre-to-centre, so
   forbidding two robots per cell is sound only when the cell **diagonal** is under 0.25 m:

   | cell | τ | vars | collision clauses | co-occupancy soundly forbiddable? |
   |---|---|---|---|---|
   | 0.15 m | 0.1 s | 534 M | 8.3 **billion** | yes |
   | 1.00 m | 1.0 s | 1.2 M | 18.6 M | no |
   | 1.00 m | 5.0 s | 0.24 M | 3.7 M | no |

   Any tractable encoding must permit ~8 robots per cell to stay a sound over-approximation,
   and a relaxation that loose returns SAT for almost anything; the sound-and-tight encoding is
   not buildable. Note also that SAT decides a **horizon**, not a solver wall-clock — an
   "1800 s budget" is not a property a SAT instance can express. **Do not revisit unless the
   abstraction changes fundamentally.**

2. **Certificate-derived guides [O].** §V shows guide quality dominates; §VI produces verified
   discrete plans. Using a certificate witness to seed guides connects the strongest result to
   a mechanism. Untested.
3. **Re-run §VIII.C arms at a budget that completes a round**, or at N=16 with a scenario the
   baseline does *not* already solve. Without this the timing direction stays unresolved rather
   than refuted.
4. **20-trial seeded runs.** Every number here is n=1. The randomised greedy colouring already
   provides a natural seed axis.
5. **Cluttered Cross sweep on the Fig. 2(b) rebuild** — 4/8/16/32, never run with good guides.
   Note §V-D of the paper states "no methods can scale over 8 robots" there, so our failures at
   16/32 may reproduce the paper rather than contradict it.

---

## X. Reproduction

```bash
# Geometry table (§III.B)
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from scripts.gen_open_cross import geometry, body_diameter
b = body_diameter()
for n in (4,8,16,32):
    w,xl,xr,ys = geometry(n); s = ys[1]-ys[0]
    print(f'N={n:3d} spacing {s:6.3f} lateral {(s-b)/2:6.3f}')"

# Feasibility certificates (§VI)
.venv/bin/python scripts/feasibility_certificate.py approach=planning \
    env=open_cross_32_unicycle2 init=fixed

# Baseline + pairwise-resolvability counters (§VII)
.venv/bin/python -m main approach=planning approach.method=karc \
    env=open_cross_16_unicycle2 init=fixed approach.karc.workers=8

# Ablation arms (§VIII.C)
.venv/bin/python -m main approach=planning approach.method=karc \
    env=open_cross_32_unicycle2 init=fixed approach.karc.workers=8 \
    approach.karc.speed_assignment=true approach.karc.speed_mode=offset \
    approach.karc.speed_rank=colour
```

Raw logs for every number in this document: `experiments/session_2026-09-09/`.

**Relevant commits.** `cc4c807` four deviations closed · `c656760` Fig. 2(b) cluttered rebuild ·
`7a0ec45` shortcut default · `c475922` guide_repair rung (dead, §VIII.A) · `9a6a30c` pairwise
counters · `d7695b2` wait rung (untested) · `b12bfa5` feasibility certificate + speed assignment.

---

## XI. Honest Accounting

This document is written to be usable by someone who was not present, so the error record
matters as much as the results.

**Retracted claims [!]**, all corrected above: shortcutting "cannot leave the homotopy class"
(§V.C); density/geometry as an explanation of *infeasibility* at N=32 (§VI.C); a monotone
worsening "trend" across timing arms that was actually a timeout artifact (§VIII.C); a "16-deep
chain" reading of `speed_ranks`, which counts *robots slowed*, not chain depth (§VIII.C).

**Methodological failures worth not repeating.** A rung was built before reading the baseline
that already implemented it (§VIII.A). A hypothesis was tested with a parameter that could not
move the measured quantity (§VIII.B). Comparisons were drawn from a column governed by a
timeout rather than by quality (§VIII.C). Several 20-minute runs were spent where a 5-minute
read or a closed-form calculation would have answered the question.

**What survives all of it**, because each is independently checkable: the guide-quality result
(§V), the pairwise-vs-joint measurement (§VII), the geometry table (§III.B, as a statement about
lane-preserving solutions), and the feasibility certificates (§VI) — the last being a
constructive proof that does not depend on our planner at all.
