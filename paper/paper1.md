# Admissible Distance Shaping is Exploration-Harmful for Kinodynamic Robots

*Paper 1 (theory + empirical). Working draft / research plan. Branch:
`paper1-shaping-theory`.*

## Thesis (one sentence)

> For acceleration- and curvature-constrained (kinodynamic) robots, potential-based
> reward shaping with an **admissible distance potential** (Euclidean, `φ = −‖p−g‖`) —
> which classical PBRS theory certifies as policy-invariant and which is admissible in the
> A\* sense — is **exploration-harmful**: it slows or prevents learning relative to no
> shaping in constrained/cluttered settings, and the harm grows with the constraint
> tightness (turning radius ρ and braking ratio B). A constraint-aware potential removes it.

## The gap in the literature (why this is novel)

- **Ng, Harada & Russell (1999)** prove PBRS leaves the *optimal policy set* unchanged.
  This is a statement about optima, **not about learning dynamics or sample complexity.**
- **Admissibility** (an underestimate of cost-to-go, as in A\*) is widely treated as the
  "safe" property of a heuristic. Euclidean distance is admissible. Yet — see our Paper 2 —
  Euclidean shaping *underperforms no shaping* in clutter for non-holonomic robots.
- The literature has Dubins-path shaping (aircraft guidance) and obstacle-aware Dijkstra
  cost-map potentials, but **no result tying admissible *distance* shaping to an exploration
  penalty that scales with kinodynamic constraints.** That scaling law is the contribution.

The one-line reframing reviewers should remember:
**policy-invariance is asymptotic; admissibility is about optima; neither bounds exploration.**

## Mechanism (informal)

With γ_RL < 1, the per-step shaping `F = γφ(s′) − φ(s)` does **not** telescope away during the
*pre-goal exploration phase* (before the sparse reach reward is ever discovered). For
`φ = −‖p−g‖`, F is positive exactly when the agent reduces Euclidean distance and negative
when it increases it. The solution to a cluttered/constrained task often **requires
temporarily increasing** Euclidean distance (detour around a barrier; decelerate/curve away
to respect a turning radius). Euclidean shaping therefore puts a dense local maximum at "the
closest reachable point to the goal that is not the goal" — the agent presses against the
barrier — and exploration must escape this basin against the shaping gradient. The tighter the
kinodynamic constraint, the longer the required detour and the deeper the trap.

- **Curvature knob** ρ = v_max/ω_max: tighter turning ⇒ longer feasible detour ⇒ deeper trap.
- **Momentum knob** B = v_max² / (2·a_max·r_goal): higher B ⇒ the "charge straight at the
  goal" incentive overshoots and cannot stop/turn ⇒ Euclidean is more harmful.

In the open, holonomic, low-momentum limit (ρ→0, B→0) the basin vanishes and Euclidean
shaping helps — recovering the conventional wisdom and making the claim falsifiable.

## Formal setup (to be proved)

**Corridor MDP.** Start s₀ and goal g separated by a barrier with a single detour of length L
(the agent must move to Euclidean-distance d_max > ‖s₀−g‖ before any path to g opens). Local
exploration policy (ε-greedy or bounded-entropy softmax) on a value function bootstrapped with
discount γ_RL < 1.

**Proposition 1 (invariance, restated).** Euclidean PBRS leaves the optimal policy unchanged
(Ng 1999). *(Sanity baseline — the harm is not about optima.)*

**Proposition 2 (exploration penalty).** Under the corridor MDP with γ_RL < 1 and a local
exploration scheme, the expected first-hitting time of g under Euclidean PBRS exceeds that of
no shaping by a factor increasing in the detour length L; obstacle-/kinodynamics-aware shaping
keeps it within a constant factor of the shortest feasible path. *(Proof goal: a hitting-time
/ random-walk-with-drift argument; the shaping drift points away from the detour.)*

**Proposition 3 (constraint scaling).** The minimal detour length L is monotone increasing in
ρ and in B; combined with Prop. 2, the exploration penalty of Euclidean shaping grows with
both knobs. *(Geometric: turning radius and braking distance lower-bound the deviation from
the straight line.)*

> Status: Prop. 1 is immediate. Prop. 2/3 are the work. Even if the proofs land only for the
> 1-D corridor abstraction, the empirical scaling (below) carries the paper.

## Experiments (falsifiable, runnable on this codebase)

Single second-order robot (`dyn_unicycle`), isolating shaping from multi-agent coordination.
Shaping ∈ {`none`, `euclidean`, `dijkstra`}. Potentials already implemented; the robot exposes
both knobs (`omega_max`, `a_max`). Metric: success vs training steps; steps-to-70%; final
success at a fixed budget. Seeds 0–4.

- **E1 — core (clutter).** `theory_corridor` (goal behind a barrier). Prediction:
  `euclidean ≤ none < dijkstra`. The decisive plot: Euclidean *below* no-shaping.
- **E2 — curvature sweep.** Vary `env.omega_max_override` ∈ {π, π/2, π/3, π/6}. Plot
  (none − euclidean) success gap vs ρ. Prediction: gap grows with ρ.
- **E3 — momentum sweep.** Vary `a_max` ∈ {8, 4, 2, 1}. Plot harm vs B. Prediction: grows.
- **E4 — open-world control.** `theory_open` (no barrier). Prediction: `euclidean ≳ none`
  (harm vanishes) — proves the harm is constraint-induced, not generic.

Runner: `experiments/run_theory.sh` (sweeps shaping × knob × seed, logs success curves).

## Novelty / risk / venue

- **Contribution:** (i) a negative result — admissible distance shaping hurts exploration
  under kinodynamic constraints; (ii) a scaling law in ρ and B; (iii) the
  policy-invariance≠sample-complexity reframing; (iv) constraint-aware shaping as the fix.
- **Risk:** Prop. 2/3 may only close for the abstract corridor; mitigated by the empirical
  scaling carrying the result. Reviewer "just use a cost map" → answered: the point is *why*
  the textbook-safe choice fails, with a predictive law, not merely that a better φ exists.
- **Venue:** workshop (RLC / NeurIPS) for the empirical+abstract-theory version; conference
  (AAMAS / CoRL) if Prop. 2 closes in 2-D.

## References

- Ng, Harada, Russell. "Policy invariance under reward transformations." ICML 1999.
- Wiewiora. "Potential-based shaping and Q-value initialization are equivalent." JAIR 2003.
- Devlin & Kudenko. "Dynamic potential-based reward shaping." AAMAS 2012.
- Dubins. "On curves of minimal length…" Amer. J. Math 1957.
- Donald, Xavier, Canny, Reif. "Kinodynamic motion planning." JACM 1993.
- Schulman et al. "Proximal Policy Optimization." arXiv:1707.06347, 2017.
- Reward-shaping navigation survey, arXiv:2408.10215, 2024 (cost-map potentials).
