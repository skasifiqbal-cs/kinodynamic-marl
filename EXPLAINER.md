# Understanding This Repo: Multi-Agent Kinodynamic RL

A plain-language guide to what every component does, the math behind it,
and where to take the research next.

---

## 1. The Problem Being Solved

Two (or more) robots share a workspace. Each has a start position and a goal.
They must navigate to their goals without hitting each other or obstacles.
The challenge: each robot decides its own actions independently, in real time,
knowing only what it can observe — and what one robot does affects what the
other should do.

This is the core problem of **multi-agent motion planning under uncertainty**,
and we solve it with reinforcement learning instead of hand-crafted planners.

---

## 2. Formal Setting: MDP, Game, or What?

### Single-agent baseline: MDP
A single robot navigating alone is a **Markov Decision Process (MDP)**:

```
State s  →  Agent picks action a  →  Environment returns (s', reward)
```

The agent learns a **policy** π(a | s): given state, what action to take.

### Multi-agent: Stochastic Game (Markov Game)
With N agents, the world is a **Stochastic Game** (also called a Markov Game):

```
Joint state (s₁, s₂)  →  Each agent picks aᵢ independently
                       →  Environment returns (s₁', s₂', r₁, r₂)
```

Key difference from MDP: **each agent's reward depends on what the other does**.
If agent 0 blocks the corridor, agent 1 gets a collision penalty even if agent 1
did nothing wrong.

### Cooperative or Competitive?
This repo is **cooperative**: both agents want to succeed. Neither benefits from
the other failing. But they have *separate* reward functions (each cares about
its own goal), which makes it a **cooperative Markov game** — not a zero-sum
game, not a fully shared-reward problem.

Formal name: **Dec-POMDP** (Decentralized Partially Observable MDP) when:
- Agents share a global objective (cooperative)
- Each agent only sees a local observation, not the full joint state
- Decisions are made independently (decentralized)

This repo fits that description:
- Global objective: both agents reach their goals
- Each agent observes only its own pose + goal + relative positions (partial)
- Each agent runs its own policy (decentralized)

---

## 3. What Each Component Does

### `conf/robot/*.yaml` — Robot Specification

Defines **how a robot moves**. Not a simulation — just numbers:

```yaml
# unicycle_v1.yaml
type: unicycle
v_max: 1.0        # max forward speed (m/s)
omega_max: 3.14   # max rotation speed (rad/s)
shape:
  type: circle
  radius: 0.13    # physical footprint for collision
```

A robot config is loaded by name at runtime. Adding a new robot = one new file.

---

### `src/robot/` — Dynamics Models

Answers: **given current state and an action, what is the next state?**

**Unicycle model** — the simplest steerable robot:
- State: `[x, y, θ]` — position and heading
- Action: `[v, ω]` — forward speed and rotation rate
- Physics: `ẋ = v·cos θ`, `ẏ = v·sin θ`, `θ̇ = ω`

**Kinematic car (bicycle model)** — like a car that can only go forward:
- Same state `[x, y, θ]`
- Action: `[v, δ]` — speed and steering angle
- Physics: turning radius depends on speed and steering: `θ̇ = v/L · tan(δ)`

Both use **RK4 integration** — a 4th-order numerical integration method that
is significantly more accurate than naive Euler stepping at the same timestep.
This matters for fast-moving robots or sharp turns.

---

### `src/collision/` — Geometry and Collision Detection

Handles two questions:
1. **Are two objects touching?** (for penalty computation)
2. **How far is the nearest obstacle in a given direction?** (for lidar)

Supports two shape types:

| Shape | Used for | Collision method |
|---|---|---|
| Circle | Most robots, round obstacles | Distance < r₁ + r₂ |
| Box (OBB) | Car robot, wall obstacles | SAT (Separating Axis Theorem) |

**OBB** = Oriented Bounding Box. Unlike axis-aligned boxes, OBBs rotate with
the robot. A car-shaped robot navigating a corner keeps its box aligned to its
heading, not to the world grid.

**SAT**: to check if two convex shapes overlap, find any axis along which their
projections don't overlap — if one exists, they're separate. If no such axis
exists, they collide.

---

### `src/shaping/` — Reward Shaping Potentials

Raw rewards (reach goal: +10, collision: -5) are **sparse** — the agent rarely
hits them early in training, so learning is slow. Potential-based shaping adds
a **dense signal** that guides the agent without changing the optimal policy.

The shaping bonus per step is:
```
F(s, s') = γ · φ(s') - φ(s)
```

This rewards the agent for *moving toward a state with higher potential*.
Because this is a difference of potentials, it's **policy-invariant**: any
optimal policy under the original reward is still optimal with shaping added.

Three potentials implemented:

| Name | Formula | What it measures |
|---|---|---|
| `none` | φ = 0 | No guidance — pure sparse reward |
| `euclidean` | φ = −‖pos − goal‖ | Straight-line distance to goal |
| `dubins` | φ = −L_dubins / v_max | Shortest path a car-like robot can physically take |

**Dubins path**: the shortest curve connecting two poses (position + heading)
for a vehicle with a minimum turning radius. Unlike Euclidean distance, it
accounts for the robot's orientation — a robot facing away from its goal
needs to turn first, which the Euclidean potential ignores.

---

### `src/obs/` — What Each Agent Sees

Defines the **observation vector** fed into the neural network.
The choice of observation design significantly affects what the policy can learn.

**Full-state observer** (`full_state`):
```
obs = [x, y, sin θ, cos θ,        ← own pose (4D — sin/cos avoids θ discontinuity)
       gx, gy, sin gθ, cos gθ,    ← goal pose (4D)
       Δx, Δy, ...,               ← relative position of each other agent (2D each)
       rx, ry, hw, hl, sin a, cos a, ...]  ← each obstacle: position + shape (6D each)
```

Why `sin/cos` instead of raw angle? An angle like θ = π and θ = -π are the
same direction but have value difference 2π — a discontinuity that confuses
neural networks. Using `[sin θ, cos θ]` encodes direction continuously.

**Lidar observer** (`lidar`):
```
obs = [x, y, sin θ, cos θ,    ← own pose
       gx, gy, sin gθ, cos gθ, ← goal
       d₁, d₂, ..., d_N]       ← N range readings (normalised 0–1)
```

Casts N rays outward and returns distance to nearest object in each direction.
Lidar is **more general**: it doesn't require knowing obstacle positions
explicitly, so a policy trained with lidar can potentially transfer to new
environments (sim-to-real). Full-state is easier to train but harder to transfer.

---

### `src/init/` — Episode Initialisation

Decides where agents start and where goals are placed at the beginning of
each training episode.

| Mode | Behaviour | Good for |
|---|---|---|
| `fixed` | Same start/goal every episode | Debugging, simple tasks |
| `random_heading` | Fixed positions, random orientation | Heading generalisation |
| `random` | Fully random (collision-free) positions | Robust, general policies |

Fixed initialisation is the fastest way to see a policy work, but agents
can memorise a single trajectory instead of learning a general strategy.
Random initialisation forces the policy to generalise.

---

### `src/env/multiagent_nav.py` — The Simulation Loop

This is the **world** — it ties everything together. At each timestep:

```
1. Each agent's policy outputs an action [v, ω]
2. Robot dynamics (RK4) integrate: state → next state
3. Collision detection checks all agent-agent and agent-obstacle pairs
4. Rewards computed:
   - Step penalty (small negative — encourages speed)
   - Shaping bonus: γ·φ(s') − φ(s)
   - Collision penalty (if touching)
   - Goal bonus (if within goal_radius — one-shot)
5. Termination: all reached goals → episode ends
   Timeout: max_steps exceeded → episode truncates
6. New observations built and returned to policies
```

This follows the **PettingZoo Parallel API** — the standard interface for
multi-agent environments in Python. All agents act simultaneously each step
(parallel, not turn-based).

---

### `src/networks/` — The Policy Neural Networks

Each agent has two networks:

**Policy network** (actor): maps observation → action distribution
```
obs → [Linear → Tanh] × n_layers → Linear → mean action
                                             + learned log_std parameter
```
Outputs a **Gaussian distribution** over actions. During training, actions
are sampled from this distribution (exploration). During evaluation, the
mean is used (deterministic).

**Value network** (critic): maps observation → expected return
```
obs → [Linear → Tanh] × n_layers → Linear → scalar V(s)
```
Used only during training to compute **advantage estimates** — how much
better was the action taken compared to the average action in this state.

Both MLP (feedforward) and GRU (recurrent) architectures are available.
GRU is useful when the observation is **partially observable** and the agent
benefits from memory across timesteps.

---

### `src/training/train.py` — The Learning Algorithm: IPPO

**PPO** (Proximal Policy Optimisation): the workhorse RL algorithm for
continuous control. Core idea: improve the policy, but not too much in one
update (clip the gradient step to stay "proximal" to the old policy).

**IPPO** (Independent PPO): run one PPO instance per agent. Each agent:
- Collects its own experience (obs, action, reward, next obs)
- Trains its own policy and value network independently
- Treats other agents as part of the environment (not modelled explicitly)

This is the **simplest possible multi-agent approach** — and surprisingly
effective in cooperative settings. The non-stationarity (other agents are
also changing their policies) can cause instability, but in practice IPPO
works well for cooperative tasks.

Training loop:
```
for each rollout:
    collect rollouts episodes of experience
    for each PPO epoch:
        shuffle experience into mini-batches
        compute advantages: A(s,a) = returns - V(s)
        update policy:  maximise clipped_ratio × advantage
        update value:   minimise (V(s) - return)²
```

Checkpoints saved every 50k steps to `runs/`. wandb tracks reward curves.

---

### `conf/` — Configuration System (Hydra)

Every axis of variation has its own config file. The top-level `config.yaml`
just names which variant to use:

```yaml
defaults:
  - env: crossing      # which scenario
  - shaping: euclidean # which potential
  - network: mlp       # which architecture
  - obs: full_state    # what agents observe
  - init: fixed        # how episodes start
  - train: ppo_default # PPO hyperparameters
```

Override anything from the command line — no code changes needed:
```bash
python train.py env=corridor shaping=dubins network=gru obs=lidar
```

---

## 4. How a Training Step Works End-to-End

```
Config loaded by Hydra
    │
    ├─ load_robot_cfg("unicycle_v1") → UnicycleModel(v_max=1.0, shape=CircleShape)
    ├─ build_potential(cfg) → EuclideanPotential
    ├─ build_obs_builder(cfg) → FullStateObsBuilder (obs_dim=22)
    ├─ build_initializer(cfg) → FixedInitializer
    │
    └─ MultiAgentNav created
           │
           ▼
    SKRL wraps env → IPPO agent (one Policy + Value per agent)
           │
           ▼
    SequentialTrainer.train()
       │
       └─ for each timestep:
              obs = env.reset() or carry over
              action = policy.act(obs)          ← sample from Gaussian
              obs', reward, done = env.step(action)
                  │
                  ├─ RK4 integrate robot dynamics
                  ├─ check OBB/circle collisions
                  ├─ compute shaping: γ·φ(s') - φ(s)
                  └─ return reward + termination
              store (obs, action, reward, obs', done) in RandomMemory
              if memory full → PPO update (10 epochs, 4 mini-batches)
              checkpoint every 50k steps
```

---

## 5. Future Features to Explore

### Research Questions This Repo Is Designed For

**Reward shaping comparison** (already supported):
- Do agents trained with Dubins shaping reach goals faster than Euclidean?
- Does shaping help more with car robots (non-holonomic) than unicycles?
- At what gamma value does shaping stop helping?

**Observation design**:
- Can a lidar-trained policy transfer to a new obstacle layout?
- Does full-state knowledge create brittle policies?

**Robot heterogeneity**:
- Do two different robot types (unicycle + car) learn to coordinate?
- Does one robot's policy hurt when the other is replaced?

---

### Near-Term Extensions (Low Effort)

| Feature | Where to add | Why |
|---|---|---|
| Reward annealing (decay step penalty over training) | `conf/train/` + env | Curriculum: first learn to reach goal, then learn to be fast |
| Moving obstacles | `src/env/multiagent_nav.py` | More realistic; obstacles become agents |
| Communication channel | Obs builder + agents config | Agents share info → compare with no-comm |
| Velocity in observation | `src/obs/full_state.py` | Help agents predict collisions earlier |
| Asymmetric roles | Env yaml | One agent is a slower "loader", one is a faster "runner" |

---

### Medium-Term Extensions (Moderate Effort)

**MAPPO** (Multi-Agent PPO with centralised critic):
- Same as IPPO but each agent's value network sees the **joint state** of all agents
- Reduces non-stationarity; often outperforms IPPO
- Requires changing the value network's input to concatenate all agents' observations

**Curriculum learning**:
- Start with easy scenarios (no obstacles, short distance)
- Progressively increase difficulty as success rate improves
- Add `conf/curriculum/` config group + `RandomInitializer` difficulty parameter

**Multi-task training**:
- Train on `crossing` + `corridor` + `three_agent` simultaneously
- Tests whether one policy generalises across scenarios
- Hydra multirun: `python train.py --multirun env=crossing,corridor,three_agent`

**Sim-to-real**:
- Add noise to observations and dynamics (domain randomisation)
- Train with lidar obs (easier to match real sensors)
- Export policy to TorchScript for deployment: `torch.jit.script(policy)`

---

### Longer-Term Research Directions

**Social force / potential field baselines**:
- Implement a hand-crafted planner as a baseline
- Compare convergence speed and final performance against RL

**Graph Neural Network policies**:
- Each agent = node; relative positions = edges
- Policy is a GNN: naturally handles variable number of agents
- Replace `src/networks/policy.py` with a GNN variant

**Safe RL**:
- Add a constraint: collision rate < 5% (Constrained MDP)
- Use algorithms like CPO or Lagrangian PPO
- Critical for real-robot deployment

**Opponent modelling**:
- Each agent maintains a model of the other's policy
- Use the model to predict future positions → plan ahead
- Transforms IPPO into something closer to planning

**Hierarchical RL**:
- High-level policy: decides waypoints
- Low-level policy: unicycle/car tracks the waypoint
- Decomposes long-horizon planning into tractable subproblems

---

## 6. Key Papers to Read

| Paper | Why relevant |
|---|---|
| Schulman et al. 2017 — PPO | The algorithm used for training |
| Lowe et al. 2017 — MADDPG | Introduced centralised training + decentralised execution |
| Yu et al. 2022 — MAPPO | IPPO vs MAPPO comparison; often cited for cooperative tasks |
| Ng et al. 1999 — Potential-based shaping | Proves shaping is policy-invariant |
| Dubins 1957 — Curves of minimal length | Foundation for DubinsPotential |
| Everett et al. 2021 — Collision avoidance with DNNs | RL for multi-robot nav survey |
