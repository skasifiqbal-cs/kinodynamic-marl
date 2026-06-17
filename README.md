# Kinodynamic RL — 2-Agent Navigation with Potential-Based Reward Shaping

Two unicycle robots learn to reach their goals in a shared workspace.
Reward shaping potential is swappable via a single config field.

---

## Stack

| Component | Library |
|---|---|
| Config | Hydra 1.3 |
| Neural nets / tensors | PyTorch |
| Multi-agent RL | SKRL (IPPO) |
| Environment contract | PettingZoo Parallel API |
| Robot model | Custom RK4 unicycle |
| Dubins paths | `dubins` (optional) |

---

## Setup

### 1. Create and activate virtual environment

```bash
cd kinodynamic_rl
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note — `dubins` package:** requires C build tools (`gcc`).
> If install fails, run:
> ```bash
> sudo apt install python3-dev build-essential   # Ubuntu/Debian
> pip install dubins
> ```
> Without `dubins`, the `none` and `euclidean` shapers work fine.
> `DubinsPotential` only imports `dubins` at call time — no crash at startup.

---

## Running

Always activate the venv first:

```bash
source .venv/bin/activate
```

### Train with default config (Euclidean shaping)

```bash
python train.py
```

### Swap shaping potential — one flag

```bash
python train.py shaping=none        # no shaping
python train.py shaping=euclidean   # negative Euclidean distance (default)
python train.py shaping=dubins      # negative Dubins path length / v_max
```

### Override any config value on the command line

```bash
# More timesteps
python train.py train.timesteps=2000000

# Dubins shaping + custom turning radius
python train.py shaping=dubins shaping.min_turning_radius=0.3

# Change robot speed limits
python train.py env.robot.v_max=2.0 env.robot.omega_max=2.0

# Reproducible seed
python train.py train.seed=123
```

### Outputs

Hydra writes logs and checkpoints to:

```
outputs/YYYY-MM-DD/HH-MM-SS/
```

---

## Project Structure

```
kinodynamic_rl/
├── conf/
│   ├── config.yaml            # main config — edit defaults here
│   └── shaping/
│       ├── none.yaml          # NoPotential
│       ├── euclidean.yaml     # EuclideanPotential (default)
│       └── dubins.yaml        # DubinsPotential
├── src/
│   ├── robot/
│   │   └── unicycle.py        # UnicycleModel — state [x,y,θ], action [v,ω], RK4
│   ├── shaping/
│   │   ├── base.py            # BasePotential ABC — phi(state, goal) -> float
│   │   ├── no_potential.py    # phi = 0
│   │   ├── euclidean.py       # phi = -||pos - goal||
│   │   ├── dubins_potential.py# phi = -dubins_length / v_max
│   │   └── __init__.py        # build_potential(cfg, v_max) factory
│   ├── env/
│   │   └── two_agent_nav.py   # PettingZoo ParallelEnv
│   └── training/
│       └── train.py           # SKRL IPPO setup and trainer
└── train.py                   # Hydra entry point
```

---

## Key Config Fields (`conf/config.yaml`)

```yaml
defaults:
  - shaping: euclidean   # ← change this to: none | euclidean | dubins

env:
  dt: 0.05               # simulation timestep (seconds)
  max_steps: 500         # episode length cap
  goal_radius: 0.2       # success threshold (metres)
  collision_penalty: -5.0
  reach_reward: 10.0
  robot:
    v_max: 1.0           # max linear speed (m/s)
    omega_max: 3.14159   # max angular speed (rad/s)
  agents:
    - start: [0.0, 0.0, 0.0]   # agent 0: [x, y, theta]
      goal:  [3.0, 3.0, 0.0]
    - start: [3.0, 0.0, 0.0]   # agent 1
      goal:  [0.0, 3.0, 0.0]
  obstacles:             # [x, y, radius]
    - [1.5, 1.5, 0.3]
    - [2.0, 0.5, 0.2]

train:
  timesteps: 1_000_000
  learning_rate: 3.0e-4
  discount: 0.99
  hidden_size: 128
  num_layers: 2
  seed: 42
```

---

## Reward Structure

Per agent per step:

```
r = F_shaping + r_collision + r_goal
```

| Term | Value |
|---|---|
| Shaping `F` | `γ · φ(s') − φ(s)` |
| Agent–agent collision | `collision_penalty` |
| Agent–obstacle collision | `collision_penalty` |
| Reaching goal (once) | `reach_reward` |

---

## Adding a New Potential

1. Create `src/shaping/my_potential.py`, subclass `BasePotential`, implement `phi(state, goal) -> float`.
2. Add `"my_type"` branch in `src/shaping/__init__.py → build_potential()`.
3. Add `conf/shaping/my_type.yaml`.
4. Run: `python train.py shaping=my_type`.

---

## Swapping the Robot Model

`UnicycleModel.step(state, action, dt) -> state` is the only interface the env calls.
Replace `src/robot/unicycle.py` with any model that respects that signature,
then update `env.robot` config fields accordingly.
