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
------

# K-ARC: Kinodynamic Optimization Framework
Markdown
---

# Multi-Robot Kinodynamic Motion Planning (K-ARC Framework)

This repository contains an implementation of the **Kinodynamic Asynchronous Replanning for Conflicts (K-ARC)** algorithm for multi-robot trajectory planning (`enew.py`). The implementation computes collision-free, dynamically feasible trajectories in dense symmetric interchange environments using non-linear optimization and prioritized subproblem decomposition.

---

## 1. Algorithmic Overview

The framework operates in a decoupled-to-coupled hierarchical manner across four stages:

[Kinematic Reference Path]
│
▼
[Segmented Horizon Decomposition]
│
▼
[Uncoordinated Trajectory Optimization] ──(Conflict Detected?)──► [Prioritized Subproblem Resolution]
│                                                                     │
└───────────────────────────► (No Conflicts) ◄────────────────────────┘
│
▼
[Final Coordinated Path]

### Module Descriptions

*   **`KinematicPlanner` (Algorithm 1, Lines 3–4):**  
    Generates initial collision-free reference geometric paths. For symmetric multi-robot configurations ($N > 4$), it injects priority-based tangential offsets at the workspace center to break symmetry and avoid gradient-based local minima.
*   **`CasadiTrajectoryOptimizer` (Non-linear Programming):**  
    Formulates a continuous-time kinodynamic trajectory optimization problem using **CasADi + IPOPT**. Trajectories are integrated using Runge-Kutta 4th Order (RK4) integration over differential-drive unicycle dynamics:
    $$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \omega, \quad \dot{v} = a, \quad \dot{\omega} = \alpha$$
*   **Segmented Execution Loop (`K_ARC_Algorithm_1`):**  
    Divides the global kinematic plan into $m$ segments ($m=4$ by default) to enable sequential trajectory generation and conflict monitoring.
*   **Conflict Resolution (`FindConflicts` & `SolveSubProblem`):**  
    Detects pairwise spatio-temporal collisions ($D < 0.75\,\text{m}$). Robots involved in conflicts are grouped into a subproblem and resolved hierarchically, where lower-priority robots treat higher-priority trajectories as moving obstacles.

---

## 2. Optimization Formulation

For each robot, the optimizer minimizes control effort, control jerk, trajectory tracking error, and terminal error:

$$\min_{X, U} \sum_{k=0}^{N-1} \left( w_u \Vert{}u_k\Vert{}^2 + w_{\Delta u} \Vert{}u_{k+1} - u_k\Vert{}^2 \right) + w_{\text{track}} \sum_{k=0}^{N} \Vert{}p_k - p_{\text{ref}, k}\Vert{}^2 + w_{\text{goal}} \Vert{}p_N - p_{\text{goal}}\Vert{}^2$$

### Constraints
*   **Kinodynamic bounds:** Linear velocity $v \in [-2.8, 2.8]\,\text{m/s}$, angular velocity $\omega \in [-2.5, 2.5]\,\text{rad/s}$, control inputs $a, \alpha \in [-2.5, 2.5]\,\text{m/s}^2$.
*   **Static obstacle clearance:** Enforced via Euclidean distance constraints (circles) and superquadric approximations (rectangles).
*   **Inter-robot dynamic clearance:** Enforced on higher-priority trajectory footprints:
    $$\Vert{}p_i(t) - p_j(t)\Vert{}^2 \ge d_{\text{safe}}^2 \quad (\forall j < i)$$

---

## 3. System Requirements & Installation

### Prerequisites
*   **Python 3.8+**
*   **FFmpeg** (required by Matplotlib to render and save `.mp4` animations)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/skasifiqbal-cs/kinodynamic-marl.git](https://github.com/skasifiqbal-cs/kinodynamic-marl.git)
   cd kinodynamic-marl
Install FFmpeg:
macOS (Homebrew):
Bash
brew install ffmpeg
Ubuntu / Linux:
Bash
sudo apt update && sudo apt install -y ffmpeg
Windows:
Download from gyan.dev and add the bin folder to your system PATH.
Install Python dependencies:
Bash
pip install casadi numpy matplotlib
4. Running the Simulation
Execute the main script:
Bash
python enew.py
Scenario Configuration
To test different robot densities (e.g., N=2,4,5,6,7), modify the NUM_ROBOTS variable in enew.py:
Python
# Change to 4, 5, 6, or 7 robots
NUM_ROBOTS = 7
5. Output Deliverables
Upon completion, the script generates and opens three animation videos:
Video File	Algorithm Stage	Description
video_alg1_lines_3_4.mp4	Alg 1, Lines 3–4	Geometric kinematic reference paths with symmetry-breaking offsets.
video_alg1_lines_17_18.mp4	Alg 1, Lines 17–18	Uncoordinated local trajectories demonstrating spatial-temporal bottlenecks.
video_alg1_lines_20_21.mp4	Alg 1, Lines 20–22	Final collision-free coordinated multi-robot trajectory.
