"""Interactive checkpoint viewer.

Usage:
    cd ~/kinodynamic_rl
    streamlit run scripts/viewer.py
"""
from __future__ import annotations

import glob
import io
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st
from PIL import Image

# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_checkpoints() -> Dict[str, List[str]]:
    ckpts = sorted(glob.glob("runs/**/*.pt", recursive=True), reverse=True)
    by_run: Dict[str, List[str]] = {}
    for c in ckpts:
        run_dir = str(Path(c).parent.parent)
        by_run.setdefault(run_dir, []).append(c)
    return by_run


def infer_overrides(run_dir: str) -> List[str]:
    """Parse shaping/network/obs from directory name like 'dubins_mlp_full_state'."""
    parent_name = Path(run_dir).parent.name
    overrides: List[str] = []
    rest = parent_name
    for shaping in ("dubins", "euclidean", "none"):
        if rest.startswith(shaping):
            overrides.append(f"shaping={shaping}")
            rest = rest[len(shaping) + 1:]
            break
    for network in ("gru", "mlp"):
        if rest.startswith(network):
            overrides.append(f"network={network}")
            rest = rest[len(network) + 1:]
            break
    for obs in ("full_state", "lidar"):
        if rest.startswith(obs):
            overrides.append(f"obs={obs}")
            break
    return overrides


# ── Env + policy loading (cached per checkpoint) ──────────────────────────────

@st.cache_resource(show_spinner="Loading checkpoint…")
def load_checkpoint(ckpt_path: str, run_dir: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    overrides = infer_overrides(run_dir)
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose("config", overrides=overrides)

    from skrl.resources.preprocessors.torch import RunningStandardScaler

    from src.env.multiagent_nav import MultiAgentNav
    from src.init import build_initializer
    from src.networks import build_policy
    from src.obs import build_obs_builder
    from src.robot import build_robot, load_robot_cfg
    from src.shaping import build_potential

    agent_cfgs = list(cfg.env.agents)
    robots = [build_robot(load_robot_cfg(a.robot)) for a in agent_cfgs]
    potentials = [build_potential(cfg, v_max=getattr(r, "v_max", 1.0)) for r in robots]
    obs_builder = build_obs_builder(
        cfg.obs,
        n_agents=len(robots),
        n_obstacles=len(cfg.env.obstacles),
        world_size=float(cfg.env.world_size),
    )
    initializer = build_initializer(cfg.init)
    env = MultiAgentNav(cfg, robots, potentials, obs_builder, initializer)

    device = torch.device("cpu")
    policies = {}
    preprocessors = {}
    ckpt = torch.load(ckpt_path, map_location=device)
    for agent_id in env.possible_agents:
        policies[agent_id] = build_policy(
            env.observation_space(agent_id),
            env.action_space(agent_id),
            device,
            cfg.network,
        ).to(device)
        preprocessors[agent_id] = None
        if agent_id in ckpt and "policy" in ckpt[agent_id]:
            policies[agent_id].load_state_dict(ckpt[agent_id]["policy"], strict=False)
        elif f"{agent_id}/policy" in ckpt:
            policies[agent_id].load_state_dict(ckpt[f"{agent_id}/policy"], strict=False)
        policies[agent_id].eval()
        # Restore observation normaliser (policy trained on normalised obs)
        if agent_id in ckpt and ckpt[agent_id].get("state_preprocessor"):
            pp = RunningStandardScaler(size=env.observation_space(agent_id), device=device)
            pp.load_state_dict(ckpt[agent_id]["state_preprocessor"])
            pp.eval()
            preprocessors[agent_id] = pp

    return env, policies, preprocessors, cfg, device


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, policies, preprocessors, device, frame_skip: int = 5, seed: int = 42,
                progress_cb=None):
    from src.viz.renderer import render_frame_with_shapes

    max_steps = env.cfg.env.max_steps
    obs_dict, _ = env.reset(seed=seed)
    trails = [[] for _ in range(env._n)]
    frames: List[np.ndarray] = []
    frame_rewards: List[Dict[str, float]] = []
    total_rewards = {a: 0.0 for a in env.possible_agents}
    step = 0

    while env.agents:
        actions = {}
        for agent in env.possible_agents:
            if agent not in env.agents:
                continue
            obs_t = torch.tensor(obs_dict[agent], dtype=torch.float32,
                                 device=device).unsqueeze(0)
            if preprocessors.get(agent) is not None:
                obs_t = preprocessors[agent](obs_t, train=False)
            with torch.no_grad():
                mean_act, _, _ = policies[agent].compute({"states": obs_t}, role="policy")
            act = mean_act.squeeze(0).cpu().numpy()
            act = np.clip(act, env.action_space(agent).low, env.action_space(agent).high)
            actions[agent] = act

        obs_dict, rewards, _, _, _ = env.step(actions)
        for a, r in rewards.items():
            total_rewards[a] += r
        for i in range(env._n):
            trails[i].append(env._states[i][:2].copy())

        if step % frame_skip == 0:
            frame = render_frame_with_shapes(
                states=[s.copy() for s in env._states],
                robot_shapes=[r.shape for r in env.robots],
                goals=[g.copy() for g in env._goals],
                obstacles=env._obstacles,
                trails=[list(t) for t in trails],
                world_size=env.cfg.env.world_size,
                reached=list(env._reached),
                step=step,
                step_rewards=rewards,
                fig_px=320,
            )
            frames.append(frame)
            frame_rewards.append(dict(rewards))
            if progress_cb is not None:
                progress_cb(step, max_steps)

        step += 1

    return frames, frame_rewards, {
        "total_reward": total_rewards,
        "steps": step,
        "success": all(env._reached),
    }


def frames_to_gif(frames: List[np.ndarray], fps: int = 15) -> bytes:
    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(
        buf, format="GIF", save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps), loop=0,
    )
    return buf.getvalue()


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kinodynamic RL Viewer",
    layout="wide",
    page_icon="🤖",
)
st.title("Kinodynamic RL — Checkpoint Viewer")

checkpoints_by_run = find_checkpoints()

if not checkpoints_by_run:
    st.error("No checkpoints found in `runs/`. Train first:  `python train.py`")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    run_options = list(checkpoints_by_run.keys())
    run_dir = st.selectbox(
        "Run",
        run_options,
        format_func=lambda p: f"{Path(p).parent.name} / {Path(p).name}",
    )

    ckpt_options = checkpoints_by_run[run_dir]
    ckpt_path = st.selectbox(
        "Checkpoint",
        ckpt_options,
        format_func=lambda p: Path(p).name,
    )

    inferred = infer_overrides(run_dir)
    if inferred:
        st.caption("Inferred config: " + "  |  ".join(inferred))

    st.divider()
    frame_skip = st.slider("Frame skip", 1, 10, 5,
                           help="Render every N steps — skip=5 → ~200 frames, ~8s")
    seed = int(st.number_input("Seed", value=42, min_value=0, max_value=9999))
    fps = st.slider("Playback FPS", 5, 30, 15)

    st.divider()
    run_btn = st.button("▶ Run Episode", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([3, 1])

if run_btn:
    env, policies, preprocessors, cfg, device = load_checkpoint(ckpt_path, run_dir)
    max_steps = cfg.env.max_steps
    n_render = max_steps // frame_skip
    pbar = st.progress(0, text=f"Step 0 / {max_steps}  (rendering {n_render} frames)")

    def _progress(step, total):
        pct = min(step / total, 1.0)
        pbar.progress(pct, text=f"Step {step} / {total}")

    frames, frame_rewards, stats = run_episode(
        env, policies, preprocessors, device,
        frame_skip=frame_skip, seed=seed,
        progress_cb=_progress,
    )
    pbar.progress(1.0, text=f"Done — {len(frames)} frames")
    st.session_state.update(
        frames=frames,
        frame_rewards=frame_rewards,
        stats=stats,
        frame_idx=0,
        playing=False,
    )

if "frames" not in st.session_state:
    with col_main:
        st.info("Select a checkpoint and press **▶ Run Episode** to start.")
    st.stop()

frames: List[np.ndarray] = st.session_state["frames"]
frame_rewards: List[Dict[str, float]] = st.session_state["frame_rewards"]
stats: dict = st.session_state["stats"]
n_frames = len(frames)

# ── Stats ─────────────────────────────────────────────────────────────────────
with col_side:
    st.subheader("Episode stats")
    st.metric("Success", "✓ Yes" if stats["success"] else "✗ No")
    st.metric("Steps", stats["steps"])
    st.divider()
    for agent, r in sorted(stats["total_reward"].items()):
        st.metric(f"Total reward ({agent})", f"{r:.1f}")

    st.divider()
    st.caption("GIF download available in main panel.")

# ── Playback ──────────────────────────────────────────────────────────────────
with col_main:
    # Animated GIF — auto-plays and loops in browser
    gif_bytes = frames_to_gif(frames, fps=fps)
    st.image(gif_bytes, width=480)
    st.download_button("⬇ Download GIF", data=gif_bytes,
                       file_name="episode.gif", mime="image/gif")

    st.divider()

    # Scrub slider for per-frame inspection
    frame_idx = st.slider("Scrub frame", 0, n_frames - 1, 0)
    st.image(frames[frame_idx], width=480)

    if frame_rewards and frame_idx < len(frame_rewards):
        r_this = frame_rewards[frame_idx]
        r_cols = st.columns(len(r_this))
        for col, (agent, r) in zip(r_cols, sorted(r_this.items())):
            col.metric(f"Step reward ({agent})", f"{r:+.3f}")
