"""Recovering a run's config from its checkpoint path."""
from __future__ import annotations

from omegaconf import OmegaConf

from evaluate import merge_saved


def _cfgs():
    """`saved` is what the run trained with; `cfg` is what Hydra composed from defaults
    this time round. They disagree on every group, which is the situation that matters."""
    saved = OmegaConf.create({
        "env": {"name": "gap2_unicycle2", "world_size": 5.0},
        "shaping": {"type": "euclidean"},
        "network": {"type": "mlp"},
        "eval": {"episodes": 3, "checkpoint": None, "gif_path": "episode.gif"},
    })
    cfg = OmegaConf.create({
        "env": {"name": "swap2_unicycle2", "world_size": 9.0},   # hydra's default
        "shaping": {"type": "dijkstra"},
        "network": {"type": "gru"},
        "eval": {"episodes": 3, "checkpoint": None, "gif_path": "episode.gif"},
    })
    return cfg, saved


def test_untyped_groups_come_from_the_run_not_from_hydra_defaults():
    """The bug this prevents: Hydra always supplies env/shaping, so without consulting
    the override list the defaults look exactly like a user choice and silently replace
    what the run was trained on."""
    cfg, saved = _cfgs()
    out = merge_saved(cfg, saved, ["eval.checkpoint=runs/x/checkpoints/agent_1.pt"])
    assert out.env.name == "gap2_unicycle2"       # from the run
    assert out.shaping.type == "euclidean"
    assert out.network.type == "mlp"
    assert out.eval.checkpoint == "runs/x/checkpoints/agent_1.pt"   # typed, wins


def test_typed_group_overrides_the_run():
    """Group overrides name a file to compose, so they are taken from cfg wholesale
    rather than applied as key=value."""
    cfg, saved = _cfgs()
    out = merge_saved(cfg, saved, ["eval.checkpoint=c.pt", "env=swap2_unicycle2"])
    assert out.env.name == "swap2_unicycle2"
    assert out.env.world_size == 9.0          # the WHOLE group, not just the name
    assert out.shaping.type == "euclidean"    # untyped, still from the run


def test_typed_scalar_overrides_the_run():
    cfg, saved = _cfgs()
    out = merge_saved(cfg, saved, ["eval.checkpoint=c.pt", "eval.episodes=7",
                                   "+eval.fps=30"])
    assert out.eval.episodes == 7
    assert out.eval.fps == 30                 # '+' prefix stripped, value still applied
    assert out.eval.gif_path == "episode.gif"


def test_both_approaches_render_the_same_way(tmp_path, monkeypatch):
    """The GIF must not depend on which approach produced the episodes.

    evaluate.py and PlanningApproach.run each used to carry their own copy of this
    loop and had already drifted: only one held the last frame between episodes. Both
    now call run_episodes, so this pins the frame budget and the single GIF write.
    """
    import numpy as np

    from src.approach import rollout

    monkeypatch.setattr(rollout, "_render_frame",
                        lambda env, trails, step, rewards: np.zeros((2, 2, 3), np.uint8))

    class _Env:
        """Two steps per episode, then done."""
        possible_agents = ["r0"]

        def __init__(self):
            self._n, self._states, self._t = 1, [np.zeros(2)], 0
            self._reached, self._collision_count = [True], 0.0

        def reset(self):
            self.agents, self._t = list(self.possible_agents), 0
            return {"r0": np.zeros(2)}, {}

        def action_space(self, agent):
            from gymnasium.spaces import Box
            return Box(-1.0, 1.0, (2,), np.float32)

        def step(self, actions):
            self._t += 1
            if self._t >= 2:
                self.agents = []
            info = {"r0": {"episode": {"success": 1.0, "crashed": 0.0, "collisions": 0.0}}}
            return {"r0": np.zeros(2)}, {"r0": 0.0}, {}, {}, info

    class _Ctrl(rollout.Controller):
        def reset(self, env): pass
        def act(self, obs_dict, env): return {"r0": np.zeros(2, np.float32)}

    gif = tmp_path / "episode.gif"
    # frame_skip=1 so both steps are captured: 2 frames/episode, 3 episodes,
    # plus the hold after all but the last.
    stats, frames = rollout.run_episodes(_Env(), _Ctrl(), 3, gif_path=str(gif),
                                         frame_skip=1, hold=10, verbose=False)
    assert len(stats) == 3
    assert len(frames) == 3 * 2 + 2 * 10
    assert gif.is_file(), "run_episodes must write the GIF itself, not leave it to callers"

    # No gif_path -> no rendering at all, which is what fasteval relies on.
    _, none = rollout.run_episodes(_Env(), _Ctrl(), 3, gif_path=None, verbose=False)
    assert none == []
