"""Recovering a run's config from its checkpoint path."""
from __future__ import annotations

from omegaconf import OmegaConf

from evaluate import merge_saved


def _cfgs():
    """`saved` is what the run trained with; `cfg` is what Hydra composed from defaults
    this time round. They disagree on every group, which is the situation that matters."""
    saved = OmegaConf.create({
        "env": {"name": "gap_2agent", "world_size": 5.0},
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
    assert out.env.name == "gap_2agent"       # from the run
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
