"""conf/experiment/local.yaml must override the committed defaults, and be optional.

Guards the `- optional experiment: local` entry in conf/config.yaml. Two things can
silently break it: dropping the entry (personal settings stop applying) and listing it
before `_self_` (settings apply but lose to the defaults). Both show up here.

The rest of the suite already composes `config` with no local.yaml present, which covers
the "absent" half -- this adds the "present" half without writing into the real conf/.
"""
from __future__ import annotations

import os
import shutil

from hydra import compose, initialize_config_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL = """# @package _global_
defaults:
  - override /env: swap2_unicycle2
  - override /shaping: braking
eval:
  episodes: 7
"""


def test_local_experiment_overrides_defaults(tmp_path):
    conf = tmp_path / "conf"
    shutil.copytree(os.path.join(ROOT, "conf"), conf)
    (conf / "experiment").mkdir(exist_ok=True)
    (conf / "experiment" / "local.yaml").write_text(LOCAL)

    with initialize_config_dir(config_dir=str(conf), version_base="1.3"):
        cfg = compose("config")

    # Committed defaults are gap_2agent (world 6.0) / dijkstra / episodes 1.
    assert cfg.env.world_size == 5.0, "local.yaml did not override the env group"
    assert cfg.shaping.type == "braking", "local.yaml did not override the shaping group"
    assert cfg.eval.episodes == 7, "local.yaml lost to _self_ -- it must be listed last"


def test_composes_without_local_experiment(tmp_path):
    """A fresh clone has no local.yaml; `optional` must make that a no-op, not an error."""
    conf = tmp_path / "conf"
    shutil.copytree(os.path.join(ROOT, "conf"), conf)
    shutil.rmtree(conf / "experiment", ignore_errors=True)

    with initialize_config_dir(config_dir=str(conf), version_base="1.3"):
        cfg = compose("config")
    assert cfg.env.world_size == 6.0
    assert cfg.shaping.type == "dijkstra"
