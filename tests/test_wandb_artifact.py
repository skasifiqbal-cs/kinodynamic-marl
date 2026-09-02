"""Publishing a run to W&B so someone else can render it."""
from __future__ import annotations

import pytest

wandb = pytest.importorskip("wandb")

from src.approach.rl.train import upload_run_artifact  # noqa: E402


class _FakeArtifact:
    def __init__(self, name, type):
        self.name, self.type, self.files = name, type, {}

    def add_file(self, local_path, name):
        self.files[name] = local_path


@pytest.fixture
def fake_wandb(monkeypatch):
    logged = []
    monkeypatch.setattr(wandb, "Artifact", _FakeArtifact)
    monkeypatch.setattr(wandb, "log_artifact", logged.append)
    monkeypatch.setattr(wandb, "run", object())        # a live run
    return logged


def _run_dir(tmp_path, with_ckpt=True):
    d = tmp_path / "braking_mlp_full_state" / "2026-09-03_10-00"
    (d / "checkpoints").mkdir(parents=True)
    (d / "config.yaml").write_text("env: {name: gap_2agent}\n")
    if with_ckpt:
        (d / "checkpoints" / "best_agent.pt").write_bytes(b"weights")
    return d


def test_artifact_preserves_the_layout_evaluate_expects(tmp_path, fake_wandb):
    """evaluate.py finds a run's config at <ckpt>/../../config.yaml. If the artifact
    flattened the files, the download would render with hydra's DEFAULT env instead of
    the one that trained the policy — silently, and wrongly."""
    upload_run_artifact(_run_dir(tmp_path))

    (art,) = fake_wandb
    assert set(art.files) == {"checkpoints/best_agent.pt", "config.yaml"}
    assert art.type == "model"
    # "/" is illegal in an artifact name and both path parts contain one.
    assert art.name == "braking_mlp_full_state_2026-09-03_10-00"
    assert "/" not in art.name


def test_no_checkpoint_is_reported_not_published(tmp_path, fake_wandb, capsys):
    """A run too short to write best_agent.pt must not publish a config-only artifact
    that looks renderable and is not."""
    upload_run_artifact(_run_dir(tmp_path, with_ckpt=False))

    assert fake_wandb == []
    assert "no best_agent.pt" in capsys.readouterr().out


def test_nothing_published_when_the_run_died(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    upload_run_artifact(_run_dir(tmp_path))      # must not raise
