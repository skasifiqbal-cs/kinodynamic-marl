"""Reinforcement-learning approach: train an IPPO policy, evaluate the checkpoint."""
from __future__ import annotations

import torch
from omegaconf import DictConfig

from src.approach.base import BaseApproach
from src.approach.rl.controller import RLController
from src.approach.rl.train import run_training


class RLApproach(BaseApproach):
    """``approach=reinforcement_learning`` — the existing skrl IPPO pipeline."""

    def run(self, cfg: DictConfig) -> None:
        run_training(cfg)

    def build_controller(self, env, deterministic: bool = True) -> RLController:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = cfg_get(self.cfg, "eval", "checkpoint")
        return RLController(self.cfg, env, checkpoint, device, deterministic=deterministic)


def cfg_get(cfg, group: str, key: str):
    g = cfg.get(group, None)
    return None if g is None else g.get(key, None)
