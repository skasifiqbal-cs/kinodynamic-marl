"""Approach factory: select how the navigation problem is solved.

``cfg.approach.type`` picks the paradigm — the same ``build_*(cfg) -> Base``
idiom used across the codebase (see ``src/shaping/__init__.py``,
``src/robot/__init__.py``). Add a new paradigm by importing its class here and
adding one branch below, plus a ``conf/approach/<type>.yaml``.
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.approach.base import BaseApproach, Controller

__all__ = ["BaseApproach", "Controller", "build_approach"]


def build_approach(cfg: DictConfig) -> BaseApproach:
    """``cfg.approach.type`` in {'reinforcement_learning', 'planning'}."""
    t = cfg.approach.type
    if t == "reinforcement_learning":
        from src.approach.rl.approach import RLApproach
        return RLApproach(cfg)
    if t == "planning":
        from src.approach.planning.approach import PlanningApproach
        return PlanningApproach(cfg)
    raise ValueError(
        f"Unknown approach type: {t!r}. "
        "Choose 'reinforcement_learning' or 'planning'."
    )
