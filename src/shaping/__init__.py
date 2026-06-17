"""Factory for potential functions. Single config field selects implementation."""
from __future__ import annotations

from omegaconf import DictConfig

from .base import BasePotential
from .no_potential import NoPotential
from .euclidean import EuclideanPotential
from .dubins_potential import DubinsPotential

__all__ = ["BasePotential", "NoPotential", "EuclideanPotential", "DubinsPotential", "build_potential"]


def build_potential(cfg: DictConfig, v_max: float) -> BasePotential:
    """cfg.shaping.type in {'none', 'euclidean', 'dubins'}."""
    t = cfg.shaping.type
    if t == "none":
        return NoPotential()
    if t == "euclidean":
        return EuclideanPotential()
    if t == "dubins":
        return DubinsPotential(
            min_turning_radius=cfg.shaping.min_turning_radius,
            v_max=v_max,
        )
    raise ValueError(f"Unknown shaping type: {t!r}. Choose 'none', 'euclidean', or 'dubins'.")
