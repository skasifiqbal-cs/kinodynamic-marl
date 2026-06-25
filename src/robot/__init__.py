"""Robot factory: build_robot(cfg) and load_robot_cfg(name)."""
from __future__ import annotations

import os

from omegaconf import DictConfig, OmegaConf

from src.collision.shapes import build_shape
from src.robot.base import BaseRobot
from src.robot.car import KinematicCarModel
from src.robot.unicycle import Unicycle2Model, UnicycleModel

__all__ = ["BaseRobot", "UnicycleModel", "Unicycle2Model", "KinematicCarModel",
           "build_robot", "load_robot_cfg"]


def load_robot_cfg(robot_name: str) -> DictConfig:
    """Load conf/robot/<robot_name>.yaml relative to the project root."""
    try:
        from hydra.utils import get_original_cwd
        cwd = get_original_cwd()
    except Exception:
        cwd = os.getcwd()
    path = os.path.join(cwd, "conf", "robot", f"{robot_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Robot config not found: {path}\n"
            f"Create conf/robot/{robot_name}.yaml with fields: type, v_max, ..."
        )
    return OmegaConf.load(path)


def build_robot(cfg: DictConfig) -> BaseRobot:
    """Instantiate robot from a loaded robot config DictConfig."""
    shape = build_shape(cfg.shape)
    t = cfg.type

    if t == "unicycle":
        return UnicycleModel(
            v_max=float(cfg.v_max),
            v_min=float(cfg.get("v_min", -cfg.v_max)),
            omega_max=float(cfg.omega_max),
            omega_min=float(cfg.get("omega_min", -cfg.omega_max)),
            shape=shape,
        )
    if t == "unicycle2":
        a_min = cfg.get("a_min", None)
        alpha_min = cfg.get("alpha_min", None)
        return Unicycle2Model(
            v_max=float(cfg.v_max),
            v_min=float(cfg.get("v_min", 0.0)),
            omega_max=float(cfg.omega_max),
            omega_min=float(cfg.get("omega_min", -cfg.omega_max)),
            a_max=float(cfg.a_max),
            alpha_max=float(cfg.alpha_max),
            a_min=None if a_min is None else float(a_min),
            alpha_min=None if alpha_min is None else float(alpha_min),
            shape=shape,
        )
    if t == "car":
        return KinematicCarModel(
            v_max=float(cfg.v_max),
            delta_max=float(cfg.delta_max),
            wheelbase=float(cfg.wheelbase),
            shape=shape,
        )
    raise ValueError(f"Unknown robot type: {t!r}. Add it to src/robot/__init__.py.")
