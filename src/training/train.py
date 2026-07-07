"""Back-compat shim.

RL training moved to ``src.approach.rl.train`` and env construction to
``src.env.factory``. This module re-exports both so legacy imports
(``from src.training.train import run_training`` / ``_build_env``) keep working.
Prefer the new locations in new code.
"""
from __future__ import annotations

from src.approach.rl.train import run_training
from src.env.factory import _build_env, build_env

__all__ = ["run_training", "build_env", "_build_env"]
