"""Planning method factory.

``cfg.approach.method`` selects the planner. Same ``build_*`` idiom as the rest
of the codebase: import the class, add one branch, and add a param block in
``conf/approach/planning.yaml``. See ``docs/INTERN.md``.
"""
from __future__ import annotations

from src.planning.base import BasePlanner
from src.planning.cegar import CEGARPlanner
from src.planning.constructive import ConstructivePlanner
from src.planning.karc import KARCPlanner
from src.planning.kcbs import KCBSPlanner
from src.planning.optimization import OptimizationPlanner
from src.planning.splinecegar import SplineCEGARPlanner

__all__ = [
    "BasePlanner", "OptimizationPlanner",
    "KARCPlanner", "KCBSPlanner", "ConstructivePlanner", "CEGARPlanner", "SplineCEGARPlanner",
    "build_planner",
]

_PLANNERS = {
    "optimization": OptimizationPlanner,
    "karc": KARCPlanner,
    "kcbs": KCBSPlanner,
    "constructive": ConstructivePlanner,
    "cegar": CEGARPlanner,
    "splinecegar": SplineCEGARPlanner,
}


def build_planner(approach_cfg) -> BasePlanner:
    """``approach_cfg.method`` in {'optimization', 'karc', 'kcbs', 'constructive',
    'cegar', 'splinecegar'}.

    ``karc`` is the faithful reimplementation of arXiv:2501.01559 and is the baseline;
    ``constructive`` and ``cegar`` are ours -- the first constructs the coordination from a
    rulebook and schedules it exactly, the second samples candidates and lets unsat cores
    drive the resampling. All three are separate methods with separate config blocks so
    that none can be quietly turned into another by a flag.
    """
    method = approach_cfg.method
    cls = _PLANNERS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown planning method: {method!r}. "
            f"Choose one of {sorted(_PLANNERS)}."
        )
    # Method-specific params live in a same-named block, e.g. cfg.approach.rrt.
    params = approach_cfg.get(method, {})
    return cls(approach_cfg, params)
