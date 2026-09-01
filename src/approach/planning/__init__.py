"""Planning method factory.

``cfg.approach.method`` selects the planner. Same ``build_*`` idiom as the rest
of the codebase: import the class, add one branch, and add a param block in
``conf/approach/planning.yaml``. See ``docs/INTERN.md``.
"""
from __future__ import annotations

from src.approach.planning.base import BasePlanner
from src.approach.planning.karc import KARCPlanner
from src.approach.planning.kinodynamic_rrt import KinodynamicRRTPlanner
from src.approach.planning.optimization import OptimizationPlanner
from src.approach.planning.rrt import RRTPlanner

__all__ = [
    "BasePlanner", "RRTPlanner", "KinodynamicRRTPlanner", "OptimizationPlanner",
    "KARCPlanner",
    "build_planner",
]

_PLANNERS = {
    "rrt": RRTPlanner,
    "kinodynamic_rrt": KinodynamicRRTPlanner,
    "optimization": OptimizationPlanner,
    "karc": KARCPlanner,
}


def build_planner(approach_cfg) -> BasePlanner:
    """``approach_cfg.method`` in {'rrt', 'kinodynamic_rrt', 'optimization', 'karc'}."""
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
