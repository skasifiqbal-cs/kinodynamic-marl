"""Geometric RRT planner — STUB for the intern to implement.

Plan a collision-free geometric path (x, y) from start to goal via RRT, then
follow it with a simple controller that maps waypoints to robot controls.

TODO (see BasePlanner docstring + docs/INTERN.md):
  reset(env):
    - sample random points in [-world, world]^2; reject those in collision
      (src.collision.shapes.collides against env._obstacles + robot.shape,
       collides_wall against env._world_size)
    - grow a tree toward samples with step_size; connect when within goal_radius
      of env._goals[i][:2]; goal-bias with params['goal_sample_rate']
    - store the path per agent on self
  act(obs_dict, env):
    - steer the robot toward the next waypoint -> return {agent: control}
      (control is [v, ω] or [a, α] depending on env.robots[i]; clip to bounds)
Reuse note: DijkstraPotential._free (src/shaping/dijkstra_potential.py) is a
ready clearance-inflated occupancy grid you can sample/validate against.
"""
from __future__ import annotations

from src.approach.planning.base import BasePlanner


class RRTPlanner(BasePlanner):
    method = "rrt"

    def reset(self, env) -> None:
        raise self._todo("RRT path planning")

    def act(self, obs_dict: dict, env) -> dict:
        raise self._todo("RRT path following")
