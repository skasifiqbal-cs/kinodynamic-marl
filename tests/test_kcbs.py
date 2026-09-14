"""The K-CBS bridge must hand the C++ planner the problem the env scores, not a looser one."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.approach.planning.kcbs import problem_yaml
from src.collision.shapes import BoxShape, CircleShape, Obstacle


class Unicycle2Model:          # name-matched to the model the bridge maps
    shape = BoxShape(0.5, 0.25)


def _env(obstacles):
    return SimpleNamespace(robots=[Unicycle2Model()], _obstacles=obstacles, _world_size=17.0,
                           _states=[np.array([1.0, 1.0, 0.0, 0.0, 0.0])],
                           _goals=[np.array([16.0, 1.0, 0.0, 0.0, 0.0])])


def test_walls_are_exported_as_boxes_so_the_body_cannot_leave_the_world():
    boxes = problem_yaml(_env([]))["environment"]["obstacles"]
    assert len(boxes) == 4
    # Each wall's inner face lies exactly on the world boundary.
    faces = sorted(round(b["center"][k] + s * b["size"][k] / 2, 9)
                   for b in boxes for k in (0, 1) if b["size"][k] == 1.0
                   for s in ((1,) if b["center"][k] < 0 else (-1,)))
    assert faces == [0.0, 0.0, 17.0, 17.0]


def test_what_the_driver_cannot_express_is_refused_not_approximated():
    with pytest.raises(ValueError):
        problem_yaml(_env([Obstacle(5.0, 5.0, CircleShape(1.0))]))
    with pytest.raises(ValueError):
        problem_yaml(_env([Obstacle(5.0, 5.0, BoxShape(1.0, 2.0), angle=0.3)]))
