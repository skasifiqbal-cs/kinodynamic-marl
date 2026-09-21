"""Geometric RRT for K-ARC's initial paths (Alg. 1 line 3): free, connected, no dynamics."""
import numpy as np

from src.core.collision.shapes import CircleShape, Obstacle, collides
from src.planning import geometric_rrt


def _blocked_corridor():
    """A wall across the middle with one gap, so a straight line cannot work."""
    return [Obstacle(x=5.0, y=y, shape=CircleShape(radius=0.4)) for y in
            (0.5, 1.3, 2.1, 2.9, 7.1, 7.9, 8.7, 9.5)]


def test_path_is_collision_free_and_connects_when_a_detour_is_required():
    obs = _blocked_corridor()
    r = 0.25
    path = geometric_rrt.plan_path(np.array([1.0, 5.0]), np.array([9.0, 5.0]), obs, 10.0,
                                   radius=r, rng=np.random.default_rng(0))
    assert path is not None, "the gap at y=5 is 2.7 m wide; a 0.5 m disc fits"
    assert np.allclose(path[0], [1.0, 5.0]) and np.allclose(path[-1], [9.0, 5.0])

    probe = CircleShape(radius=r)
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, 40):
            p = a + t * (b - a)
            pose = (float(p[0]), float(p[1]), 0.0)
            assert not any(collides(probe, pose, o.shape, o.pose) for o in obs), \
                f"path passes through an obstacle near {p}"


def test_reports_failure_rather_than_returning_a_straight_line():
    """A guide through a wall is worse than no guide: it seeds the optimizer inside an
    obstacle, which then reports infeasible for a reason the caller cannot see."""
    wall = [Obstacle(x=5.0, y=y / 2, shape=CircleShape(radius=0.4)) for y in range(0, 42)]
    path = geometric_rrt.plan_path(np.array([1.0, 5.0]), np.array([9.0, 5.0]), wall, 10.0,
                                   radius=0.25, max_iters=800,
                                   rng=np.random.default_rng(0))
    assert path is None
