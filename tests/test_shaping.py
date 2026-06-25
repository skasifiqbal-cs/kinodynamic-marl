"""Shaping potentials: Euclidean, PBRS telescoping, obstacle-aware Dijkstra."""
import numpy as np
from omegaconf import OmegaConf

from src.shaping.dijkstra_potential import DijkstraPotential
from src.shaping.euclidean import EuclideanPotential


def test_euclidean_phi_increases_toward_goal():
    phi = EuclideanPotential()
    g = np.array([5.0, 5.0, 0.0])
    far = np.array([0.0, 0.0, 0.0])
    near = np.array([4.0, 4.0, 0.0])
    assert phi.phi(near, g) > phi.phi(far, g)   # closer -> less negative
    assert phi.phi(g, g) == 0.0


def test_pbrs_telescopes_to_endpoint_difference():
    """With gamma=1, sum_t F_t = phi(s_T) - phi(s_0), path-independent (Ng 1999)."""
    phi = EuclideanPotential()
    g = np.array([5.0, 0.0, 0.0])
    path = [np.array([float(x), float(y), 0.0]) for x, y in
            [(0, 0), (1, 1), (2, -1), (3, 2), (4, 0), (5, 0)]]
    total = sum(phi.shaping(path[i], path[i + 1], g, gamma=1.0) for i in range(len(path) - 1))
    assert total == np.float64(phi.phi(path[-1], g) - phi.phi(path[0], g))


def _dijkstra_with_wall():
    obstacles = OmegaConf.create([
        {"x": 3.0, "y": 3.0, "angle": 0.0, "shape": {"type": "box", "width": 0.3, "length": 2.0}},
    ])
    return DijkstraPotential(obstacles_cfg=obstacles, world_size=6.0, v_max=1.0,
                             clearance=0.18, cell_size=0.1)


def test_dijkstra_phi_finite_and_improves_toward_goal():
    pot = _dijkstra_with_wall()
    g = np.array([5.0, 3.0, 0.0])
    start = np.array([1.0, 3.0, 0.0])
    near = np.array([4.5, 3.0, 0.0])
    assert np.isfinite(pot.phi(start, g))
    assert pot.phi(near, g) > pot.phi(start, g)   # closer along the routed path


def test_dijkstra_routes_around_obstacle():
    """A point just before the wall must have a LONGER cost-to-go than the straight-line
    distance would suggest — the path is forced around, not through."""
    pot = _dijkstra_with_wall()
    g = np.array([5.0, 3.0, 0.0])
    before_wall = np.array([2.7, 3.0, 0.0])
    euclid = -np.linalg.norm(before_wall[:2] - g[:2])   # / v_max=1
    assert pot.phi(before_wall, g) < euclid             # routed cost > straight (more negative)
