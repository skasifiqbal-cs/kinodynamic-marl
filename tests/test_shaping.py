"""Shaping potentials: Euclidean, PBRS telescoping, obstacle-aware Dijkstra."""
import numpy as np
import pytest
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


def test_shaping_gamma_differs_from_learner_gamma_and_idle_is_free():
    """Pins the gamma convention the report (paper/implementation_report.tex S6.2) analyses.

    Shaping uses gamma_shape=1.0 while IPPO discounts at gamma_learn=0.99, so the
    Ng-Harada-Russell invariance theorem does NOT cover this configuration. The choice is
    deliberate: at gamma_shape=gamma_learn an idle agent would earn
    k*(1-gamma)*|phi| ~ 8.0*0.01*8.0 = +0.64/step against a -0.01 step penalty.

    If either constant moves, the report's numbers are stale -- fix the report, not this
    test.
    """
    from src.shaping.braking_potential import bangbang_time

    g_shape = float(OmegaConf.load("conf/shaping/braking.yaml").gamma)
    g_learn = float(OmegaConf.load("conf/train/ppo_default.yaml").discount)
    k = float(OmegaConf.load("conf/env/swap2_unicycle2.yaml").reward.shaping_scale)
    step_penalty = float(OmegaConf.load("conf/env/swap2_unicycle2.yaml").reward.step_penalty)

    assert g_shape == 1.0 and g_learn == 0.99, "report S6.2 assumes 1.0 vs 0.99"

    # gamma_shape=1 makes standing still worth exactly nothing.
    phi = -bangbang_time(3.0, 0.0, v_max=0.5, a_max=0.25)   # swap2 start, at rest
    assert k * (g_shape * phi - phi) == 0.0

    # ...whereas the theorem-correct gamma would pay a large multiple of the step
    # penalty for doing nothing. Pin the physics constant and the qualitative claim,
    # not the multiple itself -- the latter tracks reward.shaping_scale, which is tuned.
    assert abs(phi) == pytest.approx(8.0, abs=1e-9)   # bang-bang time over swap2's 3 m
    idle = k * (g_learn * phi - phi)
    assert idle > 0.0
    assert idle / abs(step_penalty) > 10.0, (
        "matched-gamma idling must dwarf the step penalty -- the reason gamma_shape=1"
    )
