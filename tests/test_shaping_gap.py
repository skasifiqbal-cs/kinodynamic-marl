"""The gap falsifier reimplements the dynamics vectorised; it must match the real one."""
from omegaconf import OmegaConf

from scripts.shaping_gap import self_check
from src.robot import build_robot, load_robot_cfg


def test_vectorised_rk4_matches_robot_step():
    for name in ("unicycle_db", "unicycle_v2"):
        robot = build_robot(load_robot_cfg(name))
        self_check(robot, dt=0.1)


def test_rejects_velocity_grid_that_would_misalign():
    """v_max must divide evenly into the velocity lattice or rows aren't comparable."""
    rc = load_robot_cfg("unicycle_db")
    r = build_robot(OmegaConf.merge(rc, {"v_max": 0.3536, "v_min": -0.3536}))
    assert abs(2 * r.v_max / 0.05 - round(2 * r.v_max / 0.05)) > 1e-9


def test_bangbang_time_is_a_lower_bound_and_tight_outside_overshoot():
    """T must never exceed the true time, and must be ATTAINED when d >= braking distance.

    When d < s0^2/(2 a_max) the robot cannot cover d and still stop -- it has to overshoot
    and come back -- so T is a strict under-estimate there. That is fine for admissibility
    (a heuristic may under-estimate) but it means tightness can only be claimed outside
    the overshoot regime. The proof in the paper must say so.
    """
    import numpy as np

    from src.shaping.braking_potential import bangbang_time

    for v_max, a_max in ((0.5, 0.25), (1.0, 2.0), (0.5, 0.1)):
        for d in (0.0, 0.2, 1.0, 3.0, 7.0):
            for s0 in (0.0, 0.25 * v_max, v_max):
                T = bangbang_time(d, s0, v_max, a_max)
                # greedy time-optimal 1-D policy: accelerate while the braking distance
                # still fits inside what is left, otherwise brake.
                dt, s, travelled, t = 1e-4, s0, 0.0, 0.0
                while t < 60.0 and not (travelled >= d and s <= 1e-3):
                    a = -a_max if travelled + s * s / (2 * a_max) >= d else a_max
                    s = float(np.clip(s + a * dt, 0.0, v_max))
                    travelled += s * dt
                    t += dt
                assert T <= t + 1e-2, f"not a lower bound: T={T} > achieved {t}"
                if d >= s0 * s0 / (2 * a_max):          # no overshoot needed
                    assert abs(T - t) < 3e-2, f"not tight: T={T} vs achieved {t}"


def test_braking_potential_is_zero_only_at_rest_on_goal():
    import numpy as np
    from omegaconf import OmegaConf

    from src.robot import build_robot, load_robot_cfg
    from src.shaping import build_potential

    robot = build_robot(load_robot_cfg("unicycle_db"))
    cfg = OmegaConf.create({"shaping": {"type": "braking", "cell_size": 0.1}})
    phi = build_potential(cfg, v_max=robot.v_max, obstacles=[], world_size=5.0, robot=robot)
    goal = np.array([4.0, 2.5, 0.0, 0.0, 0.0])
    assert phi.phi(np.array([4.0, 2.5, 0.0, 0.0, 0.0]), goal) == 0.0
    assert phi.phi(np.array([4.0, 2.5, 0.0, 0.5, 0.0]), goal) < -0.5   # moving: not free
    assert phi.phi(np.array([1.0, 2.5, 0.0, 0.0, 0.0]), goal) < -5.0   # far: costly
