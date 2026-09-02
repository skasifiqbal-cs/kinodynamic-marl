"""Robot dynamics: second-order integration, saturation, state lifting."""
import numpy as np
import pytest

from src.collision.shapes import CircleShape
from src.robot.unicycle import Unicycle2Model, UnicycleModel


def make_dyn(**kw):
    p = dict(v_max=1.0, v_min=0.0, omega_max=1.5, omega_min=-1.5,
             a_max=2.0, alpha_max=5.0, shape=CircleShape(0.13))
    p.update(kw)
    return Unicycle2Model(**p)


def test_dyn_dims():
    r = make_dyn()
    assert r.state_dim == 5
    assert r.action_dim == 2
    assert r.n_dynamic_features == 2
    assert np.allclose(r.action_low, [-2.0, -5.0])   # symmetric default: a_min = -a_max
    assert np.allclose(r.action_high, [2.0, 5.0])


def test_asymmetric_accel_bounds():
    r = make_dyn(a_min=-4.0, alpha_min=-3.0)         # brake harder than throttle
    assert np.allclose(r.action_low, [-4.0, -3.0])
    assert np.allclose(r.action_high, [2.0, 5.0])
    # a strong brake command is honored down to a_min, not clipped at -a_max
    s = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    s2 = r.step(s, np.array([-4.0, 0.0]), 0.05)
    assert s2[3] == pytest.approx(1.0 - 4.0 * 0.05, abs=1e-6)


def test_accel_raises_velocity_and_coast_holds():
    r = make_dyn()
    s0 = r.reset_state(np.array([1.0, 1.0, 0.0]))
    s1 = r.step(s0, np.array([r.a_max, 0.0]), 0.05)
    assert s1[3] > s0[3]                      # accel increases v
    s2 = r.step(s0, np.array([0.0, 0.0]), 0.05)
    assert s2[3] == pytest.approx(0.0)        # coasting from rest stays at rest


def test_velocity_saturates_to_v_max():
    r = make_dyn()
    s = r.reset_state(np.array([0.0, 0.0, 0.0]))
    for _ in range(200):                      # full throttle for a long time
        s = r.step(s, np.array([r.a_max, 0.0]), 0.05)
    assert s[3] <= r.v_max + 1e-9
    assert s[3] == pytest.approx(r.v_max, abs=1e-6)


def test_reset_state_three_and_five_element():
    r = make_dyn()
    a = r.reset_state(np.array([2.0, 3.0, 0.5]))          # 3-element -> v,w = 0
    assert a.shape == (5,) and a[3] == 0.0 and a[4] == 0.0
    b = r.reset_state(np.array([2.0, 3.0, 0.5, 0.4, -0.2]))  # 5-element -> used
    assert b[3] == pytest.approx(0.4) and b[4] == pytest.approx(-0.2)
    c = r.reset_state(np.array([0, 0, 0, 99.0, 99.0]))       # clipped to limits
    assert c[3] == pytest.approx(r.v_max) and c[4] == pytest.approx(r.omega_max)


def test_theta_wrapped():
    r = make_dyn()
    s = np.array([0.0, 0.0, 3.1, 1.0, 1.5])
    for _ in range(50):
        s = r.step(s, np.array([0.0, r.alpha_max]), 0.05)
        assert -np.pi - 1e-9 <= s[2] <= np.pi + 1e-9


def test_kinematic_unicycle_basic():
    r = UnicycleModel(v_max=1.0, v_min=0.0, omega_max=1.0, omega_min=-1.0, shape=CircleShape(0.13))
    assert r.state_dim == 3
    s = r.step(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0]), 0.1)
    assert s[0] > 0.0                          # drives forward along +x


def test_load_robot_cfg_does_not_depend_on_the_launch_directory(tmp_path, monkeypatch):
    """Robots are referenced by name from an env config, so they load outside hydra's
    composition. Resolving only against the cwd made every entry point silently
    dependent on where it was invoked from — scripts/fasteval.py run by absolute path
    from another directory could not find conf/robot/ at all."""
    from src.robot import load_robot_cfg

    monkeypatch.chdir(tmp_path)
    cfg = load_robot_cfg("unicycle_v2")
    assert cfg.shape.radius == 0.13


def test_load_robot_cfg_prefers_a_local_override(tmp_path, monkeypatch):
    """A conf/robot/ next to where you launched shadows the repo's, which is how you
    try a modified robot without editing the shipped one."""
    from omegaconf import OmegaConf

    from src.robot import load_robot_cfg

    d = tmp_path / "conf" / "robot"
    d.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"type": "unicycle2", "shape": {"radius": 0.99}}),
                   d / "unicycle_v2.yaml")
    monkeypatch.chdir(tmp_path)
    assert load_robot_cfg("unicycle_v2").shape.radius == 0.99
