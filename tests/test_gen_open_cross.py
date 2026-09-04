"""The Open Cross generator's geometry is only trustworthy if its own checks run in CI.

`check()` is what the generator asserts before writing anything: N=2 must come out equal
to the hand-ported swap2_unicycle2.yaml, and the spacing/margin/centring invariants that
N=2 cannot exercise must hold at every size. Running it here means a later edit to swap2
or to unicycle_db's shape fails the suite instead of silently desynchronising the
generated configs from the file they claim to extend.
"""
import yaml

from scripts.gen_open_cross import SIZES, check, geometry, render


def test_generator_self_check():
    check()


def test_every_size_renders_valid_yaml_with_n_agents():
    for n in SIZES:
        cfg = yaml.safe_load(render(n))
        assert len(cfg["agents"]) == n
        assert len({a["id"] for a in cfg["agents"]}) == n
        assert cfg["obstacles"] == []
        # Head-on: each row's pair swaps endpoints, so i's goal is its partner's start.
        for left, right in zip(cfg["agents"][::2], cfg["agents"][1::2]):
            assert left["goal"][:2] == right["start"][:2]
            assert right["goal"][:2] == left["start"][:2]
        # Every robot inside the world, with its body clear of the walls.
        world = cfg["world_size"]
        for a in cfg["agents"]:
            for field in ("start", "goal"):
                assert all(0.3 < c < world - 0.3 for c in a[field][:2]), (n, a["id"], field)


def test_world_grows_but_traverse_does_not():
    """Runtime against N has to measure congestion, not distance."""
    worlds = [geometry(n)[0] for n in SIZES]
    assert worlds == sorted(worlds) and worlds[-1] > worlds[0]
    assert {round(geometry(n)[2] - geometry(n)[1], 9) for n in SIZES} == {3.0}


def test_odd_or_undersized_n_is_rejected():
    for bad in (0, 1, 3, 7):
        try:
            geometry(bad)
        except ValueError:
            continue
        raise AssertionError(f"geometry({bad}) should have been rejected")
