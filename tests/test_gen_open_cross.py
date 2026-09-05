"""The Open Cross generator's geometry is only trustworthy if its own checks run in CI.

`check()` is what the generator asserts before writing anything: N=2 must come out equal
to the hand-ported swap2_unicycle2.yaml, and the spacing/margin/centring invariants that
N=2 cannot exercise must hold at every size. Running it here means a later edit to swap2
or to unicycle_db's shape fails the suite instead of silently desynchronising the
generated configs from the file they claim to extend.
"""
import yaml

from scripts.gen_open_cross import (
    SIZES,
    USABLE,
    body_diameter,
    check,
    geometry,
    render,
    row_spacing,
)


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


def test_only_the_robot_count_varies_with_n():
    """Runtime against N has to measure congestion, so nothing else may move with N.

    World and traverse are both fixed; N buys crowding by packing the rows closer into
    the same height. A per-N world would instead rescale full_state.py's wall features,
    which divide by world_size, confounding any curve plotted against N.
    """
    assert len({geometry(n)[0] for n in SIZES}) == 1
    assert {round(geometry(n)[2] - geometry(n)[1], 9) for n in SIZES} == {USABLE}
    area = [geometry(n)[0] ** 2 / n for n in SIZES]
    assert area == sorted(area, reverse=True), dict(zip(SIZES, area))


def test_rows_pack_closer_as_n_grows_but_never_overlap():
    """The congestion mechanism. Spacing must fall with N and stay above the body size."""
    gaps = [row_spacing(n // 2) for n in SIZES if n > 2]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert min(gaps) > body_diameter()
    # Every robot's body clears the walls at the extreme rows too.
    for n in SIZES:
        world, _, _, ys = geometry(n)
        assert ys[0] > body_diameter() / 2 and ys[-1] < world - body_diameter() / 2


def test_n_that_does_not_fit_the_world_is_rejected():
    """Rows must not be packed below the body diagonal to fit — raise instead."""
    try:
        geometry(64)
    except ValueError as e:
        assert "body diagonal" in str(e) and "overlap" in str(e), e
    else:
        raise AssertionError("geometry(64) packs rows below the body diagonal; must raise")


def test_odd_or_undersized_n_is_rejected():
    for bad in (0, 1, 3, 7):
        try:
            geometry(bad)
        except ValueError:
            continue
        raise AssertionError(f"geometry({bad}) should have been rejected")
