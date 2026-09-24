# Extending this repo

Conventions that already hold everywhere in the tree. None of them are style
preferences — each one is here because breaking it has already cost something.

## 1. Respect the layering

```
src/core/       robots, env, obs, shaping, collision, conflict — imports nothing else in src/
src/approach/   BaseApproach, Controller, rollout, build_approach — the shared contract
src/planning/   planners            } never import each other
src/rl/         training + policies }
```

If a change looks like it needs `src/planning/` to import `src/rl/` or the reverse, the
thing it needs belongs in `src/core/`. Move it there instead.

`src/core/` is shared, so a change there moves both sides at once. Say so before making
one.

## 2. New things are registered, never hard-wired

Same shape every time: subclass the base, add one branch to the factory, add a YAML with
the same name.

```python
# src/core/shaping/__init__.py
def build_potential(cfg):
    if cfg.type == "euclidean":  return EuclideanPotential(...)
    if cfg.type == "yours":      return YourPotential(...)     # <- one branch
```

```yaml
# conf/shaping/yours.yaml   <- same name as the string above
type: yours
weight: 1.0
```

Then `python train.py shaping=yours` works with no other change. Do not add a
`--your-thing` flag to an entry point; the config group *is* the interface.

## 3. A new switch defaults to the old behaviour

Any option added to an existing method must default to exactly what the code did before
it existed, so that adding it changes no result. This is what makes an ablation table
trustworthy: every column differs from the baseline in one named key.

`conf/approach/planning.yaml` is full of this — the `karc:` block's defaults are marked
`faithful` one by one (`:90`, `:154`, `:161`, `:190`, `:249`) because that block is the
baseline being compared against, and a default that quietly improved it would flatter us.

Check it: flip nothing, run the scenario, confirm the numbers are unchanged.

## 4. Comments carry the evidence, not the restatement

The repo's comments say *why*, and where a choice was measured they quote the
measurement. From `conf/approach/planning.yaml:100`:

```
# On open_cross_16 the raw guides averaged 1.25x chord over 33 vertices and wandered
# 3.16 m off-row against a 2.14 m row spacing -- so they routed each robot through one to
# two NEIGHBOURING lanes and manufactured the conflicts. The cost was not marginal:
#   raw       1800 s, FAILED, merges=10, subproblem_max=10, conflicts_remaining=9
#   shortcut    52 s, solved, merges=0,  subproblem_max=2,  rungs={'prioritized': 9}
```

Write that, not `# use shortcut guides (faster)`. A number someone can re-measure
survives; an adjective does not. If you tuned a constant, the comment says what you tried
and what it cost.

A comment that restates the line below it (`# increment the counter`) is noise — delete
it.

## 5. One definition per quantity

Two names for one thing is how two files drift into disagreeing. `src/approach/rollout.py`
carries the scar: `success` and `both_reached` once meant almost the same thing, and
`evaluate.py` scored from one while `fasteval.py` scored from the other. They agreed only
because a third setting happened to be off.

So: if two places need the same number, they call the same function. `run_episode`,
`run_episodes`, `summarize` and `save_gif` all exist for exactly this reason — every
evaluator shares them. Adding a fifth evaluator means calling those, not writing a loop.

## 6. Do not depend on where the command was run from

Resolve paths from the file or from an explicit config key, never from the working
directory. `load_robot_cfg` (`src/core/robot/__init__.py`) tries the launch directory
first *by design*, so a local `conf/robot/` can shadow the repo's, and then falls back to
the repo root found by locating the `src` package by name.

`runs/` is written only by `train.py`, under a path it chooses. `experiments/` is where
scripts write results, and takes `--out`. Neither is tracked (see the README).

## 7. Leave one runnable check

Non-trivial logic gets the smallest test that fails when it breaks — one function in
`tests/`, no new fixtures or frameworks. Two kinds earn their keep here:

- **Regression**: defaults unchanged ⇒ output unchanged. Cheap, and it is the only thing
  that catches an "unrelated" refactor altering a result.
- **Property**: the thing that must stay true regardless of inputs — a symmetric pair has
  equal slack; a stopped robot has maximal margin.

A one-line passthrough needs no test.

## 8. Before you push

```bash
ruff check . && python -m pytest
```

Both must pass. CI is **not** running yet — `docs/ci-workflow.yml` is waiting to be
installed as `.github/workflows/ci.yml` — so this is a manual gate, not a safety net.

Bare `pytest` also works now: `pyproject.toml` sets `pythonpath = ["."]`, without which
the console script cannot import `src` and all 17 test modules fail at collection. If you
ever see `ModuleNotFoundError: No module named 'src'`, that setting is missing or your
pytest is older than 7.

One test is a known failure on this branch and is not yours:
`tests/test_planning.py::test_adapt_subproblem_reopens_the_previous_segment_and_rescues_it`.

## 9. Experiments are configured in the file

Edit `conf/config.yaml` to set up a run. Command-line overrides are for sweeps, where
several scenarios run at once and the file can only name one — that is why the README's
examples use them.

Do not add a `local.yaml` escape hatch. Merge conflicts in `conf/config.yaml` are
information: they say two people changed the experiment.

## 10. Commits

Subject line says what changed. Body says why, with the evidence — the measurement, the
failure it prevents, or the thing that was silently wrong. `git log` here reads as the
record of what was learned; keep it that way.

No `Co-Authored-By` trailers.

---

## Your first change, end to end

1. Branch. Do not commit to `main`.
2. Find the factory for the kind of thing you are adding (§2) and read its two nearest
   neighbours — this repo is more consistent than it is documented, so the neighbour is
   the spec.
3. Write it. Add the YAML. Register it.
4. Add the check (§7).
5. `ruff check . && pytest`.
6. Run it end to end and keep the output:
   ```bash
   python evaluate.py approach=planning approach.method=<yours>     # renders a GIF
   python scripts/fasteval.py approach=planning approach.method=<yours> eval.episodes=100
   ```
   `fasteval.py` prints a `RESULT,...` line comparable against every other method on the
   same env. That line, not "it seems to work", is what done looks like.
