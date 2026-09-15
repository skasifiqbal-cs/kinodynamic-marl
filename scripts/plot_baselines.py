"""K-ARC-style comparison figures: success, runtime and makespan against team size.

Reads the per-run logs `scripts/karc_bench.py` writes (one RESULT/STATS pair per cell), so a
sweep still running is plotted from whatever has finished. Success is PLAN-level for every
method: our planner and K-CBS returned a plan (`solved`), K-ARC returned a plan that passes its
own validity check (`plan_valid`; its optimised timestep is off the simulator grid, so it is not
executed). Runtime and makespan are means over the successful runs only.

    python scripts/plot_baselines.py            # writes experiments/baselines_<model>.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FAMILIES = ["open_cross", "cluttered_cross", "circular_cross"]
SIZES = [4, 8, 16, 32]
# method -> (label, log dirs in priority order (later wins), success key)
METHODS = {
    "cegar": ("Ours", ["trajopt_sweep_{m}", "trajopt_sweep_{m}_rerun"], "solved"),
    "karc": ("K-ARC", ["karcfix600_{m}"], "plan_valid"),
    "kcbs": ("K-CBS", ["kcbs600"], "solved"),
}


def _stats(log: Path, method: str) -> dict | None:
    m = re.search(rf"^STATS,{method},(.*)$", log.read_text(), re.M)
    if not m:
        return None
    return dict(f.partition("=")[::2] for f in re.split(r",(?=[a-z_]+=)", m[1]))


def collect(model: str) -> dict:
    """(method, family, n) -> list of per-seed stats dicts."""
    out = {}
    for method, (_, dirs, _) in METHODS.items():
        for d in dirs:
            for log in (EXP / d.format(m=model)).glob(f"*_{model}.{method}.s*.log"):
                st = _stats(log, method)
                if st is None:
                    continue
                fam, n = re.match(rf"(.+)_(\d+)_{model}\.", log.name).groups()
                seed = log.name.rsplit(".s", 1)[1].split(".")[0]
                out.setdefault((method, fam, int(n)), {})[seed] = st
    return {k: list(v.values()) for k, v in out.items()}


def summarise(model: str):
    data, rows = collect(model), []
    for fam in FAMILIES:
        for n in SIZES:
            for method, (label, _, key) in METHODS.items():
                runs = data.get((method, fam, n), [])
                ok = [r for r in runs if float(r.get(key, 0) or 0) >= 1]
                mean = lambda f: float(np.mean([float(r[f]) for r in ok])) if ok else np.nan  # noqa: E731
                rows.append((fam, n, method, len(runs), len(ok), mean("wall_time"), mean("makespan")))
    return rows


def plot(model: str, rows) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
    for c, fam in enumerate(FAMILIES):
        for (method, (label, _, _)), (mk, ls, ms) in zip(
                METHODS.items(), [("o", "-", 9), ("s", "--", 6), ("^", ":", 6)]):
            st = dict(marker=mk, linestyle=ls, markersize=ms, label=label)
            r = [x for x in rows if x[0] == fam and x[2] == method]
            x = [v[1] for v in r]
            succ = [100 * v[4] / v[3] if v[3] else np.nan for v in r]
            axes[0, c].plot(x, succ, **st)
            axes[1, c].plot(x, [v[5] for v in r], **st)
            axes[2, c].plot(x, [v[6] for v in r], **st)
        axes[0, c].set_title(fam.replace("_", " "))
        axes[1, c].set_yscale("log")
        axes[2, c].set_xlabel("robots")
        axes[2, c].set_xscale("log", base=2)
        axes[2, c].set_xticks(SIZES, [str(s) for s in SIZES])
    for r, name in enumerate(["success (%)", "runtime (s)", "makespan (s)"]):
        axes[r, 0].set_ylabel(name)
    axes[0, 0].legend()
    fig.tight_layout()
    path = EXP / f"baselines_{model}.png"
    fig.savefig(path, dpi=200)
    return path


if __name__ == "__main__":
    for model in sys.argv[1:] or ["unicycle2", "unicycle1"]:
        rows = summarise(model)
        print(f"\n{model}\n| scenario | N | method | ok/runs | runtime s | makespan s |\n|---|---|---|---|---|---|")
        for fam, n, method, runs, ok, t, ms in rows:
            if runs:
                print(f"| {fam} | {n} | {method} | {ok}/{runs} | {t:.1f} | {ms:.1f} |")
        print("wrote", plot(model, rows))
