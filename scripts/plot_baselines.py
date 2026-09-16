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
    """(scenario, n, method, runs, ok, runtime mean/std, makespan mean/std) per cell."""
    data, rows = collect(model), []
    for fam in FAMILIES:
        for n in SIZES:
            for method, (label, _, key) in METHODS.items():
                runs = data.get((method, fam, n), [])
                ok = [r for r in runs if float(r.get(key, 0) or 0) >= 1]
                # Spread is over the runs that produced a plan: a failed run has no time to
                # average (it hit the same 600 s cap) and no makespan at all.
                stat = lambda f: (float(np.mean(v)), float(np.std(v))) if (  # noqa: E731
                    v := [float(r[f]) for r in ok if r.get(f)]) else (np.nan, np.nan)
                rows.append((fam, n, method, len(runs), len(ok),
                             *stat("wall_time"), *stat("makespan")))
    return rows


def table(models) -> Path:
    """The whole results float, one group of rows per robot model.

    Emitted as a complete `table*` rather than a body to \\input inside a tabular: LaTeX's
    \\input leaves the alignment mid-cell, so the \\midrule after it errors out.
    """
    head = [r"\begin{table*}[t]",
            r"\caption{All cells: successes out of five runs, runtime and makespan in seconds as "
            r"mean $\pm$ standard deviation over the successful runs. ``---'': no run succeeded.}",
            r"\label{tab:results}", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}ll ccc ccc ccc@{}}", r"\toprule",
            r"& & \multicolumn{3}{c}{ours} & \multicolumn{3}{c}{K-ARC \cite{karc2025}} & "
            r"\multicolumn{3}{c}{K-CBS \cite{kottinger2022kcbs}} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
            r"scenario & $N$ & succ. & time & span & succ. & time & span & succ. & time & span \\"]
    body = []
    for model, rows in models.items():
        order = {"unicycle2": "second-order", "unicycle1": "first-order"}.get(model, model)
        body += [r"\midrule", r"\multicolumn{11}{@{}l}{\textit{%s unicycles}} \\" % order]
        for fam in FAMILIES:
            for n in SIZES:
                cells = []
                for method in METHODS:
                    _, _, _, runs, ok, t, ts, m, ms = next(
                        x for x in rows if x[:3] == (fam, n, method))
                    cells += [f"{ok}/{runs}" if runs else "--",
                              "--" if np.isnan(t) else f"${t:.0f} \\pm {ts:.0f}$",
                              "--" if np.isnan(m) else f"${m:.0f} \\pm {ms:.0f}$"]
                body.append(f"{fam.split('_')[0]} & {n} & " + " & ".join(cells) + r" \\")
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path = EXP / "baselines_table.tex"
    path.write_text("\n".join(head + body + tail) + "\n")
    return path


def plot(model: str, rows) -> Path:
    """Compact box plots of runtime: one panel per scenario, one box per method and size.

    Five runs per box, so the box is the quartiles and the whiskers the range. Success
    rates are printed under each tick in legend order; makespan stays in the table.
    """
    data = collect(model)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
    colour = dict(zip(METHODS, ["#1f77b4", "#d62728", "#2ca02c"]))
    for c, fam in enumerate(FAMILIES):
        ax = axes[c]
        for off, (method, (label, _, key)) in zip([-0.26, 0.0, 0.26], METHODS.items()):
            for pos, n in enumerate(SIZES):
                v = [float(r["wall_time"]) for r in data.get((method, fam, n), [])
                     if float(r.get(key, 0) or 0) >= 1 and r.get("wall_time")]
                if not v:          # nothing solved: a bare cross, so the gap is visible
                    ax.plot(pos + off, 2.0, marker="x", ms=3, color=colour[method])
                    continue
                ax.boxplot(v, positions=[pos + off], widths=0.22, patch_artist=True,
                           medianprops=dict(color="black", linewidth=0.8),
                           boxprops=dict(facecolor=colour[method], edgecolor=colour[method],
                                         linewidth=0.8),
                           whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                           flierprops=dict(ms=2))
        # Success rate goes under the tick, in legend order, so the figure says how many
        # runs each box stands for without a second axis.
        ticks = []
        for n in SIZES:
            pct = []
            for method, (_, _, key) in METHODS.items():
                runs = data.get((method, fam, n), [])
                ok = sum(float(r.get(key, 0) or 0) >= 1 for r in runs)
                pct.append(f"{ok}" if runs else "-")
            ticks.append(f"{n}\n({'/'.join(pct)})")
        ax.set_title(fam.replace("_", " "), fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(SIZES)), ticks, fontsize=6)
        ax.set_xlim(-0.6, len(SIZES) - 0.4)
        ax.set_xlabel("robots", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("runtime [s]", fontsize=8)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=colour[m], label=METHODS[m][0])
                        for m in METHODS], ncol=3, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = EXP / f"baselines_{model}.png"
    fig.savefig(path, dpi=300)
    return path


if __name__ == "__main__":
    tables = {}
    for model in sys.argv[1:] or ["unicycle2", "unicycle1"]:
        rows = tables[model] = summarise(model)
        print(f"\n{model}\n| scenario | N | method | ok/runs | runtime s | makespan s |\n|---|---|---|---|---|---|")
        for fam, n, method, runs, ok, t, ts, m, ms in rows:
            if runs:
                print(f"| {fam} | {n} | {method} | {ok}/{runs} | {t:.1f} +- {ts:.1f} "
                      f"| {m:.1f} +- {ms:.1f} |")
        print("wrote", plot(model, rows))
    print("wrote", table(tables))
