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
    """K-ARC's layout, turned upright to fit one column: a row per team size.

    Columns are the scenarios, colour is the method. A method that solved nothing is
    dropped from its panel, as in K-ARC's figures. The success rate of each method is the
    x tick label under its box, in the method's colour, so nothing is written on the axes
    themselves.
    """
    data = collect(model)
    # N = 32 and circular 16 are all-or-nothing cells (only one method solves one of them),
    # so they are reported in prose rather than as panels of empty axes.
    sizes, skip = SIZES[:-1], {("circular_cross", 16)}
    fig, axes = plt.subplots(len(sizes), len(FAMILIES), figsize=(7.1, 3.9), sharey=True)
    colour = dict(zip(METHODS, ["#1f77b4", "#d62728", "#2ca02c"]))
    for r, n in enumerate(sizes):
        for c, fam in enumerate(FAMILIES):
            ax, labels = axes[r, c], []
            if (fam, n) in skip:
                ax.set_visible(False)
                continue
            for pos, (method, (label, _, key)) in enumerate(METHODS.items()):
                runs = data.get((method, fam, n), [])
                ok = [x for x in runs if float(x.get(key, 0) or 0) >= 1 and x.get("wall_time")]
                labels.append(f"{100 * len(ok) // len(runs)}" if runs else "-")
                if not ok:
                    continue
                v = [float(x["wall_time"]) for x in ok]
                # A cell with no spread draws a box of zero height, which the white median
                # line then hides: lay a coloured bar under it so the method still reads.
                ax.plot([pos - 0.28, pos + 0.28], [np.median(v)] * 2, color=colour[method],
                        linewidth=2.4, zorder=1, solid_capstyle="butt")
                ax.boxplot(v, positions=[pos], widths=0.55, zorder=2, patch_artist=True,
                           medianprops=dict(color="white", linewidth=0.6),
                           boxprops=dict(facecolor=colour[method], edgecolor=colour[method],
                                         linewidth=0.6),
                           whiskerprops=dict(linewidth=0.5, color=colour[method]),
                           capprops=dict(linewidth=0.5, color=colour[method]),
                           flierprops=dict(ms=3, markeredgecolor=colour[method]))
            ax.set_xlim(-0.7, len(METHODS) - 0.3)
            ax.set_yscale("log")
            ax.set_ylim(3, 1200)
            ax.tick_params(labelsize=7, length=2, pad=1)
            ax.set_xticks(range(len(METHODS)), labels, fontsize=7)
            for tick, method in zip(ax.get_xticklabels(), METHODS):
                tick.set_color(colour[method])
            if r == 0:
                ax.set_title(fam.replace("_cross", "").replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"$N = {n}$", fontsize=9)
    fig.supxlabel("success rate [%] per method", fontsize=8, y=0.075)
    fig.supylabel("runtime [s]", fontsize=8, x=0.012)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=colour[m], label=METHODS[m][0])
                        for m in METHODS], ncol=3, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=(0.02, 0.13, 1, 1), h_pad=0.7, w_pad=0.5)
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
