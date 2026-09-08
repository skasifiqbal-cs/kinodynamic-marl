"""Run the frozen K-ARC configuration over every cross scenario and emit one table.

One row per scenario, written to ``experiments/karc_bench.csv`` (and printed as markdown).

Caveats that belong on any table built from this, and are repeated in the CSV header:

* n=1 per scenario. K-ARC reports 20 trials (§V-A); a single deterministic trial from
  ``init=fixed`` is not a success *rate*, it is one sample of one.
* Wall times are NOT comparable to the paper's. K-ARC is C++ in the Parasol Planning
  Library on a 32-core i9-14900K; this is Python driving IPOPT through CasADi.
* Every run gets the same ``workers`` count so the 600 s budget buys the same amount of
  search in each -- concurrency across scenarios is capped to keep that true.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    "open_cross_4_unicycle2", "open_cross_8_unicycle2",
    "open_cross_16_unicycle2", "open_cross_32_unicycle2",
    "open_cross_32_wide_unicycle2",
    "cluttered_cross_4_unicycle2", "cluttered_cross_8_unicycle2",
    "cluttered_cross_16_unicycle2", "cluttered_cross_32_unicycle2",
]

COLUMNS = ["scenario", "success", "collisions", "steps", "wall_time", "path_cost",
           "solver_calls", "conflicts", "conflicts_remaining", "subproblems",
           "subproblem_max", "rounds", "adaptations", "unsolved_segments", "timed_out",
           "rungs"]


def _run(env_name: str, workers: int, extra: list[str], out_dir: Path) -> dict:
    cmd = [sys.executable, "-m", "main", "approach=planning", "approach.method=karc",
           f"env={env_name}", "init=fixed", f"approach.karc.workers={workers}", *extra]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    (out_dir / f"{env_name}.log").write_text(out)

    row = {"scenario": env_name}
    if m := re.search(r"^RESULT,karc,([\d.]+),[\d.]+,([\d.]+),([\d.nan]+)", out, re.M):
        row |= {"success": m[1], "collisions": m[2], "steps": m[3]}
    if m := re.search(r"^STATS,karc,(.*)$", out, re.M):
        # rungs={'prioritized': 8} has a comma inside it, so split on ",<key>=" not ",".
        for field in re.split(r",(?=[a-z_]+=)", m[1]):
            k, _, v = field.partition("=")
            if k in COLUMNS:
                row[k] = v
    row.setdefault("wall_time", f"{time.time() - t0:.1f}")
    if proc.returncode != 0:
        row["scenario"] += " (FAILED)"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10, help="pool size inside each run")
    ap.add_argument("--concurrency", type=int, default=3, help="scenarios run at once")
    ap.add_argument("--out", default="experiments/karc_bench.csv")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    # NOT a positional: `--scenarios a b c foo=bar` is nargs="*" and swallows the override
    # as a fourth scenario name, silently benchmarking the default config under the
    # ablation's label. Cost an hour once; do not make it a positional again.
    ap.add_argument("--extra", nargs="*", default=[], help="extra hydra overrides")
    args = ap.parse_args()

    out = ROOT / args.out
    log_dir = out.with_suffix("")
    log_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(lambda e: _run(e, args.workers, args.extra, log_dir),
                             args.scenarios))

    with out.open("w", newline="") as fh:
        fh.write(f"# n=1 per scenario; workers={args.workers} concurrency={args.concurrency}; "
                 "wall times are Python+CasADi, NOT comparable to K-ARC's C++ numbers\n")
        w = csv.DictWriter(fh, COLUMNS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    show = [c for c in COLUMNS if any(r.get(c) for r in rows)]
    print("| " + " | ".join(show) + " |")
    print("|" + "|".join("---" for _ in show) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in show) + " |")
    print(f"\nwrote {out}  logs in {log_dir}/")


if __name__ == "__main__":
    main()
