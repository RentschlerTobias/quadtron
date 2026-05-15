"""Aggregate and plot the fair sorting-strategy comparison.

Reads runs/meshtron-sorting-fair/summary.jsonl (one JSON object per run,
written by compare_sorting.py) and produces:

  1. A printed table of mean +/- std per strategy for the key metrics.
  2. runs/meshtron-sorting-fair/aggregate.csv with the same numbers.
  3. runs/meshtron-sorting-fair/comparison.png with bar plots of
     val_bpt, bits_per_mesh, val_tokens_per_epoch, peak_gpu_mem_mib and
     wall_time_s side-by-side.

The analysis treats sorting_strategy as a categorical factor and ignores
runs that did not produce a finite best_val_bpt (e.g. crashed early).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


STRATEGY_LABEL = {1: "1: YZX lex", 2: "2: rows / half-edge"}

# Metrics shown in the printed/aggregated table and (those marked True) plotted.
METRICS = [
    ("best_val_bpt",           "val bpt",         "lower is better", True),
    ("bits_per_mesh",          "bits / mesh",     "lower is better", True),
    ("val_tokens_per_epoch",   "val tokens",      "lower is better", True),
    ("max_seq_length",         "max seq len",     "lower is better", False),
    ("peak_gpu_mem_mib",       "peak GPU [MiB]",  "lower is better", True),
    ("wall_time_s",            "wall time [s]",   "lower is better", True),
    ("epochs_run",             "epochs run",      "info",            False),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-path", type=Path,
                   default=Path("runs/meshtron-sorting-fair/summary.jsonl"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for aggregate.csv and comparison.png "
                        "(default: parent of --summary-path).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip writing comparison.png.")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist - run compare_sorting.py first.")
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _finite(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        v = r.get("best_val_bpt")
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(r)
    return out


def _agg(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    if len(values) == 1:
        return float(values[0]), 0.0, 1
    return mean(values), stdev(values), len(values)


def aggregate(rows: list[dict]) -> dict[int, dict[str, tuple[float, float, int]]]:
    by_strat: dict[int, list[dict]] = {}
    for r in rows:
        by_strat.setdefault(int(r["sorting_strategy"]), []).append(r)
    summary: dict[int, dict[str, tuple[float, float, int]]] = {}
    for strat, rs in by_strat.items():
        summary[strat] = {
            key: _agg([float(r[key]) for r in rs if key in r])
            for key, *_ in METRICS
        }
    return summary


def print_table(summary: dict[int, dict[str, tuple[float, float, int]]]) -> None:
    strategies = sorted(summary.keys())
    header_strats = [STRATEGY_LABEL.get(s, f"strategy {s}") for s in strategies]
    col_w = max(28, *(len(h) for h in header_strats)) + 2

    print(f"{'metric':<22} | " + " | ".join(f"{h:<{col_w}}" for h in header_strats))
    print("-" * (22 + 3 + (col_w + 3) * len(strategies)))
    for key, label, direction, _ in METRICS:
        cells = []
        for s in strategies:
            m, sd, n = summary[s][key]
            if math.isnan(m):
                cells.append(f"{'n/a':<{col_w}}")
            elif n <= 1:
                cells.append(f"{m:>12.4g}              (n=1)".ljust(col_w))
            else:
                cells.append(f"{m:>12.4g} +/- {sd:<8.3g} (n={n})".ljust(col_w))
        print(f"{label:<22} | " + " | ".join(cells) + f"   {direction}")


def _ratio_line(summary: dict[int, dict[str, tuple[float, float, int]]]) -> None:
    if 1 not in summary or 2 not in summary:
        return
    print("\nratios strategy 2 / strategy 1 (lower < 1 means strategy 2 wins):")
    for key, label, *_ in METRICS:
        m1, _, _ = summary[1][key]
        m2, _, _ = summary[2][key]
        if not (math.isfinite(m1) and math.isfinite(m2)) or m1 == 0:
            continue
        print(f"  {label:<22}  {m2 / m1:.3f}")


def write_csv(summary: dict[int, dict[str, tuple[float, float, int]]],
              out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["strategy"] + [f"{k}_mean" for k, *_ in METRICS] \
                       + [f"{k}_std"  for k, *_ in METRICS] \
                       + [f"{k}_n"    for k, *_ in METRICS]
    with out_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for s in sorted(summary.keys()):
            row = [str(s)]
            row += [f"{summary[s][k][0]}" for k, *_ in METRICS]
            row += [f"{summary[s][k][1]}" for k, *_ in METRICS]
            row += [f"{summary[s][k][2]}" for k, *_ in METRICS]
            f.write(",".join(row) + "\n")


def plot(summary: dict[int, dict[str, tuple[float, float, int]]],
         out_path: Path) -> None:
    plotted = [m for m in METRICS if m[3]]
    strategies = sorted(summary.keys())
    labels = [STRATEGY_LABEL.get(s, f"strategy {s}") for s in strategies]

    fig, axes = plt.subplots(1, len(plotted), figsize=(3.2 * len(plotted), 3.6))
    if len(plotted) == 1:
        axes = [axes]

    for ax, (key, label, _direction, _) in zip(axes, plotted):
        means = [summary[s][key][0] for s in strategies]
        stds  = [summary[s][key][1] for s in strategies]
        bars = ax.bar(labels, means, yerr=stds, capsize=4,
                      color=["#4C72B0", "#DD8452"][:len(strategies)])
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        for b, m in zip(bars, means):
            if math.isfinite(m):
                ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3g}",
                        ha="center", va="bottom", fontsize=8)

    fig.suptitle("Sorting strategy comparison (mean +/- std across seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")


def main() -> None:
    args = parse_args()
    rows_all = load_rows(args.summary_path)
    rows = _finite(rows_all)
    if len(rows) < len(rows_all):
        print(f"warning: dropped {len(rows_all) - len(rows)} non-finite row(s)")
    if not rows:
        print("no usable rows - exiting.")
        return

    out_dir = args.out_dir or args.summary_path.parent

    print(f"loaded {len(rows)} run(s) from {args.summary_path}\n")
    summary = aggregate(rows)
    print_table(summary)
    _ratio_line(summary)

    write_csv(summary, out_dir / "aggregate.csv")
    print(f"\nwrote {out_dir / 'aggregate.csv'}")

    if not args.no_plot:
        plot(summary, out_dir / "comparison.png")


if __name__ == "__main__":
    main()
