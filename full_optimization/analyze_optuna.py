import os
import json
import sqlite3
import optuna
import optuna.visualization.matplotlib as mpl_vis
from optuna.visualization import plot_optimization_history
import optuna.importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/home/t1dde/Work/repos/meshtron/full_optimization"
OUT_DIR = f"{BASE_DIR}/optuna_report"
os.makedirs(OUT_DIR, exist_ok=True)

SORTINGS = ["0", "1", "3"]
STAGES = ["a", "b", "c"]
STAGE_NAMES = {"a": "Architecture (Stage A)", "b": "Hyperparameters (Stage B)", "c": "Training Dynamics (Stage C)"}
SORTING_LABELS = {
    "0": "Sorting 0 (Baseline)",
    "1": "Sorting 1 (Novel)",
    "3": "Sorting 3 (Adjacency + Deduplication)",
}
COLORS = {"0": "#2196F3", "1": "#4CAF50", "3": "#FF9800"}


def mpl_to_html(fig, path):
    if fig is None:
        return
    try:
        from plotly.tools import mpl_to_plotly
        pfig = mpl_to_plotly(fig)
        pfig.write_html(path)
    except Exception:
        try:
            import plotly.io as pio
            pio.write_html(fig, path)
        except Exception:
            pass


def get_figure_from_obj(obj):
    if isinstance(obj, plt.Figure):
        return obj
    if isinstance(obj, np.ndarray):
        flat = np.asarray(obj).flatten()
        figs = set(ax.figure for ax in flat if ax.figure is not None)
        if len(figs) == 1:
            return list(figs)[0]
        n = len(flat)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        for i, ax in enumerate(flat[:n]):
            if i < len(axes_flat):
                axes_flat[i].__dict__.update(ax.__dict__)
        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)
        fig.tight_layout()
        return fig
    if isinstance(obj, plt.Axes):
        return obj.figure if obj.figure is not None else None
    return None


def save_png(obj, path, dpi=150, title=""):
    fig = get_figure_from_obj(obj)
    if fig is None:
        return
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def load_study(db_path):
    studies = optuna.get_all_study_names(f"sqlite:///{db_path}")
    if not studies:
        return None
    return optuna.load_study(storage=f"sqlite:///{db_path}", study_name=studies[0])


STUDIES = {}
for s in SORTINGS:
    STUDIES[s] = {}
    for st in STAGES:
        db = f"{BASE_DIR}/sweep_s{s}_{st}.db"
        STUDIES[s][st] = load_study(db)


def get_completed(study):
    if study is None:
        return []
    return [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE]


def best_over_time(study):
    completed = get_completed(study)
    if not completed:
        return [], []
    vals = [t.value for t in completed]
    best = []
    cur = float("inf")
    for v in vals:
        cur = min(cur, v)
        best.append(cur)
    return list(range(1, len(completed) + 1)), best


# ─── 1. OPT HISTORY OVERVIEW GRID ───────────────────────────────────────────
print("Generating optimization history overview...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Optimization History — All 9 Studies", fontsize=18, fontweight="bold", y=1.01)

for col_idx, stage in enumerate(STAGES):
    for row_idx, sort in enumerate(SORTINGS):
        ax = axes[row_idx][col_idx]
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            ax.set_text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title(f"S{sort}-{stage.upper()}\n(n=0)", fontsize=10, fontweight="bold")
            ax.set_visible(False)
            continue

        x, y = best_over_time(study)
        color = COLORS[sort]
        ax.plot(x, y, marker="o", markersize=3, color=color, linewidth=1.8, label="Best-so-far")
        ax.scatter(x, [t.value for t in completed], color=color, alpha=0.25, s=12, zorder=5, label="Trial value")

        best_val = min(t.value for t in completed)
        ax.axhline(best_val, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(len(x), best_val, f" {best_val:.4f}", fontsize=8, va="center", color=color)

        ax.set_title(f"S{sort} — {STAGE_NAMES[stage]}\n(n={len(completed)}, best={best_val:.4f})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Trial", fontsize=8)
        ax.set_ylabel("Val BPT", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(fontsize=6, loc="upper right")

fig.tight_layout()
save_png(fig, f"{OUT_DIR}/opt_history_overview.png")
mpl_to_html(fig, f"{OUT_DIR}/opt_history_overview.html")
plt.close(fig)


# ─── 2. STAGE COMPARISON (overlaid) ─────────────────────────────────────────
print("Generating stage comparison...")

# Per-stage comparison (3 sortings overlaid)
for stage in STAGES:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sort in SORTINGS:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        x, y = best_over_time(study)
        color = COLORS[sort]
        label = f"S{sort} — {SORTING_LABELS[sort].split('(')[0].strip()} (best={y[-1]:.4f})"
        ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=2, label=label)

    ax.set_title(f"Optimization History — {STAGE_NAMES[stage]}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel("Validation Bits per Token", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_png(fig, f"{OUT_DIR}/stage_comparison_{stage}.png")
    mpl_to_html(fig, f"{OUT_DIR}/stage_comparison_{stage}.html")
    plt.close(fig)

# Combined all-in-one comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, stage in enumerate(STAGES):
    ax = axes[idx]
    for sort in SORTINGS:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        x, y = best_over_time(study)
        color = COLORS[sort]
        ax.plot(x, y, marker="o", markersize=3, color=color, linewidth=1.5, label=f"S{sort}")

    ax.set_title(STAGE_NAMES[stage], fontsize=10, fontweight="bold")
    ax.set_xlabel("Trial", fontsize=9)
    ax.set_ylabel("Val BPT", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0.22, 0.70)

fig.suptitle("Stage Comparison — All Sorting Strategies", fontsize=14, fontweight="bold")
fig.tight_layout()
save_png(fig, f"{OUT_DIR}/stage_comparison_all.png")
mpl_to_html(fig, f"{OUT_DIR}/stage_comparison_all.html")
plt.close(fig)


# ─── 3. BEST VALUE RANKING BAR CHART ─────────────────────────────────────────
print("Generating best value summary...")
labels_list = []
best_list = []
color_list = []
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        labels_list.append(f"S{sort}-{stage.upper()}")
        best_list.append(min(t.value for t in completed))
        color_list.append(COLORS[sort])

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(len(best_list)), best_list, color=color_list, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(best_list)))
ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Best Validation Bits per Token", fontsize=11)
ax.set_title("Best Value per Study (ranked)", fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars, best_list):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=7)

legend_patches = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s], label=f"S{s} — {SORTING_LABELS[s].split('(')[0].strip()}")
                  for s in SORTINGS]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

fig.tight_layout()
save_png(fig, f"{OUT_DIR}/best_value_summary.png")
mpl_to_html(fig, f"{OUT_DIR}/best_value_summary.html")
plt.close(fig)


# ─── 4. PARAM IMPORTANCE ─────────────────────────────────────────────────────
print("Generating param importance plots...")

def manual_param_importance(study):
    completed = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {}
    param_names = list(completed[0].params.keys())
    if not param_names:
        return {}
    values = [t.value for t in completed]
    from scipy.stats import spearmanr
    imp = {}
    for pname in param_names:
        pvals = [t.params.get(pname, None) for t in completed]
        valid = [(v, p) for v, p in zip(values, pvals) if p is not None]
        if len(valid) < 3:
            continue
        v, p = zip(*valid)
        corr, _ = spearmanr(v, p)
        imp[pname] = abs(corr) if not np.isnan(corr) else 0.0
    return imp


for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue

    all_imp = {}
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        if not study:
            continue
        try:
            imp = manual_param_importance(study)
            for k, v in imp.items():
                all_imp[k] = max(all_imp.get(k, 0), v)
        except Exception as e:
            pass

    if all_imp:
        sorted_imp = sorted(all_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [x[0] for x in sorted_imp]
        vals = [x[1] for x in sorted_imp]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(names)), vals, color=COLORS[sort], edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("Spearman |correlation| with Val BPT", fontsize=11)
        ax.set_title(f"Top 10 Parameters — {SORTING_LABELS[sort]}\n(max across stages, rank-based)", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        save_png(fig, f"{OUT_DIR}/param_importance_top10_s{sort}.png")
        try:
            mpl_to_html(fig, f"{OUT_DIR}/param_importance_top10_s{sort}.html")
        except Exception:
            pass
        plt.close("all")


# ─── 5. SLICE PLOTS ──────────────────────────────────────────────────────────
print("Generating slice plots...")
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        if not study:
            continue
        try:
            obj = mpl_vis.plot_slice(study, target=lambda t: t.value)
            if obj is not None:
                fig = get_figure_from_obj(obj)
                if fig:
                    save_png(fig, f"{OUT_DIR}/slice_s{sort}_{stage}.png", dpi=120)
                    try:
                        mpl_to_html(fig, f"{OUT_DIR}/slice_s{sort}_{stage}.html")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Slice plot S{sort}-{stage}: {e}")

# ─── 6. PARALLEL COORDINATES ─────────────────────────────────────────────────
print("Generating parallel coordinates...")
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        if not study:
            continue
        try:
            obj = mpl_vis.plot_parallel_coordinate(study, target=lambda t: t.value)
            if obj is not None:
                fig = get_figure_from_obj(obj)
                if fig:
                    save_png(fig, f"{OUT_DIR}/parallel_s{sort}_{stage}.png", dpi=120)
                    try:
                        mpl_to_html(fig, f"{OUT_DIR}/parallel_s{sort}_{stage}.html")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Parallel coords S{sort}-{stage}: {e}")

# ─── 7. CONTOUR PLOTS ────────────────────────────────────────────────────────
print("Generating contour plots...")
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        if not study:
            continue
        try:
            obj = mpl_vis.plot_contour(study, target=lambda t: t.value)
            if obj is not None:
                fig = get_figure_from_obj(obj)
                if fig:
                    save_png(fig, f"{OUT_DIR}/contour_s{sort}_{stage}.png", dpi=120)
                    try:
                        mpl_to_html(fig, f"{OUT_DIR}/contour_s{sort}_{stage}.html")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Contour S{sort}-{stage}: {e}")


# ─── 8. CONVERGENCE COMPARISON (all in one) ─────────────────────────────────
print("Generating convergence comparison...")
fig, ax = plt.subplots(figsize=(12, 6))
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        x, y = best_over_time(study)
        ls = "-" if stage == "a" else "--" if stage == "b" else ":"
        ax.plot(x, y, marker="o", markersize=3, color=COLORS[sort], linewidth=1.5, linestyle=ls, label=f"S{sort}-{stage.upper()}")

ax.set_xlabel("Trial Number", fontsize=12)
ax.set_ylabel("Best Validation Bits per Token", fontsize=12)
ax.set_title("All Studies — Convergence Comparison\nSolid=StageA, Dashed=StageB, Dotted=StageC", fontsize=13, fontweight="bold")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_png(fig, f"{OUT_DIR}/convergence_all.png")
mpl_to_html(fig, f"{OUT_DIR}/convergence_all.html")
plt.close(fig)


# ─── 9. SUMMARY JSON + MARKDOWN ──────────────────────────────────────────────
print("Generating summary files...")
rows = []
s0_baseline = {}
for sort in SORTINGS:
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        best = min(t.value for t in completed)
        if sort == "0":
            s0_baseline[stage] = best
        delta = ((best - s0_baseline.get(stage, best)) / s0_baseline.get(stage, best)) * 100 if sort != "0" else None
        rows.append({
            "sorting": sort,
            "stage": stage,
            "n_trials": len(completed),
            "best_val_bpt": best,
            "delta_vs_s0_pct": round(delta, 4) if delta is not None else None,
        })

rows.sort(key=lambda x: x["best_val_bpt"])

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(rows, f, indent=2)

md = ["# Optuna Study Summary\n",
      "| Rank | Sorting | Stage | Trials | Best Val BPT | vs S0 |\n",
      "|---|---|---|---|---|---|\n"]
for i, r in enumerate(rows):
    delta = f"{r['delta_vs_s0_pct']:.2f}%" if r["delta_vs_s0_pct"] is not None else "—"
    md.append(f"| {i+1} | S{r['sorting']} | {r['stage']} | {r['n_trials']} | {r['best_val_bpt']:.6f} | {delta} |\n")

with open(f"{OUT_DIR}/summary.md", "w") as f:
    f.writelines(md)


# ─── 10. INDEX HTML REPORT ───────────────────────────────────────────────────
print("Generating index HTML...")
idx = ["<!DOCTYPE html><html><head>",
       "<meta charset='utf-8'>",
       "<title>Meshtron Optuna Analysis</title>",
       "<style>",
       "body{font-family:sans-serif;margin:20px;background:#fafafa}",
       "h1{color:#222;border-bottom:2px solid #333;padding-bottom:8px}",
       "h2{color:#444;margin-top:40px}",
       "h3{color:#666;margin-top:20px}",
       "table{border-collapse:collapse;margin:15px 0;width:100%;max-width:900px}",
       "th,td{border:1px solid #bbb;padding:8px 12px;text-align:left}",
       "th{background:#222;color:#fff}",
       "tr:nth-child(even){background:#f2f2f2}",
       "img{max-width:100%;height:auto;margin:8px 0;border:1px solid #ddd;background:#fff}",
       ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(450px,1fr));gap:20px;margin:15px 0}",
       ".card{background:#fff;padding:15px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}",
       ".card h3{margin-top:0;color:#333}",
       ".badge{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-size:12px}",
       ".s0{background:#2196F3}.s1{background:#4CAF50}.s3{background:#FF9800}",
       "a{color:#1565C0}",
       "</style></head><body>",
       "<h1>Meshtron Full Optimization — Optuna Study Analysis</h1>",
       "<p>Sorting strategies: <span class='badge s0'>S0 = Baseline</span> "
       "<span class='badge s1'>S1 = Novel method</span> "
       "<span class='badge s3'>S3 = Adjacency + deduplication</span></p>",
       "<h2>Best Value Ranking</h2>",
       "<table><tr><th>Rank</th><th>Sort</th><th>Stage</th><th>Trials</th>"
       "<th>Best Val BPT</th><th>vs S0</th></tr>"]

for i, r in enumerate(rows):
    delta = f"{r['delta_vs_s0_pct']:.2f}%" if r["delta_vs_s0_pct"] is not None else "—"
    idx.append(f"<tr><td>{i+1}</td><td>S{r['sorting']}</td><td>{r['stage']}</td>"
               f"<td>{r['n_trials']}</td><td>{r['best_val_bpt']:.6f}</td><td>{delta}</td></tr>")

idx.append("</table>")

idx.append("<h2>Optimization History Overview</h2>")
idx.append(f"<img src='opt_history_overview.png' onclick=\"this.style.width=this.style.width==='100%'?'auto':'100%'\">")

idx.append("<h2>Stage Comparison</h2>")
idx.append("<div class='grid'>")
for stage in STAGES:
    idx.append(f"<div class='card'><h3>{STAGE_NAMES[stage]}</h3>"
               f"<img src='stage_comparison_{stage}.png'></div>")
idx.append("</div>")

idx.append("<h2>Best Value Summary</h2>")
idx.append(f"<img src='best_value_summary.png' onclick=\"this.style.width=this.style.width==='100%'?'auto':'100%'\">")

idx.append("<h2>Convergence Comparison</h2>")
idx.append(f"<img src='convergence_all.png' onclick=\"this.style.width=this.style.width==='100%'?'auto':'100%'\">")

for sort in SORTINGS:
    idx.append(f"<h2>Sorting {sort}: {SORTING_LABELS[sort]}</h2>")
    idx.append("<div class='grid'>")
    for stage in STAGES:
        study = STUDIES.get(sort, {}).get(stage)
        completed = get_completed(study)
        if not completed:
            continue
        best = min(t.value for t in completed)
        links = []
        for ext, label in [("opt_history", "Opt History"),
                            ("stage_comparison", "Stage Comparison"),
                            ("param_importance", "Param Importance"),
                            ("slice", "Slice"),
                            ("parallel", "Parallel Coords"),
                            ("contour", "Contour")]:
            fname_map = {
                "opt_history": f"opt_history_overview.png",
                "stage_comparison": f"stage_comparison_{stage}.png",
                "param_importance": f"param_importance_s{sort}_{stage}.png",
                "slice": f"slice_s{sort}_{stage}.html",
                "parallel": f"parallel_s{sort}_{stage}.html",
                "contour": f"contour_s{sort}_{stage}.html",
            }
            fpath = fname_map.get(ext, "")
            if ext.endswith(".html") and os.path.exists(f"{OUT_DIR}/{fpath}"):
                links.append(f"<a href='{fpath}' target='_blank'>{label} ↗</a>")
            elif ext.endswith(".png") and os.path.exists(f"{OUT_DIR}/{fpath}"):
                links.append(f"<a href='{fpath}' target='_blank'>{label} ↗</a>")

        img_file = f"param_importance_s{sort}_{stage}.png"
        img_tag = f"<img src='{img_file}'>" if os.path.exists(f"{OUT_DIR}/{img_file}") else ""

        links_str = " | ".join(links) if links else ""
        idx.append(f"<div class='card'><h3>Stage {stage.upper()}: {STAGE_NAMES[stage]}</h3>"
                   f"<p>Trials: {len(completed)}, Best: {best:.6f}</p>{img_tag}<p>{links_str}</p></div>")

    # Top10 param importance
    top10_file = f"param_importance_top10_s{sort}.png"
    if os.path.exists(f"{OUT_DIR}/{top10_file}"):
        idx.append(f"<div class='card'><h3>Top 10 Params (aggregated)</h3>"
                   f"<img src='{top10_file}'></div>")
    idx.append("</div>")

idx.append("</body></html>")

with open(f"{OUT_DIR}/index.html", "w") as f:
    f.write("\n".join(idx))


# ─── DONE ────────────────────────────────────────────────────────────────────
print(f"\nAll outputs saved to: {OUT_DIR}")
files = sorted(os.listdir(OUT_DIR))
print(f"Total files: {len(files)}")
for fn in files:
    sz = os.path.getsize(f"{OUT_DIR}/{fn}") / 1024
    print(f"  {fn:45s} {sz:7.1f} KB")