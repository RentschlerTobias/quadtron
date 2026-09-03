import os
import json
import optuna
import optuna.importance
import optuna.visualization.matplotlib as mpl_vis
from optuna.visualization import plot_optimization_history
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BASE_DIR = "/home/t1dde/Work/repos/meshtron/full_optimization"
OUT_DIR = f"{BASE_DIR}/sweep_report"
os.makedirs(OUT_DIR, exist_ok=True)

DB_PATH = f"sqlite:///{BASE_DIR}/sweep.db"

STUDY_INFO = {
    "meshtron-stage2": {
        "label": "Stage 2 — Architecture Search",
        "color": "#9C27B0",
        "description": "Architecture search: d_model, stage_layers",
    },
    "meshtron-stage3": {
        "label": "Stage 3 — Hyperparameter Tuning",
        "color": "#E91E63",
        "description": "Hyperparameter tuning (lr, warmup, dropout, etc.)",
    },
    "meshtron-sorting": {
        "label": "Sorting Strategy Comparison",
        "color": "#FF9800",
        "description": "Comparing sorting strategies 1 vs 2",
    },
    "meshtron-all": {
        "label": "All Parameters Combined",
        "color": "#00BCD4",
        "description": "Full joint optimization across all parameter spaces",
    },
}


def load_all_studies():
    studies = {}
    for name in STUDY_INFO:
        try:
            studies[name] = optuna.load_study(storage=DB_PATH, study_name=name)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
    return studies


def get_completed(study):
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


def save_png(obj, path, dpi=150):
    fig = get_figure_from_obj(obj)
    if fig is None:
        return
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


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


def rank_importance(study):
    completed = get_completed(study)
    if not completed:
        return {}
    if not completed[0].params:
        return {}
    values = [t.value for t in completed]
    imp = {}
    for pname in list(completed[0].params.keys()):
        pvals = [t.params.get(pname, None) for t in completed]
        valid = [(v, p) for v, p in zip(values, pvals) if p is not None]
        if len(valid) < 3:
            continue
        v, p = zip(*valid)
        corr, _ = spearmanr(v, p)
        imp[pname] = abs(corr) if not np.isnan(corr) else 0.0
    return imp


STUDIES = load_all_studies()
print(f"Loaded studies: {list(STUDIES.keys())}")

# ─── 1. OPTIMIZATION HISTORY PER STUDY ──────────────────────────────────────
print("Generating optimization histories...")
for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed:
        continue

    x, y = best_over_time(study)
    color = STUDY_INFO[name]["color"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", markersize=5, color=color, linewidth=2, label="Best-so-far")
    ax.scatter(x, [t.value for t in completed], color=color, alpha=0.3, s=20, zorder=5, label="Trial value")
    best_val = min(t.value for t in completed)
    ax.axhline(best_val, color=color, linestyle="--", alpha=0.5)
    ax.text(len(x), best_val, f" {best_val:.4f}", fontsize=9, va="center", color=color)
    ax.set_title(f"Optimization History — {STUDY_INFO[name]['label']}\n({len(completed)} trials, best={best_val:.4f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Trial", fontsize=10)
    ax.set_ylabel("Validation Bits per Token", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_png(fig, f"{OUT_DIR}/opt_history_{name}.png")
    mpl_to_html(fig, f"{OUT_DIR}/opt_history_{name}.html")
    plt.close(fig)


# ─── 2. BEST VALUE COMPARISON (bar chart) ───────────────────────────────────
print("Generating best value comparison...")
names = list(STUDIES.keys())
best_vals = []
n_trials_list = []
colors = []
for name in names:
    study = STUDIES[name]
    completed = get_completed(study)
    if not completed:
        continue
    best_vals.append(min(t.value for t in completed))
    n_trials_list.append(len(completed))
    colors.append(STUDY_INFO[name]["color"])

labels = [STUDY_INFO[n]["label"].split("—")[0].strip() for n in names]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(best_vals)), best_vals, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(best_vals)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Best Validation Bits per Token", fontsize=11)
ax.set_title("Best Value per Study in sweep.db", fontsize=13, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)

for bar, val, n in zip(bars, best_vals, n_trials_list):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}\n(n={n})", ha="center", va="bottom", fontsize=8)

legend_patches = [plt.Rectangle((0, 0), 1, 1, color=STUDY_INFO[n]["color"], label=STUDY_INFO[n]["label"])
                  for n in names]
ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

fig.tight_layout()
save_png(fig, f"{OUT_DIR}/best_value_comparison.png")
mpl_to_html(fig, f"{OUT_DIR}/best_value_comparison.html")
plt.close(fig)


# ─── 3. PARAM IMPORTANCE (fanova + rank correlation) ─────────────────────────
print("Generating param importance...")

for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed:
        continue

    info = STUDY_INFO[name]

    # Fanova (sklearn)
    try:
        fign = mpl_vis.plot_param_importances(study, target=lambda t: t.value, target_name="Val BPT")
        if fign is not None:
            save_png(fign, f"{OUT_DIR}/param_importance_fanova_{name}.png", dpi=130)
            mpl_to_html(fign, f"{OUT_DIR}/param_importance_fanova_{name}.html")
            plt.close("all")
    except Exception as e:
        print(f"  Fanova importance failed for {name}: {e}")

    # Rank correlation fallback
    imp = rank_importance(study)
    if imp:
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        names_p, vals_p = zip(*sorted_imp) if sorted_imp else ([], [])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(range(len(names_p)), vals_p, color=info["color"], edgecolor="white")
        ax.set_yticks(range(len(names_p)))
        ax.set_yticklabels(names_p, fontsize=9)
        ax.set_xlabel("Spearman |correlation| with Val BPT", fontsize=10)
        ax.set_title(f"Parameter Importance (Rank Correlation)\n{info['label']}", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        save_png(fig, f"{OUT_DIR}/param_importance_rank_{name}.png")
        mpl_to_html(fig, f"{OUT_DIR}/param_importance_rank_{name}.html")
        plt.close(fig)


# ─── 4. SLICE PLOTS ──────────────────────────────────────────────────────────
print("Generating slice plots...")
for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed or len(completed) < 3:
        print(f"  Skipping slice for {name} (only {len(completed)} trials)")
        continue
    try:
        obj = mpl_vis.plot_slice(study, target=lambda t: t.value)
        if obj is not None:
            fig = get_figure_from_obj(obj)
            if fig:
                save_png(fig, f"{OUT_DIR}/slice_{name}.png", dpi=120)
                mpl_to_html(fig, f"{OUT_DIR}/slice_{name}.html")
                plt.close("all")
    except Exception as e:
        print(f"  Slice plot failed for {name}: {e}")


# ─── 5. PARALLEL COORDINATES ─────────────────────────────────────────────────
print("Generating parallel coordinates...")
for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed or len(completed) < 3:
        print(f"  Skipping parallel coords for {name} (only {len(completed)} trials)")
        continue
    try:
        obj = mpl_vis.plot_parallel_coordinate(study, target=lambda t: t.value)
        if obj is not None:
            fig = get_figure_from_obj(obj)
            if fig:
                save_png(fig, f"{OUT_DIR}/parallel_{name}.png", dpi=120)
                mpl_to_html(fig, f"{OUT_DIR}/parallel_{name}.html")
                plt.close("all")
    except Exception as e:
        print(f"  Parallel coords failed for {name}: {e}")


# ─── 6. CONTOUR PLOTS ────────────────────────────────────────────────────────
print("Generating contour plots...")
for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed or len(completed) < 4:
        print(f"  Skipping contour for {name} (only {len(completed)} trials)")
        continue
    try:
        obj = mpl_vis.plot_contour(study, target=lambda t: t.value)
        if obj is not None:
            fig = get_figure_from_obj(obj)
            if fig:
                save_png(fig, f"{OUT_DIR}/contour_{name}.png", dpi=120)
                plt.close("all")
    except Exception as e:
        print(f"  Contour failed for {name}: {e}")


# ─── 7. ALL STUDIES OVERLAY (convergence comparison) ─────────────────────────
print("Generating convergence overlay...")
fig, ax = plt.subplots(figsize=(12, 6))

for name, study in STUDIES.items():
    completed = get_completed(study)
    if not completed:
        continue
    x, y = best_over_time(study)
    color = STUDY_INFO[name]["color"]
    label = f"{STUDY_INFO[name]['label']} (n={len(completed)}, best={y[-1]:.4f})"
    ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=2, label=label)

ax.set_xlabel("Trial Number", fontsize=12)
ax.set_ylabel("Best Validation Bits per Token", fontsize=12)
ax.set_title("All Studies — Convergence Comparison (sweep.db)", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_png(fig, f"{OUT_DIR}/convergence_all.png")
mpl_to_html(fig, f"{OUT_DIR}/convergence_all.html")
plt.close(fig)


# ─── 8. SUMMARY JSON + MARKDOWN ──────────────────────────────────────────────
print("Generating summary...")
rows = []
for name in STUDY_INFO:
    study = STUDIES.get(name)
    if not study:
        continue
    completed = get_completed(study)
    if not completed:
        continue
    best = min(t.value for t in completed)
    rows.append({
        "study": name,
        "label": STUDY_INFO[name]["label"],
        "n_trials": len(completed),
        "best_val_bpt": best,
        "params": list(completed[0].params.keys()) if completed else [],
    })

rows.sort(key=lambda x: x["best_val_bpt"])

with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(rows, f, indent=2)

md = ["# sweep.db Study Summary\n",
      "| Rank | Study | Trials | Best Val BPT | Parameters |\n",
      "|---|---|---|---|---|\n"]
for i, r in enumerate(rows):
    md.append(f"| {i+1} | {r['label']} | {r['n_trials']} | {r['best_val_bpt']:.6f} | {', '.join(r['params'])} |\n")

with open(f"{OUT_DIR}/summary.md", "w") as f:
    f.writelines(md)


# ─── 9. INDEX HTML REPORT ────────────────────────────────────────────────────
print("Generating index HTML...")

idx = [
    "<!DOCTYPE html><html><head>",
    "<meta charset='utf-8'>",
    "<title>sweep.db Optuna Analysis</title>",
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
    ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:20px;margin:15px 0}",
    ".card{background:#fff;padding:15px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}",
    ".card h3{margin-top:0}",
    "a{color:#1565C0}",
    "</style></head><body>",
    "<h1>sweep.db — Optuna Study Analysis</h1>",
    "<p>4 studies: Stage2, Stage3, Sorting Comparison, All-in-one</p>",
    "<h2>Best Value Comparison</h2>",
    f"<img src='best_value_comparison.png'>",
    "<h2>Convergence Comparison</h2>",
    f"<img src='convergence_all.png'>",
]

for name in STUDY_INFO:
    study = STUDIES.get(name)
    if not study:
        continue
    info = STUDY_INFO[name]
    completed = get_completed(study)
    if not completed:
        continue

    idx.append(f"<h2>{info['label']}</h2>")
    idx.append(f"<p>{info['description']} — {len(completed)} trials</p>")
    idx.append("<div class='grid'>")

    for ext in ["opt_history", "param_importance_fanova", "param_importance_rank",
                "slice", "parallel", "contour"]:
        fname = f"{OUT_DIR}/{ext}_{name}"
        png = f"{ext}_{name}.png"
        html = f"{ext}_{name}.html"
        if os.path.exists(f"{OUT_DIR}/{png}"):
            label = ext.replace("_", " ").title()
            idx.append(f"<div class='card'><h3>{label}</h3>"
                       f"<img src='{png}'>"
                       f"<p><a href='{html}' target='_blank'>Interactive ↗</a></p></div>")

    idx.append("</div>")

idx.append("</body></html>")

with open(f"{OUT_DIR}/index.html", "w") as f:
    f.write("\n".join(idx))

# ─── DONE ─────────────────────────────────────────────────────────────────────
plt.close("all")
print(f"\nAll outputs saved to: {OUT_DIR}")
files = sorted(os.listdir(OUT_DIR))
print(f"Total files: {len(files)}")
for fn in files:
    sz = os.path.getsize(f"{OUT_DIR}/{fn}") / 1024
    print(f"  {fn:50s} {sz:7.1f} KB")