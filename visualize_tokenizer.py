"""
python visualize_tokenizer.py <mesh.pt> [mesh_idx|start:end] [n_faces] [alpha]

mesh_idx: single index (e.g. "0") or Python-style slice "start:end" (e.g. "0:5").
Range produces one PNG per mesh: viz_<idx>.png
"""
from tokenizer_v2 import Tokenizer2D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

PALETTE = [
    '#e41a1c', '#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#a65628',
    '#f781bf', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
    '#e6ab02', '#a6761d', '#000000', '#8dd3c7', '#fb8072', '#80b1d3',
]


def viz(mesh_path, mesh_idx=0, n_faces=None, alpha=0.30, out='viz.png'):
    mesh = torch.load(mesh_path, weights_only=False)[mesh_idx]
    verts = mesh.x[:, :2]
    tok = Tokenizer2D(quantization_levels=256, sorting_strategy=1)
    ordered = tok._order_quads(verts, mesh.faces)  # [n_total, 4]

    n_total = ordered.shape[0]
    n = n_total if n_faces is None else min(n_faces, n_total)
    v = verts.numpy()

    # Persistent color per unique vertex (first-appearance order)
    col_idx = {}
    for fi in range(n):
        for vi in range(4):
            vidx = ordered[fi, vi].item()
            if vidx not in col_idx:
                col_idx[vidx] = len(col_idx)

    def vc(vidx): return PALETTE[col_idx[vidx] % len(PALETTE)]

    # First-occurrence flag per (face, vertex-position)
    seen, first = set(), {}
    for fi in range(n):
        for vi in range(4):
            vidx = ordered[fi, vi].item()
            first[(fi, vi)] = vidx not in seen
            seen.add(vidx)

    faces_per_row = 6
    n_seq_rows = (n + faces_per_row - 1) // faces_per_row
    seq_h = max(1.8, n_seq_rows * 1.1)

    fig = plt.figure(figsize=(9, 8.5 + seq_h))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[8.5, seq_h], hspace=0.06)
    ax = fig.add_subplot(gs[0])

    # Full mesh (face-order coolwarm, all faces)
    cmap = plt.cm.coolwarm
    for fi in range(n_total):
        coords = v[ordered[fi].numpy()]
        c = cmap(fi / max(n_total - 1, 1))
        ax.add_patch(Polygon(coords, closed=True,
                             fc=(*c[:3], 0.18), ec='#888', lw=0.5, zorder=1))
        ax.text(*coords.mean(0), str(fi + 1), ha='center', va='center',
                fontsize=5, color='#444', zorder=2)

    # Mini-element per face: vertices shifted alpha-fraction towards centroid
    for fi in range(n):
        coords = v[ordered[fi].numpy()]
        inner = coords * (1 - alpha) + coords.mean(0) * alpha

        ax.add_patch(Polygon(inner, closed=True, fc='none',
                     ec='#333', lw=0.7, zorder=3))

        for vi in range(4):
            vidx = ordered[fi, vi].item()
            is_first = first[(fi, vi)]
            ax.plot(*inner[vi], 'o' if is_first else 's', color=vc(vidx),
                    ms=8 if is_first else 5, mec='black', mew=0.4, zorder=4)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(handles=[
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=10, label='First occurrence'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=7, label='Repeated vertex'),
    ], loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(
        f'{mesh_path}  [mesh {mesh_idx}]  –  Strategy 1'
        f'   ({n}/{n_total} faces,  α={alpha})',
        fontsize=9,
    )

    # Token sequence – multi-line, one face group (4 tokens) per slot
    ax_s = fig.add_subplot(gs[1])
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, n_seq_rows)
    ax_s.axis('off')

    face_w = 1.0 / faces_per_row          # width per face slot
    dx = face_w * 0.20                    # spacing between tokens within a face

    for fi in range(n):
        row = n_seq_rows - 1 - fi // faces_per_row   # top-to-bottom
        col = fi % faces_per_row
        x0 = col * face_w + face_w * 0.08

        for vi in range(4):
            vidx = ordered[fi, vi].item()
            xn = x0 + vi * dx
            ax_s.text(xn, row + 0.70, f'$v_{vi + 1}^{{{fi + 1}}}$',
                      ha='center', va='center', fontsize=9,
                      color=vc(vidx), fontweight='bold')
            if first[(fi, vi)]:
                ax_s.plot(xn, row + 0.25, 'o', color=vc(vidx),
                          ms=5, mec='black', mew=0.4)

    ax_s.text(0.0, -0.15 / n_seq_rows, '●  First occurrence of unique vertex',
              fontsize=7.5, ha='left', va='top', color='#555')

    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'→ {out}')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'centered_blades_cleaned.pt'
    midx_arg = sys.argv[2] if len(sys.argv) > 2 else '0'
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30

    if ':' in midx_arg:
        a, b = midx_arg.split(':')
        indices = list(range(int(a), int(b)))
    else:
        indices = [int(midx_arg)]

    for i in indices:
        out = f'viz_{i}.png' if len(indices) > 1 else 'viz.png'
        viz(path, i, nf, alpha, out=out)
