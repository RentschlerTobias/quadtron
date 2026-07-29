"""
chain_e2e.py

END-TO-END-Verkettung von Plan B (nur 6-Face). Trainiert die drei Koepfe
(Stufe 1 Vertices, Stufe 2 Pointer-Faces, Stufe 3 HO-Geometrie) auf denselben
6F-Meshes (index-aligned) und fuehrt dann die volle Kette auf held-out aus:

    Punktwolke --S1--> Vertices --S2--> Faces --S3--> Geometrie --> Mesh

Kernfrage: FEHLER-FORTPFLANZUNG. S2/S3 laufen im Chain auf den GENERIERTEN
Vertices (nicht GT), S3 auf der generierten Topologie. Gemessen wird, wie stark
sich S1-Vertexfehler auf Face-Exaktheit (S2) und Kurvenfehler (S3) durchschlagen,
plus Loch-offen/Quads-distinct-Gueltigkeit. Galerie GT vs generiert -> figures/e2e/.

  ~/Environments/meshtron/bin/python chain_e2e.py \
      [--data domain_data_aug.pt] [--ep1 20 --ep2 25 --ep3 25] [--eval-n 150]
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from prototype_twostage import TwoStageTokenizer
import vertex_head_prototype as vh
import pointer_head_prototype as ph
import geom_head_prototype as gh
import train_pointer as tp


def make_sched(opt, epochs, steps_per_ep, warmup_frac=0.05):
    total = max(1, epochs * steps_per_ep); warm = max(1, int(warmup_frac * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def train_head(name, model, run_epoch, examples, train_ids, val_ids, epochs, bs,
               lr, device, rng, extra=()):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = make_sched(opt, epochs, math.ceil(len(train_ids) / bs))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, rng.permutation(train_ids), examples, bs, opt, sched,
                       1.0, device, *extra, train=True)
        if ep == epochs or ep % max(1, epochs // 4) == 0:
            vl = run_epoch(model, val_ids, examples, bs, None, None, 0, device,
                           *extra, train=False)
            trm = tr[0] if isinstance(tr, tuple) else tr
            vlm = vl[0] if isinstance(vl, tuple) else vl
            print(f"  [{name}] ep {ep:3d}  tr {trm:.4f}  val {vlm:.4f}")
    print(f"  [{name}] trainiert in {time.time()-t0:.0f}s")


# ---- Chain-Inferenz-Bausteine ----------------------------------------
@torch.no_grad()
def s1_generate(model, pts, tok, START, device):
    """Punktwolke -> 12 Vertices (polar r,theta in kanonischer Reihenfolge)."""
    Qr, Qa = tok.Qr, tok.Qa; STOP = Qr + 2 * Qa + 1
    pts = pts.unsqueeze(0).to(device)
    ppad = torch.zeros(1, pts.size(1), dtype=torch.bool, device=device)
    mem = model.encode(pts, ppad)
    seq = [START]
    for step in range(36):
        din = torch.tensor(seq, device=device)[None]
        logits = model.decode(mem, ppad, din)[0, -1]
        logits = vh.group_mask(logits, step, Qr, Qa, STOP)
        seq.append(int(logits.argmax()))
    gen = seq[1:]
    polar = []
    for m in range(12):
        r = tok._dq_scalar(gen[3 * m], tok.R_MIN, tok.R_MAX)
        th = tok._dq_angle(gen[3 * m + 1] - Qr, gen[3 * m + 2] - (Qr + Qa))
        polar.append((r, th))
    return np.array(polar)                                    # [12,2] (r,theta)


def faces_to_edges(faces_new):
    """[6,4] -> [24,2] gerichtete Half-Edges (Face-Traversal)."""
    e = []
    for f in faces_new:
        for k in range(4):
            e.append((int(f[k]), int(f[(k + 1) % 4])))
    return np.array(e, dtype=np.int64)


@torch.no_grad()
def chain_one(vmodel, pmodel, gmodel, pts, tok, START, device):
    """Volle Kette. Returns cart[12,2], faces_new[6,4], edge_curves dict (a,b)->[n,2]."""
    polar = s1_generate(vmodel, pts, tok, START, device)
    r, th = polar[:, 0], polar[:, 1]
    vert_feats = torch.tensor(np.stack([r, np.sin(th), np.cos(th)], 1),
                              dtype=torch.float32)             # [12,3]
    cart = np.stack([r * np.cos(th), r * np.sin(th)], 1)       # [12,2] (um center)

    vf = vert_feats.unsqueeze(0).to(device)
    ptrs = pmodel.generate(vf, 24)                            # 24 Zeiger
    faces_new = np.array(ptrs, dtype=np.int64).reshape(6, 4)

    e_new = faces_to_edges(faces_new)
    en = torch.tensor(e_new, device=device).unsqueeze(0)
    geom = gmodel(vf, en)[0].cpu().numpy()                    # [24,4]

    curves = {}
    for j, (a, b) in enumerate(e_new):
        P0, P1 = cart[a], cart[b]
        uh, nh, L = TwoStageTokenizer._chord_frame(P0, P1)
        s1, h1, s2, h2 = geom[j]
        B1 = P0 + s1 * L * uh + h1 * L * nh
        B2 = P0 + s2 * L * uh + h2 * L * nh
        curves[(a, b)] = gh._cubic_curve(P0, P1, B1, B2, 24)
    return cart, faces_new, curves


def draw_gen(ax, cart, center, faces_new, curves):
    cmap = plt.get_cmap('tab20')
    for fi, f in enumerate(faces_new):
        loop = []
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            loop.append(curves[(a, b)] + center[None])
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6, ec='k', lw=0.5)
    ax.set_aspect('equal'); ax.axis('off')


def draw_gt(ax, d):
    vc = d['vertices_cartesian'].numpy(); faces = d['faces'].numpy().T
    e2s = d['edge_to_streamline']; cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = []
        for k in range(4):
            u, v = int(q[k]), int(q[(k + 1) % 4])
            c = e2s.get((u, v))
            if c is None:
                c = e2s.get((v, u))
                c = np.asarray(c, float)[::-1] if c is not None else np.stack([vc[u], vc[v]])
            else:
                c = np.asarray(c, float)
            loop.append(c)
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6, ec='k', lw=0.5)
    ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_aug.pt')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ep1', type=int, default=20)
    ap.add_argument('--ep2', type=int, default=25)
    ap.add_argument('--ep3', type=int, default=25)
    ap.add_argument('--eval-n', type=int, default=150)
    ap.add_argument('--gallery', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='figures/e2e/e2e_gallery.png')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = TwoStageTokenizer(repr_mode='cubic_bezier')
    Qr, Qa = tok.Qr, tok.Qa; START = Qr + 2 * Qa; VOCAB = Qr + 2 * Qa + 2

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] == 6]
    if args.limit:
        six = six[:args.limit]
    print(f"6F-Meshes: {len(six)}  |  baue Beispiele (S1/S2/S3, index-aligned) ...")
    t0 = time.time()
    ex_v = vh.build_vertex_examples(six, tok)
    ex_p = ph.build_examples(six, tok)
    ex_g = gh.build_geom_examples(six, tok)
    print(f"  Beispiele in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(six))
    n_val = max(1, len(six) // 10)
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    print("== Training 3 Koepfe ==")
    vmodel = vh.VertexGen(VOCAB, d=args.d_model, start_id=START).to(device)
    pmodel = ph.PointerFaceModel(d_model=args.d_model).to(device)
    gmodel = gh.GeomHeadModel(d_model=args.d_model).to(device)
    train_head('S1', vmodel, vh.run_epoch, ex_v, train_ids, val_ids, args.ep1,
               args.batch, 5e-4, device, rng, extra=(START,))
    train_head('S2', pmodel, tp.run_epoch, ex_p, train_ids, val_ids, args.ep2,
               args.batch, 5e-4, device, rng)
    train_head('S3', gmodel, gh.run_epoch, ex_g, train_ids, val_ids, args.ep3,
               args.batch, 5e-4, device, rng)

    print("== Chain-Inferenz auf held-out ==")
    vmodel.eval(); pmodel.eval(); gmodel.eval()
    ev_ids = val_ids[:min(args.eval_n, len(val_ids))]
    vert_errs, face_exact, quads_distinct, curve_errs = [], 0, 0, []
    n_faces = 0
    for i in ev_ids:
        d = six[i]
        cart, faces_new, curves = chain_one(vmodel, pmodel, gmodel, ex_v[i]['pts'],
                                            tok, START, device)
        center = d['center'].numpy()
        # S1 Vertex-Fehler (generiert vs GT, index-weise, /Diag)
        gt_vc = ex_v[i]['vc']
        vert_errs.append(np.linalg.norm(cart + center - gt_vc, axis=1) / math.sqrt(2))
        # S2 Face-Exaktheit (generierte Topologie vs GT) + Quads-distinct
        gt_faces = ex_p[i][2]                                 # [6,4] GT faces_new
        for gi, fi in zip(faces_new, gt_faces):
            n_faces += 1
            if np.array_equal(gi, fi):
                face_exact += 1
            if len(set(gi.tolist())) == 4:
                quads_distinct += 1
        # S3 Kurvenfehler NUR wo Topologie exakt (Kanten korrespondieren zu GT)
        e2s = d['edge_to_streamline']; e_glob = ex_g[i]['e_glob']; gcart = ex_g[i]['cart']
        e_new = faces_to_edges(faces_new)
        for j, (a, b) in enumerate(e_new):
            if not np.array_equal(faces_new, gt_faces):
                continue
            p0, p1 = int(e_glob[j, 0]), int(e_glob[j, 1])
            pts = e2s.get((p0, p1))
            if pts is None:
                continue
            pts = np.asarray(pts, float)
            cur = curves[(a, b)] + center[None]
            chord = np.linalg.norm(gcart[p1] - gcart[p0]) + 1e-9
            dd = np.linalg.norm(pts[:, None, :] - cur[None, :, :], axis=2).min(1)
            curve_errs.append(float(dd.max() / chord))

    ve = np.concatenate(vert_errs)
    print(f"\n== E2E-Ergebnis ({len(ev_ids)} held-out Meshes) ==")
    print(f"S1 vert-err%(diag):   med {100*np.median(ve):.2f}  p90 {100*np.percentile(ve,90):.2f}")
    print(f"S2 face-exact:        {face_exact/n_faces:.3f}  ({face_exact}/{n_faces})")
    print(f"S2 quads-distinct:    {quads_distinct/n_faces:.3f}")
    if curve_errs:
        ce = np.array(curve_errs)
        print(f"S3 curve-err% (top-exakt): med {100*np.median(ce):.2f}  p90 {100*np.percentile(ce,90):.2f}")

    # Galerie
    print("== Galerie ==")
    gids = ev_ids[:args.gallery]
    fig, axes = plt.subplots(2, len(gids), figsize=(3 * len(gids), 6))
    for c, i in enumerate(gids):
        d = six[i]
        cart, faces_new, curves = chain_one(vmodel, pmodel, gmodel, ex_v[i]['pts'],
                                            tok, START, device)
        draw_gt(axes[0, c], d); axes[0, c].set_title(f"GT #{i}", fontsize=9)
        draw_gen(axes[1, c], cart, d['center'].numpy(), faces_new, curves)
        axes[1, c].set_title("E2E generiert", fontsize=9)
    axes[0, 0].set_ylabel("GT", fontsize=11); axes[1, 0].set_ylabel("generiert", fontsize=11)
    fig.suptitle("Plan B end-to-end: Punktwolke -> S1 Vertices -> S2 Faces -> S3 Geometrie",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
