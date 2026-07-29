"""
vertex_head_prototype.py

STUFE 1: Punktwolken-Konditionierung -> Vertices ERZEUGEN (der eigentliche
generative Teil; Stufe 2/3 nehmen Vertices als gegeben an). Konditioniert auf die
Punktwolke tri_coordinates [Np,3] (x, y in [0,1], Label {0=Singularitaet/Ecke,
1=Rand, 2=Feld}) generiert das Modell autoregressiv den Vertex-Satz als polar-
Tokens (r, sin theta, cos theta), kanonische Reihenfolge lexsort(theta, r).

Befund (siehe Analyse): die 6 Label-0-Punkte SIND exakt 6 der 12 Vertices, die
restlichen 6 liegen nahe Rand (<=0.021). -> Stufe 1 = Selektion + Feinregression
aus der Punktwolke, nicht Generierung aus dem Nichts.

Prototyp: nur 6-Face-Basis (M=12 fix). Punktwolken-Encoder (bidirektional) +
autoregressiver Decoder (kausal, Cross-Attn zur Punktwolke). Gueltigkeit: Logits
je Decoder-Position auf die richtige Token-Gruppe (r / sin / cos) maskiert.

  ~/Environments/meshtron/bin/python vertex_head_prototype.py \
      [--data domain_data_smoke.pt] [--d-model 256] [--batch 32] [--epochs 40]
"""

import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_twostage import TwoStageTokenizer


# ------------------------------------------------------------------
# Daten
# ------------------------------------------------------------------
def build_vertex_examples(data, tok):
    """Pro 6F-Mesh: pts [Np,3], seq [3M+1] (r,ts,tc)*M + stop, verts_cart [M,2],
    center [2] (fuer Eval)."""
    Qr, Qa = tok.Qr, tok.Qa
    off_ts, off_tc = tok.off_ts, tok.off_tc
    STOP = Qr + 2 * Qa + 1
    examples = []
    for d in data:
        if d['faces'].shape[1] != 6:
            continue
        vp = d['vertices_polar'].numpy()
        order = tok._sort_order(d['vertices_polar'])
        seq = []
        for oi in order:
            r, th = float(vp[oi, 0]), float(vp[oi, 1])
            rt = tok._q_scalar(r, tok.R_MIN, tok.R_MAX)
            ts, tc = tok._q_angle(th)
            seq += [rt, ts + off_ts, tc + off_tc]
        seq.append(STOP)
        examples.append({
            'pts': torch.tensor(d['tri_coordinates'].numpy(), dtype=torch.float32),
            'seq': torch.tensor(seq, dtype=torch.long),
            'vc': d['vertices_cartesian'].numpy()[order],
            'center': d['center'].numpy()})
    return examples


# ------------------------------------------------------------------
# Modell
# ------------------------------------------------------------------
class VertexGen(nn.Module):
    def __init__(self, vocab, d=256, heads=8, n_enc=4, n_dec=4, max_len=64, start_id=0):
        super().__init__()
        self.start_id = start_id
        self.pt_proj = nn.Linear(3, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, n_dec)
        self.head = nn.Linear(d, vocab)

    def encode(self, pts, ppad):
        return self.encoder(self.pt_proj(pts), src_key_padding_mask=ppad)

    def decode(self, mem, ppad, dec_in):
        L = dec_in.size(1)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=dec_in.device), 1)
        h = self.tok_emb(dec_in) + self.pos(torch.arange(L, device=dec_in.device))[None]
        out = self.decoder(h, mem, tgt_mask=causal, memory_key_padding_mask=ppad)
        return self.head(out)

    def forward(self, pts, ppad, dec_in):
        return self.decode(self.encode(pts, ppad), ppad, dec_in)


# ------------------------------------------------------------------
# Gruppen-Maske: Decoder-Position -> erlaubte Token-Gruppe
# ------------------------------------------------------------------
def group_mask(logits, pos0, Qr, Qa, stop_id):
    """logits [.,V] an absoluter Sequenzposition pos0 (0-basiert, VOR dem Ziel):
    Ziel-Token an Position p ist Gruppe (p mod 3): 0=r, 1=sin, 2=cos; nach 3M=36 -> stop.
    Setzt verbotene Logits auf -inf. pos0 = Index des ZIELS."""
    V = logits.shape[-1]
    mask = torch.full((V,), float('-inf'), device=logits.device)
    if pos0 >= 36:
        mask[stop_id] = 0.0
    else:
        g = pos0 % 3
        if g == 0:
            mask[0:Qr] = 0.0
        elif g == 1:
            mask[Qr:Qr + Qa] = 0.0
        else:
            mask[Qr + Qa:Qr + 2 * Qa] = 0.0
    return logits + mask


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------
def collate(batch, start_id):
    Npmax = max(e['pts'].shape[0] for e in batch)
    B = len(batch)
    pts = torch.zeros(B, Npmax, 3); ppad = torch.ones(B, Npmax, dtype=torch.bool)
    seqs = torch.stack([e['seq'] for e in batch])           # [B, 3M+1] (6F: fix 37)
    for i, e in enumerate(batch):
        Np = e['pts'].shape[0]
        pts[i, :Np] = e['pts']; ppad[i, :Np] = False
    # Decoder-Input = [start] + seq[:-1]; Ziel = seq
    din = torch.cat([torch.full((B, 1), start_id, dtype=torch.long), seqs[:, :-1]], 1)
    return pts, ppad, din, seqs


def run_epoch(model, order, examples, bs, opt, sched, clip, device, start_id, train=True):
    model.train() if train else model.eval()
    tot = cnt = correct = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for s in range(0, len(order), bs):
            batch = [examples[i] for i in order[s:s + bs]]
            pts, ppad, din, tgt = collate(batch, start_id)
            pts, ppad, din, tgt = pts.to(device), ppad.to(device), din.to(device), tgt.to(device)
            logits = model(pts, ppad, din)                  # [B,L,V]
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            with torch.no_grad():
                pred = logits.argmax(-1)
                correct += float((pred == tgt).sum()); cnt += tgt.numel()
                tot += loss.item() * tgt.numel()
    return tot / cnt, correct / cnt


@torch.no_grad()
def vertex_eval(model, examples, ids, device, tok, start_id):
    """Free-Run: 36 Vertex-Tokens generieren (Gruppen-maskiert), dequantisieren,
    index-weise mit GT vergleichen. Fehler = L2 / Einheitsbox-Diagonale (sqrt2)."""
    model.eval()
    Qr, Qa = tok.Qr, tok.Qa
    STOP = Qr + 2 * Qa + 1
    errs = []
    for i in ids:
        e = examples[i]
        pts = e['pts'].unsqueeze(0).to(device)
        ppad = torch.zeros(1, pts.size(1), dtype=torch.bool, device=device)
        mem = model.encode(pts, ppad)
        seq = [start_id]
        for step in range(36):
            din = torch.tensor(seq, device=device)[None]
            logits = model.decode(mem, ppad, din)[0, -1]
            logits = group_mask(logits, step, Qr, Qa, STOP)
            seq.append(int(logits.argmax()))
        gen = seq[1:]                                        # 36 Tokens
        rec = []
        for m in range(12):
            rt, tsk, tck = gen[3 * m], gen[3 * m + 1] - Qr, gen[3 * m + 2] - (Qr + Qa)
            r = tok._dq_scalar(rt, tok.R_MIN, tok.R_MAX)
            th = tok._dq_angle(tsk, tck)
            rec.append([r * math.cos(th), r * math.sin(th)])
        rec = np.array(rec) + e['center'][None]
        d = np.linalg.norm(rec - e['vc'], axis=1) / math.sqrt(2.0)
        errs.append(d)
    return np.concatenate(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_smoke.pt')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-enc', type=int, default=4)
    ap.add_argument('--n-dec', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--ve-every', type=int, default=5)
    ap.add_argument('--ve-n', type=int, default=80)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = TwoStageTokenizer(repr_mode='cubic_bezier')
    Qr, Qa = tok.Qr, tok.Qa
    START = Qr + 2 * Qa            # start
    vocab = Qr + 2 * Qa + 2        # + start + stop

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    if args.limit:
        data = data[:args.limit]
    print("baue Vertex-Beispiele (nur 6F) ...")
    t0 = time.time()
    examples = build_vertex_examples(data, tok)
    Nps = [e['pts'].shape[0] for e in examples]
    print(f"{len(examples)} 6F-Beispiele in {time.time()-t0:.0f}s  |  Np "
          f"{min(Nps)}..{max(Nps)}  |  vocab {vocab}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    n_val = max(1, int(args.val_frac * len(examples)))
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    model = VertexGen(vocab, d=args.d_model, n_enc=args.n_enc, n_dec=args.n_dec,
                      start_id=START).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"VertexGen d={args.d_model} enc={args.n_enc} dec={args.n_dec}  "
          f"Params {n_par/1e6:.2f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    spe = math.ceil(len(train_ids) / args.batch)
    total = args.epochs * spe; warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    ve_ids = val_ids[:min(args.ve_n, len(val_ids))]
    tg = time.time()
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, rng.permutation(train_ids), examples, args.batch,
                             opt, sched, args.clip, device, START, train=True)
        vll, vla = run_epoch(model, val_ids, examples, args.batch, None, None, 0,
                             device, START, train=False)
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        if ep % args.ve_every == 0 or ep == args.epochs:
            er = vertex_eval(model, examples, ve_ids, device, tok, START)
            ves = (f"vert-err%(diag) med {100*np.median(er):.2f} p90 {100*np.percentile(er,90):.2f} "
                   f"max {100*er.max():.2f}")
        else:
            ves = "vert-err% -"
        print(f"ep {ep:3d}  tr-loss {trl:.4f} tr-tokacc {tra:.3f}  |  "
              f"val-loss {vll:.4f} val-tokacc {vla:.3f}  {ves}  "
              f"lr {sched.get_last_lr()[0]:.1e} peakVRAM {vram:.2f}GB")
    print(f"\nfertig in {time.time()-tg:.0f}s.")


if __name__ == '__main__':
    main()
