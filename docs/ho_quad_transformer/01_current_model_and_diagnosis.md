# 01 — Aktuelles Modell `MeshtronDomain` und Diagnose

> Ziel dieses Dokuments: exakt festhalten, **was schon existiert**, **wie es funktioniert**, und
> **was die tatsächlichen Ergebnisse** eines bereits durchgeführten Trainings sind. Das ist die
> Baseline, gegen die alle Pläne in [03_plans_and_feasibility.md](03_plans_and_feasibility.md) bewertet werden.

## TL;DR

- Es existiert **bereits** ein punktwolken-konditionierter Transformer für genau diese Aufgabe:
  `MeshtronDomain` (`meshtron_domain.py`).
- Er **wurde trainiert** (`runs_domain/e23bf276/best.pt`, 100 Epochen, ~7 min). Die Annahme
  „nie trainiert" ist falsch — es gibt Checkpoint, Metriken und Inferenz-Bilder.
- **Ergebnis: starkes Overfitting und invalide Generierung.**
  - `train_bpt = 1.47` vs. `val_bpt = 4.15` (ppl **2.8 vs. 17.8**) → memoriert die ~80 Trainings-Meshes.
  - Inferenz-Bilder (`runs_domain/e23bf276/inference/mesh_*.png`): rechtes „Generated"-Panel ist
    **leer**, weil die generierte Tokensequenz nicht zu einer gültigen Block-Struktur
    rekonstruiert (`reconstruct_domain` wirft, wird in `inference_domain.py` gefangen).
- **Kernproblem ist nicht die Architektur, sondern (a) Datenmenge (~100 Meshes) und (b) fehlende
  strukturelle Gültigkeitsgarantie der Tokenisierung.** Das priorisiert die Pläne.

## Architektur (Ist-Zustand)

Pipeline: Punktwolke → Konditionierung → autoregressiver Decoder → Tokens → Rekonstruktion → TFI.

```
tri_coordinates [N,3]           face_count
      │                              │
PerceiverPointEncoder          FaceCountEncoder
 (Fourier-Feats + Cross/       (Embedding 1..50)
  Self-Attn → n_latents)            │
      └──────────── concat ─────────┘
                    │  latent_condition
              HourglassTransformer  ◄── Token-Embeddings (DomainEmbedding, split-vocab)
                    │
               output_head (Linear → vocab)
                    │  autoregressiv
        Token-Sequenz (8 Tokens / Vertex-Place)
                    │  tokenizer.detokenize
     vertex_places (r,θ) + Hermite-Tangenten + faces_as_places
                    │  reconstruct_domain_coons
        Block-Kanten (quadr. Bézier) + per-Block Coons-TFI
```

Belege im Code:
- `meshtron_domain.py:85` `forward(input_tokens, point_cloud, face_count)` — Konditionierung per
  `torch.cat([latent_points, latent_faces_seq])`.
- `point_encoder.py:19` `PerceiverPointEncoder` — Fourier-Features (`n_freqs=8`) + eine
  Cross-Attention (Latents attenden auf Punkte) + eine Self-Attention. Set-invariant, gut für
  Punktwolken, **aber** komprimiert auf feste `n_latents` Latents (Detailverlust bei scharfer
  Blade-Kontur).
- `domain_extractor.py` — erzeugt `vertices_polar (r,θ)`, `edge_tangents (α_start, t_norm, α_end,
  t_norm)`, `faces`, `tri_coordinates`, `center`. Das ist die Tokenizer-Eingabe.
- Tokenisierung: **8 Tokens pro Vertex-Place** (polar r,θ + Hermite: t_norm, α_in, α_out mit
  sin/cos-Kodierung), Faces als Folgen von Places. Details siehe `tokenizer_domain.py`.

## Trainingsergebnis (`runs_domain/e23bf276`)

Konfiguration (aus `config.json`):
`qr=256, qa=128, d_model=256, n_latents=256, stage_layers=(2,4,6,8,10), n_heads=4,
n_sample_points=1500, batch=8, 100 Epochen, bf16`. Datensatz: `domain_data.pt` (100 Meshes,
80/20 Split).

| Metrik | Wert | Deutung |
|---|---|---|
| final `train_bpt` | 1.47 (ppl 2.77) | Modell passt Trainingsdaten fast perfekt |
| best `val_bpt` | 4.15 (ppl 17.77) | Validierung plateauisiert früh, **~10× schlechter** |
| best_epoch | 99 | kein echtes Optimum — val bewegt sich kaum |
| Gen-Rekonstruktion | **fehlgeschlagen** | rechte Panels leer, Sequenz nicht dekodierbar |

**Diagnose:**
1. **Overfitting durch Datenmangel.** ~80 Trainings-Meshes bei ~5 M Parametern. Die val-Kurve ist
   quasi flach ab Epoche ~5 — das Modell lernt keine generalisierende Abbildung Punktwolke→Blocking,
   sondern memoriert.
2. **Keine Gültigkeitsgarantie.** Die Tokenisierung kodiert Koordinaten frei; ein Sampling-Fehler
   (falsche Token-Anzahl pro Place, Face verweist auf nicht-existente Places, ungerade Sequenz)
   macht `detokenize`/`reconstruct` ungültig → leeres Ergebnis. Bei Teacher-Forcing-Training ohne
   strukturelle Constraints ist das der Normalfall bei kleinen Daten.
3. **Konditionierungs-Engpass.** Perceiver komprimiert die ganze Punktwolke in `n_latents`
   Vektoren; die scharfe Blade-Kontur (die die Blockstruktur bestimmt) kann verwaschen.
4. **Higher-order-Grenze.** Aktuell 1 quadratische Bézier pro Kante (kein Wendepunkt/S-Kurve) —
   siehe `[[meshtron-known-bugs]]`. Für „higher order" Ziel strukturell limitiert.

## Update (2026-07-27): Volldatensatz verfügbar — 10.014 Meshes

Das Overfitting-Training nutzte **nur `domain_data.pt` (100 Meshes)**. Es existiert jetzt der
**volle Datensatz**: `quad_domain_data/` (102 Dateien `checkpoint_mesh_*.pt`, **10.014 Meshes**, 5.2 GB
entpackt). Das ist **~100× mehr** — der wichtigste Engpass (K1) ist damit weitgehend gelöst, ohne dass
synthetische Augmentation zwingend nötig wäre.

**Bonus:** jeder Mesh trägt bereits reiche Felder:
`frame_field_angle`, `frame_field_u`, `singularities`, `singularities_coords`, `streamlines`,
`quad_coordinates`, `quad_faces`, `blocking_nodes`, `blocking_faces`, `edge_to_streamline`,
`tri_coordinates`. → **Cross-Field und Singularitäten sind schon vorhanden** — Plan D
(Layout/Cross-Field) braucht die teure Feld-Berechnung nicht mehr, nur noch das Lernen darauf.

Format = **roh** (wie `checkpoint_mesh_100.pt`), also muss `domain_extractor.py` über alle 102 Dateien
laufen, um die Tokenizer-Eingabe (`vertices_polar`, `edge_tangents`, …) für alle 10k zu erzeugen.

**Konsequenz:** Erste Maßnahme ist nicht mehr „Daten synthetisieren", sondern schlicht
**Baseline auf vollen 10k neu trainieren** — das allein sollte den Overfit-Gap massiv schließen.

## Konsequenz für die Pläne

Die Baseline zeigt: **das Paradigma (Punktwolke → autoregressive Block-Tokens → TFI) funktioniert
technisch, scheitert aber an Daten + Validität.** Daher bewerten wir Pläne primär nach:

1. **Generalisierung bei kleinen Datenmengen** (Data-Augmentation, Symmetrie, Induktive Bias).
2. **Gültigkeit-per-Konstruktion** (kann das Modell überhaupt eine kaputte Struktur erzeugen?).
3. **Higher-order-Fähigkeit** (Kontrollpunkte/Grad pro Blockkante).
4. **Repo-Fit / Retrain-Kosten** (voller Retrain ist laut Nutzer OK).

Siehe [04_rating_and_roadmap.md](04_rating_and_roadmap.md).
