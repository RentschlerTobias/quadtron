# HANDOFF — Zwei-Stufen Quad-Block-Generator (kontextfreie Übergabe)

> Stand: 2026-07-27. Diese Datei ist so geschrieben, dass eine KI **ohne jeden vorherigen
> Kontext** an genau dieser Stelle weiterarbeiten kann. Alles Nötige steht hier oder ist verlinkt.

---

## 1. Projektziel (in einem Satz)

Ein Transformer soll aus einer **2D-Punktwolke** (Rand-Knoten *oder* trianguliertes Mesh) eine
**quadrilaterale Blockstruktur** (Faces + higher-order Blockkanten) erzeugen, die per
**Transfinite/Coons-Interpolation (TFI)** zum CFD-Netz um ein Blade/Profil wird. Konditionierung:
Punktwolke + optional **FaceCount** (Anzahl Blöcke). Das Blade-Loch bleibt hohl; die Blade-Geometrie
ist bekannt und wird nachträglich exakt re-injiziert (`blade_inject.py`).

## 2. Umgebung / Fakten

- **Repo:** `/home/t1dde/Duty/projects/meshtron`, Branch `blade-known-reinsertion`.
- **Python/venv:** `~/Environments/meshtron/bin/python` (torch 2.11.0+cu128, **kein scipy**).
- **Lokale GPU:** RTX 4060 8 GB (nur Smoke-Tests; volles Training braucht Blackwell 48 GB / Cluster).
- **Daten:**
  - `domain_data_10k.pt` (900 MB, **10.014 Meshes**, vorverarbeitet) — der echte Trainings-/Testdatensatz.
  - `domain_data.pt` (8 MB, 100 Meshes) — schneller Sanity-Datensatz.
  - `quad_domain_data/` (roh, 102 `checkpoint_mesh_*.pt`) — Quelle; `.gitignore`d.
  - **Alle `*.pt` sind git-ignored** (zu groß, nie committen).
- **Commit-Regel (HART):** NIEMALS Claude/Assistant in Commits erwähnen — kein `Co-Authored-By`,
  kein „Generated with". `~/.claude/settings.json` hat `includeCoAuthoredBy: false`.

## 3. Datenstruktur eines Meshes (Dict, aus `domain_extractor.py`)

```
vertices_cartesian : [M,2]        Blockecken kartesisch
vertices_polar     : [M,2]        (r, theta) relativ zu center; theta in [0,2pi)
faces              : [4, F]       GLOBALE Vertex-Indizes in [0, M-1]  <-- schon dedupliziert!
edge_index         : [2, E]       gerichtete Kanten (u,v)
edge_tangents      : [E, 4]       [alpha_start, t_norm_start, alpha_end, t_norm_end]
edge_to_streamline : dict{(u,v): ndarray[N,2]}   Original-Kurvenpunkte (zur Rekonstruktion)
tri_coordinates    : [N,3]        Punktwolke fuer Konditionierung; Spalte 2 = Boundary-Flag
center             : [2]          Referenzpunkt der Polarkoordinaten
bounds             : [xmin,ymin,xmax,ymax]
```
Typische Größen: M ≈ 12–50 Ecken, F ≈ 6–~40 Faces. **`faces` sind globale Indizes** — das ist der
Schlüssel, der die Zwei-Stufen-Idee trivial macht.

## 4. Kernproblem des Alt-Modells (WARUM die Zwei-Stufen-Idee)

`MeshtronDomain` (`meshtron_domain.py`) existiert und **wurde trainiert** (`runs_domain/e23bf276`),
Ergebnis: **Overfit** (train ppl 2.8 vs val 17.8) + **invalide Generierung** (leere Inferenz-Panels).
Zwei Ursachen:
1. **Datenmangel** — trainiert auf nur 100 Meshes. → gelöst: jetzt 10.014.
2. **Faces sind *implizit*** im alten `DomainTokenizer` (`tokenizer_domain.py`): 1 Quad = 4
   aufeinanderfolgende „Vertex-Plätze" à 8 Tokens; `detokenize` macht `n_faces = n_places // 4`.
   Geteilte Ecken werden **dupliziert** (jede mit eigenem Quant-Fehler) → Blöcke driften, keine
   Konnektivität als Signal → invalide/leere Rekonstruktion.

Die Konditionierung selbst ist **gesund** und bleibt: `PerceiverPointEncoder` (Punktwolke →
`n_latents` Latents) + `FaceCountEncoder` (sinusoidal) → in `MeshtronDomain.forward` zu
`latent_condition` konkateniert, per Cross-Attention in `HourglassTransformer`.

## 5. Zielarchitektur (Plan B, Rating 4.35 — bester Plan)

Trenne **Topologie** von **Geometrie**, drei Stufen, geteilte Konditionierung:

```
Punktwolke + FaceCount --> latent_condition (bestehende Encoder, unveraendert)
  Stufe 1  Vertex-Modell:   AR ueber EINDEUTIGE Blockecken (r, theta)
  Stufe 2  Face-Pointer:     jedes Quad = 4 ZEIGER auf Stufe-1-Vertices (PolyGen-Pointer-Softmax)
                             -> Gueltigkeit per Konstruktion, keine ungueltigen Indizes moeglich
  Stufe 3  HO-Kantenkopf:    pro GERICHTETER Blockkante Tangenten/Kontrollpunkte (einmal -> buendig)
  --> reconstruct_domain_coons + blade_inject
```
Details/Begründung: `docs/ho_quad_transformer/05_face_block_generator.md`. Studie gesamt: `README.md`
+ `01`–`04` im selben Ordner.

## 6. WAS BEREITS ERLEDIGT IST (Stand jetzt)

### 6a. Konzept-Doc
`docs/ho_quad_transformer/05_face_block_generator.md` — vollständiges Fokus-Konzept für den
Face-/Block-Generator, auf realen Code gemappt.

### 6b. Prototyp Stufe 1+2 — VALIDIERT ✅
Datei `prototype_twostage.py`, Klasse `TwoStageTokenizer`. Standalone (Produktions-`DomainTokenizer`
bewusst noch unangetastet).

**Token-Layout (embedding_mode 0 / split ranges):**
```
r          : [0,        Qr-1]                (Qr=512)
theta_sin  : [Qr,       Qr+Qa-1]            (Qa=256)
theta_cos  : [Qr+Qa,    Qr+2Qa-1]
vertex_idx : [Qr+2Qa,   Qr+2Qa+Vmax-1]      Pointer-Ziele fuer Stufe 2
specials   : start, end, sep, stop, pad     (danach)

Sequenz:  [start] (r,ts,tc)*M  [sep]  (i0,i1,i2,i3)*F  [end]
```
Vertices werden lexikografisch nach `(theta, r)` sortiert; `old2new`-Mapping remappt die Face-Indizes.

**Round-trip-Ergebnis (real ausgeführt):**
| Datensatz | Topologie exakt | Seq-Len mean | max kart. Fehler |
|---|---|---|---|
| domain_data.pt (100) | 100/100 | 75 | 3.6e-3 |
| **domain_data_10k.pt (10.014)** | **10.014/10.014** | **81** | 4.4e-3 |

Topologie ist **Integer-exakt** (Pointer-Quadrupel matchen nach Remapping). Geometriefehler ist reine
Quantisierung (mit höherem Qr/Qa reduzierbar). Sequenzen ~60 % kürzer als implizit.

**Ausführen:**
```
~/Environments/meshtron/bin/python prototype_twostage.py --data domain_data.pt
~/Environments/meshtron/bin/python prototype_twostage.py --data domain_data_10k.pt
# Flags: --n <limit>  --qr 512  --qa 256
```

## 7. WAS ALS NÄCHSTES ANSTEHT (in Reihenfolge)

1. **Stufe 3 — HO-Kantengeometrie** (unmittelbarer nächster Schritt, im Prototyp):
   - Aus den Faces (Stufe 2) die **gerichteten Kanten** ableiten; jede physische Kante **einmal**
     mit `(t_norm, alpha_in, alpha_out)` aus `edge_tangents` kodieren (sincos für Winkel, wie im
     alten Tokenizer `_encode_vertex_place`). Round-trip **gegen `edge_tangents`** verifizieren.
   - Alternative HO-Repr: k=2 Bézier-Kontrollpunkte (S-Kurven; behebt Wendepunkt-Limit, siehe
     Memory `meshtron-known-bugs`). Start mit Hermite (erprobt).
2. **Integration in `DomainTokenizer`** als `tokenize_twostage` / `detokenize_twostage` (+ `sep_token`
   existiert schon), mit Round-trip-Test analog `testing()`.
3. **Pointer-Kopf im Modell** (`meshtron_domain.py`): Query = Decoder-State, Keys =
   Stufe-1-Vertex-Embeddings; Pointer-Softmax über `M+1` (inkl. stop). Zweiphasiger `generate`.
4. **Constrained Decoding** in `policy.py`/`inference_domain.py`: Pointer-Maske (nur gültige
   Vertex-Indizes), zweiphasig (Vertices bis `end`, Faces bis `stop`).
5. **Metriken:** Gen-Rekonstruktionsrate (Ziel >90 %), Bündigkeit geteilter Kanten (Sagitta-Diff),
   HO-Kantenfehler.

**Roadmap-Kontext** (`04_rating_and_roadmap.md`): Phase 0 = Alt-Baseline auf 10k neu trainieren als
ehrliche Messlatte (`run_domain_10k.py`, auf Blackwell — **Nutzer startet das**, nicht die KI). Phase 1
= Constrained Decoding. Phase 2 = dieses Zwei-Stufen-Modell.

## 8. Wichtige Dateien (Landkarte)

| Datei | Rolle |
|---|---|
| `prototype_twostage.py` | **Hier weitermachen.** Zwei-Stufen-Tokenizer-Prototyp (Stufe 1+2 fertig) |
| `docs/ho_quad_transformer/05_face_block_generator.md` | Konzept des Face-Generators |
| `docs/ho_quad_transformer/{README,01..04}.md` | Studie, Diagnose, Survey, Pläne, Rating |
| `domain_extractor.py` | Roh-Mesh → Dict (Feldstruktur §3) |
| `tokenizer_domain.py` | Alt-Tokenizer (implizite Faces) — Quant-Helfer wiederverwendbar |
| `meshtron_domain.py` | Modell (Encoder + Hourglass + output_head) — Pointer-Kopf hier ergänzen |
| `point_encoder.py` / `faceCount_encoder.py` | Konditionierung (bleibt) |
| `reconstruct_domain*.py` / `blade_inject.py` | TFI-Backend + Blade-Re-Injektion |
| `run_domain_10k.py` / `domain_trainer.py` / `config.py` | Training (Nutzer startet auf Blackwell) |

## 9. Fallstricke

- `vertices_polar` ist **float64**; `theta` in `[0,2pi)`, aber `_dq_angle` (atan2) liefert `[-pi,pi]`
  → Winkelvergleiche wrap-sicher machen (`_ang_diff` im Prototyp).
- `faces` sind globale Indizes — **nicht** die impliziten 4er-Blöcke des Alt-Tokenizers verwechseln.
- Volles Training NICHT lokal (8 GB OOM bei dichter O(seq²)-Attention × Layer). Nur Smoke lokal.
- Keine Selbstnennung in Commits (§2). `*.pt` nie committen.
