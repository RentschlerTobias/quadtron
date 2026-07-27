# 04 — Rating & Roadmap

> Bewertung der sechs Pläne aus [03_plans_and_feasibility.md](03_plans_and_feasibility.md) gegen die
> aus der Diagnose ([01](01_current_model_and_diagnosis.md)) abgeleiteten Kriterien.

## Kriterien & Gewichte

| # | Kriterium | Gewicht | Begründung |
|---|---|---|---|
| K1 | **Generalisierung bei kleinen Daten** | 0.30 | Der reale Baseline-Killer (Overfit ppl 2.8→17.8) |
| K2 | **Gültigkeit per Konstruktion** | 0.25 | Baseline generiert unrekonstruierbar (leere Panels) |
| K3 | **Higher-order-Fähigkeit** | 0.15 | Explizites Nutzerziel (S-Kurven, echte HO-Kanten) |
| K4 | **Repo-Fit / Umsetzungsrisiko** | 0.20 | Vorhandener Meshtron/TFI/Blade-Stack nutzbar? |
| K5 | **Retrain-/Datenaufwand** | 0.10 | Voller Retrain erlaubt, aber Aufwand zählt |

Skala 1 (schlecht) – 5 (sehr gut). K5: höher = weniger Aufwand.

## Rating-Matrix

| Plan | K1 Gen. | K2 Gültig | K3 HO | K4 Repo/Risiko | K5 Aufwand | **Gewichtet** |
|---|---|---|---|---|---|---|
| **A** Baseline härten | 3 | 4¹ | 2 | 5 | 5 | **3.65** |
| **B** PolyGen 2-Stufen + HO-Kanten | 4 | 5 | 5 | 4 | 3 | **4.35** |
| **C** Link-basiert (QuadLink) | 4 | 4 | 4 | 2 | 2 | **3.30** |
| **D** Layout/Cross-Field (TopGen) | 5 | 4 | 4 | 2 | 2 | **3.55** |
| **E** Unsupervised (MGNet) | 5 | 3 | 3 | 2 | 3 | **3.35** |
| **F** VQ-Codebook (MeshGPT) | 2² | 4 | 3 | 3 | 2 | **2.80** |

¹ nur mit Constrained-Decoding-Nachrüstung. ² Codebook-Collapse bei 100 Meshes.

Berechnung Beispiel B: 0.30·4 + 0.25·5 + 0.15·5 + 0.20·4 + 0.10·3 = 1.2+1.25+0.75+0.8+0.3 = **4.35**.

## Ranking

1. **Plan B — PolyGen-artige Zwei-Stufen (4.35)** — bester Kompromiss: löst Gültigkeit **strukturell**
   (Pointer-Faces), liefert echtes higher-order (Kontrollpunkt-Kopf), nutzt den vorhandenen
   Transformer-/TFI-/Blade-Stack.
2. **Plan A — Baseline härten (3.65)** — geringstes Risiko, schnellster Nutzen; ideal **parallel/zuerst**
   als Datengrundlage und Validitäts-Fix.
3. **Plan D — Layout/Cross-Field (3.55)** — beste Generalisierung, aber höchstes Umsetzungsrisiko →
   Forschungsspur.
4. Plan E (3.35), Plan C (3.30), Plan F (2.80).

## Update (2026-07-27): 10.014 Meshes verfügbar

Der Datenengpass ist real gelöst: `quad_domain_data/` enthält **10.014 Meshes** (statt 100). Damit:
- **Plan A** wird stark aufgewertet (K1 Generalisierung 3→4½): oft reicht **nur** „auf vollen 10k neu
  trainieren", um den Overfit-Gap zu schließen — Augmentation optional statt zwingend.
- **Plan D** (Cross-Field/Layout) wird billiger: `frame_field_*` und `singularities` liegen **schon in
  den Daten** → kein teurer Feld-Solver nötig, nur Lernen darauf. K4-Risiko sinkt.
- **Plan F** (VQ-Codebook) wird erstmals realistisch (10k reichen gegen Codebook-Collapse).

## Empfehlung (kombiniert, nicht exklusiv)

**Phase 0 (sofort, ~Stunden) — Baseline auf vollen 10k neu trainieren**
- `domain_extractor.py` über alle 102 `quad_domain_data/*.pt` → ein `domain_data_10k.pt`.
- Bestehende `MeshtronDomain`-Config unverändert trainieren. **Erwartung:** val-ppl fällt deutlich
  (Overfit war Datenmangel). Das ist die neue, ehrliche Baseline — **erst danach** Architektur ändern.
- **Messbar:** val_bpt/ppl vs. altes (4.15 / 17.8); train-val-Gap.

**Phase 1 (~1–2 Wo) — Plan A härten**
- Constrained Decoding in den Sampler (`inference_domain.py`/`policy.py`) → keine leeren Panels mehr.
- Winkel relativ zur Sehne. Blade-Prior über `blade_inject.py`.
- Augmentation nur falls Phase 0 noch Generalisierungslücken zeigt.
- **Messbar:** Gen-Rekonstruktionsrate von ~0 % auf >90 %.

**Phase 2 (~4 Wo) — Plan B als Ziel-Architektur**
- Vertex-Modell + Pointer-Face-Modell + Kontrollpunkt-Kopf (higher-order).
- Reuse: `PerceiverPointEncoder`, `HourglassTransformer`, `reconstruct_domain_coons`, `blade_inject`.
- **Messbar:** 100 % gültige Topologie per Konstruktion; HO-Kantenfehler (Sagitta) < Baseline;
  Generalisierung auf Hold-out-Blades.

**Phase 3 (optional, Forschung) — Plan D**
- Wenn Generalisierung auf stark neue Topologien (andere Blade-Klassen) nötig: Cross-Field/Layout
  als vorgeschaltete Topologie-Wahl.

## Sofort-Nächstschritte (konkret, im Repo)

1. **10k extrahieren:** `domain_extractor.py` über alle `quad_domain_data/*.pt` → `domain_data_10k.pt`
   (evtl. sharded/streamed wegen 5.2 GB), dann Phase-0-Retrain. (Datengenerator-Audit nur noch, falls
   zusätzliche Augmentation gewünscht.)
2. **Constrained Sampler** in `policy.py`: Zustandsautomat über Token-Rollen (Place hat 8 Tokens;
   Face-Tokens nur gültige Place-Indizes). Verhindert invalide Sequenzen ohne Retrain.
3. **Baseline-Reproduktion sauber verifizieren:** `inference_domain.py` auf aktuelle
   `reconstruct_domain_coons` umstellen (die alten Inferenz-Bilder nutzten die veraltete
   `reconstruct_domain`) und die Gen-Rekonstruktionsrate als harte Metrik loggen.

## Offene Entscheidungen für den Nutzer

- **Datenquelle:** Steht der klassische Multiblock-Generator headless/scriptbar zur Verfügung?
  (Bestimmt Machbarkeit von Phase 1.)
- **HO-Repräsentation:** kubischer Bézier (k=2) vs. rationaler NURBS pro Kante — Genauigkeit vs.
  Komplexität.
- **Vertex-Koordinaten:** polar (aktuell, gut für O-/C-Grid um Zentrum) beibehalten oder kartesisch?
