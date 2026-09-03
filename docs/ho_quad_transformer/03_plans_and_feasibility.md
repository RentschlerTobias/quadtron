# 03 — Pläne & Durchführbarkeit

> Sechs konkrete Pläne, jeweils auf **unser** Problem gemappt: 2D, Punktwolke (Rand **oder**
> trianguliert) → **Block-Struktur** (Ecken + higher-order Blockkanten + Quad-Topologie) → Coons/TFI,
> Blade-Loch bleibt hohl, Blade-Geometrie bekannt (re-injizierbar). Voller Retrain ist erlaubt.
> Bewertung/Rating in [04_rating_and_roadmap.md](04_rating_and_roadmap.md).

Gemeinsame Randbedingungen für alle Pläne:
- **Output-Ziel:** Block-Struktur für Transfinite/Coons (nicht direktes Element-Mesh). Die
  bestehende `reconstruct_domain_coons` / `blade_inject.py` bleibt als Backend nutzbar.
- **Konditionierung:** Punktwolke via `PerceiverPointEncoder` (oder Alternative), optional
  face_count. Blade-Kontur als Teil-Input markierbar (Boundary-Flag ist schon in
  `tri_coordinates[:,2]`).
- **Daten:** aktuell 100 Meshes → Datenstrategie ist Teil jedes Plans.

---

## Plan A — Baseline härten (Daten + Validität + Blade-Prior)

**Idee:** Architektur behalten (`MeshtronDomain`), die zwei realen Defekte beheben: Overfitting und
Ungültigkeit. Kein neues Paradigma.

**Maßnahmen:**
1. **Datenmenge ×10–×50** durch Augmentation, die die Blocktopologie invariant lässt:
   Rotation, Spiegelung, Skalierung, Chord-Reparametrisierung, kleine Blade-Morphs. Ideal: den
   **klassischen parametrischen Multiblock-Generator** (vgl. MDPI-Airfoil-Paper) anzapfen, um
   beliebig viele (Punktwolke, Blocking)-Paare zu erzeugen.
2. **Constrained Decoding / Grammatik:** beim Sampling erzwingen, dass (a) Token-Anzahl pro Place
   stimmt, (b) Faces nur auf existierende Places zeigen, (c) Sequenz wohlgeformt endet. Verhindert
   die „leeres Panel"-Fehler ohne Retrain-Änderung.
3. **Winkel relativ zur Sehne** kodieren (gerade Kante = 0 exakt) — reduziert Quantisierungsfehler,
   siehe `[[meshtron-known-bugs]]` Punkt (3).
4. **Blade-Prior:** Blade-Randknoten aus der Konditionierung als „fixe" Tokens vorgeben
   (Teacher-Forcing-Prefix) oder per `blade_inject.py` post-hoc einsetzen.

**Durchführbarkeit:** Hoch. Keine neuen Module nötig, nur Trainer/Sampler + Daten. Risiko:
Augmentation allein behebt Overfitting evtl. nur teilweise, wenn die 100 Basisformen zu ähnlich sind.

**Aufwand:** ~1–2 Wochen. **Retrain:** ja, aber gleiche Pipeline.

---

## Plan B — PolyGen-artige Zwei-Stufen (Vertices → Pointer-Faces → Kantengeometrie)

**Idee:** Trenne **Topologie** von **Geometrie** und garantiere Gültigkeit per Konstruktion.

**Stufen:**
1. **Vertex-Modell:** autoregressiv die Blocking-Knoten (quantisiert, polar oder kartesisch),
   konditioniert auf Punktwolke.
2. **Face-Modell (Pointer-Network, PolyGen):** Quads als **Zeiger auf bereits generierte Vertices**.
   → Kann **keine** ungültigen Indizes erzeugen. Löst den Baseline-Fehler direkt.
3. **Kanten-Geometrie-Kopf:** pro Blockkante **k Kontrollpunkte** (Bézier/NURBS) — echtes
   higher-order (S-Kurven möglich), regressiv oder quantisiert. Blade-Kanten werden durch
   `blade_inject.py` exakt ersetzt.

**Durchführbarkeit:** Mittel-hoch. Pointer-Network ist Standard, gut dokumentiert (PolyGen). Passt
zum Repo (Transformer-Stack vorhanden). Datenbedarf moderat, da Topologie-Vokabular klein
(6–~50 Faces).

**Aufwand:** ~3–5 Wochen (zwei Köpfe + neuer Tokenizer). **Retrain:** voll.

---

## Plan C — Link-basiert aus Punktwolke (QuadLink / Point2Quad)

**Idee:** Blocking-Vertices sind (fast) Teilmenge markanter Punktwolken-Punkte. Modell lernt
**Verknüpfungen** statt Koordinaten.

**Ablauf (QuadLink-adaptiert):** (1) Anker vorhersagen = Blocking-Vertices + Blockzentroide;
(2) centroid-conditioned Links (welche Vertices bilden welchen Block); (3) Quads mit geometrischer
Verifikation assemblieren; higher-order-Kanten separat als Kontrollpunkte.

**Durchführbarkeit:** Mittel. Stärke: **kohärenter Edge-Flow**, anisotrope sparse Blöcke — genau der
bei der Baseline verlorene „Verlauf". Schwäche: Blocking-Vertices sind bei uns **nicht** exakt
Randknoten → Stufe (1) muss echte Regression der Ankerpositionen leisten, nicht nur Auswahl.

**Aufwand:** ~4–6 Wochen (drei Stufen + Verifikation, wenig Vorlage im Repo). **Retrain:** voll.

---

## Plan D — Layout / Cross-Field (TopGen-artig)

**Idee:** Erst **Blocktopologie** (C/O/H-Grid um Blade, Singularitäten) als diskretes Layout lernen,
dann Kantengeometrie/TFI. Trennt „welche Topologie" von „wo genau".

**Ablauf:** (1) Layout-/Cross-Field-Netz klassifiziert Topologie-Template + Singularitätenlage aus
Punktwolke; (2) Geometrie-Kopf platziert Ecken + Kontrollpunkte; (3) TFI.

**Durchführbarkeit:** Mittel-niedrig. Konzeptuell ideal für „Block-Struktur" und
datensparsam (Topologie-Templates sind wenige). Aber: Cross-Field-Maschinerie in 2D-CFD-Blocking ist
Forschung, wenig fertige Bausteine, höchstes Umsetzungsrisiko.

**Aufwand:** ~6–8 Wochen. **Retrain:** voll + neue Label (Topologie-Klassen).

---

## Plan E — Unsupervised / physik-informiert (MGNet-artig)

**Idee:** Datenmangel umgehen: **kein** Ground-Truth-Blocking nötig. Netz lernt die
Parametrisierungsabbildung aus **Randkurven**; Loss = Netzqualität (Orthogonalität, Glätte,
PDE-/Winkel-Residuum) statt Token-Matching.

**Durchführbarkeit:** Mittel. Stark bei wenig Labels und guter Generalisierung. Aber: liefert primär
**Netz-/Parametrisierung**, die Ableitung einer diskreten **Blockstruktur** (Ecken/Topologie) daraus
ist zusätzlicher Schritt. Weniger „Transformer-Tokenizer", mehr Optimierungsnetz — passt weniger zum
bestehenden Meshtron-Stack.

**Aufwand:** ~5–7 Wochen. **Retrain:** ja, anderes Trainingsziel.

---

## Plan F — VQ-Codebook der Blockmotive (MeshGPT-artig)

**Idee:** Ein VQ-VAE lernt ein **Vokabular latenter Blockmotive**; der Transformer generiert nur
Codebook-Indizes. Sehr kurze Sequenzen.

**Durchführbarkeit:** Niedrig **bei aktueller Datenmenge** — VQ-Codebooks brauchen viele Daten, sonst
Codebook-Collapse. Erst sinnvoll, wenn Plan A die Daten bereits ×20+ gebracht hat.

**Aufwand:** ~6–8 Wochen. **Retrain:** voll + VQ-Vortraining.

---

## Querschnitt: Higher-order-Kanten (gilt für B/C/D)

Statt 1 quadratischer Bézier pro Kante: **k Kontrollpunkte pro Blockkante** als eigener Ausgabe-Kanal
(à la BrepGen-Poles). Vorschlag: kubischer/rationaler Bézier (k=2 innere Kontrollpunkte) →
S-Kurven möglich, behebt das Wendepunkt-Limit aus `[[meshtron-known-bugs]]`. Blade-Kanten immer per
`blade_inject.py` exakt überschreiben (Blade ist bekannt).

## Datenstrategie (planübergreifend, kritisch)

Der limitierende Faktor ist Datenmenge. Unabhängig vom Plan empfohlen:
1. **Synthese via klassischem Multiblock-Generator** (MDPI-Ansatz / bestehendes
   `domain_partition`-Tooling) → beliebig viele (Punktwolke, Blocking, higher-order-Kanten)-Paare.
2. **Augmentation** (Rotation/Spiegelung/Skalierung/Blade-Morph) mit topologie-erhaltender Semantik.
3. **Blade-Bibliothek** verbreitern (mehr Profile, mehr Anströmwinkel).
