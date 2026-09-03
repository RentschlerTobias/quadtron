# 02 — Literatur-Survey: Tokenisierung & Generierung von (higher-order) Quad-Strukturen

> Stand der Recherche: Juli 2026. Fokus: Verfahren, die **aus einer Punktwolke / Randkurve** eine
> **Quad-/Block-Struktur** erzeugen, und die zugehörigen **Tokenisierungs-Paradigmen**. Jede Methode
> ist am Ende auf unser Problem gemappt (2D, Blade-Loch, Output = Block-Struktur für Transfinite/Coons).

## Taxonomie

Vier Familien, nach Art des Outputs:

| Familie | Idee | Vertreter |
|---|---|---|
| **A. Autoregressive Token-Sequenz** | Mesh = Sequenz quantisierter Koordinaten-Tokens, Decoder-only Transformer | Meshtron, PolyGen, QuadGPT, EdgeRunner, MeshGPT |
| **B. Link-/Face-Prediction aus Punktwolke** | Punkte gegeben → Modell lernt, welche Punkte zu Faces verbunden werden | QuadLink, Point2Quad |
| **C. Layout / Cross-Field** | Erst singularitäten-/richtungsfeld-basiertes Blocklayout, dann Parametrisierung | TopGen, klassische Cross-Field-Quadrangulierung |
| **D. Feld-/Diffusions-/RL-basiert** | Struktur über gelernte Felder, Diffusion oder sequentielle RL-Entscheidungen | BrepGen (Diffusion, NURBS-Poles), DRL-MeshGen, Dmsh, MGNet |

---

## A. Autoregressive Token-Sequenz (unser aktuelles Paradigma)

**Meshtron** ([arXiv:2412.09548](https://arxiv.org/abs/2412.09548), NVIDIA). Hourglass-Transformer +
Sliding-Window-Attention. Zwei-Ebenen-Hierarchie: je 3 Tokens = Vertex, je 9/12 Tokens = Face
(Quads mit Padding). Shortening/Upsampling zwischen Ebenen. Bis 64k Faces, 1024-Level-Quantisierung.
→ **Genau die Basis unseres Repos.** Wichtig: Meshtron generiert **Elemente**, nicht Block-Struktur;
wir haben es auf Blocking umgewidmet.

**PolyGen** ([arXiv:2002.10880](https://arxiv.org/abs/2002.10880), DeepMind). **Zwei-Stufen**:
(1) Vertex-Modell (Transformer-Decoder über quantisierte Koordinaten), (2) Face-Modell als
**Pointer-Network** — Faces werden als Zeiger auf **bereits generierte** Vertices erzeugt.
Konditionierung auf Klasse/Bild/Voxel via Cross-Attention. NLL ~2.5 bits/vertex, ~85–90 %
Vorhersage-Genauigkeit. → **Schlüsselidee für uns: das Pointer-Face-Modell kann per Konstruktion
keine ungültigen Vertex-Indizes erzeugen** — löst genau unseren „leeres Panel"-Fehler.

**QuadGPT** ([arXiv:2509.21420](https://arxiv.org/abs/2509.21420)). **Native Quadrilaterale**
autoregressiv (keine Triangulierung als Umweg). Behandelt die spezifischen Symmetrien/Constraints von
Quads im Tokenizer. → Direkt relevant: zeigt, dass reine Quad-Erzeugung ohne Tri-Zwischenschritt
tragfähig ist.

**EdgeRunner** ([arXiv:2409.18114](https://arxiv.org/abs/2409.18114)) und **MeshGPT**
([arXiv:2311.15475](https://arxiv.org/abs/2311.15475)). MeshGPT lernt per **VQ-VAE + Graph-Convs**
ein Vokabular latenter Face-Embeddings (Residual-VQ), dann Decoder-only über Token-Indizes; scharfe
Kanten, kompakt. EdgeRunner: autoregressiver Auto-Encoder, effiziente Tokenisierung über Half-Edges.
→ **VQ-Idee**: statt rohe Koordinaten zu quantisieren, ein gelerntes Vokabular von „Block-Motiven".
Für kleine Datenmengen aber riskant (VQ-Codebook braucht Daten).

**Gemeinsame Schwäche für uns:** rohe Koordinaten-Tokenisierung hat **keine strukturelle
Gültigkeitsgarantie** und ist datenhungrig.

---

## B. Link-/Face-Prediction direkt aus Punktwolke

**Point2Quad** ([arXiv:2504.19545](https://arxiv.org/abs/2504.19545)). „Generating Quad Meshes from
Point Clouds via Face Prediction." Punkte sind gegeben; das Netz **klassifiziert Kandidaten-Quads**.
→ Passt, wenn die **Blocking-Vertices selbst Teilmenge der Eingabe-Punktwolke** sind. In unserem Fall
sind Blocking-Knoten aber i. d. R. **nicht** identisch mit Randknoten — Einschränkung.

**QuadLink** ([arXiv:2605.16813](https://arxiv.org/abs/2605.16813)). Drei-Stufen, **quad-first**:
(1) Anker = Vertices **und Face-Zentroide** vorhersagen, (2) **centroid-conditioned links** lernen
(welche Vertices gehören zu welchem Zentroid), (3) Faces mit geometrischer Verifikation assemblieren.
Erzeugt **anisotrope, sparse** Quad-dominante Meshes aus Punktwolken mit kohärentem Edge-Flow.
Tri-to-Quad-Operator zur Trainingsdaten-Erzeugung. → **Sehr nah an unserem Bedarf**: sparse,
anisotrope Quad-Blöcke mit sauberem Kantenfluss um eine Kontur. Der „Edge-Flow"-Fokus adressiert
genau den Verlust, den wir bei der Baseline sehen.

---

## C. Layout / Cross-Field

**TopGen** ([arXiv:2603.10606](https://arxiv.org/abs/2603.10606)). „Learning Structural Layouts and
Cross-Fields for Quadrilateral Mesh Generation." Lernt **erst das strukturelle Layout
(Singularitäten) + Cross-Field**, dann folgt die Quad-Erzeugung. → **Konzeptuell am nächsten an
„Block-Struktur"**: eine Blockzerlegung IST ein grobes Quad-Layout mit wenigen Singularitäten. Für
CFD um ein Blade ist die Blocktopologie (C-Grid/O-Grid/H-Grid) genau so ein Layout.

Klassische Referenz für den Baustein: **Cross-Field-basierte Quadrangulierung** (Instant Meshes /
QuadCover-Linie) — nützlich als geometrischer Prior / Post-Processing, nicht als Transformer.

---

## D. Feld-, Diffusions-, RL-basiert (inkl. CFD-spezifisch)

**BrepGen** ([arXiv:2401.15563](https://arxiv.org/abs/2401.15563)). Diffusion über **strukturierte
latente Geometrie** in einem Baum (Solid→Face→Edge→Vertex). Flächen als UV-Grids, danach **NURBS-Fit
(Poles = Kontrollpunkte)**. → **Higher-order-Idee**: Kanten/Flächen als **Kontrollnetze**
generieren statt punktweise. Für 2D-Blockkanten hieße das: pro Kante ein kleines Kontrollpolygon
(Bézier/NURBS) — direkt „higher order". **NURBGen**
([arXiv:2511.06194](https://arxiv.org/abs/2511.06194)) treibt das per LLM für Text→CAD.

**DRL-MeshGen** ([Springer, Eng. w. Computers 2025](https://link.springer.com/article/10.1007/s00366-025-02199-9)).
Automatisierte **block-strukturierte** Netzerzeugung via Deep-RL + optimaler konformer Abbildung.
→ Direkt in unserer Zielklasse (Blockstruktur), aber RL ist sample-ineffizient und schwer zu tunen.

**Dmsh** ([arXiv:2606.10601](https://arxiv.org/abs/2606.10601)). Multi-Agent-RL für **All-Quad**
Meshing (sequentielle Element-/Template-Entscheidungen).

**Airfoil-spezifisch (am direktesten):**
- **„Automatic Structured Mesh Generation … Parametric Multi-Block Topology"**
  ([MDPI Appl. Sci. 16(2):1116](https://www.mdpi.com/2076-3417/16/2/1116)). Parametrisiert
  **Kontroll-Vertices und Kontroll-Kanten** zur Multiblock-Topologie, dann CFD-Netz via
  **Transfinite Interpolation** — **exakt unsere Ausgabe-Pipeline**, klassisch/parametrisch statt
  gelernt. Gute Quelle für Topologie-Templates (C/O/H) und TFI-Randbedingungen.
- **Loop2Net** ([arXiv:2507.01057](https://arxiv.org/abs/2507.01057)). Datengetriebene Erzeugung &
  Optimierung von **Airfoil-CFD-Netzen aus sparsen Randkoordinaten** — bestätigt „Rand-Punktwolke
  als Input" als tragfähig.
- **MGNet / 3DMeshNet** ([arXiv:2407.01560](https://arxiv.org/abs/2407.01560)). **Unsupervised**
  strukturierte Netzerzeugung: Netz lernt die Parametrisierungsabbildung aus **Randkurven**,
  PDE-Residuum als Loss (kein Ground-Truth-Netz nötig). → **Wichtig für unser Datenproblem**:
  unsupervised/physik-informiert umgeht den Mangel an gelabelten Blockings.

**Übersichtsarbeiten:**
- „A Survey of AI Methods for Geometry Preparation and Mesh Generation in Engineering Simulation"
  ([arXiv:2512.23719](https://arxiv.org/abs/2512.23719)).
- Quad-Meshing-Survey-Repo ([GitHub](https://github.com/Bigger-and-Stronger/quad-meshing-survey)).

---

## Erkenntnisse für unser Problem

1. **Gültigkeit per Konstruktion** ist der wichtigste Hebel gegen unseren „leeres-Panel"-Fehler.
   PolyGen (Pointer-Faces) und QuadLink (Link-Verifikation) bauen das strukturell ein.
2. **Higher-order** = Kanten als **Kontrollnetze** (Bézier/NURBS-Poles) generieren, à la BrepGen/NURBGen
   — nicht mehr 1 Tangente pro Endpunkt, sondern k Kontrollpunkte pro Blockkante.
3. **Kleine Datenmengen**: entweder (a) Daten massiv augmentieren/synthetisieren (klassischer
   Multiblock-Generator wie MDPI liefert unbegrenzt Labels), oder (b) **unsupervised/physik-informiert**
   (MGNet) trainieren, oder (c) starke geometrische Priors (Cross-Field/TopGen).
4. **Bekannte Blade-Geometrie** ist ein starker Prior: sie fixiert einen Teil der Randkurven exakt —
   das Modell muss nur noch die **Außentopologie + Kantenverläufe** lernen, nicht die Blade-Form.

Diese vier Punkte strukturieren die Pläne in [03_plans_and_feasibility.md](03_plans_and_feasibility.md).
