# 06 — Kantengeometrie-Studie (Stufe 3: HO-Kantenrepräsentation)

Untersuchung, **wie die gekrümmten Blockkanten (Half-Edges) tokenisiert** werden — die dritte Stufe
des Zwei-Stufen-Tokenizers aus [05_face_block_generator.md](05_face_block_generator.md). Ziel: eine
Kanten-Repräsentation, die (a) die echte Kurve trifft, (b) den Multigraph (Blade Druck/Saug) respektiert
und (c) möglichst wenige Tokens braucht.

Prototyp: `prototype_twostage.py` (`TwoStageTokenizer`, `repr_mode`). Viz: `viz_twostage.py`
(Frames 09–11) → `figures/twostage/`. Alle Zahlen aus Round-trip über 300 Meshes von
`domain_data_10k.pt`.

---

## 1. Aufbau Stufe 3

Sequenz: `[start] verts [sep] faces [sep2] geom·4F [end]`.

Geometrie hängt an der **gerichteten Half-Edge**: jede Face-Seite `p0→p1` (Traversal-Reihenfolge)
erscheint mesh-weit **genau einmal** (der Twin liegt im Nachbarblock, Gegenrichtung). Daraus folgt:

- **Kein `{u,v}`-Dedup.** Innenkanten (geteilt) und Blade-Parallelkanten (Druck/Saug, gleiche
  Endpunkte, verschiedene Splines) werden dadurch korrekt getrennt gehalten.
- Fehlende Richtungen im Datensatz: **0 %** (jede Face-Seite hat einen Streamline-Eintrag).

Endpunkte kommen aus Stufe 1/2 (Vertices); Stufe 3 kodiert nur den **Verlauf** dazwischen.

---

## 2. Drei Repräsentationen (`repr_mode`)

| Modus | Tok/Kante | Was kodiert wird | Rekonstruktion |
|---|---|---|---|
| `hermite` | 6 | `α_start(sin,cos), tn_start, α_end(sin,cos), tn_end` | kubischer Hermite, Tangenten aus `edge_tangents` |
| `bezier` (k=2) | 4 | `α_start(sin,cos), α_end(sin,cos)` | quadratischer Bézier, Kontrollpunkt = Tangentenschnitt |
| `cubic_bezier` | 4 | `s1,h1,s2,h2` (Chord-lokal) | kubischer Bézier, 2 Kontrollpunkte **best-fit** an Streamline |

- **Winkel** werden als `(sin,cos)`-Paar kodiert (kein 0/2π-Wrap).
- **`hermite`** nutzt die vom Extractor gelieferten Tangenten direkt (`T0=tn·[cosα,sinα]`).
- **`bezier` k=2** = Port von `reconstruct_domain.quadratic_bezier_2d`: Kontrollpunkt ist der
  Schnittpunkt der beiden Tangentenlinien, **kein** Magnituden-Freiheitsgrad.
- **`cubic_bezier`** fittet die 2 inneren Kontrollpunkte per Least-Squares an die tatsächliche
  Streamline (Endpunkte fix) und kodiert sie **Chord-lokal**: `s` = Position längs der Sehne,
  `h` = Auslenkung quer (in Chord-Einheiten), Bounds `s∈[-0.5,1.5]`, `h∈[-1.2,1.2]`.

---

## 3. Ergebnisse (Round-trip, 300 Meshes)

Topologie in **allen** Modi 300/300 exakt (Stufe 1/2 unberührt). Geometrie-Fehler =
max. Punktabstand rekonstruierte Kurve ↔ `edge_to_streamline`, relativ zur Sehnenlänge:

| Modus | Tok/Kante | Seq-Len mean | Fehler median | Fehler mean | worst |
|---|---|---|---|---|---|
| `hermite` | 6 | 261 | 1.34 % | 2.16 % | 58 % |
| `bezier` k=2 | 4 | 199 (−24 %) | 5.99 % | 8.54 % | 62 % |
| **`cubic_bezier`** | **4** | **199 (−24 %)** | **0.74 %** | **1.18 %** | 58 % |

**`cubic_bezier` gewinnt beide Achsen gleichzeitig:** kürzeste Sequenz (wie `bezier`) **und**
genaueste Kurve (besser als `hermite`).

### Warum

- `hermite` ist an die **Extractor-Tangenten** gebunden — ein *ein* kubisches Segment mit
  vorgegebenen Endtangenten trifft die Kurve nur so gut wie diese Tangenten passen.
- `bezier` k=2 hat **0 Freiheitsgrade** beim Kontrollpunkt (fix am Tangentenschnitt) → auf
  asymmetrischen Bögen schlecht. Das ist der Grund für die 6 %, **nicht** die fehlende S-Kurve.
- `cubic_bezier` hat **2 freie Kontrollpunkte**, direkt an die Punktwolke der Kurve gefittet → beste
  mögliche Einzel-Kubik, mit weniger Zahlen als `hermite` (4 statt 6, weil Endpunkte schon aus
  Stufe 1 bekannt sind).

Auf einer einzelnen, monoton gekrümmten Kante (Frame 11, Sagitta/Chord≈0.10) liegen `hermite`
(2.4 %) und `bezier` (2.6 %) fast gleichauf; `cubic_bezier` trifft mit 0.3 %. Der große Satz-Ø-Abstand
von `bezier` kommt von den asymmetrischen/stark gekrümmten Kanten.

---

## 4. Daten-Befund: keine Wendepunkte

Über **alle 330.164 Kanten** des 10k-Datensatzes (geglättete Krümmung, Vorzeichenwechsel-Test):

> **0 Kanten mit Wendepunkt.** Die Streamlines sind reine Einzelbögen.

Konsequenzen:

- Das theoretische **Wendepunkt-Limit der Quadratik ist irrelevant** — es wird von den Daten nie
  getroffen. Es ist **kein** Argument gegen `bezier` k=2.
- Ein **einzelnes** kubisches Segment pro Kante reicht geometrisch aus (1 Wendepunkt Kapazität, 0
  gebraucht). Mehr-Segment/Kante bringt nur bei den entarteten Mini-Kanten etwas.

### worst-case (~58 %)

Bei allen Kubik-Modi identisch — eine **entartete Mini-Kante** (chord≈0.016). Inhärent für *ein*
Polynom pro Kante (roh, ohne Quantisierung, sogar 62 %). Kein Repräsentations-Bug, siehe
[[meshtron-known-bugs]] Bug 2. Bei Bedarf: Mehr-Segment nur für solche Kanten.

---

## 5. Empfehlung

**`cubic_bezier` als Default-Repräsentation für Stufe 3.**

- Kürzeste Sequenz (4 Tok/Kante, −24 % Gesamt-Seq ggü. `hermite`).
- Genaueste Rekonstruktion (median 0.7 %).
- Multigraph-treu (Half-Edge, kein `{u,v}`-Dedup) — Blade Druck/Saug bleiben getrennt.
- Chord-lokale Kodierung ist translations-/rotations-invariant → für das Modell leichter zu lernen
  als absolute Winkel.

Offene Punkte: (1) Kontrollpunkt-Kopf im Modell (`meshtron_domain.py`) — Stufe 3 als eigener
Regressions-/Klassifikationskopf; (2) Mehr-Segment nur für entartete Mini-Kanten; (3) Blade-Kanten
werden ohnehin per `blade_inject.py` exakt überschrieben (bekannte Geometrie).

---

## 6. Reproduktion

```bash
# Round-trip je Modus (Topologie + Geometrie-Fehler)
~/Environments/meshtron/bin/python prototype_twostage.py \
    --data domain_data_10k.pt --n 300 --mode hermite       # bzw. bezier / cubic_bezier

# Bildserie inkl. Frame 09 (gekrümmte Faces), 10 (Kante→Tokens), 11 (Methodenvergleich)
~/Environments/meshtron/bin/python viz_twostage.py         # -> figures/twostage/
```

Verwandt: [05_face_block_generator.md](05_face_block_generator.md) (Gesamt-Architektur Stufe 1–3),
[[meshtron-known-bugs]] (Bug 1 Parallelkanten, Bug 2 Kubik-Limit, Bug 3 Twin-Tangenten),
[[meshtron-blade-injection]].
