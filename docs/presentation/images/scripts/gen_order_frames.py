"""Generate step-frame PNGs for the "Why Order Matters" click-through animation.

For each variant (row-major / column-major) and each reveal step 0..9:
  step 0 = empty 3x3 grid (order-gradient backgrounds only, no glyphs, empty
           sequence-bar placeholders)
  step k = the first k faces (in traversal order) show their 4 corner glyphs
           in the grid AND their 4 tokens filled in the sequence bar; not-yet
           revealed sequence slots stay as thin outline placeholders.

Color rules (constant mesh identity, same in both variants):
  square  = face identity (9 colors)
  circles = coordinate-value identity (4 colors, shared scale for x and y)
Only the face background (order-gradient blue->white->red) differs between
variants, since that encodes the traversal order itself.

Output: images/frames/order_<mode>_<step>.png  (mode: row | col), 20 files.
"""

from PIL import Image, ImageDraw
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "frames")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- palettes --
FACE_PALETTE = {  # (row, col) -> color, constant mesh identity — pastel, organic-inspired
    (0, 0): "#8fa9c4", (0, 1): "#d99a72", (0, 2): "#a9b98c",
    (1, 0): "#c47f6e", (1, 1): "#b490a8", (1, 2): "#b8ab7a",
    (2, 0): "#c98fa0", (2, 1): "#d9b96a", (2, 2): "#7fa89c",
}
COORD_PALETTE = {0: "#e3b565", 1: "#7f9a6b", 2: "#8ea3c9", 3: "#c97b6d"}

GRAD_STOPS = [(0.0, (59, 76, 192)), (0.5, (242, 242, 242)), (1.0, (180, 4, 38))]

BORDER = "#9a9490"
BG = "#f3f2f2"
INK = "#201e1d"
PLACEHOLDER = "#d8d5d3"

# ---------------------------------------------------------------- sizing (compact, ~0.66x of the original static design) --
CELL = 100
MARGIN = 20
SQ, CIRC_D, GAP = 28, 11, 3
INSET = 22

SQ2, CIRC2, GAP2 = 20, 7, 2
GLYPH_W = SQ2 + 8
PER_ROW = 12
SEQ_ROW_H = 41
SEQ_TOP_PAD = 17


def grad_color(t):
    for (t0, c0), (t1, c1) in zip(GRAD_STOPS, GRAD_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = round(c0[0] + f * (c1[0] - c0[0]))
            g = round(c0[1] + f * (c1[1] - c0[1]))
            b = round(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#f2f2f2"


def face_order(mode):
    seq = []
    if mode == "row":
        for row in range(3):
            for col in range(3):
                seq.append((row, col))
    else:
        for col in range(3):
            for row in range(3):
                seq.append((row, col))
    order = {rc: i for i, rc in enumerate(seq)}
    return order, seq


def draw_glyph(d, cx, cy, sq, circ_d, gap, face_color, x_color, y_color):
    x0, y0 = cx - sq / 2, cy - sq / 2
    x1, y1 = cx + sq / 2, cy + sq / 2
    d.rounded_rectangle([x0, y0, x1, y1], radius=max(2, sq * 0.13),
                         fill=face_color, outline=INK, width=2)
    circ_y0 = cy - circ_d / 2
    circ_y1 = cy + circ_d / 2
    lx0 = cx - gap / 2 - circ_d
    lx1 = cx - gap / 2
    rx0 = cx + gap / 2
    rx1 = cx + gap / 2 + circ_d
    d.ellipse([lx0, circ_y0, lx1, circ_y1], fill=x_color, outline=INK, width=1)
    d.ellipse([rx0, circ_y0, rx1, circ_y1], fill=y_color, outline=INK, width=1)


def draw_placeholder(d, cx, cy, sq):
    x0, y0 = cx - sq / 2, cy - sq / 2
    x1, y1 = cx + sq / 2, cy + sq / 2
    d.rounded_rectangle([x0, y0, x1, y1], radius=max(2, sq * 0.13),
                         outline=PLACEHOLDER, width=2)


def render(mode, reveal, out_path):
    order, seq = face_order(mode)
    grid_w = grid_h = CELL * 3
    W = grid_w + MARGIN * 2

    seq_rows = 3
    seq_area_h = SEQ_TOP_PAD + seq_rows * SEQ_ROW_H
    H = MARGIN + grid_h + seq_area_h + MARGIN

    total_row_w = PER_ROW * GLYPH_W
    W = max(W, total_row_w + MARGIN * 2)

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    gx0 = (W - grid_w) / 2
    gy0 = MARGIN

    for row in range(3):
        for col in range(3):
            fx0 = gx0 + col * CELL
            fy0 = gy0 + row * CELL
            fx1, fy1 = fx0 + CELL, fy0 + CELL
            t = order[(row, col)] / 8
            d.rectangle([fx0, fy0, fx1, fy1], fill=grad_color(t), outline=BORDER, width=2)

            if order[(row, col)] >= reveal:
                continue  # not yet revealed this step

            face_color = FACE_PALETTE[(row, col)]
            corners = [
                (fx0 + INSET, fy0 + INSET, col, row),
                (fx1 - INSET, fy0 + INSET, col + 1, row),
                (fx0 + INSET, fy1 - INSET, col, row + 1),
                (fx1 - INSET, fy1 - INSET, col + 1, row + 1),
            ]
            for cx, cy, xv, yv in corners:
                draw_glyph(d, cx, cy, SQ, CIRC_D, GAP,
                           face_color, COORD_PALETTE[xv], COORD_PALETTE[yv])

    # -- sequence bar: 36 slots, filled up to reveal*4, rest = placeholders --
    tokens = []
    for (row, col) in seq:
        face_color = FACE_PALETTE[(row, col)]
        for (xv, yv) in [(col, row), (col + 1, row), (col, row + 1), (col + 1, row + 1)]:
            tokens.append((face_color, COORD_PALETTE[xv], COORD_PALETTE[yv]))

    start_y = gy0 + grid_h + SEQ_TOP_PAD
    start_x = (W - total_row_w) / 2 + GLYPH_W / 2
    filled = reveal * 4

    for i, (fc, xc, yc) in enumerate(tokens):
        r = i // PER_ROW
        c = i % PER_ROW
        cx = start_x + c * GLYPH_W
        cy = start_y + r * SEQ_ROW_H + SEQ_ROW_H / 2
        if i < filled:
            draw_glyph(d, cx, cy, SQ2, CIRC2, GAP2, fc, xc, yc)
        else:
            draw_placeholder(d, cx, cy, SQ2)

    img.save(out_path)


for mode in ("row", "col"):
    for step in range(10):
        render(mode, step, os.path.join(OUT_DIR, f"order_{mode}_{step}.png"))

print("done:", len(os.listdir(OUT_DIR)), "frames in", OUT_DIR)
