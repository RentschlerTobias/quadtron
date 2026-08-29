"""Generate images/intro_order_a.png (row-major) and intro_order_b.png (column-major)
for the "Why Order Matters" slide.

Design:
- 3x3 face grid. Face background = order-gradient (blue=first visited, red=last),
  DIFFERS between row-major (a) and column-major (b) since traversal differs.
- Each face corner carries a [square, circle, circle] glyph:
    square  = face identity color (9 colors), constant across a/b (same physical face)
    circles = coordinate-value identity color (4 colors, shared palette for x and y),
              constant across a/b (same physical coordinates)
- Below the grid: full 36-token sequence (9 faces x 4 corners), in traversal order,
  wrapped 3 rows x 12 glyphs. Same color rules as above -> same colors reappear in a
  different arrangement between a/b, visually showing "different order, different
  sequence" while the underlying mesh identity stays the same.
"""

from PIL import Image, ImageDraw

# ---------------------------------------------------------------- palettes --
FACE_PALETTE = {  # (row, col) -> color, constant mesh identity — soft/pastel, organic-inspired
    (0, 0): "#8fa9c4", (0, 1): "#d99a72", (0, 2): "#a9b98c",
    (1, 0): "#c47f6e", (1, 1): "#b490a8", (1, 2): "#b8ab7a",
    (2, 0): "#c98fa0", (2, 1): "#d9b96a", (2, 2): "#7fa89c",
}
COORD_PALETTE = {0: "#e3b565", 1: "#7f9a6b", 2: "#8ea3c9", 3: "#c97b6d"}

GRAD_STOPS = [(0.0, (59, 76, 192)), (0.5, (242, 242, 242)), (1.0, (180, 4, 38))]


def grad_color(t):
    for (t0, c0), (t1, c1) in zip(GRAD_STOPS, GRAD_STOPS[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = round(c0[0] + f * (c1[0] - c0[0]))
            g = round(c0[1] + f * (c1[1] - c0[1]))
            b = round(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#f2f2f2"


BORDER = "#9a9490"
BG = "#f3f2f2"
INK = "#201e1d"


def face_order(mode):
    """Return dict (row,col) -> order index 0..8, and list of (row,col) in
    traversal order."""
    seq = []
    if mode == "row":
        for row in range(3):
            for col in range(3):
                seq.append((row, col))
    else:  # column-major
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


def render(mode, out_path):
    order, seq = face_order(mode)

    # -- layout: grid --
    cell = 150
    grid_w = grid_h = cell * 3
    margin = 30
    W = grid_w + margin * 2

    seq_row_h = 62
    seq_rows = 3
    seq_top_pad = 26
    seq_area_h = seq_top_pad + seq_rows * seq_row_h
    H = margin + grid_h + seq_area_h + margin

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    gx0, gy0 = margin, margin
    SQ, CIRC_D, GAP = 42, 16, 4
    INSET = 34

    for row in range(3):
        for col in range(3):
            fx0 = gx0 + col * cell
            fy0 = gy0 + row * cell
            fx1, fy1 = fx0 + cell, fy0 + cell
            t = order[(row, col)] / 8
            d.rectangle([fx0, fy0, fx1, fy1], fill=grad_color(t), outline=BORDER, width=2)

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

    # -- sequence bar: 9 faces x 4 corners = 36 glyphs, 3 rows x 12 --
    sq2, circ2, gap2 = 30, 11, 3
    glyph_w = sq2 + 10
    tokens = []
    for (row, col) in seq:
        face_color = FACE_PALETTE[(row, col)]
        for (xv, yv) in [(col, row), (col + 1, row), (col, row + 1), (col + 1, row + 1)]:
            tokens.append((face_color, COORD_PALETTE[xv], COORD_PALETTE[yv]))

    per_row = 12
    start_y = gy0 + grid_h + seq_top_pad
    total_row_w = per_row * glyph_w
    start_x = gx0 + (grid_w - total_row_w) / 2 + glyph_w / 2

    for i, (fc, xc, yc) in enumerate(tokens):
        r = i // per_row
        c = i % per_row
        cx = start_x + c * glyph_w
        cy = start_y + r * seq_row_h + seq_row_h / 2
        draw_glyph(d, cx, cy, sq2, circ2, gap2, fc, xc, yc)

    img.save(out_path)
    print("saved", out_path, img.size)


render("row", "/tmp/claude-1000/-home-t1dde-Duty-projects-presentations-privat-AIFLUIDs/75b6f084-2fbe-4e1b-aa16-11639d8580e5/scratchpad/intro_order_a.png")
render("col", "/tmp/claude-1000/-home-t1dde-Duty-projects-presentations-privat-AIFLUIDs/75b6f084-2fbe-4e1b-aa16-11639d8580e5/scratchpad/intro_order_b.png")
