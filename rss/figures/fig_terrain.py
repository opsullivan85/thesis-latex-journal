"""Training terrain with its legend (fig:training-terrain-combined, Fig. 4 of the L4DC paper).

The picture is the author's render from Sullivan's thesis images,
gaitnet_thesis_images/figures/environment-combined.png (2379 x 568 px):
  left   one Go1 on the training terrain. The red patch is the area converted into a
         heightmap for the front-left leg, the black squares are voids, and the green
         arrow is the desired (commanded) base velocity.
  right  the full terrain. A black box and two black lines mark where the left view
         lies on it.
This script neither redraws nor resamples the picture. It puts the picture at the top of
the page and one legend row in a white strip under it, so the legend covers no pixel of
the picture. The page is no taller than the picture alone was (the paper is at its page
limit): the strip's height is taken from the picture, which is placed at 90 % of the page
width, centred.

What happens to the picture:
- The PNG is RGBA with a binary alpha channel: the right panel's background is fully
  transparent (alpha 0), everything else is opaque. On the paper's white page that
  background prints white (an image viewer may show it black). The picture is flattened
  onto white here, which is what the page shows anyway, so the PDF needs no soft mask.
  Every opaque pixel keeps its RGB value (asserted), and the PDF embeds the picture at
  its full 2379 x 568 px (asserted on the finished PDF with pdfimages); only its printed
  size changes.
- The legend swatches use the picture's own colours: the median colour of the red patch
  and of the green arrow (asserted against the picture, tolerance COLOUR_TOL).

Legend style, the same in every figure of the paper: Fira Sans 7 pt, ink #1F2937, no
frame, handlelength 1.6, handletextpad 0.5, columnspacing 1.2, labelspacing 0.3, entries
of at most four words, in lower case as in fig_teaser.py and fig_action.py. The figure
has no other text.

Size: the page is 0.92 x 152 mm = 139.84 mm wide and 33.38 mm tall, the size the picture
alone has at 0.92 of the text width in methodology.tex (139.84 x 33.39 mm), so replacing
the picture by this figure moves nothing in the paper. On that page the picture is
126.06 x 30.10 mm (90.1 %) and the legend strip under it 3.29 mm. Checked on the rendered
pixels: legend ink stays at least MIN_GAP_MM below the picture, inside the picture's
width and off the page edge.
The other layout, the picture at the full page width on a page 3.00 mm taller
(139.84 x 36.39 mm), is PAGE_W_PX, STRIP_PX = 2379, 51 with MAX_GROWTH_MM = 3.0,
MIN_GAP_MM = 0.4 and BOTTOM_PAD_MM = 0.10; use it only if the paper has the room.

Outputs:
  fig_terrain.pdf  picture (raster, /Interpolate true) under the legend (vector, text
                   converted to outlines with Inkscape, as in the other figures)
  fig_terrain.png  the same page on the picture's own pixel grid (2639 px wide,
                   479 ppi): the picture's pixels are copied 1:1 (asserted)

usage: python3 fig_terrain.py
       (needs inkscape, pdflatex, pdfimages, pdffonts, pdfinfo, pdftoppm)
"""
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cadence_names  # noqa: F401  (paper name: CADENCE; no method name is drawn here)
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
PICTURE = HERE / "gaitnet_thesis_images" / "figures" / "environment-combined.png"

# ---------------------------------------------------------------- geometry
MM = 1 / 25.4                           # inch per mm
PT_PER_MM = 72 / 25.4
FIG_W = 0.92 * 152.0                    # [mm] page width: 0.92 x \textwidth, as in methodology.tex
PIC_W_PX, PIC_H_PX = 2379, 568          # the picture, checked below
OLD_H = FIG_W * PIC_H_PX / PIC_W_PX     # [mm] 33.39: the picture alone at FIG_W (Fig. 4 before the legend)
# The page is laid out on the picture's pixel grid, so that the PNG copies the picture 1:1:
# PAGE_W_PX x (PIC_H_PX + STRIP_PX) picture pixels are printed FIG_W mm wide. This pair
# makes the page as tall as OLD_H to within 0.01 mm (never taller) with equal side margins.
PAGE_W_PX, STRIP_PX = 2639, 62          # page width, legend strip under the picture [picture px]
PAGE_H_PX = PIC_H_PX + STRIP_PX
assert (PAGE_W_PX - PIC_W_PX) % 2 == 0, "the picture cannot be centred on whole pixels"
X0_PX = (PAGE_W_PX - PIC_W_PX) // 2     # white margin left and right of the picture [picture px]
MM_PER_PX = FIG_W / PAGE_W_PX           # 0.0530 mm: the picture is embedded at 479 ppi
PIC_W, PIC_H = PIC_W_PX * MM_PER_PX, PIC_H_PX * MM_PER_PX   # [mm] 126.06 x 30.10
X0 = X0_PX * MM_PER_PX                  # [mm] 6.89
STRIP_H = STRIP_PX * MM_PER_PX          # [mm] 3.29
FIG_H = PAGE_H_PX * MM_PER_PX           # [mm] 33.38
MAX_STRIP_MM = 4.0                      # the strip stays a single thin row
MAX_GROWTH_MM = 0.0                     # page height the figure may gain over OLD_H
MIN_GAP_MM = 0.6                        # least white between the picture and legend ink
BOTTOM_PAD_MM = 0.20                    # white under the descenders, so none is clipped
assert STRIP_H <= MAX_STRIP_MM, STRIP_H
assert FIG_H <= OLD_H + MAX_GROWTH_MM, (FIG_H, OLD_H)

# ---------------------------------------------------------------- picture
src = Image.open(PICTURE)
assert src.mode == "RGBA" and src.size == (PIC_W_PX, PIC_H_PX), (src.mode, src.size)
rgba = np.asarray(src)
alpha = rgba[..., 3]
assert set(np.unique(alpha).tolist()) <= {0, 255}, "alpha is not binary; flattening would blend"
opaque = alpha == 255
# Flatten onto the page colour: transparent pixels become white, opaque ones are kept.
flat = np.where(opaque[..., None], rgba[..., :3], 255).astype(np.uint8)
assert np.array_equal(flat[opaque], rgba[..., :3][opaque])
assert (flat[~opaque] == 255).all()

# Colours of the two coloured marks, taken from the picture (median over the mark).
RED, GREEN, BLACK = "#F04A4A", "#34EB34", "#000000"
COLOUR_TOL = 3                          # 8-bit levels per channel
r, g, b = (flat[..., k].astype(int) for k in range(3))
MARKS = {"red patch": ((r > 180) & (g < 110) & (b < 110), RED),
         "green arrow": ((g > 180) & (r < 120) & (b < 120), GREEN)}
for name, (mask, colour) in MARKS.items():
    assert mask.sum() > 2000, (name, int(mask.sum()))
    assert not mask[:, PIC_W_PX // 2:].any(), f"{name}: also found in the right panel"
    median = np.median(flat[mask], axis=0)
    want = np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)])
    assert np.abs(median - want).max() <= COLOUR_TOL, (name, median, colour)
# The zoom box and lines are pure black; the voids are black to within DARK levels.
DARK = 10
assert (opaque & (flat.max(axis=2) == 0))[:, PIC_W_PX // 2:].sum() > 5_000
assert (opaque & (flat.max(axis=2) <= DARK))[:, :PIC_W_PX // 2].sum() > 50_000
print("picture checks passed")

# ---------------------------------------------------------------- style
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7, "legend.fontsize": 7,
                     "text.color": "#1F2937", "pdf.fonttype": 42})
INK = "#1F2937"
LEGEND_PT = 7


# ---------------------------------------------------------------- legend handles
def block_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    """A short, wide arrow like the one in the picture, in the handle box."""
    x0, y0 = -xdescent, -ydescent
    yc = y0 + height / 2
    length = 0.86 * width                         # a little shorter than a swatch
    xa, xb = x0 + (width - length) / 2, x0 + (width + length) / 2
    # the handle box runs from the baseline to cap height; 0.56 keeps the head
    # inside the text line (ascender to descender)
    head_len, head_half, shaft_half = 0.50 * length, 0.56 * height, 0.30 * height
    xh = xb - head_len
    return Polygon([(xa, yc - shaft_half), (xh, yc - shaft_half), (xh, yc - head_half),
                    (xb, yc), (xh, yc + head_half), (xh, yc + shaft_half),
                    (xa, yc + shaft_half)], closed=True)


class ZoomHandler(HandlerBase):
    """The picture's zoom mark in small: a box and the two lines that leave its right
    corners and spread towards the left view."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        yc = y0 + height / 2
        box_w, box_h = 0.34 * width, 0.56 * height
        bx, by = x0 + width - box_w - 0.4, yc - box_h / 2
        spread = 0.52 * height                    # half-height of the lines' left end
        box = Rectangle((bx, by), box_w, box_h, fc="none", ec=BLACK, lw=0.8,
                        joinstyle="miter", transform=trans)
        lines = [Line2D([x0, bx + box_w], [yc + s * spread, yc + s * box_h / 2],
                        color=BLACK, lw=0.6, solid_capstyle="butt", transform=trans)
                 for s in (+1, -1)]
        return [box, *lines]


class ZoomMark:                                   # legend key for ZoomHandler
    pass


# Legend key of the arrow: it only carries the colour, block_arrow() draws the shape.
ARROW_KEY = Polygon([(0, 0), (1, 0), (1, 1)], facecolor=GREEN, edgecolor="none")
ENTRIES = [                                       # (handle, label), left to right
    (Patch(facecolor=RED, edgecolor="none"), "front-left leg's scan area"),
    (Patch(facecolor=BLACK, edgecolor="none"), "void"),
    (ARROW_KEY, "commanded velocity"),
    (ZoomMark(), "location of left view"),
]
for _, label in ENTRIES:
    assert len(label.split()) <= 4, label
    assert label == label.lower(), label          # lower case, as in fig_teaser and fig_action

# ---------------------------------------------------------------- legend (vector overlay)
W_IN, H_IN = FIG_W * MM, FIG_H * MM
fig = plt.figure(figsize=(W_IN, H_IN))
leg = fig.legend(handles=[h for h, _ in ENTRIES], labels=[t for _, t in ENTRIES],
                 loc="lower center", bbox_to_anchor=(0.5, BOTTOM_PAD_MM / FIG_H),
                 ncol=len(ENTRIES),
                 frameon=False, fontsize=LEGEND_PT, labelcolor=INK,
                 handlelength=1.6, handletextpad=0.5, columnspacing=1.2, labelspacing=0.3,
                 borderpad=0.0, borderaxespad=0.0,
                 handler_map={ARROW_KEY: HandlerPatch(patch_func=block_arrow),
                              ZoomMark: ZoomHandler()})

# ---------------------------------------------------------------- layout checks
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
box = leg.get_tightbbox(renderer)                 # display px at fig.dpi
to_mm = 25.4 / fig.dpi
print(f"legend box: {box.width * to_mm:.1f} x {box.height * to_mm:.2f} mm in a "
      f"{STRIP_H:.2f} mm strip")
assert box.x0 >= 0 and box.x1 <= fig.bbox.x1 and box.y0 >= 0, "legend outside the canvas"
assert box.y1 * to_mm <= STRIP_H - MIN_GAP_MM, "legend box too close to the picture"
texts = leg.get_texts()
assert [t.get_text() for t in texts] == [t for _, t in ENTRIES]
assert all(t.get_fontsize() >= 7 for t in texts), "legend text below 7 pt"
assert len(fig.texts) == 0 and not fig.axes, "the figure carries no text besides the legend"

# ---------------------------------------------------------------- PNG on the picture's pixel grid
# 0.01 px of slack keeps int() in the Agg backend from rounding PAGE_W_PX down.
png_dpi = (PAGE_W_PX + 0.01) / W_IN
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=png_dpi, transparent=True)
overlay_px = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
assert overlay_px.size == (PAGE_W_PX, PAGE_H_PX), overlay_px.size
ink = np.asarray(overlay_px)[..., 3] > 0
ink_rows, ink_cols = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
# Legend ink (text and handles, measured on the rendered pixels) lies in the strip only,
# clear of the picture above and of the page edge below, and within the picture's width.
ink_gap = (ink_rows.min() - PIC_H_PX) * MM_PER_PX
ink_foot = (PAGE_H_PX - 1 - ink_rows.max()) * MM_PER_PX
print(f"legend ink: {ink_gap:.2f} mm under the picture, {ink_foot:.2f} mm above the page edge, "
      f"{(ink_cols.max() - ink_cols.min() + 1) * MM_PER_PX:.1f} mm wide")
assert ink_gap >= MIN_GAP_MM, f"legend ink {ink_gap:.2f} mm from the picture"
assert ink_foot > 0, "legend ink reaches the page edge"
assert ink_cols.min() >= X0_PX and ink_cols.max() < X0_PX + PIC_W_PX, "legend wider than the picture"
page = Image.new("RGBA", (PAGE_W_PX, PAGE_H_PX), (255, 255, 255, 255))
page.paste(Image.fromarray(flat), (X0_PX, 0))
page = Image.alpha_composite(page, overlay_px).convert("RGB")
page_px = np.asarray(page)
assert np.array_equal(page_px[:PIC_H_PX, X0_PX:X0_PX + PIC_W_PX], flat), "picture pixels changed in the PNG"
ppi = PAGE_W_PX / W_IN
page.save(HERE / "fig_terrain.png", dpi=(ppi, ppi), optimize=True)

# ---------------------------------------------------------------- PDF: picture + vector legend
# Same construction as teaser_stack.stack (raster base with /Interpolate true under a
# vector overlay, one pdflatex page); here the base is the picture alone, at the top,
# X0 from the left edge.
TEX = rf"""\documentclass{{article}}
\usepackage[paperwidth={FIG_W:.4f}mm,paperheight={FIG_H:.4f}mm,margin=0pt]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\topskip=0pt
\begin{{document}}%
% zero-height box hung from the top edge, so nothing can spill to a second page
\noindent\raisebox{{-\height}}[0pt][0pt]{{%
\makebox[0pt][l]{{\hspace*{{{X0:.4f}mm}}\raisebox{{{STRIP_H:.4f}mm}}{{%
\includegraphics[interpolate=true,width={PIC_W:.4f}mm,height={PIC_H:.4f}mm]{{picture.png}}}}}}%
\includegraphics[width={FIG_W:.4f}mm,height={FIG_H:.4f}mm]{{overlay.pdf}}}}%
\end{{document}}
"""
OUT_PDF = HERE / "fig_terrain.pdf"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    Image.fromarray(flat).save(td / "picture.png", optimize=True)
    fig.savefig(td / "overlay_raw.pdf", transparent=True)
    subprocess.run(["inkscape", str(td / "overlay_raw.pdf"), "--export-type=pdf",
                    "--export-text-to-path", f"--export-filename={td / 'overlay.pdf'}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (td / "page.tex").write_text(TEX)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "page.tex"],
                   cwd=td, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---- checks on the finished page
    info = subprocess.run(["pdfinfo", str(td / "page.pdf")], capture_output=True, text=True).stdout
    assert re.search(r"^Pages:\s+1$", info, re.M), info
    w_pt, h_pt = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info).groups())
    assert abs(w_pt - FIG_W * PT_PER_MM) < 0.1 and abs(h_pt - FIG_H * PT_PER_MM) < 0.1, (w_pt, h_pt)
    fonts = subprocess.run(["pdffonts", str(td / "page.pdf")], capture_output=True, text=True).stdout
    assert len(fonts.strip().splitlines()) == 2, f"text was not converted to outlines:\n{fonts}"
    subprocess.run(["pdfimages", "-png", str(td / "page.pdf"), str(td / "img")], check=True)
    images = sorted(td.glob("img-*.png"))
    assert len(images) == 1, [p.name for p in images]   # one image, no soft mask
    embedded = Image.open(images[0])
    assert embedded.mode == "RGB" and np.array_equal(np.asarray(embedded), flat), \
        "the picture embedded in the PDF differs from the source"
    # The page rendered on the PNG's pixel grid has the picture where the PNG has it and the
    # legend ink within 2 px of the PNG's, so the gaps measured on the PNG hold for the PDF.
    # The picture is compared on means over 8 x 8 px blocks, which tolerates the renderer's
    # sub-pixel resampling: the mean difference is below 1 level with the picture in place
    # and above 4 levels for a shift of 4 px (0.2 mm).
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(PAGE_W_PX),
                    "-scale-to-y", str(PAGE_H_PX), str(td / "page.pdf"), str(td / "render")],
                   check=True)
    render = np.asarray(Image.open(td / "render.png").convert("RGB"))
    assert render.shape == page_px.shape, render.shape

    def block_means(px, n=8):
        """Mean colour of each n x n pixel block."""
        h, w = px.shape[0] // n * n, px.shape[1] // n * n
        return px[:h, :w].reshape(h // n, n, w // n, n, 3).mean(axis=(1, 3))

    pic_diff = np.abs(block_means(render[:PIC_H_PX, X0_PX:X0_PX + PIC_W_PX])
                      - block_means(flat)).mean()
    assert pic_diff < 3.0, f"picture misplaced in the PDF (mean difference {pic_diff:.2f} levels)"

    def strip_ink_box(px, skip=3):
        """First and last row and column of non-white pixels in the legend strip. The
        `skip` rows next to the picture are left out: a renderer may bleed the picture's
        last row into them."""
        dark = px[PIC_H_PX + skip:].min(axis=2) < 255
        rows, cols = np.nonzero(dark.any(axis=1))[0], np.nonzero(dark.any(axis=0))[0]
        return np.array([rows.min(), rows.max(), cols.min(), cols.max()])

    ink_shift = np.abs(strip_ink_box(render) - strip_ink_box(page_px)).max()
    assert ink_shift <= 2, f"legend in the PDF is {ink_shift} px from where the PNG has it"
    size_mb = (td / "page.pdf").stat().st_size / 1e6
    assert size_mb < 2.0, size_mb
    shutil.copy(td / "page.pdf", OUT_PDF)
plt.close(fig)
print(f"page {FIG_W:.2f} x {FIG_H:.2f} mm ({FIG_H - OLD_H:+.3f} mm against the picture alone): "
      f"picture {PIC_W:.2f} x {PIC_H:.2f} mm ({100 * PIC_W / FIG_W:.1f} %) + strip {STRIP_H:.2f} mm; "
      f"PDF {size_mb:.2f} MB, picture embedded at {PIC_W_PX} x {PIC_H_PX} px")
print("wrote", OUT_PDF)
