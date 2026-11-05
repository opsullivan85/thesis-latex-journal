"""Plot transfer and retrained ablations (fig:transfer-ablations, Sec. 5.2).

Source of every number: the author's final results of 2026-11-03 (full record on the cluster,
/scratch/sselvaraj/gaitnet-runs/idm/inv/paper_exp/SUMMARY.md),
sections "Same checkpoint on Gazebo (OCS2) vs Isaac" and
"Retrained ablations (P2b)". Nothing here is invented or re-derived from
another source.

Design rule (user, 2026-11-03: "the bx and wisker is not good readapple ...
head map is notr good"): plain bars only. No error bars, whiskers,
confidence bands, heatmaps, p-values or per-seed marks. Every bar carries
its value, and every bar starts at 0.

(a) The benchmark checkpoint c0 (iteration 350, depth-camera terrain) on OCS2
    in Gazebo (G125 worlds, 100 trials per cell) and on the training controller
    in Isaac (120 robots per cell), each with its own IDM, then without an IDM.
    Groups: with IDM at 0.10 m/s (26/38/49 % voids pooled), with IDM at
    0.20 m/s, with IDM over all six cells, and, after a gap, without IDM over
    all six cells. Isaac's void fractions are nominal (20/30/40 %), matched to
    OCS2's by void area.
(b) The retrained ablations ("P2b"): a separate training recipe, two training
    seeds per arm, 300 trials per seed and command on the same OCS2 benchmark.
    Plotted: the mean of the two seeds per command, one bar each. The per-seed
    counts below are used only to form that mean and to check it against the
    source; no per-seed value is drawn (user rule, 2026-11-03).

Data conventions:
- Counts are integers out of n. Every derived number is a fractions.Fraction,
  rounded half up with decimal.ROUND_HALF_UP (Python's round() would give,
  e.g., 21.2 where the paper prints 21.3). Every printed value is asserted
  against the figure spec before anything is drawn.
- Isaac without IDM: the 49 %/0.10 m/s cell was reported as 8.2 %, which is
  not k/120 for any k. The author said on 2026-11-03 that the rate is about
  8 % and chose the nearest whole count, 10/120 = 8.3 %. All six cells are
  then counts, and "without IDM, all" is 153/720 = 21.25 -> 21.3
  (= \\corlBenchIsaacNoIdm, unchanged).
- Full retrains: the source gives only whole-percent per-command rates
  (92/92 at 0.10 m/s, 72/73 at 0.20 m/s) and no per-void cells, so their seed
  means are 92 and 72.5 and are printed exactly so ("92", not "92.0").

Open \\verify items that touch this figure's data:
- OCS2 with IDM at 49 %/0.20 m/s is 77, as in the G125 main table; the author
  confirmed 77 on 2026-11-03 (the Gazebo-vs-Isaac table's 87 is superseded).
  The sweep (Fig. 3) does not enter this figure.

usage: ~/miniforge3/envs/manim/bin/python -B fig_transfer_ablations.py
       (writes fig_transfer_ablations.pdf and .png next to this file)
"""
import itertools
import subprocess
from decimal import ROUND_HALF_UP, Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox, blended_transform_factory

HERE = Path(__file__).resolve().parent
getcontext().prec = 40


# ---------------------------------------------------------------- helpers
def dec(x: Fraction) -> Decimal:
    """Exact-enough Decimal of a Fraction (40 significant digits)."""
    return Decimal(x.numerator) / Decimal(x.denominator)


def half_up(x, places: str = "0.1") -> Decimal:
    """Round a Fraction or Decimal half up to the given quantum."""
    d = dec(x) if isinstance(x, Fraction) else Decimal(x)
    return d.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def pct(k: int, n: int) -> Fraction:
    """Success rate in percent as an exact Fraction."""
    return Fraction(100 * k, n)


def r1(x) -> str:
    """One-decimal, half-up string of a Fraction or Decimal."""
    return str(half_up(x))


# ---------------------------------------------------------------- data (a)
# Cell order: 0.10 m/s at 26/38/49 % voids, then 0.20 m/s at 26/38/49 %.
N_OCS2 = 100                               # trials per cell (\corlBenchTrialsPerCell)
N_ISAAC = 120                              # robots per cell (\corlBenchIsaacRobots)

OCS2_IDM = [100, 93, 96, 96, 87, 77]       # OCS2 IDM; 77 not 87, see docstring
ISAAC_IDM = [116, 120, 114, 120, 114, 107]  # training-controller IDM
OCS2_NOIDM = [48, 34, 11, 33, 17, 8]       # the real no-IDM result (author)
# Isaac without IDM, successes of 120. Cell 3 was reported as 8.2 %; the
# author's answer (2026-11-03: "about 8 %") fixes it at 10/120 = 8.3 %.
ISAAC_NOIDM_K = [50, 41, 10, 30, 14, 8]
ISAAC_NOIDM_ALL_GIVEN = Decimal("21.3")    # \corlBenchIsaacNoIdm

# ---------------------------------------------------------------- data (b)
# Per-void successes out of 100 per seed (s1, s2), order as in (a); GT P2b table.
ABLATION_CELLS = {
    "No history": ([98, 95, 84, 64, 36, 17], [96, 90, 84, 82, 70, 48]),
    "No candidate mixer": ([97, 89, 84, 89, 81, 58], [93, 77, 80, 81, 69, 52]),
    "No IDM in training,\nnone deployed": ([81, 74, 59, 69, 47, 22],
                                           [89, 85, 73, 76, 59, 34]),
    "No IDM in training,\nIsaac IDM deployed": ([72, 66, 52, 68, 52, 37],
                                                [89, 84, 68, 83, 64, 33]),
}
# GT's per-seed, per-command whole-percent rates and per-seed pooled rates,
# used only to cross-check the cells above (never plotted).
ABLATION_GIVEN_PER_SEED = {
    "No history": ((92, 39, "65.7"), (90, 67, "78.3")),
    "No candidate mixer": ((90, 76, "83.0"), (83, 67, "75.3")),
    "No IDM in training,\nnone deployed": ((71, 46, "58.7"), (82, 56, "69.3")),
    "No IDM in training,\nIsaac IDM deployed": ((63, 52, "57.8"), (80, 60, "70.2")),
}
# Full retrains: only whole-percent per-command rates are given (s1, s2).
FULL_RETRAIN_GIVEN = {"0.10": (92, 92), "0.20": (72, 73)}

# ---------------------------------------------------------------- derived (a)
# Successes and trials per group: 0.10 m/s = cells 0-2, 0.20 m/s = cells 3-5.
GROUPS_A = ["with IDM, 0.10 m/s", "with IDM, 0.20 m/s", "with IDM, all", "without IDM, all"]
ocs2_counts = {
    "with IDM, 0.10 m/s": (sum(OCS2_IDM[:3]), 3 * N_OCS2),
    "with IDM, 0.20 m/s": (sum(OCS2_IDM[3:]), 3 * N_OCS2),
    "with IDM, all": (sum(OCS2_IDM), 6 * N_OCS2),
    "without IDM, all": (sum(OCS2_NOIDM), 6 * N_OCS2),
}
isaac_counts = {
    "with IDM, 0.10 m/s": (sum(ISAAC_IDM[:3]), 3 * N_ISAAC),
    "with IDM, 0.20 m/s": (sum(ISAAC_IDM[3:]), 3 * N_ISAAC),
    "with IDM, all": (sum(ISAAC_IDM), 6 * N_ISAAC),
    "without IDM, all": (sum(ISAAC_NOIDM_K), 6 * N_ISAAC),
}
vals_a = {"ocs2": {g: pct(*kn) for g, kn in ocs2_counts.items()},
          "isaac": {g: pct(*kn) for g, kn in isaac_counts.items()}}
labels_a = {ctrl: {g: r1(v) for g, v in d.items()} for ctrl, d in vals_a.items()}

# ---------------------------------------------------------------- derived (b)
# Seed mean per command = (k_s1 + k_s2) / 600, k = successes over 3 cells.
ARMS = ["Full retrain"] + list(ABLATION_CELLS)
vals_b = {"Full retrain": (Fraction(sum(FULL_RETRAIN_GIVEN["0.10"]), 2),
                           Fraction(sum(FULL_RETRAIN_GIVEN["0.20"]), 2))}
for arm, (s1, s2) in ABLATION_CELLS.items():
    vals_b[arm] = (pct(sum(s1[:3]) + sum(s2[:3]), 600), pct(sum(s1[3:]) + sum(s2[3:]), 600))
labels_b = {arm: tuple(r1(v) for v in vs) for arm, vs in vals_b.items()}
# Full retrains are printed as given (whole-percent inputs): "92", not "92.0".
labels_b["Full retrain"] = (str(vals_b["Full retrain"][0]), str(dec(vals_b["Full retrain"][1])))

# ---------------------------------------------------------------- asserts
# (a) counts behind each bar
assert ocs2_counts == {"with IDM, 0.10 m/s": (289, 300), "with IDM, 0.20 m/s": (260, 300),
                       "with IDM, all": (549, 600), "without IDM, all": (151, 600)}
assert isaac_counts == {"with IDM, 0.10 m/s": (350, 360), "with IDM, 0.20 m/s": (341, 360),
                        "with IDM, all": (691, 720), "without IDM, all": (153, 720)}
# (a) Isaac cells reproduce GT's comparison table
assert [r1(pct(k, N_ISAAC)) for k in ISAAC_IDM] == ["96.7", "100.0", "95.0", "100.0", "95.0", "89.2"]
# The five other cells reproduce GT's rates; cell 3 is 8.3 (author: about 8 %).
assert [r1(pct(k, N_ISAAC)) for k in ISAAC_NOIDM_K] == ["41.7", "34.2", "8.3", "25.0", "11.7", "6.7"]
assert half_up(vals_a["isaac"]["without IDM, all"]) == ISAAC_NOIDM_ALL_GIVEN
# (a) every printed value, as in the figure spec (OCS2 / Isaac per group)
assert [(labels_a["ocs2"][g], labels_a["isaac"][g]) for g in GROUPS_A] == [
    ("96.3", "97.2"), ("86.7", "94.7"), ("91.5", "96.0"), ("25.2", "21.3")]
# (b) cells reproduce GT's per-seed rates
for arm, (s1, s2) in ABLATION_CELLS.items():
    for cells, (g10, g20, gpool) in zip((s1, s2), ABLATION_GIVEN_PER_SEED[arm]):
        assert half_up(pct(sum(cells[:3]), 300), "1") == g10, arm
        assert half_up(pct(sum(cells[3:]), 300), "1") == g20, arm
        assert r1(pct(sum(cells), 600)) == gpool, arm
# (b) every printed value, as in the figure spec (0.10 / 0.20 m/s per arm)
assert [labels_b[a] for a in ARMS] == [
    ("92", "72.5"), ("91.2", "52.8"), ("86.7", "71.7"), ("76.8", "51.2"), ("71.8", "56.2")]
assert vals_b["Full retrain"] == (Fraction(92), Fraction(145, 2))
print("all asserts passed")

# ---------------------------------------------------------------- style
# Same font, rcParams and export path as fig_bench.py and Fig. 1.
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Fira Sans", "font.size": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#1F2937", "axes.labelcolor": "#1F2937",
    "xtick.color": "#1F2937", "ytick.color": "#1F2937",
    "pdf.fonttype": 42,
})
INK = "#1F2937"
MUTED = "#4B5563"
GRID = "#E5E7EB"
C_OCS2 = "#4F46E5"        # GaitNet on OCS2 (GaitNet's colour in every figure)
C_ISAAC = "#94A3B8"       # the same checkpoint on the training controller in Isaac
C_V10 = "#A5B4FC"         # 0.10 m/s in (b)
C_V20 = "#312E81"         # 0.20 m/s in (b)
FS_VALUE = 6.5            # value labels on the bars
FS_TICK = 7.0             # tick and category labels
FS_KEY = 7.0              # legend keys and subtitle (7 pt, as in Figs. 2-3)
FS_TITLE = 7.5

# ---------------------------------------------------------------- layout [mm]
MM = 1 / 25.4
FIG_W, FIG_H = 152.0, 42.0
LEFT_A, WIDTH_A = 9.6, 63.0       # panel (a) axes
GAP_AB = 27.0                     # (a) right edge to (b) axes left; holds (b)'s row labels
RIGHT_MARGIN = 4.4                # holds (b)'s value labels past 100 %
WIDTH_B = FIG_W - LEFT_A - WIDTH_A - GAP_AB - RIGHT_MARGIN
BOTTOM, TOP = 8.0, 33.6           # axes bottom and top [mm from the canvas bottom]
TITLE_Y, HEADER_Y = 39.3, 35.6    # baselines of panel titles and the header row [mm]
ROW1_Y, ROW2_Y = BOTTOM - 3.3, BOTTOM - 6.7   # baselines of (a)'s two category rows [mm]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W * MM, FIG_H * MM),
                                 gridspec_kw=dict(width_ratios=[WIDTH_A, WIDTH_B]))
fig.subplots_adjust(left=LEFT_A / FIG_W, right=1 - RIGHT_MARGIN / FIG_W,
                    bottom=BOTTOM / FIG_H, top=TOP / FIG_H,
                    wspace=GAP_AB / ((WIDTH_A + WIDTH_B) / 2))
# Canvas coordinates in mm from the bottom-left corner, for titles, the
# header row and the category rows (all placed by baseline).
T_MM = Affine2D().scale(1 / FIG_W, 1 / FIG_H) + fig.transFigure


def text_mm(x, y, s, **kw):
    return fig.text(x, y, s, transform=T_MM, va="baseline", **kw)


def key_row(x0, y, items):
    """One-row legend at baseline y [mm]: a square swatch, then its label."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    side = 2.0                                     # swatch side [mm], about the cap height
    x = x0
    for colour, label in items:
        fig.add_artist(Rectangle((x, y - 0.05), side, side, transform=T_MM,
                                 facecolor=colour, edgecolor="none", zorder=3))
        t = text_mm(x + side + 1.0, y, label, ha="left", fontsize=FS_KEY, color=INK)
        x = T_MM.inverted().transform((t.get_window_extent(r).x1, 0))[0] + 3.2
    return x


def axes_left_mm(ax):
    return ax.get_position().x0 * FIG_W


def value_text(ax, xy, s, **kw):
    # White box under the digits masks the grid line behind them; zorder 1.5
    # keeps the box below every bar (zorder 2), so it can never hide one.
    return ax.annotate(s, xy=xy, textcoords="offset points", fontsize=FS_VALUE, color=INK,
                       annotation_clip=False, zorder=1.5,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), **kw)


# ---------------------------------------------------------------- panel (a)
# x positions of the four groups; the gap before "without IDM" is 0.45 wider.
X_A = [0.0, 1.0, 2.0, 3.45]
BAR_W = 0.38                      # one bar [category units]
DX = BAR_W / 2 + 0.01             # bar centre offset from the group centre
ax_a.set_xlim(-0.55, 3.95)
ax_a.set_ylim(0, 110)
ax_a.set_yticks(range(0, 101, 20))
ax_a.spines["left"].set_bounds(0, 100)
ax_a.set_ylabel("Success [%]", labelpad=2)
ax_a.grid(axis="y", color=GRID, lw=0.5)
ax_a.set_axisbelow(True)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.set_xticks([])

for xg, g in zip(X_A, GROUPS_A):
    for ctrl, dx, colour in (("ocs2", -DX, C_OCS2), ("isaac", +DX, C_ISAAC)):
        v = float(vals_a[ctrl][g])
        ax_a.bar(xg + dx, v, width=BAR_W, color=colour, edgecolor="none", zorder=2)
        value_text(ax_a, (xg + dx, v), labels_a[ctrl][g], xytext=(0, 1.8),
                   ha="center", va="baseline")

# Category labels in two rows under the axis: what was pooled, then the IDM condition.
blend_a = blended_transform_factory(ax_a.transData, T_MM)
for xg, lab in zip(X_A, ["0.10 m/s", "0.20 m/s", "all six cells", "all six cells"]):
    ax_a.text(xg, ROW1_Y, lab, transform=blend_a, ha="center", va="baseline",
              fontsize=FS_TICK, color=INK, clip_on=False)
for xg, lab in ((1.0, "with each controller's own IDM"), (X_A[3], "without IDM")):
    ax_a.text(xg, ROW2_Y, lab, transform=blend_a, ha="center", va="baseline",
              fontsize=FS_TICK, color=INK, clip_on=False)

x_a0 = axes_left_mm(ax_a)
text_mm(x_a0, TITLE_Y, "(a) Benchmark checkpoint on both controllers", ha="left",
        fontsize=FS_TITLE, fontweight="semibold", color=INK)
key_row(x_a0, HEADER_Y, [(C_OCS2, "OCS2 in Gazebo"), (C_ISAAC, "training controller in Isaac")])

# ---------------------------------------------------------------- panel (b)
# Arms top to bottom (y axis inverted). The pitch grows where a two-line
# label meets another label, so the gap between labels stays well above the
# gap between the lines of one label.
Y_ROWS = [0.0, 1.0, 2.0, 3.15, 4.6]
H_B = 0.36                        # one bar [row units]
# Bar centre offset from the arm centre. The 0.08-unit (~1 pt) gap between an
# arm's two bars keeps a value label, centred on its bar and about as tall as
# it, clear of the neighbouring bar.
DY = H_B / 2 + 0.04
ax_b.set_ylim(Y_ROWS[-1] + 0.6, -0.6)
ax_b.set_xlim(0, 100)
ax_b.set_xticks(range(0, 101, 20))
# Tick labels and the axis label sit on the same two baselines as (a)'s
# category rows, so both panels share one band under the axes.
ax_b.set_xticklabels([])
blend_b = blended_transform_factory(ax_b.transData, T_MM)
for xt in range(0, 101, 20):
    ax_b.text(xt, ROW1_Y, str(xt), transform=blend_b, ha="center", va="baseline",
              fontsize=FS_TICK, color=INK, clip_on=False)
ax_b.text(50, ROW2_Y, "Success [%]", transform=blend_b, ha="center", va="baseline",
          fontsize=plt.rcParams["axes.labelsize"], color=INK, clip_on=False)
ax_b.set_yticks(Y_ROWS)
ax_b.set_yticklabels(ARMS, fontsize=FS_TICK, linespacing=0.9)
ax_b.tick_params(axis="y", length=0, pad=3.0)
ax_b.grid(axis="x", color=GRID, lw=0.5)
ax_b.set_axisbelow(True)
ax_b.spines[["top", "right"]].set_visible(False)

for y, arm in zip(Y_ROWS, ARMS):
    for dy, colour, v, s in ((-DY, C_V10, vals_b[arm][0], labels_b[arm][0]),
                             (+DY, C_V20, vals_b[arm][1], labels_b[arm][1])):
        ax_b.barh(y + dy, float(v), height=H_B, color=colour, edgecolor="none", zorder=2)
        # Fira Sans digits span -0.022..0.679 em about the baseline, so a
        # baseline 0.33 em below the bar centre centres the digits on the bar.
        value_text(ax_b, (float(v), y + dy), s, xytext=(1.6, -0.33 * FS_VALUE), ha="left",
                   va="baseline")

# Title, subtitle and key start at the left edge of the row labels.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
x_b0 = T_MM.inverted().transform(
    (min(t.get_window_extent(renderer).x0 for t in ax_b.get_yticklabels()), 0))[0]
text_mm(x_b0, TITLE_Y, "(b) Retrained ablations", ha="left",
        fontsize=FS_TITLE, fontweight="semibold", color=INK)
sub = text_mm(x_b0, HEADER_Y, "separate recipe, mean of two seeds", ha="left",
              fontsize=FS_KEY, color=MUTED)
fig.canvas.draw()
x_key_b = T_MM.inverted().transform((sub.get_window_extent(renderer).x1, 0))[0] + 4.0
key_row(x_key_b, HEADER_Y, [(C_V10, "0.10 m/s"), (C_V20, "0.20 m/s")])

# Inkscape's text-to-path drops OpenType ligature glyphs ("filled" came out as
# "f lled"), so ligatures are switched off for every text in the figure.
for t in fig.findobj(Text):
    t.set_fontfeatures(["-liga"])


# ---------------------------------------------------------------- layout check
def ink_boxes(t: Text, r) -> list[Bbox]:
    """Glyph-ink boxes (one per line, display px) of an unrotated text.

    The layout bbox of a text includes the font's full ascent and descent, so
    it flags neighbours whose glyphs do not touch. TextPath gives the outlines.
    """
    if t.get_rotation() % 360:
        return [t.get_window_extent(r)]
    if isinstance(t, Annotation):
        t.update_positions(r)      # sets the transform of the text position (xytext)
    _, info, _ = t._get_layout(r)
    posx, posy = t.get_transform().transform(t.get_unitless_position())
    scale = r.points_to_pixels(1.0)
    boxes = []
    for line, _, (lx, ly) in info:
        if not line.strip():
            continue
        ext = TextPath((0, 0), line, size=t.get_fontsize(), prop=t.get_fontproperties()).get_extents()
        boxes.append(Bbox.from_extents(posx + lx + ext.x0 * scale, posy + ly + ext.y0 * scale,
                                       posx + lx + ext.x1 * scale, posy + ly + ext.y1 * scale))
    return boxes


def layout_report(fig, min_gap_pt: float = 0.5) -> list[str]:
    """Ink overlaps text-text and text-bar, texts off the canvas, small fonts."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    gap = r.points_to_pixels(min_gap_pt)
    fb = fig.bbox
    texts = []                     # (label, [ink boxes], fontsize)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            texts.append((t.get_text().replace("\n", "/"), ink_boxes(t, r), t.get_fontsize()))
    hit = lambda a, b: (a.x0 < b.x1 + gap and a.x1 > b.x0 - gap and
                        a.y0 < b.y1 + gap and a.y1 > b.y0 - gap)
    problems = []
    for s, boxes, fs in texts:
        if fs < 6.3:
            problems.append(f"font {fs} pt < 6.3: {s!r}")
        for bb in boxes:
            if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1:
                problems.append(f"off canvas: {s!r}")
    for (s1, b1, _), (s2, b2, _) in itertools.combinations(texts, 2):
        if any(hit(a, b) for a in b1 for b in b2):
            problems.append(f"text overlap: {s1!r} x {s2!r}")
    # Bars and key swatches in display px.
    marks = [("bar", p.get_window_extent(r)) for ax in (ax_a, ax_b) for p in ax.patches]
    marks += [("swatch", a.get_window_extent(r)) for a in fig.artists if isinstance(a, Rectangle)]
    for s, boxes, _ in texts:
        for kind, mb in marks:
            if any(hit(b, mb) for b in boxes):
                problems.append(f"{kind} touches text {s!r}")
    return problems


report = layout_report(fig)
print("layout:", "clean" if not report else "\n  " + "\n  ".join(report))
digits = sorted({t.get_text().replace("\n", "/") for t in fig.findobj(Text)
                 if any(ch.isdigit() for ch in t.get_text())})
print("texts with digits:", digits)

# ---------------------------------------------------------------- export
fig.savefig(HERE / "fig_transfer_ablations.png", dpi=300)
# Matplotlib embeds the OpenType (CFF) Fira Sans in a way some PDF checkers
# flag, so the PDF for the paper has its text converted to outlines.
raw = HERE / "fig_transfer_ablations.raw.pdf"
fig.savefig(raw)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                "-o", str(HERE / "fig_transfer_ablations.pdf")], check=True, capture_output=True)
raw.unlink()
print("wrote", HERE / "fig_transfer_ablations.pdf")
