"""Plot the footstep-correction figure of the L4DC paper (fig:transfer, Sec. 5, Q3).

Claim supported: with the same planner weights, GaitNet works on OCS2 (the new
controller) only when the IFM is refitted on OCS2; no correction, the training
controller's IFM, and a constant per-leg offset all fall short, and overall
only the refitted IFM lifts GaitNet above OCS2's own trot. (On the 49 % cells
alone the per-leg offset also beats the trot; the caption must say "overall".)

One panel, 74 x 42 mm, to sit beside fig_swing.pdf (74 x 42 mm) in one row;
margins as fig_swing.py so the two axes align. Plain bars from 0, value printed
on every bar; no error bars, whiskers, bands, heatmaps or p-values; one
aggregate per multi-seed method:
  GaitNet's OCS2 benchmark success over all six cells with each footstep
  correction: none (the desired foothold is sent uncorrected), the training
  controller's IFM, a constant per-leg offset, and the IFM refitted on OCS2.
  All bars indigo: same planner, only the correction changes.
  One thin dashed grey reference line at OCS2's own trot over the same six
  cells (\\corlBenchTrot). Trot style as fig_swing.py / fig_gait.py
  (#4B5563, dashes 3.2/1.8), thinner because it is a reference.

  Legend (author, 2026-11-13: "add legend all the figure"): one row inside
  the axes, under the title, above every bar (the tallest is 85.1 %) and
  above the end of the y axis (100 %; the y range runs to Y_TOP = 114 for
  that, so that the handles cannot be read as marks at 100 %), in
  the paper's common legend style: indigo bar = GaitNet, dashed grey line =
  OCS2's own trot. The direct label that sat on the reference line was
  removed (it said the same as the legend entry). Legend text is 7.1 pt on
  the 74 mm canvas, i.e. 7.0 pt when printed at 0.48\\textwidth (72.96 mm).

  Value labels: at this height a value printed above the per-leg-offset bar
  (60.0) would span about 62-71 % and the trot line (67.8) would run through
  it. Hence every bar tall enough to hold its value (>= INSIDE_MIN) carries it
  inside its top, in white (contrast 6.3:1 on #4F46E5); the 5.5 bar keeps it
  above. The layout check asserts that an inside label lies within its bar,
  that the reference line crosses no text, and that the legend covers no bar,
  value, line or spine.

  All text >= 7 pt at final size (rewrite decisions: "Never put fonts below
  7 pt at final size"); fig_bench.py's 6.3 pt value size is not reused here.

  (a) REMOVED 2026-11-12 (reviewer): "training controller in Isaac vs OCS2"
      paired \\corlBenchIsaac{Idm,Mlp,Single} (96.0 / 80.3 / 72.1) with the
      five-seed OCS2 means. The Isaac values have no documented per-seed source
      (results_to_fill.md section 3 is empty; \\corlBenchIsaacIdm's comment
      "691/720" and the per-cell k/120 values in constants.tex read as ONE
      run), and they were entered on 2026-11-05 next to OCS2 values that have
      since been replaced (e.g. \\corlBenchSingle 82.8 -> 39.6, constants.tex
      "was 82.8"; commit 569c6dc). Restore (a) only once
      section 3 holds the Isaac counts of the same five iteration-300 policies.
      The rewrite decisions of 2026-11-12 say not to restore it at all.

Source (nothing here is invented): per-seed success counts (of 100 per cell,
6 cells, five training seeds at iteration 300 per arm) entered by the author on
2026-11-12 in ../../results_to_fill.md, section 2, and the one-run counts of
OCS2's own trot from the same table. They are copied below, re-checked against
that file when it exists, pooled exactly, and asserted against the \\corlBench*
macros of ../constants.tex (the source of truth).

Provenance note: the protocol table of results_to_fill.md lists the three
control arms with 5 seeds at iteration 300 but does not itself say that their
seeds s1-s5 are GaitNet's s1-s5 policies; results.tex (Q3) states that the
controls keep GaitNet's weights.

usage: python3 fig_transfer.py   (writes fig_transfer.pdf and fig_transfer.png here)
"""
import re
import shutil
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cadence_names  # noqa: F401  (paper name: CADENCE)
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"
RESULTS_MD = HERE.parent.parent / "results_to_fill.md"

# ---------------------------------------------------------------- data
N_CELL = 100                            # \corlBenchTrialsPerCell
DENSE = (2, 5)                          # 49 % voids at 0.10 and 0.20 m/s
# Per-seed success counts of 100 per cell, results_to_fill.md section 2
# (2026-11-12). Cell order: 0.10 m/s at 26/38/49 % voids, then 0.20 m/s.
# Keys are the method names of that table.
OCS2_SEEDS = {
    "GaitNet without the IFM": ((21, 20, 15, 19, 18, 10), (18, 19, 16, 16, 17, 12),
                                (22, 23, 16, 18, 17, 13), (23, 21, 12, 19, 17, 11),
                                (21, 19, 13, 17, 15, 14)),
    "GaitNet + training IFM": ((9, 9, 5, 7, 6, 7), (7, 1, 8, 5, 2, 0),
                               (10, 6, 5, 7, 3, 3), (7, 7, 9, 3, 5, 0),
                               (9, 8, 6, 6, 4, 0)),
    "GaitNet + offset": ((68, 67, 55, 65, 63, 45), (65, 64, 53, 64, 62, 48),
                         (66, 63, 58, 63, 65, 42), (67, 65, 48, 59, 58, 48),
                         (70, 70, 54, 68, 66, 50)),
    "GaitNet": ((92, 88, 78, 90, 91, 69), (94, 91, 77, 93, 89, 73),
                (94, 94, 75, 96, 92, 71), (98, 93, 80, 96, 93, 75),
                (87, 85, 68, 83, 84, 63)),
}
# OCS2's own trot: one run (no training seeds), same table, same cell order.
TROT = "OCS2's own trot"
TROT_COUNTS = (91, 72, 46, 88, 70, 40)
# Bar order (left to right), tick label, and the macros of the all-cell and
# 49 % pools. The 49 % pools are asserted but not drawn (Table 1 prints them).
BARS = (   # results_to_fill.md name, tick label, all cells, 49 % voids
    ("GaitNet without the IFM", "None",            "corlBenchNoIdm",    "corlBenchNoIdmDense"),
    ("GaitNet + training IFM",  "Training\nIFM",   "corlBenchTrainIfm", "corlBenchTrainIfmDense"),
    ("GaitNet + offset",        "Per-leg\noffset", "corlBenchOffset",   "corlBenchOffsetDense"),
    ("GaitNet",                 "Refitted\nIFM",   "corlBenchFull",     "corlBenchFullDense"),
)
SPEC = {   # macro name -> printed string
    "corlBenchNoIdm": "17.1", "corlBenchTrainIfm": "5.5",
    "corlBenchOffset": "60.0", "corlBenchFull": "85.1",
    "corlBenchNoIdmDense": "13.2", "corlBenchTrainIfmDense": "4.3",
    "corlBenchOffsetDense": "50.1", "corlBenchFullDense": "72.9",
    "corlBenchTrot": "67.8",            # the reference line
}


def half_up(x: Fraction, places: str = "0.1") -> Decimal:
    """Round an exact fraction half up (Python's round() is half-even)."""
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
        Decimal(places), rounding=ROUND_HALF_UP)


def pooled(seeds, idx=range(6)) -> Fraction:
    """Exact success rate [%] over the given cells and all seeds. Every seed
    ran the same trials, so this equals the mean of the per-seed rates."""
    k = sum(s[i] for s in seeds for i in idx)
    return Fraction(100 * k, N_CELL * len(idx) * len(seeds))


def check(got, want, what):
    assert str(got) == want, f"{what}: computed {got}, expected {want}"


# (1) Re-check the copied counts against results_to_fill.md (if present).
if RESULTS_MD.exists():
    rows, trot_rows = {}, []
    for line in RESULTS_MD.read_text().split("## 3.")[0].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 8 and re.fullmatch(r"s\d", cells[1]):
            rows.setdefault(cells[0], []).append(tuple(int(c) for c in cells[2:]))
        if len(cells) == 8 and cells[0] == TROT and cells[1] == "one run":
            trot_rows.append(tuple(int(c) for c in cells[2:]))
    for method, seeds in OCS2_SEEDS.items():
        assert tuple(rows[method]) == seeds, f"{method}: counts differ from {RESULTS_MD}"
        assert len(seeds) == 5, f"{method}: expected five training seeds"
    assert trot_rows == [TROT_COUNTS], f"{TROT}: counts differ from {RESULTS_MD}"
    print("per-seed counts match", RESULTS_MD.name)
else:
    print(f"note: {RESULTS_MD} not found; per-seed counts not re-checked")

# (2) Pool exactly and check against the spec.
for method, _, macro, macro_dense in BARS:
    check(half_up(pooled(OCS2_SEEDS[method])), SPEC[macro], method)
    check(half_up(pooled(OCS2_SEEDS[method], DENSE)), SPEC[macro_dense], method + ", 49 %")
check(half_up(pooled((TROT_COUNTS,))), SPEC["corlBenchTrot"], TROT)

# (3) The macros of constants.tex are the source of truth: each must be defined
# and equal to the spec (a missing macro is an error: the prose uses them).
MACROS = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}",
                         CONSTANTS_TEX.read_text()))
for name, want in SPEC.items():
    assert name in MACROS, f"\\{name} is not defined in {CONSTANTS_TEX}"
    check(MACROS[name], want, "\\" + name)
check(MACROS["corlBenchTrialsPerCell"], str(N_CELL), "\\corlBenchTrialsPerCell")
check(MACROS["corlNumTrainSeeds"], "five", "\\corlNumTrainSeeds")
print("all value checks passed")

# ---------------------------------------------------------------- style
# rcParams, halo and grid as fig_bench.py; trot colour and dashes as fig_swing.py.
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
INK, GRID, INDIGO, GRAY = "#1F2937", "#E5E7EB", "#4F46E5", "#4B5563"
MIN_PT = 7.0                            # smallest text at final size
VALUE_PT = 7.0                          # value labels, as fig_swing.py's labels
PAD_PT = 1.5                            # value label to bar top
HALO = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
REF_STYLE = dict(color=GRAY, lw=0.8, ls=(0, (3.2, 1.8)))   # thin trot dashes
# Common legend style of the paper's figures (2026-11-13): Fira Sans, ink, no
# frame, short handles. 7.1 pt here prints as 7.0 pt: the 74 mm canvas is
# included at 0.48\textwidth = 72.96 mm.
LEGEND_PT = 7.1
LEGEND_KW = dict(frameon=False, fontsize=LEGEND_PT, labelcolor=INK, handlelength=1.6,
                 handletextpad=0.5, columnspacing=1.2, labelspacing=0.3,
                 borderpad=0.0, borderaxespad=0.0)
LEGEND_X_MM = 2.0                       # legend starts this far right of the y axis
LEGEND_TOP_MM = 0.4                     # legend top below the axes top
LEGEND_CLEAR_MM = 1.5                   # least distance from the legend to a bar or line
LEGEND_ABOVE_AXIS_MM = 0.7              # least height of the legend above the y axis's end
INSIDE_MIN = 15.0                       # % : a bar this tall holds its value inside

# ---------------------------------------------------------------- layout [mm]
MM = 1 / 25.4
FIG_W, FIG_H = 74.0, 42.0               # half \textwidth, beside fig_swing (74 x 42)
TOP, BOTTOM, LEFT, RIGHT = 5.0, 9.0, 10.5, 3.0   # as fig_swing.py: axes align in the row
BAR_MM = 6.5                            # wide enough for "85.1" inside at 7 pt
# The y axis ends at 100 %; the band above it holds the legend row. With 114
# the legend's lower edge is at 103 %, above the axis end. With 106 (until
# 2026-11-13) the row was centred on the 100 % level, next to the "100" tick,
# and its handles read as marks at 100 %. No value changes: the bars are drawn
# 7 % shorter.
Y_AXIS_END = 100
Y_TOP = 114
GRID_AT = (25, 50, 75)

fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax = fig.add_axes([LEFT / FIG_W, BOTTOM / FIG_H,
                   (FIG_W - LEFT - RIGHT) / FIG_W, (FIG_H - TOP - BOTTOM) / FIG_H])

units = (FIG_W - LEFT - RIGHT) / len(BARS)   # mm per bar slot
bar_w = BAR_MM / units                       # bar width in data units
inside = {}                                  # value label -> the bar it sits in
for i, (_, _, macro, _) in enumerate(BARS):
    value = MACROS[macro]               # printed string (= spec, asserted above)
    bar = ax.bar(i, float(value), width=bar_w, facecolor=INDIGO, linewidth=0, zorder=2)[0]
    if float(value) >= INSIDE_MIN:
        t = ax.annotate(value, (i, float(value)), xytext=(0, -PAD_PT),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=VALUE_PT, color="white", zorder=3)
        inside[t] = bar
    else:
        ax.annotate(value, (i, float(value)), xytext=(0, PAD_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=VALUE_PT, color=INK, zorder=3,
                    bbox=HALO)

# OCS2's own trot over the same six cells: behind the bars (zorder 1), above
# the grid (zorder 0). The legend names it (no label on the line).
trot = float(MACROS["corlBenchTrot"])
ref_line = ax.axhline(trot, zorder=1, **REF_STYLE)

ax.set_xlim(-0.5, len(BARS) - 0.5)
ax.set_xticks(range(len(BARS)))
ax.set_xticklabels([tick for _, tick, _, _ in BARS], linespacing=1.0)
ax.tick_params(axis="x", length=0, pad=3)
ax.set_ylim(0, Y_TOP)
ax.set_yticks(range(0, Y_AXIS_END + 1, 25))
ax.spines["left"].set_bounds(0, Y_AXIS_END)
for g in GRID_AT:
    ax.axhline(g, color=GRID, lw=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Success [%]", labelpad=2)
ax.set_title("GaitNet on OCS2, by footstep correction", loc="left", fontsize=7.5,
             fontweight="semibold", color=INK, pad=3)

# Legend: one row in the empty band under the axes top, wholly above the end
# of the y axis (100 %) and so above the tallest bar (85.1 %). Handles repeat
# the drawn marks: an indigo bar and the dashed reference line.
_ax_w, _ax_h = FIG_W - LEFT - RIGHT, FIG_H - TOP - BOTTOM      # mm
legend = ax.legend(handles=[Patch(facecolor=INDIGO, linewidth=0), Line2D([], [], **REF_STYLE)],
                   labels=["GaitNet", TROT], ncol=2, loc="upper left",
                   bbox_to_anchor=(LEGEND_X_MM / _ax_w, 1 - LEGEND_TOP_MM / _ax_h), **LEGEND_KW)


# ---------------------------------------------------------------- layout checks
def layout_problems(fig):
    """Text outside the canvas or too small, text on text, labels on bars or
    spines (an inside value label must lie within its own bar), the
    reference line through any text, and the legend on or too close to any
    mark (as fig_bench.py plus the last three)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]

    def extent(t):
        b, patch = t.get_window_extent(r), t.get_bbox_patch()
        return Bbox.union([b, patch.get_window_extent(r)]) if patch else b

    boxes = [(t, extent(t)) for t in texts]
    values = [(t, extent(t)) for a in fig.axes for t in a.texts]   # incl. the line label
    out, canvas = [], fig.bbox
    for t, b in boxes:
        if not (b.x0 >= canvas.x0 and b.y0 >= canvas.y0
                and b.x1 <= canvas.x1 and b.y1 <= canvas.y1):
            out.append(f"outside canvas: {t.get_text()!r}")
        if t.get_fontsize() < MIN_PT:
            out.append(f"font below {MIN_PT} pt: {t.get_text()!r}")
        if t.get_fontsize() < 6.3:
            out.append(f"font below 6.3 pt: {t.get_text()!r}")
    for i, (t, b) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if b.overlaps(b2):
                out.append(f"text overlap: {t.get_text()!r} / {t2.get_text()!r}")
    margin = 0.5 * fig.dpi / 72                        # 0.5 pt inside the bar edge
    for t, b in values:
        own = inside.get(t)
        for a in fig.axes:
            for p in a.patches:
                pb = p.get_window_extent(r)
                if p is own:
                    if not (b.x0 >= pb.x0 + margin and b.x1 <= pb.x1 - margin
                            and b.y0 >= pb.y0 + margin and b.y1 <= pb.y1):
                        out.append(f"inside label not within its bar: {t.get_text()!r}")
                elif b.overlaps(pb):
                    out.append(f"label on data: {t.get_text()!r}")
            for s in a.spines.values():
                if s.get_visible() and b.overlaps(s.get_window_extent(r)):
                    out.append(f"label on data: {t.get_text()!r}")
    # The dashed line spans the axes width; no text box may meet its stroke.
    y = ax.transData.transform((0, trot))[1]
    half = REF_STYLE["lw"] / 2 * fig.dpi / 72
    ab = ax.get_window_extent(r)
    for t, b in boxes:
        if b.x1 > ab.x0 and b.x0 < ab.x1 and b.y0 < y + half and b.y1 > y - half:
            out.append(f"reference line through text: {t.get_text()!r}")
    # The legend: inside the axes, two entries, above the end of the y axis
    # (not level with the "100" tick), and at least LEGEND_CLEAR_MM
    # from every bar, value label, the reference line and the visible spines;
    # no grid line may run behind it either.
    lg = legend.get_window_extent(r)
    y_end = ax.transData.transform((0, Y_AXIS_END))[1]
    if lg.y0 < y_end + LEGEND_ABOVE_AXIS_MM / 25.4 * fig.dpi:
        out.append(f"legend not above the end of the y axis ({Y_AXIS_END} %)")
    if len(legend.get_texts()) != 2:
        out.append("legend must have two entries")
    if not (lg.x0 >= ab.x0 and lg.x1 <= ab.x1 and lg.y0 >= ab.y0 and lg.y1 <= ab.y1):
        out.append("legend outside the axes")
    clear = LEGEND_CLEAR_MM / 25.4 * fig.dpi
    grown = Bbox.from_extents(lg.x0 - clear, lg.y0 - clear, lg.x1 + clear, lg.y1 + clear)
    for p in ax.patches:
        if grown.overlaps(p.get_window_extent(r)):
            out.append("legend too close to a bar")
    for t, b in values:
        if grown.overlaps(b):
            out.append(f"legend too close to value {t.get_text()!r}")
    for s in ax.spines.values():
        if s.get_visible() and grown.overlaps(s.get_window_extent(r)):
            out.append("legend too close to a spine")
    for level in (trot,) + GRID_AT:
        y_l = ax.transData.transform((0, level))[1]
        if grown.y0 < y_l < grown.y1:
            out.append(f"legend on the line at {level} %")
    return sorted(set(out))


problems = layout_problems(fig)
for p in problems:
    print("LAYOUT:", p)
assert not problems, "layout problems, see above"

fig.savefig(HERE / "fig_transfer.png", dpi=300)
# As fig_bench.py: text to outlines with Inkscape for the paper's PDF.
raw = HERE / "fig_transfer.raw.pdf"
fig.savefig(raw)
if shutil.which("inkscape"):
    subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                    "-o", str(HERE / "fig_transfer.pdf")], check=True, capture_output=True)
    raw.unlink()
else:
    print("warning: inkscape not found; fig_transfer.pdf keeps embedded fonts")
    raw.replace(HERE / "fig_transfer.pdf")
print("wrote", HERE / "fig_transfer.pdf")
