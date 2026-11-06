"""Plot the G125 benchmark for Sec. 5.1 (Fig. fig:bench of the L4DC paper).

PROVENANCE NOTE (2026-11-05): the author states that every learned value
plotted here is the mean over five training seeds at iteration 350, each seed
run on all trials; the per-seed logs are not in this repository. Comments below
that describe a value as a single run or another iteration predate this
statement and are kept as written.

Source of every number: the author's final results of 2026-11-03 (full record on the cluster,
/scratch/sselvaraj/gaitnet-runs/idm/inv/paper_exp/SUMMARY.md),
the G125 tables. The prose macros
\\corlBench* in ../constants.tex carry the same values; this script checks
them against its own arithmetic before it draws anything.

Setting: OCS2 in Gazebo, simulated depth-camera terrain, 12.5 cm tiles,
20 s trials, world seeds 111-210 for every arm. A cell is one void fraction
(26 / 38 / 49 %) at one commanded speed (0.10 / 0.20 m/s) and holds 100
trials, so a cell's success count equals its success rate in percent.
Success: no fall and at least half of v * duration walked.

Panels:
  (a) 0.10 m/s and (b) 0.20 m/s: grouped bars by void fraction, one bar per
      arm (GaitNet, GaitNet-Diagonal, OCS2's own trot), the cell's success
      count printed on top of every bar.
  (c) REMOVED 2026-11-05 at the author's request: it repeated Table 1's
      benchmark column. Its values are still computed and checked below
      (PANEL_C, SPEC_C), because the \\corlBench* macros use them.
  (c, former) All six cells: one horizontal bar per arm, its pooled rate printed at
      the bar end.

Design rule (user, 2026-11-03, after commit 098def2): no error bars, whiskers,
confidence bands or heatmaps. Plain bars starting at 0, with the values a
reader needs printed on the marks. The rerun is not drawn (author,
2026-11-04: "remove gaitnet rerun"); its counts stay below and are checked;
the prose states 95 % intervals instead. History: on 2026-11-04 the per-cell
panels were dropped for a swing diagram, then restored at the author's
request ("then add back the image"); the swing diagram is in App. E.

Arms (names as in the paper):
  GaitNet          = c0, iteration 350 of one of five seeded runs, OCS2 IFM.
  GaitNet, rerun   = c0b, the SAME checkpoint evaluated a second time. It is
                     run-to-run variation of the benchmark, not a training seed.
  GaitNet-Diagonal = #6, diagonal-only second swing.
  OCS2's own trot  = #7.
  Per-candidate + IFM = #4, four training seeds at iteration 1599. The figure
                     shows ONLY the mean over the four seeds (user rule of
                     2026-11-03: no per-seed markers, bars, rows or ranges in
                     figures). The per-seed counts are kept below only to
                     compute that mean and to check them against the source.

Data conventions:
  * Counts are integers. Every rate is Fraction(100 * k, n) rounded half up
    (decimal.ROUND_HALF_UP): integers for cells, one decimal for pools.
  * Per-candidate + IFM appears only in (c): no claim uses its per-cell means,
    and a bare mean next to GaitNet's cells would rank the two networks cell
    by cell, which the paper does not do.

Open \\verify items that touch this figure:
  * c0 at 49 % / 0.20 m/s is 77, confirmed by the author on 2026-11-03 (the
    Gazebo-vs-Isaac table's 87 is superseded). 77 matches the row sums:
    96 + 87 + 77 = 260 of 300 gives 86.7 % at 0.20 m/s and 549/600 = 91.5 %
    pooled.
  * The checkpoint was fixed \\verify{before any trial on OCS2}, and the world
    seeds are disjoint from those that tuned the execution layer, supplied
    the OCS2 IFM's data or developed the camera pipeline
    \\verify{(confirm for the 12.5 cm worlds)}.
  The speed x void sweep's tile size and success criterion are open as
  well, but they do not enter this figure.

usage: python fig_bench.py   (writes fig_bench.pdf and fig_bench.png here)
"""
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"

# ---------------------------------------------------------------- data
VOIDS = (26, 38, 49)                    # void fraction of a cell [%]
N_CELL = 100                            # \corlBenchTrialsPerCell
# Success [%] per cell, mean of five training seeds rounded to whole percent
# (author, 2026-11-05), equal to successes of N_CELL for one run; pooled = mean of
# the cells. Order: 0.10 m/s at 26/38/49 % voids, then 0.20 m/s.
CELLS = {
    "GaitNet":          (100, 93, 96, 96, 87, 77),   # c0; 77, not 87 (see header)
    "GaitNet, rerun":   (96, 97, 92, 96, 84, 74),    # c0b, same checkpoint
    "GaitNet-Diagonal": (99, 96, 89, 96, 84, 59),    # #6
    "OCS2's own trot":  (97, 89, 80, 94, 81, 63),    # #7
    "OCS2's static walk": (98, 91, 80, 90, 73, 51), # author, 2026-11-05 (deterministic, one run)
}
# Per-candidate + IDM (#4), one tuple per training seed (1-4), same order.
# Used only for the mean over seeds; no seed is drawn.
PER_CANDIDATE_SEEDS = (
    (98, 91, 82, 92, 80, 70),
    (98, 95, 87, 93, 84, 68),
    (99, 93, 88, 93, 88, 77),
    (98, 87, 79, 83, 63, 43),
)


# ---------------------------------------------------------------- arithmetic
def half_up(x: Fraction, places: str = "0.1") -> Decimal:
    """Round an exact fraction half up, e.g. places='0.1' or '1'."""
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
        Decimal(places), rounding=ROUND_HALF_UP)


def rate(k: int, n: int) -> Fraction:
    """Exact success rate [%]."""
    return Fraction(100 * k, n)


def pool(counts, idx) -> tuple[int, int]:
    """(successes, trials) over the cells at positions idx."""
    return sum(counts[i] for i in idx), N_CELL * len(idx)


SLOW, FAST, ALL = (0, 1, 2), (3, 4, 5), tuple(range(6))
BAR_ARMS = ("GaitNet", "GaitNet-Diagonal", "OCS2's own trot", "OCS2's static walk")   # (a), (b)

# (a), (b): per arm, the three cell counts (= rates in %, n = 100).
PANEL_A = {a: CELLS[a][:3] for a in BAR_ARMS}
PANEL_B = {a: CELLS[a][3:] for a in BAR_ARMS}
# (c): per arm, the exact pooled rate over all six cells.
MLP_K = sum(sum(s) for s in PER_CANDIDATE_SEEDS)
MLP_N = N_CELL * 6 * len(PER_CANDIDATE_SEEDS)
# Equal trials per seed, so the pooled rate equals the mean of seed rates.
MLP_MEAN = rate(MLP_K, MLP_N)
assert MLP_MEAN == sum(rate(sum(s), 600) for s in PER_CANDIDATE_SEEDS) / 4
PANEL_C = {a: rate(*pool(CELLS[a], ALL)) for a in CELLS}
PANEL_C["Per-candidate + IFM"] = MLP_MEAN


# ---------------------------------------------------------------- checks
def check(got, want, what):
    """Strings are compared as printed (so 90.0 is not 90), others by value."""
    ok = str(got) == want if isinstance(want, str) else got == want
    assert ok, f"{what}: computed {got}, expected {want}"


# Data entry against the source's per-speed (integer) and pooled columns.
SOURCE_ROWS = {"GaitNet": ("96", "87", "91.5"), "GaitNet, rerun": ("95", "85", "89.8"),
               "GaitNet-Diagonal": ("95", "80", "87.2"),
               "OCS2's own trot": ("89", "79", "84.0"),
               "OCS2's static walk": ("90", "71", "80.5")}
for arm, (s10, s20, s_all) in SOURCE_ROWS.items():
    check(half_up(rate(*pool(CELLS[arm], SLOW)), "1"), s10, f"{arm} 0.10")
    check(half_up(rate(*pool(CELLS[arm], FAST)), "1"), s20, f"{arm} 0.20")
    check(half_up(rate(*pool(CELLS[arm], ALL))), s_all, f"{arm} all")
for seed, want in zip(PER_CANDIDATE_SEEDS, ("85.5", "87.5", "89.7", "75.5")):
    check(half_up(rate(sum(seed), 600)), want, "per-candidate seed")
check(half_up(rate(sum(sum(s[:3]) for s in PER_CANDIDATE_SEEDS), 1200)), "91.3", "MLP 0.10")
check(half_up(rate(sum(sum(s[3:]) for s in PER_CANDIDATE_SEEDS), 1200)), "77.8", "MLP 0.20")
check(MLP_K, 2029, "per-candidate successes")


def cell_text(k: int) -> str:
    """Printed value of one cell: its count of 100, i.e. its rate in %."""
    return str(half_up(rate(k, N_CELL), "1"))


def pool_text(r: Fraction) -> str:
    """Printed value of a pooled rate: one decimal."""
    return str(half_up(r))


# Every value the panel map prints, before plotting.
SPEC_A = {"GaitNet": ("100", "93", "96"), "GaitNet-Diagonal": ("99", "96", "89"),
          "OCS2's own trot": ("97", "89", "80"), "OCS2's static walk": ("98", "91", "80")}
SPEC_B = {"GaitNet": ("96", "87", "77"), "GaitNet-Diagonal": ("96", "84", "59"),
          "OCS2's own trot": ("94", "81", "63"), "OCS2's static walk": ("90", "73", "51")}
SPEC_C = {"GaitNet": "91.5", "GaitNet, rerun": "89.8", "GaitNet-Diagonal": "87.2",
          "Per-candidate + IFM": "84.5", "OCS2's own trot": "84.0",
          "OCS2's static walk": "80.5"}
for panel, spec, name in ((PANEL_A, SPEC_A, "(a)"), (PANEL_B, SPEC_B, "(b)")):
    for arm, want in spec.items():
        check(tuple(cell_text(k) for k in panel[arm]), want, f"{name} {arm}")
for arm, want in SPEC_C.items():
    check(pool_text(PANEL_C[arm]), want, f"(c) {arm}")
check(sorted(PANEL_C), sorted(SPEC_C), "(c) rows")

# The printed pooled values must equal the prose macros of constants.tex. The
# file belongs to the prose; a macro it no longer defines is reported, not
# fatal, but a macro it defines with another value is an error.
MACROS = dict(re.findall(r"\\newcommand\{\\(corlBench\w+)\}\{([^}]*)\}",
                         CONSTANTS_TEX.read_text()))
for macro, arm in {"Full": "GaitNet", "Rerun": "GaitNet, rerun",
                   "Diag": "GaitNet-Diagonal", "Trot": "OCS2's own trot",
                   "MlpMean": "Per-candidate + IFM", "StaticWalk": "OCS2's static walk"}.items():
    name = "corlBench" + macro
    if name in MACROS:
        check(pool_text(PANEL_C[arm]), MACROS[name], "\\" + name)
    else:
        print(f"note: \\{name} is not defined in constants.tex; not checked")
if "corlBenchTrialsPerCell" in MACROS:
    check(N_CELL, int(MACROS["corlBenchTrialsPerCell"]), "\\corlBenchTrialsPerCell")
print("all value checks passed")

# ---------------------------------------------------------------- style
# Same font and export path as plot_ocs2_results.py and Fig. 1.
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
INK, GRID = "#1F2937", "#E5E7EB"
INDIGO, GRAY, TEAL = "#4F46E5", "#4B5563", "#0D9488"
LIGHT_INDIGO = "#A5B4FC"
VALUE_PT = 6.3                          # value labels on the bars
PAD_PT = 1.5                            # gap between a bar end and its value [pt]
# A white box behind every value label cuts the light grid lines where a label
# crosses one (the label of the 63 % bar in (b) reaches the 75 % line). Its
# pad (0.1 x 7 pt = 0.7 pt) is less than PAD_PT, so it never touches a bar.
HALO = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
# GaitNet and the trot are both dark (#4F46E5 and #4B5563 have about the same
# luminance), so GaitNet-Diagonal is light: every group reads dark-light-dark
# in the same order, in colour and in grey.
BAR_STYLE = {
    "GaitNet":             dict(facecolor=INDIGO, edgecolor=INDIGO),
    "GaitNet, rerun":      dict(facecolor="white", edgecolor=INDIGO),
    "GaitNet-Diagonal":    dict(facecolor=LIGHT_INDIGO, edgecolor=LIGHT_INDIGO),
    "Per-candidate + IFM": dict(facecolor=TEAL, edgecolor=TEAL),
    "OCS2's own trot":     dict(facecolor=GRAY, edgecolor=GRAY),
    "OCS2's static walk":  dict(facecolor="#B8BEC8", edgecolor="#B8BEC8"),
}
EDGE_LW = 0.8                           # bar edge width (all bars filled; the rerun is not drawn)

# ---------------------------------------------------------------- layout [mm]
FIG_W, FIG_H = 152.0, 42.0              # \textwidth of the PMLR class x 42 mm
LEFT, RIGHT, TOP, BOTTOM = 10.0, 3.0, 7.6, 8.8
GAP = 4.0                               # between (a) and (b)
# Two equal panels; wspace is a fraction of the panel width.
_panel_w = (FIG_W - LEFT - RIGHT - GAP) / 2
WSPACE = GAP / _panel_w

SLOT, BAR_W = 0.20, 0.17               # bar pitch and width within a group [groups]
OFFSET = {"GaitNet": -1.5 * SLOT, "GaitNet-Diagonal": -0.5 * SLOT,
          "OCS2's own trot": 0.5 * SLOT, "OCS2's static walk": 1.5 * SLOT}
Y_TOP = 114                             # headroom for the labels on 100 % bars
GRID_AT = (25, 50, 75)                  # no line at 100: labels of 93-100 % bars cross it


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="semibold", color=INK, pad=3)


def draw_cells(ax, panel):
    """Grouped bars by void fraction, the cell's value on top of every bar."""
    for arm, counts in panel.items():
        for i, k in enumerate(counts):
            x = i + OFFSET[arm]
            ax.bar(x, k, width=BAR_W, linewidth=0, zorder=2, **BAR_STYLE[arm])
            ax.annotate(cell_text(k), (x, k), xytext=(0, PAD_PT),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=VALUE_PT, color=INK, zorder=3, bbox=HALO)
    ax.set_xlim(-0.5, len(VOIDS) - 0.5)
    ax.set_xticks(range(len(VOIDS)))
    ax.set_xticklabels([str(v) for v in VOIDS])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlabel("Void fraction [%]", labelpad=2)
    ax.set_ylim(0, Y_TOP)
    ax.set_yticks(range(0, 101, 25))
    ax.spines["left"].set_bounds(0, 100)
    for g in GRID_AT:
        ax.axhline(g, color=GRID, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


MM = 1 / 25.4
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
# Columns: (a), (b).
gs = fig.add_gridspec(1, 2, wspace=WSPACE)
fig.subplots_adjust(left=LEFT / FIG_W, right=1 - RIGHT / FIG_W,
                    top=1 - TOP / FIG_H, bottom=BOTTOM / FIG_H)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1], sharey=ax_a)

draw_cells(ax_a, PANEL_A)
draw_cells(ax_b, PANEL_B)
ax_a.set_ylabel("Success [%]", labelpad=2)
ax_b.tick_params(labelleft=False)
title(ax_a, "(a) 0.10 m/s")
title(ax_b, "(b) 0.20 m/s")


# Legend in its own strip above (a) and (b): every bar in (a) and (b) reaches
# at least 59 %, so no area inside the panels is free.
_x_mid = (ax_a.get_position().x0 + ax_b.get_position().x1) / 2
fig.legend(handles=[Patch(label=a, linewidth=0, **BAR_STYLE[a]) for a in BAR_ARMS],
           loc="upper center", bbox_to_anchor=(_x_mid, 1.0), ncol=4, frameon=False,
           fontsize=7, handlelength=1.0, handleheight=0.9, handletextpad=0.4,
           columnspacing=1.6, borderaxespad=0.0, borderpad=0.15)


# ---------------------------------------------------------------- layout checks
def data_boxes(fig, renderer, grid=True):
    """Display-space boxes of every bar, visible spine and (if grid) grid line."""
    boxes = []
    for ax in fig.axes:
        boxes += [(ax, p.get_window_extent(renderer)) for p in ax.patches]
        if grid:
            boxes += [(ax, ln.get_window_extent(renderer)) for ln in ax.lines]
        boxes += [(ax, s.get_window_extent(renderer)) for s in ax.spines.values()
                  if s.get_visible()]
    return boxes


def layout_problems(fig):
    """Text outside the canvas or too small, text on text, labels on data."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]

    def extent(t):
        """The text's box, grown to its white halo if it has one."""
        b, patch = t.get_window_extent(r), t.get_bbox_patch()
        return Bbox.union([b, patch.get_window_extent(r)]) if patch else b

    boxes = [(t, extent(t)) for t in texts]
    out, canvas = [], fig.bbox
    for t, b in boxes:
        if not (b.x0 >= canvas.x0 and b.y0 >= canvas.y0
                and b.x1 <= canvas.x1 and b.y1 <= canvas.y1):
            out.append(f"outside canvas: {t.get_text()!r}")
        if t.get_fontsize() < 7 and t not in [v for ax in fig.axes for v in ax.texts]:
            out.append(f"label font below 7 pt: {t.get_text()!r}")
        if t.get_fontsize() < 6.3:
            out.append(f"font below 6.3 pt: {t.get_text()!r}")
    for i, (t, b) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if b.overlaps(b2):
                out.append(f"text overlap: {t.get_text()!r} / {t2.get_text()!r}")
    values = [(t, extent(t)) for ax in fig.axes for t in ax.texts]
    legend = fig.legends[0].get_window_extent(r)
    # Value labels may pass over a grid line: their white box (zorder 3) hides
    # it (zorder 0). Bars and spines they must clear.
    for ax, b2 in data_boxes(fig, r, grid=False):
        out += [f"value label on data: {t.get_text()!r}" for t, b in values
                if b.overlaps(b2)]
    for ax, b2 in data_boxes(fig, r):
        if legend.overlaps(b2):
            out.append("legend covers data")
    out += [f"legend covers {t.get_text()!r}" for t, b in boxes
            if t.figure is fig and t not in fig.legends[0].get_texts()
            and legend.overlaps(b)]
    return sorted(set(out))


problems = layout_problems(fig)
for p in problems:
    print("LAYOUT:", p)
assert not problems, "layout problems, see above"

fig.savefig(HERE / "fig_bench.png", dpi=300)
# Matplotlib embeds the OpenType (CFF) Fira Sans in a way some PDF checkers
# flag, so the PDF that goes into the paper has its text converted to
# outlines with Inkscape, like Fig. 1 and Fig. 2.
raw = HERE / "fig_bench.raw.pdf"
fig.savefig(raw)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                "-o", str(HERE / "fig_bench.pdf")], check=True, capture_output=True)
raw.unlink()
print("wrote", HERE / "fig_bench.pdf")
