"""Plot the G125 benchmark for Sec. 5.1 (Fig. fig:bench of the L4DC paper).

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

Panels (since 2026-11-04; author: drop the per-cell panels, the sweep of
Fig. fig:sweep covers the trend over void fractions):
  (a) All six cells: one horizontal bar per arm, its pooled rate printed at
      the bar end. (b) and (c) of the same figure are fig_gait.pdf, placed to
      the right in LaTeX. The per-cell counts are no longer drawn; they are
      still checked below and printed for App. D. The former panels showed
      GaitNet, GaitNet-Diagonal and the trot per cell at 0.10 and 0.20 m/s.

Design rule (user, 2026-11-03, after commit 098def2): no error bars, whiskers,
confidence bands or heatmaps. Plain bars starting at 0, with the values a
reader needs printed on the marks. Run-to-run variation is shown only by the
rerun bar in (a) and is put into words in the prose.

Arms (names as in the paper):
  GaitNet          = c0, iteration 350 of one of five seeded runs, OCS2 IDM.
  GaitNet, rerun   = c0b, the SAME checkpoint evaluated a second time. It is
                     run-to-run variation of the benchmark, not a training seed.
  GaitNet-Diagonal = #6, diagonal-only second swing.
  OCS2's own trot  = #7.
  Per-candidate + IDM = #4, four training seeds at iteration 1599. The figure
                     shows ONLY the mean over the four seeds (user rule of
                     2026-11-03: no per-seed markers, bars, rows or ranges in
                     figures). The per-seed counts are kept below only to
                     compute that mean and to check them against the source.

Data conventions:
  * Counts are integers. Every rate is Fraction(100 * k, n) rounded half up
    (decimal.ROUND_HALF_UP): integers for cells, one decimal for pools.
  * Per-candidate + IDM appears only pooled: no claim uses its per-cell means,
    and a bare mean next to GaitNet's cells would rank the two networks cell
    by cell, which the paper does not do.

Open \\verify items that touch this figure:
  * c0 at 49 % / 0.20 m/s is 77, confirmed by the author on 2026-11-03 (the
    Gazebo-vs-Isaac table's 87 is superseded). 77 matches the row sums:
    96 + 87 + 77 = 260 of 300 gives 86.7 % at 0.20 m/s and 549/600 = 91.5 %
    pooled.
  * The checkpoint was fixed \\verify{before any trial on OCS2}, and the world
    seeds are disjoint from those that tuned the execution layer, supplied
    the OCS2 IDM's data or developed the camera pipeline
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
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"

# ---------------------------------------------------------------- data
VOIDS = (26, 38, 49)                    # void fraction of a cell [%]
N_CELL = 100                            # \corlBenchTrialsPerCell
# Successes out of N_CELL. Order: 0.10 m/s at 26/38/49 % voids, then 0.20 m/s.
CELLS = {
    "GaitNet":          (100, 93, 96, 96, 87, 77),   # c0; 77, not 87 (see header)
    "GaitNet, rerun":   (96, 97, 92, 96, 84, 74),    # c0b, same checkpoint
    "GaitNet-Diagonal": (99, 96, 89, 96, 84, 59),    # #6
    "OCS2's own trot":  (97, 89, 80, 94, 81, 63),    # #7
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
BAR_ARMS = ("GaitNet", "GaitNet-Diagonal", "OCS2's own trot")   # per-cell checks (App. D)

# Per arm, the three cell counts (= rates in %, n = 100) at 0.10 and at 0.20 m/s;
# no longer drawn (former panels a, b), kept for the checks and App. D.
PANEL_A = {a: CELLS[a][:3] for a in BAR_ARMS}
PANEL_B = {a: CELLS[a][3:] for a in BAR_ARMS}
# Panel (a): per arm, the exact pooled rate over all six cells.
MLP_K = sum(sum(s) for s in PER_CANDIDATE_SEEDS)
MLP_N = N_CELL * 6 * len(PER_CANDIDATE_SEEDS)
# Equal trials per seed, so the pooled rate equals the mean of seed rates.
MLP_MEAN = rate(MLP_K, MLP_N)
assert MLP_MEAN == sum(rate(sum(s), 600) for s in PER_CANDIDATE_SEEDS) / 4
PANEL_C = {a: rate(*pool(CELLS[a], ALL)) for a in CELLS}
PANEL_C["Per-candidate + IDM"] = MLP_MEAN


# ---------------------------------------------------------------- checks
def check(got, want, what):
    """Strings are compared as printed (so 90.0 is not 90), others by value."""
    ok = str(got) == want if isinstance(want, str) else got == want
    assert ok, f"{what}: computed {got}, expected {want}"


# Data entry against the source's per-speed (integer) and pooled columns.
SOURCE_ROWS = {"GaitNet": ("96", "87", "91.5"), "GaitNet, rerun": ("95", "85", "89.8"),
               "GaitNet-Diagonal": ("95", "80", "87.2"),
               "OCS2's own trot": ("89", "79", "84.0")}
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
          "OCS2's own trot": ("97", "89", "80")}
SPEC_B = {"GaitNet": ("96", "87", "77"), "GaitNet-Diagonal": ("96", "84", "59"),
          "OCS2's own trot": ("94", "81", "63")}
SPEC_C = {"GaitNet": "91.5", "GaitNet, rerun": "89.8", "GaitNet-Diagonal": "87.2",
          "Per-candidate + IDM": "84.5", "OCS2's own trot": "84.0"}
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
                   "MlpMean": "Per-candidate + IDM"}.items():
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
VALUE_PT = 7.0                          # value labels on the bars
ROW_PT = 7.0                            # row labels of (a)
PAD_PT = 1.5                            # gap between a bar end and its value [pt]
# A white box behind every value label cuts the light grid lines where a label
# crosses one. Its
# pad (0.1 x 7 pt = 0.7 pt) is less than PAD_PT, so it never touches a bar.
HALO = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
# GaitNet and the trot are both dark (#4F46E5 and #4B5563 have about the same
# luminance), GaitNet-Diagonal light. Light indigo means GaitNet-Diagonal only:
# fig_gait.py draws both GaitNet speeds in (c) in indigo.
BAR_STYLE = {
    "GaitNet":             dict(facecolor=INDIGO, edgecolor=INDIGO),
    "GaitNet, rerun":      dict(facecolor="white", edgecolor=INDIGO),
    "GaitNet-Diagonal":    dict(facecolor=LIGHT_INDIGO, edgecolor=LIGHT_INDIGO),
    "Per-candidate + IDM": dict(facecolor=TEAL, edgecolor=TEAL),
    "OCS2's own trot":     dict(facecolor=GRAY, edgecolor=GRAY),
}
EDGE_LW = 0.8                           # visible only on the open rerun bar

# ---------------------------------------------------------------- layout [mm]
# Since 2026-11-04 (author: drop the per-cell panels, the sweep covers the void
# trend) this script draws only the pooled panel, as (a) of a figure whose (b)
# and (c) are fig_gait.pdf. The two PDFs sit side by side in one figure
# environment, each scaled by \textwidth / 152 mm like every other figure, so
# their heights and their TOP and BOTTOM margins must match fig_gait.py's
# (checked there). The per-cell counts above stay: they are checked against
# the source and printed in App. D (printed below for that paragraph).
FIG_W, FIG_H = 52.0, 40.0               # left part of the 152 mm figure
LEFT, RIGHT, TOP, BOTTOM = 22.0, 4.0, 5.0, 9.0   # LEFT: the row labels
XTICK_PAD = 3.5                         # pt; must equal fig_gait.py's (x labels on one line)
GRID_AT = (25, 50, 75)                  # light vertical lines behind the bars


def title(ax, text):
    """Panel title, left-aligned with the figure's left edge (over the row labels)."""
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="semibold", color=INK, pad=3,
                 x=-(LEFT - 1.0) / (FIG_W - LEFT - RIGHT))


MM = 1 / 25.4
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax_c = fig.add_axes([LEFT / FIG_W, BOTTOM / FIG_H,
                     (FIG_W - LEFT - RIGHT) / FIG_W, (FIG_H - TOP - BOTTOM) / FIG_H])

# All six cells per arm, one row each, top to bottom; the pooled rate right of
# the bar end. The rerun is the same checkpoint as GaitNet, so it keeps
# GaitNet's colour as an open bar.
ROWS = ("GaitNet", "GaitNet, rerun", "GaitNet-Diagonal", "Per-candidate + IDM",
        "OCS2's own trot")
# One two-line label only: rows are 5.2 mm apart, too close for two adjacent
# two-line labels. The caption says what the rerun and the seed mean are.
ROW_LABELS = {"GaitNet, rerun": "GaitNet rerun",
              "Per-candidate + IDM": "Per-candidate\n+ IDM"}
for r, arm in enumerate(ROWS):
    y, v = len(ROWS) - 1 - r, PANEL_C[arm]
    ax_c.barh(y, float(v), height=0.62, linewidth=EDGE_LW, zorder=2, **BAR_STYLE[arm])
    ax_c.annotate(pool_text(v), (float(v), y), xytext=(EDGE_LW / 2 + PAD_PT, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=VALUE_PT, color=INK, zorder=3, bbox=HALO)
ax_c.set_ylim(-0.55, len(ROWS) - 0.45)
ax_c.set_yticks(range(len(ROWS))[::-1])
ax_c.set_yticklabels([ROW_LABELS.get(a, a) for a in ROWS], fontsize=ROW_PT,
                     linespacing=1.0)
ax_c.tick_params(axis="y", length=0, pad=3)
ax_c.set_xlim(0, 100)
ax_c.set_xticks(range(0, 101, 25))
ax_c.set_xticklabels(["0", "", "50", "", "100"])
ax_c.tick_params(axis="x", pad=XTICK_PAD)
ax_c.spines["bottom"].set_bounds(0, 100)
ax_c.set_xlabel("Success [%]", labelpad=2)
for g in GRID_AT:
    ax_c.axvline(g, color=GRID, lw=0.5, zorder=0)
ax_c.spines[["top", "right"]].set_visible(False)
title(ax_c, "(a) Success, all six cells")

# The per-cell counts for App. D (one sentence per arm, 0.10 then 0.20 m/s).
for arm in ("GaitNet", "GaitNet, rerun", "GaitNet-Diagonal", "OCS2's own trot"):
    print(f"App. D {arm}: " + ", ".join(cell_text(k) for k in CELLS[arm]))


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
    # Value labels may pass over a grid line: their white box (zorder 3) hides
    # it (zorder 0). Bars and spines they must clear.
    for ax, b2 in data_boxes(fig, r, grid=False):
        out += [f"value label on data: {t.get_text()!r}" for t, b in values
                if b.overlaps(b2)]
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
