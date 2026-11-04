"""Plot the OCS2 void x speed sweep (Fig. fig:sweep, Sec. 5.2 of the L4DC paper).

Source of every number: the author's final results of 2026-11-03 (full record on the cluster,
/scratch/sselvaraj/gaitnet-runs/idm/inv/paper_exp/SUMMARY.md):
the speed x void sweep table, and the trot's "existing data" of the earlier
benchmark. Macros that must agree with the printed values live in
constants.tex (\\corlSweep*, \\corlOcsTrialsPerPoint); this script checks them
before it draws anything.

Design rule (user, 2026-11-03: "the bx and wisker is not good readapple ...
head map is notr good"): no error bars, whiskers, confidence bands or
heatmaps. Lines with markers over the void fraction, with the values a reader
needs printed next to the marks.

Panels
- (a) Success vs void fraction, GaitNet against OCS2's own trot. Every point
  carries its value (one decimal, as for the pooled rates of Fig. 2; an
  exact 100 %, the trot's 80 of 80 at 0 and 10 %, is printed as "100"):
  GaitNet's in indigo, mostly above its line, the trot's in gray, mostly
  below. Positions are chosen by a collision check (see "(a) values").
- (b) GaitNet only, one line per commanded speed. The legend carries each
  command's success over all eight void fractions (integers, as in the
  prose macros). The 48 per-cell values are not printed: they do not fit.
  The legend sits in the empty lower left of (b) (added 2026-11-04, to free
  the right edge for (c)); dashed lines are the commands above the training
  range, which the caption states.
- (c) The EARLIER benchmark (author's data of 2026-10-14, the figure of the
  2026-10-28 draft, git 486a92a:figures/plot_ocs2_results.py), added at the
  author's request on 2026-11-04: an earlier GaitNet checkpoint (about 600
  iterations of a seeded run of the same preset) and its variants, on the
  SAME worlds as the trot of (a): GaitNet, GaitNet-Single (one leg in swing
  at a time), Per-candidate + IDM (one training seed), GaitNet without the
  IDM, and OCS2's own trot. One bar each: success over the whole void sweep
  (8 void fractions x 80 trials = 640 trials, spread over 0.05-0.25 m/s).
  The earlier checkpoint is NOT the benchmark checkpoint of (a), (b) and
  Figs. 2 and 4; compare the bars of (c) only with each other.

Data conventions
- GaitNet is the benchmark checkpoint c0 (iteration 350, OCS2 IDM, depth-camera
  terrain). Its sweep has 100 trials per cell over 8 void fractions (0-70 %) x
  6 commanded forward speeds (0.05-0.30 m/s). Data are stored as integer
  success counts out of 100, so a cell's count is its success rate in %.
- Panel (a), GaitNet: per void fraction, pooled over the five commands
  0.05-0.25 m/s (500 trials per point) to match the trot's void sweep.
- Panel (a), OCS2's own trot: the EARLIER benchmark's void sweep (author:
  "the trot sweep is the existing data"), 80 trials per point spread over
  0.05-0.25 m/s, on different worlds from GaitNet's sweep. The figure labels it
  "earlier runs, other worlds"; the comparison is therefore across worlds.
- Panel (b): GaitNet's 48 cells, one line per command, plus the all-voids
  value per command pooled over the 8 void fractions (800 trials per
  command; equal trials per cell, so it is also the mean of the 8 cell
  rates). Printed as integers so that they equal \\corlSweepFullSlowest (88),
  \\corlSweepFullTwentyFive (72) and \\corlSweepFullThirty (62).
- Commands 0.25 and 0.30 m/s are above the 0.2 m/s training range
  (\\corlCmdVxMax); they are dashed and grouped under "above training range"
  in the legend.
- Every derived number is computed from integer counts with fractions.Fraction
  and rounded half up (decimal.ROUND_HALF_UP); Python's round() is not used for
  any printed or asserted value.

Open \\verify items that affect this figure
- The sweep's tile size and success criterion are unconfirmed. The text assumes
  the earlier benchmark's 24 cm tiles and "reach the end of a 3 m track within
  1.5x the commanded time"; both stay under \\verify in the paper.
- If the author says GaitNet's sweep did NOT use the earlier benchmark's tiles
  and criterion, the plan's fallback (fidelity R12b) re-renders (a) with GaitNet
  only: set SHOW_TROT = False below.
- The trot's SPEED sweep (63 63 59 51 41 of 80) is not plotted because its
  void fractions are unconfirmed (\\verify in the 5.2 prose; counts in App. D).
- Not used here: the 12.5 cm benchmark cells (incl. c0's 77 at 49 % / 0.20 m/s)
  and the Isaac comparison belong to Figs. 2 and 4, not to this sweep.

No per-seed information: the sweep is a single checkpoint, so every marker is
one run of that checkpoint.

usage: ~/miniforge3/envs/manim/bin/python -B fig_sweep.py
       (writes fig_sweep.pdf and fig_sweep.png next to this file)
"""
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"
SHOW_TROT = True        # fidelity R12b fallback: False plots GaitNet only in (a)

# ---------------------------------------------------------------- data
VOIDS = [0, 10, 20, 30, 40, 50, 60, 70]                 # void fraction [%]
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]           # commanded speed [m/s]
TRAINED_MAX_SPEED = 0.2                                 # \corlCmdVxMax
# GaitNet (c0) successes out of 100 per cell; rows = VOIDS, columns = SPEEDS.
GRID = [
    [100, 100, 100, 100, 99, 99],
    [100, 100, 99, 99, 100, 98],
    [98, 100, 97, 99, 96, 93],
    [98, 96, 96, 97, 89, 79],
    [89, 97, 99, 93, 87, 63],
    [89, 87, 87, 76, 60, 38],
    [79, 71, 68, 50, 34, 21],
    [50, 34, 23, 17, 8, 2],
]
TRIALS_PER_CELL = 100                                   # \corlSweepTrialsPerCell
N_POOLED_SPEEDS = 5                                     # 0.05-0.25 m/s in (a)
TROT_BY_VOID = [80, 80, 70, 60, 56, 30, 24, 10]         # successes of 80
TROT_TRIALS = 80                                        # \corlOcsTrialsPerPoint
# (c) Earlier benchmark, success counts of 80 per void fraction (0-70 %),
# recovered from the 2026-10-28 figure's percentages (each maps to exactly one
# count under round-half-even); totals equal that draft's macros (529, 381,
# 352, 471 and 410 of 640).
EARLY_TRIALS = 80
EARLY_BY_VOID = {
    "GaitNet": [80, 80, 72, 72, 73, 64, 48, 40],
    "GaitNet-Single": [69, 59, 66, 53, 54, 34, 24, 22],
    "Per-candidate + IDM": [78, 74, 75, 56, 53, 56, 42, 37],
    "GaitNet, no IDM": [77, 64, 51, 45, 35, 30, 29, 21],
    "OCS2's own trot": TROT_BY_VOID,
}
EARLY_PCT_SOURCE = {   # the percentages printed in the 2026-10-28 figure
    "GaitNet": [100, 100, 90, 90, 91, 80, 60, 50],
    "GaitNet-Single": [86, 74, 82, 66, 68, 42, 30, 28],
    "Per-candidate + IDM": [98, 92, 94, 70, 66, 70, 52, 46],
    "GaitNet, no IDM": [96, 80, 64, 56, 44, 38, 36, 26],
    "OCS2's own trot": [100, 100, 88, 75, 70, 38, 30, 12],
}
for _name, _ks in EARLY_BY_VOID.items():
    assert [round(100 * k / EARLY_TRIALS) for k in _ks] == EARLY_PCT_SOURCE[_name], _name
EARLY_N = len(VOIDS) * EARLY_TRIALS                    # 640 per bar


def half_up(x: Fraction, places: str = "0.1") -> Decimal:
    """Round an exact fraction half up to the given decimal places."""
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
        Decimal(places), ROUND_HALF_UP)


def pct(k: int, n: int) -> Fraction:
    """Exact success rate [%] of k successes out of n trials."""
    return Fraction(100 * k, n)


# Derived series.
N_VOID = N_POOLED_SPEEDS * TRIALS_PER_CELL              # 500, \corlSweepTrialsPerVoid
N_SPEED = len(VOIDS) * TRIALS_PER_CELL                  # 800, \corlSweepTrialsPerSpeed
by_void_k = [sum(row[:N_POOLED_SPEEDS]) for row in GRID]
by_speed_k = [sum(row[j] for row in GRID) for j in range(len(SPEEDS))]
by_void_pct = [half_up(pct(k, N_VOID)) for k in by_void_k]
trot_pct = [half_up(pct(k, TROT_TRIALS)) for k in TROT_BY_VOID]
all_voids_int = [half_up(pct(k, N_SPEED), "1") for k in by_speed_k]
dense = slice(4, 8)                                     # 40-70 % voids
dense_k, dense_n = sum(by_void_k[dense]), 4 * N_VOID
trot_dense_k, trot_dense_n = sum(TROT_BY_VOID[dense]), 4 * TROT_TRIALS

# The strings drawn on the figure. One decimal, except that an exact 100 %
# (k == n, the trot's 80 of 80 at 0 and 10 %) is printed as "100": it is
# exact, and the shorter label fits between the two curves at 0-10 %.


def value_label(k: int, n: int) -> str:
    return "100" if k == n else str(half_up(pct(k, n)))


LABELS_A_GAITNET = [value_label(k, N_VOID) for k in by_void_k]
LABELS_A_TROT = [value_label(k, TROT_TRIALS) for k in TROT_BY_VOID]
LEGEND_B = [f"{SPEEDS[0]:.2f} m/s ({all_voids_int[0]} %)"] + [
    f"{s:.2f} ({v} %)" for s, v in zip(SPEEDS[1:], all_voids_int[1:])]

# ---------------------------------------------------------------- asserts
# Every value the spec lists, checked before plotting.
D = lambda s: [Decimal(v) for v in s.split()]  # noqa: E731
assert len(GRID) == 8 and all(len(r) == 6 for r in GRID)
assert all(0 <= c <= TRIALS_PER_CELL for r in GRID for c in r)
# Data entry against the source table, row by row (rows = void 0..70 %).
assert [" ".join(map(str, r)) for r in GRID] == [
    "100 100 100 100 99 99", "100 100 99 99 100 98", "98 100 97 99 96 93",
    "98 96 96 97 89 79", "89 97 99 93 87 63", "89 87 87 76 60 38",
    "79 71 68 50 34 21", "50 34 23 17 8 2"]
assert TROT_BY_VOID == [80, 80, 70, 60, 56, 30, 24, 10] and TROT_TRIALS == 80
assert by_void_k == [499, 498, 490, 476, 465, 399, 302, 132], by_void_k
assert by_void_pct == D("99.8 99.6 98.0 95.2 93.0 79.8 60.4 26.4"), by_void_pct
assert trot_pct == D("100.0 100.0 87.5 75.0 70.0 37.5 30.0 12.5"), trot_pct
assert LABELS_A_GAITNET == "99.8 99.6 98.0 95.2 93.0 79.8 60.4 26.4".split()
assert LABELS_A_TROT == "100 100 87.5 75.0 70.0 37.5 30.0 12.5".split()
assert by_speed_k == [703, 685, 669, 631, 573, 493], by_speed_k
assert all_voids_int == D("88 86 84 79 72 62"), all_voids_int
assert LEGEND_B == ["0.05 m/s (88 %)", "0.10 (86 %)", "0.15 (84 %)",
                    "0.20 (79 %)", "0.25 (72 %)", "0.30 (62 %)"], LEGEND_B
# Equal trials per cell: the pooled all-voids rate is the mean of the cell rates.
for j, k in enumerate(by_speed_k):
    assert pct(k, N_SPEED) == sum(Fraction(r[j]) for r in GRID) / len(VOIDS)
# The prose's derived numbers (not drawn, but they come from these counts).
# Dense voids (40-70 %): 1298/2000 = 64.9 % (macro 65) vs 120/320 = 37.5 % (38).
assert (dense_k, dense_n) == (1298, 2000) and half_up(pct(dense_k, dense_n)) == Decimal("64.9")
assert half_up(pct(dense_k, dense_n), "1") == 65
assert (trot_dense_k, trot_dense_n) == (120, 320)
assert half_up(pct(trot_dense_k, trot_dense_n)) == Decimal("37.5")
assert half_up(pct(trot_dense_k, trot_dense_n), "1") == 38
# All voids 0-70 % over 0.05-0.25 m/s: 3261/4000 (macro 82) vs 410/640 (64).
assert half_up(pct(sum(by_void_k), 8 * N_VOID), "1") == 82
assert half_up(pct(sum(TROT_BY_VOID), 8 * TROT_TRIALS), "1") == 64

# The legend values and trial counts must equal the prose macros.
MACROS = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}",
                         CONSTANTS_TEX.read_text()))
for macro, want in {"corlSweepFullSlowest": all_voids_int[0],
                    "corlSweepFullTwentyFive": all_voids_int[4],
                    "corlSweepFullThirty": all_voids_int[5],
                    "corlSweepTrialsPerCell": TRIALS_PER_CELL,
                    "corlSweepTrialsPerVoid": N_VOID,
                    "corlSweepTrialsPerSpeed": N_SPEED,
                    "corlOcsTrialsPerPoint": TROT_TRIALS}.items():
    assert Decimal(MACROS[macro]) == Decimal(want), (macro, MACROS[macro], want)
assert Decimal(MACROS["corlCmdVxMax"]) == Decimal(str(TRAINED_MAX_SPEED))
early_k = {m: sum(ks) for m, ks in EARLY_BY_VOID.items()}
assert early_k == {"GaitNet": 529, "GaitNet-Single": 381, "Per-candidate + IDM": 471,
                   "GaitNet, no IDM": 352, "OCS2's own trot": 410}, early_k
EARLY_LABELS = {m: str(half_up(pct(k, EARLY_N))) for m, k in early_k.items()}
for _m, _macro in {"GaitNet": "corlEarlyFull", "GaitNet-Single": "corlEarlySingle",
                   "Per-candidate + IDM": "corlEarlyMlp", "GaitNet, no IDM": "corlEarlyNoIdm",
                   "OCS2's own trot": "corlEarlyTrot"}.items():
    assert MACROS[_macro] == EARLY_LABELS[_m], (_macro, MACROS[_macro], EARLY_LABELS[_m])
assert int(MACROS["corlEarlyTrials"]) == EARLY_N
assert EARLY_LABELS == {"GaitNet": "82.7", "GaitNet-Single": "59.5",
                        "Per-candidate + IDM": "73.6", "GaitNet, no IDM": "55.0",
                        "OCS2's own trot": "64.1"}, EARLY_LABELS
print("all value checks passed")

# ---------------------------------------------------------------- style
# Same font and rcParams as plot_ocs2_results.py, Fig. 1 and fig_bench.py.
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
INK, MUTED, GRIDLINE = "#1F2937", "#4B5563", "#E5E7EB"
GAITNET, TROT = "#4F46E5", "#4B5563"
TITLE = dict(loc="left", fontsize=7.5, fontweight="semibold", color=INK, pad=4)
VALUE_PT = 6.5                  # value labels of (a)
PAD_PT = 0.8                    # gap between a marker's edge and its value [pt]
SHIFT_PT = 3.0                  # sideways shift of a value off a line [pt]
# Panel (b): one indigo shade per command, light (slow) to dark (fast). The
# six steps are about evenly spaced in OKLab lightness (0.79, 0.68, 0.59,
# 0.46, 0.36, 0.26), so neighbouring commands stay apart in grayscale.
SPEED_COLORS = ["#A5B4FC", "#818CF8", "#6366F1", "#4338CA", "#312E81", "#1E1B4B"]
GAITNET_MS, TROT_MS = 3.6, 4.0  # marker sizes [pt]; the trot's open square
                                # is drawn over and around GaitNet's dot

# ---------------------------------------------------------------- layout [mm]
# Three panels in one row (2026-11-04): (a) and (b) as before but narrower,
# (b)'s legend inside its empty lower left, and (c) on the right, where (b)'s
# legend used to be. Axes are placed in mm.
FIG_W, FIG_H = 152.0, 46.0      # \textwidth of the PMLR class x 46 mm
LEFT, TOP, BOTTOM = 9.6, 4.6, 9.4
W_A, GAP_AB, W_B, GAP_BC, W_C = 55.0, 7.0, 35.5, 17.5, 25.0
MM = 1 / 25.4
PT_PER_MM = 72 / 25.4
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
_h = (FIG_H - TOP - BOTTOM) / FIG_H
_x = [LEFT, LEFT + W_A + GAP_AB, LEFT + W_A + GAP_AB + W_B + GAP_BC]
assert _x[2] + W_C < FIG_W - 1.5, _x
ax_a = fig.add_axes([_x[0] / FIG_W, BOTTOM / FIG_H, W_A / FIG_W, _h])
ax_b = fig.add_axes([_x[1] / FIG_W, BOTTOM / FIG_H, W_B / FIG_W, _h])
ax_c = fig.add_axes([_x[2] / FIG_W, BOTTOM / FIG_H, W_C / FIG_W, _h])

Y_MAX = 114                     # headroom for the labels above 100 % in (a)


def style_axes(ax, xmax):
    ax.set_xlim(-4.5, xmax)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(VOIDS)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Void fraction [%]", labelpad=2)
    ax.grid(axis="y", color=GRIDLINE, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 100)   # the headroom above 100 holds labels


# ---------------------------------------------------------------- (a) by void
# x extends past 70 % so that the values at 70 % fit right of their markers
# (left of them the two steep segments leave no room).
style_axes(ax_a, 78)
ax_a.set_ylabel("Success [%]", labelpad=2)
ax_a.tick_params(labelleft=True)
ax_a.set_title("(a) By void fraction, 0.05–0.25 m/s", **TITLE)
series_a = [("GaitNet", [float(pct(k, N_VOID)) for k in by_void_k], LABELS_A_GAITNET,
             dict(color=GAITNET, lw=1.6, marker="o", ms=GAITNET_MS, mew=0,
                  zorder=4), GAITNET_MS)]
if SHOW_TROT:
    series_a.append(("OCS2's own trot\n(earlier runs,\nother worlds)",
                     [float(pct(k, TROT_TRIALS)) for k in TROT_BY_VOID], LABELS_A_TROT,
                     dict(color=TROT, ls=(0, (3.2, 1.8)), lw=1.2, marker="s",
                          ms=TROT_MS, mfc="none", mec=TROT, mew=0.9, zorder=5),
                     TROT_MS + 0.9))   # drawn extent: side + edge width
lines_a = []
for label, ys, _, style, _ in series_a:
    (ln,) = ax_a.plot(VOIDS, ys, label=label, **style)
    lines_a.append(ln)
# Lower left is empty: both curves start near 100 % and fall to the right.
leg_a = ax_a.legend(loc="lower left", frameon=False, handlelength=2.2,
                    borderaxespad=0.3, labelspacing=0.35, handletextpad=0.6)

# ---------------------------------------------------------------- (b) by speed
style_axes(ax_b, 73)
ax_b.tick_params(labelleft=True)        # sharey hides them by default
ax_b.set_title("(b) GaitNet by speed", **TITLE)
n_in = SPEEDS.index(TRAINED_MAX_SPEED) + 1               # 4 commands in range
handles_b = []
# One marker shape per command, so that lines with similar values (0.10 and
# 0.15 m/s overlap at 40-60 % voids) stay distinguishable without colour.
MARKERS_B = ["o", "s", "^", "D", "v", "X"]
for j, (speed, color, label) in enumerate(zip(SPEEDS, SPEED_COLORS, LEGEND_B)):
    above = j >= n_in
    (ln,) = ax_b.plot(VOIDS, [row[j] for row in GRID], color=color, lw=1.3,
                      ls=(0, (3.2, 1.6)) if above else "-",
                      marker=MARKERS_B[j], ms=2.8, mew=0, zorder=3 + j / 10,
                      label=label)
    handles_b.append(ln)
# One legend in the empty lower left of (b): every line is near 100 % up to
# 30 % voids. Values: each speed's success over all eight void fractions.
# Dashed: above the training range (stated in the caption).
# Compact: speeds only, two columns; each speed's success over all voids
# (LEGEND_B values, = the prose macros) goes in the caption.
LEGEND_B_SHORT = [f"{v:.2f}" for v in SPEEDS]
LEGEND_B_SHORT[0] += " m/s"
leg_b = ax_b.legend(handles=handles_b, labels=LEGEND_B_SHORT, loc="lower left", ncol=2,
                    frameon=False, fontsize=6.3, handlelength=1.7, handletextpad=0.35,
                    labelspacing=0.15, columnspacing=0.7, borderaxespad=0.25, borderpad=0.1)
print("caption, (b) success over all voids per speed:", LEGEND_B)

# ---------------------------------------------------------------- (c) earlier benchmark
C_TEAL, C_LIGHT_INDIGO, C_LIGHT_GRAY = "#0D9488", "#A5B4FC", "#B6BCC6"
C_BARS = [("GaitNet", "GaitNet", dict(color=GAITNET)),
          ("GaitNet-Single", "GaitNet-\nSingle", dict(color=C_LIGHT_INDIGO)),
          ("Per-candidate + IDM", "Per-candidate\n+ IDM", dict(color=C_TEAL)),
          ("GaitNet, no IDM", "GaitNet,\nno IDM", dict(color="#CBD5E1")),   # not white+indigo: that is Fig. 2's rerun
          ("OCS2's own trot", "OCS2's\nown trot", dict(color=TROT))]
for yi, (key, _, kw) in enumerate(C_BARS):
    val = float(pct(early_k[key], EARLY_N))
    ax_c.barh(yi, val, height=0.62, zorder=2, **kw)
    ax_c.annotate(EARLY_LABELS[key], (val, yi), xytext=(1.5, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=VALUE_PT, color=INK, zorder=3)
ax_c.set_yticks(range(len(C_BARS)))
ax_c.set_yticklabels([lab for _, lab, _ in C_BARS], fontsize=6.5, linespacing=0.95)
ax_c.tick_params(axis="y", length=0, pad=2)
ax_c.invert_yaxis()
ax_c.set_xlim(0, 100)
ax_c.set_xticks([0, 50, 100])
ax_c.set_xlabel("Success [%]", labelpad=2)
for g in (25, 50, 75, 100):
    ax_c.axvline(g, color=GRIDLINE, lw=0.5, zorder=0)
ax_c.spines[["top", "right", "left"]].set_visible(False)
ax_c.set_title("(c) Earlier checkpoint", x=-GAP_BC / W_C + 0.02, **TITLE)

# ---------------------------------------------------------------- (a) values
# Each value is placed at the first candidate position around its marker that
# overlaps no line, marker, legend, title or earlier value and stays inside
# the axes. GaitNet prefers above its marker, the trot below (GaitNet's curve
# lies above the trot's).
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
px_per_pt = fig.dpi / 72


def polyline_px(line):
    xy = line.get_xydata()
    return ax_a.transData.transform(xy)


def hits_polyline(box: Bbox, pts) -> bool:
    """True if any segment of the polyline passes through the box."""
    for p, q in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, 400)[:, None]
        s = p + t * (q - p)
        inside = ((s[:, 0] > box.x0) & (s[:, 0] < box.x1)
                  & (s[:, 1] > box.y0) & (s[:, 1] < box.y1))
        if inside.any():
            return True
    return False


def grow(box: Bbox, pt: float) -> Bbox:
    d = pt * px_per_pt
    return Bbox.from_extents(box.x0 - d, box.y0 - d, box.x1 + d, box.y1 + d)


obstacle_boxes = [leg_a.get_window_extent(renderer), ax_a.title.get_window_extent(renderer)]
marker_boxes = []
for (_, ys, _, _, ms), _ in zip(series_a, lines_a):
    for x, y in zip(VOIDS, ys):
        cx, cy = ax_a.transData.transform((x, y))
        r = 0.5 * ms * px_per_pt
        marker_boxes.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
polylines = [polyline_px(ln) for ln in lines_a]
ax_box = ax_a.get_window_extent(renderer)
placed = []


def candidates(r_pt: float, prefer_up: bool):
    """(name, offset [pt], ha, va) around a marker of half-extent r_pt."""
    d = e = r_pt + PAD_PT          # box-to-box clearance along both axes
    f = SHIFT_PT                   # small sideways shift, still under the marker
    up = [("N", (0, d), "center", "bottom"), ("N<", (-f, d), "center", "bottom"),
          ("N>", (f, d), "center", "bottom"), ("NE", (e, e), "left", "bottom"),
          ("NW", (-e, e), "right", "bottom")]
    down = [("S", (0, -d), "center", "top"), ("S<", (-f, -d), "center", "top"),
            ("S>", (f, -d), "center", "top"), ("SW", (-e, -e), "right", "top"),
            ("SE", (e, -e), "left", "top")]
    side = [("E", (d, 0), "left", "center"), ("W", (-d, 0), "right", "center")]
    return (up + side + down) if prefer_up else (down + side + up)


def blocked_by(box: Bbox):
    """None if the label box is free, else what it hits."""
    if not (box.x0 >= ax_box.x0 and box.x1 <= ax_box.x1
            and box.y0 >= ax_box.y0 and box.y1 <= ax_box.y1):
        return "outside axes"
    g = grow(box, 0.4)
    if any(g.overlaps(b) for b in obstacle_boxes):
        return "legend or title"
    if any(g.overlaps(b) for b in marker_boxes):
        return "marker"
    if any(g.overlaps(b) for b in placed):
        return "value label"
    # Lines: 0.4 pt of air plus half the line's width.
    if any(hits_polyline(grow(box, 0.4 + 0.5 * ln.get_linewidth()), pl)
           for ln, pl in zip(lines_a, polylines)):
        return "line"
    return None


choice = {}
R_CLEAR_PT = 0.5 * max(s[4] for s in series_a)
for si, (name, ys, labels, style, ms) in enumerate(series_a):
    for x, y, text in zip(VOIDS, ys, labels):
        tried = []
        # Offsets clear the larger of the two markers: at 0 and 10 % they coincide.
        for cname, off, ha, va in candidates(R_CLEAR_PT, prefer_up=(si == 0)):
            t = ax_a.annotate(text, (x, y), xytext=off, textcoords="offset points",
                              ha=ha, va=va, fontsize=VALUE_PT, color=style["color"],
                              zorder=6)
            box = t.get_window_extent(renderer)
            why = blocked_by(box)
            if why is None:
                placed.append(box)
                choice[(si, x)] = cname
                break
            tried.append(f"{cname}: {why}")
            t.remove()
        else:
            raise AssertionError(f"no free place for {name!r} {text} at {x} %: {tried}")
print("label positions:", choice)

# ---------------------------------------------------------------- checks
# Nothing may leave the figure; (b)'s legend must not touch a line of (b);
# (c)'s category labels must not reach into (b).
fig.canvas.draw()
fig_box = fig.bbox
for artist in [leg_a, leg_b, ax_a.title, ax_b.title, ax_c.title, ax_a.yaxis.label,
               ax_a.xaxis.label, ax_b.xaxis.label, ax_c.xaxis.label] + ax_a.texts + ax_c.texts \
        + ax_c.get_yticklabels():
    b = artist.get_window_extent(renderer)
    assert (b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig_box.x1 and b.y1 <= fig_box.y1), \
        (artist, b, fig_box)
_lb = grow(leg_b.get_window_extent(renderer), 0.6)
for ln in handles_b:
    assert not hits_polyline(_lb, ax_b.transData.transform(ln.get_xydata())), ln.get_label()
_bx1 = ax_b.get_window_extent(renderer).x1
for t in ax_c.get_yticklabels():
    assert t.get_window_extent(renderer).x0 > _bx1 + 2, t.get_text()
assert ax_c.title.get_window_extent(renderer).x0 > ax_b.title.get_window_extent(renderer).x1
assert not ax_c.title.get_window_extent(renderer).overlaps(ax_b.title.get_window_extent(renderer))

fig.savefig(HERE / "fig_sweep.png", dpi=300)
# Matplotlib embeds the OpenType (CFF) Fira Sans in a way some PDF checkers
# flag, so the PDF that goes into the paper has its text converted to
# outlines with Inkscape, like Fig. 1.
raw = HERE / "fig_sweep.raw.pdf"
fig.savefig(raw)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                "-o", str(HERE / "fig_sweep.pdf")], check=True, capture_output=True)
raw.unlink()
print("wrote", HERE / "fig_sweep.pdf")
