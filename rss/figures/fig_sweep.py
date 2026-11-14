"""CORRECTION 2026-11-12 (author): the checkpoint is iteration 300; every '350'
below was a mistake.
Plot the OCS2 sweep (Fig. fig:sweep, Sec. 5 of the L4DC paper).

PROVENANCE NOTE (2026-11-05): the author states that every learned value
plotted here is the mean over five training seeds at iteration 350, each seed
run on all trials; the per-seed logs are not in this repository. Comments below
that describe a value as a single run or another iteration predate this
statement and are kept as written.

Source of every number: the author's OCS2 sweep data of 2026-10-14 (the figure
of the 2026-10-28 draft, git 486a92a:figures/plot_ocs2_results.py), confirmed
on 2026-11-04 as the paper's sweep: GaitNet is the benchmark checkpoint
(iteration 350); every method ran on the same worlds, 12.5 cm tiles, 80 trials
per point; success = reach the end of a 3 m track within 1.5x the nominal time.
A 500-trial-per-point GaitNet sweep (100 per cell, 0.05-0.30 m/s) used in drafts
of 2026-11-03/04 had an evaluation mistake (author, 2026-11-04) and was removed
from the paper; it is kept only in git history.

Design rule (user, 2026-11-03): no error bars, whiskers, confidence bands or
heatmaps. Lines with markers, the values a reader needs printed next to the
marks. Here only GaitNet's values are printed (five lines in (a), three in
(b) leave no room for more); App. D lists every count. Author, 2026-11-12: no
shaded bands either, so the speeds above training in (b) are marked by a thin
dashed vertical divider between 0.20 and 0.25 m/s with the label "beyond
training" on its right (it replaced a grey axvspan).
Author, 2026-11-13 ("add legend all the figure"): the legend follows the
paper's common legend style (Fira Sans 7 pt, ink text, no frame, short
handles) and now also keys the divider ("Beyond training →", drawn as a
short vertical dashed handle); the floating label "beyond training" in (b) is
removed, so that no mark is explained twice. No data, axis or value label moved.
The entry names the side of the line, not the line: the divider is drawn at
0.225 m/s, halfway between the last trained speed (0.20 m/s, \\corlCmdVxMax)
and 0.25 m/s, so calling it the training speed limit would put the limit at the
wrong speed (review of 2026-11-13).

Panels
- (a) Success vs void fraction (0-70 %), 80 trials per point spread over
  0.05-0.25 m/s: GaitNet, OCS2's own trot, GaitNet-Single (trained with one
  leg in swing at a time, same training seed), GaitNet with the IFM switched
  off (same weights), and Per-candidate + IFM (one training seed, trained
  without the IFM in the loop, run with the OCS2 IFM).
- (b) Success vs commanded speed (0.05-0.25 m/s), 80 trials per point, 10 at
  each of the eight void fractions: GaitNet, GaitNet-Single, OCS2's own trot.
  0.25 m/s is above GaitNet's 0.2 m/s training range (\\corlCmdVxMax).

Data conventions: integer success counts; every printed or asserted value is
computed with fractions.Fraction and rounded half up (decimal.ROUND_HALF_UP).
The counts were recovered from the 2026-10-28 figure's integer percentages
(each maps to exactly one count of 80 under round-half-even), and their totals
equal that draft's macros.

usage: python fig_sweep.py   (writes fig_sweep.pdf and fig_sweep.png here)
"""
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cadence_names  # noqa: F401  (paper name: CADENCE)
import numpy as np
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"

# ---------------------------------------------------------------- data
VOIDS = [0, 10, 20, 30, 40, 50, 60, 70]                 # void fraction [%]
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25]                 # commanded speed [m/s]
TRAINED_MAX_SPEED = 0.2                                 # \corlCmdVxMax
# Divider in (b) halfway between the last trained speed and the first one above
# it: a line at exactly 0.20 m/s would cross three markers and a value.
X_DIVIDER = 0.5 * (TRAINED_MAX_SPEED + min(s for s in SPEEDS if s > TRAINED_MAX_SPEED))
TRIALS = 80                                             # trials per point and training seed
# Author, 2026-11-05: every value is the success rate [%] averaged over the five
# training seeds, rounded to a whole percent. The script therefore works in whole
# percentages (N = 100 "units"), prints them as given, and pools by plain means.
# The sweep's no-IFM run is not used (author: the benchmark's no-IFM result is the
# correct one).
N = 100
BY_VOID = {          # success [%] per void fraction (trials spread over 0.05-0.25 m/s)
    "GaitNet":             [100, 100, 90, 90, 91, 80, 60, 50],
    "OCS2's own trot":     [100, 100, 88, 75, 70, 38, 30, 12],
    "GaitNet-Single":      [86, 74, 82, 66, 68, 42, 30, 28],
    "Per-candidate + IFM": [98, 92, 94, 70, 66, 70, 52, 46],
}
BY_SPEED = {         # success [%] per speed (10 trials at each void fraction)
    "GaitNet":         [84, 89, 86, 81, 66],
    "GaitNet-Single":  [74, 68, 63, 54, 40],
    "OCS2's own trot": [79, 79, 74, 64, 51],
}
def half_up(x: Fraction, places: str = "0.1") -> Decimal:
    """Round an exact fraction half up to the given decimal places."""
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
        Decimal(places), ROUND_HALF_UP)


def pct(k: int, n: int) -> Fraction:
    """Exact success rate [%] of k successes out of n trials."""
    return Fraction(100 * k, n)


def value_label(k: int, n: int = N) -> str:
    """Whole percent, as the source gives it."""
    return str(k)


# ---------------------------------------------------------------- asserts
for _m, _ks in BY_VOID.items():
    assert len(_ks) == len(VOIDS) and all(0 <= k <= N for k in _ks), _m
for _m, _ks in BY_SPEED.items():
    assert len(_ks) == len(SPEEDS) and all(0 <= k <= N for k in _ks), _m
TOTAL = {m: sum(ks) for m, ks in BY_VOID.items()}
N_ALL = N * len(VOIDS)                                  # 8 points -> mean of percentages
POOLED = {m: str(half_up(pct(k, N_ALL))) for m, k in TOTAL.items()}
assert POOLED == {"GaitNet": "82.6", "OCS2's own trot": "64.1", "GaitNet-Single": "59.5",
                  "Per-candidate + IFM": "73.5"}, POOLED
# Prose claims that come from these values.
dense = slice(4, 8)                                     # 40-70 % voids
g_dense, t_dense = sum(BY_VOID["GaitNet"][dense]), sum(BY_VOID["OCS2's own trot"][dense])
assert (g_dense, t_dense) == (281, 150)                 # means 70.25 (70) vs 37.5 (38)
G, T, S = BY_SPEED["GaitNet"], BY_SPEED["OCS2's own trot"], BY_SPEED["GaitNet-Single"]
assert all(g > t for g, t in zip(G, T))                 # beat the trot at every speed
assert all(g > s for g, s in zip(G, S))                 # Single lower at every speed
assert all(g > s for g, s in zip(BY_VOID["GaitNet"], BY_VOID["GaitNet-Single"]))
_gt = [g - t for g, t in zip(BY_VOID["GaitNet"], BY_VOID["OCS2's own trot"])]
assert min(_gt[3:]) == 15 and max(_gt[3:]) == 42, _gt   # "from 30 % on, by 15 to 42 points"
assert G[1] - S[1] == 21                                # "21 points at 0.10 m/s"
assert all(g > 80 for g in G[:4]) and G[4] == 66        # "above 80 % up to 0.20, 66 % at 0.25"
# The printed pooled values must equal the prose macros.
MACROS = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}",
                         CONSTANTS_TEX.read_text()))
for _m, _macro in {"GaitNet": "corlSweepFull", "GaitNet-Single": "corlSweepSingle",
                   "Per-candidate + IFM": "corlSweepMlp", "OCS2's own trot": "corlSweepTrot"}.items():
    assert MACROS[_macro] == POOLED[_m], (_macro, MACROS[_macro], POOLED[_m])
assert int(MACROS["corlOcsTrialsPerPoint"]) == TRIALS
assert MACROS["corlOcsDenseFull"] == str(half_up(pct(g_dense, 4 * N), "1"))
assert MACROS["corlOcsDenseTrot"] == str(half_up(pct(t_dense, 4 * N), "1"))
assert Decimal(MACROS["corlOcsSpeedMax"]) == Decimal(str(SPEEDS[-1]))
assert Decimal(MACROS["corlCmdVxMax"]) == Decimal(str(TRAINED_MAX_SPEED))
# Exactly the speeds right of the divider are above training (here only 0.25).
assert [s for s in SPEEDS if s > X_DIVIDER] == [s for s in SPEEDS if s > TRAINED_MAX_SPEED] == [0.25]
# The divider is NOT at the training limit, so its legend entry (DIVIDER_NAME,
# below) must not call it one.
assert X_DIVIDER > TRAINED_MAX_SPEED
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
DIVIDER_LW = 0.7                # dashed divider in (b) [pt]
DIVIDER_DASH = (3.0, 2.0)       # its dash pattern: on, off [line widths]
# Its legend entry. It names the side right of the line (the author's earlier
# label "beyond training"), not the line: the line is at X_DIVIDER = 0.225 m/s,
# the training limit at 0.20 m/s (\corlCmdVxMax), so "limit" would be wrong.
DIVIDER_NAME = "Beyond training →"
# The paper's common legend style, shared by every figure script.
LEGEND_STYLE = dict(frameon=False, fontsize=7, labelcolor=INK, handlelength=1.6,
                    handletextpad=0.5, columnspacing=1.2, labelspacing=0.3)
PAD_PT = 0.8                    # gap between a marker's edge and its value [pt]
SHIFT_PT = 3.0                  # sideways shift of a value off a line [pt]
GAITNET_MS, TROT_MS, VARIANT_MS = 3.6, 4.0, 3.0      # marker sizes [pt]
STYLE = {   # method -> plot style; teal = Per-candidate + IFM as in Fig. 2a
    "GaitNet": dict(color=GAITNET, lw=1.6, marker="o", ms=GAITNET_MS, mew=0, zorder=5),
    "OCS2's own trot": dict(color=TROT, ls=(0, (3.2, 1.8)), lw=1.2, marker="s", ms=TROT_MS,
                            mfc="none", mec=TROT, mew=0.9, zorder=4),
    "GaitNet-Single": dict(color="#D97706", lw=1.0, marker="^", ms=VARIANT_MS, mew=0, zorder=3),
    "GaitNet, no IFM": dict(color="#DB2777", lw=1.0, marker="v", ms=VARIANT_MS, mew=0, zorder=3),
    "Per-candidate + IFM": dict(color="#0D9488", lw=1.0, marker="D", ms=VARIANT_MS, mew=0,
                                zorder=3),
}
EXTENT = {m: s["ms"] + (s.get("mew", 0) if s.get("mfc") == "none" else 0)
          for m, s in STYLE.items()}                    # drawn marker extent [pt]

# ---------------------------------------------------------------- layout [mm]
FIG_W, FIG_H = 152.0, 42.0      # \textwidth of the PMLR class x 42 mm
LEFT, TOP, BOTTOM = 9.6, 4.6, 9.4
W_A, GAP_AB, W_B = 66.0, 37.0, 37.5   # GAP_AB holds the legend column and (b)'s y labels
LEG_X = 1.0                           # legend: mm right of (a)
MM = 1 / 25.4
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
_h = (FIG_H - TOP - BOTTOM) / FIG_H
_x = [LEFT, LEFT + W_A + GAP_AB]
assert _x[1] + W_B < FIG_W - 1.5, _x
ax_a = fig.add_axes([_x[0] / FIG_W, BOTTOM / FIG_H, W_A / FIG_W, _h])
ax_b = fig.add_axes([_x[1] / FIG_W, BOTTOM / FIG_H, W_B / FIG_W, _h])
Y_MAX = 118                     # headroom for the labels above 100 % (42 mm tall)


def style_axes(ax, xs, pad, xlabel):
    ax.set_xlim(xs[0] - pad, xs[-1] + 1.8 * pad)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks(xs)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Success [%]", labelpad=2)
    ax.grid(axis="y", color=GRIDLINE, lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 100)   # the headroom above 100 holds labels
    ax.spines["bottom"].set_bounds(xs[0], xs[-1])


# (a) by void fraction
style_axes(ax_a, VOIDS, 4.5, "Nominal void fraction [%]")
ax_a.set_title("(a) By void fraction, 0.05–0.25 m/s", **TITLE)
lines = {"a": {}, "b": {}}
for m in ("Per-candidate + IFM", "GaitNet-Single", "OCS2's own trot", "GaitNet"):
    (lines["a"][m],) = ax_a.plot(VOIDS, [float(pct(k, N)) for k in BY_VOID[m]], label=m, **STYLE[m])

# (b) by speed; 0.25 m/s is above the training range: a thin dashed divider
# (no shaded band), from 0 to 100 % like the left spine. It is named in the
# legend (2026-11-13), not by a label inside the panel.
style_axes(ax_b, SPEEDS, 0.0225, "Commanded speed [m/s]")
ax_b.set_xticklabels([f"{s:.2f}" for s in SPEEDS])
ax_b.set_ylabel("")              # same axis as (a); its label is there
divider = ax_b.axvline(X_DIVIDER, ymin=0, ymax=100 / Y_MAX, color=MUTED, lw=DIVIDER_LW,
                       ls=(0, DIVIDER_DASH), zorder=1)
ax_b.set_title("(b) By speed, all void fractions", **TITLE)
for m in ("GaitNet-Single", "OCS2's own trot", "GaitNet"):
    (lines["b"][m],) = ax_b.plot(SPEEDS, [float(pct(k, N)) for k in BY_SPEED[m]], label=m,
                                 **STYLE[m])


class VerticalDividerHandle(HandlerBase):
    """Legend handle of (b)'s divider: a short VERTICAL dashed line.

    The divider and the trot's line are both dashed and #4B5563, so the default
    horizontal handle would read as a second trot entry. The handle is centred in
    the handle box and is three dashes and two gaps long (9.1 pt), so it starts
    and ends on a full dash.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                       fontsize, trans):
        on, off = (d * orig_handle.get_linewidth() for d in DIVIDER_DASH)
        half = 0.5 * (3 * on + 2 * off)
        x, y = 0.5 * width - xdescent, 0.5 * height - ydescent
        handle = Line2D([x, x], [y - half, y + half])
        self.update_prop(handle, orig_handle, legend)   # colour, width, dashes
        handle.set_transform(trans)
        return [handle]


# One legend for both panels, a column between them: every line of (a) and (b)
# (Per-candidate + IFM is drawn in (a) only) and (b)'s divider.
order = ["GaitNet", "OCS2's own trot", "GaitNet-Single", "Per-candidate + IFM"]
assert set(lines["a"]) == set(order) and set(lines["b"]) <= set(order)
# The baseline's display name (cadence_names._BASELINE) is long for the legend
# column, so it is wrapped onto two lines here (2026-11-13).
BASELINE_LABEL = "CADENCE without\nmixer or history"
assert BASELINE_LABEL.replace("\n", " ") == cadence_names._BASELINE
leg_labels = [BASELINE_LABEL if m == "Per-candidate + IFM" else m for m in order]
leg = fig.legend(handles=[lines["a"][m] for m in order] + [divider],
                 labels=leg_labels + [DIVIDER_NAME], loc="center left", ncol=1,
                 handler_map={divider: VerticalDividerHandle()},
                 bbox_to_anchor=((LEFT + W_A + LEG_X) / FIG_W,
                                 (BOTTOM + 0.5 * (FIG_H - TOP - BOTTOM)) / FIG_H),
                 borderaxespad=0.0, borderpad=0.0, **LEGEND_STYLE)

# ---------------------------------------------------------------- GaitNet's values
# Each value goes to the first candidate position around its marker that
# overlaps no line, marker, legend, title or earlier value and stays inside
# the axes; preferred above the marker (GaitNet's line is mostly the top one).
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
px_per_pt = fig.dpi / 72


def grow(box: Bbox, pt: float) -> Bbox:
    d = pt * px_per_pt
    return Bbox.from_extents(box.x0 - d, box.y0 - d, box.x1 + d, box.y1 + d)


def hits_polyline(box: Bbox, pts) -> bool:
    """True if any segment of the polyline passes through the box."""
    for p, q in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, 400)[:, None]
        s = p + t * (q - p)
        if ((s[:, 0] > box.x0) & (s[:, 0] < box.x1)
                & (s[:, 1] > box.y0) & (s[:, 1] < box.y1)).any():
            return True
    return False


def candidates(r_pt: float):
    d = r_pt + PAD_PT
    f = SHIFT_PT
    return [("N", (0, d), "center", "bottom"), ("N<", (-f, d), "center", "bottom"),
            ("N>", (f, d), "center", "bottom"), ("NE", (d, d), "left", "bottom"),
            ("NW", (-d, d), "right", "bottom"), ("E", (d, 0), "left", "center"),
            ("W", (-d, 0), "right", "center"), ("S", (0, -d), "center", "top"),
            ("SE", (d, -d), "left", "top"), ("SW", (-d, -d), "right", "top")]


def marker_boxes(ax, panel_lines):
    """Pixel boxes of every drawn marker of a panel."""
    boxes = []
    for m, ln in panel_lines.items():
        r = 0.5 * EXTENT[m] * px_per_pt
        for cx, cy in ax.transData.transform(ln.get_xydata()):
            boxes.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))
    return boxes


def polylines(ax, panel_lines):
    """(line, pixel vertices) of every data line of a panel."""
    return [(ln, ax.transData.transform(ln.get_xydata())) for ln in panel_lines.values()]


def divider_box() -> Bbox:
    """Pixel box of (b)'s dashed divider: x from data, y from 0 to 100 % (axvline is
    drawn in a blended data/axes transform, so it is not a data polyline)."""
    x = ax_b.transData.transform((X_DIVIDER, 0))[0]
    y0, y1 = ax_b.transData.transform([(X_DIVIDER, 0), (X_DIVIDER, 100)])[:, 1]
    d = 0.5 * DIVIDER_LW * px_per_pt
    return Bbox.from_extents(x - d, y0, x + d, y1)


def place_values(ax, xs, ks, panel_lines, extra_obstacles=()):
    ax_box = ax.get_window_extent(renderer)
    markers = marker_boxes(ax, panel_lines)
    polys = polylines(ax, panel_lines)
    obstacles = [leg.get_window_extent(renderer), ax.title.get_window_extent(renderer),
                 *extra_obstacles]
    placed, chosen = [], []
    for x, k in zip(xs, ks):
        y, text = float(pct(k, N)), value_label(k)
        tried = []
        for name, off, ha, va in candidates(0.5 * max(EXTENT.values())):
            t = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                            fontsize=VALUE_PT, color=GAITNET, zorder=6)
            b = t.get_window_extent(renderer)
            g = grow(b, 0.4)
            if not (b.x0 >= ax_box.x0 and b.x1 <= ax_box.x1 and b.y0 >= ax_box.y0
                    and b.y1 <= ax_box.y1):
                why = "outside axes"
            elif any(g.overlaps(o) for o in obstacles):
                why = "legend or title"
            elif any(g.overlaps(o) for o in markers):
                why = "marker"
            elif any(g.overlaps(o) for o in placed):
                why = "value label"
            elif any(hits_polyline(grow(b, 0.4 + 0.5 * ln.get_linewidth()), pl)
                     for ln, pl in polys):
                why = "line"
            else:
                placed.append(b)
                chosen.append(name)
                break
            tried.append(f"{name}: {why}")
            t.remove()
        else:
            raise AssertionError(f"no free place for GaitNet {text} at {x}: {tried}")
    return chosen


print("label positions (a):", place_values(ax_a, VOIDS, BY_VOID["GaitNet"], lines["a"]))
print("label positions (b):", place_values(
    ax_b, SPEEDS, BY_SPEED["GaitNet"], lines["b"], extra_obstacles=[divider_box()]))

# ---------------------------------------------------------------- checks
# Nothing may leave the figure; the legend must not touch either panel or a
# tick label of (b).
fig.canvas.draw()
fig_box = fig.bbox
for artist in [leg, ax_a.title, ax_b.title, ax_a.xaxis.label, ax_b.xaxis.label,
               ax_a.yaxis.label, ax_b.yaxis.label] + ax_a.texts + ax_b.texts:
    b = artist.get_window_extent(renderer)
    assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig_box.x1 and b.y1 <= fig_box.y1, artist
_lb = grow(leg.get_window_extent(renderer), 0.6)
assert _lb.x0 > ax_a.get_window_extent(renderer).x1, "legend reaches into (a)"
_bleft = min([t.get_window_extent(renderer).x0 for t in ax_b.get_yticklabels() if t.get_text()]
             + [ax_b.yaxis.label.get_window_extent(renderer).x0])
assert _lb.x1 < _bleft - 1, "legend reaches (b)'s y labels"
assert ax_b.title.get_window_extent(renderer).x0 > ax_a.title.get_window_extent(renderer).x1
# The divider is named by the legend only: (b) holds GaitNet's five values and no
# other text, and the divider crosses none of them.
_div = divider_box()
assert [t.get_text() for t in ax_b.texts] == [value_label(k) for k in BY_SPEED["GaitNet"]]
assert not any(_div.overlaps(t.get_window_extent(renderer)) for t in ax_b.texts), \
    "divider through a value"
# Legend: one entry per drawn line and one for the divider, in the common style
# (compared on the data keys; cadence_names shows "CADENCE" for "GaitNet").
assert [t.get_text() for t in leg.get_texts()] == [
    m.replace("GaitNet", "CADENCE") for m in leg_labels] + [DIVIDER_NAME], "legend entries"   # display names: cadence_names.py
assert all(t.get_fontsize() >= 7 for t in leg.get_texts()), "legend text below 7 pt"
assert "limit" not in DIVIDER_NAME.lower(), "the divider is at 0.225 m/s, not at the limit"
# The divider's vertical handle (drawn past its handle box) stays clear of the
# legend entry above it and inside the figure.
_handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
_vh = _handles[-1].get_window_extent(renderer)
_above = grow(_handles[-2].get_window_extent(renderer), 0.5 * VARIANT_MS)
assert _vh.height > _vh.width and not _vh.overlaps(_above) and _vh.y0 > 0, _vh
assert abs(_vh.height / px_per_pt - (3 * DIVIDER_DASH[0] + 2 * DIVIDER_DASH[1]) * DIVIDER_LW) < 0.1
print("divider at %.3f m/s; legend %.1f x %.1f mm" % (
    X_DIVIDER, *(np.array([_lb.width, _lb.height]) / fig.dpi * 25.4)))

fig.savefig(HERE / "fig_sweep.png", dpi=300)
# Text to outlines with Inkscape, like Figs. 1 and 2.
raw = HERE / "fig_sweep.raw.pdf"
fig.savefig(raw)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                "-o", str(HERE / "fig_sweep.pdf")], check=True, capture_output=True)
raw.unlink()
print("wrote", HERE / "fig_sweep.pdf")
