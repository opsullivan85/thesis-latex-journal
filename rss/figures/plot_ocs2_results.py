"""Plot the OCS2 sweep for Sec. 4 (Fig. 2 of the L4DC paper).

Data (author, 2026-11-03): GaitNet is the benchmark checkpoint c0
(iteration 350, OCS2 IDM, depth-camera terrain), 100 trials per cell over
8 void fractions x 6 commands (0.05-0.30 m/s). OCS2's own trot is the
earlier benchmark's data (author: "the trot sweep is the existing data"),
80 trials per point; its void sweep spreads them over 0.05-0.25 m/s. The two
methods ran on different worlds.
Left: success by void fraction, GaitNet pooled over 0.05-0.25 m/s to match.
Right: success by commanded speed, GaitNet pooled over all eight voids.
The arms of the earlier ~600-iteration checkpoint (GaitNet-Single, no IDM,
per-candidate) are no longer plotted; their data are in git history.

usage: python3 plot_ocs2_results.py   (writes ocs2_results.pdf next to this file)
"""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data
VOIDS = [0, 10, 20, 30, 40, 50, 60, 70]                 # void fraction [%]
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]           # commanded speed [m/s]
# c0 successes out of 100 per cell; rows = VOIDS, columns = SPEEDS.
C0_GRID = np.array([
    [100, 100, 100, 100, 99, 99],
    [100, 100, 99, 99, 100, 98],
    [98, 100, 97, 99, 96, 93],
    [98, 96, 96, 97, 89, 79],
    [89, 97, 99, 93, 87, 63],
    [89, 87, 87, 76, 60, 38],
    [79, 71, 68, 50, 34, 21],
    [50, 34, 23, 17, 8, 2],
])
C0_TRIALS_PER_CELL = 100
TROT_TRIALS = 80                                        # \corlOcsTrialsPerPoint
# Per series and panel: (x values, successes, trials per point).
BY_VOID = {
    "GaitNet": (VOIDS, C0_GRID[:, :5].sum(axis=1), 5 * C0_TRIALS_PER_CELL),
    "OCS2's own trot": (VOIDS, np.array([80, 80, 70, 60, 56, 30, 24, 10]), TROT_TRIALS),
}
BY_SPEED = {
    "GaitNet": (SPEEDS, C0_GRID.sum(axis=0), len(VOIDS) * C0_TRIALS_PER_CELL),
    "OCS2's own trot": (SPEEDS[:5], np.array([63, 63, 59, 51, 41]), TROT_TRIALS),
}
TRAINED_MAX_SPEED = 0.2                                 # \corlCmdVxMax

# ---------------------------------------------------------------- style
# Same palette and font as Fig. 1 (fig_overview.svg).
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
STYLE = {
    "GaitNet": dict(color="#4F46E5", lw=1.8, marker="o", ms=3.6, zorder=5),
    "OCS2's own trot": dict(color="#4B5563", lw=1.4, marker="s", ms=3.2, zorder=4),
}


def wilson(k, n, z=1.96):
    """95 % Wilson interval [%] for k successes out of n trials."""
    p, d = np.asarray(k) / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def draw(ax, series):
    for name, (xs, k, n) in series.items():
        ax.plot(xs, 100 * np.asarray(k) / n, label=name, **STYLE[name])
        lo, hi = wilson(k, n)
        ax.fill_between(xs, lo, hi, color=STYLE[name]["color"], alpha=0.15, lw=0, zorder=1)


MM = 1 / 25.4
# Width of the PMLR text block (6 in) so the fonts stay at their set size.
fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(152 * MM, 46 * MM),
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))

draw(ax_v, BY_VOID)
# No "above training range" shading on the void axis: the sweep's tiles
# (24 cm, to confirm) differ from the training cells (12.5 cm), so the void
# fractions are not directly comparable.
ax_v.set_xlim(-2.5, 72.5)
ax_v.set_xlabel("Void fraction [%]")
ax_v.set_ylabel("Success [%]")
ax_v.set_xticks(VOIDS)
ax_v.set_title("(a) By void fraction, 0.05–0.25 m/s", loc="left", fontsize=7.5,
               fontweight="semibold", color="#1F2937", pad=4)

draw(ax_s, BY_SPEED)
ax_s.axvspan(TRAINED_MAX_SPEED + 0.0125, 0.32, color="#E5E7EB", zorder=0, lw=0)
ax_s.text(0.2625, 101, "above training\nrange", ha="center", va="top",
          fontsize=6.5, color="#4B5563", linespacing=1.0)
ax_s.set_xlabel("Commanded speed [m/s]")
ax_s.set_xticks(SPEEDS)
ax_s.set_xlim(0.035, 0.315)
# GaitNet: all eight void fractions. The trot's speed sweep: void fractions
# still to be confirmed.
ax_s.set_title("(b) By commanded speed", loc="left",
               fontsize=7.5, fontweight="semibold", color="#1F2937", pad=4)

for ax in (ax_v, ax_s):
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#E5E7EB", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

# Legend inside panel (a), whose lower left is empty (both curves start
# near 100 % and fall to the right).
ax_v.legend(loc="lower left", frameon=False, handlelength=2.6)
fig.subplots_adjust(left=0.065, right=0.995, top=0.89, bottom=0.24, wspace=0.14)
fig.savefig(HERE / "ocs2_results.png", dpi=300)
# Matplotlib embeds the OpenType (CFF) Fira Sans in a way some PDF checkers
# flag, so the PDF that goes into the paper has its text converted to
# outlines with Inkscape, like Fig. 1.
raw = HERE / "ocs2_results.raw.pdf"
fig.savefig(raw)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                "-o", str(HERE / "ocs2_results.pdf")], check=True, capture_output=True)
raw.unlink()
print("wrote", HERE / "ocs2_results.pdf")
