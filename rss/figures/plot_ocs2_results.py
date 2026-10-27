"""Plot the OCS2 benchmark results for Sec. IV (Fig. 2 of the RSS paper).

Data: the author's OCS2 benchmark of 2026-10-14 (OCS2-fitted IDM,
depth-camera terrain). Every point is 80 trials, and the void sweep and the
speed sweep are separate sets of trials (author, 2026-10-27).
Left: success by void fraction.
Right: success by commanded speed.

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
SUCCESS_BY_VOID = {                                     # success [%], 80 trials each
    "GaitNet": [100, 100, 90, 90, 91, 80, 60, 50],
    # Corrected by the author on 2026-10-14: the first table repeated the
    # 20 % entry at 30 % and shifted the rest one column to the right.
    "OCS2's own trot": [100, 100, 88, 75, 70, 38, 30, 12],
    "GaitNet-Single": [86, 74, 82, 66, 68, 42, 30, 28],
    "GaitNet, no IDM": [96, 80, 64, 56, 44, 38, 36, 26],
    "Per-candidate + IDM": [98, 92, 94, 70, 66, 70, 52, 46],
}
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25]                 # commanded speed [m/s]
SUCCESS_BY_SPEED = {                                    # success [%], 80 trials each
    "GaitNet": [84, 89, 86, 81, 66],
    "OCS2's own trot": [79, 79, 74, 64, 51],
    "GaitNet-Single": [74, 68, 62, 54, 40],
}
TRIALS_PER_POINT = 80                                   # \corlOcsTrialsPerPoint
TRAINED_MAX_SPEED = 0.2                                 # \corlCmdVxMax
TRAINED_MAX_VOID = 40                                  # \corlMaxVoidPct

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
    "GaitNet-Single": dict(color="#8B87F0", lw=1.1, marker="^", ms=3.2, ls=(0, (4, 2))),
    "GaitNet, no IDM": dict(color="#0F9F6E", lw=1.1, marker="v", ms=3.2, ls=(0, (4, 2))),
    "Per-candidate + IDM": dict(color="#D97706", lw=1.1, marker="D", ms=2.8, ls=(0, (4, 2))),
}

def wilson(rate_pct, n=TRIALS_PER_POINT, z=1.96):
    """95 % Wilson interval [%] for a rate given in %, from the count k = round(rate * n)."""
    k = np.round(np.asarray(rate_pct) / 100 * n)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# Wilson bands only for GaitNet and OCS2's trot, the main comparison; the
# ablations would make the plot unreadable and are compared in the text.
BANDED = ("GaitNet", "OCS2's own trot")

MM = 1 / 25.4
# Width of the PMLR text block (6 in) so the fonts stay at their set size.
fig, (ax_v, ax_s) = plt.subplots(1, 2, figsize=(152 * MM, 56 * MM),
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))

for name, ys in SUCCESS_BY_VOID.items():
    ax_v.plot(VOIDS, ys, label=name, **STYLE[name])
    if name in BANDED:
        lo, hi = wilson(ys)
        ax_v.fill_between(VOIDS, lo, hi, color=STYLE[name]["color"], alpha=0.15, lw=0, zorder=1)
# No "above training range" shading on the void axis: the training cells
# (12.5 cm) and the Gazebo tiles (24 cm) make the void fractions not
# directly comparable, and the trained void range of this checkpoint is
# still to be confirmed (review of 2026-10-27).
ax_v.set_xlim(-2.5, 72.5)
ax_v.set_xlabel("Void fraction [%]")
ax_v.set_ylabel("Success [%]")
ax_v.set_xticks(VOIDS)
ax_v.set_title("(a) By void fraction, all speeds", loc="left", fontsize=7.5,
               fontweight="semibold", color="#1F2937", pad=4)

for name, ys in SUCCESS_BY_SPEED.items():
    ax_s.plot(SPEEDS, ys, label=name, **STYLE[name])
    if name in BANDED:
        lo, hi = wilson(ys)
        ax_s.fill_between(SPEEDS, lo, hi, color=STYLE[name]["color"], alpha=0.15, lw=0, zorder=1)
ax_s.axvspan(TRAINED_MAX_SPEED + 0.0125, 0.27, color="#E5E7EB", zorder=0, lw=0)
ax_s.text(0.2375, 101, "above\ntraining\nrange", ha="center", va="top",
          fontsize=6.5, color="#4B5563", linespacing=1.0)
ax_s.set_xlabel("Commanded speed [m/s]")
ax_s.set_xticks(SPEEDS)
ax_s.set_xlim(0.035, 0.265)
# Void fractions of the speed sweep are still to be confirmed.
ax_s.set_title("(b) By commanded speed", loc="left",
               fontsize=7.5, fontweight="semibold", color="#1F2937", pad=4)

for ax in (ax_v, ax_s):
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#E5E7EB", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = ax_v.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.01), handlelength=2.6, columnspacing=1.2)
fig.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.27, wspace=0.14)
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
