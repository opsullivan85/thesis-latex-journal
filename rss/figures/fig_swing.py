"""Swing-duration figure (fig:swing): measured swing durations, CADENCE vs OCS2's own trot.

Plain, comparative counterpart of the swing-duration histogram of the
predecessor thesis (images/data/swing_duration_histogram.png).

DATA CAVEAT (same as fig_gait.py, 2026-11-12). ../timming.md comes from a
SEPARATE, EARLIER evaluation run on the OCS2 benchmark worlds: one CADENCE
checkpoint of one training seed with the OCS2 IFM, and one run of OCS2's own
trot. It is NOT the set of runs behind the current Table 1 and fig_bench.py. Its
successful trials (CADENCE 549, trot 504) are asserted by gait_stats.sanity().
Captions and prose must call this a separate evaluation run.
PROVENANCE FLAG (review, 2026-11-12): 549/600 = 91.5 % and 504/600 = 84.0 %,
and the per-cell counts reproduce every superseded Table 1 entry ("was 91.5",
"was 84.0", 96.3/86.7/86.5, 88.7/79.3/71.5 in constants.tex). So this file is
most likely the superseded single-seed Table 1 run. Table 1 now reports
67.8 % for the same deterministic trot; that gap is unexplained. Do NOT print
the per-arm trial counts (above all the trot's 504, used nowhere in the
paper) in the caption until the author has explained it.

What is plotted (definitions from gait_stats.py; read its header):
  * swing_s of the COUNTED swings only, successful trials only. counted ==
    True exactly when field_entry_s <= touchdown_s <= 20 s, and field_entry_s
    is when the REAR feet reach the void field. So: swings that END after
    the rear feet reach the field, within the 20 s trial (front-foot swings
    that land on the field earlier are not counted).
  * swing_s is a MEASURED swing (foot-height detector on simulator ground
    truth), not the commanded swing time T. The trot measures 0.235 s
    (\\corlGaitTrotSwingMeasured) against its scheduled 0.3 s, so measured
    swings are shorter than commanded ones and the T range marker (commanded
    limits \\corlSwingDurationMin, \\corlSwingDurationMax) is a reference, not
    a bound on the measured values. Only the CADENCE-vs-trot comparison
    carries meaning.
  * Step-style line histograms, each bin in % of ALL of the arm's counted
    swings; swings beyond X_MAX are in the denominator but off the axis
    (printed below for the caption).

BINS (aliasing). Lift-off and touchdown times are samples of a logger with
about 21 ms spacing (gait_stats.py header: median 21 ms, 20-34 ms, varying
within a trial), so measured swings cluster at whole multiples of the sample
spacing (CADENCE: 42, 63, 85, 106, 127, 149 ... ms; trot: 213, 235, 256 ms).
10 ms bins would alias against that grid (alternate bins empty or holding a
whole cluster). Here the spacing DELTA is fitted to the data (the spacing that
minimises the median distance of all counted swings to their nearest
multiple), and the bin edges sit half-way between multiples, (k + 1/2) DELTA,
so each bin holds exactly one cluster: one bin per number of log samples. The
grid is approximate (the spacing varies within a trial), so the script asserts
that only a small share of swings lies within 3 ms of a bin edge.

Independent check: medians and counts are recomputed from a bare
pd.read_csv of the data file (no gait_stats code) and asserted equal.

Design rules (user, 2026-11-03): no error bars, whiskers, bands or heatmaps;
plain lines with the values printed; at most about four short labels.
Style (rcParams, colours, line styles, layout check) copied from fig_gait.py.

Size: 74 mm x 42 mm, half of the PMLR \\textwidth (152 mm).

usage: python fig_swing.py   (writes fig_swing.pdf and fig_swing.png here)
"""
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import gait_stats as gs

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data
df = gs.load()
chk = gs.sanity(df)                      # also asserts 549 / 504 successful trials
_tex = (HERE.parent / "constants.tex").read_text()
_macros = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}", _tex))
T_MIN = float(_macros["corlSwingDurationMin"])          # commanded limits [s]
T_MAX = float(_macros["corlSwingDurationMax"])
assert (T_MIN, T_MAX) == (0.1, 0.3), (T_MIN, T_MAX)
assert chk["trials"] == {"c0": int(_macros["corlGaitTrialsGaitNet"]),
                         "trot": int(_macros["corlGaitTrialsTrot"])}, chk["trials"]

SWING = {a: df[(df.arm == a) & df.counted].swing_s.to_numpy() for a in gs.ARMS}
MEDIAN = {a: float(np.median(SWING[a])) for a in gs.ARMS}
COUNT = {a: len(SWING[a]) for a in gs.ARMS}

# Independent computation: bare pandas on the raw file, counted parsed from text.
_raw = pd.read_csv(gs.DATA, dtype={"counted": str})
assert set(_raw.counted.unique()) <= {"True", "False"}, _raw.counted.unique()
_ind = _raw[_raw.counted == "True"].groupby("arm").swing_s.agg(["median", "size"])
for a in gs.ARMS:
    assert abs(_ind.loc[a, "median"] - MEDIAN[a]) < 1e-12, (a, _ind.loc[a], MEDIAN[a])
    assert int(_ind.loc[a, "size"]) == COUNT[a] == chk["field_swings"][a], (a, _ind.loc[a], COUNT[a])
# The trot's printed median must be the paper's macro.
assert f"{MEDIAN['trot']:.3f}" == _macros["corlGaitTrotSwingMeasured"], MEDIAN["trot"]

# ---------------------------------------------------------------- bins on the log's sample grid
ALL_MS = np.concatenate([SWING[a] for a in gs.ARMS]) * 1e3


def grid_residual_ms(delta_ms: float) -> float:
    """Median distance [ms] of the swings to the nearest multiple of delta_ms."""
    r = ALL_MS / delta_ms
    return float(np.median(np.abs(r - np.round(r))) * delta_ms)


_cands = np.round(np.arange(19.5, 23.0, 0.01), 2)
DELTA_FIT = min(_cands, key=lambda d: (grid_residual_ms(d), d))
DELTA_MS = round(DELTA_FIT, 1)          # one decimal is enough: 0.05 ms x 25 samples < 1.3 ms
assert 21.0 <= DELTA_MS <= 21.6, DELTA_FIT
X_MAX = 0.5                              # [s] axis end; right of T_MAX holds the labels
EDGES_MS = (np.arange(0, int(X_MAX * 1e3 / DELTA_MS) + 2) + 0.5) * DELTA_MS
assert ALL_MS.min() >= EDGES_MS[0], ALL_MS.min()          # no swing below the first edge
_near = np.min(np.abs(ALL_MS[:, None] - EDGES_MS[None, :]), axis=1) < 3.0
NEAR_EDGE = float(_near.mean())
assert NEAR_EDGE < 0.03, NEAR_EDGE      # 2026-11-12: 1.6 % of swings within 3 ms of an edge

HIST, OFF_AXIS = {}, {}
for a in gs.ARMS:
    ms = SWING[a] * 1e3
    counts, _ = np.histogram(ms, EDGES_MS)
    HIST[a] = 100 * counts / COUNT[a]                   # % of ALL counted swings
    OFF_AXIS[a] = int((ms >= X_MAX * 1e3).sum())
    # Every swing is either in a drawn bin or off the axis (beyond X_MAX).
    assert counts.sum() + int((ms >= EDGES_MS[-1]).sum()) == COUNT[a]

# ---------------------------------------------------------------- style (fig_gait.py)
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-Italic.otf"):
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
    # Only addition to fig_gait.py's rcParams: an italic Fira T for the
    # paper's symbol T (swing time) in the "T range" label.
    "mathtext.fontset": "custom", "mathtext.it": "Fira Sans:italic",
    "mathtext.rm": "Fira Sans", "mathtext.cal": "Fira Sans",
})
INK, GRID = "#1F2937", "#E5E7EB"
INDIGO, GRAY = "#4F46E5", "#4B5563"
MARK, MARK_TEXT = "#B6BCC6", "#6B7280"   # light T-range marker and its label
VALUE_PT = 7.0
HALO = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
ARM_COLOR = {"c0": INDIGO, "trot": GRAY}
# Lines as in fig_gait.py (b) and fig_sweep.py: CADENCE solid, the trot dashed and thinner.
LINE_STYLE = {"c0": dict(color=INDIGO, lw=1.4),
              "trot": dict(color=GRAY, lw=1.2, ls=(0, (3.2, 1.8)))}
XTICK_PAD = 3.5                          # pt, as in fig_gait.py

# ---------------------------------------------------------------- layout [mm]
MM = 1 / 25.4
FIG_W, FIG_H = 74.0, 42.0                # half \textwidth
TOP, BOTTOM, LEFT, RIGHT = 5.0, 9.0, 10.5, 3.0
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax = fig.add_axes([LEFT / FIG_W, BOTTOM / FIG_H,
                   (FIG_W - LEFT - RIGHT) / FIG_W, (FIG_H - TOP - BOTTOM) / FIG_H])

# ---------------------------------------------------------------- histograms
# The trot's peak bin (71 %) needs the y ticks and spine to run to 80 (with
# fig_gait.py (b)'s 0-60 it read as clipped); the T-range dimension arrow sits
# above the 80 grid line.
Y_TOP_TICK = 80
Y_MAX = 86.0
for a in ("trot", "c0"):                 # CADENCE on top
    share = HIST[a]
    nz = np.flatnonzero(share)
    i0, i1 = nz[0], nz[-1] + 1           # draw only the bins that hold swings
    ax.stairs(share[i0:i1], EDGES_MS[i0:i1 + 1] / 1e3, baseline=0, fill=False,
              zorder=3 if a == "c0" else 2, **LINE_STYLE[a])

# Commanded swing-time range [T_MIN, T_MAX]: two light verticals and a
# double-headed dimension arrow between them above the data and the 80 grid
# line (it must clear the trot's 71 % bin). A plain top bar instead of the
# arrow read as a box.
Y_BRACKET = 84.0
assert max(HIST["trot"].max(), HIST["c0"].max()) < Y_TOP_TICK - 5, "peak too close to the arrow"
for x in (T_MIN, T_MAX):
    ax.plot([x, x], [0, Y_BRACKET + 1.5], color=MARK, lw=0.7, zorder=1, solid_capstyle="butt")
ax.annotate("", xy=(T_MIN, Y_BRACKET), xytext=(T_MAX, Y_BRACKET), annotation_clip=False,
            arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.13", color=MARK,
                            lw=0.7, shrinkA=0, shrinkB=0))
ax.annotate("$T$ range", ((T_MIN + T_MAX) / 2, Y_BRACKET), xytext=(0, 1.5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=VALUE_PT, color=MARK_TEXT, annotation_clip=False)

ax.set_xlim(0, X_MAX)
ax.set_xticks(np.arange(0, X_MAX + 1e-9, 0.1))
ax.set_ylim(0, Y_MAX)
ax.set_yticks(range(0, Y_TOP_TICK + 1, 20))
ax.spines["left"].set_bounds(0, Y_TOP_TICK)
ax.spines[["top", "right"]].set_visible(False)
for g in range(20, Y_TOP_TICK + 1, 20):
    ax.axhline(g, color=GRID, lw=0.5, zorder=0)
ax.set_xlabel("Measured swing [s]", labelpad=2)
ax.set_ylabel("Share of swings [%]", labelpad=2)
ax.tick_params(axis="x", pad=XTICK_PAD)

# One label per arm, right of the T_MAX line (x [s], top [%]).
LABEL = {"trot": (0.318, 77.0), "c0": (0.318, 41.0)}   # blocks ~26 % tall; ~10 % gap
for a, name in (("trot", "OCS2's own trot"), ("c0", "CADENCE")):
    ax.text(*LABEL[a], f"{name}\nmedian {MEDIAN[a]:.3f} s\n{COUNT[a]:,} swings",
            ha="left", va="top", fontsize=VALUE_PT, color=ARM_COLOR[a],
            linespacing=1.15, zorder=4, bbox=HALO)


# ---------------------------------------------------------------- layout checks
def layout_problems(fig, ax):
    """Text outside the canvas or too small, text on text, labels on drawn lines.

    As fig_gait.py, but labels are tested against the drawn paths (sampled
    densely), not against the bounding boxes of the step lines, which span
    the whole axis."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]

    def extent(t):
        b, patch = t.get_window_extent(r), t.get_bbox_patch()
        return Bbox.union([b, patch.get_window_extent(r)]) if patch else b

    boxes = [(t, extent(t)) for t in texts]
    out, canvas = [], fig.bbox
    for t, b in boxes:
        if not (b.x0 >= canvas.x0 and b.y0 >= canvas.y0 and b.x1 <= canvas.x1 and b.y1 <= canvas.y1):
            out.append(f"outside canvas: {t.get_text()!r}")
        if t.get_fontsize() < 7:
            out.append(f"font below 7 pt: {t.get_text()!r}")
    for i, (t, b) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if b.overlaps(b2):
                out.append(f"text overlap: {t.get_text()!r} / {t2.get_text()!r}")
    # Drawn paths in display coordinates: step lines, T-range lines, visible spines.
    paths = []
    for art in list(ax.patches) + [l for l in ax.lines if l.get_color() != GRID]:
        v = art.get_transform().transform(art.get_path().vertices)
        paths.append(v)
    for s in ax.spines.values():
        if s.get_visible():
            paths.append(s.get_transform().transform(s.get_path().vertices))
    clip = ax.get_window_extent(r)
    for t in (t for t in ax.texts if t.get_text().strip()):
        b = extent(t)
        for v in paths:
            for p, q in zip(v[:-1], v[1:]):
                pts = p + np.linspace(0, 1, 200)[:, None] * (q - p)
                inside_ax = ((pts[:, 0] >= clip.x0 - 1) & (pts[:, 0] <= clip.x1 + 1)
                             & (pts[:, 1] >= clip.y0 - 1) & (pts[:, 1] <= clip.y1 + 1))
                hit = ((pts[:, 0] >= b.x0) & (pts[:, 0] <= b.x1)
                       & (pts[:, 1] >= b.y0) & (pts[:, 1] <= b.y1) & inside_ax)
                if hit.any():
                    out.append(f"label on data: {t.get_text()!r}")
                    break
    return sorted(set(out))


problems = layout_problems(fig, ax)
for p in problems:
    print("LAYOUT:", p)
assert not problems, "layout problems, see above"

fig.savefig(HERE / "fig_swing.png", dpi=300)
raw = HERE / "fig_swing.raw.pdf"
fig.savefig(raw)
try:
    subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                    "-o", str(HERE / "fig_swing.pdf")], check=True, capture_output=True)
    raw.unlink()
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    raw.rename(HERE / "fig_swing.pdf")
    print("inkscape failed, kept matplotlib PDF:", e)
print("wrote", HERE / "fig_swing.pdf")

# ---------------------------------------------------------------- numbers for the caption
print(f"log sample spacing fitted {DELTA_FIT:.2f} ms -> bins {DELTA_MS} ms, edges at "
      f"(k + 1/2) x {DELTA_MS} ms; {100 * NEAR_EDGE:.2f} % of swings within 3 ms of an edge")
print(f"trials (successful): {chk['trials']}")
for a in gs.ARMS:
    s = SWING[a]
    print(f"{gs.ARM_NAME[a]}: {COUNT[a]} counted swings, median {MEDIAN[a]:.4f} s, "
          f"IQR {np.quantile(s, .25):.3f}-{np.quantile(s, .75):.3f} s, "
          f"< {T_MIN} s {100 * (s < T_MIN).mean():.1f} %, > {T_MAX} s {int((s > T_MAX).sum())}, "
          f"off axis (>= {X_MAX} s) {OFF_AXIS[a]}, max {s.max():.3f} s")
    nz = np.flatnonzero(HIST[a])
    print("    bins [%] (centre ms: share):",
          " ".join(f"{(k + 1) * DELTA_MS:.0f}:{HIST[a][k]:.1f}" for k in range(nz[0], nz[-1] + 1)))

# Proposed constants.tex lines for the caption's new numbers (the author adds
# them; this script does not edit .tex files).
print("% ---- fig:swing (figures/fig_swing.py, from ../timming.md, counted swings)")
print(f"\\newcommand{{\\corlGaitSwingMedian}}{{{MEDIAN['c0']:.3f}}}  "
      f"% CADENCE measured swing, median [s] ({MEDIAN['c0']:.4f})")
print(f"\\newcommand{{\\corlGaitSwingsGaitNet}}{{{COUNT['c0']:,}}}".replace(",", "{,}")
      + "  % counted swings, CADENCE")
print(f"\\newcommand{{\\corlGaitSwingsTrot}}{{{COUNT['trot']:,}}}".replace(",", "{,}")
      + "  % counted swings, trot")
print(f"\\newcommand{{\\corlGaitSwingBinMs}}{{{DELTA_MS}}}  "
      f"% bin width = fitted log sample spacing ({DELTA_FIT:.2f} ms)")
print(f"\\newcommand{{\\corlGaitSwingOffAxis}}{{{OFF_AXIS['c0']}}}  "
      f"% CADENCE swings >= {X_MAX} s, off the axis (trot {OFF_AXIS['trot']})")
