"""Gait figure (fig:gait): CADENCE keeps the trot's diagonal pairing, not its timing.

DATA CAVEAT (2026-11-12). ../timming.md comes from a SEPARATE, EARLIER
evaluation run on the OCS2 benchmark worlds: one CADENCE checkpoint of one
training seed with the OCS2 IFM, and one run of OCS2's own trot. It is NOT the
set of runs behind Table 1 and fig_bench.py (five training seeds at iteration
300, entered 2026-11-12). Its successful trials per cell (CADENCE 100 93 96 |
96 87 77 = 549, trot 97 89 80 | 94 81 63 = 504) are not Table 1's (trot there:
91 72 46 | 88 70 40). gait_stats.SUCCESS and gait_stats.sanity() still assert
those earlier counts: they describe this data file, not the paper's
benchmark. Captions and prose must call this a separate evaluation run.

Source: ../timming.md (CSV of measured swings, one row per swing, added by the
author on 2026-11-04). All definitions come from gait_stats.py; read its
header for the column meanings and caveats:
  * ONLY SUCCESSFUL TRIALS are in the file (549 CADENCE, 504 trot), so every
    statistic is conditioned on success.
  * Swing times are MEASURED (foot-height detector, presumably 1.5 cm above
    the rolling minimum), not the commanded swing time. The trot's measured
    swings are about 0.235 s against its scheduled 0.3 s.
  * Time resolution is about 21 ms (the logger), so lags of one sample are at
    the resolution limit.
  * Statistics use the void field only: counted swings, i.e. swings that land
    between field entry (rear feet on the void field) and 20 s.

Panels (152 mm x 45 mm, full \\textwidth since 2026-11-12; it was a 120 mm
single panel before):
  (a) Swing (footfall) diagram, UNCHANGED from the single-panel version. One
      world seed, the same for both arms, in the hardest cell (49 % voids,
      0.20 m/s), the first WINDOW_S seconds after field entry. One row per
      leg, rows ordered FL, RR, FR, RL so that diagonal partners sit next to
      each other. A bar is a measured swing (foot up); white space is stance.
      Filled bars are the FL+RR pair and open bars the FR+RL pair (no light
      shade: light indigo means CADENCE-Diagonal elsewhere in the paper).
      The seed is not hand-picked: among the seeds that succeeded in that
      cell for BOTH arms, it is the one whose CADENCE feet-up shares
      (0 / 1 / 2 feet up) are closest (L1) to the cell's pooled shares.
  (b) Time between consecutive lift-offs of the same leg on the void field,
      pooled over all six cells and all four legs: gait_stats.summarize()'s
      'stride' (counted swings only; NOT the report's "same-leg liftoff
      interval" line, which uses another filter and gives median 0.559 s).
      Step-style line histograms, 0.1 s bins on multiples of 0.1 s, each bin
      in % of ALL of the arm's intervals; intervals above X_MAX_B are in the
      denominator but off the axis (printed below for the caption). The
      trot's two-bin block is its 0.6 s period straddling a bin edge. The
      panel carries exactly two labels, "CADENCE, CV 0.61" and "OCS2's own
      trot, CV 0.02" (coefficient of variation = sample std / mean); the
      medians were dropped from the panel on 2026-11-12 (CADENCE's has no
      macro) and are printed below only. The CVs and the trot's median must
      reproduce \\corlGaitStrideCv, \\corlGaitStrideCvTrot and
      \\corlGaitTrotPeriodMeasured (asserted).
  (c) CADENCE's overlapping swing pairs (two legs in measured swing at once,
      gait_stats.py's 'overlap pair') by leg-pair type, as shares of all its
      pairs. Must reproduce \\corlGaitOverlapPairs (24,793) and
      \\corlGaitDiagPairPct (98) (asserted). The trot's pairs (all diagonal)
      are printed below for the caption, not drawn.

Removed panels (author, 2026-11-04): a step-rate panel and a feet-in-swing
(0 / 1 / 2 feet up) panel. Their numbers are still in gait_stats.py's report
(steps_per_s, feet_up) and the \\corlGaitSteps* / \\corlGaitAllDown* macros.
The diagonal share of the two-feet-up time (CADENCE 99.1 %, trot 100 %,
\\corlGaitDiagTimePct) is printed below.

Design rules (user, 2026-11-03): no error bars, whiskers, bands or heatmaps;
plain bars and lines with the values printed on them; at most about four
short labels per panel.

usage: python fig_gait.py   (writes fig_gait.pdf and fig_gait.png here)
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
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import gait_stats as gs

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data
df = gs.load()
gs.sanity(df)
trials = gs.all_trials(df)
POOLED = {a: gs.summarize(trials, df, gs.cell_sel(a)) for a in gs.ARMS}

EX_SPEED, EX_VOID = 0.2, 49
WINDOW_S = 3.0
EDGE_S = 0.05                        # [s], see the footfall loop
by_key = {t.key: t for t in trials}
cell = gs.summarize(trials, df, gs.cell_sel("c0", EX_SPEED, EX_VOID))
target = np.array(cell["feet_up"][:3])
both = sorted(s for (a, v, p, s) in by_key if a == "c0" and v == EX_SPEED and p == EX_VOID
              and ("trot", v, p, s) in by_key)
dist = {s: np.abs(by_key[("c0", EX_SPEED, EX_VOID, s)].feet_up_field[:3]
                  / by_key[("c0", EX_SPEED, EX_VOID, s)].field_time - target).sum() for s in both}
EX_SEED = min(dist, key=lambda s: (dist[s], s))
print(f"(a) world seed {EX_SEED}: closest of {len(both)} shared successful seeds "
      f"(L1 distance {dist[EX_SEED]:.3f})")

# Diagonal share of the two-feet-up time, printed for the caption.
DIAG_TIME = {a: 100 * POOLED[a]["pair_time_share_of_two"]["diagonal"] for a in gs.ARMS}

_tex = (HERE.parent / "constants.tex").read_text()
_macros = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}", _tex))
assert str(EX_SEED) == _macros["corlGaitExampleSeed"], (EX_SEED, _macros["corlGaitExampleSeed"])


def half_up(x: Fraction, places: str) -> Decimal:
    """Round an exact fraction half up, e.g. places='0.1' or '1'."""
    return (Decimal(x.numerator) / Decimal(x.denominator)).quantize(
        Decimal(places), rounding=ROUND_HALF_UP)


# (b) Same-leg lift-off intervals, exactly gait_stats.summarize()'s 'stride':
# counted (void-field) swings, sorted by trial, leg and lift-off; the
# difference of consecutive lift-offs of one leg within one trial.
def lift_off_intervals(arm: str) -> np.ndarray:
    f = df[(df.arm == arm) & df.counted].sort_values(gs.TRIAL + ["leg", "liftoff_s"])
    return f.groupby(gs.TRIAL + ["leg"]).liftoff_s.diff().dropna().to_numpy()


BIN_MS = 100                          # (b) bin width [ms]
X_MAX_B = 2.0                         # (b) x-axis end [s]
INTERVALS, HIST, TIMING = {}, {}, {}
for a in gs.ARMS:
    iv = lift_off_intervals(a)
    med, cv = float(np.median(iv)), float(np.std(iv, ddof=1) / np.mean(iv))
    assert abs(med - POOLED[a]["stride_median"]) < 1e-12, (a, med)
    assert abs(cv - POOLED[a]["stride_cv"]) < 1e-12, (a, cv)
    # Lift-off times have 1 ms resolution: bin on integer ms so that an
    # interval of exactly 0.600 s always falls in [600, 700) ms.
    ms = np.round(iv * 1e3).astype(int)
    edges_ms = np.arange(0, int(X_MAX_B * 1e3) + BIN_MS, BIN_MS)
    counts = np.array([((ms >= lo) & (ms < hi)) for lo, hi in zip(edges_ms[:-1], edges_ms[1:])]).sum(1)
    INTERVALS[a] = iv
    HIST[a] = (edges_ms / 1e3, 100 * counts / len(iv))      # % of ALL intervals
    TIMING[a] = (med, cv, 100 * float((ms >= edges_ms[-1]).mean()), len(iv))
assert f"{TIMING['c0'][1]:.2f}" == _macros["corlGaitStrideCv"], TIMING["c0"]
assert f"{TIMING['trot'][1]:.2f}" == _macros["corlGaitStrideCvTrot"], TIMING["trot"]
assert f"{TIMING['trot'][0]:.3f}" == _macros["corlGaitTrotPeriodMeasured"], TIMING["trot"]
assert TIMING["trot"][2] == 0.0       # every trot interval is on the axis

# (c) CADENCE's overlapping swing pairs by leg-pair type.
PAIR_ORDER = ("diagonal", "rear", "side", "front")
PAIRS = {p: POOLED["c0"]["pair_counts"][p] for p in PAIR_ORDER}
N_PAIRS = sum(PAIRS.values())
# Counts in the comment of \corlGaitDiagPairPct (constants.tex, 2026-11-04).
assert PAIRS == dict(diagonal=24370, rear=227, side=158, front=38), PAIRS
_n = re.search(r"\\newcommand\{\\corlGaitOverlapPairs\}\{(.*?)\}\s*%", _tex).group(1)
assert int(_n.replace("{,}", "")) == N_PAIRS, (_n, N_PAIRS)
PAIR_PCT = {p: half_up(Fraction(100 * k, N_PAIRS), "0.1") for p, k in PAIRS.items()}
assert str(half_up(Fraction(100 * PAIRS["diagonal"], N_PAIRS), "1")) == _macros["corlGaitDiagPairPct"]
TROT_PAIRS = POOLED["trot"]["pair_counts"]
assert all(TROT_PAIRS[p] == 0 for p in PAIR_ORDER if p != "diagonal"), TROT_PAIRS

# ---------------------------------------------------------------- style
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
INDIGO, LIGHT_INDIGO = "#4F46E5", "#A5B4FC"
GRAY, LIGHT_GRAY = "#4B5563", "#B6BCC6"
VALUE_PT = 7.0
PAD_PT = 1.5
HALO = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
ARM_COLOR = {"c0": (INDIGO, LIGHT_INDIGO), "trot": (GRAY, LIGHT_GRAY)}   # (FL+RR, FR+RL)
# (b) lines as in fig_sweep.py: CADENCE solid, the trot dashed and thinner.
LINE_STYLE = {"c0": dict(color=INDIGO, lw=1.4),
              "trot": dict(color=GRAY, lw=1.2, ls=(0, (3.2, 1.8)))}

# ---------------------------------------------------------------- layout [mm]
MM = 1 / 25.4
FIG_W, FIG_H = 152.0, 45.0              # PMLR \textwidth x 45 mm
TOP, BOTTOM = 5.0, 9.0
LEFT = 17.0                             # (a): arm names and leg labels
# Axes [x0, width] in mm. Gaps hold (b)'s y labels and (c)'s pair-type labels.
X_A, W_A = LEFT, 55.0
X_B, W_B = 83.0, 32.0
X_C, W_C = 129.5, 20.5
assert X_C + W_C <= FIG_W - 1.5
XTICK_PAD = 3.5                         # pt, as in fig_bench.py
_h = (FIG_H - TOP - BOTTOM) / FIG_H

fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
ax_a = fig.add_axes([X_A / FIG_W, BOTTOM / FIG_H, W_A / FIG_W, _h])
ax_b = fig.add_axes([X_B / FIG_W, BOTTOM / FIG_H, W_B / FIG_W, _h])
ax_c = fig.add_axes([X_C / FIG_W, BOTTOM / FIG_H, W_C / FIG_W, _h])


def title(ax, text, x=0.0):
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="semibold", color=INK, pad=3, x=x)


# ---------------------------------------------------------------- (a) swing (footfall) diagram
ROW_H = 0.72
GROUP_GAP = 0.5                       # extra space between the two arms [rows]
y_ticks, y_labels, arm_mid = [], [], {}
y = 0.0
for a in ("trot", "c0"):              # bottom to top: trot, then CADENCE
    tr = df[(df.arm == a) & (df.speed_mps == EX_SPEED) & (df.void_pct == EX_VOID)
            & (df.world_seed == EX_SEED)]
    fe = tr.field_entry_s.iloc[0]
    rows = list(reversed(gs.DIAG_ORDER))   # top row FL, then RR, FR, RL
    ys = []
    for leg in rows:
        # Swings with less than EDGE_S inside the window are not drawn: clipped to a
        # few ms, they would print as slivers that look like ticks.
        sw = tr[(tr.leg == leg) & (tr.touchdown_s > fe + EDGE_S)
                & (tr.liftoff_s < fe + WINDOW_S - EDGE_S)]
        col = ARM_COLOR[a][0]
        filled = gs.DIAG_PAIR[leg] == "A"
        segs = [(max(lo - fe, 0.0), min(td - fe, WINDOW_S) - max(lo - fe, 0.0))
                for lo, td in zip(sw.liftoff_s, sw.touchdown_s)]
        # Filled = FL+RR, open = FR+RL (see the header for why not a light shade).
        ax_a.broken_barh(segs, (y - ROW_H / 2, ROW_H), facecolors=col if filled else "white",
                         edgecolors=col, linewidth=0 if filled else 0.8, zorder=2)
        y_ticks.append(y)
        y_labels.append(leg)
        ys.append(y)
        y += 1
    arm_mid[a] = np.mean(ys)
    y += GROUP_GAP - 1 + 1
ax_a.set_yticks(y_ticks)
ax_a.set_yticklabels(y_labels, fontsize=7)
ax_a.tick_params(axis="y", length=0, pad=2)
ax_a.set_ylim(-0.5, y_ticks[-1] + 0.5)
ax_a.set_xlim(0, WINDOW_S)
ax_a.set_xticks(np.arange(0, WINDOW_S + 1e-9, 0.5))
ax_a.set_xlabel("Time [s]", labelpad=2)
ax_a.tick_params(axis="x", pad=XTICK_PAD)
for x in np.arange(0.5, WINDOW_S, 0.5):
    ax_a.axvline(x, color=GRID, lw=0.5, zorder=0)
ax_a.spines[["top", "right", "left"]].set_visible(False)
for a, name in (("c0", "CADENCE"), ("trot", "OCS2's\nown trot")):
    ax_a.annotate(name, xy=(0, arm_mid[a]), xycoords=("axes fraction", "data"),
                  xytext=(-17, 0), textcoords="offset points", ha="right", va="center",
                  fontsize=7.5, fontweight="semibold", color=ARM_COLOR[a][0], linespacing=1.0)
# Starts over the arm labels.
title(ax_a, f"(a) Swings, {EX_VOID} % voids, {EX_SPEED:.2f} m/s", x=-(LEFT - 1.0) / W_A)

# ---------------------------------------------------------------- (b) lift-off intervals
Y_MAX_B = 70.0
for a in ("trot", "c0"):              # CADENCE on top
    edges, share = HIST[a]
    nz = np.flatnonzero(share)
    i0, i1 = nz[0], nz[-1] + 1        # draw only the bins that hold intervals
    ax_b.stairs(share[i0:i1], edges[i0:i1 + 1], baseline=0, fill=False,
                zorder=3 if a == "c0" else 2, **LINE_STYLE[a])
ax_b.set_xlim(0, X_MAX_B)
ax_b.set_xticks(np.arange(0, X_MAX_B + 1e-9, 0.5))
ax_b.set_ylim(0, Y_MAX_B)
ax_b.set_yticks(range(0, 61, 20))
ax_b.spines["left"].set_bounds(0, 60)
ax_b.spines[["top", "right"]].set_visible(False)
for g in (20, 40, 60):
    ax_b.axhline(g, color=GRID, lw=0.5, zorder=0)
ax_b.set_xlabel("Time between lift-offs [s]", labelpad=2)
ax_b.set_ylabel("Intervals per 0.1 s [%]", labelpad=2)
ax_b.tick_params(axis="x", pad=XTICK_PAD)
# One single-line label per arm, name and CV only (2026-11-12): the medians
# were dropped from the panel since CADENCE's has no macro; both medians are
# still asserted/printed below. The trot's label (26.8 mm) is too wide to
# start right of its block, so it sits above the block, under the axes top;
# CADENCE's sits right of the trot's block and above CADENCE's line.
LABEL_B = {"trot": (0.06, 69.0), "c0": (0.80, 41.0)}       # (x [s], top [%])
for a, name in (("trot", "OCS2's own trot"), ("c0", "CADENCE")):
    cv = TIMING[a][1]
    ax_b.text(*LABEL_B[a], f"{name}, CV {cv:.2f}", ha="left", va="top",
              fontsize=VALUE_PT, color=ARM_COLOR[a][0], zorder=4, bbox=HALO)
assert len(ax_b.texts) == 2, [t.get_text() for t in ax_b.texts]   # at most two labels in (b)
title(ax_b, "(b) Timing of each leg")

# ---------------------------------------------------------------- (c) overlapping pairs
X_MAX_C = 140.0                        # room for the value right of the 98 % bar
ys_c = np.arange(len(PAIR_ORDER))[::-1]    # diagonal on top
ax_c.barh(ys_c, [float(PAIR_PCT[p]) for p in PAIR_ORDER], height=0.62,
          color=INDIGO, linewidth=0, zorder=2)
for yy, p in zip(ys_c, PAIR_ORDER):
    ax_c.annotate(f"{PAIR_PCT[p]}", (float(PAIR_PCT[p]), yy), xytext=(PAD_PT, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=VALUE_PT, color=INK, zorder=3)
ax_c.set_yticks(ys_c)
ax_c.set_yticklabels([p.capitalize() for p in PAIR_ORDER], fontsize=7)
ax_c.tick_params(axis="y", length=0, pad=2)
ax_c.set_ylim(-0.6, len(PAIR_ORDER) - 0.4)
ax_c.set_xlim(0, X_MAX_C)
ax_c.set_xticks([0, 50, 100])
ax_c.spines["bottom"].set_bounds(0, 100)
ax_c.spines[["top", "right", "left"]].set_visible(False)
ax_c.tick_params(axis="x", pad=XTICK_PAD)
ax_c.set_xlabel(f"Share of {N_PAIRS:,} pairs [%]", labelpad=2)
# Centred under the 0-100 axis; set_x keeps matplotlib's automatic y (below
# the tick labels), unlike set_label_coords.
ax_c.xaxis.label.set_x(50 / X_MAX_C)
title(ax_c, "(c) CADENCE's swing pairs", x=-(X_C - X_B - W_B - 2.0) / W_C)


# ---------------------------------------------------------------- layout checks
def layout_problems(fig):
    """Text outside the canvas or too small, text on text, values on data."""
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
    # Labels inside (b) and (c) must clear the drawn data (bars, step lines)
    # and visible spines; light grid lines may pass under their white halo.
    for ax in (ax_b, ax_c):
        data = [p.get_window_extent(r) for p in ax.patches]
        data += [s.get_window_extent(r) for s in ax.spines.values() if s.get_visible()]
        for t in ax.texts:
            out += [f"label on data: {t.get_text()!r}" for d in data if extent(t).overlaps(d)]
    return sorted(set(out))


problems = layout_problems(fig)
for p in problems:
    print("LAYOUT:", p)
assert not problems, "layout problems, see above"

fig.savefig(HERE / "fig_gait.png", dpi=300)
raw = HERE / "fig_gait.raw.pdf"
fig.savefig(raw)
try:
    subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                    "-o", str(HERE / "fig_gait.pdf")], check=True, capture_output=True)
    raw.unlink()
except (FileNotFoundError, subprocess.CalledProcessError) as e:
    raw.rename(HERE / "fig_gait.pdf")
    print("inkscape failed, kept matplotlib PDF:", e)
print("wrote", HERE / "fig_gait.pdf")

# ---------------------------------------------------------------- numbers for the caption
print("diagonal share of two-up time", {a: round(DIAG_TIME[a], 2) for a in gs.ARMS})
for a in gs.ARMS:
    med, cv, off, n = TIMING[a]
    print(f"(b) {gs.ARM_NAME[a]}: {n} intervals, median {med:.3f} s, CV {cv:.3f}, "
          f"{off:.2f} % above {X_MAX_B} s (off the axis, max {INTERVALS[a].max():.3f} s)")
    print("    bins [%]:", " ".join(f"{x:.1f}" for x in HIST[a][1]))
print(f"(c) CADENCE pairs {N_PAIRS}: " + ", ".join(f"{p} {PAIRS[p]} ({PAIR_PCT[p]} %)" for p in PAIR_ORDER))
print(f"    trot pairs {sum(TROT_PAIRS.values())}: {TROT_PAIRS}")
