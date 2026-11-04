"""Gait timing figure: GaitNet's footfalls against OCS2's own trot on OCS2.

Source: ../timming.md (CSV of measured swings, one row per swing, added by
the author on 2026-11-04). All statistics come from gait_stats.py; read its
header for the column meanings, the definitions, and the caveats:
  * ONLY SUCCESSFUL TRIALS are in the file (549 GaitNet, 504 trot); trials
    per cell equal the benchmark's success counts.
  * Swing times are MEASURED (foot-height detector, presumably 1.5 cm above
    the rolling minimum), not the commanded swing time. The trot's measured
    swings are about 0.235 s against its scheduled 0.3 s.
  * Statistics are taken on the void field, from field entry to 20 s.

Panels ((b) and (c) of the benchmark figure since 2026-11-04; its (a), the
pooled success bars, is fig_bench.pdf):
  (b) Footfall diagram. One world seed, the same for both arms, in the
      hardest cell (49 % voids, 0.20 m/s), the first WINDOW_S seconds after
      field entry. One row per leg, rows ordered FL, RR, FR, RL so that
      diagonal partners sit next to each other. A bar is a measured swing
      (foot up); white space is stance. Filled bars are the FL+RR pair and
      open bars the FR+RL pair (no light shade here: in (c) the light indigo
      means 0.10 m/s).
      The seed is not hand-picked: among the seeds that succeeded in that
      cell for BOTH arms, it is the one whose GaitNet feet-up shares
      (0 / 1 / 2 feet up) are closest (L1) to the cell's pooled shares.
  (c) Steps per second against void fraction, GaitNet at each speed and the
      trot (pooled over both speeds, since it does not change with either:
      6.66-6.67 steps/s in every cell). Lines are labelled at their right end.

The diagonal share of the two-feet-up time (GaitNet 99.1 %, trot 100 %) is
reported in App. E (\corlGaitDiagTimePct). A feet-in-swing panel (share of time with 0 / 1 / 2 feet up)
was removed at the author's request on 2026-11-04; its numbers are still in
gait_stats.py's report (feet_up, by speed).

Design rules (user, 2026-11-03): no error bars, whiskers, bands or heatmaps;
plain bars and lines with the values printed on them.

usage: python fig_gait.py   (writes fig_gait.pdf and fig_gait.png here)
"""
import re
import subprocess
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
STEPS = {(a, v, p): gs.summarize(trials, df, gs.cell_sel(a, v, p))["steps_per_s"]
         for a in gs.ARMS for v in gs.SPEEDS for p in gs.VOIDS}
TROT_STEPS = {p: gs.summarize(trials, df, lambda k, p=p: k[0] == "trot" and k[2] == p)["steps_per_s"]
              for p in gs.VOIDS}

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
print(f"(b) world seed {EX_SEED}: closest of {len(both)} shared successful seeds "
      f"(L1 distance {dist[EX_SEED]:.3f})")

# Diagonal share of the two-feet-up time, printed for the caption.
DIAG_TIME = {a: 100 * POOLED[a]["pair_time_share_of_two"]["diagonal"] for a in gs.ARMS}

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
# Both GaitNet speeds in indigo (0.10 m/s dotted): in (a), light indigo means
# GaitNet-Diagonal, whose gait was not measured.
SPEED_STYLE = {0.1: dict(color=INDIGO, marker="o", ls=(0, (1, 1.6))),
               0.2: dict(color=INDIGO, marker="o")}

# ---------------------------------------------------------------- layout [mm]
MM = 1 / 25.4
# Since 2026-11-04 this is the right part, (b) and (c), of the benchmark figure
# (fig:bench); its (a) is fig_bench.pdf (52 mm wide), placed to the left in
# LaTeX with a 2 mm gap. Both are scaled by \textwidth / 152 mm, so the heights
# and the TOP and BOTTOM margins must match fig_bench.py's (checked below).
FIG_W, FIG_H = 98.0, 40.0
LEFT, RIGHT, TOP, BOTTOM = 17.0, 3.0, 5.0, 9.0
RATIOS = (2.2, 1.0)
WSPACE = 0.38
XTICK_PAD = 3.5                         # pt; must equal fig_bench.py's
_bench = (HERE / "fig_bench.py").read_text()
_fw, _fh = map(float, re.search(r"^FIG_W, FIG_H = ([\d.]+), ([\d.]+)", _bench, re.M).groups())
_bl, _br, _bt, _bb = map(float, re.search(
    r"^LEFT, RIGHT, TOP, BOTTOM = ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)", _bench, re.M).groups())
assert (_fh, _bt, _bb) == (FIG_H, TOP, BOTTOM), "fig_bench.py and fig_gait.py must share height and margins"
_bpad = float(re.search(r"^XTICK_PAD = ([\d.]+)", _bench, re.M).group(1))
assert _bpad == XTICK_PAD, "x tick pads differ: the x labels of (a) and (b, c) would not line up"
# The LaTeX widths must be FIG_W / 152 of \textwidth for both parts.
_res = (HERE.parent / "results.tex").read_text()
for _name, _w in (("bench", _fw), ("gait", FIG_W)):
    _m = re.search(r"\\includegraphics\[width=([\d.]+)\\textwidth\]\{figures/fig_" + _name + r"\}", _res)
    assert _m and abs(float(_m.group(1)) * 152 - _w) < 0.05, ("LaTeX width of fig_" + _name, _m and _m.group(1))
# The example seed must equal the caption's macro.
_macros = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}", (HERE.parent / "constants.tex").read_text()))
assert str(EX_SEED) == _macros["corlGaitExampleSeed"], (EX_SEED, _macros["corlGaitExampleSeed"])
assert _fw + FIG_W <= 150.0, "the two parts must fit in 152 mm with a 2 mm gap"

fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
gsp = fig.add_gridspec(1, 2, width_ratios=RATIOS, wspace=WSPACE)
fig.subplots_adjust(left=LEFT / FIG_W, right=1 - RIGHT / FIG_W,
                    top=1 - TOP / FIG_H, bottom=BOTTOM / FIG_H)
ax_a, ax_b = (fig.add_subplot(gsp[i]) for i in range(2))


def title(ax, text, x=0.0):
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="semibold", color=INK, pad=3, x=x)


# ---------------------------------------------------------------- (b) swing (footfall) diagram
ROW_H = 0.72
GROUP_GAP = 0.5                       # extra space between the two arms [rows]
y_ticks, y_labels, arm_mid = [], [], {}
y = 0.0
for a in ("trot", "c0"):              # bottom to top: trot, then GaitNet
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
ax_a.set_yticklabels(y_labels, fontsize=7)     # rows about 3.1 mm apart at 40 mm height
ax_a.tick_params(axis="y", length=0, pad=2)
ax_a.set_ylim(-0.5, y_ticks[-1] + 0.5)
ax_a.set_xlim(0, WINDOW_S)
ax_a.set_xticks(np.arange(0, WINDOW_S + 1e-9, 0.5))
ax_a.set_xlabel("Time [s]", labelpad=2)
ax_a.tick_params(axis="x", pad=XTICK_PAD)
for x in np.arange(0.5, WINDOW_S, 0.5):
    ax_a.axvline(x, color=GRID, lw=0.5, zorder=0)
ax_a.spines[["top", "right", "left"]].set_visible(False)
for a, name in (("c0", "GaitNet"), ("trot", "OCS2's\nown trot")):
    ax_a.annotate(name, xy=(0, arm_mid[a]), xycoords=("axes fraction", "data"),
                  xytext=(-17, 0), textcoords="offset points", ha="right", va="center",
                  fontsize=7.5, fontweight="semibold", color=ARM_COLOR[a][0], linespacing=1.0)
# Starts over the arm labels, as (a) starts over its row labels.
title(ax_a, f"(b) Swings, {EX_VOID} % voids, {EX_SPEED:.2f} m/s",
      x=-(LEFT - 1.0) / (ax_a.get_position().width * FIG_W))

# ---------------------------------------------------------------- (c) steps/s
for v in gs.SPEEDS:
    ys_ = [STEPS[("c0", v, p)] for p in gs.VOIDS]
    ax_b.plot(gs.VOIDS, ys_, lw=1.3, ms=3.0, mew=0, zorder=3, **SPEED_STYLE[v])
    above = v == 0.2            # 0.10 m/s values go below its line, clear of the trot's
    for p, val in zip(gs.VOIDS, ys_):
        ax_b.annotate(f"{val:.1f}", (p, val), xytext=(0, 3.0 if above else -3.0),
                      textcoords="offset points", ha="center",
                      va="bottom" if above else "top", fontsize=VALUE_PT, color=INK, zorder=4,
                      bbox=HALO)
ys_t = [TROT_STEPS[p] for p in gs.VOIDS]
ax_b.plot(gs.VOIDS, ys_t, color=GRAY, lw=1.2, ls=(0, (3.2, 1.8)), marker="s", ms=2.6, mew=0, zorder=3)
for p, val in zip(gs.VOIDS, ys_t):
    ax_b.annotate(f"{val:.1f}", (p, val), xytext=(0, -3.0), textcoords="offset points",
                  ha="center", va="top", fontsize=VALUE_PT, color=INK, zorder=4, bbox=HALO)
ax_b.set_xticks(gs.VOIDS)
ax_b.spines["bottom"].set_bounds(gs.VOIDS[0] - 5, gs.VOIDS[-1] + 5)
# Direct labels right of the last point instead of a legend.
for txt, yv, col in (("0.20", STEPS[("c0", 0.2, 49)], INDIGO),
                     ("0.10", STEPS[("c0", 0.1, 49)], INDIGO),
                     ("trot", TROT_STEPS[49], GRAY)):
    ax_b.annotate(txt, (gs.VOIDS[-1], yv), xytext=(7, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=VALUE_PT, color=col, fontweight="semibold",
                  bbox=HALO)
ax_b.set_xlim(21, 66)
ax_b.set_ylim(4.0, 8.4)
ax_b.set_yticks([4, 5, 6, 7, 8])
ax_b.spines["left"].set_bounds(4, 8)
ax_b.set_xlabel("Void fraction [%]", labelpad=2)
ax_b.tick_params(axis="x", pad=XTICK_PAD)
ax_b.set_ylabel("Steps per second", labelpad=2)
for g in (5, 6, 7, 8):
    ax_b.axhline(g, color=GRID, lw=0.5, zorder=0)
ax_b.spines[["top", "right"]].set_visible(False)
title(ax_b, "(c) Step rate")   # line labels: speed in m/s (caption)


# ---------------------------------------------------------------- layout checks
def layout_problems(fig):
    """Text outside the canvas, or text on text."""
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
    for i, (t, b) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if b.overlaps(b2):
                out.append(f"text overlap: {t.get_text()!r} / {t2.get_text()!r}")
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
print("diagonal share of two-up time", {a: round(DIAG_TIME[a], 2) for a in gs.ARMS})
print("steps/s", {k: round(v, 2) for k, v in STEPS.items()}, "trot by void", {k: round(v, 2) for k, v in TROT_STEPS.items()})
