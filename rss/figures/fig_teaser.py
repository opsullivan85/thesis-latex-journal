"""Page-1 teaser (Fig. 1): three panels of one GaitNet step in a POSED MuJoCo scene.

(a) The decision tick: FR's safe cells, its 16 candidates and the chosen foothold u*.
(b) Later in the same step: the controller swings FR over the void toward u* (same
    camera as (a), panned forward, same scale).
(c) Close-up of u*: OCS2's RMS landing error on flat ground, without and with the
    refitted IFM, as circles about u*.
A legend row under the panels names every mark (2026-11-13, author: "add legend all
the figure"); "void" with a dark swatch added on 2026-11-13. It replaces the leader labels "safe candidates" and "chosen foothold",
and the words of "no IFM: 6 cm" and "refitted IFM: 2 cm" (the two values stay on (c)).
The canvas is unchanged: the panels are the same camera frames as before with CROP_TOP
cut at the top and the rest of LEG_H at the bottom (no mark, no part of the robot).
The script asserts both, and that the legend row lies clear of the panels.

What is exact and what is posed (the caption says so):
- Posed: the terrain (a seeded 40 % draw with hand edits near the stepping leg; not
  a benchmark world), the robot poses, the stepping leg (FR), the chosen foothold
  (FR's pinned nominal candidate, not a policy output), and the swing arc's shape
  (apex inside the training range \\corlRandApexMin-\\corlRandApexMax m).
  No checkpoint is run.
- Exact for this terrain and pose: FR's safe cells (1 cm grid, +-18 cm about the
  hip, unsafe cells grown by k = 2, so a 2 cm margin; Sec. 4.1) and its 16
  candidates (coverage sampler, App. E), drawn by 3D projection through the
  render camera. Candidates hidden behind the FR leg are not drawn.
- (c): circles of radius \\corlOcsLandRmsNoIdm and \\corlOcsLandRmsIdm cm about u*,
  OCS2's RMS of ||e|| on flat ground without and with the refitted IFM (Sec. 5,
  App. A remark). Centring them on u* is schematic.

Earlier teasers are in git history: frames of the MuJoCo re-render of one OCS2 trial
(commit cf888d4).

Pipeline: MuJoCo renders at 2x (teaser_hero.py, teaser_scene.py) -> safe-cell tint
and Lanczos downsample -> one base PNG at PPI; dots, arcs, circles and labels ->
transparent vector PDF (text to paths, inkscape); teaser_stack.py stacks the two
with pdflatex, flagging the raster /Interpolate so PDF viewers smooth it.

usage: MUJOCO_GL=egl ~/miniconda3/bin/python fig_teaser.py   (MuJoCo 3.10, EGL)
       TEASER_PPI=150 TEASER_OUT=/some/dir ... for a draft that leaves the paper's PDF alone
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import teaser_cands as C
import teaser_hero as hero
import teaser_scene as S
import teaser_legend
import teaser_stack

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TEASER_OUT", HERE))      # fig_teaser.pdf/.png go here
TMP = Path(tempfile.gettempdir()) / "gaitnet_teaser_build"
TMP.mkdir(exist_ok=True)

# ---- numbers drawn on the figure, checked against constants.tex --------------------
_const = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}",
                                                       (HERE.parent / "constants.tex").read_text())}
TICK_MS = int(_const["corlControlTickMs"])
RMS_NO, RMS_FIT = float(_const["corlOcsLandRmsNoIdm"]), float(_const["corlOcsLandRmsIdm"])
APEX_MIN, APEX_MAX = float(_const["corlRandApexMin"]), float(_const["corlRandApexMax"])
assert int(_const["corlTrainCandidatesPerLeg"]) == C.K_CANDIDATES == 16
assert float(_const["corlMarginCm"]) == 100 * C.GROW_CELLS * C.GRID_H_M == 2
assert float(_const["corlScanResCm"]) == 100 * C.GRID_H_M
assert float(_const["corlGridReachCm"]) == 100 * C.GRID_REACH_M
assert float(_const["corlBenchTileCm"]) == 100 * S.TILE
assert float(_const["corlVoidDepth"]) == S.DEPTH
assert abs(float(_const["corlNomHorizonS"]) - C.NOMINAL_T_S) < 1e-9

# ---- style and layout (inches; origin at the figure's bottom-left) ----------------------
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7.5, "pdf.fonttype": 42})
INDIGO, INK, EMERALD, EMERALD_TEXT, CHIP_EDGE = "#4F46E5", "#1F2937", "#0F9F6E", "#047857", "#D0D5DD"

PPI = int(os.environ.get("TEASER_PPI", 450))   # embedded raster resolution (lower for drafts)
SS = 2                               # render supersampling
W_IN, H_IN = 6.02, 2.15              # \textwidth = 433.6 pt (canvas, unchanged by the legend)
GAP, RAD_IN = 0.06, 0.04             # white gap between panels; panel corner radius
# Legend row under the panels, inside the same canvas. The cameras still render the
# H_REN-high frames of the version without a legend (same pose, fovy and scale); each
# frame loses CROP_TOP at the top and LEG_H - CROP_TOP at the bottom, nothing else.
LEG_H = 0.14                         # height of the legend row (7 pt text, 3 pt under the panels)
H_REN, CROP_TOP = 2.15, 0.06         # rendered frame height; part of it cut at the top
WA, WB = 2.62, 1.80                  # panel widths; (c) takes the rest
WC = W_IN - WA - WB - 2 * GAP
XA, XB, XC = 0.0, WA + GAP, WA + WB + 2 * GAP
px = lambda inch: int(round(inch * PPI))
H_PAN_PX, CROP_PX = px(H_IN - LEG_H), px(CROP_TOP)          # panel height; rows cut at the top
assert 0 <= CROP_PX and CROP_PX + H_PAN_PX <= px(H_REN)
Y_PAN = H_IN * (1 - H_PAN_PX / px(H_IN))                    # panels' bottom edge [figure in]

# robot camera, shared by (a) and (b): azimuth 98 deg, elevation -36 deg, 2.4 m from
# the look point. (b) pans the look point along x only, so scale and view direction match.
FOVY = 16.1
LOOK_A = (-0.01, 0.0, 0.14)
LOOK_B = (0.20, 0.0, 0.14)
# (b): base advanced by 1.5 cm (0.15 m/s for 0.1 s); FR halfway along its swing (posed)
BASE_B = (hero.BASE[0] + 0.015, hero.BASE[1], hero.BASE[2])
# close-up (c): steeper camera on u*, robot hidden
ZOOM = dict(zoom_center=(0.288, -0.124, 0.0), zoom_el=-62.0, zoom_dist=2.4, zoom_fovy=5.7)


def render_a(apex):
    return hero.render_hero(px(WA) * SS, px(H_REN) * SS, look=LOOK_A, fovy=FOVY, apex=apex,
                            zoom_px=(px(WC) * SS, px(H_REN) * SS), **ZOOM)


# ---- (a): pick the swing apex (training range) that keeps the arc clear of every dot ----
best = None
for apex in np.round(np.arange(APEX_MIN, APEX_MAX + 1e-9, 0.01), 2):
    h = render_a(apex)
    near = ndi.binary_dilation(h.robot, iterations=int(0.03 * PPI * SS))
    shown = [k for k, (u, v) in enumerate(h.cand_uv) if k > 0 and not near[int(v), int(u)]]
    arc_vis = np.array([p for p in h.arc_uv if not h.robot[int(p[1]), int(p[0])]])
    clear = min(np.hypot(*(arc_vis - h.cand_uv[k]).T).min() for k in shown) / (PPI * SS)
    if best is None or clear > best[0]:
        best = (clear, apex, h, shown)
clear, apex, ha, shown = best
assert clear >= 0.06, f"swing arc within {clear:.3f} in of a candidate dot"
assert ha.info["clearance_u"] >= 0.02 - 1e-9          # u* keeps the 2 cm margin

# ---- (b): FR at the middle of the same swing path (sphere centre on the arc) ------------
p0 = np.array(hero.FEET["FR"]); p2 = np.array([*ha.cand_xy[0], S.FOOT_R])
mid = 0.5 * (p0 + p2); mid[2] = S.FOOT_R + apex
feet_b = dict(hero.FEET, FR=tuple(mid))
hb = hero.render_hero(px(WB) * SS, px(H_REN) * SS, look=LOOK_B, fovy=FOVY, apex=apex,
                      base=BASE_B, feet=feet_b, swing=("FR",))
assert hero.tile_of(mid[0], mid[1]) in hero.void_layout()   # the foot is over the void
print(f"apex {apex} m, arc-dot clearance {clear:.3f} in, step {ha.info['step']:.3f} m, "
      f"u* {np.round(ha.info['u_star'], 4)}, clearance {ha.info['clearance_u']:.4f} m, "
      f"cells safe/void/grown {ha.info['safe']}/{ha.info['void_cells']}/{ha.info['grown']}, "
      f"dots drawn {len(shown) + 1} of 16, (b) foot {np.round(mid, 3)}")


# ---- raster layer ----------------------------------------------------------------------
def tinted(rgb, mask, alpha=0.18):
    a = alpha * mask[..., None]
    return (rgb.astype(float) * (1 - a) + np.array([79, 70, 229], float) * a).round().astype(np.uint8)


def rounded(size, r_px):
    m = Image.new("L", (size[0] * 4, size[1] * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] * 4 - 1, size[1] * 4 - 1], radius=r_px * 4, fill=255)
    return m.resize(size, Image.LANCZOS)


canvas = Image.new("RGB", (px(W_IN), px(H_IN)), "white")
rgb_a = tinted(ha.rgb, ha.safe_mask)
for x0, w, rgb in ((XA, WA, rgb_a), (XB, WB, hb.rgb),
                   (XC, WC, tinted(ha.zoom["rgb"], ha.zoom["safe_mask"]))):
    frame = Image.fromarray(rgb).resize((px(w), px(H_REN)), Image.LANCZOS)
    size = (px(w), H_PAN_PX)
    canvas.paste(frame.crop((0, CROP_PX, px(w), CROP_PX + H_PAN_PX)), (px(x0), 0), rounded(size, px(RAD_IN)))
canvas.save(TMP / "base.png")
# the legend's swatch is the tint as it shows on a lit tile top in (a)
TINT = to_hex(np.median(rgb_a[ha.safe_mask], axis=0) / 255)
# and the void's swatch is a void as it shows in (a): neither tile top nor robot
VOID = to_hex(np.median(ha.rgb[~ha.cap & ~ha.robot], axis=0) / 255)

# ---- vector layer (figure inches) --------------------------------------------------------
fig = plt.figure(figsize=(W_IN, H_IN))
fig.patch.set_alpha(0)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, W_IN)
ax.set_ylim(0, H_IN)


def to_in(uv, img_shape, x0, w):
    """Render pixels (u right, v down) of a panel's full frame -> figure inches.
    Frame row CROP_PX (at PPI) is the panel's top edge, the canvas's top edge."""
    uv = np.atleast_2d(uv)
    hr, wr = img_shape[:2]
    return np.column_stack([x0 + uv[:, 0] / wr * w,
                            H_IN * (1 - (uv[:, 1] / hr * px(H_REN) - CROP_PX) / px(H_IN))])


CHIP = dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc="white", ec=CHIP_EDGE, lw=0.4, alpha=0.94)
chips = []


def chip_text(x, y, text, color=INK, ha="left", size=7.5, z=12):
    chips.append(ax.text(x, y, text, color=color, fontsize=size, fontweight="semibold", ha=ha,
                         va="center", bbox=CHIP, zorder=z))


def header(x0, text):
    chip_text(x0 + 0.08, H_IN - 0.13, text, size=7.5)


def leader(p_from, p_to, color=INK, lw=0.5):
    ax.annotate("", xy=p_to, xytext=p_from, zorder=11,
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=0, shrinkB=0))


def halo_line(xy, color, lw, ls="-", z=5):
    ax.plot(xy[:, 0], xy[:, 1], color="white", lw=lw + 1.8, solid_capstyle="round", zorder=z - 0.5)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)


def dot(p, filled, z=8):
    if filled:
        ax.plot(*p, "o", ms=8, mfc=INDIGO, mec="white", mew=1.2, zorder=z)
    else:
        ax.plot(*p, "o", ms=5.5, mfc="white", mec=INDIGO, mew=1.1, zorder=z)


# (a) decision: swing arc (schematic shape, exact endpoints), candidate dots, u*
A = lambda uv: to_in(uv, ha.rgb.shape, XA, WA)
arc_a = A(np.array([p for p in ha.arc_uv if not ha.robot[int(p[1]), int(p[0])]]))
u_a = A(ha.cand_uv[0])[0]
arc_a = arc_a[np.hypot(*(arc_a - u_a).T) > 0.075]
halo_line(arc_a, INDIGO, 2.0)
ax.add_patch(FancyArrowPatch(arc_a[-3], arc_a[-1] + (u_a - arc_a[-1]) * 0.45, arrowstyle="-|>",
                             mutation_scale=9, color=INDIGO, lw=0, zorder=6, shrinkA=0, shrinkB=0))
dots = A(ha.cand_uv[shown])
for p in dots:
    dot(p, False, z=7)
dot(u_a, True)
header(XA, f"(a) Every {TICK_MS} ms: one step or none")

# (b) swing in progress: path behind the foot solid, ahead of it dashed, u* filled
B = lambda uv: to_in(uv, hb.rgb.shape, XB, WB)
n = len(hb.arc_uv)
vis = np.array([not hb.robot[int(v), int(u)] for u, v in hb.arc_uv])
done, ahead = np.arange(n) < n // 2, np.arange(n) >= n // 2
u_b = B(hb.cand_uv[0])[0]
trail = B(hb.arc_uv[done & vis])
halo_line(trail, INDIGO, 2.0)
rest = B(hb.arc_uv[ahead & vis])
rest = rest[np.hypot(*(rest - u_b).T) > 0.06]
halo_line(rest, INDIGO, 1.2, ls=(0, (2.5, 1.8)))
dot(u_b, True)
header(XB, "(b) The controller swings the leg")

# (c) close-up of u*: RMS landing-error circles (radii exact, centring schematic)
Z = lambda uv: to_in(uv, ha.zoom["rgb"].shape, XC, WC)
ring_no = Z(ha.zoom["ring"](RMS_NO / 100))
ring_fit = Z(ha.zoom["ring"](RMS_FIT / 100))
for r, col, ls, lw in ((ring_no, INK, (0, (3, 2)), 1.3), (ring_fit, EMERALD, "-", 1.6)):
    ax.plot(r[:, 0], r[:, 1], color="white", lw=lw + 1.6, zorder=13)
    ax.plot(r[:, 0], r[:, 1], color=col, lw=lw, ls=ls, zorder=14)
u_c = Z(ha.zoom["u_px"])[0]
dot(u_c, True, z=15)
header(XC, "(c) Landing error on OCS2")
# the two values stay on the picture; the legend says which circle is which
p6 = ring_no[np.argmax(ring_no[:, 1])]                             # top of the 6 cm circle
y6 = p6[1] + 0.16
chip_text(p6[0], y6, f"{RMS_NO:g} cm", ha="center", z=16)
leader((p6[0], y6 - 0.05), p6)
p2 = ring_fit[np.argmin(ring_fit[:, 1])]                           # bottom of the 2 cm circle
y2 = Y_PAN + 0.13
chip_text(p2[0], y2, f"{RMS_FIT:g} cm", color=EMERALD_TEXT, ha="center", z=16)
leader((p2[0], y2 + 0.06), p2, color=EMERALD_TEXT)

# ---- legend row under the panels: one entry per kind of mark (teaser_legend.py) ---------
legend, LEGEND_LABELS = teaser_legend.draw(ax, W_IN / 2, TINT, VOID)

# every chip inside its panel
fig.canvas.draw()
inv = ax.transData.inverted()
chip_bb = []
for t in chips:
    bb = inv.transform(t.get_bbox_patch().get_window_extent())
    x0b, x1b = bb[:, 0]
    panel = [(a, a + w) for a, w in ((XA, WA), (XB, WB), (XC, WC)) if a - 0.01 <= x0b <= a + w]
    assert panel and x1b <= panel[0][1] - 0.02, (t.get_text(), np.round(bb, 3))
    assert bb[:, 1].min() >= Y_PAN + 0.02 and bb[:, 1].max() <= H_IN - 0.02, (t.get_text(), np.round(bb, 3))
    chip_bb.append(bb)


def on_robot(bb, h, x0, w, pad=0.02):
    """True if a figure-inch box, grown by pad, lies on a robot pixel of panel frame h."""
    hr, wr = h.robot.shape
    u = np.clip(((bb[:, 0] + [-pad, pad] - x0) / w * wr).astype(int), 0, wr)
    row = (1 - (bb[:, 1] + [-pad, pad]) / H_IN) * px(H_IN) + CROP_PX          # frame rows at PPI
    v = np.clip((row / px(H_REN) * hr).astype(int), 0, hr)
    return bool(h.robot[v.min():v.max(), u.min():u.max()].any())


# the crop moved the header chips CROP_TOP toward the robots: no chip on the robot
assert not on_robot(chip_bb[0], ha, XA, WA) and not on_robot(chip_bb[1], hb, XB, WB)
# the crop cuts no mark and no part of the robot: each at least 0.04 in inside its panel
MARK_R = 0.064                       # largest mark radius: the filled dot with its white rim
marks = np.vstack([arc_a, dots, [u_a], trail, rest, [u_b], ring_no, ring_fit, [u_c]])
assert marks[:, 1].min() >= Y_PAN + 0.04 + MARK_R and marks[:, 1].max() <= H_IN - 0.04 - MARK_R, \
    (marks[:, 1].min(), marks[:, 1].max())
for h in (ha, hb):
    v = np.nonzero(h.robot.any(axis=1))[0] / h.robot.shape[0] * px(H_REN)
    assert v.min() >= CROP_PX + 0.04 * PPI and v.max() <= CROP_PX + H_PAN_PX - 0.04 * PPI
# the legend row: one entry per kind of mark, inside the canvas, clear of the panels
assert [t.get_text() for t in legend.get_texts()] == LEGEND_LABELS
lb = inv.transform(legend.get_tightbbox(fig.canvas.get_renderer()))
assert lb[:, 0].min() >= 0.02 and lb[:, 0].max() <= W_IN - 0.02, np.round(lb, 3)
assert lb[:, 1].min() >= 0.0 and lb[:, 1].max() <= Y_PAN - 0.03, np.round(lb, 3)
print(f"legend row x {lb[0, 0]:.3f}-{lb[1, 0]:.3f} in, y {lb[0, 1]:.3f}-{lb[1, 1]:.3f} in; "
      f"panels from y = {Y_PAN:.3f} in; tint swatch {TINT}, void swatch {VOID}")

fig.savefig(TMP / "overlay_raw.pdf", transparent=True)
plt.close(fig)
subprocess.run(["inkscape", str(TMP / "overlay_raw.pdf"), "--export-type=pdf", "--export-text-to-path",
                f"--export-filename={TMP / 'overlay.pdf'}"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
teaser_stack.stack(TMP / "base.png", TMP / "overlay.pdf", OUT / "fig_teaser.pdf", W_IN, H_IN)
subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(OUT / "fig_teaser.pdf"),
                str(OUT / "fig_teaser")], check=True)
print("wrote", OUT / "fig_teaser.pdf")
