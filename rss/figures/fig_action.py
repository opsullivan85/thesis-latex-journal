"""Fig. 3 (fig:action): GaitNet's action space at one tick, in the paper's render style.

Two panels, full text width, posed MuJoCo scene (illustrative), same look as the page-1
teaser (light tile caps over dark voids, white background, indigo candidates, Fira Sans):
(a) 4 legs x 16 candidates: the Go1 standing OPAQUE on a training patch; every leg's 16
    candidates (front-right dark indigo, the other legs light indigo). Dots that the
    robot hides from this camera are not drawn.
(b) one leg's candidates: the front-right leg ghosted at each of its 16 candidates
    (inverse kinematics, base and other legs fixed), body opaque.
Legend (author, 2026-11-13: "add legend all the figure"): one white chip in the empty
lower-right corner of (b), in the paper's common legend style (Fira Sans 7 pt, short
handles): dark dot, light dot, ghosted leg. The script asserts that it covers no dot and
no pixel of the robot or of a ghosted leg. Headers and legend use one term, "candidates".

Exact for this terrain and pose (as fig_env.py, whose scene code this reuses): the 1 cm
grid, the 2-cell (2 cm) growth of unsafe cells and the 16 coverage candidates per leg
(teaser_cands.leg_candidates; FL/RL mirrored), Sec. 4.1 and App. E. Posed: the terrain
draw, the robot's position and stance, the 0.15 m/s command for the nominal footholds.

Replaces Owen's renders (author, 2026-11-13: "regenerate the fig3 with match theme of the
paper"). Earlier rejected versions: fig_cands.pdf (2D) and the translucent fig_env panels.

usage: MUJOCO_GL=egl ~/miniconda3/bin/python fig_action.py
       FIG_SEARCH=1 ... prints the visible-dot count for a set of (a) cameras and exits.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import font_manager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

import fig_env_near as N
import fig_env_scene as E
import fig_env_terrain as T
import teaser_cands as C
import teaser_scene as S
import teaser_stack

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir()) / "gaitnet_fig_action_build"
TMP.mkdir(exist_ok=True)

for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7.0, "pdf.fonttype": 42})
INDIGO, INDIGO_LIGHT, INK, CHIP_EDGE = "#4F46E5", "#A5A0F2", "#1F2937", "#D0D5DD"

PPI = int(os.environ.get("FIG_PPI", 450))
SS = 2
W_IN, H_IN = 6.02, float(os.environ.get("FIG_H_IN", 1.80))     # \textwidth x ~46 mm
GAP, RAD_IN = 0.06, 0.04
WA = float(os.environ.get("FIG_WA", 3.20))
WB = W_IN - WA - GAP
XA, XB = 0.0, WA + GAP
px = lambda inch: int(round(inch * PPI))

BASE_WORLD = (42.4875, 11.5875)            # same posed tick as fig_env.py (level-10 patch)
CAM_A = dict(az=float(os.environ.get("CAM_A_AZ", 120.0)), el=float(os.environ.get("CAM_A_EL", -35.0)),
             dist=3.0, fovy=float(os.environ.get("CAM_A_FOVY", 14.5)))
CAM_B = dict(az=float(os.environ.get("CAM_B_AZ", 140.0)), el=float(os.environ.get("CAM_B_EL", -36.0)),
             dist=3.0, fovy=float(os.environ.get("CAM_B_FOVY", 13.0)))
GHOST_ALPHA = 0.30

voids, _ = T.training_terrain()
tick, is_void = N.make_tick(voids, BASE_WORLD)
for leg in E.LEGS:
    assert not any(is_void(x, y) for x, y in tick.cands[leg])
    assert len(tick.cands[leg]) == C.K_CANDIDATES == 16
look_a = (tick.base[0] + 0.06, tick.base[1] + 0.0, float(os.environ.get('LOOK_A_Z', 0.14)))


def view_a(az, el, fovy, w, h):
    return N.render_view(voids, tick, look_a, az, el, 3.0, fovy, w, h)


def visible(v, pts):
    uv = N.proj(v, pts)
    h, w = v.robot.shape
    ok = (uv[:, 0] > 2) & (uv[:, 0] < w - 2) & (uv[:, 1] > 2) & (uv[:, 1] < h - 2)
    vis = np.array([o and not v.robot[int(q[1]), int(q[0])] for o, q in zip(ok, uv)])
    return uv, vis


if os.environ.get("FIG_SEARCH"):
    w, h = px(WA) // 2, px(H_IN) // 2
    for az in (120, 135, 150, 165, 180, 200, 215):
        for el in (-35, -45, -55, -65):
            v = view_a(az, el, CAM_A["fovy"], w, h)
            n = sum(int(visible(v, tick.cands[l])[1].sum()) for l in E.LEGS)
            nfr = int(visible(v, tick.cands["FR"])[1].sum())
            print(f"az {az:4d} el {el:4d}: visible {n:2d}/64, FR {nfr:2d}/16")
    raise SystemExit

# ---- (a) all legs' candidates, robot opaque ------------------------------------------------
wa, ha = px(WA) * SS, px(H_IN) * SS
va = view_a(CAM_A["az"], CAM_A["el"], CAM_A["fovy"], wa, ha)
rgb_a = va.rgb
uv_a, vis_a = {}, {}
for leg in E.LEGS:
    uv_a[leg], vis_a[leg] = visible(va, tick.cands[leg])

# ---- (b) the front-right leg ghosted at its 16 candidates ----------------------------------
hip_fr = tick.hips["FR"]
look_b = (hip_fr[0] + 0.05, hip_fr[1] - 0.04, float(os.environ.get('LOOK_B_Z', 0.13)))
wb, hb = px(WB) * SS, px(H_IN) * SS
vb = N.render_view(voids, tick, look_b, CAM_B["az"], CAM_B["el"], CAM_B["dist"], CAM_B["fovy"], wb, hb)
acc = vb.rgb.astype(float)
ghost_px = np.zeros(vb.robot.shape, bool)                  # union of the ghosted-leg pixels
for x, y in tick.cands["FR"]:
    vk = N.render_view(voids, tick, look_b, CAM_B["az"], CAM_B["el"], CAM_B["dist"], CAM_B["fovy"], wb, hb,
                       legs_pose={"FR": np.array([x, y, S.FOOT_R])})
    foot = vk.data.site_xpos[mujoco.mj_name2id(vk.model, mujoco.mjtObj.mjOBJ_SITE, "FR")]
    assert np.linalg.norm(foot - [x, y, S.FOOT_R]) < 1e-3          # IK reached the candidate
    leg_px = N.leg_mask(vk, "FR")
    ghost_px |= leg_px
    mk = leg_px[..., None] * GHOST_ALPHA
    acc = acc * (1 - mk) + vk.rgb * mk
rgb_b = acc.round().astype(np.uint8)
uv_b, vis_b = visible(vb, tick.cands["FR"])

print(f"(a) visible dots: " + ", ".join(f"{l} {int(vis_a[l].sum())}/16" for l in E.LEGS)
      + f"; (b) visible FR dots {int(vis_b.sum())}/16")


# ---- raster layer ---------------------------------------------------------------------------
def rounded(size, r_px):
    m = Image.new("L", (size[0] * 4, size[1] * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] * 4 - 1, size[1] * 4 - 1], radius=r_px * 4, fill=255)
    return m.resize(size, Image.LANCZOS)


canvas = Image.new("RGB", (px(W_IN), px(H_IN)), "white")
for x0, w, rgb in ((XA, WA, rgb_a), (XB, WB, rgb_b)):
    size = (px(w), px(H_IN))
    canvas.paste(Image.fromarray(np.asarray(rgb, np.uint8)).resize(size, Image.LANCZOS),
                 (px(x0), 0), rounded(size, px(RAD_IN)))
canvas.save(TMP / "base.png")

# ---- vector layer: dots, two header chips and the legend -------------------------------------
fig = plt.figure(figsize=(W_IN, H_IN))
fig.patch.set_alpha(0)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, W_IN)
ax.set_ylim(0, H_IN)


def to_in(uv, shape, x0, w):
    uv = np.atleast_2d(uv)
    hr, wr = shape[:2]
    return np.column_stack([x0 + uv[:, 0] / wr * w, H_IN - uv[:, 1] / hr * H_IN])


CHIP = dict(boxstyle="round,pad=0.26,rounding_size=0.32", fc="white", ec=CHIP_EDGE, lw=0.4, alpha=0.94)
chips = []


def header(x0, text):
    chips.append(ax.text(x0 + 0.07, H_IN - 0.12, text, color=INK, fontsize=7.5, fontweight="semibold",
                         ha="left", va="center", bbox=CHIP, zorder=12))


A = lambda uv: to_in(uv, rgb_a.shape, XA, WA)
for leg in ("FL", "RR", "RL"):
    for (x, y) in A(uv_a[leg][vis_a[leg]]):
        ax.plot(x, y, "o", ms=3.4, mfc=INDIGO_LIGHT, mec="white", mew=0.45, zorder=7)
for (x, y) in A(uv_a["FR"][vis_a["FR"]]):
    ax.plot(x, y, "o", ms=4.0, mfc=INDIGO, mec="white", mew=0.6, zorder=8)
header(XA, "(a) 4 legs × 16 candidates")

Bf = lambda uv: to_in(uv, rgb_b.shape, XB, WB)
for (x, y) in Bf(uv_b[vis_b]):
    ax.plot(x, y, "o", ms=4.0, mfc=INDIGO, mec="white", mew=0.6, zorder=8)
# "candidates", the term of header (a), the legend and the caption (was "16 options"). No "16"
# here: with this camera the rear of the body lies under the chip, and a longer title covers more.
header(XB, "(b) one leg's candidates")


# ---- legend: one chip in the empty lower-right corner of (b) --------------------------------
class GhostLeg:
    """Legend key of a ghosted leg."""


class GhostLegHandler(HandlerBase):
    """Legend icon of a ghosted leg: thigh, calf and round foot, translucent as in (b)."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx, cy = width / 2 - xdescent, height / 2 - ydescent           # handle-box centre [pt]
        hip, knee, foot = (cx + 1.4, cy + 3.3), (cx - 3.0, cy + 0.5), (cx + 2.9, cy - 2.6)
        kw = dict(alpha=0.45, solid_capstyle="round", transform=trans)
        return [Line2D(*zip(hip, knee), lw=2.2, color="#6B7280", **kw),
                Line2D(*zip(knee, foot), lw=1.4, color="#212124", **kw),
                Circle(foot, 1.2, fc="#212124", ec="none", alpha=0.45, transform=trans)]


legend = ax.legend(
    [Line2D([], [], ls="none", marker="o", ms=4.0, mfc=INDIGO, mec="white", mew=0.6),
     Line2D([], [], ls="none", marker="o", ms=3.4, mfc=INDIGO_LIGHT, mec="white", mew=0.45),
     GhostLeg()],
    ["front-right candidates", "other legs' candidates", "leg posed at candidate"],
    handler_map={GhostLeg: GhostLegHandler()},
    # 0.04 in from the panel's right and bottom edges, as the header chips are from its left edge
    loc="lower right", bbox_to_anchor=(XB + WB - 0.04, 0.04), bbox_transform=ax.transData, borderaxespad=0.0,
    # the paper's common legend style; over a picture: a white box with a thin edge, as the chips
    fontsize=7, labelcolor=INK, handlelength=1.6, handletextpad=0.5, columnspacing=1.2, labelspacing=0.3,
    frameon=True, fancybox=True, facecolor="white", edgecolor=CHIP_EDGE, framealpha=0.94, borderpad=0.35)
legend.get_frame().set_linewidth(0.4)
legend.get_frame().set_boxstyle("round", pad=0.0, rounding_size=0.34)
legend.set_zorder(12)

def under(box, x0, w, mask, pad=0.0):
    """Pixels of a panel's mask under a box [in], (x0, y0), (x1, y1), grown by pad; rows count from the top."""
    hr, wr = mask.shape
    c0, c1 = int((box[0, 0] - pad - x0) / w * wr), int(np.ceil((box[1, 0] + pad - x0) / w * wr))
    r0, r1 = int((H_IN - box[1, 1] - pad) / H_IN * hr), int(np.ceil((H_IN - box[0, 1] + pad) / H_IN * hr))
    return mask[max(r0, 0):r1, max(c0, 0):c1]


fig.canvas.draw()
inv = ax.transData.inverted()
dots = np.vstack([A(uv_a[l][vis_a[l]]) for l in E.LEGS] + [Bf(uv_b[vis_b])])
# What a header must leave clear: every dot; in (a) the robot; in (b) the ghosted legs. The rear of
# the body does lie under the (b) header with this camera; its area is printed, not asserted.
for t, (x0, w, clear) in zip(chips, ((XA, WA, va.robot), (XB, WB, ghost_px))):
    bb = inv.transform(t.get_bbox_patch().get_window_extent())
    inside = (dots[:, 0] > bb[0, 0] - 0.03) & (dots[:, 0] < bb[1, 0] + 0.03) & \
             (dots[:, 1] > bb[0, 1] - 0.03) & (dots[:, 1] < bb[1, 1] + 0.03)
    assert not inside.any(), (t.get_text(), dots[inside])
    assert not under(bb, x0, w, clear).any(), (t.get_text(), "header covers the robot in (a) or a ghosted leg in (b)")
bb_b = inv.transform(chips[1].get_bbox_patch().get_window_extent())    # the (b) header's box [in]
body_mm2 = under(bb_b, XB, WB, vb.robot).sum() * (WB / vb.robot.shape[1]) * (H_IN / vb.robot.shape[0]) * 25.4 ** 2
print(f"(b) header {(bb_b[1, 0] - bb_b[0, 0]) * 25.4:.1f} mm wide, over {body_mm2:.0f} mm2 of the rear of the body")
lb_agg = inv.transform(legend.get_frame().get_window_extent())   # legend box as Agg lays it out [in]

fig.savefig(TMP / "overlay_raw.pdf", transparent=True)
# The PDF backend lays the legend out again with its own text metrics (rows about 0.7 pt taller
# than Agg's), so the box that is checked is the one of that draw, read back after the save. Its
# display units are PDF points (72 per inch); the comparison with Agg guards the unit.
lb = legend.get_frame().get_bbox().get_points() / 72.0     # legend box as drawn [in]: (x0, y0), (x1, y1)
assert np.abs(lb - lb_agg).max() < 0.06, (lb, lb_agg)
plt.close(fig)
PAD = 0.04                                                 # clearance kept around the box [in]
assert XB + PAD <= lb[0, 0] and lb[1, 0] <= XB + WB and 0 <= lb[0, 1] and lb[1, 1] <= H_IN, lb
inside = (dots[:, 0] > lb[0, 0] - PAD) & (dots[:, 0] < lb[1, 0] + PAD) & \
         (dots[:, 1] > lb[0, 1] - PAD) & (dots[:, 1] < lb[1, 1] + PAD)
assert not inside.any(), ("legend covers a dot", dots[inside])
assert not under(lb, XB, WB, vb.robot | ghost_px, PAD).any(), "legend covers the robot or a ghosted leg"
print(f"legend box {(lb[1, 0] - lb[0, 0]) * 25.4:.1f} x {(lb[1, 1] - lb[0, 1]) * 25.4:.1f} mm at "
      f"x {lb[0, 0] * 25.4:.1f}, y {lb[0, 1] * 25.4:.1f} mm; clear of dots, robot and ghosted legs by {PAD * 25.4:.1f} mm")
subprocess.run(["inkscape", str(TMP / "overlay_raw.pdf"), "--export-type=pdf", "--export-text-to-path",
                f"--export-filename={TMP / 'overlay.pdf'}"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
teaser_stack.stack(TMP / "base.png", TMP / "overlay.pdf", HERE / "fig_action.pdf", W_IN, H_IN)
subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(HERE / "fig_action.pdf"),
                str(HERE / "fig_action")], check=True)
print("wrote", HERE / "fig_action.pdf")
