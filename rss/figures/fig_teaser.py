"""Page-1 teaser: frames of GaitNet's MuJoCo-rendered run with schematic drawings.

Source video: ~/gaitnet_videos/gaitnet_c0_g125_ramp10to60_0.15mps_mujoco.mp4
(author, 2026-11-05: checkpoint c0, 12.5 cm tiles, voids rising from 10 % to
60 % along the track, 0.15 m/s, rendered in MuJoCo; 1920x1080 at 60 fps).

(a) one frame (t = 25.0 s) with SCHEMATIC drawings: candidate dots on a few
    safe tile tops near the near-side front leg, one of them marked as the
    chosen foothold, a dashed swing arc from that foot, and a pointer to a
    void. The dots are placed on real tile tops (segmented by colour), but
    they are not the policy's logged candidates; the caption says so.
(b) three frames along the same run, showing the voids getting denser.

usage: python fig_teaser.py   (writes fig_teaser.pdf next to this file)
"""
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
VIDEO = Path.home() / "gaitnet_videos" / "gaitnet_c0_g125_ramp10to60_0.15mps_mujoco.mp4"
# Extracted frames are cached outside the repository.
FRAMES = Path(tempfile.gettempdir()) / "gaitnet_teaser_frames"
FRAMES.mkdir(exist_ok=True)

# ------------------------------------------------------------------ style
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7.5, "pdf.fonttype": 42})
INDIGO = "#4F46E5"
INK = "#111827"
HALO_DARK = [pe.withStroke(linewidth=2.2, foreground="#0B1B33")]
HALO_LIGHT = [pe.withStroke(linewidth=2.2, foreground="white")]


# Optional lossless re-render of the same run (same camera): a folder of PNGs
# named t_<seconds with 2 decimals>.png, e.g. t_25.00.png, at any resolution
# with the clip's 16:9 framing. All pixel coordinates below are written for the
# 1920x1080 clip and are scaled to the frames actually used.
RENDER_DIR = os.environ.get("GAITNET_TEASER_FRAMES")
REF_W = 1920


def frame(t: float) -> np.ndarray:
    """Frame at time t [s] as an RGB array, resized to the clip's 1920-px
    coordinate frame only if it is not already 1920 px wide (see SCALE)."""
    if RENDER_DIR:
        src = Path(RENDER_DIR) / f"t_{t:05.2f}.png"
        if not src.exists():
            raise FileNotFoundError(f"{src} (set by GAITNET_TEASER_FRAMES)")
        return np.asarray(Image.open(src).convert("RGB"))
    out = FRAMES / f"t_{t:05.2f}.png"
    if not out.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", str(VIDEO),
                        "-frames:v", "1", str(out)], check=True)
    # H.264 frames are soft: a mild unsharp mask, nothing else.
    img = Image.open(out).convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=2))
    return np.asarray(img)


# ------------------------------------------------------------------ (a) data
T_HERO = 25.0
_full = frame(T_HERO)
SCALE = _full.shape[1] / REF_W                   # 1.0 for the clip, 2.0 for a 4K re-render
HERO_BOX = (250, 140, 1450, 790)                 # x0, y0, x1, y1 in the 1920x1080 frame
OX, OY = HERO_BOX[0], HERO_BOX[1]              # full-frame -> crop offset
_hb = [int(round(v * SCALE)) for v in HERO_BOX]
hero = _full[_hb[1]:_hb[3], _hb[0]:_hb[2]]

# Tile tops render as one flat grey (179, 180, 172); segment them so the dots
# sit on real tile tops. Coordinates below are hero-crop pixels.
top = np.all(np.abs(hero.astype(int) - (179, 180, 172)) <= 6, axis=-1)
lab, n = ndi.label(top)
cent = np.array(ndi.center_of_mass(top, lab, range(1, n + 1)))[:, ::-1] / SCALE  # (x, y), 1920-frame crop px
size = ndi.sum(top, lab, range(1, n + 1)) / SCALE ** 2


def tile_near(xy):
    """Centroid of the segmented tile top nearest to xy (tops of >= 400 px only)."""
    ok = size >= 400
    d = np.where(ok, ((cent - np.asarray(xy)) ** 2).sum(1), np.inf)
    return cent[int(np.argmin(d))]


# Positions below are full-frame pixels, shifted into the crop.
FOOT = np.array([995.0, 645.0]) - (OX, OY)       # near-side front foot at t = 25.0 s
CAND_GUESS = [np.array(p) - (OX, OY) for p in
              [(1110, 600), (1200, 615), (907, 762), (1105, 515), (1000, 672), (1250, 578)]]
CANDS = np.array([tile_near(p) for p in CAND_GUESS])
CHOSEN = CANDS[0]
VOID_AT = np.array([1056.0, 751.0]) - (OX, OY)   # blue floor seen through a gap

# ------------------------------------------------------------------ (b) data
T_STRIP = (10.0, 27.0)
STRIP_CENTER = (830, 420)                        # robot, full-frame pixels (camera follows it)
STRIP_H_PX = 560                                 # crop height; width follows the panel aspect


# ------------------------------------------------------------------ layout [in]
W, H = 6.02, 2.30                                # \textwidth = 433.6 pt
GAP, VGAP = 0.07, 0.05
hero_w = H * (HERO_BOX[2] - HERO_BOX[0]) / (HERO_BOX[3] - HERO_BOX[1])
col_x = hero_w + GAP
col_w = W - col_x
strip_h = (H - VGAP * (len(T_STRIP) - 1)) / len(T_STRIP)
_sw = int(round(STRIP_H_PX * col_w / strip_h))
STRIP_BOX = (STRIP_CENTER[0] - _sw // 2, STRIP_CENTER[1] - STRIP_H_PX // 2,
             STRIP_CENTER[0] - _sw // 2 + _sw, STRIP_CENTER[1] + STRIP_H_PX // 2)
_sb = [int(round(v * SCALE)) for v in STRIP_BOX]
strips = [frame(t)[_sb[1]:_sb[3], _sb[0]:_sb[2]] for t in T_STRIP]

fig = plt.figure(figsize=(W, H))


def axes_in(x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_axis_off()
    return ax


# (a)
ax = axes_in(0, 0, hero_w, H)
HW, HH = HERO_BOX[2] - HERO_BOX[0], HERO_BOX[3] - HERO_BOX[1]
# interpolation="none": the PDF embeds the native pixels (no resampling).
ax.imshow(hero, interpolation="none", extent=(0, HW, HH, 0))
ax.set_xlim(0, HW); ax.set_ylim(HH, 0)

# swing arc: quadratic Bezier from the foot to the chosen foothold
ctrl = (FOOT + CHOSEN) / 2 + np.array([0.0, -110.0])
s = np.linspace(0, 1, 60)[:, None]
arc = (1 - s) ** 2 * FOOT + 2 * (1 - s) * s * ctrl + s ** 2 * CHOSEN
ax.plot(arc[:-3, 0], arc[:-3, 1], color="white", lw=2.6, solid_capstyle="round", zorder=4)
ax.plot(arc[:-3, 0], arc[:-3, 1], color=INDIGO, lw=1.3, ls=(0, (3, 2)), zorder=5)
ax.add_patch(FancyArrowPatch(arc[-6], arc[-2], arrowstyle="-|>", mutation_scale=8,
                             color=INDIGO, lw=1.2, zorder=6))

for c in CANDS[1:]:
    ax.plot(*c, "o", ms=5.2, mfc="white", mec=INDIGO, mew=1.2, zorder=6)
ax.plot(*CHOSEN, "o", ms=6.5, mfc=INDIGO, mec="white", mew=1.2, zorder=7)

ax.annotate("candidates", xy=CANDS[1], xytext=(1260 - OX, 685 - OY), color=INK, fontsize=7.5,
            fontweight="semibold", path_effects=HALO_LIGHT, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=4), zorder=8)
ax.annotate("chosen foothold", xy=CHOSEN, xytext=(1155 - OX, 460 - OY), color="white", fontsize=7.5,
            fontweight="semibold", path_effects=HALO_DARK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="white", lw=0.8, shrinkA=1, shrinkB=5), zorder=8)
ax.annotate("void", xy=VOID_AT, xytext=(1160 - OX, 750 - OY), color=INK, fontsize=7.5,
            fontweight="semibold", path_effects=HALO_LIGHT, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=2), zorder=8)
ax.text(18, 30, "(a) Every 40 ms: which leg, where, for how long, or no step",
        color="white", fontsize=7.5, fontweight="semibold", path_effects=HALO_DARK,
        ha="left", va="top", zorder=8)

# (b)
for i, img in enumerate(strips):
    y = H - (i + 1) * strip_h - i * VGAP
    axb = axes_in(col_x, y, col_w, strip_h)
    sw, sh = STRIP_BOX[2] - STRIP_BOX[0], STRIP_BOX[3] - STRIP_BOX[1]
    axb.imshow(img, interpolation="none", extent=(0, sw, sh, 0))
    axb.set_xlim(0, sw); axb.set_ylim(sh, 0)
    if i == 0:
        axb.text(16, 22, "(b) Voids rise from 10 % to 60 %", color="white", fontsize=7.5,
                 fontweight="semibold", path_effects=HALO_DARK, ha="left", va="top")

raw = FRAMES / "fig_teaser_raw.pdf"
fig.savefig(raw, dpi=320)
png = HERE / "fig_teaser.png"
fig.savefig(png, dpi=200)
subprocess.run(["inkscape", str(raw), "--export-type=pdf", "--export-text-to-path",
                f"--export-filename={HERE / 'fig_teaser.pdf'}"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", HERE / "fig_teaser.pdf", "hero_w=%.2f in" % hero_w, "chosen", CHOSEN.round(1))
