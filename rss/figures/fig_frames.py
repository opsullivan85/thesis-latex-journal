"""Filmstrips of two recorded OCS2 trials of GaitNet (Fig. fig:frames).

No new trials: every frame is cut from a video that already exists.

  (a) gaitnet_c0_g125_ramp10to60_0.15mps_mujoco_4k.mp4 (3840x2160, 60 fps,
      30.9 s): one OCS2 trial of GaitNet on a track whose void fraction ramps
      from 10 % to 60 % along its length, at 0.15 m/s, re-rendered in MuJoCo.
      There is no state log of this trial, so the void fraction under the
      robot in a frame is NOT known and is not printed.
  (b) gaitnet_c0_g125_49void_0.20mps.mp4 (1920x1080, 60 fps, 21.0 s): one
      OCS2 trial of GaitNet, 49 % voids, 0.20 m/s. The renderer of this clip
      is NOT verified (it may be an offline render of the trial), so the
      figure does not name a simulator or renderer for row (b).

PROVENANCE: both videos come from a separate, earlier evaluation run, NOT
from the runs behind Table 1 (their per-cell counts differ). Every number on
the figure comes from the video file names (void fractions, speeds) or from
the video clock (times); 49 and 0.20 are also checked against the
\\corlBenchVoidHigh and \\corlBenchSpeedFast macros of ../constants.tex.

Frame choice (by eye, from 0.1 s scans of each video, 2026-11-12):
  (a) 7.0, 12.5, 18.0, 23.5, 29.0 s, 5.5 s apart over the ramp. The crop is
      wide (robot about 39 % of the frame width) so that the terrain around
      and ahead of the robot shows; it has more and deeper voids from left to
      right (the start platform is still in view at 7.0 s, the void floor
      shows through from 23.5 s on). Which feet are in the air is not stated
      anywhere: it cannot be read reliably from this oblique view.
  (b) 13.7, 14.1, 14.5, 14.9 s, 0.4 s (24 frames) apart, chosen from a
      0.1 s scan of 11.0-15.6 s so that successive frames show visibly
      different leg configurations on pillars between dark voids. Still no
      claim about which leg is in swing in which frame.
Times are the video clock: frame n is printed as n / 60 s, and the script
checks with ffmpeg's showinfo that the decoded frame has exactly that time.

Clarity rules (author): one fixed crop per row (both cameras follow the
robot), Lanczos downscale only, never upscale, plain frames with one short
time label each and one label per row. Row (a) is cropped wide to show the
terrain; row (b) is cropped closer to show the legs.

Pipeline: ffmpeg decodes each frame losslessly -> PIL crops and (row a)
downscales with Lanczos to ~600 dpi at print size; row (b) is embedded at its
native ~525 dpi -> matplotlib places the frames with interpolation="none",
so the PDF holds exactly these pixels -> ghostscript re-encodes the frames as
JPEG (DCT, QFactor 0.15, no chroma subsampling) and turns all text into
outlines (same purpose as the Inkscape step of fig_bench.py) -> pdftoppm
renders fig_frames.png (300 dpi) from the final PDF.

usage: python3 fig_frames.py   (writes fig_frames.pdf and fig_frames.png here)
"""
import io
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.text import Text
from PIL import Image

HERE = Path(__file__).resolve().parent
CONSTANTS_TEX = HERE.parent / "constants.tex"
VIDEO_DIR = Path("/home/sourav/gaitnet_videos")
FFMPEG, GS = "/usr/bin/ffmpeg", "/usr/bin/gs"
FPS = 60

# ---------------------------------------------------------------- rows
# crop = (x, y, w, h) in source pixels, fixed per row. The robot body sits at
# about the same place in every frame of a row (the cameras follow it).
ROWS = (
    # Row (a): crop 2600 x 1713 px, the bottom edge of the 4K frame; the
    # robot spans about x = 1240-2250 px (39 % of the crop width) in the
    # upper left, with the terrain ahead of and below it in view.
    dict(label="(a) MuJoCo re-render of an OCS2 trial, 10 to 60 % voids, 0.15 m/s",
         video=VIDEO_DIR / "gaitnet_c0_g125_ramp10to60_0.15mps_mujoco_4k.mp4",
         size=(3840, 2160), crop=(620, 447, 2600, 1713),
         times=(7.0, 12.5, 18.0, 23.5, 29.0)),
    # Row (b): crop 960 x 503 px; the robot spans about 57 % of the width.
    dict(label="(b) OCS2 trial, 49 % voids, 0.20 m/s",
         video=VIDEO_DIR / "gaitnet_c0_g125_49void_0.20mps.mp4",
         size=(1920, 1080), crop=(365, 268, 960, 503),
         times=(13.7, 14.1, 14.5, 14.9)),
)

# ---------------------------------------------------------------- label checks
# Row labels must match the video file names and the paper's macros, and row
# (b) must not name a renderer (unverified for that clip).
_ma = re.search(r"ramp(\d+)to(\d+)_([\d.]+)mps", ROWS[0]["video"].name)
assert _ma and f"{_ma[1]} to {_ma[2]} % voids, {_ma[3]} m/s" in ROWS[0]["label"], \
    ROWS[0]["label"]
assert "mujoco" in ROWS[0]["video"].name and "MuJoCo" in ROWS[0]["label"]
_mb = re.search(r"_(\d+)void_([\d.]+)mps", ROWS[1]["video"].name)
assert _mb and f"{_mb[1]} % voids, {_mb[2]} m/s" in ROWS[1]["label"], ROWS[1]["label"]
assert not re.search(r"gazebo|mujoco|isaac", ROWS[1]["label"], re.I), ROWS[1]["label"]
MACROS = dict(re.findall(r"\\newcommand\{\\(corl\w+)\}\{([^}]*)\}",
                         CONSTANTS_TEX.read_text()))
assert MACROS["corlBenchVoidHigh"] == _mb[1], "49 % vs \\corlBenchVoidHigh"
assert MACROS["corlBenchSpeedFast"] == _mb[2], "0.20 m/s vs \\corlBenchSpeedFast"
for row in ROWS:
    for t in row["times"]:
        assert abs(t * FPS - round(t * FPS)) < 1e-6, f"{t} s is not a frame time"

# ---------------------------------------------------------------- layout [mm]
FIG_W = 152.0          # \textwidth of the PMLR class
FIG_H_MAX = 52.0       # page budget for the whole figure
GUTTER = 1.0           # white gap between frames of a row
TITLE_H = 3.4          # strip above a row for its label
TIME_H = 2.7           # strip below a row for the time labels
ROW_GAP = 0.6          # between row (a)'s time strip and row (b)'s label
DPI_TARGET = 600       # embedded resolution at print size (capped at source)
NATIVE_IF_BELOW = 1.2  # keep source pixels if the downscale would be milder
MM = 1 / 25.4


def frame_size_mm(row):
    """Width and height [mm] of one frame of the row, from the crop aspect."""
    n = len(row["times"])
    w = (FIG_W - (n - 1) * GUTTER) / n
    cw, ch = row["crop"][2:]
    return w, w * ch / cw


SIZES = [frame_size_mm(r) for r in ROWS]
# Both rows the same height (to 0.1 mm), so the figure reads as one grid.
assert abs(SIZES[0][1] - SIZES[1][1]) < 0.1, SIZES
FIG_H = sum(TITLE_H + h + TIME_H for _, h in SIZES) + ROW_GAP * (len(ROWS) - 1)
assert FIG_H <= FIG_H_MAX, f"figure height {FIG_H:.1f} mm > {FIG_H_MAX} mm"


# ---------------------------------------------------------------- frames
def grab(video: Path, t: float, size, crop) -> Image.Image:
    """Decode the frame at video time t (exact), losslessly, and crop it."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "info", "-copyts",
           "-ss", f"{t:.4f}", "-i", str(video), "-frames:v", "1",
           "-vf", "showinfo", "-f", "image2pipe", "-vcodec", "png", "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    pts = float(re.search(rb"pts_time:([0-9.]+)", r.stderr).group(1))
    assert abs(pts - t) < 0.25 / FPS, f"{video.name}: asked {t} s, got {pts} s"
    im = Image.open(io.BytesIO(r.stdout))
    im.load()
    assert im.size == tuple(size), (video.name, im.size)
    x, y, w, h = crop
    assert x >= 0 and y >= 0 and x + w <= size[0] and y + h <= size[1], crop
    return im.convert("RGB").crop((x, y, x + w, y + h))


def to_print(im: Image.Image, width_mm: float) -> Image.Image:
    """Lanczos downscale to DPI_TARGET at print size; never upscale."""
    target = round(width_mm * MM * DPI_TARGET)
    if im.width < target * NATIVE_IF_BELOW:
        return im                       # native pixels (row b)
    h = round(target * im.height / im.width)
    return im.resize((target, h), Image.LANCZOS)


# ---------------------------------------------------------------- style
for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7.5,
                     "pdf.fonttype": 42})
INK = "#1F2937"
INK_SOFT = "#4B5563"                    # time labels: same size floor, quieter
TITLE_PT, TIME_PT = 7.5, 7.0

fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
embedded = []                           # (w, h) of every embedded raster
top = FIG_H                             # running top edge [mm from bottom]
for row, (fw, fh) in zip(ROWS, SIZES):
    # Row label, left-aligned like the panel titles of fig_bench.py.
    fig.text(0, (top - TITLE_H + 0.9) / FIG_H, row["label"], ha="left",
             va="baseline", fontsize=TITLE_PT, fontweight="semibold", color=INK)
    y0 = top - TITLE_H - fh
    for k, t in enumerate(row["times"]):
        x0 = k * (fw + GUTTER)
        im = to_print(grab(row["video"], t, row["size"], row["crop"]), fw)
        embedded.append(im.size)
        ax = fig.add_axes((x0 / FIG_W, y0 / FIG_H, fw / FIG_W, fh / FIG_H))
        ax.imshow(np.asarray(im), interpolation="none", aspect="auto")
        ax.set_axis_off()
        fig.text((x0 + fw / 2) / FIG_W, (y0 - TIME_H + 0.75) / FIG_H,
                 f"t = {t:.1f} s", ha="center", va="baseline",
                 fontsize=TIME_PT, color=INK_SOFT)
    top = y0 - TIME_H - ROW_GAP


# ---------------------------------------------------------------- layout checks
def layout_problems(fig):
    """Text off the canvas, too small, on other text, or on a frame."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    frames = [ax.get_window_extent(r) for ax in fig.axes]
    out, canvas = [], fig.bbox
    for t, b in boxes:
        if not (b.x0 >= canvas.x0 and b.y0 >= canvas.y0
                and b.x1 <= canvas.x1 and b.y1 <= canvas.y1):
            out.append(f"outside canvas: {t.get_text()!r}")
        if t.get_fontsize() < 7:
            out.append(f"font below 7 pt: {t.get_text()!r}")
        out += [f"text on a frame: {t.get_text()!r}" for f in frames if b.overlaps(f)]
    for i, (t, b) in enumerate(boxes):
        out += [f"text overlap: {t.get_text()!r} / {t2.get_text()!r}"
                for t2, b2 in boxes[i + 1:] if b.overlaps(b2)]
    return sorted(set(out))


problems = layout_problems(fig)
for p in problems:
    print("LAYOUT:", p)
assert not problems, "layout problems, see above"

# ---------------------------------------------------------------- export
raw = HERE / "fig_frames.raw.pdf"
pdf = HERE / "fig_frames.pdf"
png = HERE / "fig_frames.png"
fig.savefig(raw)                        # exact figure size, no bbox cropping
subprocess.run(
    [GS, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite",
     "-dCompatibilityLevel=1.5", "-dNoOutputFonts", "-dAutoRotatePages=/None",
     "-sColorConversionStrategy=LeaveColorUnchanged",
     "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
     "-dDownsampleColorImages=false", f"-sOutputFile={pdf}",
     "-c", "<< /ColorImageDict << /QFactor 0.15 /Blend 1 "
           "/HSamples [1 1 1 1] /VSamples [1 1 1 1] >> >> setdistillerparams",
     "-f", str(raw)], check=True)
raw.unlink()
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf),
                str(png.with_suffix(""))], check=True)

# ---------------------------------------------------------------- output checks
fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True,
                       check=True).stdout.splitlines()[2:]
assert not fonts, f"fonts left in the PDF (text should be outlines): {fonts}"
listing = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True,
                         text=True, check=True).stdout.splitlines()[2:]
images = [(int(c[3]), int(c[4]), c[8]) for c in (l.split() for l in listing)]
assert [(w, h) for w, h, _ in images] == embedded, (images, embedded)
assert all(enc == "jpeg" for _, _, enc in images), images
info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True,
                      check=True).stdout
pw, ph = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info).groups())
assert abs(pw - FIG_W / 25.4 * 72) < 0.5 and abs(ph - FIG_H / 25.4 * 72) < 0.5, (pw, ph)
size_mb = pdf.stat().st_size / 1e6
assert size_mb < 2.0, f"{pdf.name} is {size_mb:.2f} MB"

for row, (fw, fh) in zip(ROWS, SIZES):
    print(f"{row['label']}: {len(row['times'])} frames of {fw:.1f} x {fh:.1f} mm, "
          f"crop {row['crop'][2]}x{row['crop'][3]} px")
print("embedded px:", embedded,
      "-> dpi:", [round(w / (s[0] * MM)) for (w, _), s in
                  zip(embedded, [SIZES[0]] * len(ROWS[0]["times"]) +
                      [SIZES[1]] * len(ROWS[1]["times"]))])
print(f"figure {FIG_W:.0f} x {FIG_H:.1f} mm, {pdf.name} {size_mb:.2f} MB, no fonts, "
      f"{len(images)} JPEG frames")
print("wrote", pdf, "and", png)
