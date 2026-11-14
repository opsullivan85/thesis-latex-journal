"""Redraw only the legend row of fig_teaser.pdf (Fig. 1), keeping its panels as they are.

For machines without MuJoCo or the Go1 scene that fig_teaser.py renders. The panels
of the current fig_teaser.pdf (the canvas above Y_PAN) are kept as one clipped PDF
include, raster and vector marks untouched; the row below them is redrawn from
teaser_legend.py, so it matches what fig_teaser.py would draw. Running it twice gives
the same figure.

Swatch colours (fig_teaser.py derives them from the render's masks):
- TINT: the safe-cells swatch of the 2026-11-04 build, read off its legend.
- VOID: mean colour inside the void below the swing arc in (a) of the embedded raster.

usage: ~/miniconda3/bin/python fig_teaser_relegend.py
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

import teaser_legend

HERE = Path(__file__).resolve().parent
PDF = HERE / "fig_teaser.pdf"
TINT, VOID = "#aba8be", "#21242a"
W_IN, H_IN = 6.02, 2.15              # as in fig_teaser.py

for f in ("/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
          "/usr/share/fonts/opentype/fira/FiraSans-SemiBold.otf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Fira Sans", "font.size": 7.5, "pdf.fonttype": 42})

info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True, check=True).stdout
w_pt, h_pt = map(float, re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
assert abs(w_pt / 72 - W_IN) < 1e-3 and abs(h_pt / 72 - H_IN) < 1e-3, (w_pt, h_pt)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(PDF, td / "old.pdf")
    # panels' bottom edge: the lowest raster row that is not plain white
    subprocess.run(["pdfimages", "-png", str(td / "old.pdf"), str(td / "img")], check=True)
    base = np.asarray(Image.open(td / "img-000.png").convert("RGB"))
    rows = np.nonzero((base < 250).any(axis=(1, 2)))[0]
    y_pan = H_IN * (1 - (rows.max() + 1) / base.shape[0])
    assert 0.1 < y_pan < 0.2, y_pan

    fig = plt.figure(figsize=(W_IN, H_IN))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    legend, labels = teaser_legend.draw(ax, W_IN / 2, TINT, VOID)
    fig.canvas.draw()
    lb = ax.transData.inverted().transform(legend.get_tightbbox(fig.canvas.get_renderer()))
    assert [t.get_text() for t in legend.get_texts()] == labels
    assert lb[:, 0].min() >= 0.02 and lb[:, 0].max() <= W_IN - 0.02, np.round(lb, 3)
    assert lb[:, 1].min() >= 0.0 and lb[:, 1].max() <= y_pan - 0.03, np.round(lb, 3)
    print(f"legend row x {lb[0, 0]:.3f}-{lb[1, 0]:.3f} in, y {lb[0, 1]:.3f}-{lb[1, 1]:.3f} in; "
          f"panels from y = {y_pan:.3f} in")
    fig.savefig(td / "legend_raw.pdf", transparent=True)
    plt.close(fig)
    subprocess.run(["inkscape", str(td / "legend_raw.pdf"), "--export-type=pdf", "--export-text-to-path",
                    f"--export-filename={td / 'legend.pdf'}"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cut = y_pan - 0.01                   # keep the panels, drop the old legend row
    (td / "stack.tex").write_text(rf"""\documentclass{{article}}
\usepackage[paperwidth={W_IN}in,paperheight={H_IN}in,margin=0pt]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\topskip=0pt
\begin{{document}}%
\noindent\raisebox{{-\height}}[0pt][0pt]{{%
\makebox[0pt][l]{{\raisebox{{{cut}in}}{{\includegraphics[trim=0 {cut}in 0 0,clip]{{old.pdf}}}}}}%
\includegraphics[width={W_IN}in,height={H_IN}in]{{legend.pdf}}}}%
\end{{document}}
""")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "stack.tex"],
                   cwd=td, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pages = subprocess.run(["pdfinfo", str(td / "stack.pdf")], capture_output=True, text=True).stdout
    assert re.search(r"^Pages:\s+1$", pages, re.M), pages
    shutil.copy(td / "stack.pdf", PDF)
subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(PDF), str(HERE / "fig_teaser")], check=True)
print("wrote", PDF)
