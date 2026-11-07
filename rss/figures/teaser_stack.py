"""Stack a raster base (PNG) and a vector overlay (PDF) into one PDF page with pdflatex.

The base is embedded with /Interpolate true (graphicx `interpolate=true`), so PDF
viewers smooth it when they scale it to the screen; without the flag some viewers
(poppler's Splash, PDFium) pick pixels when downscaling and tile edges come out
jagged. Both layers must have the same aspect ratio as the page.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def stack(base_png: Path, overlay_pdf: Path, out_pdf: Path, width_in: float, height_in: float) -> None:
    tex = rf"""\documentclass{{article}}
\usepackage[paperwidth={width_in}in,paperheight={height_in}in,margin=0pt]{{geometry}}
\usepackage{{graphicx}}
\pagestyle{{empty}}
\topskip=0pt
\begin{{document}}%
% zero-height box hung from the top edge: a box exactly as tall as the page can
% round past \textheight and push the figure onto a second page
\noindent\raisebox{{-\height}}[0pt][0pt]{{%
\makebox[0pt][l]{{\includegraphics[interpolate=true,width={width_in}in,height={height_in}in]{{base.png}}}}%
\includegraphics[width={width_in}in,height={height_in}in]{{overlay.pdf}}}}%
\end{{document}}
"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        from PIL import Image                       # drop alpha: no soft mask in the PDF
        Image.open(base_png).convert("RGB").save(td / "base.png", optimize=True)
        shutil.copy(overlay_pdf, td / "overlay.pdf")
        (td / "stack.tex").write_text(tex)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "stack.tex"],
                       cwd=td, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages = subprocess.run(["pdfinfo", str(td / "stack.pdf")], capture_output=True, text=True).stdout
        assert re.search(r"^Pages:\s+1$", pages, re.M), pages
        shutil.copy(td / "stack.pdf", out_pdf)
