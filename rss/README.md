# GaitNet: Where Do I Step?

RSS 2027 submission (double-blind). Build with

    latexmk -pdf main.tex

`main.tex` is the main document (IEEEtran conference template).

- `figures/fig_overview.svg` is Fig. 1, drawn in Inkscape. After editing it, save a PDF copy as `figures/fig_overview.pdf` with text converted to paths.
- `figures/plot_ocs2_results.py` holds the OCS2 benchmark data and writes Fig. 2 (`figures/ocs2_results.pdf`). It needs matplotlib, Fira Sans and Inkscape.
- Numbers used in the text are macros in `constants.tex`.
- `\pending{}` and `\verify{}` mark open items. Set `\revmarkuptrue` in `main.tex` to highlight them.
