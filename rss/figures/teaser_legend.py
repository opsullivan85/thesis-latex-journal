"""Legend row of the teaser (Fig. 1): one entry per kind of mark.

Shared by fig_teaser.py (full MuJoCo build) and fig_teaser_relegend.py (redraws only
this row on an existing fig_teaser.pdf, for machines without the Go1 scene).
The paper's common legend style (Fira Sans 7 pt, ink, no frame, short handles), as in
the other figure scripts. Handles repeat the marks' own colours, sizes and dashes.
"""
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

INDIGO, INK, EMERALD = "#4F46E5", "#1F2937", "#0F9F6E"
STYLE = dict(frameon=False, fontsize=7, labelcolor=INK, handlelength=1.6,
             handletextpad=0.5, columnspacing=1.0, labelspacing=0.3)   # 1.0, not 1.2: eight entries
R_NO = 3.2                           # legend ring radius [pt]; six whole dashes around it
DASH = 2 * np.pi * R_NO / 6


def entries(tint, void):
    """tint: safe-cell swatch as on a lit tile top in (a); void: a void as it shows in (a)."""
    return [
        (Patch(fc=void, ec="none"), "void"),                                          # dark hole, all panels
        (Patch(fc=tint, ec="none"), "safe cells"),                                    # tint, (a) and (c)
        (Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=INDIGO, mew=1.1), "candidate"),
        (Line2D([], [], ls="none", marker="o", ms=6.8, mfc=INDIGO, mew=0), "chosen foothold"),
        (Line2D([], [], color=INDIGO, lw=2.0, solid_capstyle="round"), "swing path"),
        (Line2D([], [], color=INDIGO, lw=1.2, ls=(0, (2.5, 1.8))), "path ahead"),      # (b), ahead of the foot
        (Circle((0, 0), R_NO, fc="none", ec=INK, lw=1.0, ls=(0, (0.58 * DASH, 0.42 * DASH))), "no IFM"),
        (Circle((0, 0), 2.3, fc="none", ec=EMERALD, lw=1.3), "refitted IFM"),
    ]


# a ring's handle is a ring of its own radius [pt], centred in the handle box
_ring = HandlerPatch(patch_func=lambda legend, orig_handle, xdescent, ydescent, width, height, fontsize:
                     Circle((width / 2 - xdescent, height / 2 - ydescent), orig_handle.get_radius()))


def draw(ax, x_center, tint, void, y=0.008):
    """Legend row centred at x_center [data units], bottom at y; returns (legend, labels)."""
    items = entries(tint, void)
    legend = ax.legend([h for h, _ in items], [t for _, t in items], ncol=len(items), loc="lower center",
                       bbox_to_anchor=(x_center, y), bbox_transform=ax.transData,
                       borderpad=0.0, borderaxespad=0.0, handler_map={Circle: _ring}, **STYLE)
    legend.set_zorder(20)
    return legend, [t for _, t in items]
