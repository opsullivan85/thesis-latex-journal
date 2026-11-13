"""Display-name hook (2026-11-13): the method is called CADENCE in the paper.

Imported by the figure scripts before plotting. Every matplotlib Text (titles, labels,
legend entries, annotations) shows "CADENCE" where the data and checks say "GaitNet",
so the scripts' value checks and dict keys stay unchanged. The baseline is shown
as "Per-candidate + IFM" (its key); the name GaitNet is never printed.
"""
import matplotlib.text as _mt

_orig_set_text = _mt.Text.set_text


def _set_text(self, s):
    if isinstance(s, str):
        s = s.replace("GaitNet", "CADENCE")
    return _orig_set_text(self, s)


_mt.Text.set_text = _set_text
