"""Display-name hook (2026-11-13): the method is called CADENCE in the paper.

Imported by the figure scripts before plotting. Every matplotlib Text (titles, labels,
legend entries, annotations) shows "CADENCE" where the data and checks say "GaitNet",
so the scripts' value checks and dict keys stay unchanged. The baseline key
"Per-candidate + IFM" is shown as "GaitNet + IFM".
"""
import matplotlib.text as _mt

_orig_set_text = _mt.Text.set_text
_BASELINE = "GaitNet + IFM"


def _set_text(self, s):
    if isinstance(s, str):
        # 2026-11-13: the baseline row is printed as "GaitNet + IFM" (first author's
        # naming; GaitNet = the earlier planner that Sec. 4 presents). The scripts keep
        # the key "Per-candidate + IFM". matplotlib passes strings through set_text more
        # than once, so the baseline name is shielded from the method rename below.
        hold = "\x00BASELINE\x00"
        s = s.replace("Per-candidate + IFM", hold).replace(_BASELINE, hold)
        s = s.replace("GaitNet", "CADENCE")
        s = s.replace(hold, _BASELINE)
    return _orig_set_text(self, s)


_mt.Text.set_text = _set_text
