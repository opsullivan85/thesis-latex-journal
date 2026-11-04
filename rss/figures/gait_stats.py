"""Gait timing statistics of the G125 benchmark from ../timming.md.

Source: rss/timming.md (added by the author on 2026-11-04). Despite the .md
extension it is a CSV with one row per recorded swing:

    arm, speed_mps, void_pct, world_seed, leg, liftoff_s, touchdown_s,
    swing_s, field_entry_s, counted

Arms: c0 = GaitNet (iteration-350 checkpoint, OCS2 IDM), trot = OCS2's own
trot. Cells: speed 0.10 / 0.20 m/s x void fraction 26 / 38 / 49 %, world seeds
111-210, 20 s trials. Times are seconds of simulated time from trial start.

What the file holds (checked in `sanity()`; author's answers of 2026-11-04
marked "author"):
  * ONLY SUCCESSFUL TRIALS (author). The trials per cell equal the success counts of
    the benchmark (c0: 100 93 96 | 96 87 77 = 549; trot: 97 89 80 | 94 81 63
    = 504), and every trial runs to about 20 s. Every statistic below is
    therefore conditioned on success, and the arms contribute different
    numbers of trials per cell.
  * `swing_s` is a MEASURED swing from simulation ground truth (author),
    `touchdown_s - liftoff_s`, not the commanded swing time T. The trot's measured swings are about 0.235 s
    against its scheduled 0.3 s, so the detector (presumably gait_metrics.py:
    a foot is "up" when 1.5 cm above its rolling 0.5 s minimum) sees only part
    of each swing. Absolute swing times and duty factors depend on it.
  * TIME RESOLUTION about 21 ms. Every liftoff and touchdown is a sample of
    the ground-truth log (documented as 50 Hz; the observed spacing is 20-34 ms
    of simulated time, median 21 ms, varying within a trial). Both arms show
    the same grid (the trot's swings cluster at 213 / 235 / 256 ms), so it is
    the logger, not GaitNet's 40 ms policy tick. Unchecked guess: a 50 Hz
    wall-clock logger on a simulation running slightly slower than real time.
    Lags and overlaps of one sample (about 21 ms) are at the resolution limit.
  * `counted` is True exactly when field_entry_s <= touchdown_s <= 20 s
    (3 rows with touchdown printed as 20.0 fall either way by rounding), i.e.
    the swing ended on the void field within the 20 s trial. 441 swings that
    land after 20 s are logged but not counted. `field_entry_s` is one value
    per trial: the time the REAR feet reach the void field (author), so the
    front feet are already on the field at the end of the run-up.

Definitions used here:
  field window   [field_entry_s, 20 s] of a trial (swings clipped at 20 s).
  run-up window  [first liftoff, field_entry_s] of a trial (before the rear
                 feet reach the void field; NOT flat ground throughout, since
                 the front feet enter first; includes the start-up
                 transient). Run-up
                 steps are swings with touchdown_s < field_entry_s.
  step           one recorded swing. Field steps are the counted swings.
  feet-up share  share of window time with 0, 1, 2, ... feet in measured swing,
                 on a 1 ms grid, pooled over trials (time-weighted).
  overlap pair   two swings of different legs whose intervals intersect with
                 positive length. Type by legs: diagonal (FL+RR, FR+RL),
                 side (FL+RL, FR+RR), front (FL+FR), rear (RL+RR).
                 Liftoff lag = later liftoff - earlier liftoff.
  solo swing     a swing that overlaps no other swing. Also reported with a
                 one-sample tolerance (another leg's swing within 22 ms).
  pause          an all-feet-down interval longer than 0.5 s.
  event          a connected group of overlapping swings (graph components).
  next leg       the leg of the next liftoff in the same trial.

usage: python gait_stats.py   (prints the report)
"""
from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# The data file (6.4 MB) lives only in thesis-latex-journal/rss and is not part
# of the paper repo; elsewhere point GAIT_DATA at it.
DATA = Path(os.environ.get("GAIT_DATA", HERE.parent / "timming.md"))

LEGS = ("FL", "FR", "RL", "RR")
DIAG_ORDER = ("FL", "RR", "FR", "RL")      # diagonal partners adjacent
ARMS = ("c0", "trot")
ARM_NAME = {"c0": "GaitNet", "trot": "OCS2's own trot"}
SPEEDS = (0.1, 0.2)
VOIDS = (26, 38, 49)
TRIAL = ["arm", "speed_mps", "void_pct", "world_seed"]
DT = 0.001                                  # time grid for feet-up shares [s]
T_TRIAL = 20.0                              # trial duration [s]

PAIR_TYPE = {
    frozenset(("FL", "RR")): "diagonal", frozenset(("FR", "RL")): "diagonal",
    frozenset(("FL", "RL")): "side", frozenset(("FR", "RR")): "side",
    frozenset(("FL", "FR")): "front", frozenset(("RL", "RR")): "rear",
}
PAIR_TYPES = ("diagonal", "side", "front", "rear")
DIAG_PARTNER = {"FL": "RR", "RR": "FL", "FR": "RL", "RL": "FR"}
DIAG_PAIR = {"FL": "A", "RR": "A", "FR": "B", "RL": "B"}   # A = FL+RR, B = FR+RL

# Benchmark success counts per cell (fig_bench.py, constants.tex). Order:
# 0.10 m/s at 26/38/49 %, then 0.20 m/s.
SUCCESS = {"c0": (100, 93, 96, 96, 87, 77), "trot": (97, 89, 80, 94, 81, 63)}


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["counted"] = df["counted"].astype(bool)
    return df.sort_values(TRIAL + ["liftoff_s", "leg"]).reset_index(drop=True)


# ---------------------------------------------------------------- checks
def sanity(df: pd.DataFrame) -> dict:
    """Data checks; asserts the structural ones, returns counts of the rest."""
    out = {}
    assert set(df.arm) == set(ARMS) and set(df.leg) == set(LEGS)
    assert np.allclose(df.touchdown_s - df.liftoff_s, df.swing_s, atol=1e-9)
    assert (df.groupby(TRIAL).field_entry_s.nunique() == 1).all()
    rule = (df.touchdown_s >= df.field_entry_s) & (df.touchdown_s <= T_TRIAL)
    off = df.counted != rule
    assert (abs(df.touchdown_s[off] - T_TRIAL) < 1e-9).all(), "counted rule"
    out["counted_rule_mismatch_at_20s"] = int(off.sum())
    # Same leg never swings twice at once.
    g = df.sort_values(TRIAL + ["leg", "liftoff_s"]).groupby(TRIAL + ["leg"])
    stance_before = (df.sort_values(TRIAL + ["leg", "liftoff_s"]).liftoff_s
                     - g.touchdown_s.shift())
    assert (stance_before.dropna() > 0).all(), "overlapping swings on one leg"
    # Trials per cell = benchmark success counts (=> successful trials only).
    n = df.groupby(["arm", "speed_mps", "void_pct"]).world_seed.nunique()
    for arm in ARMS:
        got = tuple(int(n[(arm, v, p)]) for v in SPEEDS for p in VOIDS)
        assert got == SUCCESS[arm], f"{arm}: trials per cell {got} != {SUCCESS[arm]}"
    out["trials"] = {a: int(df[df.arm == a].groupby(TRIAL).ngroups) for a in ARMS}
    t_end = df.groupby(TRIAL).touchdown_s.max()
    out["t_end_min"] = {a: float(t_end.loc[a].min()) for a in ARMS}
    out["swings"] = {a: int((df.arm == a).sum()) for a in ARMS}
    out["field_swings"] = {a: int(((df.arm == a) & df.counted).sum()) for a in ARMS}
    for a in ARMS:
        s = df[(df.arm == a) & df.counted].swing_s
        out[f"{a}_short_lt_50ms"] = int((s < 0.05).sum())
        out[f"{a}_long_gt_300ms"] = int((s > 0.30).sum())
        out[f"{a}_long_gt_400ms"] = int((s > 0.40).sum())
    return out


# ---------------------------------------------------------------- per trial
@dataclass
class TrialStats:
    key: tuple
    field_time: float = 0.0
    runup_time: float = 0.0
    feet_up_field: np.ndarray = field(default_factory=lambda: np.zeros(5))   # seconds
    feet_up_runup: np.ndarray = field(default_factory=lambda: np.zeros(5))
    pair_time_field: dict = field(default_factory=lambda: dict.fromkeys(PAIR_TYPES, 0.0))
    leg_swing_time_field: dict = field(default_factory=lambda: dict.fromkeys(LEGS, 0.0))
    pairs: list = field(default_factory=list)       # overlap pairs on the field
    n_field_steps: int = 0
    n_runup_steps: int = 0
    n_solo: int = 0
    n_solo_tol: int = 0
    runup_nondiag: int = 0
    pause_time: float = 0.0
    events: list = field(default_factory=list)      # (t_liftoff, kind) on the field
    next_leg: list = field(default_factory=list)    # (leg, next leg) on the field


def _feet_up(mask: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    w = mask[:, i0:i1]
    n = w.sum(0)
    return np.bincount(n, minlength=5)[:5] * DT, w


def trial_stats(key: tuple, tr: pd.DataFrame) -> TrialStats:
    ts = TrialStats(key)
    tr = tr.sort_values("liftoff_s")
    lo, td, lg = tr.liftoff_s.to_numpy(), tr.touchdown_s.to_numpy(), tr.leg.to_numpy()
    counted = tr.counted.to_numpy()
    fe, t_end, t_first = tr.field_entry_s.iloc[0], T_TRIAL, lo.min()

    # Feet-up time on a 1 ms grid; swing [lo, td) occupies cells round(lo/DT)..round(td/DT)-1.
    n_cells = int(round(max(td.max(), t_end) / DT)) + 1
    mask = np.zeros((4, n_cells), bool)
    for l, a, b in zip(lg, lo, td):
        mask[LEGS.index(l), int(round(a / DT)):int(round(b / DT))] = True
    i_fe, i_end, i_first = int(round(fe / DT)), int(round(t_end / DT)), int(round(t_first / DT))
    ts.field_time, ts.runup_time = (i_end - i_fe) * DT, max(i_fe - i_first, 0) * DT
    ts.feet_up_field, w = _feet_up(mask, i_fe, i_end)
    if i_fe > i_first:
        ts.feet_up_runup, _ = _feet_up(mask, i_first, i_fe)
    two = w.sum(0) == 2
    for a, b in itertools.combinations(range(4), 2):
        ts.pair_time_field[PAIR_TYPE[frozenset((LEGS[a], LEGS[b]))]] += (w[a] & w[b] & two).sum() * DT
    for i, l in enumerate(LEGS):
        ts.leg_swing_time_field[l] = w[i].sum() * DT

    # Overlap graph on field swings (counted). Sorted by liftoff, so swing j > i
    # overlaps i iff lo[j] < td[i].
    idx = np.flatnonzero(counted)
    # Run-up steps land before field entry. (Not ~counted: that also holds
    # the swings that land after 20 s.)
    ts.n_field_steps, ts.n_runup_steps = len(idx), int((td < fe).sum())
    ru = np.flatnonzero(td < fe)
    ts.runup_nondiag = sum(1 for p_, i in enumerate(ru) for j in ru[p_ + 1:]
                           if lo[j] < td[i] and lg[i] != lg[j]
                           and PAIR_TYPE[frozenset((lg[i], lg[j]))] != "diagonal")
    # Solo with a one-sample (22 ms) tolerance: no other leg's counted swing
    # within 22 ms of this one.
    ts.n_solo_tol = sum(1 for i in idx if not any(
        lg[j] != lg[i] and lo[j] < td[i] + 0.022 and lo[i] < td[j] + 0.022 for j in idx if j != i))
    # Pauses: all-feet-down runs longer than 0.5 s inside the field window.
    down = (w.sum(0) == 0).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], down, [0]))))
    runs = (edges[1::2] - edges[::2]) * DT
    ts.pause_time = float(runs[runs > 0.5].sum())
    nbr = {i: [] for i in idx}
    for p, i in enumerate(idx):
        for j in idx[p + 1:]:
            if lo[j] >= td[i]:
                break
            nbr[i].append(j)
            nbr[j].append(i)
            ts.pairs.append(dict(
                type=PAIR_TYPE[frozenset((lg[i], lg[j]))], first=lg[i], second=lg[j],
                lag=lo[j] - lo[i], td_lag=td[j] - td[i],
                overlap=min(td[i], td[j]) - lo[j],
                overlap_frac=(min(td[i], td[j]) - lo[j]) / min(td[i] - lo[i], td[j] - lo[j])))
    ts.n_solo = sum(1 for i in idx if not nbr[i])

    # Events = connected components; kind = 'single', 'A' / 'B' (diagonal
    # pairs FL+RR / FR+RL), 'other-pair', or 'chain' (3+ swings).
    seen = set()
    for i in idx:
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack.extend(nbr[k])
        legs = sorted(lg[k] for k in comp)
        if len(comp) == 1:
            kind = "single"
        elif len(comp) == 2 and PAIR_TYPE[frozenset(legs)] == "diagonal":
            kind = DIAG_PAIR[legs[0]]
        elif len(comp) == 2:
            kind = "other-pair"
        else:
            kind = "chain"
        ts.events.append((min(lo[k] for k in comp), kind, tuple(legs)))
    ts.events.sort()
    f = [k for k in range(len(lo)) if counted[k]]
    ts.next_leg = [(lg[a], lg[b]) for a, b in zip(f, f[1:])]
    return ts


def all_trials(df: pd.DataFrame) -> list[TrialStats]:
    return [trial_stats(k, tr) for k, tr in df.groupby(TRIAL, sort=True)]


# ---------------------------------------------------------------- aggregation
def summarize(trials: list[TrialStats], df: pd.DataFrame, sel=lambda k: True) -> dict:
    """Pooled statistics over the trials whose key passes `sel`."""
    T = [t for t in trials if sel(t.key)]
    keys = {t.key for t in T}
    sub = df[df.set_index(TRIAL).index.isin(keys)]
    fsub = sub[sub.counted]
    ft = sum(t.field_time for t in T)
    rt = sum(t.runup_time for t in T)
    fu = sum(t.feet_up_field for t in T)
    ru = sum(t.feet_up_runup for t in T)
    two = fu[2]
    pt = {p: sum(t.pair_time_field[p] for t in T) for p in PAIR_TYPES}
    pairs = pd.DataFrame([p for t in T for p in t.pairs])
    nsteps = sum(t.n_field_steps for t in T)
    out = dict(
        trials=len(T), field_time_s=ft,
        steps_per_s=nsteps / ft,
        runup_steps_per_s=sum(t.n_runup_steps for t in T) / rt if rt else float("nan"),
        feet_up=(fu / ft).tolist(), runup_feet_up=(ru / rt).tolist() if rt else None,
        pair_time_share_of_two={p: pt[p] / two if two else float("nan") for p in PAIR_TYPES},
        duty={l: 1 - sum(t.leg_swing_time_field[l] for t in T) / ft for l in LEGS},
        swing_median=float(fsub.swing_s.median()),
        swing_q=(float(fsub.swing_s.quantile(.25)), float(fsub.swing_s.quantile(.75))),
        runup_swing_median=float(sub[sub.touchdown_s < sub.field_entry_s].swing_s.median()),
        runup_nondiag_per_s=sum(t.runup_nondiag for t in T) / rt if rt else float("nan"),
        pause_share=sum(t.pause_time for t in T) / ft,
        steps=nsteps, solo=sum(t.n_solo for t in T),
    )
    out["solo_share"] = out["solo"] / nsteps
    out["solo_share_tol"] = sum(t.n_solo_tol for t in T) / nsteps
    out["pair_counts"] = ({p: int((pairs.type == p).sum()) for p in PAIR_TYPES}
                          if len(pairs) else dict.fromkeys(PAIR_TYPES, 0))
    npairs = sum(out["pair_counts"].values())
    out["nondiag_share"] = 1 - out["pair_counts"]["diagonal"] / npairs if npairs else float("nan")
    out["nondiag_per_trial"] = (npairs - out["pair_counts"]["diagonal"]) / len(T)
    out["nondiag_per_s"] = (npairs - out["pair_counts"]["diagonal"]) / ft
    out["trials_with_nondiag"] = len({t.key for t in T if any(p["type"] != "diagonal" for p in t.pairs)})
    if len(pairs):
        d = pairs[pairs.type == "diagonal"]
        out["diag_lag_ms_q"] = (d.lag * 1e3).quantile([.1, .25, .5, .75, .9]).round(1).tolist()
        out["diag_simultaneous_share"] = float((d.lag < 0.005).mean())   # < 5 ms
        out["diag_td_lag_ms_median"] = float((d.td_lag.abs() * 1e3).median())
        # Signed: later lifter's touchdown minus earlier lifter's (0 = land together).
        out["diag_td_lag_signed_ms_q"] = (d.td_lag * 1e3).quantile([.25, .5, .75]).round(1).tolist()
        out["diag_td_within_5ms"] = float((d.td_lag.abs() < 0.005).mean())
        stag = d[d.lag >= 0.005]
        out["diag_lead_leg"] = {l: float((stag["first"] == l).mean()) for l in LEGS}
        out["diag_front_first"] = float(d["first"].isin(["FL", "FR"]).mean())
        out["diag_nested_share"] = float((d.overlap_frac > 0.999).mean())
        nd = pairs[pairs.type != "diagonal"]
        out["nondiag_overlap_ms_median"] = float(nd.overlap.median() * 1e3) if len(nd) else float("nan")
    # Stride: same-leg liftoff-to-liftoff on the field.
    s = fsub.sort_values(TRIAL + ["leg", "liftoff_s"])
    stride = s.groupby(TRIAL + ["leg"]).liftoff_s.diff().dropna()
    out["stride_median"] = float(stride.median())
    out["stride_q"] = (float(stride.quantile(.25)), float(stride.quantile(.75)))
    out["stride_cv"] = float(stride.std() / stride.mean())
    out["stride_q10_q90"] = (float(stride.quantile(.1)), float(stride.quantile(.9)))
    out["stride_lt_0p3"] = float((stride < 0.3).mean())     # quick same-leg re-steps
    out["stride_gt_1p5"] = float((stride > 1.5).mean())     # pauses
    out["steps_by_leg"] = {l: int((fsub.leg == l).sum()) for l in LEGS}
    # Next-leg transitions.
    tr = collections.Counter(p for t in T for p in t.next_leg)
    out["next_leg"] = {a: {b: tr[(a, b)] for b in DIAG_ORDER} for a in DIAG_ORDER}
    out["next_is_partner"] = sum(tr[(a, DIAG_PARTNER[a])] for a in LEGS) / max(sum(tr.values()), 1)
    out["next_is_same"] = sum(tr[(a, a)] for a in LEGS) / max(sum(tr.values()), 1)
    # Events and pair alternation.
    ev = collections.Counter(k for t in T for _, k, _ in t.events)
    out["events"] = dict(ev)
    alt = same = 0
    for t in T:
        seq = [k for _, k, _ in t.events if k in ("A", "B")]
        for a, b in zip(seq, seq[1:]):
            alt += a != b
            same += a == b
    out["pair_alternation"] = alt / (alt + same) if alt + same else float("nan")
    # Single steps: which legs, and what the event before/after was.
    singles = collections.Counter(legs[0] for t in T for _, k, legs in t.events if k == "single")
    out["single_legs"] = dict(singles)
    return out


def cell_sel(arm, v=None, p=None, runup=False):
    return lambda k: k[0] == arm and (v is None or k[1] == v) and (p is None or k[2] == p)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def report(df=None, trials=None):
    df = load() if df is None else df
    chk = sanity(df)
    trials = all_trials(df) if trials is None else trials
    print("== data checks")
    for k, v in chk.items():
        print(f"  {k}: {v}")
    # Trot schedule check.
    s = df.sort_values(TRIAL + ["leg", "liftoff_s"])
    ili = s.groupby(TRIAL + ["leg"]).liftoff_s.diff()
    for a in ARMS:
        x = ili[(s.arm == a) & s.counted].dropna()
        print(f"  {a}: same-leg liftoff interval median {x.median():.3f} s, IQR "
              f"{x.quantile(.25):.3f}-{x.quantile(.75):.3f}")
    S = {a: summarize(trials, df, cell_sel(a)) for a in ARMS}
    print("\n== pooled over all cells, void field only")
    for a in ARMS:
        r = S[a]
        print(f"-- {ARM_NAME[a]} ({r['trials']} successful trials, {r['field_time_s']:.0f} s on the field)")
        print(f"  steps/s {r['steps_per_s']:.2f} (run-up {r['runup_steps_per_s']:.2f})")
        print(f"  feet up 0/1/2/3/4: {[pct(x) for x in r['feet_up']]} %"
              f"   run-up: {[pct(x) for x in r['runup_feet_up']]} %")
        print(f"  two-feet-up time by pair: { {k: pct(v, 2) for k, v in r['pair_time_share_of_two'].items()} } %")
        print(f"  overlap pairs: {r['pair_counts']}  non-diagonal {pct(r['nondiag_share'], 2)} % "
              f"({r['nondiag_per_trial']:.2f}/trial, {r['trials_with_nondiag']} trials with >= 1)")
        print(f"  solo swings {pct(r['solo_share'])} % of {r['steps']} field steps; events {r['events']}")
        print(f"  diagonal liftoff lag ms q10/25/50/75/90: {r.get('diag_lag_ms_q')}, "
              f"simultaneous (<5 ms) {pct(r.get('diag_simultaneous_share', np.nan))} %, "
              f"signed touchdown lag q25/50/75 {r.get('diag_td_lag_signed_ms_q')} ms "
              f"(within 5 ms {pct(r.get('diag_td_within_5ms', np.nan))} %), "
              f"front leg first {pct(r.get('diag_front_first', np.nan))} %, "
              f"nested {pct(r.get('diag_nested_share', np.nan))} %")
        print(f"  non-diagonal overlap median {r.get('nondiag_overlap_ms_median', np.nan):.0f} ms")
        print(f"  measured swing median {r['swing_median']*1e3:.0f} ms "
              f"(IQR {r['swing_q'][0]*1e3:.0f}-{r['swing_q'][1]*1e3:.0f}; run-up {r['runup_swing_median']*1e3:.0f})")
        print(f"  stride median {r['stride_median']:.3f} s (IQR {r['stride_q'][0]:.3f}-{r['stride_q'][1]:.3f}, CV {r['stride_cv']:.2f})")
        print(f"  duty factor (measured) { {k: round(v, 3) for k, v in r['duty'].items()} }")
        print(f"  next leg is diagonal partner {pct(r['next_is_partner'])} %, same leg {pct(r['next_is_same'])} %; "
              f"diagonal pair events alternate A/B {pct(r['pair_alternation'])} %")
        print(f"  single steps by leg {r['single_legs']}")
        print("  next-leg counts (row = this liftoff, col = next):")
        print("        " + " ".join(f"{b:>6}" for b in DIAG_ORDER))
        for x in DIAG_ORDER:
            tot = sum(r["next_leg"][x].values())
            print(f"    {x}  " + " ".join(f"{100*r['next_leg'][x][b]/tot:5.1f}%" for b in DIAG_ORDER))
    print("\n== per cell (void field), GaitNet | trot")
    hdr = ("trials", "steps/s", "0 up %", "1 up %", "2 up %", "nondiag pairs %", "nondiag/trial",
           "solo %", "swing ms", "stride s")
    print("  cell        " + " | ".join(hdr))
    cells = {}
    for v in SPEEDS:
        for p in VOIDS:
            for a in ARMS:
                r = summarize(trials, df, cell_sel(a, v, p))
                cells[(a, v, p)] = r
                print(f"  {a:4} {v:.2f} {p}%  {r['trials']:4d} | {r['steps_per_s']:.2f} | "
                      f"{pct(r['feet_up'][0])} | {pct(r['feet_up'][1])} | {pct(r['feet_up'][2])} | "
                      f"{pct(r['nondiag_share'], 2)} | {r['nondiag_per_trial']:.2f} | {pct(r['solo_share'])} | "
                      f"{r['swing_median']*1e3:.0f} | {r['stride_median']:.3f}")
    print("\n== per speed (void field unless 'run-up')")
    for a in ARMS:
        for v in SPEEDS:
            r = summarize(trials, df, cell_sel(a, v))
            print(f"-- {ARM_NAME[a]} {v:.2f} m/s ({r['trials']} trials)")
            print(f"  steps/s field {r['steps_per_s']:.2f} | run-up {r['runup_steps_per_s']:.2f}")
            print(f"  feet up 0/1/2 field {[pct(x) for x in r['feet_up'][:3]]} % | run-up "
                  f"{[pct(x) for x in r['runup_feet_up'][:3]]} %")
            print(f"  non-diagonal overlaps per s: field {r['nondiag_per_s']:.3f} | run-up {r['runup_nondiag_per_s']:.3f}"
                  f"; share of pairs {pct(r['nondiag_share'], 2)} %")
            print(f"  pauses (> 0.5 s all down) {pct(r['pause_share'])} % of field time "
                  f"(all down {pct(r['feet_up'][0])} %)")
            print(f"  solo {pct(r['solo_share'])} % (one-sample tolerance {pct(r['solo_share_tol'])} %), "
                  f"pair alternation {pct(r['pair_alternation'])} %")
            print(f"  stride median {r['stride_median']:.3f} s, q10-q90 {r['stride_q10_q90'][0]:.3f}-"
                  f"{r['stride_q10_q90'][1]:.3f}, CV {r['stride_cv']:.2f}, <0.3 s {pct(r['stride_lt_0p3'])} %, "
                  f">1.5 s {pct(r['stride_gt_1p5'])} %")
            print(f"  duty {{{', '.join(f'{k} {x:.3f}' for k, x in r['duty'].items())}}}; steps by leg {r['steps_by_leg']}")
            print(f"  diagonal lag q10..q90 {r.get('diag_lag_ms_q')} ms; signed touchdown lag q25/50/75 "
                  f"{r.get('diag_td_lag_signed_ms_q')} ms; lead leg (staggered pairs) "
                  f"{ {k: pct(x) for k, x in r.get('diag_lead_leg', {}).items()} } %")
    return S, cells, chk


if __name__ == "__main__":
    report()
