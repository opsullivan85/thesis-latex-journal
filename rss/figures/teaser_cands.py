"""Candidate rule of the paper, pure numpy (no rendering imports).

Copied verbatim from gaitnet-video/anim/gn_base.py (grow_mask, coverage_candidates,
_project_distinct, CoverageTrace), which was checked against the author's sampler
spec (distinct safe cells, pinned slot, determinism, mirror symmetry). Paper:
Sec. 4.1 (1 cm grid, unsafe grown by k = 2 cells, margin 2 cm) and App. E
(coverage sampler, K = 16). Grid: 37 x 37 cells, +-18 cm around the hip.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

GRID_H_M = 0.01       # m, cell size h (\corlScanResCm = 1)
GRID_REACH_M = 0.18   # m, grid half-span (\corlGridReachCm = 18): 37 x 37 cells
GROW_CELLS = 2        # k, unsafe region grown by 2 cells = candidate margin m = 2 cm
K_CANDIDATES = 16     # K per leg (\corlTrainCandidatesPerLeg)


def grow_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """Dilate a boolean mask by k cells in the inf-norm (a (2k+1) x (2k+1) square), the
    same norm as the clearance d(u*).  Cells outside the grid do not propagate."""
    out = mask.copy()
    n0, n1 = mask.shape
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            src = mask[max(0, -di):n0 - max(0, di), max(0, -dj):n1 - max(0, dj)]
            out[max(0, di):n0 - max(0, -di), max(0, dj):n1 - max(0, -dj)] |= src
    return out



NOMINAL_T_S = 0.02 + 0.18 + 0.09   # s, lift-off lead + swing + half stance (priors, not fitted)
ANNEAL_TAU0_M2 = 0.02 ** 2         # (2 cm)^2
ANNEAL_TAU_END_M2 = 0.005 ** 2     # (0.5 cm)^2
ANNEAL_STEPS = 4



@dataclass
class CoverageTrace:
    """Stages of coverage_candidates(), for animating the sampler.  All in cell units."""
    S: np.ndarray                 # (M, 2) int: the allowed (safe) cells, row-major order
    q_nom: np.ndarray             # (2,) float: nominal foothold
    fps: np.ndarray               # (K',) int rows of S: q0, then farthest-point order
    targets: list[np.ndarray]     # per annealing step, (K', 2) float centroid targets
    steps: list[np.ndarray]       # per annealing step, (K',) int rows of S after projection

    @property
    def final(self) -> np.ndarray:
        return self.steps[-1] if self.steps else self.fps


def _project_distinct(S: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Priority-ordered projection: slot i takes the S cell nearest its target among the
    cells not held by slots < i (lowest cell index on ties).  Slot 0 is the pinned q0."""
    held = np.zeros(len(S), bool)
    out = np.empty(len(targets), int)
    for i, t in enumerate(targets):
        d = ((S - t) ** 2).sum(1)
        d[held] = np.inf
        out[i] = int(np.argmin(d))            # argmin returns the first (lowest index) tie
        held[out[i]] = True
    return out


def coverage_candidates(safe: np.ndarray, q_nom_ab, K: int = 16, h: float = 0.01,
                        mirror: bool = False) -> CoverageTrace:
    """The author's coverage sampler on one leg's grid.  `safe` is the (n, n) allowed mask
    S; `q_nom_ab` the nominal foothold in grid coordinates.  With mirror=True (left legs)
    the problem is solved on M(a, b) = (a, n-1-b) and mapped back, so FL/RL results are
    the mirror images of FR/RR ones on mirrored inputs."""
    n = safe.shape[1]
    q_nom = np.asarray(q_nom_ab, float)
    if mirror:
        tr = coverage_candidates(safe[:, ::-1], (q_nom[0], n - 1 - q_nom[1]), K, h, False)
        M = lambda X: np.column_stack([X[:, 0], n - 1 - X[:, 1]]) if len(X) else X
        S_m = M(tr.S)
        order = np.lexsort((S_m[:, 1], S_m[:, 0]))          # back to row-major order
        rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
        return CoverageTrace(S_m[order], q_nom, rank[tr.fps], [M(t) for t in tr.targets],
                             [rank[st] for st in tr.steps])
    S = np.argwhere(safe)                                     # row-major, so index = priority
    if len(S) == 0:
        return CoverageTrace(S, q_nom, np.zeros(0, int), [], [])
    if len(S) <= K:                                           # Q = S; the other slots stay empty
        allr = np.arange(len(S))
        return CoverageTrace(S, q_nom, allr, [], [allr])
    Sf = S.astype(float)
    # 1. pinned slot: the allowed cell nearest the nominal foothold
    chosen = [int(np.argmin(((Sf - q_nom) ** 2).sum(1)))]
    # 2. farthest-point sampling: next = the cell farthest from all chosen so far
    mind = ((Sf - Sf[chosen[0]]) ** 2).sum(1)
    for _ in range(1, K):
        m = mind.copy(); m[chosen] = -1.0
        j = int(np.argmax(m)); chosen.append(j)
        mind = np.minimum(mind, ((Sf - Sf[j]) ** 2).sum(1))
    fps = np.array(chosen)
    # 3. deterministic annealing, tau_t = tau0 (tau_end / tau0)^(t / 3), in cell^2
    tau0, tau_end = ANNEAL_TAU0_M2 / h ** 2, ANNEAL_TAU_END_M2 / h ** 2
    cur = fps.copy()
    targets, steps = [], []
    for t in range(ANNEAL_STEPS):
        tau = tau0 * (tau_end / tau0) ** (t / (ANNEAL_STEPS - 1))
        Q = Sf[cur]
        D = ((Q[:, None, :] - Sf[None, :, :]) ** 2).sum(-1)  # (K, M)
        logit = -D / tau
        logit -= logit.max(0, keepdims=True)
        p = np.exp(logit); p /= p.sum(0, keepdims=True)      # soft assignment over slots i
        mass = p.sum(1)
        tgt = (p @ Sf) / np.maximum(mass, 1e-12)[:, None]
        tgt[mass < 1e-9] = Q[mass < 1e-9]                    # an empty slot stays put
        tgt[0] = Q[0]                                        # q0 is pinned
        tgt = np.round(tgt * 16) / 16                        # 1/16-cell rounding (exact distances)
        cur = _project_distinct(Sf, tgt)
        targets.append(tgt); steps.append(cur.copy())
    return CoverageTrace(S, q_nom, fps, targets, steps)


def leg_candidates(is_void, hip_xy, nominal_xy, mirror: bool, h: float = GRID_H_M,
                   reach: float = GRID_REACH_M, k: int = GROW_CELLS, K: int = K_CANDIDATES):
    """One leg's grid at one tick, heading +x. is_void(x, y) -> True over a void or
    outside the field. Returns (centers (n,n,2), unsafe_raw, unsafe, candidates (K',2) world xy,
    pinned index 0 = the candidate nearest the nominal foothold)."""
    hip_xy = np.asarray(hip_xy, float)[:2]
    half = int(round(reach / h))
    offs = np.arange(-half, half + 1) * h
    A, B = np.meshgrid(offs, offs, indexing="ij")
    centers = np.stack([hip_xy[0] + A, hip_xy[1] + B], axis=-1)
    unsafe_raw = np.array([[is_void(x, y) for x, y in row] for row in centers])
    unsafe = grow_mask(unsafe_raw, k)
    d = np.asarray(nominal_xy, float)[:2] - hip_xy
    nominal_ab = np.array([half + d[0] / h, half + d[1] / h])
    tr = coverage_candidates(~unsafe, nominal_ab, K=K, h=h, mirror=mirror)
    idx = tr.S[tr.final] if len(tr.S) else np.zeros((0, 2), int)
    cand = centers[idx[:, 0], idx[:, 1]] if len(idx) else np.zeros((0, 2))
    return centers, unsafe_raw, unsafe, cand
