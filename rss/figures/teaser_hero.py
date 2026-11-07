"""Hero scene of the teaser: one GaitNet decision on void terrain, POSED (not logged).

The terrain, the pose and the choice of the stepping leg and foothold are posed.
The safe cells and the 16 candidates are exact for this terrain under the
paper's rule (Sec. 4.1, App. E). World frame: x heading, y left, z up, metres;
tile tops at z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import teaser_cands as C
import teaser_scene as S

# --- terrain (12.5 cm tiles, 0.5 m deep voids) ------------------------------------
N_X, N_Y, X0, Y0 = 20, 16, -1.1, -0.8125
FIXED_SOLID = {(9, 5), (10, 7), (7, 5), (6, 7), (7, 7), (11, 5), (10, 4), (10, 6), (11, 4),
               (9, 4), (9, 6), (9, 7), (11, 7), (8, 5), (8, 6), (8, 7), (6, 5), (6, 6), (7, 6),
               (8, 4), (11, 6)}
FIXED_VOID = {(10, 5), (12, 5)}          # (10, 5): the void ahead of FR's nominal foothold
BAND = {(i, j) for i in range(4, 14) for j in (8, 9, 10)}   # kept solid behind the trunk


def void_layout() -> set:
    """Seeded 40 % draw (illustrative, not a benchmark world), then hand edits near FR."""
    draw = np.random.default_rng(42).random((N_X, 13)) < 0.40
    voids = {(i, j) for i in range(N_X) for j in range(13) if draw[i, j]}
    ext = np.random.default_rng(43).random((N_X, 3)) < 0.40
    voids |= {(i, 13 + j) for i in range(N_X) for j in range(3) if ext[i, j]}
    voids -= FIXED_SOLID
    voids |= FIXED_VOID
    voids -= BAND
    return voids


def tile_of(x: float, y: float) -> tuple[int, int]:
    return int(np.floor((x - X0) / S.TILE)), int(np.floor((y - Y0) / S.TILE))


# --- pose -------------------------------------------------------------------------
BASE = (0.0, 0.0, 0.285)
R = S.FOOT_R
FEET = {"FR": (0.12, -0.127, R),     # end of stance: the leg GaitNet starts at this tick
        "FL": (0.24, 0.127, R), "RR": (-0.14, -0.127, R), "RL": (-0.19, 0.127, R)}
V_CMD = 0.15                         # m/s, illustrative command for the nominal foothold


def zoom_cam(target, az, el, dist, fovy):
    target = np.asarray(target, float)
    f = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    return (target - dist * f, target, fovy)


@dataclass
class Hero:
    rgb: np.ndarray            # (H, W, 3) uint8, MSAA render
    gid: np.ndarray            # (H, W) geom id per pixel (no MSAA)
    robot: np.ndarray          # (H, W) bool
    cap: np.ndarray            # (H, W) bool: tile tops
    safe_mask: np.ndarray      # (H, W) bool: FR's safe cells, on visible tile tops
    cand_uv: np.ndarray        # (16, 2) px
    cand_xy: np.ndarray        # (16, 2) world
    arc_uv: np.ndarray         # (n, 2) px, swing path
    foot_uv: np.ndarray        # (2,) px, FR foot
    info: dict
    zoom: dict | None = None   # close-up of u*: rgb, safe_mask, ring(r), u_px, cand_px, arc_px


def render_hero(width: int, height: int, look=(0.07, 0.0, 0.13), az: float = 98.0,
                el: float = -36.0, dist: float = 2.4, fovy: float = 14.5,
                apex: float = 0.10, zoom_px=None, zoom_center=(0.27, -0.12, 0.0),
                zoom_el: float = -62.0, zoom_dist: float = 2.4, zoom_fovy: float = 4.6,
                base=BASE, feet: dict | None = None, swing: tuple = ()) -> Hero:
    """Render the posed scene. base: trunk position (world, m); feet: foot-site
    (sphere-centre) targets per leg, default FEET (the decision tick); swing: legs in
    the air, exempt from the on-a-solid-tile check. The candidates are always FR's at
    the decision tick (hip of the default pose), so every frame shares u*."""
    feet = FEET if feet is None else feet
    # close-up size in px: an int (square) or (width, height)
    zw, zh = (None, None) if zoom_px is None else (
        (zoom_px, zoom_px) if isinstance(zoom_px, (int, np.integer)) else tuple(zoom_px))
    voids = void_layout()

    def is_void(x, y):
        i, j = tile_of(x, y)
        return (i, j) in voids or not (0 <= i < N_X and 0 <= j < N_Y)

    look = np.asarray(look, float)
    f = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    cam = look - dist * f
    hx = lambda s: [int(s[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    fb = max(width, height, zw or 0, zh or 0)          # offscreen framebuffer must fit both views
    m = S.build(voids, N_X, N_Y, X0, Y0, cam, look, fovy=fovy, width=max(width, fb), height=max(height, fb),
                top_rgb=hx("#E9E6DF"), column_rgb=hx("#2A2E35"), pit_rgb=hx("#14161A"),
                light_dir=(0.1, 0.8, -1.0),
                zoom=None if zoom_px is None else zoom_cam(zoom_center, az, zoom_el, zoom_dist, zoom_fovy))
    d = mujoco.MjData(m)
    # FR's candidates come from the decision-tick pose (default BASE and FEET)
    S.pose(m, d, BASE, feet={k: np.array(v) for k, v in FEET.items()})
    hip = S.body_pos(m, d, "FR_hip")[:2]
    errs = S.pose(m, d, base, feet={k: np.array(v) for k, v in feet.items()})
    assert max(errs.values()) < 1e-3, errs
    for k in range(1, 13):                                   # joints within range
        lo, hi = m.jnt_range[k]
        assert lo - 1e-9 <= d.qpos[6 + k] <= hi + 1e-9, (k, d.qpos[6 + k])
    for leg, p in feet.items():                              # stance feet on solid tiles
        assert leg in swing or tile_of(p[0], p[1]) not in voids, leg

    q_nom = hip + np.array([C.NOMINAL_T_S * V_CMD, FEET["FR"][1] - hip[1]])
    centers, uraw, uns, cand = C.leg_candidates(is_void, hip, q_nom, mirror=False)
    assert len(cand) == 16

    rgb = S.render(m, d, width, height)
    gid = S.segment(m, d, width, height)
    body = np.where(gid >= 0, m.geom_bodyid[np.clip(gid, 0, m.ngeom - 1)], -1)
    robot = body > 0
    cap = np.isin(gid, S.cap_ids(m))

    # safe cells: union of the projected 1 cm squares of FR's safe cells, on visible caps
    h = C.GRID_H_M
    sm = Image.new("L", (width, height), 0)
    dr = ImageDraw.Draw(sm)
    for a, b in np.argwhere(~uns):
        cx, cy = centers[a, b]
        sq = np.array([[cx - h / 2, cy - h / 2, 0], [cx + h / 2, cy - h / 2, 0],
                       [cx + h / 2, cy + h / 2, 0], [cx - h / 2, cy + h / 2, 0]])
        dr.polygon([tuple(p) for p in S.project(m, d, sq, width, height)], fill=255)
    safe_mask = (np.asarray(sm) > 0) & cap & ~robot

    # swing path (schematic shape, exact endpoints): foot -> chosen candidate (#0)
    p0 = np.array(FEET["FR"]); p2 = np.array([*cand[0], R])
    s = np.linspace(0, 1, 120)[:, None]
    arc = p0 + s * (p2 - p0)
    arc[:, 2] = p0[2] + (p2[2] - p0[2]) * s[:, 0] + 4 * apex * s[:, 0] * (1 - s[:, 0])
    # the drawn path starts at the foot's top and ends at the tile top (sphere bottom)
    arc[:, 2] -= R * s[:, 0]
    info = dict(hip=hip, q_nom=q_nom, u_star=cand[0], safe=int((~uns).sum()),
                void_cells=int(uraw.sum()), grown=int((uns & ~uraw).sum()),
                step=float(np.linalg.norm(cand[0] - np.array(FEET["FR"][:2]))),
                clearance_u=float(abs(cand[0][0] - (X0 + 11 * S.TILE))))   # to void (10, 5)'s far edge
    zoom = None
    if zoom_px is not None:
        # zoom on u*: same azimuth as the hero, steeper; robot hidden, no shadows
        m.light_castshadow[:] = 0
        trgb = S.render(m, d, zw, zh, camera="zoom", hide_robot=True)
        tgid = S.segment(m, d, zw, zh, camera="zoom")
        tcap = np.isin(tgid, S.cap_ids(m))
        tm = Image.new("L", (zw, zh), 0)
        tdr = ImageDraw.Draw(tm)
        for a, b in np.argwhere(~uns):
            cx, cy = centers[a, b]
            sq = np.array([[cx - h / 2, cy - h / 2, 0], [cx + h / 2, cy - h / 2, 0],
                           [cx + h / 2, cy + h / 2, 0], [cx - h / 2, cy + h / 2, 0]])
            tdr.polygon([tuple(p) for p in S.project(m, d, sq, zw, zh, "zoom")], fill=255)
        ring = lambda r: S.project(m, d, np.c_[cand[0][0] + r * np.cos(np.linspace(0, 2 * np.pi, 361)),
                                               cand[0][1] + r * np.sin(np.linspace(0, 2 * np.pi, 361)),
                                               np.zeros(361)], zw, zh, "zoom")
        zoom = dict(rgb=trgb, safe_mask=(np.asarray(tm) > 0) & tcap, ring=ring,
                   u_px=S.project(m, d, np.array([[*cand[0], 0.0]]), zw, zh, "zoom")[0],
                   cand_px=S.project(m, d, np.c_[cand, np.zeros(len(cand))], zw, zh, "zoom"),
                   arc_px=S.project(m, d, arc, zw, zh, "zoom"),
                   void_edge_px=S.project(m, d, np.array([[0.275, cand[0][1], 0.0]]), zw, zh, "zoom")[0])
    return Hero(rgb, gid, robot, cap, safe_mask,
                S.project(m, d, np.c_[cand, np.zeros(len(cand))], width, height), cand,
                S.project(m, d, arc, width, height),
                S.project(m, d, np.array([feet["FR"]]), width, height)[0], info, zoom)
