"""MuJoCo scene for the GaitNet teaser (fig_teaser.py): the Go1 on the paper's void terrain.

World frame: x forward (walking direction), y left, z up; tile tops at z = 0.
Tiles are 12.5 cm squares (\\corlBenchTileCm); a void is a missing tile over a
0.5 m deep pit (\\corlVoidDepth). Each tile is drawn as a 1 mm light cap on a dark
column, so voids read as dark holes and tile tops as light squares.
Go1 model: go1_mjx.xml from the skadi/TerraStride scene (not in this repository).

Go1 qpos layout (go1_mjx.xml): [x y z, qw qx qy qz, FR(hip thigh calf),
FL(...), RR(...), RL(...)]; quaternion order is MuJoCo's (w, x, y, z).
"""
from __future__ import annotations

import numpy as np
import mujoco

GO1_SCENE = "/home/sourav/dev/repos/skadi/TerraStride/scene/scene_mjx_flat_terrain.xml"
TILE = 0.125
DEPTH = 0.5
LEGS = ("FR", "FL", "RR", "RL")          # qpos order
FOOT_R = 0.023                            # foot sphere radius [m] (go1_mjx.xml class "foot")


def look_at_quat(pos, target) -> np.ndarray:
    """Quaternion (w, x, y, z) of a MuJoCo camera at pos looking at target, z-up.
    MuJoCo cameras look along their local -z, with local +y up and +x right."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.column_stack([right, up, -fwd])          # camera axes in world
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def build(voids: set, n_x: int, n_y: int, x0: float, y0: float, cam_pos, cam_target,
          fovy: float = 30.0, width: int = 3840, height: int = 2160,
          top_rgb=(0.80, 0.79, 0.76), column_rgb=(0.30, 0.31, 0.34),
          pit_rgb=(0.10, 0.11, 0.13), gap: float = 0.004,
          light_dir=(-0.30, 0.45, -1.0), cap_t: float = 0.001, key_diffuse: float = 0.45,
          key_specular: float = 0.08, head_ambient: float = 0.35, head_diffuse: float = 0.20,
          offsamples: int = 8, zoom=None) -> mujoco.MjModel:
    """Compile the scene. voids: set of (i, j) tile indices (i along x, j along y)
    with tile (i, j) covering [x0 + i*TILE, x0 + (i+1)*TILE] x [y0 + j*TILE, ...]."""
    spec = mujoco.MjSpec.from_file(GO1_SCENE)
    # the scene's checker floor becomes the pit floor
    floor = [g for g in spec.geoms if g.name == "floor"][0]
    floor.pos = [0, 0, -DEPTH]
    floor.material = ""
    floor.rgba = [*pit_rgb, 1]

    top_t = cap_t                      # thin caps: a thick cap's light side fills the seams
    for i in range(n_x):
        for j in range(n_y):
            if (i, j) in voids:
                continue
            cx, cy = x0 + (i + 0.5) * TILE, y0 + (j + 0.5) * TILE
            hs = TILE / 2 - gap / 2
            top = spec.worldbody.add_geom()
            top.name = f"cap_{i}_{j}"
            top.type = mujoco.mjtGeom.mjGEOM_BOX
            top.size = [hs, hs, top_t / 2]
            top.pos = [cx, cy, -top_t / 2]
            top.rgba = [*top_rgb, 1]
            top.contype = 0; top.conaffinity = 0
            col = spec.worldbody.add_geom()
            col.type = mujoco.mjtGeom.mjGEOM_BOX
            col.size = [hs, hs, (DEPTH - top_t) / 2]
            col.pos = [cx, cy, -top_t - (DEPTH - top_t) / 2]
            col.rgba = [*column_rgb, 1]
            col.contype = 0; col.conaffinity = 0

    # lighting: a soft key light from front-left-above with shadows, a dim fill
    for l in list(spec.lights):
        spec.delete(l)
    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.dir = np.asarray(light_dir, float) / np.linalg.norm(light_dir)   # travel direction
    key.pos = [1.5, 2.0, 4.0]
    key.diffuse = [key_diffuse] * 3
    key.specular = [key_specular] * 3
    key.ambient = [0.0, 0.0, 0.0]
    key.castshadow = True
    spec.visual.headlight.ambient = [head_ambient] * 3
    spec.visual.headlight.diffuse = [head_diffuse] * 3
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 8192
    spec.visual.quality.offsamples = offsamples
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.map.znear = 0.01
    spec.visual.map.zfar = 50
    spec.visual.global_.ipd = 0.0

    # Robot colours by part (the mjx model uses one dark material everywhere):
    # silver thigh covers as on the real Go1, charcoal trunk and hips, near-black calves.
    for g in spec.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        n = g.meshname or ""                         # visual meshes are unnamed
        if "thigh" in n:
            rgba = [0.80, 0.81, 0.84, 1]
        elif "calf" in n:
            rgba = [0.13, 0.13, 0.14, 1]
        elif "hip" in n:
            rgba = [0.26, 0.26, 0.28, 1]
        else:                                           # trunk shell and head
            rgba = [0.24, 0.24, 0.26, 1]
        g.material = ""
        g.rgba = rgba

    cam = spec.worldbody.add_camera()
    cam.name = "teaser"
    cam.pos = list(cam_pos)
    cam.quat = list(look_at_quat(cam_pos, cam_target))
    cam.fovy = fovy
    if zoom is not None:
        # second perspective camera: zoom = (pos, target, fovy)
        zc = spec.worldbody.add_camera()
        zc.name = "zoom"
        zc.pos = list(zoom[0])
        zc.quat = list(look_at_quat(zoom[0], zoom[1]))
        zc.fovy = zoom[2]
    return spec.compile()


def solve_leg_ik(model, data, leg: str, target, iters: int = 200) -> float:
    """Damped least squares on the leg's 3 joints so that the foot site reaches
    target (world, m). Returns the final position error [m]."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
    k = LEGS.index(leg)
    dofs = np.arange(6 + 3 * k, 9 + 3 * k)
    qadr = np.arange(7 + 3 * k, 10 + 3 * k)
    jacp = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)               # mj_jacSite needs cdof
        err = np.asarray(target) - data.site_xpos[sid]
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J = jacp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
        data.qpos[qadr] += 0.7 * dq
        for a, j in zip(qadr, range(3)):
            jid = 1 + 3 * k + j
            lo, hi = model.jnt_range[jid]
            data.qpos[a] = np.clip(data.qpos[a], lo, hi)
    mujoco.mj_kinematics(model, data)
    return float(np.linalg.norm(np.asarray(target) - data.site_xpos[sid]))


def pose(model, data, base_pos, base_rpy=(0.0, 0.0, 0.0), feet: dict | None = None) -> dict:
    """Set the base pose and solve each leg's IK for the foot-site targets in feet."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = base_pos
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(base_rpy, float), "xyz")
    data.qpos[3:7] = q
    errs = {}
    for leg, tgt in (feet or {}).items():
        errs[leg] = solve_leg_ik(model, data, leg, tgt)
    mujoco.mj_forward(model, data)
    return errs


def site_pos(model, data, name: str) -> np.ndarray:
    return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def body_pos(model, data, name: str) -> np.ndarray:
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def render(model, data, width: int, height: int, camera: str = "teaser",
           hide_robot: bool = False) -> np.ndarray:
    r = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    if hide_robot:
        opt.geomgroup[2] = 0                         # Go1 visual meshes are group 2
    r.update_scene(data, camera=camera, scene_option=opt)
    img = r.render()
    r.close()
    return img


def project(model, data, pts, width: int, height: int, camera: str = "teaser") -> np.ndarray:
    """World points (N, 3) -> pixel coordinates (N, 2) of a perspective camera
    (pinhole, vertical fovy, principal point at the image centre; y down)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    R = data.cam_xmat[cid].reshape(3, 3)            # columns: camera x, y, z in world
    p = data.cam_xpos[cid]
    f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    P = (np.atleast_2d(pts) - p) @ R                 # camera-frame coordinates
    u = width / 2 + f * P[:, 0] / -P[:, 2]
    v = height / 2 - f * P[:, 1] / -P[:, 2]
    return np.column_stack([u, v])


def cap_ids(model) -> np.ndarray:
    """Geom ids of the tile caps (tile tops)."""
    return np.array([g for g in range(model.ngeom)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("cap_")])


def segment(model, data, width: int, height: int, camera: str = "teaser") -> np.ndarray:
    """Per-pixel geom id (-1 for none). Rendered without MSAA: blended id colours
    from multisampling are not valid ids."""
    ms = model.vis.quality.offsamples
    model.vis.quality.offsamples = 0
    r = mujoco.Renderer(model, height, width)
    r.enable_segmentation_rendering()
    r.update_scene(data, camera=camera)
    seg = r.render()
    r.close()
    model.vis.quality.offsamples = ms
    gid = seg[..., 0].astype(int)
    gid[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
    return gid
