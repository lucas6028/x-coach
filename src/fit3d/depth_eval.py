"""Experiment 1 -- how much true depth does a monocular 3D method recover?

The depth-bottleneck thread (see ``src/rehab24/nlf_skeleton_features.py`` and the
``lift_2d_to_3d*`` modules) concluded *indirectly*, from REHAB24-6 LOSO accuracy
with n=9, that 2D->3D lifting cannot recover out-of-plane depth while NLF direct
image->3D might. Fit3D's mocap ground truth lets us measure that **directly**, with
thousands of joints instead of n=9, and -- crucially -- decompose the error into the
**in-plane** part (which lifting already gets) versus the **depth** part (the axis
toward the camera, which is the actual bottleneck).

Protocol
--------
GT ``joints3d_25`` is world-frame; we map it into each camera frame with the camera
extrinsics so predictions and GT share one frame. Errors are **root-relative** (pelvis
subtracted) by default, which keeps the camera ``z`` axis meaning "depth" -- the whole
point. We report, per method, on the shared Human3.6M-17 core joints:

* ``mpjpe``    -- mean per-joint position error (mm),
* ``inplane``  -- error within the image plane: ``sqrt(dx^2 + dy^2)`` (mm),
* ``depth``    -- error along the camera axis: ``|dz|`` (mm),
* ``pa_mpjpe`` -- Procrustes-aligned MPJPE (removes global rotation/scale),

plus the biomechanical-angle errors that actually drive squat faults (knee/hip angle
are rotation-invariant; torso-lean / hip-below-knee are evaluated in the gravity-aligned
world frame recovered with the GT camera rotation).

Prediction contract
-------------------
The expensive monocular inference runs once on GPU (Kaggle, mirroring the NLF kernel)
and is saved per ``(subject, action, camera)`` as an ``.npz`` under ``--pred-root``:

* ``joints_cam``  (F, 17, 3)  metric 3D in the **camera frame**, Human3.6M-17 order,
                              millimetres (or metres -- set ``--pred-units``).

File name: ``<subject>__<action>__<camera>.npz``. NaN frames are ignored. If a method
emits a different joint layout (NLF SMPL-24, MediaPipe-33), resample it to the H36M-17
core in the kernel; this harness compares only on shared joints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.biomech import WORLD3D, frame_metrics

# Human3.6M-17 core shared by Fit3D GT, lifting outputs, and resampled NLF/MediaPipe.
CORE_JOINTS = tuple(range(17))
ROOT_IDX = ds.ROOT  # 0, pelvis


def pred_npz_name(subject: str, action: str, camera: str) -> str:
    return f"{subject}__{action}__{camera}.npz"


# --- NLF SMPL-24 -> Human3.6M-17 -------------------------------------------------
# NLF (detect_smpl_batched) emits SMPL-24:
#   0 pelv,1 lhip,2 rhip,3 spi1,4 lkne,5 rkne,6 spi2,7 lank,8 rank,9 spi3,10 ltoe,11 rtoe,
#   12 neck,13 lcla,14 rcla,15 head,16 lsho,17 rsho,18 lelb,19 relb,20 lwri,21 rwri,22 lhan,23 rhan.
# H36M-17 target (Fit3D core): 0 pelvis,1 Rhip,2 Rknee,3 Rankle,4 Lhip,5 Lknee,6 Lankle,
#   7 spine,8 thorax,9 neck,10 head,11 Rsho,12 Relb,13 Rwri,14 Lsho,15 Lelb,16 Lwri.
# Spine/thorax are approximate (SMPL spi2/spi3). Whether Fit3D's index-1 side is anatomically
# right is NOT assumed -- ``resolve_lr`` picks the L/R orientation that matches the GT.
SMPL24_TO_H36M17 = (0, 2, 5, 8, 1, 4, 7, 6, 9, 12, 15, 17, 19, 21, 16, 18, 20)
SMPL24_TO_H36M17_SWAP = (0, 1, 4, 7, 2, 5, 8, 6, 9, 12, 15, 16, 18, 20, 17, 19, 21)


def map_smpl24_to_h36m17(joints_smpl24: np.ndarray, swap_lr: bool = False) -> np.ndarray:
    """(F, 24, 3) SMPL-24 -> (F, 17, 3) Human3.6M-17."""
    idx = SMPL24_TO_H36M17_SWAP if swap_lr else SMPL24_TO_H36M17
    return joints_smpl24[:, idx, :]


# H36M-17 left<->right index swap (used to resolve the L/R convention of a model that
# already emits H36M-17, e.g. MediaPipe, against the GT -- same idea as ``resolve_lr``
# for SMPL-24). Pairs: hip 1<->4, knee 2<->5, ankle 3<->6, shoulder 11<->14,
# elbow 12<->15, wrist 13<->16; midline joints (0,7,8,9,10) unchanged.
H36M17_LR_SWAP = (0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13)


def resolve_lr_h36m17(pred_h36m17: np.ndarray, gt_cam_core: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Pick the L/R orientation of an H36M-17 prediction with lower root-relative MPJPE vs GT.

    Mirrors ``resolve_lr`` for the SMPL-24 path: a model that emits H36M-17 directly may
    label left/right opposite to Fit3D's fixed convention. Because pa_mpjpe is *not*
    swap-invariant (Procrustes rotates/scales but never permutes joints), leaving the swap
    unresolved silently inflates it. Returns (pred (F,17,3), swap_lr_used, chosen_mpjpe).
    """
    best = None
    for swap in (False, True):
        pred = pred_h36m17[:, H36M17_LR_SWAP, :] if swap else pred_h36m17
        d = root_relative(pred) - root_relative(gt_cam_core)
        finite = np.isfinite(d).all(axis=2)
        mpjpe = float(np.nanmean(np.linalg.norm(d, axis=2)[finite])) if finite.any() else float("inf")
        if best is None or mpjpe < best[2]:
            best = (pred, swap, mpjpe)
    return best


def resolve_lr(pred_smpl24: np.ndarray, gt_cam_core: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Map SMPL-24 to H36M-17 choosing the L/R orientation with lower root-relative MPJPE vs GT.

    Fit3D's left/right labelling is a fixed dataset convention; this resolves it once against the
    mocap GT instead of guessing. Returns (mapped_pred (F,17,3), swap_lr_used, chosen_mpjpe).
    """
    best = None
    for swap in (False, True):
        pred = map_smpl24_to_h36m17(pred_smpl24, swap)
        d = root_relative(pred) - root_relative(gt_cam_core)
        finite = np.isfinite(d).all(axis=2)
        mpjpe = float(np.nanmean(np.linalg.norm(d, axis=2)[finite])) if finite.any() else float("inf")
        if best is None or mpjpe < best[2]:
            best = (pred, swap, mpjpe)
    return best


def gt_in_camera_frame(j3d_world: np.ndarray, cam_params: dict) -> np.ndarray:
    """Fit3D GT (F, 25, 3) world -> camera frame, metres."""
    return ds.world_to_camera(j3d_world, cam_params)


def root_relative(points: np.ndarray, root_idx: int = ROOT_IDX) -> np.ndarray:
    return points - points[:, root_idx : root_idx + 1, :]


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align each frame of ``pred`` (F, J, 3) onto ``gt`` (rotation+scale+translation)."""
    out = np.full_like(pred, np.nan)
    for f in range(pred.shape[0]):
        P, G = pred[f], gt[f]
        ok = np.isfinite(P).all(1) & np.isfinite(G).all(1)
        if ok.sum() < 3:
            continue
        P0, G0 = P[ok], G[ok]
        muP, muG = P0.mean(0), G0.mean(0)
        Pc, Gc = P0 - muP, G0 - muG
        H = Pc.T @ Gc
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        Rm = Vt.T @ D @ U.T
        scale = float(np.dot(S, [1, 1, d])) / (Pc ** 2).sum() if (Pc ** 2).sum() > 1e-12 else 1.0
        out[f] = (scale * (P - muP) @ Rm.T) + muG
    return out


@dataclass
class DepthError:
    mpjpe: float
    inplane: float
    depth: float
    pa_mpjpe: float
    ex: float  # per-axis mean-abs error: image x
    ey: float  # image y (vertical)
    ez: float  # camera depth -- compare to ex/ey on equal (single-axis) footing
    n_joints: int


def depth_decomposition(
    pred_cam: np.ndarray,
    gt_cam: np.ndarray,
    joints: tuple[int, ...] = CORE_JOINTS,
    root_idx: int = ROOT_IDX,
) -> DepthError:
    """Per-axis error of a camera-frame prediction vs camera-frame GT (same units in/out).

    ``inplane`` is the (x, y) image-plane error and ``depth`` is the |z| camera-axis
    error -- the decomposition that separates "lifting already solves this" from the
    actual bottleneck.
    """
    pred_r = root_relative(pred_cam, root_idx)[:, joints, :]
    gt_r = root_relative(gt_cam, root_idx)[:, joints, :]
    diff = pred_r - gt_r
    finite = np.isfinite(diff).all(axis=2)  # (F, J)

    per_joint = np.linalg.norm(diff, axis=2)
    inplane = np.linalg.norm(diff[..., :2], axis=2)
    depth = np.abs(diff[..., 2])

    pa = procrustes_align(pred_cam[:, joints, :], gt_cam[:, joints, :])
    pa_err = np.linalg.norm(pa - gt_cam[:, joints, :], axis=2)

    abs_axis = np.abs(diff)  # (F, J, 3)
    m = lambda a, mask: float(np.nanmean(a[mask])) if mask.any() else float("nan")  # noqa: E731
    return DepthError(
        mpjpe=m(per_joint, finite),
        inplane=m(inplane, finite),
        depth=m(depth, finite),
        pa_mpjpe=m(pa_err, np.isfinite(pa_err)),
        ex=m(abs_axis[..., 0], finite),
        ey=m(abs_axis[..., 1], finite),
        ez=m(abs_axis[..., 2], finite),
        n_joints=len(joints),
    )


def biomech_error(pred_cam: np.ndarray, gt_world: np.ndarray, cam_params: dict) -> dict[str, float]:
    """Abs error of squat cues. Angles are rotation-invariant; gravity-dependent cues are
    compared after rotating the prediction into the world frame via the GT camera rotation."""
    R = cam_params["extrinsics"]["R"]  # camera axes expressed in world == world<-camera is R.T
    pred_world = pred_cam @ R  # (X_cam) @ R  == R.T @ X_cam per row: camera-frame -> world-aligned directions
    # pad to 25 joints so frame_metrics indexing is valid (only core joints are used by the cues)
    def pad(x):
        out = np.zeros((x.shape[0], ds.NUM_JOINTS, 3))
        out[:, : x.shape[1], :] = x
        return out

    pm = frame_metrics(pad(pred_world), WORLD3D)
    gm = frame_metrics(gt_world, WORLD3D)
    return {
        k: float(np.nanmean(np.abs(pm[k] - gm[k])))
        for k in ("knee_angle", "hip_angle", "torso_lean_deg", "depth_ratio")
    }


def make_perfect_prediction(j3d_world: np.ndarray, cam_params: dict) -> np.ndarray:
    """GT expressed in the camera frame on the core joints -- a zero-error reference/baseline."""
    return gt_in_camera_frame(j3d_world, cam_params)[:, CORE_JOINTS, :]


BIOMECH_KEYS = ("knee_angle", "hip_angle", "torso_lean_deg", "depth_ratio")


def load_prediction_h36m17(
    npz_path: Path, gt_cam_core: np.ndarray, source: str = "smpl3d", units_scale: float = 1.0
) -> tuple[np.ndarray, dict]:
    """Load a prediction npz as (F, 17, 3) camera-frame H36M-17, resolving the format.

    Accepts a direct ``joints_cam`` (F,17,3) layout or an NLF SMPL-24 layout (``smpl3d`` /
    ``smpl3d_np``), mapping the latter with the GT-resolved L/R orientation. Returns
    (pred (F,17,3), info).
    """
    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.files)
        if "joints_cam" in keys:
            pred = np.asarray(data["joints_cam"], dtype=np.float64) * units_scale
            n = min(len(pred), len(gt_cam_core))
            pred, swap, mpjpe = resolve_lr_h36m17(pred[:n], gt_cam_core[:n])
            return pred, {"format": "h36m17", "swap_lr": swap, "resolve_mpjpe": mpjpe}
        key = source if source in keys else ("smpl3d" if "smpl3d" in keys else next(iter(keys & {"smpl3d_np"})))
        smpl = np.asarray(data[key], dtype=np.float64) * units_scale  # (F,24,3) mm
    n = min(len(smpl), len(gt_cam_core))
    pred, swap, mpjpe = resolve_lr(smpl[:n], gt_cam_core[:n])
    return pred, {"format": f"smpl24:{key}", "swap_lr": swap, "resolve_mpjpe": mpjpe}


def evaluate(
    pred_root: Path,
    action: str = "squat",
    split: str = "train",
    pred_units: str = "mm",
    source: str = "smpl3d",
    subjs: list[str] | None = None,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
) -> dict:
    """Evaluate every available prediction npz against Fit3D GT and aggregate (errors in mm/deg)."""
    units_scale = 1000.0 if pred_units == "m" else 1.0  # report in mm; GT metres -> mm below
    h36m = tuple(range(17))
    rows: list[dict] = []
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        j3d_world = ds.load_joints3d(split, subj, action, root) * 1000.0  # m -> mm
        for cam in ds.cameras(split, subj, root):
            npz = pred_root / pred_npz_name(subj, action, cam)
            if not npz.exists():
                continue
            cp = ds.read_cam_params(split, subj, cam, action, root)
            gt_cam = gt_in_camera_frame(j3d_world, cp)
            pred, info = load_prediction_h36m17(npz, gt_cam[:, h36m, :], source, units_scale)
            n = min(len(pred), len(gt_cam))
            de = depth_decomposition(pred[:n], gt_cam[:n, h36m, :], joints=h36m)
            bm = biomech_error(pred[:n], j3d_world[:n], cp)
            rows.append({"subject": subj, "camera": cam, **info, **de.__dict__,
                         **{f"err_{k}": v for k, v in bm.items()}})
    if not rows:
        return {"action": action, "split": split, "n": 0, "pred_root": str(pred_root), "rows": []}
    agg_keys = ["mpjpe", "inplane", "depth", "pa_mpjpe", "ex", "ey", "ez"] + [f"err_{k}" for k in BIOMECH_KEYS]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in agg_keys}
    return {"action": action, "split": split, "source": source, "n": len(rows),
            "pred_root": str(pred_root), "swap_lr": rows[0]["swap_lr"], "aggregate": agg, "rows": rows}
