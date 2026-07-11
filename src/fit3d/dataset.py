"""Fit3D dataset access: 3D ground truth, camera calibration, and projection.

Layout (already extracted under ``data/Fit3D``)::

    Fit3D/
      fit3d_info.json                       # train/test subject split + camera names
      {train,test}/<subj>/
        joints3d_25/<action>.json           # {"joints3d_25": (F, 25, 3)} world metres  (train only)
        camera_parameters/<cam>/<action>.json
        videos/<cam>/<action>.mp4
        smplx/<action>.json
        gpp/<action>.json
        rep_ann.json                        # {action: [boundary_frame, ...]}  (train only)

The ``joints3d_25`` skeleton is the **Human3.6M-17 convention** for indices 0..16
plus 8 extremity points (feet/hands), verified against the official limb
connectivity in ``sminchisescu-research/imar_vision_datasets_tools`` (``util/dataset_util.py``)::

    limbs = [[10,9],[9,8],[8,11],[8,14],[11,12],[14,15],[12,13],[15,16],
             [8,7],[7,0],[0,1],[0,4],[1,2],[4,5],[2,3],[5,6],
             [13,21],[13,22],[16,23],[16,24],[3,17],[3,18],[6,19],[6,20]]

L/R labels below follow the H36M ordering (index 1 = right hip). Squat metrics are
bilateral, so an L/R swap relative to the physical body does not affect them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT3D_ROOT = REPO_ROOT / "data" / "Fit3D"
INFO_FILE = DEFAULT_FIT3D_ROOT / "fit3d_info.json"

NUM_JOINTS = 25

# --- Human3.6M-17 core (indices 0..16) -------------------------------------
ROOT = 0  # pelvis / mid-hip
R_HIP, R_KNEE, R_ANKLE = 1, 2, 3
L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
SPINE = 7
THORAX = 8
NECK = 9
HEAD = 10
R_SHOULDER, R_ELBOW, R_WRIST = 11, 12, 13
L_SHOULDER, L_ELBOW, L_WRIST = 14, 15, 16
# --- extremity points (indices 17..24) -------------------------------------
R_FOOT_A, R_FOOT_B = 17, 18  # two foot points hanging off the right ankle (3)
L_FOOT_A, L_FOOT_B = 19, 20  # two foot points hanging off the left ankle (6)
R_HAND_A, R_HAND_B = 21, 22  # two hand points off the right wrist (13)
L_HAND_A, L_HAND_B = 23, 24  # two hand points off the left wrist (16)

JOINT_NAMES = (
    "pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
    "spine", "thorax", "neck", "head",
    "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_foot_a", "r_foot_b", "l_foot_a", "l_foot_b",
    "r_hand_a", "r_hand_b", "l_hand_a", "l_hand_b",
)
assert len(JOINT_NAMES) == NUM_JOINTS

# Bone connectivity (from the official imar visualiser), useful for plotting/debug.
LIMBS: tuple[tuple[int, int], ...] = (
    (10, 9), (9, 8), (8, 11), (8, 14), (11, 12), (14, 15), (12, 13), (15, 16),
    (8, 7), (7, 0), (0, 1), (0, 4), (1, 2), (4, 5), (2, 3), (5, 6),
    (13, 21), (13, 22), (16, 23), (16, 24), (3, 17), (3, 18), (6, 19), (6, 20),
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def load_info(root: Path = DEFAULT_FIT3D_ROOT) -> dict:
    return json.loads((root / "fit3d_info.json").read_text())


def subjects(split: str, root: Path = DEFAULT_FIT3D_ROOT) -> list[str]:
    """Subject ids present on disk for ``split`` ('train' or 'test')."""
    split_dir = root / split
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def cameras(split: str, subj: str, root: Path = DEFAULT_FIT3D_ROOT) -> list[str]:
    cam_dir = root / split / subj / "camera_parameters"
    return sorted(p.name for p in cam_dir.iterdir() if p.is_dir())


def actions(split: str, subj: str, root: Path = DEFAULT_FIT3D_ROOT) -> list[str]:
    """Action names for a subject (from joints3d_25 on train, videos on test)."""
    subj_dir = root / split / subj
    j3d_dir = subj_dir / "joints3d_25"
    if j3d_dir.is_dir():
        return sorted(p.stem for p in j3d_dir.glob("*.json"))
    # test split: only videos exist
    cam0 = cameras(split, subj, root)[0]
    return sorted(p.stem for p in (subj_dir / "videos" / cam0).glob("*.mp4"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_joints3d(split: str, subj: str, action: str, root: Path = DEFAULT_FIT3D_ROOT) -> np.ndarray:
    """World-frame 3D joints, shape (F, 25, 3) in metres (Z is up)."""
    path = root / split / subj / "joints3d_25" / f"{action}.json"
    data = json.loads(path.read_text())
    return np.asarray(data["joints3d_25"], dtype=np.float64)


def read_cam_params(split: str, subj: str, camera: str, action: str, root: Path = DEFAULT_FIT3D_ROOT) -> dict:
    """Camera calibration with every leaf turned into a numpy array (imar convention)."""
    path = root / split / subj / "camera_parameters" / camera / f"{action}.json"
    raw = json.loads(path.read_text())
    return {k1: {k2: np.asarray(v2, dtype=np.float64) for k2, v2 in v1.items()} for k1, v1 in raw.items()}


def load_rep_ann(split: str, subj: str, root: Path = DEFAULT_FIT3D_ROOT) -> dict[str, list[int]]:
    path = root / split / subj / "rep_ann.json"
    return json.loads(path.read_text())


def rep_segments(boundaries: list[int]) -> list[tuple[int, int]]:
    """Turn a list of rep boundary frames into [start, end) windows (consecutive pairs)."""
    bounds = sorted(int(b) for b in boundaries)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# ---------------------------------------------------------------------------
# Geometry: world -> camera -> image
# ---------------------------------------------------------------------------
def world_to_camera(points_world: np.ndarray, cam_params: dict) -> np.ndarray:
    """Transform world points (..., 3) into the camera frame.

    Fit3D stores ``T`` as the camera position in world coordinates, so
    ``X_cam = R @ (X_world - T)`` (verified against the GHUM helper, which sets the
    camera translation to ``-R @ T``).
    """
    R = cam_params["extrinsics"]["R"]            # (3, 3)
    T = cam_params["extrinsics"]["T"].reshape(3)  # (3,)
    return (points_world - T) @ R.T


def project_3d_to_2d(points_cam: np.ndarray, cam_params: dict, with_distortion: bool = True) -> np.ndarray:
    """Project camera-frame points (..., 3) to image pixels (..., 2).

    Ported verbatim from ``imar_vision_datasets_tools/util/dataset_util.py`` and
    generalised over leading batch dimensions.
    """
    key = "intrinsics_w_distortion" if with_distortion else "intrinsics_wo_distortion"
    intr = cam_params[key]
    f = np.asarray(intr["f"], dtype=np.float64).reshape(2)
    c = np.asarray(intr["c"], dtype=np.float64).reshape(2)

    pts = np.asarray(points_cam, dtype=np.float64)
    lead = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    z = np.where(np.abs(flat[:, 2:3]) < 1e-9, np.nan, flat[:, 2:3])
    x = flat[:, :2] / z  # (N, 2) normalised image coords

    if with_distortion:
        k = np.asarray(intr["k"], dtype=np.float64).reshape(1, 3)
        p = np.asarray(intr["p"], dtype=np.float64).reshape(1, 2)[:, [1, 0]]
        r2 = np.sum(x ** 2, axis=1)
        radial = 1 + np.transpose(np.matmul(k, np.array([r2, r2 ** 2, r2 ** 3])))  # (N, 1)
        tan = np.matmul(x, np.transpose(p))  # (N, 1)
        xx = x * (tan + radial) + r2[:, None] * p
    else:
        xx = x
    proj = f * xx + c
    return proj.reshape(*lead, 2)


def project_world_to_image(points_world: np.ndarray, cam_params: dict, with_distortion: bool = True) -> np.ndarray:
    """Convenience: world (..., 3) -> image pixels (..., 2) for one camera."""
    return project_3d_to_2d(world_to_camera(points_world, cam_params), cam_params, with_distortion)


# ---------------------------------------------------------------------------
# Iteration helper
# ---------------------------------------------------------------------------
def iter_sequences(
    split: str,
    action: str,
    subjs: list[str] | None = None,
    root: Path = DEFAULT_FIT3D_ROOT,
) -> Iterator[tuple[str, np.ndarray, dict[str, dict]]]:
    """Yield (subject, joints3d (F,25,3), {camera: cam_params}) for one action across subjects."""
    for subj in subjs or subjects(split, root):
        if action not in actions(split, subj, root):
            continue
        j3d = load_joints3d(split, subj, action, root)
        cams = {cam: read_cam_params(split, subj, cam, action, root) for cam in cameras(split, subj, root)}
        yield subj, j3d, cams
