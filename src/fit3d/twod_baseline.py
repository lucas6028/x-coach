"""Real 2D-detector keypoints (RTMPose) mapped into the Fit3D-25 biomech layout.

This is the *real* 2D arm of the 2D-vs-3D comparison: a genuine 2D pose detector on the Fit3D
videos, as opposed to the **mocap-2D** baseline (GT 3D projected to the image = a *perfect*
detector, used in experiment 2 / model_comparison). Comparing the two decomposes a single-view
2D pipeline's cue error into:

    real-2D error  =  detector error (real-2D - mocap-2D)  +  projection error (mocap-2D - GT-3D)

RTMPose (``rtmlib.Wholebody``) emits COCO-WholeBody 133 keypoints in **pixels**; we take the
COCO-17 body subset and place it into the Fit3D H36M-25 slots the biomech cues use. Fit3D squat
cues are **bilateral** (mean of L/R), so the L/R assignment is free -- we map anatomically for
clarity. Kept in pixels so it shares the mocap-2D coordinate space (``project_world_to_image``),
which the biomech ``IMAGE2D`` cues require (angles distort under anisotropic x/w, y/h normalisation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds

# COCO-WholeBody body indices (first 17 == COCO-17).
COCO_NOSE = 0
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6
COCO_L_HIP, COCO_R_HIP = 11, 12
COCO_L_KNEE, COCO_R_KNEE = 13, 14
COCO_L_ANKLE, COCO_R_ANKLE = 15, 16


def coco_wholebody_to_fit3d25(keypoints: np.ndarray, scores: np.ndarray | None = None,
                              score_thr: float = 0.3) -> np.ndarray:
    """(K>=17, 2) COCO-WholeBody pixels -> (25, 2) Fit3D-25 biomech slots (NaN where unused/low-conf).

    Only the joints the squat cues need are filled: pelvis (mid-hip), L/R hip-knee-ankle, thorax
    (mid-shoulder), L/R shoulder. A joint is NaN'd if either contributing COCO keypoint scores
    below ``score_thr``.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] < 17:
        return np.full((ds.NUM_JOINTS, 2), np.nan)
    sc = None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)

    def pt(i: int) -> np.ndarray:
        if sc is not None and i < sc.shape[0] and sc[i] < score_thr:
            return np.array([np.nan, np.nan])
        return kp[i, :2]

    def mid(i: int, j: int) -> np.ndarray:
        return 0.5 * (pt(i) + pt(j))

    out = np.full((ds.NUM_JOINTS, 2), np.nan)
    out[ds.ROOT] = mid(COCO_L_HIP, COCO_R_HIP)
    out[ds.R_HIP], out[ds.R_KNEE], out[ds.R_ANKLE] = pt(COCO_R_HIP), pt(COCO_R_KNEE), pt(COCO_R_ANKLE)
    out[ds.L_HIP], out[ds.L_KNEE], out[ds.L_ANKLE] = pt(COCO_L_HIP), pt(COCO_L_KNEE), pt(COCO_L_ANKLE)
    out[ds.THORAX] = mid(COCO_L_SHOULDER, COCO_R_SHOULDER)
    out[ds.R_SHOULDER], out[ds.L_SHOULDER] = pt(COCO_R_SHOULDER), pt(COCO_L_SHOULDER)
    return out


# Joints the biomech cues actually index -- used to decide a frame "has a usable detection".
BIOMECH_JOINTS = (ds.ROOT, ds.R_HIP, ds.R_KNEE, ds.R_ANKLE, ds.L_HIP, ds.L_KNEE, ds.L_ANKLE,
                  ds.THORAX, ds.R_SHOULDER, ds.L_SHOULDER)


def load_rtmpose_2d(npz_path: Path) -> np.ndarray:
    """Load a per-video RTMPose npz -> (F, 25, 2) pixel keypoints (NaN where not inferred)."""
    with np.load(npz_path, allow_pickle=True) as d:
        return np.asarray(d["kp2d"], dtype=np.float64)


def pred_npz_name(subject: str, action: str, camera: str) -> str:
    return f"{subject}__{action}__{camera}.npz"
