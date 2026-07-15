"""MediaPipe BlazePose-33 world landmarks mapped to the Human3.6M-17 core.

The **sparse + weak-depth** arm of the depth-bottleneck study. MediaPipe Pose
(BlazePose GHUM) emits 33 landmarks and a metric ``pose_world_landmarks`` field --
3D coordinates in metres, origin at the mid-hip, axes roughly **image-aligned**
(x right, y down, z depth, larger z = farther from camera). Unlike the dense SMPL
regressors (NLF/HMR2.0/Multi-HMR) it is a lightweight on-device model whose depth
(z) is a *weak* estimate, so it is the natural counterpart to MeTRAbs's sparse but
metric (真深度) 3D.

We keep the same evaluation contract as the other 3D methods: resample to H36M-17,
camera frame, millimetres, saved per ``(subject, action, camera)`` npz. Because the
world-landmark frame is only *roughly* the true camera rotation (the model never sees
the extrinsics), MediaPipe carries the same frame caveat as HMR2.0's crop frame --
its raw per-axis ``ez`` is not literal depth-mm; lean on ``pa_mpjpe``, the
rotation-invariant knee/hip cues, the scale-free ``ez/exy`` ratio, and verdict-flip.

BlazePose-33 landmark indices (MediaPipe Pose):
    0 nose, 1-6 eyes, 7 l_ear, 8 r_ear, 9-10 mouth, 11 l_shoulder, 12 r_shoulder,
    13 l_elbow, 14 r_elbow, 15 l_wrist, 16 r_wrist, 17-22 hand pts, 23 l_hip,
    24 r_hip, 25 l_knee, 26 r_knee, 27 l_ankle, 28 r_ankle, 29-32 heel/foot pts.

H36M-17 has no direct landmarks for pelvis/spine/thorax/neck/head, so those are
derived from midpoints. Only pelvis/hip/knee/ankle/shoulder feed the squat cues; the
derived spine/neck/head affect only ``mpjpe``/``pa_mpjpe`` and are approximate. L/R is
mapped anatomically (right side -> H36M index 1) and re-resolved against the GT at eval
time by ``depth_eval.resolve_lr_h36m17``, so the mapping's handedness is not load-bearing.
"""

from __future__ import annotations

import numpy as np

# --- BlazePose-33 landmark indices we use ----------------------------------------
BP_NOSE = 0
BP_L_EAR, BP_R_EAR = 7, 8
BP_L_SHOULDER, BP_R_SHOULDER = 11, 12
BP_L_ELBOW, BP_R_ELBOW = 13, 14
BP_L_WRIST, BP_R_WRIST = 15, 16
BP_L_HIP, BP_R_HIP = 23, 24
BP_L_KNEE, BP_R_KNEE = 25, 26
BP_L_ANKLE, BP_R_ANKLE = 27, 28

NUM_BLAZEPOSE = 33


def blazepose33_to_h36m17(world_landmarks: np.ndarray) -> np.ndarray:
    """(F, 33, 3) BlazePose world landmarks -> (F, 17, 3) Human3.6M-17.

    Derived joints: pelvis = mid-hip, thorax = mid-shoulder, head = mid-ear,
    neck = midway thorax->head, spine = midway pelvis->thorax. NaN-safe (a NaN in any
    contributing landmark propagates to the derived joint, which the eval then ignores).
    """
    j = np.asarray(world_landmarks, dtype=np.float64)
    if j.ndim != 3 or j.shape[1] < NUM_BLAZEPOSE:
        raise ValueError(f"expected (F, >=33, 3) BlazePose landmarks, got {j.shape}")

    def mid(a: int, b: int) -> np.ndarray:
        return 0.5 * (j[:, a, :] + j[:, b, :])

    pelvis = mid(BP_L_HIP, BP_R_HIP)
    thorax = mid(BP_L_SHOULDER, BP_R_SHOULDER)
    head = mid(BP_L_EAR, BP_R_EAR)
    neck = 0.5 * (thorax + head)
    spine = 0.5 * (pelvis + thorax)

    out = np.full((j.shape[0], 17, 3), np.nan, dtype=np.float64)
    out[:, 0] = pelvis
    out[:, 1] = j[:, BP_R_HIP]
    out[:, 2] = j[:, BP_R_KNEE]
    out[:, 3] = j[:, BP_R_ANKLE]
    out[:, 4] = j[:, BP_L_HIP]
    out[:, 5] = j[:, BP_L_KNEE]
    out[:, 6] = j[:, BP_L_ANKLE]
    out[:, 7] = spine
    out[:, 8] = thorax
    out[:, 9] = neck
    out[:, 10] = head
    out[:, 11] = j[:, BP_R_SHOULDER]
    out[:, 12] = j[:, BP_R_ELBOW]
    out[:, 13] = j[:, BP_R_WRIST]
    out[:, 14] = j[:, BP_L_SHOULDER]
    out[:, 15] = j[:, BP_L_ELBOW]
    out[:, 16] = j[:, BP_L_WRIST]
    return out


def pred_npz_name(subject: str, action: str, camera: str) -> str:
    return f"{subject}__{action}__{camera}.npz"
