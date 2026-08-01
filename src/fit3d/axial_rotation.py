"""Experiment 0 -- is segment axial rotation observable from 3D joint centres?

The depth thread (:mod:`src.fit3d.depth_eval`, :mod:`src.fit3d.twod_vs_threed`) established
that a *2D* keypoint pipeline is blind to out-of-plane depth and that direct image->3D
fixes it. This asks the follow-up question for *3D* keypoints: a joint-centre skeleton
places one point per joint, so it pins down where each segment *is* but not how it is
*rolled about its own long axis*. Femoral internal rotation is the textbook mechanism
behind knee valgus, and a three-point knee angle is invariant to it -- so on paper this
looks like a blind spot that survives even a perfect 3D detector.

The measurement uses Fit3D ground truth on both sides, so no estimator quality is involved:

* **target** -- the twist component of the SMPLX ``body_pose`` hip rotation, taken about
  the rest-pose femur axis by swing-twist decomposition (:func:`swing_twist`).
* **input** -- Fit3D ``joints3d_25`` joint centres, i.e. a *perfect* keypoint detector.

``joints3d_25`` limb segments are rigid to ~1e-5 m across frames (verified: hip->knee sd
1.4e-6 m), so it is regressed from the same SMPLX fit as the target. That makes the
residual a clean information-loss measurement rather than fit noise.

Two findings, both against the naive "blind spot" reading:

1. A *single segment* really is uninformative -- ``femur_only`` (hip + knee, the femur's
   own two endpoints) is unstable across actions and often strongly negative.
2. But a *whole-body* skeleton recovers the twist to roughly 2.6-4.0 deg MAE
   (R2_within 0.30-0.76). Keypoints estimate axial rotation through bilateral posture
   correlations even though they never observe it geometrically.

So the deployable claim is not "keypoints cannot see rotation" but a quantitative bar:
an explicit rotation channel only earns its place if it can be *estimated* from video to
better than the ~3 deg a perfect keypoint skeleton already achieves.

Frame caveat that materially changes the numbers: a canonicaliser built from
``L_HIP - R_HIP`` leaks bilateral information into every feature set, including nominally
unilateral ones. :func:`canonicalize_gt` re-expresses the skeleton in the true SMPLX
pelvis frame instead; it is an ORACLE (uses ground truth, not deployable) and plays the
same role as the oracle per-view offset removal in :mod:`src.fit3d.decision_eval`.

Numbers and caveats: ``notes/fit3d_axial_rotation_summary.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMPLX_MODEL = REPO_ROOT / ".kaggle_tmp" / "smplx_ds" / "SMPLX_NEUTRAL.npz"

# SMPLX body joint indices, verified empirically against the model's own ``joint2num``
# map and ``kintree_table`` (parent of L_Knee is L_Hip, femur length 0.383 m).
SMPLX_PELVIS = 0
SMPLX_L_HIP, SMPLX_R_HIP = 1, 2
SMPLX_L_KNEE, SMPLX_R_KNEE = 4, 5
SMPLX_L_ANKLE, SMPLX_R_ANKLE = 7, 8

#: ``body_pose`` is stored as joints 1..21, so joint ``j`` lives at ``body_pose[j - 1]``.
BODY_POSE_OFFSET = 1

# joints3d_25 indices, re-exported from :mod:`src.fit3d.dataset` for local readability.
J_ROOT = ds.ROOT
J_RHIP, J_RKNEE, J_RANK = ds.R_HIP, ds.R_KNEE, ds.R_ANKLE
J_LHIP, J_LKNEE, J_LANK = ds.L_HIP, ds.L_KNEE, ds.L_ANKLE
J_THORAX, J_NECK, J_HEAD = ds.THORAX, ds.NECK, ds.HEAD
J_RSHO, J_RELB, J_RWRI = ds.R_SHOULDER, ds.R_ELBOW, ds.R_WRIST
J_LSHO, J_LELB, J_LWRI = ds.L_SHOULDER, ds.L_ELBOW, ds.L_WRIST

#: Number of H36M-17 core joints in ``joints3d_25`` (indices 17..24 are extremity points).
NUM_H36M17 = 17

#: Nested keypoint sets, ordered so a result localises *where* information comes from.
#: Indices come from :mod:`src.fit3d.dataset` constants so a joint-layout renumbering
#: cannot silently turn one of these into a different experiment.
KEYPOINT_SETS: dict[str, list[int]] = {
    "femur_only": [J_ROOT, J_LHIP, J_LKNEE],
    "leg": [J_ROOT, J_LHIP, J_LKNEE, J_LANK],
    "both_legs": [J_ROOT, J_LHIP, J_RHIP, J_LKNEE, J_RKNEE, J_LANK, J_RANK],
    # no leg joints at all -- arms cannot geometrically constrain femoral roll
    "upper_only": [J_THORAX, J_NECK, J_HEAD, J_RSHO, J_RELB, J_RWRI, J_LSHO, J_LELB, J_LWRI],
    "h36m17": list(range(NUM_H36M17)),
    "full25": list(range(ds.NUM_JOINTS)),
}


# ---------------------------------------------------------------------------
# Quaternion helpers (numpy only)
#
# scipy would supply these, but this repo treats scipy as OPTIONAL -- it is absent from
# both requirements files and its two other users in ``src/`` import it lazily inside
# functions (see ``mediapipe_skeleton_features``, ``paired_loso``). CI installs only
# ``requirements-ci.txt``, so a module-level scipy import here would break the suite.
# ---------------------------------------------------------------------------
def _quat_from_matrix(mats: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` -> unit quaternions ``(N, 4)``, scalar-LAST (x,y,z,w).

    Shepperd's branch-stable method: pick the largest of ``w, x, y, z`` to divide by, so no
    branch ever divides by something near zero. Unlike ``scipy``'s version this does **not**
    re-orthogonalise the input; the callers here feed exact GT rotmats and network outputs
    verified orthogonal to ~4e-7, and the ``sqrt`` arguments are clipped at 0 for safety.
    """
    m = np.asarray(mats, dtype=np.float64)
    n = m.shape[0]
    out = np.empty((n, 4))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    b0 = trace > 0.0
    b1 = ~b0 & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    b2 = ~b0 & ~b1 & (m[:, 1, 1] >= m[:, 2, 2])
    b3 = ~b0 & ~b1 & ~b2

    def _s(vals: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(vals, 0.0, None)) * 2.0

    if b0.any():
        k = m[b0]
        s = _s(trace[b0] + 1.0)
        out[b0] = np.stack([(k[:, 2, 1] - k[:, 1, 2]) / s, (k[:, 0, 2] - k[:, 2, 0]) / s,
                            (k[:, 1, 0] - k[:, 0, 1]) / s, 0.25 * s], axis=1)
    if b1.any():
        k = m[b1]
        s = _s(1.0 + k[:, 0, 0] - k[:, 1, 1] - k[:, 2, 2])
        out[b1] = np.stack([0.25 * s, (k[:, 0, 1] + k[:, 1, 0]) / s,
                            (k[:, 0, 2] + k[:, 2, 0]) / s, (k[:, 2, 1] - k[:, 1, 2]) / s], axis=1)
    if b2.any():
        k = m[b2]
        s = _s(1.0 + k[:, 1, 1] - k[:, 0, 0] - k[:, 2, 2])
        out[b2] = np.stack([(k[:, 0, 1] + k[:, 1, 0]) / s, 0.25 * s,
                            (k[:, 1, 2] + k[:, 2, 1]) / s, (k[:, 0, 2] - k[:, 2, 0]) / s], axis=1)
    if b3.any():
        k = m[b3]
        s = _s(1.0 + k[:, 2, 2] - k[:, 0, 0] - k[:, 1, 1])
        out[b3] = np.stack([(k[:, 0, 2] + k[:, 2, 0]) / s, (k[:, 1, 2] + k[:, 2, 1]) / s,
                            0.25 * s, (k[:, 1, 0] - k[:, 0, 1]) / s], axis=1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of scalar-last quaternions, ``(N, 4) x (N, 4) -> (N, 4)``."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=1)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate == inverse for unit quaternions."""
    return np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)


def _quat_angle(q: np.ndarray) -> np.ndarray:
    """Unsigned rotation angle in ``[0, pi]`` of scalar-last unit quaternions."""
    return 2.0 * np.arctan2(np.linalg.norm(q[:, :3], axis=1), np.abs(q[:, 3]))


def rotation_matrices_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle ``(..., 3)`` -> rotation matrices ``(..., 3, 3)`` via Rodrigues.

    Provided so callers and tests can build rotations without pulling in scipy.
    """
    v = np.asarray(rotvec, dtype=np.float64)
    lead = v.shape[:-1]
    flat = v.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = np.divide(flat, theta, out=np.zeros_like(flat), where=theta > 1e-12)
    kx, ky, kz = unit[:, 0], unit[:, 1], unit[:, 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=1).reshape(-1, 3, 3)
    t = theta.reshape(-1, 1, 1)
    eye = np.broadcast_to(np.eye(3), (len(flat), 3, 3))
    return (eye + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)).reshape(*lead, 3, 3)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def swing_twist(rotations: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split rotations into a twist about ``axis`` plus the residual swing.

    Parameters
    ----------
    rotations:
        ``(..., 3, 3)`` rotation matrices -- a joint's local rotation relative to its
        parent. Non-finite entries propagate as NaN instead of raising.
    axis:
        ``(3,)`` twist axis in the parent frame; normalised internally. For a hip this is
        the rest-pose hip->knee direction (:func:`rest_bone_axis`).

    Returns
    -------
    ``(twist_rad, swing_rad)``, both shaped like ``rotations[..., 0, 0]``. ``twist_rad``
    is *signed* about ``axis`` and lies in ``[-pi, pi]``; ``swing_rad`` is the unsigned
    magnitude of the residual rotation, which carries no component along ``axis``.

    Notes
    -----
    Uses the quaternion form ``twist = 2 * atan2(dot(q.xyz, axis), q.w)`` after forcing
    ``q.w >= 0`` (shortest arc). A rotation of ~180 deg *perpendicular* to ``axis`` leaves
    the twist undefined; those entries come back NaN rather than as an arbitrary value.
    """
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("twist axis must be a finite non-zero vector")
    a = a / norm

    mats = np.asarray(rotations, dtype=np.float64)
    if mats.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) rotation matrices, got {mats.shape}")
    lead = mats.shape[:-2]
    flat = mats.reshape(-1, 3, 3)

    finite = np.isfinite(flat).all(axis=(1, 2))
    quat = np.full((flat.shape[0], 4), np.nan)  # scalar-LAST: (x, y, z, w)
    if finite.any():
        quat[finite] = _quat_from_matrix(flat[finite])

    w = quat[:, 3].copy()
    v = quat[:, :3].copy()
    flip = w < 0
    w[flip] *= -1.0
    v[flip] *= -1.0
    quat_pos = np.concatenate([v, w[:, None]], axis=1)

    along = v @ a
    twist = 2.0 * np.arctan2(along, w)
    degenerate = (np.abs(w) < 1e-8) & (np.abs(along) < 1e-8)
    twist[degenerate] = np.nan

    q_twist = np.concatenate([along[:, None] * a[None, :], w[:, None]], axis=1)
    n = np.linalg.norm(q_twist, axis=1, keepdims=True)
    q_twist = np.divide(q_twist, n, out=np.full_like(q_twist, np.nan), where=n > 1e-12)
    swing = np.full(flat.shape[0], np.nan)
    usable = finite & np.isfinite(q_twist).all(axis=1) & ~degenerate
    if usable.any():
        residual = _quat_mul(quat_pos[usable], _quat_conj(q_twist[usable]))
        swing[usable] = _quat_angle(residual)

    return twist.reshape(lead), swing.reshape(lead)


def rest_joints(model_path: Path = DEFAULT_SMPLX_MODEL) -> np.ndarray:
    """SMPLX rest-pose joint locations ``(55, 3)`` = ``J_regressor @ v_template``."""
    with np.load(model_path, allow_pickle=True) as data:
        regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        template = np.asarray(data["v_template"], dtype=np.float64)
    return regressor @ template


def rest_bone_axis(child: int, parent: int, model_path: Path = DEFAULT_SMPLX_MODEL) -> np.ndarray:
    """Unit rest-pose direction ``parent -> child``, i.e. the segment's twist axis.

    At rest every SMPL local rotation is the identity, so the parent frame coincides with
    the template frame and this direction is the correct axis for :func:`swing_twist`.
    """
    joints = rest_joints(model_path)
    vec = joints[child] - joints[parent]
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
def load_smplx(
    split: str, subj: str, action: str, root: Path = ds.DEFAULT_FIT3D_ROOT
) -> dict[str, np.ndarray]:
    """Fit3D SMPLX fit for one sequence (``body_pose`` ``(F, 21, 3, 3)``, rotation matrices)."""
    path = root / split / subj / "smplx" / f"{action}.json"
    raw = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def hip_twist_series(
    body_pose: np.ndarray,
    side: str,
    model_path: Path = DEFAULT_SMPLX_MODEL,
    rest: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame femoral axial rotation in **degrees** for ``side`` in ``{'L', 'R'}``.

    ``rest`` overrides the rest-pose joints used to build the twist axis. Pass the
    predicting model's OWN rest joints when the estimate comes from a different body model
    than the ground truth -- HMR2.0 predicts SMPL while Fit3D GT is SMPLX, and although
    their body-joint indices agree for 0..21, their rest femur directions differ slightly,
    which would otherwise show up as a spurious systematic twist offset. The Kaggle
    rotation kernel exports SMPL's rest joints as ``rest_j_smpl`` in each npz for this.

    Left and right come out mirrored for the same anatomical direction (the template is
    left-right symmetric, so the two femur axes are mirror images and the twist sign flips
    with them). The sign is therefore consistent within a side but the mapping onto
    "internal" vs "external" rotation is **not** verified -- do not read the sign as
    anatomy without checking it first.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    hip = SMPLX_L_HIP if side == "L" else SMPLX_R_HIP
    knee = SMPLX_L_KNEE if side == "L" else SMPLX_R_KNEE
    if rest is None:
        axis = rest_bone_axis(knee, hip, model_path)
    else:
        vec = np.asarray(rest, dtype=np.float64)[knee] - np.asarray(rest, dtype=np.float64)[hip]
        axis = vec / np.linalg.norm(vec)
    twist, _ = swing_twist(body_pose[:, hip - BODY_POSE_OFFSET], axis)
    return np.degrees(twist)


# ---------------------------------------------------------------------------
# Estimate vs ground truth
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwistAgreement:
    """How close a monocular twist estimate lands to the SMPLX ground truth.

    Three MAEs, mirroring the three-valued structure in :mod:`src.fit3d.decision_eval`:

    ``mae_raw``
        no calibration at all -- the honest deployed number, but it carries the
        SMPL-vs-SMPLX parameterisation offset, which is not the estimator's fault.
    ``mae_debiased``
        one global offset, fitted on the OTHER subjects (leave-one-subject-out), so it
        removes the parameterisation offset without leaking anything subject-specific.
        **This is the number to compare against the keypoint bar.**
    ``mae_oracle``
        per-subject offset removal using the held-out subject's own mean -- an upper bound
        on what any calibration could buy, not deployable.

    ``n_groups`` guards a trap: with a single subject there are no "other subjects" to fit
    the offset on, so ``mae_debiased`` silently degenerates to ``mae_oracle`` and stops
    being a LOSO number. Check :attr:`loso_is_degenerate` before comparing it to a bar that
    was itself computed under LOSO.
    """

    n: int
    n_groups: int
    mae_raw: float
    mae_debiased: float
    mae_oracle: float
    bias: float
    pearson: float

    @property
    def loso_is_degenerate(self) -> bool:
        """True when ``mae_debiased`` is really an oracle number (fewer than 2 subjects)."""
        return self.n_groups < 2


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def compare_twist(gt_deg: np.ndarray, est_deg: np.ndarray, groups: np.ndarray) -> TwistAgreement:
    """Agreement between an estimated and a ground-truth twist series.

    Frames where either side is non-finite are dropped (the kernel writes NaN off the
    subsample grid and on detection failures), so pass full-length arrays freely.
    """
    gt = np.asarray(gt_deg, dtype=np.float64)
    est = np.asarray(est_deg, dtype=np.float64)
    grp = np.asarray(groups)
    if not (gt.shape == est.shape == grp.shape):
        raise ValueError(f"shape mismatch: gt {gt.shape}, est {est.shape}, groups {grp.shape}")
    keep = np.isfinite(gt) & np.isfinite(est)
    gt, est, grp = gt[keep], est[keep], grp[keep]
    if gt.size < 2:
        raise ValueError("need at least 2 paired finite frames")

    err = est - gt
    debiased = est.copy()
    oracle = est.copy()
    subjects = set(grp.tolist())
    for subject in subjects:
        held = grp == subject
        others = ~held
        # LOSO offset: what the OTHER subjects say the constant offset is. With a single
        # subject there are none, so this degenerates to the oracle -- flagged via n_groups.
        debiased[held] -= err[others].mean() if others.any() else err.mean()
        oracle[held] -= err[held].mean()

    centred_gt = gt - gt.mean()
    centred_est = est - est.mean()
    denom = float(np.linalg.norm(centred_gt) * np.linalg.norm(centred_est))
    return TwistAgreement(
        n=int(gt.size),
        n_groups=len(subjects),
        mae_raw=_mae(est, gt),
        mae_debiased=_mae(debiased, gt),
        mae_oracle=_mae(oracle, gt),
        bias=float(err.mean()),
        pearson=float(centred_gt @ centred_est / denom) if denom > 1e-12 else float("nan"),
    )


# ---------------------------------------------------------------------------
# Skeleton canonicalisation
# ---------------------------------------------------------------------------
def canonicalize(joints3d: np.ndarray) -> np.ndarray:
    """Root-centre, remove global heading, scale-normalise -- using keypoints only.

    WARNING: the lateral axis is ``L_HIP - R_HIP``, so this leaks bilateral information
    into *every* feature set built on top of it, including nominally single-leg ones. Use
    :func:`canonicalize_gt` when the question is how much a unilateral chain carries.
    """
    x = np.asarray(joints3d, dtype=np.float64)
    x = x - x[:, J_ROOT : J_ROOT + 1]
    lat = x[:, J_LHIP] - x[:, J_RHIP]
    lat = lat / np.linalg.norm(lat, axis=1, keepdims=True)
    up = x[:, J_THORAX] - x[:, J_ROOT]
    up = up - np.sum(up * lat, axis=1, keepdims=True) * lat
    scale = np.linalg.norm(up, axis=1, keepdims=True)
    up = up / scale
    basis = np.stack([lat, up, np.cross(lat, up)], axis=1)
    return np.einsum("fij,fkj->fki", basis, x) / scale[:, :, None]


def canonicalize_gt(joints3d: np.ndarray, pelvis_rotation: np.ndarray) -> np.ndarray:
    """Root-centre then rotate into the **true** SMPLX pelvis frame (``global_orient``).

    Oracle preprocessing: it consumes ground truth and is not deployable. Its purpose is to
    separate "this feature set carries hip information" from "this feature set helps pin
    down the reference frame the target is defined in".
    """
    x = np.asarray(joints3d, dtype=np.float64)
    x = x - x[:, J_ROOT : J_ROOT + 1]
    lat = x[:, J_LHIP] - x[:, J_RHIP]
    lat = lat / np.linalg.norm(lat, axis=1, keepdims=True)
    up = x[:, J_THORAX] - x[:, J_ROOT]
    up = up - np.sum(up * lat, axis=1, keepdims=True) * lat
    scale = np.linalg.norm(up, axis=1, keepdims=True)
    return np.einsum("fji,fkj->fki", pelvis_rotation, x) / scale[:, :, None]


# ---------------------------------------------------------------------------
# Observability regression (numpy-only; matches the repo's dependency-light style)
# ---------------------------------------------------------------------------
def ridge_predictor(x_train: np.ndarray, y_train: np.ndarray, lam: float):
    """Closed-form ridge on standardised features; the intercept is left unpenalised."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-9
    z = np.hstack([(x_train - mu) / sd, np.ones((len(x_train), 1))])
    gram = z.T @ z + lam * np.eye(z.shape[1])
    gram[-1, -1] -= lam
    weights = np.linalg.solve(gram, z.T @ y_train)

    def predict(x_test: np.ndarray) -> np.ndarray:
        zt = np.hstack([(x_test - mu) / sd, np.ones((len(x_test), 1))])
        return zt @ weights

    return predict


def rbf_krr_predictor(x_train: np.ndarray, y_train: np.ndarray, lam: float = 1.0):
    """RBF kernel ridge with the median-distance bandwidth heuristic.

    The nonlinear control: without it, a near-zero linear score cannot distinguish
    "unobservable" from "observable but not linearly".
    """
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-9
    z = (x_train - mu) / sd
    sq = np.sum(z ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * z @ z.T
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-d2 / (2.0 * sigma2))
    offset = y_train.mean()
    alpha = np.linalg.solve(kernel + lam * np.eye(len(kernel)), y_train - offset)

    def predict(x_test: np.ndarray) -> np.ndarray:
        zt = (x_test - mu) / sd
        dt = np.sum(zt ** 2, axis=1)[:, None] + sq[None, :] - 2.0 * zt @ z.T
        return np.exp(-dt / (2.0 * sigma2)) @ alpha + offset

    return predict


@dataclass(frozen=True)
class FoldScore:
    """One held-out subject.

    ``r2_within`` scores against the held-out subject's OWN mean, so it measures whether
    within-subject variation -- the movement-quality signal -- is predictable, rather than
    rewarding a model for merely reproducing between-subject offsets (``r2_global``).
    """

    subject: str
    r2_global: float
    r2_within: float
    mae: float
    target_sd: float


def loso_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray, make_predictor) -> list[FoldScore]:
    """Leave-one-subject-out evaluation, the convention used across this repo's studies."""
    out: list[FoldScore] = []
    for subject in sorted(set(groups.tolist())):
        held = groups == subject
        predict = make_predictor(x[~held], y[~held])
        pred = predict(x[held])
        truth = y[held]
        ss_res = float(np.sum((truth - pred) ** 2))
        out.append(
            FoldScore(
                subject=str(subject),
                r2_global=1.0 - ss_res / float(np.sum((truth - y[~held].mean()) ** 2)),
                r2_within=1.0 - ss_res / float(np.sum((truth - truth.mean()) ** 2)),
                mae=float(np.mean(np.abs(truth - pred))),
                target_sd=float(np.std(truth)),
            )
        )
    return out
