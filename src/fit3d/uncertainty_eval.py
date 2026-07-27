"""Does a pose estimator's own uncertainty carry information its keypoints do not?

Blind spot C of the "what do keypoints structurally miss" thread. A keypoint skeleton is a
**point estimate**: it says where each joint is and has nowhere to record how much that should
be believed. Two readings with identical coordinates but different reliability warrant
different coaching decisions, so on paper no amount of accuracy improvement substitutes for a
confidence channel -- a representational gap, like A (axial rotation) and B (the implement).

This project already has such a channel and has never used it. NLF writes ``unc`` (F, 24)
per-joint uncertainty in millimetres into every prediction npz, for both Fit3D and REHAB24-6;
its only mention anywhere in ``src/`` is one docstring line. Fit3D has mocap ground truth and a
synchronised 4-camera rig, so it can answer whether the channel is meaningful and whether it is
redundant, before any downstream work is spent on it.

Three traps this module is built around
---------------------------------------
**1. The SMPL-24 -> H36M-17 index list must follow the L/R swap.** ``depth_eval.resolve_lr``
picks one of two index lists by minimising MPJPE. Indexing ``unc`` with the other one pairs
each joint's uncertainty with a different joint's error, which shows up as a near-null
correlation that reads exactly like "uncertainty is uncalibrated". :func:`load_sequences`
returns them already consistent.

**2. Never compare uncertainty ACROSS joints.** The SMPL and H36M joint conventions disagree by
a large constant: measured on Fit3D squats the per-joint offset is 176 mm at the thorax, 95 mm
at the left hip, 95 mm at the head. Raw per-joint MPJPE therefore ranks thorax and hips as the
"worst" joints, which is anatomically absurd -- they are the *best localised* and the *worst
matched*. Against that, ``unc`` (which correctly calls wrists and elbows hardest) looks
anti-correlated: Spearman -0.15 between per-joint median error and per-joint median
uncertainty. Within joints the same data gives +0.44. Every comparison here is within-joint.

**3. A per-joint lookup table is the baseline to beat, not zero.** "Which joint is this"
already predicts most of the error spread. A confidence channel has to improve on that
constant, and improving on the grand mean proves nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d import depth_eval as de

DEFAULT_PRED_ROOT = ds.DEFAULT_FIT3D_ROOT / "derived" / "preds" / "nlf"
H36M17_NAMES = ("pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
                "spine", "thorax", "neck", "head", "r_shoulder", "r_elbow", "r_wrist",
                "l_shoulder", "l_elbow", "l_wrist")
N_JOINTS = 17


@dataclass(frozen=True)
class Sequence:
    """One (subject, camera) prediction aligned to ground truth, all in millimetres."""

    subject: str
    camera: str
    action: str
    delta: np.ndarray       # (F, 17, 3) root-relative prediction minus ground truth
    uncertainty: np.ndarray  # (F, 17) NLF `unc`, mapped through the SAME L/R index list
    pose: np.ndarray        # (F, 17, 3) root-relative PREDICTED pose (what a consumer sees)
    swap_lr: bool

    def __len__(self) -> int:
        return len(self.delta)


def load_sequences(
    action: str,
    pred_root: Path = DEFAULT_PRED_ROOT,
    split: str = "train",
    root: Path = ds.DEFAULT_FIT3D_ROOT,
) -> list[Sequence]:
    """Load NLF predictions for ``action`` alongside Fit3D ground truth.

    ``uncertainty`` is indexed with the *same* SMPL-24 -> H36M-17 list that ``resolve_lr``
    chose for the pose. See trap 1 in the module docstring.
    """
    out: list[Sequence] = []
    for path in sorted(Path(pred_root).glob("*.npz")):
        parts = path.stem.split("__")
        if len(parts) != 3 or parts[1] != action:
            continue
        data = np.load(path)
        pred24 = np.asarray(data["smpl3d"], dtype=np.float64)
        unc24 = np.asarray(data["unc"], dtype=np.float64)
        subject, camera = str(data["subject"]), str(data["camera"])
        joints = ds.load_joints3d(split, subject, action, root)
        cam_params = ds.read_cam_params(split, subject, camera, action, root)
        gt_cam = de.gt_in_camera_frame(joints, cam_params)[:, :N_JOINTS] * 1000.0

        n = min(len(pred24), len(gt_cam))
        pred24, unc24, gt_cam = pred24[:n], unc24[:n], gt_cam[:n]
        pred17, swap, _ = de.resolve_lr(pred24, gt_cam)
        index = de.SMPL24_TO_H36M17_SWAP if swap else de.SMPL24_TO_H36M17

        out.append(Sequence(
            subject=subject, camera=camera, action=action,
            delta=de.root_relative(pred17) - de.root_relative(gt_cam),
            uncertainty=unc24[:, list(index)],
            pose=de.root_relative(pred17),
            swap_lr=bool(swap),
        ))
    return out


def convention_bias(delta: np.ndarray) -> np.ndarray:
    """Per-joint constant offset (1, 17, 3) -- the SMPL/H36M definition mismatch, not error.

    Estimated as a median so a minority of tracking failures cannot move it. Callers doing
    leave-one-subject-out must estimate this on TRAINING sequences only.
    """
    return np.median(np.asarray(delta, dtype=np.float64), axis=0, keepdims=True)


def corrected_error(delta: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """(F, 17) error magnitude with the joint-convention offset removed."""
    delta = np.asarray(delta, dtype=np.float64)
    if bias is None:
        bias = convention_bias(delta)
    return np.linalg.norm(delta - bias, axis=2)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged.

    ``argsort(argsort(x))`` is NOT a rank function when there are ties: on a constant array it
    returns 0, 1, 2, ... which has full variance and correlates with whatever order the rows
    happened to arrive in. That fabricated a rho of +0.229 for the pelvis -- a joint whose
    root-relative error is identically zero -- until this was fixed.
    """
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    ends = np.append(starts[1:], n)
    group_rank = (starts + ends - 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_rank[np.cumsum(is_new) - 1]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, numpy-only (scipy is an optional dependency in this repo)."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _average_ranks(a[ok]), _average_ranks(b[ok])
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def within_joint_calibration(error: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """(17,) Spearman between uncertainty and error, computed separately per joint."""
    return np.array([spearman(uncertainty[:, j], error[:, j]) for j in range(N_JOINTS)])


# ---------------------------------------------------------------------------
# Cross-view: the controlled contrast
# ---------------------------------------------------------------------------
def cross_view_agreement(sequences: list[Sequence]) -> dict:
    """Do the cameras that are actually worse at this instant also report higher uncertainty?

    Fit3D's cameras are synchronised, so for a fixed (subject, frame, joint) the underlying
    motion is *identical* across views and any difference in error is purely view-induced.
    This is the cleanest possible test of the "learned reliability routing" idea, and it is the
    one thing that could upgrade the hand-written ``view_type`` gate in the rule detector into
    something measured.

    Both quantities are standardised **within (camera, joint)** first, so a camera that is
    uniformly worse -- or a joint whose convention offset differs per camera -- cannot create
    agreement on its own. What is left is "which view is unusually bad *right now*".
    """
    by_subject: dict[str, dict[str, Sequence]] = {}
    for seq in sequences:
        by_subject.setdefault(seq.subject, {})[seq.camera] = seq

    per_subject, pooled_rho = {}, []
    for subject, cams in sorted(by_subject.items()):
        if len(cams) < 3:
            continue
        names = sorted(cams)
        length = min(len(cams[c]) for c in names)
        err = np.stack([corrected_error(cams[c].delta)[:length] for c in names])       # (V,F,J)
        unc = np.stack([cams[c].uncertainty[:length] for c in names])                  # (V,F,J)
        # standardise within (view, joint): removes any per-camera or per-joint constant
        err_z = (err - err.mean(axis=1, keepdims=True)) / (err.std(axis=1, keepdims=True) + 1e-9)
        unc_z = (unc - unc.mean(axis=1, keepdims=True)) / (unc.std(axis=1, keepdims=True) + 1e-9)
        rho = spearman(unc_z.transpose(1, 2, 0).ravel(), err_z.transpose(1, 2, 0).ravel())
        per_subject[subject] = {"n_views": len(names), "n_frames": int(length), "rho": rho}
        pooled_rho.append(rho)
    return {"per_subject": per_subject,
            "mean_rho": float(np.mean(pooled_rho)) if pooled_rho else float("nan"),
            "n_subjects": len(pooled_rho)}


# ---------------------------------------------------------------------------
# Redundancy: does `unc` add anything over the pose itself?
# ---------------------------------------------------------------------------
def cross_view_routing(sequences: list[Sequence], lam: float = 100.0) -> dict:
    """The product question: to pick the most reliable view, is ``unc`` better than the pose?

    :func:`cross_view_agreement` shows ``unc`` knows which view is unusually bad right now. That
    only matters if the pose does not already know it -- and the pose is in CAMERA coordinates,
    so it is itself view-dependent and could carry the same signal.

    At each (frame, joint) the available views are ranked by (a) ``unc``, (b) an error predicted
    from the pose alone by a leave-one-subject-out ridge, and each ranking is scored against the
    ranking by true error. Reported as mean Spearman over frames and as top-1 accuracy: how
    often the view each signal calls best really is the best.
    """
    subjects = sorted({s.subject for s in sequences})
    scores = {"unc_rho": [], "pose_rho": [], "unc_top1": [], "pose_top1": [], "chance_top1": []}
    for held in subjects:
        train = [s for s in sequences if s.subject != held]
        test = [s for s in sequences if s.subject == held]
        if not train or not test:
            continue
        bias = convention_bias(np.concatenate([s.delta for s in train]))
        err_tr = np.concatenate([corrected_error(s.delta, bias) for s in train])
        pose_tr = np.concatenate([s.pose for s in train]).reshape(len(err_tr), -1)
        models = [_ridge(pose_tr, err_tr[:, j], lam) for j in range(N_JOINTS)]

        cams = sorted({s.camera for s in test})
        if len(cams) < 3:
            continue
        by_cam = {s.camera: s for s in test}
        length = min(len(by_cam[c]) for c in cams)
        true_err = np.stack([corrected_error(by_cam[c].delta, bias)[:length] for c in cams])
        unc = np.stack([by_cam[c].uncertainty[:length] for c in cams])
        pose_err = np.stack([
            np.column_stack([models[j](by_cam[c].pose[:length].reshape(length, -1))
                             for j in range(N_JOINTS)]) for c in cams])

        # standardise within (view, joint) so a uniformly-worse camera cannot carry the score
        def z(a):
            return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)
        tz, uz, pz = z(true_err), z(unc), z(pose_err)
        # joint 0 is the root: its error is identically zero, so it carries no ranking signal
        tz, uz, pz = tz[:, :, 1:], uz[:, :, 1:], pz[:, :, 1:]
        scores["unc_rho"].append(spearman(uz.ravel(), tz.ravel()))
        scores["pose_rho"].append(spearman(pz.ravel(), tz.ravel()))
        best = np.argmin(tz, axis=0)
        scores["unc_top1"].append(float((np.argmin(uz, axis=0) == best).mean()))
        scores["pose_top1"].append(float((np.argmin(pz, axis=0) == best).mean()))
        scores["chance_top1"].append(1.0 / len(cams))
    return {k: float(np.mean(v)) for k, v in scores.items()} | {"n_subjects": len(subjects)}


def _ridge(x: np.ndarray, y: np.ndarray, lam: float):
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    gram = z.T @ z + lam * np.eye(z.shape[1])
    gram[-1, -1] -= lam
    weights = np.linalg.solve(gram, z.T @ y)

    def predict(xt: np.ndarray) -> np.ndarray:
        return np.hstack([(xt - mu) / sd, np.ones((len(xt), 1))]) @ weights

    return predict


def redundancy_test(
    sequences: list[Sequence],
    lambdas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> dict:
    """LOSO: can ``unc`` predict error beyond a per-joint constant, and beyond the pose itself?

    Four predictors, each scored as mean absolute error (mm) on the held-out subject, pooled
    over joints. Every model is fitted per joint, because cross-joint comparison is invalid
    (trap 2).

    * ``lookup``     -- training-fold mean error for this joint. **Zero features.**
    * ``unc``        -- ridge on the uncertainty alone.
    * ``pose``       -- ridge on the predicted root-relative pose (51 dims). The pose already
      signals difficulty: a self-occluded or extreme configuration is recognisable from
      coordinates alone.
    * ``pose+unc``   -- both. If this does not beat ``pose``, the channel is redundant and
      blind spot C dies the same way A and B did.
    """
    subjects = sorted({s.subject for s in sequences})
    results = {k: [] for k in ("lookup", "unc", "pose", "pose_unc")}
    for held in subjects:
        train = [s for s in sequences if s.subject != held]
        test = [s for s in sequences if s.subject == held]
        if not train or not test:
            continue
        bias = convention_bias(np.concatenate([s.delta for s in train]))
        err_tr = np.concatenate([corrected_error(s.delta, bias) for s in train])
        err_te = np.concatenate([corrected_error(s.delta, bias) for s in test])
        unc_tr = np.concatenate([s.uncertainty for s in train])
        unc_te = np.concatenate([s.uncertainty for s in test])
        pose_tr = np.concatenate([s.pose for s in train]).reshape(len(err_tr), -1)
        pose_te = np.concatenate([s.pose for s in test]).reshape(len(err_te), -1)

        fold = {k: [] for k in results}
        for j in range(N_JOINTS):
            y_tr, y_te = err_tr[:, j], err_te[:, j]
            fold["lookup"].append(np.abs(y_te - y_tr.mean()))
            for key, xtr, xte in (
                ("unc", unc_tr[:, j:j + 1], unc_te[:, j:j + 1]),
                ("pose", pose_tr, pose_te),
                ("pose_unc", np.hstack([pose_tr, unc_tr[:, j:j + 1]]),
                             np.hstack([pose_te, unc_te[:, j:j + 1]])),
            ):
                best = None
                for lam in lambdas:
                    pred = _ridge(xtr, y_tr, lam)(xte)
                    mae = float(np.mean(np.abs(y_te - pred)))
                    if best is None or mae < best:
                        best = mae
                fold[key].append(np.full(len(y_te), best))
        for key in results:
            results[key].append(float(np.mean(np.concatenate(fold[key]))))
    return {k: float(np.mean(v)) for k, v in results.items()} | {"n_subjects": len(subjects)}
