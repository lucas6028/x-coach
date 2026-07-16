"""Experiment 2 -- how view-dependent are the 2D squat-rule readings?

For every squat repetition we read each biomechanical cue two ways:

* from the **view-invariant 3D ground truth** (the right answer), and
* from the **2D projection in each of the 4 cameras** (what a single-camera 2D
  pipeline would actually see).

Both use the identical formulas in :mod:`src.fit3d.biomech`, so the gap is pure
projection distortion. We then aggregate, per cue:

* ``mae`` / ``bias`` of each camera's reading vs truth (same units: degrees or
  dimensionless ratios),
* ``pearson`` -- does the 2D reading still *rank* reps like the truth does,
* ``cross_view_std`` -- how much the reading swings between the 4 cameras for the
  *same* rep, and ``noise_to_signal`` = that swing divided by the real
  between-rep signal. ``>= 1`` means the camera you happened to use matters as
  much as the athlete's actual form -> that cue needs 3D.

The output is a per-cue verdict (view-robust / view-sensitive / view-corrupted)
that feeds straight back into ``pose_rule_detector`` threshold calibration and the
``view_estimation`` confidence gating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics, rep_summary

METRICS = tuple(frame_metrics(np.zeros((1, ds.NUM_JOINTS, 3)), WORLD3D).keys())

# Verdict thresholds (heuristic, documented in the module docstring).
ROBUST_PEARSON, ROBUST_NTS = 0.80, 0.50
CORRUPT_PEARSON, CORRUPT_NTS = 0.50, 1.00


@dataclass
class RepRecord:
    subject: str
    rep_index: int
    truth: dict[str, float]
    views: dict[str, dict[str, float]]  # camera -> metric -> value


def collect_records(
    action: str = "squat",
    split: str = "train",
    subjs: list[str] | None = None,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    min_rep_frames: int = 5,
) -> tuple[list[RepRecord], list[str]]:
    """Per-rep 3D-truth and per-camera 2D readings for one action."""
    records: list[RepRecord] = []
    camera_set: list[str] = []
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        j3d = ds.load_joints3d(split, subj, action, root)
        rep_ann = ds.load_rep_ann(split, subj, root).get(action)
        if not rep_ann:
            continue
        segments = ds.rep_segments(rep_ann)
        cams = ds.cameras(split, subj, root)
        if not camera_set:
            camera_set = cams
        # Pre-project the whole sequence once per camera.
        proj = {
            cam: ds.project_world_to_image(j3d, ds.read_cam_params(split, subj, cam, action, root))
            for cam in cams
        }
        for rep_index, (start, end) in enumerate(segments):
            if end - start < min_rep_frames:
                continue
            truth = rep_summary(j3d, WORLD3D, start, end)
            views = {cam: rep_summary(proj[cam], IMAGE2D, start, end) for cam in cams}
            records.append(RepRecord(subj, rep_index, truth, views))
    return records, camera_set


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _verdict(pearson: float, nts: float) -> str:
    if np.isfinite(pearson) and np.isfinite(nts) and pearson >= ROBUST_PEARSON and nts <= ROBUST_NTS:
        return "view-robust"
    if (np.isfinite(pearson) and pearson < CORRUPT_PEARSON) or (np.isfinite(nts) and nts >= CORRUPT_NTS):
        return "view-corrupted"
    return "view-sensitive"


def analyse(records: list[RepRecord], cameras: list[str]) -> dict:
    """Aggregate per-rep records into per-metric, per-camera statistics + verdicts."""
    per_metric: dict[str, dict] = {}
    for metric in METRICS:
        truth = np.array([r.truth[metric] for r in records], dtype=np.float64)
        truth_std = float(np.nanstd(truth))

        per_camera = {}
        pooled_err, pooled_truth, pooled_view = [], [], []
        view_matrix = []  # (n_reps, n_cameras) of the 2D reading
        for cam in cameras:
            view = np.array([r.views[cam][metric] for r in records], dtype=np.float64)
            view_matrix.append(view)
            err = view - truth
            per_camera[cam] = {
                "mae": float(np.nanmean(np.abs(err))),
                "bias": float(np.nanmean(err)),
                "pearson": _pearson(view, truth),
                "n": int(np.isfinite(err).sum()),
            }
            pooled_err.append(err)
            pooled_truth.append(truth)
            pooled_view.append(view)

        view_matrix = np.vstack(view_matrix).T  # (n_reps, n_cameras)
        cross_view_std = float(np.nanmean(np.nanstd(view_matrix, axis=1)))
        nts = cross_view_std / truth_std if truth_std > 1e-9 else float("nan")
        pooled_err = np.concatenate(pooled_err)
        pooled_pearson = _pearson(np.concatenate(pooled_view), np.concatenate(pooled_truth))

        per_metric[metric] = {
            "truth_std": truth_std,
            "truth_mean": float(np.nanmean(truth)),
            "per_camera": per_camera,
            "pooled_mae": float(np.nanmean(np.abs(pooled_err))),
            "pooled_bias": float(np.nanmean(pooled_err)),
            "pooled_pearson": pooled_pearson,
            "cross_view_std": cross_view_std,
            "noise_to_signal": nts,
            "verdict": _verdict(pooled_pearson, nts),
        }
    return per_metric


def run(
    action: str = "squat",
    split: str = "train",
    subjs: list[str] | None = None,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
) -> dict:
    records, cameras = collect_records(action, split, subjs, root)
    return {
        "action": action,
        "split": split,
        "cameras": cameras,
        "n_subjects": len({r.subject for r in records}),
        "n_reps": len(records),
        "per_metric": analyse(records, cameras) if records else {},
    }


def format_report(result: dict) -> str:
    lines = [
        f"Fit3D view-dependence -- action={result['action']} split={result['split']}",
        f"  {result['n_reps']} reps from {result['n_subjects']} subjects, "
        f"{len(result['cameras'])} cameras: {', '.join(result['cameras'])}",
        "",
        f"  {'metric':<17}{'truth(mean±sd)':>18}{'pooledMAE':>11}{'bias':>8}"
        f"{'r':>7}{'viewStd':>9}{'noise/sig':>11}  verdict",
        "  " + "-" * 96,
    ]
    for metric, m in result["per_metric"].items():
        lines.append(
            f"  {metric:<17}"
            f"{m['truth_mean']:>9.1f}±{m['truth_std']:<7.1f}"
            f"{m['pooled_mae']:>11.2f}{m['pooled_bias']:>8.2f}"
            f"{m['pooled_pearson']:>7.2f}{m['cross_view_std']:>9.2f}"
            f"{m['noise_to_signal']:>11.2f}  {m['verdict']}"
        )
    lines.append("")
    lines.append("  per-camera MAE (vs 3D truth):")
    lines.append(f"  {'metric':<17}" + "".join(f"{c:>12}" for c in result["cameras"]))
    for metric, m in result["per_metric"].items():
        lines.append(
            f"  {metric:<17}" + "".join(f"{m['per_camera'][c]['mae']:>12.2f}" for c in result["cameras"])
        )
    return "\n".join(lines)
