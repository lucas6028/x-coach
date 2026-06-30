"""Cross-model comparison of direct image->3D methods on Fit3D (NLF vs HMR2.0 vs ...).

The question is two-layered, and the layers need different metrics because the models use
different body conventions (NLF SMPL-24, HMR2.0 SMPL, Multi-HMR SMPL-X) whose joints sit at
systematically different anatomical points -- a per-joint bias that is NOT depth recovery:

* **Mechanism (headline, bias-tolerant):** do *all* direct-3D models show ``ez/exy`` near 1
  (depth error on par with in-plane, unlike 2D-lifting's ez>>exy) and recover the sagittal
  cues/verdicts that single-view 2D corrupts? If yes -> "direct image->3D recovers depth" is a
  general mechanism, not an NLF quirk. This is what closes the REHAB24 thread (which was
  confounded there by 75% detection). Lean on **pa_mpjpe** (procrustes; removes global
  rotation+scale), the **ez/exy pattern**, **rotation-invariant knee/hip angle** recovery, and
  the **debiased** verdict-flip (per-camera offset removed) -- never raw MPJPE rankings.

* **Ranking (secondary):** which model is best. Only claim a gap that survives debiasing AND
  holds per-subject; small raw-mm differences are likely joint-convention artifact.

All models are mapped to H36M-17 with the SAME ``resolve_lr`` + ``SMPL24_TO_H36M17`` (they share
the SMPL-family body-joint order), so the mapping artifact cancels in the cross-model compare.
Caveat carried per model: HMR2.0 regresses orientation in the CROP camera frame, so its
gravity-dependent cues (torso lean, depth axis) and ez/exy carry a crop-rotation term that the
rotation-invariant knee/hip angles and pa_mpjpe do not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d import decision_eval as dec
from src.fit3d import depth_eval as de
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics

# Cues whose angle is rotation-invariant (fair even for HMR2.0's crop frame) vs gravity-dependent.
ROTATION_INVARIANT = ("knee_angle", "hip_angle")
GRAVITY_DEPENDENT = ("torso_lean_deg", "depth_ratio")


def projection_biomech_error(action: str, split: str, root: Path) -> dict[str, float]:
    """Single-view 2D-projection cue error vs 3D truth (per-frame mean-abs) -- the 2D baseline,
    identical for every model. Mirrors scripts/fit3d/run_depth_eval.py."""
    acc: dict[str, list] = {k: [] for k in de.BIOMECH_KEYS}
    for subj in ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        j3d_m = ds.load_joints3d(split, subj, action, root)
        gm = frame_metrics(j3d_m, WORLD3D)
        for cam in ds.cameras(split, subj, root):
            cp = ds.read_cam_params(split, subj, cam, action, root)
            pm = frame_metrics(ds.project_world_to_image(j3d_m, cp), IMAGE2D)
            for k in de.BIOMECH_KEYS:
                acc[k].append(float(np.nanmean(np.abs(pm[k] - gm[k]))))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def compare(
    models: dict[str, str | Path],
    action: str = "squat",
    split: str = "train",
    source: str = "smpl3d",
    root: Path = ds.DEFAULT_FIT3D_ROOT,
) -> dict:
    """Run depth_eval + decision_eval per model and collect the bias-tolerant comparison metrics.

    ``models`` maps a display name to its ``--pred-root`` (each holding <subj>__<action>__<cam>.npz
    with an ``smpl3d`` SMPL-24 camera-frame key, mm).
    """
    out: dict = {"action": action, "split": split, "models": {}, "projection_2d": None}
    cues = tuple(dec.CUES)
    for name, pred_root in models.items():
        pred_root = Path(pred_root)
        dep = de.evaluate(pred_root, action=action, split=split, pred_units="mm", source=source)
        dcn = dec.run(pred_root, action=action, split=split, source=source)
        if dep["n"] == 0 or dcn["n_pairs"] == 0:
            out["models"][name] = {"n_seq": 0, "missing": True}
            continue
        agg = dep["aggregate"]
        exy = 0.5 * (agg["ex"] + agg["ey"])
        out["models"][name] = {
            "n_seq": dep["n"],
            "n_pairs": dcn["n_pairs"],
            "swap_lr": dep["swap_lr"],
            "mpjpe": agg["mpjpe"],
            "pa_mpjpe": agg["pa_mpjpe"],
            "ez": agg["ez"], "exy": exy, "ez_exy": agg["ez"] / exy if exy > 1e-9 else float("nan"),
            "cue_err": {c.name: agg[f"err_{c.name}"] for c in cues if f"err_{c.name}" in agg},
            # this model's debiased + raw verdict-flip per cue (swept thresholds, fair)
            "verdict_flip_deb": {c.name: dcn["per_cue"][c.name]["swept_flip"]["debnlf"] for c in cues},
            "verdict_flip_raw": {c.name: dcn["per_cue"][c.name]["swept_flip"]["nlf"] for c in cues},
            "knee_at_thr": dcn["per_cue"]["knee_angle"]["at_threshold"],
        }
        if out["projection_2d"] is None:
            out["projection_2d"] = {
                "cue_err": projection_biomech_error(action, split, root),
                "verdict_flip_deb": {c.name: dcn["per_cue"][c.name]["swept_flip"]["deb2d"] for c in cues},
            }
    return out


def format_comparison(result: dict) -> str:
    names = [n for n, m in result["models"].items() if not m.get("missing")]
    lines = [
        f"Fit3D direct image->3D model comparison -- action={result['action']} split={result['split']}",
        f"  models: {', '.join(names)}",
        "  Bias-tolerant metrics (joint-convention differs per body model): pa_mpjpe, ez/exy pattern,",
        "  rotation-invariant knee/hip cues, and DEBIASED verdict-flip. Raw MPJPE rankings are not safe.",
        "",
    ]

    # 1. Position + depth-axis pattern (the mechanism signal).
    lines.append("  [DEPTH PATTERN] position error vs mocap GT (mm, root-relative) + depth axis:")
    lines.append(f"  {'model':<12}{'n':>5}{'MPJPE':>8}{'PA-MPJPE':>10}{'ez(depth)':>11}{'exy':>8}{'ez/exy':>8}")
    lines.append("  " + "-" * 62)
    for n in names:
        m = result["models"][n]
        lines.append(f"  {n:<12}{m['n_seq']:>5}{m['mpjpe']:>8.1f}{m['pa_mpjpe']:>10.1f}"
                     f"{m['ez']:>11.1f}{m['exy']:>8.1f}{m['ez_exy']:>8.2f}")
    lines.append("  (ez/exy ~1 => depth on par with in-plane = recovered; >>1 => 2D-lifting-like depth failure)")

    # 2. Cue recovery vs the single-view 2D baseline (knee/hip are rotation-invariant = fair for all).
    p2 = result["projection_2d"]
    lines.append("")
    lines.append("  [CUE RECOVERY] per-frame cue error vs 3D truth (deg / ratio); knee,hip = rotation-invariant:")
    head = f"  {'cue':<16}{'2D-view':>9}" + "".join(f"{n:>11}" for n in names)
    lines.append(head)
    for c in de.BIOMECH_KEYS:
        inv = "*" if c in ROTATION_INVARIANT else " "
        row = f"  {c:<15}{inv}{p2['cue_err'][c]:>9.2f}"
        for n in names:
            row += f"{result['models'][n]['cue_err'].get(c, float('nan')):>11.2f}"
        lines.append(row)
    lines.append("  (* rotation-invariant: fair even for HMR2.0's crop frame; others carry a crop-rotation term)")

    # 3. Debiased verdict-flip (needs-3D map), fair across models.
    lines.append("")
    lines.append("  [VERDICT FLIP] debiased verdict-flip over swept thresholds (lower=better; oracle-calib both):")
    lines.append(f"  {'cue':<16}{'deb-2D':>9}" + "".join(f"{n:>11}" for n in names))
    for c in [cc.name for cc in dec.CUES]:
        row = f"  {c:<16}{p2['verdict_flip_deb'][c] * 100:>8.0f}%"
        for n in names:
            row += f"{result['models'][n]['verdict_flip_deb'][c] * 100:>10.0f}%"
        lines.append(row)
    return "\n".join(lines)
