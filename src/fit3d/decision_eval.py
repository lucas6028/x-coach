"""Experiment 3 -- does the perception layer flip the *coaching verdict*?

Experiments 1 and 2 are stated in cue units (a knee angle is off by N degrees). A
coaching app does not output degrees; it outputs a **pass/fault verdict** ("you didn't
reach parallel", "knees caved"). This experiment translates the cue findings into that
verdict and asks the decision-level question directly:

    Reading squat depth from a single 2D camera, how often is the verdict WRONG versus
    the mocap-truth verdict -- and does direct image->3D (NLF) fix it?

For every rep x camera we compute each cue three ways with the identical
:func:`src.fit3d.biomech.rep_summary` (the per-rep *extreme* -- the bottom-of-squat
reading a coach actually judges), threshold it into pass/fault, and compare to the
verdict from the view-invariant 3D ground truth:

* ``gt``      -- mocap 3D truth (defines the correct verdict),
* ``view2d``  -- the single-camera 2D projection (what a 2D pipeline deploys today),
* ``nlf``     -- NLF monocular 3D (the proposed fix).

The 2D arm is **three-valued** so the result is not inflated by *correctable* bias.
Experiment 2 found constant per-view offsets (knee +41 deg, torso -22 deg) that one
calibration constant removes without any 3D. So we also report ``view2d`` after
subtracting each camera's oracle mean offset vs GT -- the upper bound on what per-view
calibration can buy. The same debiasing is applied to ``nlf`` (it has its own residual
cue bias), so the only fair comparisons are **raw-2D vs raw-NLF** and
**debiased-2D vs debiased-NLF**. The deciding number for the depth-bottleneck thesis:
after oracle debiasing, is the knee/depth verdict-flip rate for 2D still well above NLF's?
If yes, even perfect calibration cannot fix single-view 2D and you need direct 3D.

Framing caveat: this is **verdict fidelity vs mocap truth**, not accuracy vs human
correctness labels (that is the n-limited REHAB24 thread). The Fit3D population is all
competent reps, so the meaningful error at a real threshold is the **false-alarm rate**
(falsely failing a good rep). The 4 cameras of one rep are not independent, so we report
descriptive rates and per-subject spread, not a p-value over the pooled readings.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d import depth_eval as de
from src.fit3d.biomech import IMAGE2D, WORLD3D, rep_summary


@dataclass(frozen=True)
class CueSpec:
    """A cue with a coaching verdict: ``fault_when`` 'high' (value above threshold is a
    fault, e.g. knee never bent below parallel) or 'low' (below, e.g. knees caved in).
    ``canonical`` is the biomechanical threshold (parallel etc.); ``None`` -> verdicts use
    the GT median split only (a ranking robustness check, not a real fault threshold)."""

    name: str
    fault_when: str
    canonical: float | None
    unit: str


# Primary depth cues carry a real biomechanical threshold (parallel). The others have no
# clean universal cutoff here, so they are reported on the median split as a robustness
# check (which mostly re-expresses ranking, i.e. experiment-2 MAE, not fault detection).
CUES: tuple[CueSpec, ...] = (
    CueSpec("knee_angle", "high", 90.0, "deg"),       # didn't reach parallel: deepest knee > 90 deg
    # depth_ratio uses the hip JOINT CENTRE, which sits above the knee even at parallel ("parallel"
    # is defined on the hip CREASE), so 0.0 is not the parallel cutoff for this skeleton -- it labels
    # every rep a fault. No universal threshold -> median split (ranking check) only.
    CueSpec("depth_ratio", "high", None, "ratio"),
    CueSpec("torso_lean_deg", "high", None, "deg"),   # excessive forward lean (no universal cutoff)
    CueSpec("hip_angle", "high", None, "deg"),
    CueSpec("knee_width_ratio", "low", None, "ratio"),  # valgus: knees narrower than ankles
)


def nlf_world_points(pred_cam_h36m17: np.ndarray, cam_params: dict) -> np.ndarray:
    """NLF camera-frame H36M-17 (F,17,3) -> world-aligned (F,25,3), padded for biomech indexing.

    Mirrors :func:`src.fit3d.depth_eval.biomech_error`: rotate by the GT camera rotation so
    the gravity-dependent cues (torso lean, hip-below-knee) read against true vertical, then
    zero-pad to 25 joints (only the H36M-17 core is used by the cue formulas)."""
    R = cam_params["extrinsics"]["R"]
    pred_world = pred_cam_h36m17 @ R
    out = np.zeros((pred_world.shape[0], ds.NUM_JOINTS, 3))
    out[:, :17, :] = pred_world
    return out


def mask_to_stride(points: np.ndarray, stride: int) -> np.ndarray:
    """NaN out every frame not on ``stride`` (keeping 0, stride, 2*stride, ...).

    Used to put an every-frame model on the same frame grid as a subsampled one. The verdict
    reduces a rep window to its *extreme* (nanmin/nanmax), which is sample-count biased -- fewer
    frames can only make an extreme less extreme -- so a dense-vs-sparse verdict comparison is
    confounded unless both arms see the same frames. Sparse Fit3D predictions start at frame 0
    (strides: Multi-HMR 6; MeTRAbs/MediaPipe/RTMPose 15), so this reproduces their grid exactly.
    """
    if stride <= 1:
        return points
    out = np.array(points, dtype=np.float64, copy=True)
    keep = np.zeros(len(out), dtype=bool)
    keep[::stride] = True
    out[~keep] = np.nan
    return out


def collect_records(
    pred_root: Path,
    action: str = "squat",
    split: str = "train",
    source: str = "smpl3d",
    subjs: list[str] | None = None,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    min_rep_frames: int = 5,
    frame_stride: int = 1,
) -> tuple[list[dict], list[str]]:
    """Per (subject, rep, camera): the GT / 2D-projection / NLF cue readings at the rep extreme.

    One record per (rep, camera); the GT reading repeats across cameras (it is view-invariant)
    -- matching experiment 2's rep x camera pairing.

    ``frame_stride`` > 1 subsamples the **model arm only** (see :func:`mask_to_stride`); GT and
    the 2D projection stay at full rate, matching how the sparse-model tables were built.
    """
    records: list[dict] = []
    cams_seen: list[str] = []
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        rep_ann = ds.load_rep_ann(split, subj, root).get(action)
        if not rep_ann:
            continue
        segments = ds.rep_segments(rep_ann)
        j3d_m = ds.load_joints3d(split, subj, action, root)   # world metres (cues are scale-free)
        j3d_mm = j3d_m * 1000.0                                 # for resolve_lr vs NLF mm
        for cam in ds.cameras(split, subj, root):
            npz = pred_root / de.pred_npz_name(subj, action, cam)
            if not npz.exists():
                continue
            cp = ds.read_cam_params(split, subj, cam, action, root)
            proj = ds.project_world_to_image(j3d_m, cp)         # 2D arm
            gt_cam = de.gt_in_camera_frame(j3d_mm, cp)
            pred_cam, _info = de.load_prediction_h36m17(npz, gt_cam[:, :17, :], source=source)
            nlf_world = mask_to_stride(nlf_world_points(pred_cam, cp), frame_stride)  # NLF arm
            n = min(len(j3d_m), len(proj), len(nlf_world))
            if cam not in cams_seen:
                cams_seen.append(cam)
            for rep_index, (start, end) in enumerate(segments):
                e = min(end, n)
                if e - start < min_rep_frames:
                    continue
                records.append({
                    "subject": subj,
                    "rep_index": rep_index,
                    "camera": cam,
                    "gt": rep_summary(j3d_m, WORLD3D, start, e),
                    "view2d": rep_summary(proj, IMAGE2D, start, e),
                    "nlf": rep_summary(nlf_world, WORLD3D, start, e),
                })
    return records, cams_seen


def _debias(reading: np.ndarray, gt: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Subtract each camera's oracle mean offset (reading - gt) -- the per-view calibration upper bound."""
    out = reading.astype(np.float64).copy()
    offsets: dict[str, float] = {}
    for c in np.unique(cam):
        sel = cam == c
        m = sel & np.isfinite(reading) & np.isfinite(gt)
        off = float(np.nanmean(reading[m] - gt[m])) if m.any() else 0.0
        offsets[str(c)] = off
        out[sel] = reading[sel] - off
    return out, offsets


def _verdict_metrics(gt: np.ndarray, reading: np.ndarray, thr: float, fault_when: str) -> dict:
    """Pass/fault verdict agreement of ``reading`` vs GT at threshold ``thr``.

    flip        -- verdict disagrees with GT,
    false_alarm -- GT says OK but reading flags a fault (the meaningful error on a competent
                   population: falsely failing a good rep),
    miss        -- GT says fault but reading passes it (undetected fault).
    """
    mask = np.isfinite(gt) & np.isfinite(reading)
    g, r = gt[mask], reading[mask]
    if g.size == 0:
        return {"flip": float("nan"), "false_alarm": float("nan"), "miss": float("nan"),
                "true_fault_rate": float("nan"), "n": 0}
    if fault_when == "high":
        true_f, pred_f = g > thr, r > thr
    else:
        true_f, pred_f = g < thr, r < thr
    ok, fault = ~true_f, true_f
    return {
        "flip": float(np.mean(true_f != pred_f)),
        "false_alarm": float(np.mean(pred_f[ok])) if ok.any() else float("nan"),
        "miss": float(np.mean(~pred_f[fault])) if fault.any() else float("nan"),
        "true_fault_rate": float(np.mean(true_f)),
        "n": int(g.size),
    }


def _swept_flip(gt: np.ndarray, reading: np.ndarray, fault_when: str,
                lo: float = 20.0, hi: float = 80.0, steps: int = 13) -> float:
    """Mean verdict-flip rate over thresholds swept across the central GT range.

    Threshold-agnostic robustness check. NOTE: a swept/median split on a competent
    population measures *ranking* fidelity (does the reading order reps like the truth),
    which restates experiment-2 MAE -- it is not fault detection. The canonical-threshold
    false-alarm rate is the fault-detection metric."""
    finite = np.isfinite(gt)
    if finite.sum() < 5:
        return float("nan")
    thrs = np.nanpercentile(gt[finite], np.linspace(lo, hi, steps))
    flips = [_verdict_metrics(gt, reading, float(t), fault_when)["flip"] for t in thrs]
    return float(np.nanmean(flips))


def _per_subject_flip(gt: np.ndarray, reading: np.ndarray, subj: np.ndarray,
                      thr: float, fault_when: str) -> dict:
    """Verdict-flip rate computed per subject then summarised -- honest spread given that the
    4 cameras of one rep are not independent samples."""
    vals: dict[str, float] = {}
    for s in np.unique(subj):
        m = subj == s
        vals[str(s)] = _verdict_metrics(gt[m], reading[m], thr, fault_when)["flip"]
    arr = np.array([v for v in vals.values() if np.isfinite(v)], dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "n_subjects": int(arr.size),
        "per_subject": vals,
    }


def analyse(records: list[dict]) -> dict:
    """Per-cue verdict fidelity for the four readouts (raw/debiased 2D, raw/debiased NLF)."""
    subj = np.array([r["subject"] for r in records])
    cam = np.array([r["camera"] for r in records])
    out: dict[str, dict] = {}
    for cue in CUES:
        gt = np.array([r["gt"][cue.name] for r in records], dtype=np.float64)
        v2 = np.array([r["view2d"][cue.name] for r in records], dtype=np.float64)
        nl = np.array([r["nlf"][cue.name] for r in records], dtype=np.float64)
        v2_deb, off2 = _debias(v2, gt, cam)
        nl_deb, offn = _debias(nl, gt, cam)
        arms = {"raw2d": v2, "deb2d": v2_deb, "nlf": nl, "debnlf": nl_deb}

        thr = cue.canonical if cue.canonical is not None else float(np.nanmedian(gt))
        cue_out: dict = {
            "fault_when": cue.fault_when,
            "unit": cue.unit,
            "canonical_threshold": cue.canonical,
            "threshold_used": thr,
            "threshold_is_canonical": cue.canonical is not None,
            "gt_mean": float(np.nanmean(gt)),
            "gt_std": float(np.nanstd(gt)),
            "n_pairs": int(np.isfinite(gt).sum()),
            "per_camera_offset": {"raw2d": off2, "nlf": offn},
            "at_threshold": {name: _verdict_metrics(gt, arm, thr, cue.fault_when)
                             for name, arm in arms.items()},
            "swept_flip": {name: _swept_flip(gt, arm, cue.fault_when) for name, arm in arms.items()},
            "per_subject_flip": {
                "raw2d": _per_subject_flip(gt, v2, subj, thr, cue.fault_when),
                "deb2d": _per_subject_flip(gt, v2_deb, subj, thr, cue.fault_when),
                "nlf": _per_subject_flip(gt, nl, subj, thr, cue.fault_when),
                "debnlf": _per_subject_flip(gt, nl_deb, subj, thr, cue.fault_when),
            },
        }
        out[cue.name] = cue_out
    return out


def run(
    pred_root: Path,
    action: str = "squat",
    split: str = "train",
    source: str = "smpl3d",
    subjs: list[str] | None = None,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    frame_stride: int = 1,
) -> dict:
    records, cams = collect_records(pred_root, action, split, source, subjs, root,
                                    frame_stride=frame_stride)
    return {
        "action": action,
        "split": split,
        "source": source,
        "pred_root": str(pred_root),
        "frame_stride": frame_stride,
        "cameras": cams,
        "n_subjects": len({r["subject"] for r in records}),
        "n_reps": len({(r["subject"], r["rep_index"]) for r in records}),
        "n_pairs": len(records),
        "per_cue": analyse(records) if records else {},
    }


def needs_3d_verdict(deb2d_flip: float, debnlf_flip: float, margin: float = 0.05) -> str:
    """Cross-cue summary on the FAIR debiased-vs-debiased flip rates (oracle calibration applied
    to both arms). 'needs-3d' = even calibrated 2D flips the verdict materially more than NLF."""
    if not (np.isfinite(deb2d_flip) and np.isfinite(debnlf_flip)):
        return "n/a"
    if deb2d_flip > debnlf_flip + margin:
        return "needs-3D"
    if debnlf_flip > deb2d_flip + margin:
        return "2D-better"
    return "tie"


def format_report(result: dict) -> str:
    lines = [
        f"Fit3D verdict fidelity -- action={result['action']} split={result['split']} source={result['source']}",
        f"  {result['n_pairs']} rep x camera readings "
        f"({result['n_reps']} reps x {len(result['cameras'])} cameras, {result['n_subjects']} subjects)",
        "  Verdict fidelity vs mocap truth. 'debiased' = oracle per-view mean-offset removed (the cap on",
        "  what per-view calibration can buy); applied to BOTH arms, so the fair pairs are raw-2D vs raw-NLF",
        "  and deb-2D vs deb-NLF. Population is all competent reps -> false-alarm (falsely failing a good rep)",
        "  is the meaningful error.",
        "",
    ]

    # --- Headline: the depth verdict (knee angle at parallel), the geometrically-corrupted cue ---
    km = result["per_cue"].get("knee_angle", {})
    if km.get("threshold_is_canonical"):
        m = km
        t = m["at_threshold"]
        prevalence = t["raw2d"]["true_fault_rate"]
        lines.append(f"  [DEPTH VERDICT] knee angle at parallel ({m['threshold_used']:.0f} deg), "
                     f"true-fault prevalence {prevalence*100:.0f}%:")
        if prevalence > 0.9:
            # The true-OK class is near-empty (e.g. hip-hinge deadlift: knees never bend to parallel),
            # so the false-alarm rate -- the meaningful error here -- has too few reps to be reliable.
            # Knee depth simply isn't this movement's cue; read the needs-3D map below, not this box.
            lines.append("  (degenerate: almost no good-rep reference -> knee depth is not this movement's"
                         " cue; see the needs-3D map below, not this box)")
        else:
            lines.append(f"  {'readout':<22}{'flip':>8}{'false-alarm':>14}{'miss':>8}")
            lines.append("  " + "-" * 52)
            labels = [("raw2d", "raw 2D (deployed)"), ("deb2d", "oracle-calib 2D"),
                      ("nlf", "raw NLF"), ("debnlf", "oracle-calib NLF")]
            for key, lbl in labels:
                a = t[key]
                fa = "n/a" if not np.isfinite(a["false_alarm"]) else f"{a['false_alarm']*100:.0f}%"
                ms = "n/a" if not np.isfinite(a["miss"]) else f"{a['miss']*100:.0f}%"
                lines.append(f"  {lbl:<22}{a['flip']*100:>7.0f}%{fa:>14}{ms:>8}")
            ps = m["per_subject_flip"]
            lines.append(f"  => fair (calibrated both): 2D flips depth verdict {t['deb2d']['flip']*100:.0f}% of reps, "
                         f"NLF {t['debnlf']['flip']*100:.0f}%.")
            lines.append(f"     per-subject calibrated flip: 2D {ps['deb2d']['mean']*100:.0f}+/-{ps['deb2d']['std']*100:.0f}%"
                         f" vs NLF {ps['debnlf']['mean']*100:.0f}+/-{ps['debnlf']['std']*100:.0f}% "
                         f"(raw 2D {ps['raw2d']['mean']*100:.0f}+/-{ps['raw2d']['std']*100:.0f}%).")
            lines.append(f"     calibrated 2D trades false-alarms for misses ({t['deb2d']['false_alarm']*100:.0f}% FA "
                         f"+ {t['deb2d']['miss']*100:.0f}% miss); one per-view offset can't fix both -- NLF gets both low.")
            lines.append(f"     as deployed (no calibration) 2D false-fails {t['raw2d']['false_alarm']*100:.0f}% of good squats.")
        lines.append("")

    # --- Cross-cue needs-3D map (fair, threshold-agnostic: debiased swept flip on both arms) ---
    lines.append("  [NEEDS-3D MAP] verdict-flip over swept thresholds, oracle-calibrated both arms:")
    lines.append(f"  {'cue':<18}{'deb-2D flip':>13}{'deb-NLF flip':>14}{'verdict':>12}")
    lines.append("  " + "-" * 57)
    for cue in CUES:
        s = result["per_cue"][cue.name]["swept_flip"]
        v = needs_3d_verdict(s["deb2d"], s["debnlf"])
        lines.append(f"  {cue.name:<18}{s['deb2d']*100:>12.0f}%{s['debnlf']*100:>13.0f}%{v:>12}")
    lines.append("  (depth/hip flexion -> direct 3D; torso-lean & valgus are frontal-plane cues 2D already sees)")

    # --- Per-subject spread (honest n: the 4 views of a rep are not independent) ---
    lines.append("")
    lines.append("  per-subject verdict-flip rate (mean +/- sd over subjects; raw readouts):")
    lines.append(f"  {'cue':<18}{'2D flip':>16}{'NLF flip':>16}")
    for cue in CUES:
        m = result["per_cue"][cue.name]
        ps2, psn = m["per_subject_flip"]["raw2d"], m["per_subject_flip"]["nlf"]
        tag = "" if m["threshold_is_canonical"] else "  (median split)"
        lines.append(f"  {cue.name:<18}{ps2['mean']*100:>11.0f}+/-{ps2['std']*100:<4.0f}"
                     f"{psn['mean']*100:>11.0f}+/-{psn['std']*100:<4.0f}{tag}")
    return "\n".join(lines)


def to_json(result: dict) -> dict:
    """Compact payload for persistence (drops nothing material; full rows aren't stored)."""
    return result
