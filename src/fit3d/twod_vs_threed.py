"""2D-vs-3D-vs-mocap decomposition: where does a single-view 2D pipeline's cue error come from?

Places three "readings" of each squat cue side by side against the mocap 3D truth:

* **real-2D**  -- a genuine 2D detector (RTMPose) on the video (``src/fit3d/twod_baseline.py``),
* **mocap-2D** -- the GT 3D *projected* to the image = a **perfect** detector (zero detector error),
* **3D**       -- direct image->3D models (NLF / HMR2.0 / Multi-HMR), via ``depth_eval``.

The point: a 2D pipeline's total cue error decomposes as

    real-2D error  =  detector error (real-2D - mocap-2D)  +  projection error (mocap-2D - GT-3D)

and the two components behave oppositely by cue. On the **depth/flexion** cues (knee, hip,
hip-below-knee) the *projection* term already dominates -- even a perfect detector (mocap-2D)
can't recover them -- so a better 2D detector cannot help and you need 3D. On the **frontal-plane**
cues (valgus, torso) mocap-2D is fine, so the *detector* term is the whole story and a better 2D
detector is what helps (3D is not needed). This is the direct mocap-GT version of "depth is the
bottleneck, not 2D accuracy".

real-2D and mocap-2D cue errors are computed on the **same RTMPose-inferred frames** so their
difference is exactly the detector term; the 3D errors come from ``depth_eval`` over their own
frames (per-frame cue error is unbiased under subsampling, so the means are comparable).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d import depth_eval as de
from src.fit3d import twod_baseline as tb
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics

CUES = ("knee_angle", "hip_angle", "torso_lean_deg", "depth_ratio", "knee_width_ratio")


def cue_verdict(real: float, mocap: float, best3d: float | None) -> str:
    """Data-driven read of where a cue's 2D error lives and what fixes it.

    'need-3D'   -- the perfect detector (mocap-2D) already fails and 3D materially beats it, so the
                   error is projection geometry a better 2D detector can't remove.
    'better-2D' -- the detector term (real-2D - mocap-2D) is a big share, so a better 2D detector helps
                   (and, where 3D is worse/absent, 3D is not the answer).
    'mixed'     -- neither dominates / nothing helps much (low-signal cue)."""
    detector = abs(real - mocap)
    has3d = best3d is not None and np.isfinite(best3d)
    if has3d and best3d < 0.7 * mocap and mocap > detector:
        return "need-3D"
    if detector >= 0.5 * mocap:
        return "better-2D"
    return "mixed"


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(np.abs(a - b)))


def twod_errors(action: str, rtmpose_root: Path, split: str = "train",
                subjs: list[str] | None = None, root: Path = ds.DEFAULT_FIT3D_ROOT) -> dict:
    """Per-frame cue error of real-2D (RTMPose) and mocap-2D (GT projection) vs GT-3D truth, on the
    RTMPose-inferred frames. Also the raw keypoint MAE (px) of real-2D vs mocap-2D = detector error."""
    acc: dict[str, dict[str, list]] = {"real2d": defaultdict(list), "mocap2d": defaultdict(list)}
    per_subj: dict[str, dict] = {}
    kp_mae, n_frames, n_seq = [], 0, 0
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        subj_acc = {"real2d": defaultdict(list), "mocap2d": defaultdict(list)}
        for cam in ds.cameras(split, subj, root):
            npz = rtmpose_root / tb.pred_npz_name(subj, action, cam)
            if not npz.exists():
                continue
            j3d = ds.load_joints3d(split, subj, action, root)     # (F,25,3) m
            cp = ds.read_cam_params(split, subj, cam, action, root)
            proj = ds.project_world_to_image(j3d, cp)             # mocap-2D px
            rtm = tb.load_rtmpose_2d(npz)                          # real-2D px
            n = min(len(j3d), len(proj), len(rtm))
            finite = np.isfinite(rtm[:n][:, tb.BIOMECH_JOINTS, :]).all(axis=(1, 2))
            sel = np.where(finite)[0]
            if sel.size == 0:
                continue
            n_seq += 1; n_frames += int(sel.size)
            gt = frame_metrics(j3d[sel], WORLD3D)
            mo = frame_metrics(proj[sel], IMAGE2D)
            rl = frame_metrics(rtm[sel], IMAGE2D)
            kp_mae.append(float(np.nanmean(np.linalg.norm(rtm[sel] - proj[sel], axis=2))))
            for k in CUES:
                for arm, cues in (("real2d", rl), ("mocap2d", mo)):
                    e = _mae(cues[k], gt[k])
                    acc[arm][k].append(e); subj_acc[arm][k].append(e)
        if subj_acc["real2d"]:
            per_subj[subj] = {arm: {k: float(np.nanmean(subj_acc[arm][k])) for k in CUES}
                              for arm in ("real2d", "mocap2d")}
    agg = {arm: {k: float(np.nanmean(acc[arm][k])) if acc[arm][k] else float("nan") for k in CUES}
           for arm in ("real2d", "mocap2d")}
    return {"aggregate": agg, "per_subject": per_subj, "kp_mae_px": float(np.nanmean(kp_mae)) if kp_mae else float("nan"),
            "n_seq": n_seq, "n_frames": n_frames}


def compare(action: str, rtmpose_root: Path, models_3d: dict[str, str | Path],
            split: str = "train", source: str = "smpl3d",
            subjs: list[str] | None = None, root: Path = ds.DEFAULT_FIT3D_ROOT) -> dict:
    """2D arms (real + mocap) + each 3D model's per-cue error vs GT-3D truth."""
    two = twod_errors(action, rtmpose_root, split, subjs, root)
    three: dict[str, dict] = {}
    for name, pred_root in models_3d.items():
        r = de.evaluate(Path(pred_root), action=action, split=split, pred_units="mm", source=source, subjs=subjs)
        three[name] = {k: r["aggregate"].get(f"err_{k}", float("nan")) for k in CUES} if r["n"] else None
    return {"action": action, "split": split, "twod": two, "threed": three}


def format_comparison(result: dict) -> str:
    two, three = result["twod"], result["threed"]
    names3d = [n for n, v in three.items() if v]
    a = two["aggregate"]
    lines = [
        f"Fit3D 2D-vs-3D cue error decomposition -- action={result['action']} split={result['split']}",
        f"  real-2D = RTMPose ({two['n_seq']} seq, {two['n_frames']} frames); mocap-2D = GT projected (perfect detector);",
        f"  3D = {', '.join(names3d)}. All are cue-reading error vs mocap-3D truth (deg / ratio).",
        f"  RTMPose keypoint MAE vs GT-projected: {two['kp_mae_px']:.0f} px (the raw detector error).",
        "",
        f"  {'cue':<16}{'real-2D':>9}{'mocap-2D':>10}" + "".join(f"{n:>10}" for n in names3d)
        + f"{'detector':>10}{'verdict':>11}",
        "  " + "-" * (36 + 10 * len(names3d) + 21),
    ]
    for k in CUES:
        real, mocap = a["real2d"][k], a["mocap2d"][k]
        detector = real - mocap
        threes = [three[n][k] for n in names3d if np.isfinite(three[n][k])]
        best3d = min(threes) if threes else None
        cells = "".join((f"{three[n][k]:>10.2f}" if np.isfinite(three[n][k]) else f"{'--':>10}") for n in names3d)
        row = f"  {k:<16}{real:>9.2f}{mocap:>10.2f}{cells}{detector:>+10.2f}{cue_verdict(real, mocap, best3d):>11}"
        lines.append(row)
    lines.append("")
    lines.append("  need-3D: even the PERFECT detector (mocap-2D) fails and 3D fixes it -> projection geometry,")
    lines.append("           a better 2D detector cannot help. better-2D: mocap-2D is fine, the error is the")
    lines.append("           detector -> a better 2D detector helps, 3D not needed (valgus 3D omitted; exp3: 3D worse).")

    # per-subject spread of the real-2D vs mocap-2D gap on the headline depth cue (knee)
    ps = two["per_subject"]
    if ps:
        gaps = [ps[s]["real2d"]["knee_angle"] - ps[s]["mocap2d"]["knee_angle"] for s in ps]
        lines.append("")
        lines.append(f"  per-subject knee detector-error (real-2D - mocap-2D): "
                     f"{np.mean(gaps):+.1f} +/- {np.std(gaps):.1f} deg over {len(gaps)} subjects")
    return "\n".join(lines)
