"""Subject-level significance of the Fit3D 2D-vs-NLF verdict-flip comparisons (2026-09-26).

Exploratory, post hoc (no pre-registered plan). Run from the repo root:

    .venv\\Scripts\\python.exe notes\\exploratory\\fit3d_parity_significance_20260926\\parity_significance.py collect
    .venv\\Scripts\\python.exe notes\\exploratory\\fit3d_parity_significance_20260926\\parity_significance.py analyse

`collect` reads every (subject, rep, camera) cue reading at the rep extreme for these arms:
  gt / gt_s           mocap 3D truth, full rate / masked to RTMPose's every-15th-frame grid
  perf2d / perf2d_s   mocap 3D projected to the image = a zero-error 2D detector
  rtm                 RTMPose 2D (inferred every 15th frame)
  nlf / nlf_s         NLF 3D rotated by the GT camera rotation, full rate / 15-frame grid
`analyse` computes the oracle-debiased swept verdict flip (as in src/fit3d/decision_eval.py)
and an exact sign-flip test over the 8 subjects on per-subject flip differences.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.fit3d import dataset as ds  # noqa: E402
from src.fit3d import depth_eval as de  # noqa: E402
from src.fit3d import twod_baseline as tb  # noqa: E402
from src.fit3d.biomech import IMAGE2D, WORLD3D, rep_summary  # noqa: E402
from src.fit3d.decision_eval import CUES, _debias, _verdict_metrics, mask_to_stride, nlf_world_points  # noqa: E402

STRIDE = 15
NLF_ROOT = REPO / "data/Fit3D/derived/preds/nlf"
RTM_ROOT = REPO / "data/Fit3D/derived/preds/rtmpose"
RECORDS = REPO / "data/Fit3D/derived/parity_significance_records.json"
RESULT = REPO / "data/Fit3D/derived/parity_significance.json"
ACTIONS = ["squat", "deadlift", "overhead_extension_thruster"]
CUE_NAMES = [c.name for c in CUES]
FAULT_WHEN = {c.name: c.fault_when for c in CUES}
# Cues whose 3D reading depends on the GT camera rotation (true vertical / ground plane).
GRAVITY_ORACLE = {"torso_lean_deg", "depth_ratio", "knee_width_ratio"}

PAIRINGS = {  # name -> (arm A, arm B, truth); delta = flip(B) - flip(A), negative = B better
    "parity_rtmpose_vs_nlf": ("rtm", "nlf_s", "gt_s"),
    "published_perfect2d_vs_nlf": ("perf2d", "nlf", "gt"),
    "detector_perfect2d_vs_rtmpose": ("perf2d_s", "rtm", "gt_s"),
}


def collect() -> None:
    out = {}
    for action in ACTIONS:
        recs = []
        for subj in ds.subjects("train"):
            if action not in ds.actions("train", subj):
                continue
            rep_ann = ds.load_rep_ann("train", subj).get(action)
            if not rep_ann:
                continue
            segs = ds.rep_segments(rep_ann)
            j3d = ds.load_joints3d("train", subj, action)
            for cam in ds.cameras("train", subj):
                nz = NLF_ROOT / de.pred_npz_name(subj, action, cam)
                rz = RTM_ROOT / tb.pred_npz_name(subj, action, cam)
                if not (nz.exists() and rz.exists()):
                    continue
                cp = ds.read_cam_params("train", subj, cam, action)
                proj = ds.project_world_to_image(j3d, cp)
                gt_cam = de.gt_in_camera_frame(j3d * 1000.0, cp)
                pred_cam, _ = de.load_prediction_h36m17(nz, gt_cam[:, :17, :], source="smpl3d")
                nlf = nlf_world_points(pred_cam, cp)
                rtm = tb.load_rtmpose_2d(rz)
                n = min(len(j3d), len(proj), len(nlf), len(rtm))
                arms = {
                    "gt": (j3d[:n], WORLD3D), "gt_s": (mask_to_stride(j3d[:n], STRIDE), WORLD3D),
                    "perf2d": (proj[:n], IMAGE2D), "perf2d_s": (mask_to_stride(proj[:n], STRIDE), IMAGE2D),
                    "rtm": (rtm[:n], IMAGE2D),
                    "nlf": (nlf[:n], WORLD3D), "nlf_s": (mask_to_stride(nlf[:n], STRIDE), WORLD3D),
                }
                for ri, (s, e) in enumerate(segs):
                    e = min(e, n)
                    if e - s < 5:
                        continue
                    r = {"subject": subj, "rep": ri, "camera": cam}
                    for k, (pts, mode) in arms.items():
                        summ = rep_summary(pts, mode, s, e)
                        for cue in CUE_NAMES:
                            r[f"{k}|{cue}"] = summ[cue]
                    recs.append(r)
        out[action] = {k: [r[k] for r in recs] for k in recs[0]}
        print(action, len(recs), "records")
    RECORDS.write_text(json.dumps(out))


def _flip(gt, rd, thrs, fw):
    return float(np.nanmean([_verdict_metrics(gt, rd, float(t), fw)["flip"] for t in thrs]))


def compare(d: dict, cue: str, pair: tuple[str, str, str]) -> dict:
    """Pooled debiased swept flip for both arms + per-subject flips on the pooled thresholds.

    With thresholds fixed from the pooled truth, the pooled flip is exactly the n-weighted mean
    of the per-subject flips, so the per-subject contributions sum to the pooled delta."""
    fw = FAULT_WHEN[cue]
    subj, cam = np.array(d["subject"]), np.array(d["camera"])
    gt = np.array(d[f"{pair[2]}|{cue}"], dtype=float)
    a, _ = _debias(np.array(d[f"{pair[0]}|{cue}"], dtype=float), gt, cam)
    b, _ = _debias(np.array(d[f"{pair[1]}|{cue}"], dtype=float), gt, cam)
    thrs = np.nanpercentile(gt[np.isfinite(gt)], np.linspace(20, 80, 13))
    subs = sorted(set(subj))
    per_subject = {}
    for s in subs:
        m = subj == s
        per_subject[s] = {"a": _flip(gt[m], a[m], thrs, fw), "b": _flip(gt[m], b[m], thrs, fw), "n": int(m.sum())}
    per_camera = {}
    for c in sorted(set(cam)):
        m = cam == c
        per_camera[c] = {"a": _flip(gt[m], a[m], thrs, fw), "b": _flip(gt[m], b[m], thrs, fw)}
    N = len(gt)
    contrib = np.array([v["n"] / N * (v["b"] - v["a"]) for v in per_subject.values()])
    return {"flip_a": _flip(gt, a, thrs, fw), "flip_b": _flip(gt, b, thrs, fw),
            "per_subject": per_subject, "per_camera": per_camera, "contrib": contrib.tolist()}


def signflip_p(contrib: np.ndarray) -> float:
    """Exact two-sided sign-flip p over all 2^n subject sign patterns (min 2/2^n)."""
    pats = np.array(list(itertools.product([1, -1], repeat=len(contrib))))
    return float((np.abs(pats @ contrib) >= abs(contrib.sum()) - 1e-12).mean())


def summarise(res: dict) -> dict:
    deltas = np.array([v["b"] - v["a"] for v in res["per_subject"].values()])
    c = np.array(res["contrib"])
    return {
        "flip_a": res["flip_a"], "flip_b": res["flip_b"], "delta": float(c.sum()),
        "p_signflip": signflip_p(c),
        "subj_delta_mean": float(deltas.mean()), "subj_delta_sd": float(deltas.std(ddof=1)),
        "subj_delta_min": float(deltas.min()), "subj_delta_max": float(deltas.max()),
        "n_b_better": int((deltas < -1e-9).sum()), "n_b_worse": int((deltas > 1e-9).sum()),
        "n_subjects": int(len(deltas)),
    }


def knee90(raw: dict) -> dict:
    """Squat-depth verdict at the canonical threshold (knee > 90 deg = didn't reach parallel):
    flip / false alarm / miss, raw and oracle-debiased, per arm. Truth and arms paired by grid."""
    arm_sets = {"full rate": [("perf2d", "gt"), ("nlf", "gt")],
                "15-frame grid": [("perf2d_s", "gt_s"), ("rtm", "gt_s"), ("nlf_s", "gt_s")]}
    out: dict = {}
    print("\nknee_angle at 90 deg (flip / false alarm / miss; per-subject flip mean +/- sd)")
    for action in ("squat", "overhead_extension_thruster"):
        d = raw[action]
        cam, subj = np.array(d["camera"]), np.array(d["subject"])
        out[action] = {}
        for grid, pairs in arm_sets.items():
            for arm, truth in pairs:
                gt = np.array(d[f"{truth}|knee_angle"], dtype=float)
                rd = np.array(d[f"{arm}|knee_angle"], dtype=float)
                for mode, x in (("raw", rd), ("debiased", _debias(rd, gt, cam)[0])):
                    m = _verdict_metrics(gt, x, 90.0, "high")
                    ps = [_verdict_metrics(gt[subj == s], x[subj == s], 90.0, "high")["flip"] for s in sorted(set(subj))]
                    m["subj_flip_mean"], m["subj_flip_sd"] = float(np.nanmean(ps)), float(np.nanstd(ps))
                    out[action][f"{arm}|{mode}"] = m
                    print(f"  {action[:9]:9s} {grid:13s} {arm:8s} {mode:8s} flip {m['flip']:.2f} FA "
                          f"{m['false_alarm']:.2f} miss {m['miss']:.2f} (prev {m['true_fault_rate']:.2f}, "
                          f"n {m['n']}) subj {m['subj_flip_mean']:.2f}+/-{m['subj_flip_sd']:.2f}")
    return out


def analyse() -> None:
    raw = json.loads(RECORDS.read_text())
    result: dict = {}
    for pname, pair in PAIRINGS.items():
        cues = CUE_NAMES if pname == "parity_rtmpose_vs_nlf" else ["knee_width_ratio"]
        result[pname] = {}
        for cue in cues:
            per_action, contribs = {}, []
            for action in ACTIONS:
                r = compare(raw[action], cue, pair)
                per_action[action] = {**summarise(r), "per_subject": r["per_subject"], "per_camera": r["per_camera"]}
                contribs.append(np.array(r["contrib"]))
            comb = np.mean(contribs, axis=0)
            result[pname][cue] = {"per_action": per_action,
                                  "combined": {"delta": float(comb.sum()), "p_signflip": signflip_p(comb)},
                                  "gravity_oracle": cue in GRAVITY_ORACLE}
            print(f"\n{pname} / {cue}  (A={pair[0]} B={pair[1]} truth={pair[2]}; negative = B better)")
            for action, s in per_action.items():
                cams = " ".join(f"{k}:{v['a']:.2f}/{v['b']:.2f}" for k, v in s["per_camera"].items())
                print(f"  {action[:9]:9s} {s['flip_a']:.3f}->{s['flip_b']:.3f} d {s['delta']:+.3f} "
                      f"p={s['p_signflip']:.3f} subj d {s['subj_delta_mean']:+.3f}±{s['subj_delta_sd']:.3f} "
                      f"[{s['subj_delta_min']:+.2f},{s['subj_delta_max']:+.2f}] B better/worse "
                      f"{s['n_b_better']}/{s['n_b_worse']}  cams A/B {cams}")
            print(f"  combined d {comb.sum():+.3f} p={signflip_p(comb):.3f}")
    result["knee90"] = knee90(raw)
    RESULT.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {RESULT.relative_to(REPO)}")


if __name__ == "__main__":
    {"collect": collect, "analyse": analyse}[sys.argv[1]]()
