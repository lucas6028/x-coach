"""CLI for Fit3D step 1: does the SMPL-X collar joint move independently of the arm?

Run from the repository root:

    .venv\\Scripts\\python.exe scripts/fit3d/run_collar_elevation.py
    .venv\\Scripts\\python.exe scripts/fit3d/run_collar_elevation.py --json data/fit3d/derived/collar_elevation.json

Reads the Fit3D train-split SMPL-X fits and ``rep_ann.json`` only (no video, no estimator), plus
the SMPL-X neutral template for the rest skeleton. The pre-registered rule is in the module
docstring of :mod:`src.fit3d.collar_elevation`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fit3d import axial_rotation as ax  # noqa: E402
from src.fit3d import collar_elevation as ce  # noqa: E402
from src.fit3d import dataset as ds  # noqa: E402

SHRUG = "barbell_shrug"
LUNGE = "dumbbell_reverse_lunge"
RAISES = ("side_lateral_raise", "dumbbell_scaptions")
BPA = "band_pull_apart"
DESCRIPTIVE = ("overhead_trap_raises",)  # deliberate scapular motion; not a normal-reference action
SIDES = ("L", "R")


def load_series(root, subj, action, rest, side):
    pose = ax.load_smplx("train", subj, action, root)["body_pose"]
    c_elev, c_prot = ce.collar_angles(pose, rest, side)
    a_elev, a_horiz = ce.arm_angles(pose, rest, side)
    windows = ds.rep_segments(ds.load_rep_ann("train", subj, root)[action])
    return {"c_elev": c_elev, "c_prot": c_prot, "a_elev": a_elev, "a_horiz": a_horiz, "windows": windows}


def jitter_floor(series_list, key):
    """sqrt(2) x pooled SD of `key` around a 15-frame moving average, over rep frames only."""
    res = []
    for s in series_list:
        j = ce.jitter_residuals(s[key])
        for a, z in s["windows"]:
            res.append(j[a:z])
    r = np.concatenate(res)
    return float(np.sqrt(2.0) * np.sqrt(np.mean(r**2)))


def white_noise_floor(series_list, key):
    """sqrt(2) x pooled second-difference sigma over rep frames (window-free lower bound)."""
    parts = []
    for s in series_list:
        x = s[key]
        dd = x[2:] - 2.0 * x[1:-1] + x[:-2]
        for a, z in s["windows"]:
            parts.append(dd[max(a - 1, 0): z - 2])
    return float(np.sqrt(2.0) * np.sqrt(np.mean(np.concatenate(parts) ** 2)) / np.sqrt(6.0))


def window_sensitivity(series_list, key, residual_sd, windows=(9, 15, 25, 51)):
    """residual_sd / (sqrt(2) x moving-average jitter) for several windows -- the rule's floor is not identifiable."""
    out = {}
    for w in windows:
        res = []
        for s in series_list:
            j = s[key] - ce.moving_average(s[key], w)
            for a, z in s["windows"]:
                res.append(j[a:z])
        floor = float(np.sqrt(2.0) * np.sqrt(np.mean(np.concatenate(res) ** 2)))
        out[str(w)] = residual_sd / floor
    return out


def within_rep_sd(series_list, key):
    """Pooled within-rep SD of `key` (the loose, motion-inclusive floor)."""
    parts = []
    for s in series_list:
        for a, z in s["windows"]:
            seg = s[key][a:z]
            parts.append(seg - seg.mean())
    r = np.concatenate(parts)
    return float(np.sqrt(np.mean(r**2)))


def deltas_for(series_by_subj, driver, collar, arm):
    out = []
    for subj, s in series_by_subj.items():
        out += ce.rep_deltas(s[driver], s[collar], s[arm], s["windows"], subj)
    return out


def per_subject_resid(deltas, fit):
    xs = {}
    for d in deltas:
        xs.setdefault(d.subject, []).append(d)
    out = {}
    for subj, ds_ in xs.items():
        x = np.array([d.d_arm for d in ds_]) - np.mean([d.d_arm for d in ds_])
        y = np.array([d.d_collar for d in ds_]) - np.mean([d.d_collar for d in ds_])
        r = y - fit.slope * x
        out[subj] = float(np.sqrt(np.mean(r**2)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ds.DEFAULT_FIT3D_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rest = ax.rest_joints()
    subjects = ds.load_info(args.root)["train_subj_names"]
    clavicle_m = {s: float(np.linalg.norm(rest[ce.SHOULDER[s]] - rest[ce.COLLAR[s]])) for s in SIDES}
    actions = (SHRUG, LUNGE, *RAISES, BPA, *DESCRIPTIVE)
    data = {
        (act, side): {subj: load_series(args.root, subj, act, rest, side) for subj in subjects}
        for act in actions
        for side in SIDES
    }

    report = {"subjects": subjects, "clavicle_rest_m": clavicle_m, "sides": {}}
    for side in SIDES:
        shrug_deltas = deltas_for(data[(SHRUG, side)], "c_elev", "c_elev", "a_elev")
        shrug_amp = float(np.median([d.d_collar for d in shrug_deltas]))
        lunge_deltas = deltas_for(data[(LUNGE, side)], "c_elev", "c_elev", "a_elev")
        per_subj_shrug_vs_lunge = {
            subj: {
                "shrug_collar_amp_median_deg": float(np.median([d.d_collar for d in shrug_deltas if d.subject == subj])),
                "shrug_abs_arm_change_median_deg": float(np.median([abs(d.d_arm) for d in shrug_deltas if d.subject == subj])),
                "lunge_collar_amp_median_deg": float(np.median([d.d_collar for d in lunge_deltas if d.subject == subj])),
            }
            for subj in subjects
        }
        side_rep = {
            "shrug_control": {
                "n_reps": len(shrug_deltas),
                "collar_elev_amplitude_median_deg": shrug_amp,
                "collar_elev_amplitude_p10_p90_deg": [float(v) for v in np.percentile([d.d_collar for d in shrug_deltas], [10, 90])],
                "arm_elev_change_median_deg": float(np.median([d.d_arm for d in shrug_deltas])),
                "jitter_floor_deg": jitter_floor(list(data[(SHRUG, side)].values()), "c_elev"),
                "white_noise_floor_deg": white_noise_floor(list(data[(SHRUG, side)].values()), "c_elev"),
                "per_subject_vs_reverse_lunge": per_subj_shrug_vs_lunge,
                "subjects_shrug_gt_lunge": sum(
                    v["shrug_collar_amp_median_deg"] > v["lunge_collar_amp_median_deg"] for v in per_subj_shrug_vs_lunge.values()
                ),
            },
            "reverse_lunge_collar_amp_median_deg": float(np.median([d.d_collar for d in lunge_deltas])),
            "reverse_lunge_collar_amp_max_deg": float(np.max([d.d_collar for d in lunge_deltas])),
            "loose_floor_reverse_lunge_within_rep_sd_deg": within_rep_sd(list(data[(LUNGE, side)].values()), "c_elev"),
            "lunge_jitter_floor_deg": jitter_floor(list(data[(LUNGE, side)].values()), "c_elev"),
            "actions": {},
        }
        for act in (*RAISES, *DESCRIPTIVE):
            deltas = deltas_for(data[(act, side)], "a_elev", "c_elev", "a_elev")
            fit = ce.within_subject_fit(deltas)
            floor = jitter_floor(list(data[(act, side)].values()), "c_elev")
            side_rep["actions"][act] = {
                "pair": "collar elevation vs arm elevation",
                "fit": fit.to_dict(),
                "d_collar_median_deg": float(np.median([d.d_collar for d in deltas])),
                "d_arm_median_deg": float(np.median([d.d_arm for d in deltas])),
                "jitter_floor_deg": floor,
                "white_noise_floor_deg": white_noise_floor(list(data[(act, side)].values()), "c_elev"),
                "residual_over_jitter_floor_by_window": window_sensitivity(list(data[(act, side)].values()), "c_elev", fit.residual_sd),
                "residual_sd_mm": fit.residual_sd * np.pi / 180 * clavicle_m[side] * 1000,
                "per_subject_residual_rms_deg": per_subject_resid(deltas, fit),
                "per_subject_median_ratio_dcollar_darm": {
                    subj: float(np.median([d.d_collar / d.d_arm for d in deltas if d.subject == subj])) for subj in subjects
                },
                "verdict": ce.verdict(fit.residual_sd, floor, shrug_amp) if act in RAISES else "descriptive",
            }
        for collar_key, label in (("c_prot", "collar protraction vs horizontal abduction"), ("c_elev", "collar elevation vs horizontal abduction")):
            deltas = deltas_for(data[(BPA, side)], "a_horiz", collar_key, "a_horiz")
            fit = ce.within_subject_fit(deltas)
            floor = jitter_floor(list(data[(BPA, side)].values()), collar_key)
            above = fit.residual_sd > ce.STOP_FLOOR_MULT * floor
            side_rep["actions"][f"{BPA}:{collar_key}"] = {
                "pair": label,
                "fit": fit.to_dict(),
                "d_collar_median_deg": float(np.median([d.d_collar for d in deltas])),
                "d_arm_median_deg": float(np.median([d.d_arm for d in deltas])),
                "jitter_floor_deg": floor,
                "white_noise_floor_deg": white_noise_floor(list(data[(BPA, side)].values()), collar_key),
                "residual_sd_mm": fit.residual_sd * np.pi / 180 * clavicle_m[side] * 1000,
                "per_subject_residual_rms_deg": per_subject_resid(deltas, fit),
                # No retraction positive control exists in Fit3D, so only the STOP half applies.
                "verdict": ("above floor (no positive control)" if above else "STOP")
                if collar_key == "c_prot" else "descriptive",
            }
        report["sides"][side] = side_rep

    # Does the SMPL-X template's own L/R asymmetry make the shrug gap? Mirror each left collar
    # rotation onto the right side of the real template and re-measure the amplitude.
    mirror = np.diag([-1.0, 1.0, 1.0])
    own, mirrored = [], []
    for subj in subjects:
        pose = ax.load_smplx("train", subj, SHRUG, args.root)["body_pose"]
        swapped = pose.copy()
        swapped[:, ce.COLLAR["R"] - ce.BODY_POSE_OFFSET] = mirror @ pose[:, ce.COLLAR["L"] - ce.BODY_POSE_OFFSET] @ mirror
        left, _ = ce.collar_angles(pose, rest, "L")
        right_from_left, _ = ce.collar_angles(swapped, rest, "R")
        for a, z in ds.rep_segments(ds.load_rep_ann("train", subj, args.root)[SHRUG]):
            own.append(float(np.ptp(left[a:z])))
            mirrored.append(float(np.ptp(right_from_left[a:z])))
    report["template_mirror_check"] = {
        "left_shrug_amp_median_deg": float(np.median(own)),
        "left_rotations_on_right_template_amp_median_deg": float(np.median(mirrored)),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
