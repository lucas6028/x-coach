"""Retrospective audit of threshold weighting on saved Fit3D predictions.

Run from the repository root: python -m src.fit3d.threshold_audit
No model inference or production evaluation behavior is changed.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from src.fit3d import decision_eval as de

OUT = Path("data/Fit3D/derived/threshold_audit_20260919")
ACTIONS = ("squat", "deadlift", "overhead_extension_thruster")
MODELS = ("NLF", "MeTRAbs")
LAMBDAS = (.4, .5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.5)


def threshold_losses(g, p, thresholds):
    """One disagreement fraction per observation; all thresholds have equal mass."""
    return np.mean((g[:, None] > thresholds) != (p[:, None] > thresholds), axis=1)


def uniform_losses(g, p, low, high):
    """Exact continuous uniform-threshold disagreement, including range clipping."""
    return np.abs(np.clip(g, low, high) - np.clip(p, low, high)) / (high - low)


def integrate_decisions(g, p, low, high):
    """Independent integral: evaluate verdicts on every interval between breakpoints."""
    cuts = np.unique(np.clip(np.r_[low, high, g, p], low, high))
    mid = (cuts[:-1] + cuts[1:]) / 2
    losses = np.mean((g[:, None] > mid) != (p[:, None] > mid), axis=0)
    return float(losses @ np.diff(cuts))


def losses(g, p, bounds):
    low, high = np.percentile(g, [20, 80])
    return {
        "mae_deg": np.abs(p - g),
        "full_uniform": uniform_losses(g, p, *bounds),
        "central_uniform": uniform_losses(g, p, low, high),
        "percentile_13": threshold_losses(g, p, np.percentile(g, np.linspace(20, 80, 13))),
        "percentile_6001": threshold_losses(g, p, np.percentile(g, np.linspace(20, 80, 6001))),
        "fixed_90": ((g > 90) != (p > 90)).astype(float),
    }


def summaries(g, p, bounds):
    result = {k: float(v.mean()) for k, v in losses(g, p, bounds).items()}
    area = integrate_decisions(g, p, *bounds)
    assert np.isclose(area, result["mae_deg"], atol=1e-10)
    result["integrated_disagreement_deg"] = area
    low, high = np.percentile(g, [20, 80])
    central_area = integrate_decisions(g, p, low, high)
    assert np.isclose(central_area / (high - low), result["central_uniform"], atol=1e-10)
    result["central_integrated_disagreement_deg"] = central_area
    result["outside_central_integrated_disagreement_deg"] = area - central_area
    result["pearson_r"] = float(np.corrcoef(g, p)[0, 1])
    return result


def collect():
    """Recompute rep summaries from existing NPZ files, retaining explicit identifiers."""
    records = {}
    for action in ACTIONS:
        records[action] = {}
        for model in MODELS:
            rows, _ = de.collect_records(
                Path("data/Fit3D/derived/preds") / model.lower(),
                action=action, frame_stride=15,
            )
            records[action][model] = [
                {"subject": r["subject"], "camera": r["camera"],
                 "rep_index": r["rep_index"], "gt": r["gt"]["knee_angle"],
                 "pred": r["nlf"]["knee_angle"]}
                for r in rows
            ]
            print(f"collected {action} {model}: {len(rows)}", flush=True)
    return records


def paired_summary(delta, subjects):
    """Subject-macro paired differences; bootstrap conditional on fitted calibration/grid."""
    names = np.unique(subjects)
    vals = np.array([delta[subjects == s].mean() for s in names])
    rng = np.random.default_rng(20260919)
    boot = vals[rng.integers(len(vals), size=(10000, len(vals)))].mean(axis=1)
    return {"pooled_delta": float(delta.mean()), "subject_macro_delta": float(vals.mean()),
            "subject_sd": float(vals.std(ddof=1)),
            "conditional_bootstrap_95ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "subjects_negative": int(np.sum(vals < -1e-12)), "n_subjects": len(names),
            "per_subject": dict(zip(names.tolist(), vals.tolist()))}


def run(records):
    saved = json.loads(Path("data/Fit3D/derived/rep_extreme_decomposition.json").read_text())
    output = {"metadata": {
        "purpose": "Retrospective metric sensitivity audit, not clinical validation",
        "frame_stride": 15, "gt_rate": "full", "source": "smpl3d",
        "paired_delta": "NLF minus MeTRAbs; lower is better",
        "bootstrap": "10000 subject resamples; calibration, threshold grids and lambda fixed; conditional only",
        "calibration": "raw; global oracle mean removal; per-camera oracle mean removal",
    }, "actions": {}}
    for action in ACTIONS:
        maps = {m: {(r["subject"], r["camera"], r["rep_index"]): r
                    for r in records[action][m]} for m in MODELS}
        assert all(len(maps[m]) == len(records[action][m]) for m in MODELS)
        keys = [k for k in maps["NLF"] if all(k in maps[m] for m in MODELS)]
        keys = [k for k in keys if all(np.isfinite([maps[m][k]["gt"], maps[m][k]["pred"]]).all()
                                    for m in MODELS)]
        g = np.array([maps["NLF"][k]["gt"] for k in keys])
        cam, subj = np.array([k[1] for k in keys]), np.array([k[0] for k in keys])
        preds = {m: np.array([maps[m][k]["pred"] for k in keys]) for m in MODELS}
        for m in MODELS:
            assert np.allclose(g, [maps[m][k]["gt"] for k in keys])
            assert np.allclose(np.c_[g, preds[m]], saved[action]["pairs"][m], atol=1e-9)
        arms = {
            "raw": preds,
            "global_oracle": {m: p - (p - g).mean() for m, p in preds.items()},
            "camera_oracle": {m: de._debias(p, g, cam)[0] for m, p in preds.items()},
        }
        shrink = {m: {lam: g.mean() + lam * (p - p.mean()) for lam in LAMBDAS}
                  for m, p in preds.items()}
        all_values = np.concatenate([g] + [p for arm in arms.values() for p in arm.values()]
                                    + [p for arm in shrink.values() for p in arm.values()])
        bounds = (float(all_values.min() - 1), float(all_values.max() + 1))
        entry = {"n_pairs": len(keys), "n_subjects": len(set(subj)),
                 "n_unique_reps": len({(k[0], k[2]) for k in keys}),
                 "input_n": {m: len(maps[m]) for m in MODELS},
                 "common_support_bounds": bounds, "gt_q20_q80": np.percentile(g, [20, 80]).tolist(),
                 "gt_fault_90": float(np.mean(g > 90)),
                 "thresholds_13": np.percentile(g, np.linspace(20, 80, 13)).tolist(),
                 "saved_pairs_reproduced": True, "tracks": {}, "shrinkage": {}}
        for track, arm in arms.items():
            metrics = {m: losses(g, p, bounds) for m, p in arm.items()}
            entry["tracks"][track] = {
                "models": {m: summaries(g, p, bounds) for m, p in arm.items()},
                "paired_differences": {k: paired_summary(metrics["NLF"][k] - metrics["MeTRAbs"][k], subj)
                                       for k in metrics["NLF"]},
            }
        for m, candidates in shrink.items():
            rows = [{"lambda": lam, **summaries(g, p, bounds)} for lam, p in candidates.items()]
            best = min(rows, key=lambda r: r["mae_deg"])["lambda"]
            before, after = losses(g, candidates[1.], bounds), losses(g, candidates[best], bounds)
            entry["shrinkage"][m] = {
                "rows": rows, "mae_best_lambda": best,
                "n_pointwise_error_improved": int(np.sum(after["mae_deg"] < before["mae_deg"] - 1e-10)),
                "n_pointwise_error_worsened": int(np.sum(after["mae_deg"] > before["mae_deg"] + 1e-10)),
                "best_minus_baseline": {k: paired_summary(after[k] - before[k], subj) for k in before},
            }
        output["actions"][action] = entry
        print(f"audited {action}: {len(keys)} pairs", flush=True)
    return output


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "paired_records.json"
    if cache.exists():
        records = json.loads(cache.read_text())
    else:
        records = collect()
        cache.write_text(json.dumps(records, indent=2), encoding="utf-8")
    result = run(records)
    result["metadata"]["elapsed_seconds"] = time.perf_counter() - start
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(OUT / "results.json", flush=True)


if __name__ == "__main__":
    main()
