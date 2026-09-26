"""Pre-registered separability test for the two silent rules EgoExo pixels can speak to.

Reads the ``--json`` outputs of ``run_jumping_jacks_validation.py`` and
``run_high_knee_validation.py`` (full archive) and asks, per rule, whether the rule's own metric
ranks the human-flagged actions on the fault side. Plan, fixed before the data existed:
``docs/superpowers/specs/2026-09-26-egoexo-silent-rule-separability-prereg.md``.
Report-only: nothing here changes a rule, a cut, or a registration.

    .venv\\Scripts\\python.exe scripts/egoexo/run_silent_rule_separability.py ^
      --jj-json <jj.json> --hk-json <hk.json> --out <separability.json>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.egoexo.high_knee_validation import (  # noqa: E402
    GATED_VIEWS, KNEE_LIFT_COMMENT_DISCLOSED, knee_lift_comment_labels,
)
from src.egoexo.jumping_jacks_validation import (  # noqa: E402
    FOOT_SPLIT_COMMENT_CATEGORY, FOOT_SPLIT_CRITERION,
)
from src.egoexo.separability import (  # noqa: E402
    auc, cluster_bootstrap_auc, high_knee_action_scores, jumping_jacks_action_scores,
    leave_one_group_out, verdict, youden_cut,
)
from src.pose.movements.jumping_jacks import LEG_ROM_MILD_RATIO  # noqa: E402

DATA = REPO_ROOT / "data/EgoExo-Fitness"


def analyse(name: str, scores: dict[str, float], labels: dict[str, int],
            participant: dict[str, str], *, lower_is_fault: bool, propose_cut: bool) -> dict:
    ids = sorted(set(scores) & set(labels))
    s = [scores[i] for i in ids]
    y = [labels[i] for i in ids]
    g = [participant[i] for i in ids]
    ci = cluster_bootstrap_auc(s, y, g, lower_is_fault=lower_is_fault)
    result = {
        "analysis": name,
        "n_actions": len(ids),
        "n_positive": sum(y),
        "n_participants": len(set(g)),
        "n_positive_participants": len({gi for gi, yi in zip(g, y) if yi == 1}),
        "labelled_but_unscored": sorted(set(labels) - set(scores)),
        "auc": auc(s, y, lower_is_fault=lower_is_fault),
        "ci95": [ci.low, ci.high],
        "bootstrap_used": ci.used,
        "bootstrap_skipped": ci.skipped,
        "verdict": verdict(ci),
    }
    if propose_cut:
        held = leave_one_group_out(s, y, g, lower_is_fault=lower_is_fault)
        result["candidate_cut_all_data"] = youden_cut(s, y, lower_is_fault=lower_is_fault)
        result["held_out"] = {
            "tp": held.tp, "fp": held.fp, "tn": held.tn, "fn": held.fn,
            "sensitivity": held.sensitivity, "specificity": held.specificity,
            "folds_without_a_cut": held.folds_without_a_cut,
        }
    return result


def show(result: dict) -> None:
    lo, hi = result["ci95"]
    print(f"{result['analysis']:48s} n={result['n_actions']:3d} pos={result['n_positive']:2d} "
          f"({result['n_positive_participants']}/{result['n_participants']} people)  "
          f"AUC {result['auc']:.3f} [{lo:.3f}, {hi:.3f}]  {result['verdict']}")
    if result["labelled_but_unscored"]:
        print(f"    labelled but unscored: {len(result['labelled_but_unscored'])}")
    if "held_out" in result:
        h = result["held_out"]
        print(f"    candidate cut (all data, NOT applied) {result['candidate_cut_all_data']:.3f}; "
              f"leave-one-participant-out sens {h['sensitivity']:.3f} spec {h['specificity']:.3f} "
              f"(tp {h['tp']} fp {h['fp']} tn {h['tn']} fn {h['fn']}, "
              f"{h['folds_without_a_cut']} folds without a cut)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jj-json", type=Path, default=None)
    ap.add_argument("--hk-json", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=DATA / "processed/manifest.csv")
    ap.add_argument("--tkv", type=Path, default=DATA / "processed/labels/tkv.json")
    ap.add_argument("--judgements", type=Path,
                    default=DATA / "raw_annotations/interpretable_action_judgement.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not (args.jj_json or args.hk_json):
        raise SystemExit("pass --jj-json and/or --hk-json")

    with open(args.manifest, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    participant = {r["sample_id"]: r["participant"] for r in rows}
    annotators = {r["sample_id"]: int(r["num_annotators"]) for r in rows}
    results: list[dict] = []

    if args.jj_json:
        payload = json.loads(args.jj_json.read_text(encoding="utf-8"))
        tkv = json.loads(args.tkv.read_text(encoding="utf-8"))
        jj = jumping_jacks_action_scores(payload, spec_cut=LEG_ROM_MILD_RATIO)
        judged = {sid: c[FOOT_SPLIT_CRITERION] for sid, c in tkv.items() if FOOT_SPLIT_CRITERION in c}
        majority = {sid: int(r["fault"]) for sid, r in judged.items()}
        any_false = {sid: int(r["n_false"] > 0) for sid, r in judged.items()}
        results.append(analyse("JJ leg ROM | widest stance | majority label [PRIMARY]",
                               {k: v["score"] for k, v in jj.items()}, majority, participant,
                               lower_is_fault=True, propose_cut=True))
        results.append(analyse("JJ leg ROM | widest stance | any-false label",
                               {k: v["score"] for k, v in jj.items()}, any_false, participant,
                               lower_is_fault=True, propose_cut=False))
        # Pre-registered robustness checks (prereg "JJ label threats"). A positive is DROPPED, not
        # relabelled negative, when its comment names a fault widest stance is not about.
        def only(categories: set[str]) -> dict[str, int]:
            return {sid: y for sid, y in majority.items()
                    if y == 0 or FOOT_SPLIT_COMMENT_CATEGORY.get(sid) in categories}
        results.append(analyse("JJ leg ROM | widest stance | non-pattern positives",
                               {k: v["score"] for k, v in jj.items()},
                               only({"width", "ambiguous", "closing", "other"}), participant,
                               lower_is_fault=True, propose_cut=False))
        results.append(analyse("JJ leg ROM | widest stance | width+ambiguous positives",
                               {k: v["score"] for k, v in jj.items()},
                               only({"width", "ambiguous"}), participant,
                               lower_is_fault=True, propose_cut=False))
        results.append(analyse("JJ leg ROM | widest stance | single-annotator stratum",
                               {k: v["score"] for k, v in jj.items()},
                               {sid: y for sid, y in majority.items() if annotators.get(sid) == 1},
                               participant, lower_is_fault=True, propose_cut=False))
        results.append(analyse("JJ leg ROM | spec 1.3 rep fire rate | majority",
                               {k: v["spec_cut_rep_rate"] for k, v in jj.items()}, majority,
                               participant, lower_is_fault=False, propose_cut=False))

    if args.hk_json:
        payload = json.loads(args.hk_json.read_text(encoding="utf-8"))
        hk = high_knee_action_scores(payload, gated_views=GATED_VIEWS)
        comments = knee_lift_comment_labels(args.judgements)
        binary = {sid: int(v == "positive") for sid, v in comments.items()
                  if v in ("positive", "negative")}
        held_out = {sid: y for sid, y in binary.items() if sid not in KNEE_LIFT_COMMENT_DISCLOSED}
        # No cut is proposed for High Knee: the comment rule judges a whole action and its own
        # docstring says it CANNOT license a threshold (high_knee_validation.py).
        results.append(analyse("HK knee lift | peak thigh elevation | held-out [PRIMARY]",
                               {k: v["score"] for k, v in hk.items()}, held_out, participant,
                               lower_is_fault=True, propose_cut=False))
        results.append(analyse("HK knee lift | peak thigh elevation | all 68",
                               {k: v["score"] for k, v in hk.items()}, binary, participant,
                               lower_is_fault=True, propose_cut=False))
        results.append(analyse("HK knee lift | cited 45 deg fire rate | held-out",
                               {k: v["cited_cut_rate"] for k, v in hk.items()
                                if math.isfinite(v["cited_cut_rate"])},
                               held_out, participant, lower_is_fault=False, propose_cut=False))

    for result in results:
        show(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
