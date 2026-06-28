"""EgoExo-Fitness dataset pipeline — Stage 1: manifest, participant-disjoint splits, labels.

Mirrors ``src/rehab24/dataset.py``. The sample unit is one *judged action* — a single
entry in ``interpretable_action_judgement.json`` (key ``<record_id>_action_<N>``). Each
such entry is self-contained: it carries its own ``st_ed_frame`` and ``frame_root`` (the
``action_N`` index does NOT line up with ``action_level_annotations.json`` IDs, so we do
not join against that file), plus the interpretable labels we care about:

  * ``key_point_verification`` — the per-action technical-keypoint checklist (TKV); the
    supervised target for E1 (guidance-based execution verification).
  * ``action_quality_score`` — 1..5 holistic quality (target for E2).
  * ``comment`` / ``action_guidance`` — natural-language feedback / canonical form (E7/E8).

Splits are **participant-disjoint**. ``original_actor`` is formatted ``DATE-Name`` and the
same person recorded on several dates appears under several strings (e.g. ``Jianan`` spans
six dates), so we group by the *name* part — splitting on the raw string would leak a
person across train/test. The official ActivityNet-style substep split is deliberately not
used: it places every record in both train and test, which is fine for localization but
leaks subjects for form assessment.

Run from the repo root, e.g.::

    python scripts/egoexo/build_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "EgoExo-Fitness"
DEFAULT_ANNOTATION_ROOT = DEFAULT_DATA_ROOT / "raw_annotations"
DEFAULT_PROCESSED_ROOT = DEFAULT_DATA_ROOT / "processed"

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15

IAJ_KEY_RE = re.compile(r"^(?P<record>.+)_action_(?P<idx>\d+)$")
DATE_PREFIX_RE = re.compile(r"^\d{2,8}[-_ ]?(?P<name>.+)$")

MANIFEST_FIELDS = [
    "sample_id",
    "split",
    "record_id",
    "participant",
    "original_actor",
    "action_index",
    "action_name",
    "st_frame",
    "ed_frame",
    "num_frames_segment",
    "frame_root",
    "views",
    "num_views",
    "num_annotators",
    "quality_score_mean",
    "quality_score_mode",
    "num_criteria",
    "num_faults",
]


def participant_id(original_actor: str) -> str:
    """Map ``original_actor`` (``DATE-Name``) to a stable participant id (the name).

    Grouping by name is the safe direction for leakage: at worst it merges two distinct
    people who share a name, which only makes the split *more* conservative, never leaky.
    """
    text = original_actor.strip()
    match = DATE_PREFIX_RE.match(text)
    name = match.group("name") if match else text
    return re.sub(r"[-_\s]+", "", name).lower()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _truthy(value) -> bool | None:
    label = str(value).strip().lower()
    if label == "true":
        return True
    if label == "false":
        return False
    return None


@dataclass
class JudgedAction:
    sample_id: str
    record_id: str
    action_index: int
    action_name: str
    st_frame: int
    ed_frame: int
    frame_root: str
    original_actor: str
    participant: str
    views: list[str]
    scores: list[int]
    comments: list[str]
    guidance: str
    # criterion text -> {"n_true": int, "n_false": int}
    criteria: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def num_annotators(self) -> int:
        return len(self.scores)

    @property
    def faults(self) -> dict[str, int]:
        """Default per-criterion fault label: strict-majority False (tie -> pass)."""
        return {
            text: int(c["n_false"] > c["n_true"]) for text, c in self.criteria.items()
        }


def build_record_info(meta: dict) -> dict[str, dict]:
    """record_id -> {original_actor, participant, views} (record_id via record_index)."""
    idx_to_id = {idx: rid for rid, idx in meta["record_index"].items()}
    info: dict[str, dict] = {}
    for i, record in enumerate(meta["records"]):
        rid = record.get("record_id") or idx_to_id.get(i)
        actor = record.get("original_actor", "")
        info[rid] = {
            "original_actor": actor,
            "participant": participant_id(actor),
            "views": list(record.get("views", [])),
        }
    return info


def parse_judged_actions(iaj: dict, record_info: dict[str, dict]) -> list[JudgedAction]:
    actions: list[JudgedAction] = []
    for key, entry in iaj.items():
        match = IAJ_KEY_RE.match(key)
        if not match:
            raise ValueError(f"Unexpected IAJ key format: {key!r}")
        record_id = match.group("record")
        action_index = int(match.group("idx"))
        info = record_info.get(record_id, {"original_actor": "", "participant": "", "views": []})

        annotations = entry.get("annotations", [])
        scores = [a["action_quality_score"] for a in annotations if a.get("action_quality_score") is not None]
        comments = [a.get("comment", "") for a in annotations]
        guidances = [a.get("action_guidance", "").strip() for a in annotations if a.get("action_guidance")]
        guidance = Counter(guidances).most_common(1)[0][0] if guidances else ""
        names = [a.get("action_name", "") for a in annotations if a.get("action_name")]
        action_name = Counter(names).most_common(1)[0][0] if names else ""

        criteria: dict[str, dict[str, int]] = defaultdict(lambda: {"n_true": 0, "n_false": 0})
        for a in annotations:
            for pair in a.get("key_point_verification", []):
                if len(pair) != 2:
                    continue
                text = str(pair[0]).strip()
                verdict = _truthy(pair[1])
                if verdict is True:
                    criteria[text]["n_true"] += 1
                elif verdict is False:
                    criteria[text]["n_false"] += 1

        st_ed = entry.get("st_ed_frame", [0, 0])
        actions.append(
            JudgedAction(
                sample_id=key,
                record_id=record_id,
                action_index=action_index,
                action_name=action_name,
                st_frame=int(st_ed[0]),
                ed_frame=int(st_ed[1]),
                frame_root=entry.get("frame_root", f"frames_open/{record_id}"),
                original_actor=info["original_actor"],
                participant=info["participant"],
                views=info["views"],
                scores=scores,
                comments=comments,
                guidance=guidance,
                criteria=dict(criteria),
            )
        )
    actions.sort(key=lambda a: (a.record_id, a.action_index))
    return actions


def assign_participant_splits(
    actions: Sequence[JudgedAction],
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> dict[str, str]:
    """Greedy participant-disjoint split balanced on judged-action count.

    Deterministic: participants are processed by descending action count (name as
    tie-break) and each is placed in the split with the most remaining headroom.
    Returns participant -> split.
    """
    per_participant: Counter[str] = Counter(a.participant for a in actions)
    total = sum(per_participant.values())
    caps = {
        "train": (1.0 - val_frac - test_frac) * total,
        "val": val_frac * total,
        "test": test_frac * total,
    }
    loads = {name: 0 for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for participant, count in sorted(per_participant.items(), key=lambda kv: (-kv[1], kv[0])):
        pick = max(SPLIT_NAMES, key=lambda s: (caps[s] - loads[s], s == "train"))
        assignment[participant] = pick
        loads[pick] += count
    return assignment


# --------------------------------------------------------------------------- writing


def build_manifest_rows(actions: Sequence[JudgedAction], splits: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for a in actions:
        faults = a.faults
        rows.append(
            {
                "sample_id": a.sample_id,
                "split": splits[a.participant],
                "record_id": a.record_id,
                "participant": a.participant,
                "original_actor": a.original_actor,
                "action_index": str(a.action_index),
                "action_name": a.action_name,
                "st_frame": str(a.st_frame),
                "ed_frame": str(a.ed_frame),
                "num_frames_segment": str(max(0, a.ed_frame - a.st_frame)),
                "frame_root": a.frame_root,
                "views": ";".join(a.views),
                "num_views": str(len(a.views)),
                "num_annotators": str(a.num_annotators),
                "quality_score_mean": f"{statistics.mean(a.scores):.3f}" if a.scores else "",
                "quality_score_mode": str(Counter(a.scores).most_common(1)[0][0]) if a.scores else "",
                "num_criteria": str(len(a.criteria)),
                "num_faults": str(sum(faults.values())),
            }
        )
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_splits_and_labels(
    processed_root: Path,
    actions: Sequence[JudgedAction],
    splits: dict[str, str],
) -> None:
    split_dir = processed_root / "splits"
    label_dir = processed_root / "labels"
    split_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # splits/<name>_keys.json
    by_split: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for a in actions:
        by_split[splits[a.participant]].append(a.sample_id)
    for name in SPLIT_NAMES:
        with (split_dir / f"{name}_keys.json").open("w", encoding="utf-8") as f:
            json.dump(sorted(by_split[name]), f, indent=2)

    # labels/score.json
    score_labels = {
        a.sample_id: {
            "scores": a.scores,
            "mean": round(statistics.mean(a.scores), 3) if a.scores else None,
            "median": statistics.median(a.scores) if a.scores else None,
            "mode": Counter(a.scores).most_common(1)[0][0] if a.scores else None,
        }
        for a in actions
    }
    with (label_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score_labels, f, indent=2, sort_keys=True)

    # labels/tkv.json — per criterion: vote counts + default fault label
    tkv_labels = {
        a.sample_id: {
            text: {"n_true": c["n_true"], "n_false": c["n_false"], "fault": int(c["n_false"] > c["n_true"])}
            for text, c in a.criteria.items()
        }
        for a in actions
    }
    with (label_dir / "tkv.json").open("w", encoding="utf-8") as f:
        json.dump(tkv_labels, f, indent=2, sort_keys=True, ensure_ascii=False)

    # labels/comments.json
    comments = {a.sample_id: a.comments for a in actions}
    with (label_dir / "comments.json").open("w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2, sort_keys=True, ensure_ascii=False)

    # labels/guidance.json — canonical guidance per action_name
    guidance_by_action: dict[str, Counter] = defaultdict(Counter)
    for a in actions:
        if a.guidance:
            guidance_by_action[a.action_name][a.guidance] += 1
    guidance = {name: votes.most_common(1)[0][0] for name, votes in guidance_by_action.items()}
    with (label_dir / "guidance.json").open("w", encoding="utf-8") as f:
        json.dump(guidance, f, indent=2, sort_keys=True, ensure_ascii=False)


def write_criteria_catalog(processed_root: Path, actions: Sequence[JudgedAction]) -> dict:
    """Define E1's label space: criteria per action + a global criterion index."""
    by_action: dict[str, set[str]] = defaultdict(set)
    for a in actions:
        by_action[a.action_name].update(a.criteria.keys())
    by_action_sorted = {name: sorted(texts) for name, texts in sorted(by_action.items())}
    all_criteria = sorted({t for texts in by_action.values() for t in texts})
    catalog = {
        "by_action": by_action_sorted,
        "global_index": {text: i for i, text in enumerate(all_criteria)},
        "num_criteria_global": len(all_criteria),
    }
    with (processed_root / "criteria_catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, sort_keys=True, ensure_ascii=False)
    return catalog


# --------------------------------------------------------------------------- reporting


def print_report(actions: Sequence[JudgedAction], splits: dict[str, str], catalog: dict) -> None:
    rows_by_split: dict[str, list[JudgedAction]] = {name: [] for name in SPLIT_NAMES}
    for a in actions:
        rows_by_split[splits[a.participant]].append(a)

    total = len(actions)
    participants = {a.participant for a in actions}
    print(f"\nJudged actions: {total}  |  participants: {len(participants)}  "
          f"|  records: {len({a.record_id for a in actions})}  "
          f"|  action classes: {len({a.action_name for a in actions})}  "
          f"|  global criteria: {catalog['num_criteria_global']}")

    print("\nParticipant-disjoint split:")
    for name in SPLIT_NAMES:
        rows = rows_by_split[name]
        n_part = len({a.participant for a in rows})
        n_act = len(rows)
        print(f"  {name:5s}: {n_part:3d} participants  {n_act:4d} actions ({n_act / total:.0%})")
    # leakage guard
    seen: dict[str, str] = {}
    for a in actions:
        seen.setdefault(a.participant, splits[a.participant])
    leak = any(splits[a.participant] != seen[a.participant] for a in actions)
    print(f"  participant-disjoint: {not leak}")

    # TKV fault statistics
    n_true = sum(c["n_true"] for a in actions for c in a.criteria.values())
    n_false = sum(c["n_false"] for a in actions for c in a.criteria.values())
    print(f"\nTKV: {n_true + n_false} criterion-votes  |  fault(False) rate "
          f"{n_false / (n_true + n_false):.1%}")

    # score distribution
    scores = [s for a in actions for s in a.scores]
    print("Quality-score distribution:", dict(sorted(Counter(scores).items())))

    # per-action-class counts per split (balance sanity check)
    print("\nPer-action-class counts (train/val/test):")
    classes = sorted({a.action_name for a in actions})
    for cls in classes:
        cnt = {name: sum(a.action_name == cls for a in rows_by_split[name]) for name in SPLIT_NAMES}
        print(f"  {cls:48s} {cnt['train']:4d} / {cnt['val']:3d} / {cnt['test']:3d}")


# --------------------------------------------------------------------------- entrypoint


def build_manifest_main() -> None:
    parser = argparse.ArgumentParser(description="Build the EgoExo-Fitness judged-action manifest, "
                                                 "participant-disjoint splits, and interpretable labels.")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    args = parser.parse_args()

    meta = load_json(args.annotation_root / "meta_records.json")
    iaj = load_json(args.annotation_root / "interpretable_action_judgement.json")

    record_info = build_record_info(meta)
    actions = parse_judged_actions(iaj, record_info)

    missing = sorted({a.record_id for a in actions if a.record_id not in record_info})
    if missing:
        raise SystemExit(f"{len(missing)} IAJ records absent from meta_records.json: {missing[:5]}")

    splits = assign_participant_splits(actions, val_frac=args.val_frac, test_frac=args.test_frac)

    manifest_output = args.manifest_output or args.processed_root / "manifest.csv"
    rows = build_manifest_rows(actions, splits)
    write_manifest(manifest_output, rows)
    write_splits_and_labels(args.processed_root, actions, splits)
    catalog = write_criteria_catalog(args.processed_root, actions)

    print(f"Wrote {len(rows)} judged actions to {manifest_output}")
    print(f"Artifacts under {args.processed_root}")
    print_report(actions, splits, catalog)


if __name__ == "__main__":
    build_manifest_main()
