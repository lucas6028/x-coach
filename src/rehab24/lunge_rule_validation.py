"""Pure helpers for replaying REHAB24-6 Ex5's labeled repetitions through the production
lunge rules.

WHAT THIS CAN AND CANNOT MEASURE -- read before quoting any number it produces. REHAB24-6
labels each repetition `correct` or `incorrect` and NEVER states which fault occurred, so a
rule firing on an incorrect rep is not evidence it found that rep's actual error. Everything
here therefore measures whether a rule's signal CARRIES INFORMATION ABOUT REP CORRECTNESS --
not per-fault precision.

Nothing in this module touches the filesystem, so it is fully testable in CI while the pose
corpus under `data/` stays gitignored.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence, TypeVar

# Which camera affords each rule's required view. Segmentation.txt documents that a rep filmed
# `front` in cam17 is `side` in cam18 (the cameras are orthogonal and simultaneous), so the
# same repetition supplies both a frontal and a sagittal view. `src.rehab24.dataset
# .camera_orientation` already implements that mapping -- use it, do not restate it.
RULE_CAMERAS: dict[str, str] = {
    "lunge_knee_past_toes": "cam18",      # hard-gated to `side`
    "lunge_insufficient_depth": "cam18",  # spec rates the knee angle `high` on side
    "lunge_knee_valgus": "cam17",         # frontal-plane cue
    "lunge_pelvic_drop": "cam17",         # frontal-plane cue
}


# Dataset orientation -> the view label the ORACLE pass feeds the rules. `view_confidence` is
# pinned at 1.0 alongside it, since the premise of this pass is that the view is known.
#
# `front` DELIBERATELY maps to "front", a label production can NEVER emit: the production path
# calls estimate_view_for_pose(allow_front=False). That is the whole point of the oracle pass
# -- it asks "would this rule fire if the view label were correct?", which requires bypassing
# the gate rather than reproducing it. Any oracle-pass result on a `front` rep is therefore a
# statement about the RULE, never about what a user would see, and the writeup must say so
# wherever it quotes one.
#
# `half-profile` maps to "front_oblique" rather than "rear_oblique" because the dataset does
# not record which way the subject faced, and the two are equivalent for every lunge rule:
# `rule_knee_valgus` and `rule_pelvic_drop` treat both as fully observable, and the other two
# rules ignore the oblique labels entirely. Stated so nobody reads significance into the pick.
ORACLE_VIEWS: dict[str, str] = {
    "front": "front",
    "side": "side",
    "half-profile": "front_oblique",
    "profile": "side",   # unused by Ex5 (0 reps), present for completeness
}
ORACLE_VIEW_CONFIDENCE = 1.0


def slice_rep(frames: list[dict], first_frame: int, last_frame: int) -> list[dict]:
    """Frames `[first_frame, last_frame]` INCLUSIVE, clamped to what the clip actually holds.

    Clamping rather than raising: the labels are indexed on the mocap timeline and a video can
    run a few frames short, which is a truncated rep, not a corrupt one. A window that starts
    past the end of the clip yields [] and is reported as a skipped rep.
    """
    if first_frame < 0 or last_frame < first_frame:
        raise ValueError(f"invalid rep window [{first_frame}, {last_frame}]")
    return frames[first_frame : last_frame + 1]


def contingency(fired: Sequence[bool], correct: Sequence[bool]) -> dict[str, int]:
    """2x2 table with POSITIVE = the repetition is INCORRECT.

    `correct` is the dataset's `correctness` column as a bool (True == performed correctly), so
    a positive is `not correct`. Stated explicitly because getting this backwards silently
    inverts sensitivity and specificity in the writeup.
    """
    if len(fired) != len(correct):
        raise ValueError(f"length mismatch: {len(fired)} fired vs {len(correct)} labels")
    table = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for did_fire, is_correct in zip(fired, correct):
        positive = not is_correct
        if did_fire and positive:
            table["tp"] += 1
        elif did_fire and not positive:
            table["fp"] += 1
        elif not did_fire and positive:
            table["fn"] += 1
        else:
            table["tn"] += 1
    return table


def rank_auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Threshold-free AUC of `scores` against `positive`, by the rank-sum identity, ties = 0.5.

    Threshold-free on purpose: it says whether the underlying metric ORDERS incorrect reps
    above correct ones at all, independently of where the spec's cut happens to sit. That is
    what distinguishes "this cue carries no signal" from "this cue carries signal but the cited
    threshold sits in the wrong part of its distribution" -- and only the first of those is a
    reason to doubt the rule.

    NaN when either class is empty: a rule the dataset never exercises must not report 0.5,
    which reads as "no better than chance" rather than "not measured".
    """
    pairs = [(s, bool(p)) for s, p in zip(scores, positive) if math.isfinite(s)]
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


Record = TypeVar("Record")
Key = TypeVar("Key")


def per_subject(
    records: Iterable[Record],
    key_fn: Callable[[Record], Key],
    value_fn: Callable[[list[Record]], float],
) -> dict[Key, float]:
    """Group `records` by `key_fn` (the person id) and reduce each group with `value_fn`.

    174 reps come from only 8 people, so they are not independent observations: a pooled
    statistic lets one subject's separation masquerade as a population result. This project has
    already been burned twice by that exact shape of optimism -- a fixed-lambda ridge fabricating
    a null, and a 1-sequence oracle-debiased preview that ran 2.2x optimistic -- so every
    contingency/AUC number in the writeup must be reported per subject, not only pooled.
    """
    groups: dict[Key, list[Record]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    return {key: value_fn(group) for key, group in groups.items()}
