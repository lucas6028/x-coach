"""Pure helpers for replaying REHAB24-6 Ex5's labeled repetitions through the production
lunge rules.

WHAT THIS CAN AND CANNOT MEASURE -- read before quoting any number it produces. REHAB24-6
labels each repetition `correct` or `incorrect` and NEVER states which fault occurred, so a
rule firing on an incorrect rep is not evidence it found that rep's actual error. Everything
here therefore measures whether a rule's signal CARRIES INFORMATION ABOUT REP CORRECTNESS --
not per-fault precision.

TWO SECTIONS, and the split is load-bearing. Everything above the ORCHESTRATION banner is
pure -- it takes frames, records and numbers, never a path -- and is unit-tested in CI while
the pose corpus under `data/` stays gitignored. Everything below the banner reads pose JSON
off disk and is therefore exercised only when the corpus is present; it is kept as thin as
possible for exactly that reason, delegating every decision worth testing upward.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

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


def estimate_view_for_window(window_frames: list[dict]) -> tuple[str, float]:
    """The view label a rule would really receive for THIS repetition.

    Mirrors `estimate_view_for_pose`'s aggregation (view_estimation.py:390-414) over a rep
    window instead of a whole file, including its `allow_front=False` production default and
    its deliberate NaN -- not 0.0 -- default for `torso_width_ratio`, which exists because a 0.0
    ratio reads as "maximally narrow" and manufactures a high-confidence `side` verdict from
    clips carrying no width evidence at all.

    PER WINDOW, NEVER PER FILE, and that is a measured requirement rather than a preference:
    every Ex5 recording mixes `front` and `half-profile` repetitions roughly 50/50 because the
    subject reorients partway through (notes/lunge-view-reconnaissance.md). A whole-clip
    estimate would therefore be derived from a mixture of two orientations and be wrong for
    about half the reps in every video.
    """
    from src.pose.view_estimation import frame_view_signals, mean_finite, score_view

    signals = [frame_view_signals(f) for f in window_frames]
    valid = [s for s in signals if s is not None]
    total = len(window_frames)
    valid_frame_ratio = len(valid) / total if total else 0.0
    view_type, confidence, *_ = score_view(
        orientation_score=mean_finite([s["orientation_score"] for s in valid], default=0.0),
        face_visibility=mean_finite([s["face_visibility"] for s in valid], default=0.0),
        torso_width_ratio=mean_finite([s["torso_width_ratio"] for s in valid], default=np.nan),
        z_asymmetry_value=mean_finite([s["z_asymmetry"] for s in valid], default=0.0),
        valid_frame_ratio=valid_frame_ratio,
        allow_front=False,
    )
    return view_type, confidence


# The fire threshold each rule's continuous score is compared against, imported from the rule
# module rather than restated so the writeup's "where does the cited cut sit in the observed
# distribution" percentile can never quote a number the detector does not actually use.
def fault_thresholds() -> dict[str, float]:
    from src.pose.movements.lunge import (
        LUNGE_DEPTH_MILD_DEG, LUNGE_PELVIC_TILT_MILD_DEG, LUNGE_VALGUS_MILD,
    )
    from src.pose.pose_rule_detector import KNEE_FORWARD_MILD

    return {
        "lunge_knee_past_toes": KNEE_FORWARD_MILD,
        "lunge_knee_valgus": LUNGE_VALGUS_MILD,
        "lunge_insufficient_depth": LUNGE_DEPTH_MILD_DEG,
        "lunge_pelvic_drop": LUNGE_PELVIC_TILT_MILD_DEG,
    }


def score_spec(fault_id: str, lead: str) -> tuple[str, frozenset[str], float]:
    """(metric key, phases the rule masks on, sign) for one rule's continuous score.

    Restates each rule's own mask so an AUC can be computed for EVERY rep, not only the ones
    that fired -- a fired-only score set would be censored at the threshold and its AUC would
    measure nothing. It is a restatement and therefore a drift risk, which
    `test_score_spec_matches_each_rule_mask` pins against the rule module's own constants.

    The sign carries `rule_pelvic_drop`'s contralateral convention: `pelvis_tilt_signed_deg` is
    positive when the RIGHT hip is lower, so a LEFT-lead rep's contralateral drop is positive
    and a RIGHT-lead rep's is negative. Reading the magnitude instead would score an
    ipsilateral drop -- a different fault -- as Trendelenburg.
    """
    from src.pose.movements.lunge import LUNGE_ACTIVE_PHASES, PELVIC_DROP_PHASES

    if fault_id == "lunge_knee_past_toes":
        return f"{lead}_knee_forward_ratio", frozenset(LUNGE_ACTIVE_PHASES), 1.0
    if fault_id == "lunge_knee_valgus":
        return f"{lead}_knee_medial_offset_ratio", frozenset(LUNGE_ACTIVE_PHASES), 1.0
    if fault_id == "lunge_insufficient_depth":
        return f"{lead}_knee_angle", frozenset({"bottom"}), 1.0
    if fault_id == "lunge_pelvic_drop":
        return "pelvis_tilt_signed_deg", frozenset(PELVIC_DROP_PHASES), (1.0 if lead == "left" else -1.0)
    raise ValueError(f"unknown lunge fault id: {fault_id}")


def gate_open(fault_id: str, view_type: str, view_confidence: float) -> bool:
    """Whether this rule's VIEW GATE lets it emit anything at all under `view_type`.

    A restatement of the only two HARD gates in `lunge.py` -- `rule_knee_past_toes` requires a
    confident `side`, `rule_pelvic_drop` returns `[]` on `side`. The other two rules downgrade
    observability off-view but never go silent, so they are always open.

    A restatement is a drift risk, and every conditional contingency table in
    `notes/lunge-rule-validation.md` depends on this one being right, so it is pinned by
    EXECUTION rather than by a second copy of the same logic:
    `GateOpenTests.test_agrees_with_what_the_rules_actually_emit_across_every_view_label` drives
    each rule with a window it would fire on and compares emit/silence against this function for
    all six view labels, and a companion test walks the `side` confidence floor the same way.
    A gate change in `lunge.py` therefore fails the suite instead of silently invalidating the
    writeup.

    WHY A CONTINGENCY TABLE NEEDS THIS. Without it a rep the VIEW silenced is counted as a true
    negative, which inflates specificity and reports a rule as well-behaved on reps where it
    never ran. `rule_pelvic_drop`'s 1.000 specificity on the half-profile stratum is exactly
    that artifact: the estimator called those reps `side` and the rule was gated off on all 39
    correct ones. A specificity for a silenced rule is not a measurement.
    """
    from src.pose.pose_rule_detector import SIDE_VIEW_CONF_THRESHOLD

    if fault_id == "lunge_knee_past_toes":
        return view_type == "side" and view_confidence >= SIDE_VIEW_CONF_THRESHOLD
    if fault_id == "lunge_pelvic_drop":
        return view_type != "side"
    if fault_id in RULE_CAMERAS:
        return True
    raise ValueError(f"unknown lunge fault id: {fault_id}")


def angle_2d(points, a: int, b: int, c: int) -> float:
    """The a-b-c angle in the IMAGE PLANE only, dropping MediaPipe's pseudo-depth `z`.

    `geometry.angle_degrees` uses all three coordinates, so a production knee angle is partly a
    function of a learned depth channel rather than of anything visible. Recomputing the same
    angle without `z` says how much of a knee-flexion result that channel is carrying -- which
    is the difference between "this cue is wrong about lunges" and "this cue is unmeasurable
    from this pipeline's projection".
    """
    from src.pose.geometry import visible_point

    pa, pb, pc = (visible_point(points, i, dims=2) for i in (a, b, c))
    if pa is None or pb is None or pc is None:
        return math.nan
    ba = np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)
    bc = np.asarray(pc, dtype=np.float64) - np.asarray(pb, dtype=np.float64)
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return math.nan
    cosine = float(np.clip(float(np.dot(ba, bc)) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rules_window(result) -> list:
    """The frames the rules ACTUALLY saw, which is not always the whole labeled window.

    `run_detector` scores `core[rep.start:rep.end + 1]` for each selected rep when its own
    segmentation succeeds, and the whole clip only on a fallback path. Taking the continuous
    score over the full labeled window regardless would manufacture rows reading "the metric
    reached 0.35 and the rule stayed silent" on reps where the rule simply never saw the frame
    that produced the 0.35 -- which corrupts the exact distinction the production/oracle split
    exists to draw. Score support must equal rule support.
    """
    if result.fallback is None and result.analyzed:
        frames: list = []
        for rep in result.analyzed:
            frames.extend(result.core[rep.start : rep.end + 1])
        return frames
    return list(result.core)


def phase_frame_count(window: Sequence, phases: Iterable[str]) -> int:
    """Valid frames in `window` carrying one of `phases`.

    Compared against `run_detector`'s `min_frames` to separate "the rule did not fire" from
    "the rule COULD NOT fire": `contiguous_true_segments` needs `min_frames` consecutive
    above-threshold frames, so a window whose masked phases are shorter than that is
    structurally silent whatever the metric did. At 30 fps `min_frames` is 6, and
    `lunge_assign_phases` labels the deepest 30% `bottom` -- so a 15-frame rep has ~5 `bottom`
    frames and `rule_insufficient_depth` cannot emit on it at any knee angle. Counting these
    keeps "not exercised by this dataset" from being written where "could not fire by
    construction" is the truth.
    """
    wanted = set(phases)
    return sum(1 for frame in window if frame.valid and frame.phase in wanted)


def metric_extreme(window: Sequence, metric_key: str, phases: Iterable[str], sign: float) -> float:
    """Max of `sign * metric` over the valid, in-phase frames; NaN when there are none.

    NaN rather than a sentinel: `rank_auc` drops non-finite scores, so an unmeasurable rep is
    excluded from the AUC's denominator instead of being scored as "not faulty", which would
    quietly credit the rule for a rep it never read.
    """
    wanted = set(phases)
    values = [
        sign * frame.m(metric_key)
        for frame in window
        if frame.valid and frame.phase in wanted and math.isfinite(frame.m(metric_key))
    ]
    return max(values) if values else math.nan


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation over the pairs where both values are finite; NaN below 3 pairs.

    Rank, not Pearson: the valgus proxy is a hip-width-normalised ratio with a heavy tail, and
    the question it answers here ("does firing track step depth rather than correctness") is
    about monotone association, not linear fit.
    """
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan
    ranks_x = _average_ranks([a for a, _ in pairs])
    ranks_y = _average_ranks([b for _, b in pairs])
    n = len(pairs)
    mean_x = sum(ranks_x) / n
    mean_y = sum(ranks_y) / n
    dx = [r - mean_x for r in ranks_x]
    dy = [r - mean_y for r in ranks_y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom <= 0.0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def percentile_of(value: float, samples: Sequence[float]) -> float:
    """Fraction of finite `samples` strictly below `value`, as a percent; NaN when empty."""
    finite = [s for s in samples if math.isfinite(s)]
    if not finite:
        return math.nan
    return 100.0 * sum(1 for s in finite if s < value) / len(finite)


def median_and_range(values: Iterable[float]) -> tuple[float, float, float, int]:
    """(median, min, max, n) over the finite values; NaNs everywhere when none are finite.

    `n` is returned alongside and must be quoted with the median wherever it appears: a
    per-subject AUC is NaN for any subject whose reps are all one class, so the median's
    denominator is the number of subjects that produced a number, not the number of subjects.
    """
    finite = [float(v) for v in values if math.isfinite(v)]
    if not finite:
        return math.nan, math.nan, math.nan, 0
    return float(median(finite)), min(finite), max(finite), len(finite)


# =======================================================================================
# ORCHESTRATION -- everything below this banner reads pose JSON off disk.
# =======================================================================================

EX5_EXERCISE_ID = "5"
EX5_FPS = 30.0

# Pinned from Segmentation.csv itself, checked before any rule is run. An input that disagrees
# means the dataset, the exercise filter or the correctness polarity has moved, and every
# number downstream would be silently wrong -- so this raises rather than warns.
EX5_EXPECTED = {
    "reps": 174,
    "incorrect": 96,
    "correct": 78,
    "front": 88,
    "half_profile": 86,
    "subjects": 8,
}

POSE_FILE_SUFFIX = {
    "cam17": "-Camera17-30fps.json",
    "cam18": "-Camera18-30fps-transposed.json",
}

# `exercise_subtype` -> the lead side `resolve_lead_side` should return, for the lead-leg
# accuracy check. MediaPipe's landmark names are anatomical (LEFT_KNEE is the subject's left),
# so the mapping is the identity on the leg word and carries no camera-facing correction.
SUBTYPE_LEAD_SIDE = {
    "front leg left": "left",
    "front leg right": "right",
}


def assert_dataset_shape(segments: Sequence) -> None:
    """Raise unless the loaded Ex5 segments match the pinned counts.

    Includes the int->bool correctness conversion, which is where a polarity inversion would
    enter: `Segment.correctness` is 1 for a CORRECT rep, and `contingency` takes True == correct.
    Asserting 78 correct / 96 incorrect here means an inverted conversion cannot reach the
    writeup as a plausible-looking sensitivity/specificity swap.
    """
    counts = {
        "reps": len(segments),
        "incorrect": sum(1 for s in segments if not is_correct(s)),
        "correct": sum(1 for s in segments if is_correct(s)),
        "front": sum(1 for s in segments if s.cam17_orientation == "front"),
        "half_profile": sum(1 for s in segments if s.cam17_orientation == "half-profile"),
        "subjects": len({s.person_id for s in segments}),
    }
    if counts != EX5_EXPECTED:
        raise SystemExit(f"Ex5 shape changed: expected {EX5_EXPECTED}, loaded {counts}. STOP.")
    profile = sum(1 for s in segments if s.cam17_orientation == "profile")
    if profile:
        raise SystemExit(f"Ex5 gained {profile} `profile` reps; Phase 0 measured zero. STOP.")


def is_correct(segment) -> bool:
    """`Segment.correctness` as a bool. 1 == performed CORRECTLY, 0 == incorrect."""
    return int(segment.correctness) == 1


def load_pose_frames(path: Path) -> list[dict]:
    """Frames from a pose JSON, after asserting they are zero-origin and contiguous.

    `slice_rep` indexes by LIST POSITION, and `Segmentation.csv`'s bounds are frame numbers.
    Those agree only if `frames[i]["frame_index"] == i` throughout. If they ever diverge, every
    rep window is silently misaligned and every number in the writeup is wrong -- so this is
    checked once per file rather than assumed.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames") or []
    for position, frame in enumerate(frames):
        if int(frame.get("frame_index", -1)) != position:
            raise SystemExit(
                f"{path.name}: frame_index {frame.get('frame_index')!r} at list position "
                f"{position} -- frames are not zero-origin contiguous, so slice_rep would "
                "misalign every rep window. STOP."
            )
    return frames


def _scores_for_lead(window, lead: str | None, min_frames: int) -> tuple[dict, dict]:
    scores: dict[str, float | None] = {}
    structural: dict[str, bool] = {}
    for fault_id in RULE_CAMERAS:
        if lead is None:
            scores[fault_id] = None
            structural[fault_id] = True
            continue
        metric_key, phases, sign = score_spec(fault_id, lead)
        value = metric_extreme(window, metric_key, phases, sign)
        scores[fault_id] = None if not math.isfinite(value) else round(float(value), 6)
        structural[fault_id] = phase_frame_count(window, phases) < min_frames
    return scores, structural


def full_window_premise(window_frames: list[dict]) -> dict:
    """`resolve_lead_side`'s premise measured over the FULL labeled window.

    TWO INDEPENDENCE PROPERTIES, both of which the harness's other numbers lack.

    1. NO RE-SEGMENTATION. `run_detector` segments whatever it is handed, and on most reps it
       re-cut the labeled window and scored a sub-window of it. Anything derived from
       `rules_window` inherits that cut. This reads the labeled window directly -- raw metrics
       only, no `centered_median`, no `segment_reps`, no phases -- so the premise finding does
       not rest on the harness's own windowing.
    2. AN IMAGE-PLANE CONTROL. It returns each knee angle twice: as `lunge_compute_raw` computes
       it (all three coordinates, so partly a function of MediaPipe's learned `z`) and in the
       image plane alone. The gap between the two premise rates is how much of the result the
       pseudo-depth channel is carrying, and it is the difference between a claim about lunges
       and a claim about this pipeline's projection.
    """
    from src.pose.geometry import (
        landmarks_to_array, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    )
    from src.pose.movements.lunge import lunge_compute_raw

    raw = lunge_compute_raw(window_frames, EX5_FPS)
    best_position, best_value = None, math.inf
    for position, item in enumerate(raw):
        if not item.get("valid"):
            continue
        value = float(item.get("min_knee_angle", math.nan))
        if math.isfinite(value) and value < best_value:
            best_position, best_value = position, value
    if best_position is None:
        return {"bottom_position": None}

    points = landmarks_to_array(window_frames[best_position].get("landmarks"))
    return {
        "bottom_position": best_position,
        "valid_frames": sum(1 for item in raw if item.get("valid")),
        "left_knee_angle": _round_finite(float(raw[best_position]["left_knee_angle"])),
        "right_knee_angle": _round_finite(float(raw[best_position]["right_knee_angle"])),
        "left_knee_angle_2d": _round_finite(angle_2d(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)),
        "right_knee_angle_2d": _round_finite(angle_2d(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)),
    }


def _pass_record(
    result, window, lead: str | None, min_frames: int, truth_lead: str | None,
    view_type: str, view_confidence: float,
) -> dict:
    """One pass's outcome, plus a LEAD-ORACLE score set alongside the real one.

    `scores_lead_oracle` reads each rule's own metric off the leg `exercise_subtype` says led,
    instead of the leg `resolve_lead_side` picked. It is an AUC-ONLY diagnostic and cannot be a
    fire/no-fire result: the rules resolve the lead side internally, so no substitution outside
    them changes what they emit. Its purpose is to separate "this cue carries no information
    about correctness" from "this cue was read off the wrong leg" -- exactly the split the
    production/oracle VIEW passes draw for the view gate. No threshold moves either way.
    """
    fired = {
        detection.fault_id: {
            "severity": round(float(detection.severity), 4),
            "confidence": round(float(detection.confidence), 4),
            "observability": detection.observability,
            "primary_value": detection.evidence.get("primary_value"),
        }
        for detection in result.detections
    }
    scores, structural = _scores_for_lead(window, lead, min_frames)
    oracle_scores, _ = _scores_for_lead(window, truth_lead, min_frames)
    return {
        "fallback": result.fallback,
        "analyzed_reps": len(result.analyzed),
        "lead_side": lead,
        "fired": fired,
        "scores": scores,
        "scores_lead_oracle": oracle_scores,
        "cannot_fire": structural,
        "gate_open": {f: gate_open(f, view_type, view_confidence) for f in RULE_CAMERAS},
        # The reps on which the rule could ACTUALLY act: its view gate open AND enough masked
        # frames to clear min_frames AND a resolved lead side. Every other rep is a structural
        # silence, and counting those as true negatives is what inflates specificity.
        "actionable": {
            f: gate_open(f, view_type, view_confidence) and not structural[f] for f in RULE_CAMERAS
        },
    }


def evaluate_rep(frames: list[dict], segment, camera: str) -> dict | None:
    """Replay one labeled repetition through the lunge rules, twice.

    Returns None when the labeled window starts past the end of the clip (reported as a
    skipped rep, never silently dropped).
    """
    from src.pose.geometry import (
        landmarks_to_array, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    )
    from src.pose.movements.base import run_detector
    from src.pose.movements.lunge import LUNGE_DETECTOR, resolve_lead_side
    from src.rehab24.dataset import camera_orientation

    window_frames = slice_rep(frames, segment.first_frame, segment.last_frame)
    if not window_frames:
        return None

    estimated_view, estimated_conf = estimate_view_for_window(window_frames)
    orientation = camera_orientation(segment, camera)
    oracle_view = ORACLE_VIEWS.get(orientation, "unknown")
    min_frames = max(3, int(math.ceil(max(EX5_FPS, 1.0) * 0.20)))

    record = {
        "video_id": segment.video_id,
        "repetition_number": segment.repetition_number,
        "person_id": segment.person_id,
        "camera": camera,
        "correctness": int(segment.correctness),
        "correct": is_correct(segment),
        "exercise_subtype": segment.exercise_subtype,
        "cam17_orientation": segment.cam17_orientation,
        "camera_orientation": orientation,
        "extra_person": (
            segment.extra_person_in_cam17 if camera == "cam17" else segment.extra_person_in_cam18
        ),
        "lights_on": segment.lights_on,
        "first_frame": segment.first_frame,
        "last_frame": segment.last_frame,
        "window_frames": len(window_frames),
        "truncated": segment.last_frame >= len(frames),
        "estimated_view": estimated_view,
        "estimated_confidence": round(float(estimated_conf), 4),
        "oracle_view": oracle_view,
        "min_frames": min_frames,
        "lead_side_truth": SUBTYPE_LEAD_SIDE.get(segment.exercise_subtype),
        "full_window": full_window_premise(window_frames),
    }

    for pass_name, view, confidence in (
        ("production", estimated_view, float(estimated_conf)),
        ("oracle", oracle_view, ORACLE_VIEW_CONFIDENCE),
    ):
        result = run_detector(LUNGE_DETECTOR, window_frames, EX5_FPS, view, confidence)
        window = rules_window(result)
        lead = resolve_lead_side(window)
        record[pass_name] = _pass_record(
            result, window, lead, min_frames, record["lead_side_truth"], view, confidence
        )
        if pass_name == "production":
            record["valid_frame_ratio"] = (
                round(sum(1 for f in window if f.valid) / len(window), 4) if window else 0.0
            )
            record["rules_window_frames"] = len(window)
            record["lead_min_knee_angle"] = (
                None
                if lead is None
                else _finite_or_none(metric_extreme(window, f"{lead}_knee_angle", _all_phases(window), -1.0))
            )
            # Both knee angles at the window's deepest frame, so the report can check
            # `resolve_lead_side`'s PREMISE ("the lead leg is the more flexed one at the
            # bottom") against the label directly, separately from the heuristic's accuracy.
            bottom = _bottom_frame(window)
            record["bottom_left_knee_angle"] = None if bottom is None else _round_finite(bottom.m("left_knee_angle"))
            record["bottom_right_knee_angle"] = None if bottom is None else _round_finite(bottom.m("right_knee_angle"))
            # The SAME image-plane control `full_window_premise` applies, but at the SCORED
            # window's bottom frame. Carrying both makes the scored-window/full-window split the
            # only difference between the two sets of premise figures, so a reader can see that
            # the variants differ by which frame is chosen and by nothing else.
            record["bottom_left_knee_angle_2d"] = None
            record["bottom_right_knee_angle_2d"] = None
            if bottom is not None:
                # `lunge_compute_raw` copies each frame's own `frame_index`, and `window_frames`
                # is a contiguous slice starting at `segment.first_frame`, so this maps a
                # CoreFrame back to the landmark payload it was computed from.
                position = int(bottom.frame_index) - int(segment.first_frame)
                if 0 <= position < len(window_frames):
                    points = landmarks_to_array(window_frames[position].get("landmarks"))
                    record["bottom_left_knee_angle_2d"] = _round_finite(
                        angle_2d(points, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
                    )
                    record["bottom_right_knee_angle_2d"] = _round_finite(
                        angle_2d(points, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
                    )
    return record


def _bottom_frame(window: Sequence):
    best, best_value = None, math.inf
    for frame in window:
        if not frame.valid:
            continue
        value = frame.m("min_knee_angle")
        if math.isfinite(value) and value < best_value:
            best, best_value = frame, value
    return best


def _round_finite(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), 3)


def _all_phases(window: Sequence) -> set[str]:
    return {frame.phase for frame in window}


def _finite_or_none(value: float) -> float | None:
    # metric_extreme with sign -1 returns -min, so negate back to report the minimum itself.
    return None if not math.isfinite(value) else round(-float(value), 4)


def evaluate_dataset(pose_dir: Path, segmentation_path: Path) -> dict:
    from src.rehab24.dataset import read_segmentation

    segments = [
        s for s in read_segmentation(segmentation_path) if s.exercise_id == EX5_EXERCISE_ID
    ]
    assert_dataset_shape(segments)

    records: list[dict] = []
    skipped: list[dict] = []
    for camera, suffix in POSE_FILE_SUFFIX.items():
        for video_id in sorted({s.video_id for s in segments}):
            frames = load_pose_frames(pose_dir / f"{video_id}{suffix}")
            for segment in [s for s in segments if s.video_id == video_id]:
                record = evaluate_rep(frames, segment, camera)
                if record is None:
                    skipped.append(
                        {"video_id": video_id, "camera": camera,
                         "repetition_number": segment.repetition_number,
                         "reason": "window starts past the end of the clip"}
                    )
                    continue
                records.append(record)
    return {
        "expected": EX5_EXPECTED,
        "n_segments": len(segments),
        "records": records,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------------------
# REPORTING -- pure again from here: it reads the payload dict, never the disk. Kept as a
# separate pass over the saved JSON so a reporting bug costs a re-print, not a re-run.
# ---------------------------------------------------------------------------------------

PASSES = ("production", "oracle")


def _score_of(record: dict, pass_name: str, fault_id: str, score_key: str = "scores") -> float:
    value = record[pass_name].get(score_key, {}).get(fault_id)
    return math.nan if value is None else float(value)


def subset_stats(
    records: Sequence[dict], fault_id: str, pass_name: str, score_key: str = "scores"
) -> dict:
    """Contingency, sensitivity/specificity, pooled AUC and per-subject AUC for one subset.

    Every count is reported alongside its denominator: `rank_auc` drops reps whose score is
    non-finite (an unresolved lead side yields no score at all), and a subject whose reps are
    all one class yields a NaN AUC, so neither "174" nor "8" can be assumed downstream.
    """
    fired = [fault_id in r[pass_name]["fired"] for r in records]
    correct = [bool(r["correct"]) for r in records]
    scores = [_score_of(r, pass_name, fault_id, score_key) for r in records]
    positive = [not c for c in correct]

    table = contingency(fired, correct)
    sens = table["tp"] / (table["tp"] + table["fn"]) if (table["tp"] + table["fn"]) else math.nan
    spec = table["tn"] / (table["tn"] + table["fp"]) if (table["tn"] + table["fp"]) else math.nan

    # The same table over ONLY the reps where the rule could act -- view gate open, masked phase
    # long enough, lead side resolved. The unconditional table above counts every structurally
    # silent rep as a true negative, which deflates sensitivity and inflates specificity by an
    # amount the reader cannot see unless it is printed next to it.
    act_index = [i for i, r in enumerate(records) if r[pass_name].get("actionable", {}).get(fault_id)]
    act_table = contingency([fired[i] for i in act_index], [correct[i] for i in act_index])
    act_sens_denominator = act_table["tp"] + act_table["fn"]
    act_spec_denominator = act_table["tn"] + act_table["fp"]

    per_subject_auc = per_subject(
        list(range(len(records))),
        key_fn=lambda i: records[i]["person_id"],
        value_fn=lambda idx: rank_auc([scores[i] for i in idx], [positive[i] for i in idx]),
    )
    med, low, high, n_subj = median_and_range(per_subject_auc.values())
    scored = [s for s in scores if math.isfinite(s)]
    return {
        "n": len(records),
        "n_incorrect": sum(positive),
        "n_correct": sum(correct),
        "fired": sum(fired),
        "fired_on_incorrect": sum(1 for f, p in zip(fired, positive) if f and p),
        "fired_on_correct": sum(1 for f, p in zip(fired, positive) if f and not p),
        "table": table,
        "sensitivity": sens,
        "specificity": spec,
        "pooled_auc": rank_auc(scores, positive),
        "n_scored": len(scored),
        "n_scored_incorrect": sum(1 for s, p in zip(scores, positive) if math.isfinite(s) and p),
        "n_scored_correct": sum(1 for s, p in zip(scores, positive) if math.isfinite(s) and not p),
        "subject_auc_median": med,
        "subject_auc_min": low,
        "subject_auc_max": high,
        "subject_auc_n": n_subj,
        "n_subjects": len({r["person_id"] for r in records}),
        "cannot_fire": sum(1 for r in records if r[pass_name]["cannot_fire"].get(fault_id)),
        "view_gated": sum(
            1 for r in records if not r[pass_name].get("gate_open", {}).get(fault_id, True)
        ),
        # `view_gated` and `cannot_fire` OVERLAP -- a rep can be both. Reporting only the two
        # component counts invites a reader to add them and get a total larger than the number
        # of reps, which is why the union (and the overlap that explains it) is carried too.
        "silence_overlap": sum(
            1 for r in records
            if not r[pass_name].get("gate_open", {}).get(fault_id, True)
            and r[pass_name]["cannot_fire"].get(fault_id)
        ),
        "n_non_actionable": len(records) - len(act_index),
        "unresolved_lead": sum(1 for r in records if r[pass_name]["lead_side"] is None),
        "n_actionable": len(act_index),
        "actionable_table": act_table,
        "actionable_sensitivity": (
            act_table["tp"] / act_sens_denominator if act_sens_denominator else math.nan
        ),
        "actionable_specificity": (
            act_table["tn"] / act_spec_denominator if act_spec_denominator else math.nan
        ),
        "scores": scores,
    }


def _fmt(value: float, places: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{places}f}"


def _stats_lines(label: str, stats: dict, threshold: float) -> list[str]:
    t = stats["table"]
    a = stats["actionable_table"]
    return [
        f"    {label} (n={stats['n']}: {stats['n_incorrect']} incorrect / {stats['n_correct']} correct)",
        f"      fired {stats['fired']:>3}  "
        f"[tp {t['tp']:>3} fp {t['fp']:>3} tn {t['tn']:>3} fn {t['fn']:>3}]  "
        f"sens {_fmt(stats['sensitivity'])}  spec {_fmt(stats['specificity'])}",
        f"      STRUCTURAL SILENCE inside that table: {stats['n_non_actionable']} reps "
        f"(= {stats['view_gated']} view-gated OR {stats['cannot_fire']} could-not-fire, "
        f"overlapping on {stats['silence_overlap']}; of the latter "
        f"{stats['unresolved_lead']} are an unresolved lead side) -- all counted above as "
        f"true negatives / false negatives. THE TWO COMPONENTS DO NOT ADD; the union is what "
        f"leaves {stats['n_actionable']} actionable.",
        f"      CONDITIONAL on the {stats['n_actionable']} reps where the rule could act: "
        f"[tp {a['tp']:>3} fp {a['fp']:>3} tn {a['tn']:>3} fn {a['fn']:>3}]  "
        f"sens {_fmt(stats['actionable_sensitivity'])}  "
        f"spec {_fmt(stats['actionable_specificity'])}",
        f"      pooled AUC {_fmt(stats['pooled_auc'])}  over {stats['n_scored']} scored reps "
        f"({stats['n_scored_incorrect']} incorrect / {stats['n_scored_correct']} correct)",
        f"      PER-SUBJECT AUC median {_fmt(stats['subject_auc_median'])}  "
        f"range [{_fmt(stats['subject_auc_min'])}, {_fmt(stats['subject_auc_max'])}]  "
        f"over {stats['subject_auc_n']}/{stats['n_subjects']} subjects with both classes",
        f"      threshold {threshold:g} sits at percentile "
        f"{_fmt(percentile_of(threshold, stats['scores']), 1)} of the observed scores",
    ]


def matched_n_lines(records: Sequence[dict], fault_id: str, pass_name: str) -> list[str]:
    """Shipped-lead vs labeled-lead AUC over ONLY the reps BOTH score.

    The two score sets have different denominators (an unresolved lead side kills one but not
    the other), so the headline contrast between them could in principle be a denominator
    artifact. Restricting both to the same reps removes that reading.
    """
    both = [
        r for r in records
        if math.isfinite(_score_of(r, pass_name, fault_id))
        and math.isfinite(_score_of(r, pass_name, fault_id, "scores_lead_oracle"))
    ]
    shipped = subset_stats(both, fault_id, pass_name)
    labeled = subset_stats(both, fault_id, pass_name, "scores_lead_oracle")
    return [
        f"      MATCHED n={len(both)} (reps scored by BOTH lead choices): "
        f"shipped lead per-subject median {_fmt(shipped['subject_auc_median'])} "
        f"(pooled {_fmt(shipped['pooled_auc'])})  vs  labeled lead per-subject median "
        f"{_fmt(labeled['subject_auc_median'])} (pooled {_fmt(labeled['pooled_auc'])})",
    ]


def _lead_oracle_line(stats: dict) -> str:
    """The same cut's AUC with the metric read off the leg `exercise_subtype` names.

    AUC-only by construction: the rules resolve the lead side internally, so substituting it
    outside them cannot change what fires. It exists to separate "this cue carries no
    information about correctness" from "this cue was read off the wrong leg".
    """
    return (
        f"      LEAD-ORACLE (metric off the labeled lead leg; AUC only): "
        f"pooled {_fmt(stats['pooled_auc'])} over {stats['n_scored']} scored; "
        f"per-subject median {_fmt(stats['subject_auc_median'])} "
        f"range [{_fmt(stats['subject_auc_min'])}, {_fmt(stats['subject_auc_max'])}] "
        f"over {stats['subject_auc_n']}/{stats['n_subjects']} subjects"
    )


def build_report(payload: dict) -> str:
    records = payload["records"]
    thresholds = fault_thresholds()
    lines: list[str] = []
    add = lines.append

    add("=" * 88)
    add("LUNGE RULE VALIDATION -- REHAB24-6 Ex5 (leg lunge)")
    add("=" * 88)
    add("")
    add("WHAT THIS MEASURES: whether each rule's signal carries information about whether a")
    add("repetition was performed CORRECTLY. REHAB24-6 never names which fault occurred, so a")
    add("rule firing on an incorrect rep is NOT evidence it found that rep's actual error.")
    add("")
    add(f"segments loaded: {payload['n_segments']}  records: {len(records)} "
        f"(= reps x 2 cameras)  skipped: {len(payload['skipped'])}")
    for entry in payload["skipped"]:
        add(f"  SKIPPED {entry}")
    per_person = defaultdict(Counter)
    for record in records:
        if record["camera"] == "cam17":
            per_person[record["person_id"]][record["correct"]] += 1
    add("  reps per subject (correct/incorrect), cam17 -- a single-class subject yields a NaN "
        "per-subject AUC and drops out of every median below:")
    for person_id in sorted(per_person):
        counts = per_person[person_id]
        flag = "  <-- SINGLE CLASS" if not (counts[True] and counts[False]) else ""
        add(f"    person {person_id}: {counts[True]} correct / {counts[False]} incorrect{flag}")
    add("")

    for camera in ("cam17", "cam18"):
        subset = [r for r in records if r["camera"] == camera]
        add(f"{camera}: {len(subset)} records  "
            f"fallback={dict(Counter(r['production']['fallback'] for r in subset))}  "
            f"analyzed_reps={dict(Counter(r['production']['analyzed_reps'] for r in subset))}")
        add(f"  estimated view (production): "
            f"{dict(Counter((r['camera_orientation'], r['estimated_view']) for r in subset))}")
        vfr = [r["valid_frame_ratio"] for r in subset]
        add(f"  frame validity: mean {_fmt(sum(vfr) / len(vfr))} min {_fmt(min(vfr))}; "
            f"window frames {min(r['window_frames'] for r in subset)}-"
            f"{max(r['window_frames'] for r in subset)}; "
            f"truncated {sum(1 for r in subset if r['truncated'])}")
    add("")

    for fault_id, camera in RULE_CAMERAS.items():
        threshold = thresholds[fault_id]
        subset = [r for r in records if r["camera"] == camera]
        add("-" * 88)
        add(f"{fault_id}   [{camera}]   spec fire threshold {threshold:g}")
        add("-" * 88)
        for pass_name in PASSES:
            add(f"  {pass_name.upper()} pass")
            cuts = [("ALL", subset)]
            for orientation in ("front", "half-profile"):
                stratum = [r for r in subset if r["cam17_orientation"] == orientation]
                cuts.append(
                    (f"stratum cam17={orientation} (cam18={stratum[0]['camera_orientation']})", stratum)
                )
            if camera == "cam17":
                cuts.append(
                    ("extra-person-clean (levels 0/1 only)",
                     [r for r in subset if r["extra_person"] not in {"2", "3"}])
                )
            for label, cut in cuts:
                stats = subset_stats(cut, fault_id, pass_name)
                for line in _stats_lines(label, stats, threshold):
                    add(line)
                add(_lead_oracle_line(subset_stats(cut, fault_id, pass_name, "scores_lead_oracle")))
                for line in matched_n_lines(cut, fault_id, pass_name):
                    add(line)
            add("")

    add("-" * 88)
    add("LEAD LEG")
    add("-" * 88)
    for camera in ("cam17", "cam18"):
        subset = [r for r in records if r["camera"] == camera]
        resolved = [r for r in subset if r["production"]["lead_side"] is not None]
        agree = sum(
            1 for r in resolved
            if r["production"]["lead_side"] == SUBTYPE_LEAD_SIDE.get(r["exercise_subtype"])
        )
        add(f"  {camera}: resolved {len(resolved)}/{len(subset)} "
            f"(unresolved {len(subset) - len(resolved)}, "
            f"{_fmt(100 * (len(subset) - len(resolved)) / len(subset), 1)}%); "
            f"accuracy vs exercise_subtype {agree}/{len(resolved)} = "
            f"{_fmt(agree / len(resolved) if resolved else math.nan)}")
    paired = defaultdict(dict)
    for r in records:
        paired[(r["video_id"], r["repetition_number"])][r["camera"]] = r["production"]["lead_side"]
    both = [v for v in paired.values() if v.get("cam17") and v.get("cam18")]
    add(f"  cam17-vs-cam18 lead agreement on the same rep: "
        f"{sum(1 for v in both if v['cam17'] == v['cam18'])}/{len(both)}")
    add("  PREMISE CHECK -- is the labeled lead leg actually the MORE FLEXED knee at the bottom?")
    add("  (`resolve_lead_side` substitutes the more-flexed half of the spec's")
    add("  \"more flexed / more anterior\" definition; this asks whether that half holds here.)")
    add("  FOUR variants on a 2x2: which BOTTOM FRAME (scored window vs full labeled window) x")
    add("  which GEOMETRY (all three coordinates vs image plane only). Reading them as a grid is")
    add("  what shows that the frame choice and the pseudo-depth channel are separate effects.")
    add("    [scored window] the frames the rules saw (re-cut by segment_reps on most reps)")
    add("    [full window]   the whole labeled window, no segmentation, no smoothing")
    add("    [ +2d ]         the same frame's angle with MediaPipe's pseudo-depth z dropped")
    add("  THE FULL-WINDOW ROWS ARE THE QUOTED ONES: they are segmentation-independent, which is")
    add("  exactly the property this argument must not borrow from the harness's own windowing.")
    for camera in ("cam17", "cam18"):
        for variant, left_key, right_key, source in (
            ("scored window", "bottom_left_knee_angle", "bottom_right_knee_angle", None),
            ("scored window +2d", "bottom_left_knee_angle_2d", "bottom_right_knee_angle_2d", None),
            ("full window", "left_knee_angle", "right_knee_angle", "full_window"),
            ("full window +2d", "left_knee_angle_2d", "right_knee_angle_2d", "full_window"),
        ):
            hits, gaps, misses = 0, [], []
            for record in records:
                if record["camera"] != camera or record["lead_side_truth"] is None:
                    continue
                holder = record if source is None else record.get(source) or {}
                left, right = holder.get(left_key), holder.get(right_key)
                if left is None or right is None:
                    continue
                lead_angle, trail_angle = (
                    (left, right) if record["lead_side_truth"] == "left" else (right, left)
                )
                gaps.append(abs(lead_angle - trail_angle))
                if lead_angle < trail_angle:
                    hits += 1
                else:
                    misses.append(abs(lead_angle - trail_angle))
            total = len(gaps)
            med, _, _, _ = median_and_range(gaps)
            miss_med, _, _, _ = median_and_range(misses)
            add(f"    {camera} [{variant:>17}]: {hits}/{total} = "
                f"{_fmt(hits / total if total else math.nan)}; median |L-R| separation "
                f"{_fmt(med, 1)} deg overall, {_fmt(miss_med, 1)} deg ON THE REPS IT GETS WRONG")
    add("  CROSS-CAMERA CONTROL -- the two cameras film the SAME body at the SAME instant, so")
    add("  any disagreement about which knee is more flexed is measurement error, not anatomy.")
    add("  Reported on the same two frame populations, because the frame choice is the ONLY")
    add("  thing that differs between them:")
    for variant, left_key, right_key, source in (
        ("scored window", "bottom_left_knee_angle", "bottom_right_knee_angle", None),
        ("full window", "left_knee_angle", "right_knee_angle", "full_window"),
    ):
        paired_flex: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for record in records:
            holder = record if source is None else record.get(source) or {}
            left, right = holder.get(left_key), holder.get(right_key)
            if left is None or right is None:
                continue
            paired_flex[(record["video_id"], record["repetition_number"])][record["camera"]] = (
                "left" if left < right else "right"
            )
        pairs = [v for v in paired_flex.values() if "cam17" in v and "cam18" in v]
        disagree = sum(1 for v in pairs if v["cam17"] != v["cam18"])
        add(f"    [{variant:>13}] they disagree on {disagree}/{len(pairs)} = "
            f"{_fmt(disagree / len(pairs) if pairs else math.nan)} of reps")
    add("")

    add("-" * 88)
    add("VALGUS CONTAMINATION DIAGNOSTIC (cam17, production pass)")
    add("-" * 88)
    add("  A deep, well-tracked lunge adds ANTERIOR knee travel to the medial-offset proxy in")
    add("  every view production reaches. Correct reps have no valgus to find, so a strong")
    add("  NEGATIVE rank correlation between the valgus proxy and the lead knee's bottom-phase")
    add("  angle (deeper = smaller angle = larger proxy) on the CORRECT reps is the")
    add("  contamination signature.")
    add("  Both variants are pooled across subjects -- a diagnostic, never a headline.")
    for key, described in (
        ("scores_lead_oracle", "LABELED lead leg (the clean read)"),
        ("scores", "resolve_lead_side's lead leg (confounded by its own error rate)"),
    ):
        for want_correct, class_name in ((True, "CORRECT"), (False, "incorrect, for contrast")):
            cut = [r for r in records if r["camera"] == "cam17" and r["correct"] is want_correct]
            proxy = [_score_of(r, "production", "lunge_knee_valgus", key) for r in cut]
            # The depth read is `lunge_insufficient_depth`'s own quantity: the maximum lead-knee
            # angle during `bottom`. LARGER = shallower, so contamination shows up NEGATIVE.
            depth = [_score_of(r, "production", "lunge_insufficient_depth", key) for r in cut]
            n = sum(1 for p, d in zip(proxy, depth) if math.isfinite(p) and math.isfinite(d))
            add(f"  {described} / {class_name}: Spearman rho = "
                f"{_fmt(spearman_rho(proxy, depth))} over {n} reps")
    add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI body for `scripts/rehab24/validate_lunge_rules.py`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay REHAB24-6 Ex5's labeled repetitions through the lunge rules."
    )
    parser.add_argument("--pose-dir", type=Path,
                        default=Path("data/REHAB24-6/processed/lunge_pose_json"))
    parser.add_argument("--segmentation", type=Path, default=Path("data/REHAB24-6/Segmentation.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/REHAB24-6/processed/lunge_rule_validation.json"))
    parser.add_argument("--report-only", action="store_true",
                        help="Re-print the report from an existing --out file without re-running.")
    args = parser.parse_args(argv)

    if args.report_only:
        with args.out.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = evaluate_dataset(args.pose_dir, args.segmentation)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print(f"Wrote {len(payload['records'])} per-rep records to {args.out}")
    print(build_report(payload))
    return 0
