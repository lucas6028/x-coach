"""Replay EgoExo-Fitness's judged Jumping Jacks actions through the shipped detector.

WHY THIS SHIPS RATHER THAN LIVING IN A SCRATCH SCRIPT. Torso Twist recorded that a number in a
citation of record whose script nobody can re-run is a defect this project has already logged
once (the Row residual), and re-running its own harness after a fix corrected five quoted
figures. Every number in the Jumping Jacks design spec section 8 and in
`notes/jumping-jacks-rule-validation.md` is this module's output.

WHAT THIS CAN AND CANNOT ANSWER, STATED HERE SO NO CALLER HAS TO INFER IT.

  CAN: pipeline properties on real footage of the RIGHT exercise -- does the 8-landmark validity
  gate leave usable frames, does `segment_reps` find jumping-jack repetitions, what cadence do
  real performers hold (which decides whether `min_rep_seconds` needed lowering), what does the
  view estimator return for an upright frontal subject, and how much do three SIMULTANEOUS
  cameras disagree about the same repetition (which is pure projection error, because the three
  exo views film the same instant).

  CANNOT: validate a threshold. EgoExo's eight criteria and the parent spec's five rules overlap
  in exactly ONE pair (`jj_incomplete_leg_rom` <-> "Perform the jump by opening and closing your
  feet"), and every action reachable from the truncated archive is judged TRUE on that criterion.
  So the only rule-level question the data answers is a FALSE-POSITIVE one -- how often does the
  ROM rule fire on repetitions humans judged correct -- and there is no reachable action on which
  a firing would be right. Design spec section 2.

THE INPUT IS NOT IN THE REPOSITORY. `frames_open` is a 3 GiB-split download whose `.ac` part is
missing; `.aa`+`.ab` is a contiguous gzip PREFIX and decodes to six complete records plus a
partial seventh, which is where the ELEVEN reachable actions come from. The extraction and MediaPipe passes are recorded in
the result note; this module consumes their output (pose JSON in the schema
`src/pose/process_videos.py` writes, named `{sample_id}__{view}.json`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.pose.movements.base import run_detector
from src.pose.geometry import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, landmarks_to_array,
)
from src.pose.movements.jumping_jacks import JUMPING_JACKS_DETECTOR, LEG_ROM_MILD_RATIO

# The cut `jj_knee_valgus_landing` WOULD have fired at, transcribed from `squat.rule_knees_inward`
# where it still lives. It is not imported from the Jumping Jacks module because the withdrawal
# removed it from there -- a withdrawn rule leaves no constant behind either. It survives here
# only so the withdrawal's own evidence stays re-runnable.
WITHDRAWN_VALGUS_CUT = 0.82

# The single EgoExo criterion that any shipped rule models. Design spec section 2.2.
FOOT_SPLIT_CRITERION = "Perform the jump by opening and closing your feet."

# The three third-person cameras. They film SIMULTANEOUSLY, which is what makes cross-view
# disagreement on one action pure projection error rather than performance variation.
EXO_VIEWS = ("exo_l", "exo_m", "exo_r")


def cadence_hz(rep_count: int, frame_count: int, fps: float) -> float:
    """Repetitions per second over the span the repetitions actually occupy.

    `frame_count` must be the span from the first repetition's start to the last one's end, NOT
    the whole clip: an action file carries idle frames at both ends, and dividing by those would
    report a slower cadence than the performer held -- which is the direction that would falsely
    support leaving `min_rep_seconds` alone. Design spec section 4.5 turns on this number, so the
    bias has to run against the conclusion, not with it.

    Returns NaN rather than 0.0 when there is nothing to divide by, so an unsegmented action
    cannot be averaged in as "infinitely slow" -- the mistake `view_estimation`'s NaN-not-zero
    comment records for a different quantity.
    """
    if rep_count <= 0 or frame_count <= 0 or fps <= 0:
        return math.nan
    return float(rep_count) / (float(frame_count) / float(fps))


def floor_discarded(reps_at_shipped_floor: int, reps_at_low_floor: int) -> int:
    """How many repetitions the shipped `min_rep_seconds` threw away.

    THE DIRECT MEASUREMENT IS CIRCULAR AND THIS IS THE WAY ROUND IT. Every window `segment_reps`
    RETURNS is at least `min_rep_seconds` long by construction, so measuring the shortest returned
    repetition can never show the floor biting. Re-segmenting the same signal with a much lower
    floor and differencing the counts can. A positive result means the shipped floor is discarding
    real repetitions -- i.e. that `base.py:55`'s "must lower it" was right about this movement.
    """
    return max(0, int(reps_at_low_floor) - int(reps_at_shipped_floor))


def seconds_per_rep(cadence: float) -> float:
    if not math.isfinite(cadence) or cadence <= 0:
        return math.nan
    return 1.0 / cadence


def validity_rate(core: Sequence[object]) -> float:
    if not core:
        return 0.0
    return sum(1 for frame in core if getattr(frame, "valid", False)) / len(core)


def fired_ids(detections: Iterable[object]) -> set[str]:
    return {getattr(detection, "fault_id", "") for detection in detections}


def cross_view_agreement(per_view: dict[str, set[str]], fault_id: str) -> str | None:
    """How the simultaneous cameras voted on one fault for one action.

    "unanimous_fire" / "unanimous_silent" / "split" -- and None when fewer than two views
    produced a verdict at all, which must not be reported as agreement.
    """
    votes = [fault_id in fired for fired in per_view.values()]
    if len(votes) < 2:
        return None
    if all(votes):
        return "unanimous_fire"
    if not any(votes):
        return "unanimous_silent"
    return "split"


def spread(values: Sequence[float]) -> float:
    """max - min over the finite values; NaN if fewer than two are finite.

    The cross-camera disagreement statistic. A range rather than a standard deviation because
    three cameras is too few for a spread estimate to mean much, and the range is what an
    unlucky single-camera user actually risks.
    """
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return math.nan
    return float(max(finite) - min(finite))


def aligned_knee_ratio(points, hip_l: int, hip_r: int, knee_l: int, knee_r: int,
                       ankle_l: int, ankle_r: int) -> float:
    """The knee/ankle width ratio a PERFECTLY ALIGNED pair of knees would produce in this pose.

    THIS IS THE CONTROL THE VALGUS RULE NEVER HAD. `knee_width_to_ankle_width` compares knee
    separation with ankle separation, which in a SQUAT (feet about shoulder width, shanks near
    vertical) is close to 1.0 when the knees track the feet. In a wide side-straddle the legs
    splay from a pelvis that does not widen, so a knee sits partway along the hip->ankle line and
    its separation is necessarily SMALLER than the ankles' -- with no valgus at all.

    Each knee is replaced by its projection onto the same-side hip->ankle line at its own distance
    along that limb, i.e. the position it would occupy with a perfectly straight limb, and the
    ratio is recomputed. Comparing that with the observed ratio separates the two things the
    original metric adds together:

        observed_ratio  =  (what the stance geometry forces)  +  (genuine inward deviation)
        aligned_ratio   =   what the stance geometry forces

    so `observed - aligned` is the valgus signal and `aligned` alone is the confound.
    """
    hip_left = _xy(points, hip_l)
    hip_right = _xy(points, hip_r)
    knee_left = _xy(points, knee_l)
    knee_right = _xy(points, knee_r)
    ankle_left = _xy(points, ankle_l)
    ankle_right = _xy(points, ankle_r)
    if any(item is None for item in (hip_left, hip_right, knee_left, knee_right,
                                     ankle_left, ankle_right)):
        return math.nan

    def projected(hip, knee, ankle):
        limb = ankle - hip
        length = float(np.linalg.norm(limb))
        if length <= 1e-9:
            return None
        fraction = float(np.dot(knee - hip, limb)) / (length * length)
        return hip + fraction * limb

    left = projected(hip_left, knee_left, ankle_left)
    right = projected(hip_right, knee_right, ankle_right)
    if left is None or right is None:
        return math.nan
    ankle_width = float(np.linalg.norm(ankle_left - ankle_right))
    if ankle_width <= 1e-9:
        return math.nan
    return float(np.linalg.norm(left - right)) / ankle_width


def observed_knee_ratio(points, knee_l: int, knee_r: int, ankle_l: int, ankle_r: int) -> float:
    """`knee_width_to_ankle_width` as the withdrawn rule would have read it.

    Computed here rather than read from a metric key, because the withdrawal removed the metric
    from the module -- a withdrawn rule leaves nothing behind for something to quietly start
    reading. This function exists only so the withdrawal's own evidence stays re-runnable.
    """
    knee_left, knee_right = _xy(points, knee_l), _xy(points, knee_r)
    ankle_left, ankle_right = _xy(points, ankle_l), _xy(points, ankle_r)
    if any(item is None for item in (knee_left, knee_right, ankle_left, ankle_right)):
        return math.nan
    ankle_width = float(np.linalg.norm(ankle_left - ankle_right))
    if ankle_width <= 1e-9:
        return math.nan
    return float(np.linalg.norm(knee_left - knee_right)) / ankle_width


def _xy(points, index: int):
    if points is None or index >= len(points):
        return None
    row = points[index]
    if len(row) >= 4 and float(row[3]) < 0.5:
        return None
    return np.asarray([float(row[0]), float(row[1])], dtype=np.float64)


def load_pose_frames(path: Path) -> tuple[list[dict], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    info = payload.get("video_info", {}) or {}
    fps = float(info.get("fps", 30.0) or 30.0)
    return (frames if isinstance(frames, list) else []), fps


def load_labels(labels_path: Path) -> dict[str, dict[str, int]]:
    """sample_id -> {criterion: fault flag}, from `processed/labels/tkv.json`."""
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for sample_id, criteria in payload.items():
        out[sample_id] = {
            criterion: int(record.get("fault", 0)) for criterion, record in criteria.items()
        }
    return out


# The floor `segment_reps` is re-run with in order to see whether the shipped one is discarding
# repetitions. Low enough that a 3 Hz cadence would survive it, i.e. well past anything a jumping
# jack can reach.
PROBE_MIN_REP_SECONDS = 0.15


def _reps_at_floor(frames: list[dict], fps: float, min_rep_seconds: float) -> int:
    """Re-segment this clip's rep signal at a different duration floor. Nothing else changes."""
    from dataclasses import replace

    from src.pose.geometry import centered_median
    from src.pose.rep_segmentation import segment_reps

    detector = replace(JUMPING_JACKS_DETECTOR, min_rep_seconds=min_rep_seconds)
    raw = detector.compute_raw(frames, fps)
    signal = centered_median(
        [float(item.get(detector.rep_signal, np.nan)) for item in raw], window=5
    )
    return len(
        segment_reps(
            signal,
            fps=fps,
            polarity=detector.rep_polarity,
            rectify=detector.rep_rectify,
            rep_start=detector.rep_start,
            min_rep_seconds=min_rep_seconds,
        )
    )


def evaluate_view(frames: list[dict], fps: float, view_type: str) -> dict:
    """One (action, camera) pair through the real `run_detector`."""
    result = run_detector(JUMPING_JACKS_DETECTOR, frames, fps, view_type, 0.8)
    core = result.core
    widest = [
        max(
            (
                frame.m("stance_width_ratio")
                for frame in core[rep.start : rep.end + 1]
                if frame.valid and np.isfinite(frame.m("stance_width_ratio"))
            ),
            default=math.nan,
        )
        for rep in result.analyzed
    ]
    # THE WITHDRAWN VALGUS QUANTITY, recomputed from landmarks because the module no longer
    # carries it, alongside the ALIGNED-KNEE control that is the reason it was withdrawn.
    observed_by_frame = [
        observed_knee_ratio(
            landmarks_to_array(frame.get("landmarks")),
            LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
        )
        for frame in frames
    ]
    aligned_by_frame = [
        aligned_knee_ratio(
            landmarks_to_array(frame.get("landmarks")),
            LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
        )
        for frame in frames
    ]
    tightest = [
        min(
            (
                observed_by_frame[index]
                for index in range(rep.start, min(rep.end + 1, len(observed_by_frame)))
                if core[index].valid and math.isfinite(observed_by_frame[index])
            ),
            default=math.nan,
        )
        for rep in result.analyzed
    ]
    open_indices = [
        index
        for index, frame in enumerate(core)
        if frame.valid and frame.phase == "open" and index < len(observed_by_frame)
    ]
    open_observed = [
        observed_by_frame[index] for index in open_indices
        if math.isfinite(observed_by_frame[index])
    ]
    open_aligned = [
        aligned_by_frame[index] for index in open_indices
        if math.isfinite(aligned_by_frame[index])
    ]
    span = (
        result.reps[-1].end - result.reps[0].start + 1 if result.reps else 0
    )
    probe_reps = _reps_at_floor(frames, fps, PROBE_MIN_REP_SECONDS)

    # PER-REPETITION, NOT PER (action, camera). A pair "fires" if ANY of its analysed repetitions
    # does, and `merge_by_fault` then reports one card -- so a pair-level rate counts one narrow
    # repetition in three the same as three. Both are reported because they answer different
    # questions and the pair-level one alone would overstate the rule.
    rom_hits = sum(
        1 for value in widest if math.isfinite(value) and value < LEG_ROM_MILD_RATIO
    )
    valgus_hits = sum(
        1 for value in tightest if math.isfinite(value) and value < WITHDRAWN_VALGUS_CUT
    )
    return {
        "frames": len(core),
        "validity_rate": validity_rate(core),
        "reps_found": len(result.reps),
        "reps_analyzed": len(result.analyzed),
        "reps_at_probe_floor": probe_reps,
        "reps_lost_to_the_floor": floor_discarded(len(result.reps), probe_reps),
        "fallback": result.fallback,
        "cadence_hz": cadence_hz(len(result.reps), span, fps),
        "fired": sorted(fired_ids(result.detections)),
        # WHAT THE RULES WOULD HAVE SAID. Both are silent/withdrawn, so `fired` is empty on every
        # pair by construction and an agreement statistic computed from it would be a tautology.
        # These are the verdicts the parent spec's own cuts produce on the same windows, which is
        # the quantity the cross-camera comparison is actually about.
        "would_fire": sorted(
            ([  "jj_incomplete_leg_rom"] if rom_hits else [])
            + (["jj_knee_valgus_landing"] if valgus_hits else [])
        ),
        "per_rep_widest": [value for value in widest if math.isfinite(value)],
        "per_rep_tightest": [value for value in tightest if math.isfinite(value)],
        "rom_rep_hits": rom_hits,
        "valgus_rep_hits": valgus_hits,
        "scored_reps": len(result.analyzed),
        "max_stance_width_ratio": max([value for value in widest if math.isfinite(value)], default=math.nan),
        "min_knee_ankle_ratio": min([value for value in tightest if math.isfinite(value)], default=math.nan),
        "open_frames": len(open_observed),
        "open_observed_below_cut": sum(1 for value in open_observed if value < WITHDRAWN_VALGUS_CUT),
        "open_aligned_below_cut": sum(1 for value in open_aligned if value < WITHDRAWN_VALGUS_CUT),
        "open_observed_median": _median(open_observed),
        "open_aligned_median": _median(open_aligned),
    }


def evaluate_dataset(pose_dir: Path, labels_path: Path, view_type: str = "unknown") -> dict:
    """Every `{sample_id}__{view}.json` under `pose_dir`, grouped by action."""
    labels = load_labels(labels_path)
    actions: dict[str, dict[str, dict]] = {}
    for path in sorted(pose_dir.glob("*__*.json")):
        sample_id, _, view = path.stem.partition("__")
        if view not in EXO_VIEWS:
            continue
        frames, fps = load_pose_frames(path)
        if not frames:
            continue
        actions.setdefault(sample_id, {})[view] = evaluate_view(frames, fps, view_type)

    records = []
    for sample_id, per_view in sorted(actions.items()):
        fired = {view: set(record["would_fire"]) for view, record in per_view.items()}
        records.append(
            {
                "sample_id": sample_id,
                "foot_split_fault": labels.get(sample_id, {}).get(FOOT_SPLIT_CRITERION),
                "views": per_view,
                "agreement": {
                    fault_id: cross_view_agreement(fired, fault_id)
                    for fault_id in ("jj_incomplete_leg_rom", "jj_knee_valgus_landing")
                },
                "stance_spread": spread(
                    [record["max_stance_width_ratio"] for record in per_view.values()]
                ),
                "valgus_spread": spread(
                    [record["min_knee_ankle_ratio"] for record in per_view.values()]
                ),
            }
        )
    return {"actions": records}


def summarize(payload: dict) -> dict:
    records = payload["actions"]
    per_view_records = [record for action in records for record in action["views"].values()]
    cadences = [
        record["cadence_hz"] for record in per_view_records if math.isfinite(record["cadence_hz"])
    ]
    fired_rom = sum(
        1 for record in per_view_records if "jj_incomplete_leg_rom" in record["would_fire"]
    )
    fired_valgus = sum(
        1 for record in per_view_records if "jj_knee_valgus_landing" in record["would_fire"]
    )
    judged_correct = [
        action for action in records if action["foot_split_fault"] == 0
    ]
    return {
        "actions": len(records),
        "action_camera_pairs": len(per_view_records),
        "actions_judged_correct_on_foot_split": len(judged_correct),
        "median_validity_rate": float(np.median([r["validity_rate"] for r in per_view_records]))
        if per_view_records
        else math.nan,
        "pairs_on_fallback": sum(1 for r in per_view_records if r["fallback"]),
        "median_cadence_hz": float(np.median(cadences)) if cadences else math.nan,
        "max_cadence_hz": float(np.max(cadences)) if cadences else math.nan,
        "min_seconds_per_rep": seconds_per_rep(float(np.max(cadences))) if cadences else math.nan,
        "reps_lost_to_the_floor": sum(
            record.get("reps_lost_to_the_floor", 0) for record in per_view_records
        ),
        "reps_found_total": sum(record["reps_found"] for record in per_view_records),
        "detections_emitted": sum(len(r["fired"]) for r in per_view_records),
        "leg_rom_fire_rate": fired_rom / len(per_view_records) if per_view_records else math.nan,
        "valgus_fire_rate": fired_valgus / len(per_view_records) if per_view_records else math.nan,
        "scored_reps": sum(r["scored_reps"] for r in per_view_records),
        "leg_rom_rep_fire_rate": _rate(
            sum(r["rom_rep_hits"] for r in per_view_records),
            sum(r["scored_reps"] for r in per_view_records),
        ),
        "valgus_rep_fire_rate": _rate(
            sum(r["valgus_rep_hits"] for r in per_view_records),
            sum(r["scored_reps"] for r in per_view_records),
        ),
        "median_per_rep_widest": _median(
            [value for r in per_view_records for value in r["per_rep_widest"]]
        ),
        "median_per_rep_tightest": _median(
            [value for r in per_view_records for value in r["per_rep_tightest"]]
        ),
        # THE ZERO-PARAMETER CONTROL THAT WITHDREW THE VALGUS RULE. `aligned` replaces both knees
        # with perfectly straight-limb positions, so anything it flags is stance geometry and not
        # knee alignment. Design spec section 7.1.
        "open_frames": sum(r["open_frames"] for r in per_view_records),
        "valgus_observed_frame_rate": _rate(
            sum(r["open_observed_below_cut"] for r in per_view_records),
            sum(r["open_frames"] for r in per_view_records),
        ),
        "valgus_aligned_frame_rate": _rate(
            sum(r["open_aligned_below_cut"] for r in per_view_records),
            sum(r["open_frames"] for r in per_view_records),
        ),
        "median_open_observed": _median(
            [r["open_observed_median"] for r in per_view_records]
        ),
        "median_open_aligned": _median(
            [r["open_aligned_median"] for r in per_view_records]
        ),
        "agreement_leg_rom": _agreement_counts(records, "jj_incomplete_leg_rom"),
        "agreement_valgus": _agreement_counts(records, "jj_knee_valgus_landing"),
        "median_stance_spread": _median([action["stance_spread"] for action in records]),
        "median_valgus_spread": _median([action["valgus_spread"] for action in records]),
    }


def _agreement_counts(records: Sequence[dict], fault_id: str) -> dict[str, int]:
    counts: dict[str, int] = {"unanimous_fire": 0, "unanimous_silent": 0, "split": 0}
    for action in records:
        verdict = action["agreement"].get(fault_id)
        if verdict in counts:
            counts[verdict] += 1
    return counts


def _rate(hits: int, total: int) -> float:
    return hits / total if total else math.nan


def _median(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan
