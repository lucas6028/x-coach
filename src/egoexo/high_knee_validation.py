"""Replay EgoExo-Fitness's judged High Knee actions through the shipped detector.

WHY THIS SHIPS RATHER THAN LIVING IN A SCRATCH SCRIPT. Torso Twist recorded that a number in a
citation of record whose script nobody can re-run is a defect this project has already logged once
(the Row residual), and re-running its own harness after a fix corrected five quoted figures.
Every number in the High Knee design spec sections 5-7 and in
`notes/high-knee-rule-validation.md` is this module's output.

WHAT THIS CAN AND CANNOT ANSWER, STATED HERE SO NO CALLER HAS TO INFER IT.

  CAN: pipeline properties on real footage of the RIGHT exercise -- validity, segmentation, and
  how much of this movement the shipped `min_rep_seconds` discards. It can also refute: three
  SIMULTANEOUS exo cameras film the same instant, so any disagreement between them is pure
  projection with no performance variation in it, and that is the instrument that withdraws
  `hk_contralateral_pelvic_drop`. And it can measure a rule's REFERENCE AXIS against the fault it
  is asked to detect, which is what withdraws both trunk rules.

  CANNOT: validate a threshold from the checklist. EgoExo's seven High Knee criteria and the
  parent spec's five rules overlap in ZERO pairs -- the corpus judges cadence, arm rhythm, upper-
  body stability, alternation, gaze, back-straightness and forefoot contact, and the spec writes
  rules about knee height, trunk lean in two directions, pelvic drop and stride asymmetry. Design
  spec section 2.

THE INPUT IS NOT IN THE REPOSITORY. `frames_open` is a 3 GiB-split download whose `.ac` part is
missing; `.aa`+`.ab` is a contiguous gzip PREFIX, which is where the SIX reachable actions come
from. `scripts/egoexo/extract_action_frames.py` and `scripts/egoexo/run_pose_on_frame_dirs.py`
produce this module's input (pose JSON in the schema `src/pose/process_videos.py` writes, named
`{sample_id}__{view}.json`).
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.pose.geometry import (
    LEFT_ANKLE, LEFT_HEEL, LEFT_FOOT_INDEX, LEFT_HIP, LEFT_SHOULDER,
    RIGHT_ANKLE, RIGHT_HEEL, RIGHT_FOOT_INDEX, RIGHT_HIP, RIGHT_SHOULDER,
    centered_median, landmarks_to_array, midpoint, visible_point,
)
from src.pose.movements.base import run_detector
from src.pose.movements.high_knee import (
    HIGH_KNEE_DETECTOR, KNEE_LIFT_CITED_A_SKIP, KNEE_LIFT_IMPLEMENTED_B_SKIP,
)
from src.pose.rep_segmentation import segment_reps

# The three third-person cameras. They film SIMULTANEOUSLY, which is what makes cross-view
# disagreement on one action pure projection error rather than performance variation.
EXO_VIEWS = ("exo_l", "exo_m", "exo_r")

# The criterion with a real positive class, and the closest thing the corpus has to either trunk
# rule. Its comments describe SWAY; both rules read a signed mean. Design spec section 6.3.
STABILITY_CRITERION = "Maintain a stable upper body throughout the exercise."
# The criterion the two trunk rules would actually model. No action fails it by majority.
BACK_STRAIGHT_CRITERION = "Keep your back straight."

# The cuts the two WITHDRAWN trunk rules would have fired at, transcribed from the parent spec
# (section "High Knee", `hk_trunk_lean_back` / `hk_forward_trunk_collapse`). They are not imported
# from the movement module because the withdrawal removed them from there -- a withdrawn rule
# leaves no constant behind either. They survive here only so the withdrawals' own evidence stays
# re-runnable.
WITHDRAWN_BACK_LEAN_CUT_DEG = 10.0
WITHDRAWN_FORWARD_LEAN_CUT_DEG = 15.0
# The parent spec's pelvic-obliquity cut, same provenance, same reason for living here.
WITHDRAWN_PELVIC_DROP_CUT_DEG = 5.0

# The floor `segment_reps` is re-run with in order to see whether the shipped one is discarding
# repetitions. It equals the shipped `min_rep_seconds`' own value here, because for THIS movement
# the lowered floor IS what shipped; the probe that matters is therefore the framework DEFAULT.
PROBE_DEFAULT_MIN_REP_SECONDS = 0.4


def cadence_hz(rep_count: int, span_frames: int, fps: float) -> float:
    """Repetitions per second over the span the repetitions actually occupy.

    `span_frames` must be the span from the first repetition's start to the last one's end, NOT
    the whole clip: an action file carries idle frames at both ends, and dividing by those would
    report a slower cadence than the performer held -- the direction that would falsely support
    leaving `min_rep_seconds` alone. Design spec section 5.3 turns on this number, so the bias has
    to run against the conclusion, not with it.

    Returns NaN rather than 0.0 when there is nothing to divide by, so an unsegmented action
    cannot be averaged in as "infinitely slow".
    """
    if rep_count <= 0 or span_frames <= 0 or fps <= 0:
        return math.nan
    return float(rep_count) / (float(span_frames) / float(fps))


def floor_discarded(reps_at_shipped_floor: int, reps_at_probe_floor: int) -> int:
    """How many repetitions a duration floor throws away, relative to the other.

    THE DIRECT MEASUREMENT IS CIRCULAR AND THIS IS THE WAY ROUND IT. Every window `segment_reps`
    RETURNS is at least `min_rep_seconds` long by construction, so measuring the shortest returned
    repetition can never show the floor biting. Re-segmenting the same signal at a different floor
    and differencing the counts can.
    """
    return max(0, int(reps_at_shipped_floor) - int(reps_at_probe_floor))


def signed_trunk_lean_deg(points: np.ndarray | None) -> float:
    """THE WITHDRAWN TRUNK QUANTITY, recomputed here because the module no longer carries it.

    Positive is FORWARD (shoulders anterior of the hips). The vertical comes from the SUPPORT LIMB
    -- Leg Abduction's construction, and the only one available, since Group E established that
    the image vertical is not the world vertical and this corpus ships its side cameras rolled 90
    degrees. The support limb is the leg whose thigh hangs lowest relative to the trunk, which is
    a trunk-relative choice and so does not need the vertical it is being used to find.

    :func:`trunk_to_support_limb_deg` measures how much this construction can be trusted, and the
    answer is what withdrew both rules.
    """
    vertical_up, anterior = _support_frame(points)
    if vertical_up is None or anterior is None:
        return math.nan
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    if shoulder_mid is None or hip_mid is None:
        return math.nan
    trunk = shoulder_mid - hip_mid
    norm = float(np.linalg.norm(trunk))
    if norm <= 1e-8:
        return math.nan
    sine = float(np.dot(trunk / norm, anterior))
    return math.degrees(math.asin(max(-1.0, min(1.0, sine))))


def trunk_to_support_limb_deg(points: np.ndarray | None) -> float:
    """The angle between the trunk and the support limb -- the reference axis's own error budget.

    UNSIGNED AND REFERENCE-FREE ON PURPOSE. It needs no vertical, because it is the angle between
    two body vectors, so it can be measured on rolled frames and compared across cameras. If this
    is the size of a trunk-lean threshold, then a trunk lean measured against the support limb is
    attributing an unknown share of its own reference's inclination to the trunk.
    """
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    if shoulder_mid is None or hip_mid is None:
        return math.nan
    trunk = shoulder_mid - hip_mid
    norm = float(np.linalg.norm(trunk))
    if norm <= 1e-8:
        return math.nan
    trunk_up = trunk / norm
    ankle = _support_ankle(points, hip_mid, trunk_up)
    if ankle is None:
        return math.nan
    limb = hip_mid - ankle
    limb_norm = float(np.linalg.norm(limb))
    if limb_norm <= 1e-8:
        return math.nan
    cosine = float(np.dot(trunk_up, limb / limb_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def pelvic_obliquity_deg(points: np.ndarray | None) -> float:
    """THE WITHDRAWN PELVIC QUANTITY, recomputed here for the same reason.

    The hip line's tilt away from the trunk-perpendicular, positive when the LEFT hip sits higher
    up the trunk axis. Roll-invariant (both vectors come from the body), which is what makes a
    comparison between two simultaneous cameras a statement about the cameras.
    """
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    left = visible_point(points, LEFT_HIP, dims=2)
    right = visible_point(points, RIGHT_HIP, dims=2)
    if shoulder_mid is None or hip_mid is None or left is None or right is None:
        return math.nan
    trunk = shoulder_mid - hip_mid
    hip_line = left - right
    trunk_norm = float(np.linalg.norm(trunk))
    hip_norm = float(np.linalg.norm(hip_line))
    if trunk_norm <= 1e-8 or hip_norm <= 1e-8:
        return math.nan
    sine = float(np.dot(hip_line / hip_norm, trunk / trunk_norm))
    return math.degrees(math.asin(max(-1.0, min(1.0, sine))))


def _support_ankle(points, hip_mid, trunk_up):
    """The grounded ankle: the one farthest DOWN the trunk axis from the hips."""
    candidates = [visible_point(points, index, dims=2) for index in (LEFT_ANKLE, RIGHT_ANKLE)]
    candidates = [item for item in candidates if item is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda ankle: -float(np.dot(ankle - hip_mid, trunk_up)))


def _support_frame(points):
    """(`vertical_up`, `anterior`) unit vectors from the support limb and the feet, or (None, None)."""
    if points is None:
        return None, None
    shoulder_mid = midpoint(points, LEFT_SHOULDER, RIGHT_SHOULDER, dims=2)
    hip_mid = midpoint(points, LEFT_HIP, RIGHT_HIP, dims=2)
    if shoulder_mid is None or hip_mid is None:
        return None, None
    trunk = shoulder_mid - hip_mid
    trunk_norm = float(np.linalg.norm(trunk))
    if trunk_norm <= 1e-8:
        return None, None
    ankle = _support_ankle(points, hip_mid, trunk / trunk_norm)
    if ankle is None:
        return None, None
    vertical = hip_mid - ankle
    vertical_norm = float(np.linalg.norm(vertical))
    if vertical_norm <= 1e-8:
        return None, None
    vertical_up = vertical / vertical_norm

    feet = []
    for heel_index, toe_index in ((LEFT_HEEL, LEFT_FOOT_INDEX), (RIGHT_HEEL, RIGHT_FOOT_INDEX)):
        heel = visible_point(points, heel_index, dims=2)
        toe = visible_point(points, toe_index, dims=2)
        if heel is not None and toe is not None:
            feet.append(toe - heel)
    if not feet:
        return vertical_up, None
    mean = np.mean(np.stack(feet), axis=0)
    anterior = mean - float(np.dot(mean, vertical_up)) * vertical_up
    anterior_norm = float(np.linalg.norm(anterior))
    if anterior_norm <= 1e-8:
        return vertical_up, None
    return vertical_up, anterior / anterior_norm


def fire_rate(values: Sequence[float], cut: float, *, below: bool = True) -> float:
    """Fraction of finite `values` on the firing side of `cut`. NaN when nothing is finite."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    hits = sum(1 for value in finite if (value < cut if below else value > cut))
    return hits / len(finite)


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Correlation over the frames where BOTH series are finite. NaN below 10 usable pairs.

    The two side cameras film the same instant, so this asks whether they agree about which way
    the pelvis is moving -- not merely whether their averages differ.
    """
    count = min(len(left), len(right))
    a = np.asarray(left[:count], dtype=np.float64)
    b = np.asarray(right[:count], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 10:
        return math.nan
    if np.std(a[mask]) <= 1e-12 or np.std(b[mask]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def load_pose_frames(path: Path) -> tuple[list[dict], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    info = payload.get("metadata") or payload.get("video_info") or {}
    fps = float(info.get("fps", 30.0) or 30.0)
    return (frames if isinstance(frames, list) else []), fps


def load_judgements(path: Path) -> dict[str, dict[str, bool]]:
    """sample_id -> {criterion: strict-majority-False flag}, for High Knee actions only."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, bool]] = {}
    for sample_id, record in payload.items():
        annotations = record.get("annotations", [])
        if not annotations or annotations[0].get("action_name") != "High Knee":
            continue
        votes: dict[str, list[bool]] = {}
        for annotation in annotations:
            for criterion, value in annotation.get("key_point_verification", []):
                votes.setdefault(criterion, []).append(value == "True")
        out[sample_id] = {
            criterion: sum(not v for v in values) * 2 > len(values)
            for criterion, values in votes.items()
        }
    return out


def _reps_at_floor(frames: list[dict], fps: float, min_rep_seconds: float) -> int:
    """Re-segment this clip's rep signal at a different duration floor. Nothing else changes."""
    detector = replace(HIGH_KNEE_DETECTOR, min_rep_seconds=min_rep_seconds)
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
    result = run_detector(HIGH_KNEE_DETECTOR, frames, fps, view_type, 0.8, max_reps=None)
    core = result.core

    # THE SILENT RULE'S QUANTITY: the peak thigh elevation of the DRIVING leg, per repetition.
    peaks: list[float] = []
    for rep in result.analyzed:
        window = core[rep.start : rep.end + 1]
        candidates = [
            value
            for frame in window
            if frame.valid
            for value in (frame.m("thigh_elevation_left"), frame.m("thigh_elevation_right"))
            if np.isfinite(value)
        ]
        peaks.append(max(candidates) if candidates else math.nan)

    points_by_frame = [
        landmarks_to_array(frame.get("landmarks")) if isinstance(frame, dict) else None
        for frame in frames
    ]
    lean = [signed_trunk_lean_deg(points) for points in points_by_frame]
    axis_error = [trunk_to_support_limb_deg(points) for points in points_by_frame]
    obliquity = [pelvic_obliquity_deg(points) for points in points_by_frame]
    gate = [frame.m("anterior_axis_length") for frame in core if frame.valid]

    span = (
        result.analyzed[-1].end - result.analyzed[0].start + 1 if result.analyzed else 0
    )
    shipped_reps = len(result.reps)
    return {
        "view": view_type,
        "frames": len(core),
        "validity": (sum(1 for frame in core if frame.valid) / len(core)) if core else 0.0,
        "fallback": result.fallback,
        "reps_shipped_floor": shipped_reps,
        "reps_default_floor": _reps_at_floor(frames, fps, PROBE_DEFAULT_MIN_REP_SECONDS),
        "cadence_hz": cadence_hz(len(result.analyzed), span, fps),
        "scored_reps": len(peaks),
        "peak_elevation_median": _median(peaks),
        "fire_rate_cited_cut": fire_rate(peaks, KNEE_LIFT_CITED_A_SKIP),
        "fire_rate_implemented_cut": fire_rate(peaks, KNEE_LIFT_IMPLEMENTED_B_SKIP),
        "anterior_axis_median": _median(gate),
        "trunk_lean_median_deg": _median(lean),
        "trunk_lean_sd_deg": float(np.nanstd(np.asarray(lean, dtype=np.float64)))
        if any(math.isfinite(v) for v in lean) else math.nan,
        "back_lean_fire_rate": fire_rate(lean, -WITHDRAWN_BACK_LEAN_CUT_DEG),
        "forward_lean_fire_rate": fire_rate(lean, WITHDRAWN_FORWARD_LEAN_CUT_DEG, below=False),
        "axis_error_median_deg": _median(axis_error),
        "obliquity_median_deg": _median(obliquity),
        "obliquity_series": obliquity,
    }


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return float(np.median(finite))


def cross_camera_spread(per_view: dict[str, dict], key: str, views: Sequence[str]) -> float:
    """max-min of `key` across `views` for one action. NaN below two usable views.

    The point of restricting `views` is that a quantity read from a camera its own gate says
    cannot see it is not a second opinion -- pooling it in would understate or overstate the
    spread depending on which way the degenerate view happens to fall.
    """
    values = [
        per_view[view][key]
        for view in views
        if view in per_view and math.isfinite(float(per_view[view].get(key, math.nan)))
    ]
    if len(values) < 2:
        return math.nan
    return float(max(values) - min(values))
