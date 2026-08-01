from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from src.pose.geometry import centered_median
from src.pose.pose_rule_detector import PoseRuleDetection
from src.pose.rep_segmentation import DEFAULT_MIN_REP_SECONDS, RepWindow, segment_reps, select_reps


@dataclass(frozen=True)
class CoreFrame:
    frame_index: int
    time: float
    phase: str
    valid: bool
    lower_body_visibility: float
    metrics: dict[str, float] = field(default_factory=dict)

    def m(self, key: str) -> float:
        return float(self.metrics.get(key, math.nan))


@dataclass(frozen=True)
class RuleContext:
    fps: float
    view_type: str
    view_confidence: float
    min_frames: int


RuleFn = Callable[[list["CoreFrame"], "RuleContext"], list[PoseRuleDetection]]


@dataclass(frozen=True)
class MovementDetector:
    name: str
    metric_keys: tuple[str, ...]
    compute_raw: Callable[[Sequence[object], float], list[dict]]
    assign_phases: Callable[[list[dict]], list[str]]
    rules: tuple[RuleFn, ...]
    # Whether this detector's rules have been checked against labeled ground truth. Defaults to
    # False so a newly registered detector surfaces as Beta in the UI rather than silently
    # presenting as validated; Squat opts in explicitly.
    validated: bool = False
    # How this movement's repetitions are found. `rep_signal` names the metric (it MUST be one
    # of `metric_keys`) whose excursion defines a rep; None disables segmentation for this
    # movement and takes the whole-clip fallback. The remaining knobs exist because the 16
    # movements in the rule spec do not all share one shape -- see the spec's §3.4 audit:
    # `rep_rectify` for bipolar signals (torso twist swings to both sides), `rep_start="flexed"`
    # for movements whose rep starts at the bottom (deadlift, from the floor), and
    # `min_rep_seconds` for fast cyclic movements (high knees run ~3Hz, about 10 frames per rep
    # at 30fps, which the default would discard as noise).
    rep_signal: str | None = None
    rep_polarity: str = "min"
    rep_rectify: bool = False
    rep_start: str = "extended"
    min_rep_seconds: float = DEFAULT_MIN_REP_SECONDS


# Phase for frames belonging to no repetition: walking in, racking, resting between reps.
# Every rule gates on its movement's active phases, so "rest" frames are never scored -- that
# suppression is the point, not a side effect.
REST_PHASE = "rest"

DEFAULT_MAX_REPS = 3


@dataclass(frozen=True)
class RunResult:
    core: list[CoreFrame]
    detections: list[PoseRuleDetection]
    reps: list[RepWindow]
    analyzed: list[RepWindow]
    # None when reps were segmented normally; otherwise why the whole clip was analyzed
    # instead: "no_reps_detected", "only_partial_reps", or "segmentation_disabled".
    fallback: str | None


def merge_by_fault(detections: list[PoseRuleDetection]) -> list[PoseRuleDetection]:
    """Collapse each fault to its worst occurrence, recording which reps it fired in.

    One card per fault rather than one per rep: three near-identical "knees inward" entries
    read as three problems. Severity, timing and evidence all come from the SAME (worst)
    occurrence so the surfaced numbers stay internally consistent -- a merged entry must never
    pair one rep's severity with another rep's evidence.
    """
    worst: dict[str, PoseRuleDetection] = {}
    reps: dict[str, set[int]] = {}
    order: list[str] = []
    for detection in detections:
        fault_id = detection.fault_id
        if fault_id not in worst:
            order.append(fault_id)
            reps[fault_id] = set()
        incumbent = worst.get(fault_id)
        if incumbent is None or (detection.severity, -detection.start_frame) > (
            incumbent.severity,
            -incumbent.start_frame,
        ):
            worst[fault_id] = detection
        reps[fault_id].update(detection.occurred_reps)

    merged: list[PoseRuleDetection] = []
    for fault_id in order:
        occurred = tuple(sorted(reps[fault_id]))
        merged.append(replace(worst[fault_id], occurred_reps=occurred, rep_count=len(occurred)))
    return merged


def run_detector(
    detector: MovementDetector,
    frames: Sequence[object],
    fps: float,
    view_type: str,
    view_confidence: float,
    *,
    max_reps: int | None = DEFAULT_MAX_REPS,
) -> RunResult:
    """Compute metrics over the whole clip, then phase and score one repetition at a time.

    Smoothing stays GLOBAL (all frames exist here), and only phase assignment and rule
    execution are per-rep. RS-SP2 extracts only the selected windows, at which point smoothing
    necessarily becomes per-padded-window -- that is an SP2 change, not a constraint on this
    one.
    """
    raw = detector.compute_raw(frames, fps)
    smoothed = {
        key: centered_median([float(item.get(key, np.nan)) for item in raw], window=5)
        for key in detector.metric_keys
    }

    reps: list[RepWindow] = []
    fallback: str | None = None
    if detector.rep_signal is None:
        fallback = "segmentation_disabled"
    else:
        reps = segment_reps(
            smoothed[detector.rep_signal],
            fps=fps,
            polarity=detector.rep_polarity,
            rectify=detector.rep_rectify,
            rep_start=detector.rep_start,
            min_rep_seconds=detector.min_rep_seconds,
        )
        if not reps:
            fallback = "no_reps_detected"
        elif all(rep.partial for rep in reps):
            # A tightly-trimmed single-rep clip (the labeled research dataset) looks like this.
            # Analyzing it whole is exactly the pre-existing behavior, which is correct for it.
            fallback = "only_partial_reps"

    # `reps` is what was FOUND and gets reported; `segmented` is what is actually used to phase
    # and score. They differ on the only_partial_reps path, where the payload should still say
    # what was there rather than claiming the clip held nothing.
    segmented = reps if fallback is None else []

    # Phases: per-rep when segmented, whole-clip on any fallback (today's behavior).
    if segmented:
        phases = [REST_PHASE] * len(raw)
        for rep in segmented:
            slice_len = rep.end - rep.start + 1
            rep_phases = detector.assign_phases(raw[rep.start : rep.end + 1])
            # List slice-assignment silently RESIZES the list when the right-hand side has a
            # different length: a longer result shifts every later frame's phase without
            # raising, and a shorter one only raises IndexError incidentally, later, via the
            # frame loop below -- neither is a signal a caller can act on. Raise here, at the
            # source, naming the detector and both lengths, so a detector whose assign_phases
            # returns the wrong length fails loudly instead of silently mis-phasing every frame
            # after it. `raise`, not `assert`: assertions are stripped under `python -O`.
            if len(rep_phases) != slice_len:
                raise ValueError(
                    f"{detector.name} assign_phases returned {len(rep_phases)} phases for "
                    f"rep {rep.index} (frames {rep.start}:{rep.end}, {slice_len} frames); "
                    "assign_phases must return exactly one phase per input frame."
                )
            phases[rep.start : rep.end + 1] = rep_phases
    else:
        phases = detector.assign_phases(raw)

    core: list[CoreFrame] = []
    for i, item in enumerate(raw):
        core.append(
            CoreFrame(
                frame_index=int(item.get("frame_index", i) or i),
                time=float(item.get("time", 0.0) or 0.0),
                phase=phases[i],
                valid=bool(item.get("valid", False)),
                lower_body_visibility=float(item.get("lower_body_visibility", 0.0) or 0.0),
                metrics={key: float(smoothed[key][i]) for key in detector.metric_keys},
            )
        )

    min_frames = max(3, int(math.ceil(max(fps, 1.0) * 0.20)))
    ctx = RuleContext(fps=fps, view_type=view_type, view_confidence=view_confidence, min_frames=min_frames)
    analyzed = select_reps(segmented, max_reps)

    detections: list[PoseRuleDetection] = []
    if analyzed:
        for rep in analyzed:
            # A SLICE, not a mask: `contiguous_true_segments` over a rep-gated global mask would
            # weld a fault at the end of rep 2 to one at the start of rep 3 into a single
            # detection spanning the gap between them. CoreFrame carries absolute frame_index
            # and time, so slicing does not disturb the reported timestamps.
            window = core[rep.start : rep.end + 1]
            for rule in detector.rules:
                for detection in rule(window, ctx):
                    detections.append(
                        replace(detection, rep_index=rep.index, occurred_reps=(rep.index,), rep_count=1)
                    )
    else:
        for rule in detector.rules:
            detections.extend(rule(core, ctx))

    detections = merge_by_fault(detections)
    detections.sort(key=lambda d: (d.observability == "low", -d.severity, d.start_frame))
    return RunResult(core=core, detections=detections, reps=reps, analyzed=analyzed, fallback=fallback)
