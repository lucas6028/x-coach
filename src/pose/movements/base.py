from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.pose.geometry import centered_median
from src.pose.pose_rule_detector import PoseRuleDetection
from src.pose.rep_segmentation import DEFAULT_MIN_REP_SECONDS


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


def run_detector(
    detector: MovementDetector,
    frames: Sequence[object],
    fps: float,
    view_type: str,
    view_confidence: float,
) -> tuple[list[CoreFrame], list[PoseRuleDetection]]:
    raw = detector.compute_raw(frames, fps)
    phases = detector.assign_phases(raw)
    smoothed = {
        key: centered_median([float(item.get(key, np.nan)) for item in raw], window=5)
        for key in detector.metric_keys
    }
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
    detections: list[PoseRuleDetection] = []
    for rule in detector.rules:
        detections.extend(rule(core, ctx))
    detections.sort(key=lambda d: (d.observability == "low", -d.severity, d.start_frame))
    return core, detections
