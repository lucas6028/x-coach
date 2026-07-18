from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.pose.geometry import centered_median, contiguous_true_segments  # noqa: F401
from src.pose.pose_rule_detector import PoseRuleDetection


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
