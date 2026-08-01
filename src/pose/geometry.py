from __future__ import annotations

import numpy as np
from typing import Sequence


VISIBILITY_THRESHOLD = 0.50
LANDMARK_COUNT = 33

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def landmarks_to_array(landmarks: object) -> np.ndarray | None:
    if not isinstance(landmarks, list) or len(landmarks) < LANDMARK_COUNT:
        return None

    array = np.full((LANDMARK_COUNT, 4), np.nan, dtype=np.float32)
    for index, landmark in enumerate(landmarks[:LANDMARK_COUNT]):
        if not isinstance(landmark, dict):
            continue
        try:
            array[index, 0] = float(landmark.get("x", np.nan))
            array[index, 1] = float(landmark.get("y", np.nan))
            array[index, 2] = float(landmark.get("z", np.nan))
            array[index, 3] = float(landmark.get("visibility", np.nan))
        except (TypeError, ValueError):
            continue
    return array


def visible_point(points: np.ndarray | None, index: int, dims: int = 3) -> np.ndarray | None:
    if points is None:
        return None
    values = points[index, :dims]
    visibility = points[index, 3]
    if not np.all(np.isfinite(values)) or not np.isfinite(visibility) or visibility < VISIBILITY_THRESHOLD:
        return None
    return values.astype(np.float32, copy=False)


def distance(points: np.ndarray | None, a: int, b: int, dims: int = 2) -> float:
    pa = visible_point(points, a, dims=dims)
    pb = visible_point(points, b, dims=dims)
    if pa is None or pb is None:
        return np.nan
    return float(np.linalg.norm(pa - pb))


def angle_degrees(points: np.ndarray | None, a: int, b: int, c: int) -> float:
    pa = visible_point(points, a, dims=3)
    pb = visible_point(points, b, dims=3)
    pc = visible_point(points, c, dims=3)
    if pa is None or pb is None or pc is None:
        return np.nan

    ba = pa - pb
    bc = pc - pb
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-8:
        return np.nan
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def midpoint(points: np.ndarray | None, left_index: int, right_index: int, dims: int = 2) -> np.ndarray | None:
    left = visible_point(points, left_index, dims=dims)
    right = visible_point(points, right_index, dims=dims)
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def line_angle_from_vertical(top: np.ndarray | None, bottom: np.ndarray | None) -> float:
    if top is None or bottom is None:
        return np.nan
    delta = top[:2] - bottom[:2]
    return float(np.degrees(np.arctan2(abs(delta[0]), abs(delta[1]) + 1e-8)))


def mean_visibility(points: np.ndarray | None, indices: Sequence[int]) -> float:
    if points is None:
        return 0.0
    values = points[list(indices), 3]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def mean_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def centered_median(values: Sequence[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    radius = max(0, window // 2)
    smoothed = np.full(array.shape, np.nan, dtype=np.float32)
    for index in range(array.size):
        start = max(0, index - radius)
        end = min(array.size, index + radius + 1)
        finite = array[start:end][np.isfinite(array[start:end])]
        smoothed[index] = float(np.median(finite)) if finite.size else np.nan
    return smoothed


def knee_forward_ratio(points: np.ndarray | None, knee_index: int, ankle_index: int, toe_index: int) -> float:
    knee = visible_point(points, knee_index, dims=2)
    ankle = visible_point(points, ankle_index, dims=2)
    toe = visible_point(points, toe_index, dims=2)
    if knee is None or ankle is None or toe is None:
        return np.nan
    foot_vector = toe - ankle
    foot_length = float(np.linalg.norm(foot_vector))
    if foot_length <= 1e-8:
        return np.nan
    projection = float(np.dot(knee - ankle, foot_vector / foot_length))
    return (projection - foot_length) / foot_length


def heel_height_delta(points: np.ndarray | None, heel_index: int, toe_index: int) -> float:
    heel = visible_point(points, heel_index, dims=2)
    toe = visible_point(points, toe_index, dims=2)
    if heel is None or toe is None:
        return np.nan
    return float(heel[1] - toe[1])


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def contiguous_true_segments(mask: Sequence[bool], min_frames: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_frames:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_frames:
        segments.append((start, len(mask) - 1))
    return segments


def severity_from_range(value: float, mild: float, severe: float, *, lower_is_worse: bool) -> float:
    if not np.isfinite(value):
        return 0.0
    if lower_is_worse:
        return clip01((mild - value) / (mild - severe))
    return clip01((value - mild) / (severe - mild))
