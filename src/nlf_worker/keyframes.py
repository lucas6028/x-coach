"""Pick which sampled frames get a rendered turntable strip.

Every movement contributes one 'fault_peak' per detection. Squat additionally contributes one
'rep_bottom' per rep segment, taken as the minimum knee angle *inside that segment's time
span* -- never a global minimum over the whole clip, which would pick up a set-up crouch
before the first rep (the Phase 0 lesson this module exists to avoid repeating). Non-squat
movements score the whole clip on the browser path, so their rep segments are not used here at
all.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.nlf_worker.timeline import frame_to_seconds

_SQUAT = "squat"


def _is_squat(movement: str | None) -> bool:
    # A null/blank movement means Squat: registry.get_detector falls back the same way
    # (`(movement or "Squat").lower()`), and store.py notes older analysis rows have no
    # movement column at all.
    if not movement:
        return True
    return movement.strip().lower() == _SQUAT


def _nearest_sample(sample_times: np.ndarray, t: float, detected: np.ndarray | None = None) -> int:
    """Nearest sample index to ``t``, restricted to ``detected`` samples when any exist.

    A fault's peak_frame can land, once converted and snapped, on a sample where nobody was
    detected (no joints/vertices to render) -- so prefer the nearest sample that actually has
    a pose over the literal nearest-in-time one.
    """
    if detected is not None and detected.any():
        candidates = np.where(detected)[0]
        return int(candidates[np.argmin(np.abs(sample_times[candidates] - t))])
    return int(np.argmin(np.abs(sample_times - t)))


def _fault_peaks(
    detections: list[dict] | None,
    sample_times: np.ndarray,
    knee_angles: np.ndarray,
    pose_fps: float | None,
) -> list[dict]:
    detected = np.isfinite(knee_angles)
    out = []
    for det in detections or []:
        t = frame_to_seconds(det["peak_frame"], pose_fps)
        if t is None:
            continue  # can't place this detection in time without pose_fps
        idx = _nearest_sample(sample_times, t, detected)
        out.append(
            {
                "t_s": float(sample_times[idx]),
                "kind": "fault_peak",
                "fault_ids": [det["fault_id"]],
                "rep_index": None,
                "sample_index": idx,
            }
        )
    return out


def _rep_bottoms(
    rep_segments: list[dict] | None,
    sample_times: np.ndarray,
    knee_angles: np.ndarray,
    pose_fps: float | None,
) -> list[dict]:
    out = []
    if pose_fps is None:
        return out  # can't convert start_frame/end_frame to seconds without it
    for seg in rep_segments or []:
        start_s = frame_to_seconds(seg["start_frame"], pose_fps)
        end_s = frame_to_seconds(seg["end_frame"], pose_fps)
        in_span = (sample_times >= start_s) & (sample_times <= end_s)
        candidate_idx = np.where(in_span & np.isfinite(knee_angles))[0]
        if candidate_idx.size == 0:
            continue  # no finite knee angle inside this segment -- skip it
        best = candidate_idx[np.argmin(knee_angles[candidate_idx])]
        out.append(
            {
                "t_s": float(sample_times[best]),
                "kind": "rep_bottom",
                "fault_ids": [],
                "rep_index": seg["index"],
                "sample_index": int(best),
            }
        )
    return out


def _merge(entries: list[dict], merge_s: float) -> list[dict]:
    """Merge entries closer than ``merge_s`` apart in time, in chronological order.

    A merged group keeps 'rep_bottom' if any member has it (a bottom takes priority over a
    peak that lands close to it), unions every member's ``fault_ids``, and anchors on the
    rep_bottom's time/sample when one is present, else the earliest member's.
    """
    if not entries:
        return []
    ordered = sorted(entries, key=lambda e: e["t_s"])
    groups: list[list[dict]] = [[ordered[0]]]
    for e in ordered[1:]:
        if e["t_s"] - groups[-1][-1]["t_s"] <= merge_s:
            groups[-1].append(e)
        else:
            groups.append([e])

    merged = []
    for group in groups:
        bottoms = [g for g in group if g["kind"] == "rep_bottom"]
        anchor = bottoms[0] if bottoms else group[0]
        fault_ids: list[Any] = []
        for g in group:
            for fid in g["fault_ids"]:
                if fid not in fault_ids:
                    fault_ids.append(fid)
        merged.append(
            {
                "t_s": anchor["t_s"],
                "kind": anchor["kind"],
                "fault_ids": fault_ids,
                "rep_index": anchor["rep_index"],
                "sample_index": anchor["sample_index"],
            }
        )
    return merged


def _cap(entries: list[dict], cap: int) -> list[dict]:
    if len(entries) <= cap:
        return entries
    priority = sorted(entries, key=lambda e: (0 if e["kind"] == "rep_bottom" else 1, e["t_s"]))
    keep_ids = {id(e) for e in priority[:cap]}
    return [e for e in entries if id(e) in keep_ids]


def select_key_frames(
    movement: str | None,
    sample_times: np.ndarray,
    knee_angles: np.ndarray,
    rep_segments: list[dict] | None,
    detections: list[dict] | None,
    pose_fps: float | None,
    merge_s: float = 0.3,
    cap: int = 12,
) -> list[dict]:
    """Chronological list of {t_s, kind, fault_ids, rep_index, sample_index}."""
    sample_times = np.asarray(sample_times, dtype=float)
    knee_angles = np.asarray(knee_angles, dtype=float)

    entries = _fault_peaks(detections, sample_times, knee_angles, pose_fps)
    if _is_squat(movement):
        entries += _rep_bottoms(rep_segments, sample_times, knee_angles, pose_fps)

    merged = _merge(entries, merge_s)
    merged.sort(key=lambda e: e["t_s"])
    capped = _cap(merged, cap)
    capped.sort(key=lambda e: e["t_s"])
    return capped
