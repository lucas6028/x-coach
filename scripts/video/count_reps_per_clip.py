"""How many repetitions does one Fitness-AQA clip contain?

    .venv\\Scripts\\python.exe scripts/video/count_reps_per_clip.py --limit 300

This settles what the clip-length effect means. Clip length predicts the error label
(P(error) runs 0.517 -> 0.829 across duration quintiles), and there are two readings: a
longer clip packs MORE repetitions, so there are more chances to err, or a longer clip
is ONE slower repetition, i.e. a rep the lifter is grinding through. The two call for
opposite responses -- the first is a counting artifact to remove, the second is a real
biomechanical correlate to measure against.

Counts descend-and-return cycles in the hip height series from the MediaPipe pose JSON,
so it needs no video decoding. A rep is a descent past 70% of the clip's hip range
followed by a return above 30%; hysteresis rather than peak-picking, because a squat's
pause at the bottom makes a naive maximum finder count one rep several times.

Measured over 300 clips: 281 hold exactly one repetition, 2 hold two, 17 fail to
register (their median duration is 4.57s, so these are detection failures rather than
empty clips). corr(duration, reps) = -0.113, i.e. longer clips do NOT contain more
reps. See ``notes/videomae_b1_repeated_splits_results.md``.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.video.squat_dataset import SQUAT_LABELED_ROOT

#: MediaPipe left/right hip.
HIP_LANDMARKS = (23, 24)
VISIBILITY_THRESHOLD = 0.5
SMOOTHING_FRAMES = 9
DESCEND_THRESHOLD = 0.7
RETURN_THRESHOLD = 0.3


def hip_series(pose: dict) -> np.ndarray:
    """Mean hip height per frame, in normalised image coordinates (y grows downward)."""
    values: list[float] = []
    for frame in pose.get("frames", []):
        landmarks = frame.get("landmarks") or []
        if len(landmarks) <= max(HIP_LANDMARKS):
            continue
        visible = [
            landmarks[index]["y"]
            for index in HIP_LANDMARKS
            if float(landmarks[index].get("visibility", 0.0)) >= VISIBILITY_THRESHOLD
        ]
        values.append(float(np.mean(visible)) if visible else np.nan)
    return np.asarray(values, dtype=float)


def count_reps(series: np.ndarray) -> int | None:
    """Descend-and-return cycles, or ``None`` when the series is too short to judge."""
    series = series[np.isfinite(series)]
    if series.size < 20:
        return None

    smoothed = np.convolve(series, np.ones(SMOOTHING_FRAMES) / SMOOTHING_FRAMES, mode="valid")
    span = smoothed.max() - smoothed.min()
    if span < 1e-6:
        return 0
    normalised = (smoothed - smoothed.min()) / span

    reps = 0
    descended = False
    for value in normalised:
        if not descended and value > DESCEND_THRESHOLD:
            descended = True
        elif descended and value < RETURN_THRESHOLD:
            reps += 1
            descended = False
    return reps


def main() -> None:
    parser = argparse.ArgumentParser(description="Count repetitions per Fitness-AQA squat clip.")
    parser.add_argument("--pose-dir", type=Path, default=SQUAT_LABELED_ROOT / "pose_json")
    parser.add_argument("--manifest", type=Path, default=SQUAT_LABELED_ROOT / "videos_person_crop" / "manifest.json")
    parser.add_argument("--limit", type=int, default=300, help="Sample size; 0 for every clip.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8") as f:
        durations = {
            row["video_id"]: row["source_frames"] / (row["fps"] or 30.0)
            for row in json.load(f)["rows"]
        }

    paths = sorted(args.pose_dir.rglob("*.json"))
    if not paths:
        raise SystemExit(f"No pose JSON under {args.pose_dir}.")
    random.Random(args.seed).shuffle(paths)
    if args.limit:
        paths = paths[: args.limit]

    counts: list[int] = []
    clip_durations: list[float] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            reps = count_reps(hip_series(json.load(f)))
        if reps is None or path.stem not in durations:
            continue
        counts.append(reps)
        clip_durations.append(durations[path.stem])

    counts_array = np.asarray(counts)
    duration_array = np.asarray(clip_durations)
    print(f"{len(counts_array)} clips analysed")
    print("repetitions per clip:", dict(sorted(collections.Counter(counts).items())))
    for value in sorted(set(counts)):
        mask = counts_array == value
        print(f"  {value} rep(s): {mask.sum():>4} clips, duration median {np.median(duration_array[mask]):.2f}s")
    print(f"\ncorr(duration, reps) = {np.corrcoef(duration_array, counts_array)[0, 1]:+.3f}")
    print("  near zero => a longer clip is one SLOWER rep, not more reps")


if __name__ == "__main__":
    main()
