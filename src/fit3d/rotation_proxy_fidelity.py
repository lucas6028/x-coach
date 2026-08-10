"""Sensing fidelity of the Torso Twist rotation proxy, and of its brace angle, on Fit3D.

Backs `docs/superpowers/specs/2026-08-10-torso-twist-detector-design.md` sections 8.1 and 8.2 --
every number quoted there is produced by `scripts/fit3d/run_rotation_proxy_fidelity.py`, which
calls this module. It exists because the Row status note recorded that Fit3D can support a
2-D-cue-vs-3-D-truth fidelity comparison even though it carries no correctness labels, and
because a number in a citation-of-record whose script does not exist is a defect this project has
already logged once (see the Row residual in the parent spec).

WHAT IS MEASURED, AND WHAT IT IS NOT
------------------------------------
The parent spec's Group F heuristic reads axial trunk rotation as the change in the **projected
horizontal separation** of a paired landmark line (`|x11-x12|` for the shoulders, `|x23-x24|` for
the hips). That quantity is `width * |cos theta|`. This module asks how much true 3-D rotation
survives into it, and what that does to the decision `tt_lumbar_rotation_dominant` would make.

The 2-D side is **MOCAP-2D**: the mocap ground truth projected through the real per-camera
calibration, i.e. a PERFECT detector. Every error reported here is therefore projection alone,
with zero landmark noise. `twod_vs_threed.py` uses the same construction and explains why.

THE VARIANT DOES NOT MATCH, AND THAT BOUNDS THE CONCLUSIONS. Fit3D's `standing_ab_twists` is a
standing cross-body knee-to-elbow twist with a FREE pelvis; the app models a seated Russian twist
with the hips pinned. So the *distribution* of the true hip/shoulder rotation ratio does not
transfer and NO THRESHOLD may be taken from these numbers. What does transfer is the projection
geometry: `width * |cos theta|` compresses, is even in theta, and loses the hip line the same way
whatever the subject is doing with their legs.

The pure helpers above the banner take arrays and are unit-tested in `tests/
test_rotation_proxy_fidelity.py`; everything below the banner touches the gitignored corpus.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# The parent spec's own cut for `tt_lumbar_rotation_dominant`: flag when the hip line's rotation
# magnitude reaches ~0.6x the shoulder line's. Not a number this module chose or may move.
SPEC_RATIO_CUT = 0.6


# --------------------------------------------------------------------------------------
# Pure helpers -- arrays in, numbers out. No dataset access, so CI runs these.
# --------------------------------------------------------------------------------------
def wrap_radians(angles: np.ndarray) -> np.ndarray:
    """Fold angles into (-pi, pi]."""
    return (np.asarray(angles, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def horizontal_azimuth(vectors: np.ndarray) -> np.ndarray:
    """Heading of each row in the world horizontal plane. Fit3D's world up axis is +Z."""
    v = np.asarray(vectors, dtype=np.float64)
    return np.arctan2(v[:, 1], v[:, 0])


def rotation_about_own_median(vectors: np.ndarray) -> np.ndarray:
    """Signed rotation of a paired-landmark line, referenced to its own median heading.

    The reference has to come from the clip itself: a subject does not stand square to the
    camera on request, and the parent spec's proxy is likewise defined against a "resting"
    separation rather than an absolute one. Using the median rather than the first frame keeps
    the reference from landing on a frame that is already turned.
    """
    azimuth = horizontal_azimuth(vectors)
    return wrap_radians(azimuth - np.median(azimuth))


def signed_axial_twist(hip_line: np.ndarray, shoulder_line: np.ndarray, trunk_axis: np.ndarray) -> np.ndarray:
    """Angle of the shoulder line about the trunk axis, measured IN THE PELVIS FRAME.

    Body-internal, so it is invariant to where the subject stands and which way they face -- and
    it is the quantity the parent spec's rules are actually about, since a whole-body turn is not
    a twist.
    """
    axis = np.asarray(trunk_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)

    def perpendicular(line: np.ndarray) -> np.ndarray:
        line = np.asarray(line, dtype=np.float64)
        residual = line - (line * axis).sum(1, keepdims=True) * axis
        return residual / np.linalg.norm(residual, axis=1, keepdims=True)

    hip_p, shoulder_p = perpendicular(hip_line), perpendicular(shoulder_line)
    cosine = (hip_p * shoulder_p).sum(1)
    sine = (np.cross(hip_p, shoulder_p) * axis).sum(1)
    return np.arctan2(sine, cosine)


def proxy_rotation_deg(projected_width: np.ndarray) -> np.ndarray:
    """The parent spec's proxy, inverted to degrees: `arccos(width / resting_width)`.

    `resting_width` is taken as the window's own MAXIMUM, which is the most generous reading of
    "resting separation" a single clip can supply -- it credits the proxy with a perfect estimate
    of the square-on width whenever the subject passes through square at any point.
    """
    width = np.asarray(projected_width, dtype=np.float64)
    if not np.isfinite(width).any():
        return np.full(width.shape, np.nan)
    resting = float(np.nanmax(width))
    if not math.isfinite(resting) or resting <= 0.0:
        return np.full(width.shape, np.nan)
    return np.degrees(np.arccos(np.clip(width / resting, -1.0, 1.0)))


def sensitivity_per_degree(resting_width: float, rotation_deg: float) -> float:
    """|d(width * cos theta)/d theta| per DEGREE, in the units `resting_width` is given in.

    The analytic statement behind section 8.1's small-angle finding: the derivative is
    `width * sin(theta)`, so it is ZERO at the braced centre and largest near 90 degrees.
    """
    return float(resting_width) * math.sin(math.radians(float(rotation_deg))) * math.radians(1.0)


def smoothing_residual(series: np.ndarray, window: int = 15) -> float:
    """Median |x - rolling_median(x)|: the jitter a median filter of that width removes.

    Used to put a REAL noise floor next to `sensitivity_per_degree`, because an analytic
    derivative alone cannot say whether a degree of rotation is measurable.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.size < window:
        return float("nan")
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    rolled = np.array([np.nanmedian(padded[i : i + window]) for i in range(values.size)])
    return float(np.nanmedian(np.abs(values - rolled)))


def decision_agreement(true_ratio: np.ndarray, proxy_ratio: np.ndarray, cut: float = SPEC_RATIO_CUT) -> dict:
    """How often the proxy puts the hip/shoulder ratio on the SAME SIDE of the cut as the truth.

    The framing this project already uses for "does the reading change the verdict" (see
    `notes/fit3d_decision_fidelity_summary.md`): a correlation can be high while the decision
    flips, so both are reported and neither is allowed to stand for the other.
    """
    truth = np.asarray(true_ratio, dtype=np.float64) >= cut
    proxy = np.asarray(proxy_ratio, dtype=np.float64) >= cut
    disagree = truth != proxy
    return {
        "n": int(truth.size),
        "cut": float(cut),
        "truth_fires": int(truth.sum()),
        "proxy_fires": int(proxy.sum()),
        "disagree": int(disagree.sum()),
        "disagree_fraction": float(disagree.mean()) if truth.size else float("nan"),
        "proxy_only": int((proxy & ~truth).sum()),
        "truth_only": int((truth & ~proxy).sum()),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, so a monotone-but-biased proxy is distinguished from a noisy one."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    rank_a = np.argsort(np.argsort(a[finite]))
    rank_b = np.argsort(np.argsort(b[finite]))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def trunk_thigh_deg(points2d: np.ndarray, indices: dict[str, tuple[int, int]]) -> np.ndarray:
    """`angle(hip_mid -> shoulder_mid, hip_mid -> knee_mid)`, degrees, per frame.

    The shipped rule's own quantity, computed on a projected skeleton so that four simultaneous
    cameras can be compared against each other. MIDPOINTS, matching
    `src/pose/movements/torso_twist.py` -- a same-side construction would move under axial
    rotation and the cross-camera spread would then be measuring the twist, not the projection.
    """
    def mid(pair: tuple[int, int]) -> np.ndarray:
        return 0.5 * (points2d[:, pair[0]] + points2d[:, pair[1]])

    trunk = mid(indices["shoulders"]) - mid(indices["hips"])
    thigh = mid(indices["knees"]) - mid(indices["hips"])
    trunk = trunk / np.linalg.norm(trunk, axis=1, keepdims=True)
    thigh = thigh / np.linalg.norm(thigh, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((trunk * thigh).sum(1), -1.0, 1.0)))


def summarize(values: np.ndarray) -> dict:
    values = np.asarray([v for v in np.asarray(values, dtype=np.float64) if math.isfinite(v)])
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------------------
# Orchestration below this banner touches `data/Fit3D`, which is gitignored, so CI never
# reaches it. Same split as `src/rehab24/lunge_rule_validation.py`.
# --------------------------------------------------------------------------------------
def run(action: str = "standing_ab_twists", split: str = "train") -> dict:
    """Every number in design-spec sections 8.1 and 8.2, in one pass over the corpus."""
    from src.fit3d import dataset as ds

    indices = {
        "shoulders": (ds.L_SHOULDER, ds.R_SHOULDER),
        "hips": (ds.L_HIP, ds.R_HIP),
        "knees": (ds.L_KNEE, ds.R_KNEE),
    }
    per_line: dict[str, list[dict]] = defaultdict(list)
    true_ratios: list[float] = []
    proxy_ratios: list[float] = []
    twist_peaks: list[float] = []
    brace_absolute_spread: list[float] = []
    brace_sag_spread: list[float] = []
    brace_sag_values: list[float] = []
    sensitivity: dict[str, list[float]] = defaultdict(list)

    for subject in ds.subjects(split):
        try:
            joints = ds.load_joints3d(split, subject, action)
            reps = ds.rep_segments(ds.load_rep_ann(split, subject)[action])
        except (FileNotFoundError, KeyError):
            continue

        shoulder_line = joints[:, ds.L_SHOULDER] - joints[:, ds.R_SHOULDER]
        hip_line = joints[:, ds.L_HIP] - joints[:, ds.R_HIP]
        rotation = {
            "shoulder": np.degrees(rotation_about_own_median(shoulder_line)),
            "hip": np.degrees(rotation_about_own_median(hip_line)),
        }
        twist = np.degrees(
            wrap_radians(
                signed_axial_twist(hip_line, shoulder_line, joints[:, ds.THORAX] - joints[:, ds.ROOT])
            )
        )
        twist = twist - np.median(twist)

        projected: dict[str, dict[str, np.ndarray]] = {}
        brace: dict[str, np.ndarray] = {}
        image_widths: dict[str, float] = {}
        for camera in ds.cameras(split, subject):
            params = ds.read_cam_params(split, subject, camera, action)
            image = ds.project_world_to_image(joints, params)
            projected[camera] = {
                "shoulder": np.abs(image[:, ds.L_SHOULDER, 0] - image[:, ds.R_SHOULDER, 0]),
                "hip": np.abs(image[:, ds.L_HIP, 0] - image[:, ds.R_HIP, 0]),
            }
            # Per camera, not reused from the loop variable: Fit3D's four cameras happen to share
            # a resolution, and relying on that silently would be the kind of accident that only
            # shows up on the next dataset.
            image_widths[camera] = 2.0 * float(params["intrinsics_wo_distortion"]["c"][0])
            brace[camera] = trunk_thigh_deg(image, indices)

        for index, (start, end) in enumerate(reps):
            window = slice(start, end + 1)
            twist_peaks.append(float(np.nanmax(np.abs(twist[window]))))

            for camera, widths in projected.items():
                peaks: dict[str, float] = {}
                for line in ("shoulder", "hip"):
                    truth = np.abs(rotation[line][window])
                    estimate = proxy_rotation_deg(widths[line][window])
                    peaks[line] = float(np.nanmax(estimate))
                    per_line[line].append(
                        {
                            "subject": subject, "camera": camera, "rep": index,
                            "true_peak": float(np.nanmax(truth)),
                            "proxy_peak": peaks[line],
                            "mae": float(np.nanmean(np.abs(estimate - truth))),
                            "corr": float(np.corrcoef(truth, estimate)[0, 1])
                            if np.std(estimate) > 1e-9 else float("nan"),
                        }
                    )
                true_shoulder = float(np.nanmax(np.abs(rotation["shoulder"][window])))
                if true_shoulder > 1e-6 and peaks["shoulder"] > 1e-6:
                    true_ratios.append(float(np.nanmax(np.abs(rotation["hip"][window]))) / true_shoulder)
                    proxy_ratios.append(peaks["hip"] / peaks["shoulder"])

                # Sensitivity is reported as a fraction of the IMAGE WIDTH so it can be compared
                # against a pose estimator's own jitter, which is measured in the same units.
                resting = float(np.nanmax(widths["shoulder"][window]))
                for band, centre in (("0-15deg", 7.5), ("45-75deg", 60.0)):
                    sensitivity[band].append(
                        sensitivity_per_degree(resting / image_widths[camera], centre)
                    )

            # Four simultaneous cameras, so any disagreement between them is pure projection.
            setup = max(1, int((end - start + 1) * 0.15))
            medians, sags = [], []
            for series in brace.values():
                segment = series[window]
                medians.append(float(np.nanmedian(segment)))
                # SIGNED, matching the shipped rule: a maximum of `angle - baseline`, not of its
                # absolute value. An unsigned version would credit the rule with detecting a
                # subject who TIGHTENED, which it deliberately does not.
                sag = float(np.nanmax(segment - np.nanmedian(segment[:setup])))
                sags.append(sag)
                brace_sag_values.append(sag)
            brace_absolute_spread.append(max(medians) - min(medians))
            brace_sag_spread.append(max(sags) - min(sags))

    out: dict = {
        "action": action, "split": split,
        "records": sum(len(v) for v in per_line.values()),
        "true_relative_trunk_twist_peak_deg": summarize(np.array(twist_peaks)),
        "lines": {},
        "ratio": {
            "true": summarize(np.array(true_ratios)),
            "proxy": summarize(np.array(proxy_ratios)),
            "rank_correlation": spearman(np.array(true_ratios), np.array(proxy_ratios)),
            "decision": decision_agreement(np.array(true_ratios), np.array(proxy_ratios)),
        },
        "shoulder_width_change_per_degree_of_image_width": {
            band: float(np.median(values)) for band, values in sensitivity.items()
        },
        "brace_angle": {
            "absolute_cross_camera_spread_deg": summarize(np.array(brace_absolute_spread)),
            "sag_value_deg": summarize(np.array(brace_sag_values)),
            "sag_cross_camera_spread_deg": summarize(np.array(brace_sag_spread)),
        },
    }
    for line, rows in per_line.items():
        corr = np.array([r["corr"] for r in rows], dtype=np.float64)
        out["lines"][line] = {
            "true_peak_deg": summarize(np.array([r["true_peak"] for r in rows])),
            "proxy_peak_deg": summarize(np.array([r["proxy_peak"] for r in rows])),
            "per_frame_mae_deg": summarize(np.array([r["mae"] for r in rows])),
            "per_rep_correlation": summarize(corr),
            "fraction_anticorrelated": float(np.nanmean(corr < 0)),
        }
    return out


def mediapipe_width_jitter(cache_dir=None) -> dict:
    """The real noise floor `sensitivity_per_degree` has to be compared against.

    Frame-to-frame movement of MediaPipe's own shoulder and hip widths on REHAB24-6's cached
    landmarks, in normalized image widths -- the same units as the sensitivity above. REHAB24-6
    rather than Fit3D because Fit3D has no MediaPipe pass and this quantity is a property of the
    ESTIMATOR, not of the exercise.
    """
    from pathlib import Path

    from src.pose.geometry import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

    root = Path(cache_dir) if cache_dir is not None else (
        Path(__file__).resolve().parents[2]
        / "data" / "REHAB24-6" / "processed" / "mediapipe_landmarks_cache"
    )
    out: dict = {}
    for name, (left, right) in (
        ("shoulder", (LEFT_SHOULDER, RIGHT_SHOULDER)),
        ("hip", (LEFT_HIP, RIGHT_HIP)),
    ):
        steps, residuals, widths = [], [], []
        for path in sorted(root.glob("*.npz")):
            image = np.load(path)["image"]
            if image.shape[0] < 200:
                continue
            width = np.abs(image[:, left, 0] - image[:, right, 0]).astype(np.float64)
            steps.append(float(np.nanmedian(np.abs(np.diff(width)))))
            residuals.append(smoothing_residual(width, window=15))
            widths.append(float(np.nanmedian(width)))
        out[name] = {
            "videos": len(steps),
            "median_width_of_image": float(np.median(widths)) if widths else float("nan"),
            "frame_to_frame_step": float(np.median(steps)) if steps else float("nan"),
            "residual_vs_median15": float(np.median(residuals)) if residuals else float("nan"),
        }
    return out
