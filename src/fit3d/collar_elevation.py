"""Step 1 -- does the SMPL-X collar joint carry shoulder-girdle motion independent of the arm?

MediaPipe's "shoulder" landmark rides on the humerus, so every shrug and retraction rule built on
it is permanently silent (``arm_abduction.rule_shoulder_shrug``, ``arm_vw.rule_shrug_substitution``,
``band_pull_apart.rule_loss_of_scapular_retraction``). SMPL/SMPL-X has a separate collar joint
(13 = L_collar, 14 = R_collar, child of spine3) whose rotation moves the shoulder joint. Before
asking whether an image->mesh model can *predict* that rotation (step 2), this asks whether
Fit3D's SMPL-X fits *contain* it at all, as rep-to-rep variation the arm does not explain.

Quantities, all in the thorax (spine3) frame, so global orientation and trunk posture drop out:

* **collar elevation** -- the collar->shoulder direction's angle above horizontal, minus its
  rest-pose value (0 deg = rest; positive = shoulder raised = shrug direction).
* **collar protraction** -- the same direction's forward (+z) angle minus rest (negative =
  retraction). Used on band pull-apart only.
* **arm elevation** -- humerothoracic: angle between shoulder->elbow and thorax-down
  (0 = arm at side, 90 = horizontal).
* **horizontal abduction** -- the arm's angle in the thorax transverse plane from straight
  forward toward its own side (0 = forward, 90 = out to the side). Band pull-apart only.

Side conventions are mirror-consistent: the SMPL-X template has +x = subject's left, +y up,
+z forward, and every angle above reads only y and z (plus a side-signed x), so a rotation
mirrored across the sagittal plane gives the same angle on the other side (pinned by a test).

Per rep (Fit3D ``rep_ann``), the rep's own driver picks two frames: the driver's minimum and
maximum. Delta-collar and delta-arm are taken between those two frames. Drivers: arm elevation
for raises, horizontal abduction for band pull-apart, collar elevation for the shrug control.

Rule written before the rep-level residuals were computed. Already seen at that point: the
shrug control, and per-rep collar/arm ranges plus within-clip correlations on 3 subjects:

    per (action, side): pool reps across subjects, remove per-subject means, fit
    delta_collar ~ slope * delta_arm; residual SD = the independent signal.
    Noise floor = sqrt(2) x frame jitter (collar SD around a 15-frame moving average,
    pooled over the same rep windows) -- a delta is a difference of two frames.
      STOP       residual SD <= 2 x noise floor (collar is slaved to the arm or to fit noise)
      NOISY      residual SD > 2 x floor, but shrug-control median amplitude < 3 x residual SD
                 (independent, but a shrug-sized event would not stand out)
      PASS       residual SD > 2 x floor and shrug amplitude >= 3 x residual SD

A second, looser floor is reported but not used by the rule: collar-elevation SD within reps of
``dumbbell_reverse_lunge`` (arms hanging with dumbbells; includes real girdle motion, so it
over-states noise).

The collar "ground truth" is itself a fit (53 markers + 4-view keypoints + a normalizing-flow body
prior), so a residual is fit behaviour, not certified anatomy. No Fit3D action carries fault
labels, so passing validates no rule. Numbers and caveats: ``notes/fit3d_collar_elevation_summary.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# SMPL-X joint indices (body_pose[j - 1] holds joint j's local rotation).
SPINE3 = 9
COLLAR = {"L": 13, "R": 14}
SHOULDER = {"L": 16, "R": 17}
ELBOW = {"L": 18, "R": 19}
SIDE_SIGN = {"L": 1.0, "R": -1.0}  # +x is the subject's left in the SMPL-X template
BODY_POSE_OFFSET = 1

JITTER_WINDOW = 15  # frames; 0.3 s at Fit3D's 50 fps
STOP_FLOOR_MULT = 2.0
PASS_SHRUG_MULT = 3.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _local(body_pose: np.ndarray, joint: int) -> np.ndarray:
    return body_pose[:, joint - BODY_POSE_OFFSET]


def collar_direction(body_pose: np.ndarray, rest: np.ndarray, side: str) -> np.ndarray:
    """``(F, 3)`` collar->shoulder unit direction in the spine3 frame."""
    axis = _unit(rest[SHOULDER[side]] - rest[COLLAR[side]])
    return _local(body_pose, COLLAR[side]) @ axis


def arm_direction(body_pose: np.ndarray, rest: np.ndarray, side: str) -> np.ndarray:
    """``(F, 3)`` shoulder->elbow unit direction in the spine3 frame (collar, then shoulder)."""
    axis = _unit(rest[ELBOW[side]] - rest[SHOULDER[side]])
    rot = _local(body_pose, COLLAR[side]) @ _local(body_pose, SHOULDER[side])
    return rot @ axis


def _asin_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(x, -1.0, 1.0)))


def collar_angles(body_pose: np.ndarray, rest: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """``(elevation_deg, protraction_deg)`` of the collar->shoulder direction, relative to rest."""
    axis = _unit(rest[SHOULDER[side]] - rest[COLLAR[side]])
    d = collar_direction(body_pose, rest, side)
    elevation = _asin_deg(d[:, 1]) - _asin_deg(axis[1])
    protraction = _asin_deg(d[:, 2]) - _asin_deg(axis[2])
    return elevation, protraction


def arm_angles(body_pose: np.ndarray, rest: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """``(humerothoracic_elevation_deg, horizontal_abduction_deg)``."""
    a = arm_direction(body_pose, rest, side)
    elevation = np.degrees(np.arccos(np.clip(-a[:, 1], -1.0, 1.0)))
    horizontal = np.degrees(np.arctan2(SIDE_SIGN[side] * a[:, 0], a[:, 2]))
    return elevation, horizontal


def moving_average(x: np.ndarray, window: int = JITTER_WINDOW) -> np.ndarray:
    """Centred moving average with edge renormalisation (no phase shift, no padding bias)."""
    kernel = np.ones(window)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / den


def jitter_residuals(x: np.ndarray, window: int = JITTER_WINDOW) -> np.ndarray:
    """Frame-level deviation from a centred moving average -- the high-frequency part of ``x``."""
    return x - moving_average(x, window)


def second_difference_sigma(x: np.ndarray) -> float:
    """White-noise SD estimate with no window: ``sd(x[t+1] - 2x[t] + x[t-1]) / sqrt(6)``.

    Smooth motion contributes only acceleration x dt^2 to a second difference, so this is
    window-free. It is a LOWER bound for these fits: Fit3D's SMPL-X series are temporally
    smooth, so any fit error is correlated in time and invisible to it.
    """
    dd = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.sqrt(np.mean(dd**2)) / np.sqrt(6.0))


@dataclass
class RepDelta:
    subject: str
    rep: int
    d_collar: float  # collar quantity at driver max minus at driver min
    d_arm: float  # arm quantity at driver max minus at driver min


def rep_deltas(
    driver: np.ndarray, collar: np.ndarray, arm: np.ndarray, windows: list[tuple[int, int]], subject: str
) -> list[RepDelta]:
    """Per rep: values at the driver's max frame minus values at its min frame."""
    out = []
    for i, (a, z) in enumerate(windows):
        seg = driver[a:z]
        if len(seg) < 2 or not np.isfinite(seg).any():
            continue
        lo = a + int(np.nanargmin(seg))
        hi = a + int(np.nanargmax(seg))
        out.append(RepDelta(subject, i, float(collar[hi] - collar[lo]), float(arm[hi] - arm[lo])))
    return out


@dataclass
class CouplingFit:
    n_reps: int
    n_subjects: int
    slope: float  # d_collar per d_arm, within subject
    r2_within: float
    residual_sd: float  # deg, dof-corrected (n - subjects - 1)
    d_arm_sd_within: float  # how much the arm excursion itself varied rep to rep

    def to_dict(self) -> dict:
        return asdict(self)


def within_subject_fit(deltas: list[RepDelta]) -> CouplingFit:
    """OLS of d_collar on d_arm after removing each subject's mean from both (fixed effects)."""
    subjects = sorted({d.subject for d in deltas})
    x = np.array([d.d_arm for d in deltas], dtype=np.float64)
    y = np.array([d.d_collar for d in deltas], dtype=np.float64)
    g = np.array([d.subject for d in deltas])
    xc, yc = x.copy(), y.copy()
    for s in subjects:
        m = g == s
        xc[m] -= x[m].mean()
        yc[m] -= y[m].mean()
    sxx = float(xc @ xc)
    slope = float(xc @ yc / sxx) if sxx > 0 else 0.0
    resid = yc - slope * xc
    dof = len(deltas) - len(subjects) - 1
    ss_tot = float(yc @ yc)
    return CouplingFit(
        n_reps=len(deltas),
        n_subjects=len(subjects),
        slope=slope,
        r2_within=1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
        residual_sd=float(np.sqrt(resid @ resid / dof)) if dof > 0 else float("nan"),
        d_arm_sd_within=float(np.sqrt(sxx / max(len(deltas) - len(subjects), 1))),
    )


def verdict(residual_sd: float, noise_floor: float, shrug_amplitude: float) -> str:
    """Apply the pre-registered STOP / NOISY / PASS rule (module docstring)."""
    if not residual_sd > STOP_FLOOR_MULT * noise_floor:
        return "STOP"
    if shrug_amplitude < PASS_SHRUG_MULT * residual_sd:
        return "NOISY"
    return "PASS"
