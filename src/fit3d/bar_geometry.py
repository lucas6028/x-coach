"""Is the barbell's pose recoverable from body keypoints alone?

Blind-spot B of the "what do keypoints structurally miss" thread. A keypoint skeleton
encodes the *body* and nothing else, so on paper the implement is invisible to it. That
matters concretely here: ``src/pose/movements/overhead_press.py`` had to **withdraw** a
bar-path sub-criterion, partly because referencing the bar to the shoulders conflates it
with trunk lean and partly because no bar reference exists at all. This module measures
whether that withdrawal was forced by a representation limit or merely by missing effort.

B splits into two claims with very different testability:

* **Geometry** -- *where* the implement is. Measurable, and this module measures it.
* **Magnitude** -- *how heavy* it is. Not measurable in any dataset on this machine:
  Fit3D uses one unloaded bar for every subject, Fitness-AQA labels faults not loads,
  and REHAB24-6 has no implement. The Newtonian argument (identical kinematics + unknown
  load => joint moments unidentifiable) stays an argument here, not a result.

Ground truth
------------
Fit3D ships a calibrated 4-camera rig, so the bar can be measured rather than annotated:
segment the (yellow) bar by colour, fit a 2D line per view with RANSAC, back-project each
2D line to a plane through its camera centre, and intersect the planes across views. The
intersection is the 3D bar **axis**. Nothing about the body enters the measurement; the
keypoints are used only to place a search ROI.

The self-consistency check is ``plane_residual``: the distance from the recovered 3D point
to each view's back-projected plane. Four independent views agreeing to ~4 mm is the
analogue of the "limbs rigid to 1e-5 m" check that validated Exp 0's target.

Scope, honestly
---------------
The axis is recovered well; the bar's **endpoints** are not (occlusion by the torso breaks
the colour blob into pieces, and carving the mask gave lengths of 0.79 +- 0.39 m against a
physically constant bar). Every target here is therefore **endpoint-free** -- a property of
the infinite line, not of the segment.

``squat`` does **not** extract reliably and is excluded. With the arms up at the bar the ROI
grows tall enough to admit background fixtures, and RANSAC locks onto those instead: the
perpendicular shoulder-to-bar distance comes out 34 +- 20 cm where the bar physically rests
on the traps at ~11 cm. Filtering on ``plane_residual`` does not rescue it -- residual
*anti*-correlates with the error (-0.55), because the wrong structure is straight and is
seen consistently by all four views. A quality metric that cannot see the failure must not
be used to launder it. Bar-in-hands actions (deadlift, rows) do not have this problem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds

# Bar colour window. The bar and bare skin overlap substantially in HSV (bar hue median 28,
# skin 19, identical saturation), so colour alone CANNOT separate them -- it only narrows the
# candidate set. Collinearity over ~250 px is what actually identifies the bar, which is why
# the line fit below is RANSAC and not a plain total-least-squares fit over the mask.
BAR_HSV_LO = (18, 55, 55)
BAR_HSV_HI = (45, 255, 255)

DEFAULT_STRIDE = 20
AXIS_GRID = np.arange(-1.0, 1.0001, 0.002)

# Nested keypoint sets, ordered from "physical contact only" to "whole body", mirroring the
# nested design of the axial-rotation study. The point of nesting is to localise *where* any
# predictive power comes from, not merely to report that some exists.
BAR_KEYPOINT_SETS: dict[str, list[int]] = {
    "wrists": [ds.R_WRIST, ds.L_WRIST],
    "hands": [ds.R_WRIST, ds.L_WRIST, ds.R_HAND_A, ds.R_HAND_B, ds.L_HAND_A, ds.L_HAND_B],
    "arms": [ds.R_SHOULDER, ds.R_ELBOW, ds.R_WRIST, ds.L_SHOULDER, ds.L_ELBOW, ds.L_WRIST,
             ds.R_HAND_A, ds.R_HAND_B, ds.L_HAND_A, ds.L_HAND_B],
    "h36m17": list(range(17)),
    "full25": list(range(25)),
}


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def opencv_intrinsics(cam_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fit3D (imar) calibration -> OpenCV ``(K, distCoeffs)``.

    The coefficient ORDER was determined empirically, not assumed: round-tripping GT joints
    through ``project_world_to_image(..., True)`` and then ``cv2.undistortPoints`` reproduces
    the undistorted projection to 0.068 px mean with ``[k1, k2, p1, p2, k3]``, versus 0.323 px
    with the tangential pair swapped. Do not reorder without re-running that check.
    """
    intr = cam_params["intrinsics_w_distortion"]
    f = np.asarray(intr["f"], dtype=np.float64).reshape(2)
    c = np.asarray(intr["c"], dtype=np.float64).reshape(2)
    k = np.asarray(intr["k"], dtype=np.float64).reshape(3)
    p = np.asarray(intr["p"], dtype=np.float64).reshape(2)
    K = np.array([[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]])
    return K, np.array([k[0], k[1], p[0], p[1], k[2]], dtype=np.float64)


# ---------------------------------------------------------------------------
# 2D: find the bar in one image
# ---------------------------------------------------------------------------
def _contiguous_run(inliers: np.ndarray, along: np.ndarray, max_gap: float) -> np.ndarray:
    """Restrict an inlier mask to its longest gap-free run along the line.

    A barbell is a *connected* rod, and this is the property that identifies it. Scoring a
    hypothesis by raw extent instead is gameable: a dense compact blob plus two distant
    stragglers spans far while covering nothing in between, which is how the naive version
    locked onto background fixtures instead of the bar.

    ``max_gap`` must stay LARGE (default 150 px). The bar is routinely occluded mid-span by
    the torso or head, so a strict gap forces the fit onto one short fragment: at 15 px the
    median multi-view plane residual on deadlift is 2.9 mm, at 150 px it is 1.2 mm, and the
    per-view fragments stop disagreeing. Contiguity here rejects *disconnected* structures,
    it does not demand an unbroken one.
    """
    order = np.argsort(along)
    sorted_along = along[order]
    breaks = np.nonzero(np.diff(sorted_along) > max_gap)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(sorted_along)]])
    best = int(np.argmax(ends - starts))
    keep_local = order[starts[best]:ends[best]]
    indices = np.nonzero(inliers)[0][keep_local]
    restricted = np.zeros_like(inliers)
    restricted[indices] = True
    return restricted


def ransac_line_2d(
    points: np.ndarray,
    threshold: float = 3.0,
    iterations: int = 250,
    min_separation: float = 50.0,
    min_inliers: int = 30,
    max_gap: float = 150.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Longest strongly-collinear CONNECTED structure in a pixel cloud.

    Returns ``(mean, direction, inlier_mask, span_px)`` or ``None``.

    Hypotheses are scored by ``inliers * contiguous_span`` -- see :func:`_contiguous_run` for
    why contiguity, not raw extent, is what identifies a barbell. A compact blob of skin
    pixels can out-count the bar; it cannot out-span it *without gaps*.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 40:
        return None
    rng = np.random.default_rng(seed)
    best_score, best_inliers = -1.0, None
    for _ in range(iterations):
        i, j = rng.integers(0, len(pts), 2)
        vec = pts[j] - pts[i]
        length = float(np.linalg.norm(vec))
        if length < min_separation:
            continue
        direction = vec / length
        normal = np.array([-direction[1], direction[0]])
        inliers = np.abs((pts - pts[i]) @ normal) < threshold
        if inliers.sum() < min_inliers:
            continue
        inliers = _contiguous_run(inliers, (pts[inliers] - pts[i]) @ direction, max_gap)
        if inliers.sum() < min_inliers:
            continue
        along = (pts[inliers] - pts[i]) @ direction
        score = float(inliers.sum()) * float(along.max() - along.min())
        if score > best_score:
            best_score, best_inliers = score, inliers
    if best_inliers is None:
        return None

    inliers = best_inliers
    for _ in range(3):  # total-least-squares refinement on the selected structure
        chosen = pts[inliers]
        mean = chosen.mean(axis=0)
        _, _, vt = np.linalg.svd(chosen - mean, full_matrices=False)
        normal = np.array([-vt[0][1], vt[0][0]])
        inliers = np.abs((pts - mean) @ normal) < threshold
        if inliers.sum() < min_inliers:
            return None
        inliers = _contiguous_run(inliers, (pts[inliers] - mean) @ vt[0], max_gap)
        if inliers.sum() < min_inliers:
            return None
    chosen = pts[inliers]
    mean = chosen.mean(axis=0)
    _, _, vt = np.linalg.svd(chosen - mean, full_matrices=False)
    along = (chosen - mean) @ vt[0]
    return mean, vt[0], inliers, float(along.max() - along.min())


def back_project_line(
    inlier_pixels: np.ndarray, cam_params: dict
) -> tuple[np.ndarray, float]:
    """2D line (as its inlier pixels) -> the world plane containing it and the camera centre.

    Returns ``(normal_world, offset)`` such that every world point ``X`` on the plane
    satisfies ``normal_world @ X == offset``. The 3D bar axis lies in this plane, so
    intersecting the planes from >=2 views determines the axis with no point correspondence
    between views -- which matters because the bar is a featureless rod with no matchable
    points along it.
    """
    import cv2  # local import: OpenCV is a heavy optional dep, matching this repo's style

    K, dist = opencv_intrinsics(cam_params)
    undistorted = cv2.undistortPoints(
        np.asarray(inlier_pixels, dtype=np.float64).reshape(-1, 1, 2), K, dist
    ).reshape(-1, 2)
    mean = undistorted.mean(axis=0)
    _, _, vt = np.linalg.svd(undistorted - mean, full_matrices=False)
    normal_img = np.array([-vt[0][1], vt[0][0], 0.0])
    normal_img[2] = -float(normal_img[:2] @ mean)

    rot = cam_params["extrinsics"]["R"]
    centre = cam_params["extrinsics"]["T"].reshape(3)
    normal_world = rot.T @ normal_img
    return normal_world, float(normal_world @ centre)


def triangulate_axis(
    normals: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Intersect >=2 back-projected planes into a 3D line.

    Returns ``(point_on_line, unit_direction, plane_residual_m)``. The direction is the
    null space of the stacked normals; the point is the minimum-norm least-squares solution
    of ``normals @ X = offsets``. ``plane_residual`` is the worst per-view disagreement and
    is the *only* keypoint-independent quality signal available -- but see the module
    docstring: on ``squat`` it is anti-correlated with the true error, so a small residual
    is necessary, not sufficient.
    """
    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if len(normals) < 2:
        raise ValueError("need at least two views to triangulate a line")
    _, _, vt = np.linalg.svd(normals)
    direction = vt[-1]
    stacked = np.vstack([normals, direction])
    point, *_ = np.linalg.lstsq(stacked, np.append(offsets, 0.0), rcond=None)
    residual = float(np.abs(normals @ point - offsets).max())
    return point, direction, residual


# ---------------------------------------------------------------------------
# Track extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarTrack:
    """Per-frame 3D bar axis for one (subject, action)."""

    subject: str
    action: str
    frames: np.ndarray          # (N,) frame indices into joints3d
    point: np.ndarray           # (N, 3) a point on the bar axis, world metres
    direction: np.ndarray       # (N, 3) unit bar axis, world
    plane_residual: np.ndarray  # (N,) worst per-view plane disagreement, metres
    n_views: np.ndarray         # (N,) views contributing

    def __len__(self) -> int:
        return len(self.frames)


def extract_bar_track(
    subject: str,
    action: str,
    split: str = "train",
    stride: int = DEFAULT_STRIDE,
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    min_views: int = 3,
) -> BarTrack:
    """Measure the 3D bar axis per frame from all available calibrated views."""
    import cv2

    joints = ds.load_joints3d(split, subject, action, root)
    frame_ids = list(range(0, len(joints), stride))
    per_frame: dict[int, list[tuple[np.ndarray, float]]] = {}

    for camera in ds.cameras(split, subject, root):
        cam_params = ds.read_cam_params(split, subject, camera, action, root)
        projected = ds.project_world_to_image(joints, cam_params, True)
        path = os.path.join(str(root), split, subject, "videos", camera, action + ".mp4")
        cap = cv2.VideoCapture(path)
        wanted, position = set(frame_ids), 0
        try:
            while position <= frame_ids[-1]:
                if position not in wanted:
                    if not cap.grab():
                        break
                    position += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                index, position = position, position + 1
                found = _bar_pixels(frame, projected[index])
                if found is None:
                    continue
                fitted = ransac_line_2d(found)
                if fitted is None:
                    continue
                _, _, inliers, _ = fitted
                per_frame.setdefault(index, []).append(
                    back_project_line(found[inliers], cam_params)
                )
        finally:
            cap.release()

    frames, points, directions, residuals, counts = [], [], [], [], []
    for index in sorted(per_frame):
        views = per_frame[index]
        if len(views) < min_views:
            continue
        point, direction, residual = triangulate_axis(
            np.stack([v[0] for v in views]), np.array([v[1] for v in views])
        )
        frames.append(index)
        points.append(point)
        directions.append(direction)
        residuals.append(residual)
        counts.append(len(views))

    return BarTrack(
        subject=subject,
        action=action,
        frames=np.asarray(frames, dtype=int),
        point=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        direction=np.asarray(directions, dtype=np.float64).reshape(-1, 3),
        plane_residual=np.asarray(residuals, dtype=np.float64),
        n_views=np.asarray(counts, dtype=int),
    )


def _bar_pixels(frame_bgr: np.ndarray, keypoints_2d: np.ndarray) -> np.ndarray | None:
    """Colour-threshold inside an ROI anchored on the upper body. Returns (M, 2) pixels.

    The keypoints place the search window only; no keypoint enters the resulting measurement.
    """
    import cv2

    anchors = keypoints_2d[[ds.R_SHOULDER, ds.L_SHOULDER, ds.R_WRIST, ds.L_WRIST,
                            ds.R_ELBOW, ds.L_ELBOW, ds.HEAD]]
    if not np.isfinite(anchors).all():
        return None
    height, width = frame_bgr.shape[:2]
    x0, y0 = np.maximum(0, anchors.min(axis=0) - 100).astype(int)
    x1, y1 = np.minimum([width, height], anchors.max(axis=0) + 100).astype(int)
    if x1 <= x0 or y1 <= y0:
        return None
    window = cv2.inRange(
        cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV), BAR_HSV_LO, BAR_HSV_HI
    )
    ys, xs = np.nonzero(window)
    if len(xs) < 40:
        return None
    return np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Target: the bar axis expressed in a body-anchored frame
# ---------------------------------------------------------------------------
def body_frame(joints3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Root-centred, heading-removed body basis. Returns ``(origin, basis, torso_len)``.

    Rows of ``basis`` are (lateral, up, anterior). Same construction as
    ``axial_rotation.canonicalize``, but the scale is returned instead of divided out so
    that residuals stay in metres. The lateral axis is ``L_HIP - R_HIP``; that leaks
    *bilateral* information, which was a real trap for the unilateral questions in the
    axial-rotation study but is harmless here -- the bar is a bilateral object and no
    feature set in :data:`BAR_KEYPOINT_SETS` is unilateral.
    """
    x = np.asarray(joints3d, dtype=np.float64)
    origin = x[:, ds.ROOT]
    centred = x - origin[:, None, :]
    lateral = centred[:, ds.L_HIP] - centred[:, ds.R_HIP]
    lateral = lateral / np.linalg.norm(lateral, axis=1, keepdims=True)
    up = centred[:, ds.THORAX] - centred[:, ds.ROOT]
    up = up - np.sum(up * lateral, axis=1, keepdims=True) * lateral
    torso = np.linalg.norm(up, axis=1)
    up = up / torso[:, None]
    return origin, np.stack([lateral, up, np.cross(lateral, up)], axis=1), torso


def axis_offset_in_body_frame(
    track: BarTrack, joints3d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bar axis -> ``(offset, direction)`` in the body frame, both endpoint-free.

    ``offset`` is the closest point of the bar axis to the body root, expressed in the body
    basis, in **metres**. Being a perpendicular foot it has only 2 free dimensions (its
    component along the bar is zero by construction), so the lateral column is degenerate
    for a bar held square to the body and only ``up``/``anterior`` carry information.

    ``anterior`` is the coaching axis: "bar over mid-foot" is a statement about exactly this
    number, and it is the quantity the withdrawn OHP sub-criterion needed.
    """
    joints = np.asarray(joints3d, dtype=np.float64)[track.frames]
    origin, basis, _ = body_frame(joints)
    rel = track.point - origin
    along = np.sum(rel * track.direction, axis=1, keepdims=True)
    foot = rel - along * track.direction                    # perpendicular foot from root
    offset = np.einsum("fij,fj->fi", basis, foot)
    direction = np.einsum("fij,fj->fi", basis, track.direction)
    direction = direction * np.sign(direction[:, :1] + 1e-12)  # kill the axis sign ambiguity
    return offset, direction


def shoulder_width(joints3d: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Per-frame shoulder width in metres -- the unit the withdrawn OHP threshold uses."""
    joints = np.asarray(joints3d, dtype=np.float64)[frames]
    return np.linalg.norm(joints[:, ds.R_SHOULDER] - joints[:, ds.L_SHOULDER], axis=1)


def mid_hand_in_body_frame(joints3d: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Centroid of the four hand-extremity points, root-relative, in the body frame (metres)."""
    joints = np.asarray(joints3d, dtype=np.float64)[frames]
    _, basis, _ = body_frame(joints)
    hands = joints[:, [ds.R_HAND_A, ds.R_HAND_B, ds.L_HAND_A, ds.L_HAND_B]].mean(axis=1)
    return np.einsum("fij,fj->fi", basis, hands - joints[:, ds.ROOT])


def constant_offset_baseline(
    target: np.ndarray, reference: np.ndarray, groups: np.ndarray
) -> float:
    """LOSO MAE of "bar = reference point + a constant offset". **Zero parameters.**

    This is the control that decides how the regression results may be worded. In a
    bar-in-hands lift the bar axis passes through the hands by physical necessity, so a model
    predicting bar position from wrist keypoints is close to the identity function plus a grip
    offset. Beating a coaching threshold therefore shows nothing on its own -- it may only show
    that the implement is bolted to a tracked body part. Report this number FIRST; a learned
    model has to beat it, not the threshold.

    The offset is estimated on the training folds only, so it leaks nothing about the held-out
    subject.
    """
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    errors = []
    for subject in sorted(set(np.asarray(groups).tolist())):
        held = np.asarray(groups) == subject
        offset = float(np.mean(target[~held] - reference[~held]))
        errors.append(float(np.mean(np.abs(target[held] - (reference[held] + offset)))))
    return float(np.mean(errors))


def keypoint_features(joints3d: np.ndarray, frames: np.ndarray, subset: list[int]) -> np.ndarray:
    """Flattened body-frame keypoint coordinates for ``subset``, scale-normalised."""
    joints = np.asarray(joints3d, dtype=np.float64)[frames]
    origin, basis, torso = body_frame(joints)
    centred = joints - origin[:, None, :]
    rotated = np.einsum("fij,fkj->fki", basis, centred) / torso[:, None, None]
    return rotated[:, subset].reshape(len(frames), -1)
