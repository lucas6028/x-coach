"""Analysis orchestration: wrap the existing ``src/`` pose pipeline for the web API.

This module performs *no* biomechanics logic of its own. It calls:
  - ``src.pose.process_videos.process_video`` (video -> pose JSON), and
  - ``src.pose.pose_rule_detector.detect_pose_rules_from_json`` (pose JSON -> faults + retrieval).

It then attaches a slimmed ``pose`` block (x, y, visibility only) so the frontend can draw a
skeleton overlay synced to the video without re-downloading the heavy world-landmark data.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app import config
from backend.app.services import storage

logger = logging.getLogger(__name__)

# NOTE: ``src.pose.process_videos`` / ``src.pose.pose_rule_detector`` are imported lazily inside
# ``analyze_video_file`` (not at module load) so importing this service does not drag in MediaPipe
# / OpenCV / torch. That keeps the web process import-light and lets the API layer be tested
# without the heavy ML stack installed; the upload path imports them on first use.

# Indices to keep from each 33-landmark frame: x, y, visibility (drop z / world landmarks).
_LANDMARK_COUNT = 33


def build_pose_block(pose_json_path: Path) -> dict[str, Any]:
    """Read a pose JSON file and return a compact overlay payload.

    Returns ``{"fps": float, "width": int, "height": int, "frames": [{"i", "lm"}]}`` where
    ``lm`` is a list of ``[x, y, visibility]`` triples (or ``None`` for frames with no pose).
    """
    payload = json.loads(pose_json_path.read_text(encoding="utf-8"))
    return build_pose_block_from_payload(payload)


def build_pose_block_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {}) or {}
    frames_out: list[dict[str, Any]] = []
    for frame in payload.get("frames", []) or []:
        landmarks = frame.get("landmarks")
        if not landmarks:
            slim = None
        else:
            slim = [
                [round(float(lm["x"]), 5), round(float(lm["y"]), 5), round(float(lm["visibility"]), 4)]
                for lm in landmarks
            ]
        frames_out.append({"i": int(frame.get("frame_index", len(frames_out))), "lm": slim})
    return {
        "fps": float(metadata.get("fps", 30.0) or 30.0),
        "width": int(metadata.get("width", 0) or 0),
        "height": int(metadata.get("height", 0) or 0),
        "frames": frames_out,
    }


def _strip_frame_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Drop the verbose per-frame metrics block; the slim pose block replaces it for the UI."""
    result.pop("frame_metrics", None)
    return result


def analyze_video_file(
    source_path: Path,
    *,
    video_id: str | None = None,
    pose_json_path: Path,
    movement: str | None = None,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    """Run the full pipeline on an arbitrary video file (the live-upload flow).

    Extracts pose to ``pose_json_path``, runs rule detection with retrieval enrichment, and
    returns the detector result with a slimmed ``pose`` block attached.

    ``pose_json_path`` is supplied by the caller (``stage_upload`` puts it in the upload's temp
    directory) rather than derived from a runtime dir: pose JSON is uploaded to object storage
    after the analysis, so its on-disk location is scratch space with the same lifetime as the
    request.

    ``max_reps`` follows the ``-1`` sentinel convention: ``-1`` means "caller said nothing" and
    resolves to ``config.DEFAULT_MAX_REPS``; ``None`` means "analyze every repetition" and is
    passed through unchanged.
    """
    # Deferred imports: pull in MediaPipe/OpenCV (process_videos) and the detector only when an
    # upload is actually analyzed, keeping module import (and server startup) lightweight.
    from src.pose.pose_rule_detector import detect_pose_rules_from_json
    from src.pose.process_videos import process_video

    vid = video_id or source_path.stem

    ok = process_video(str(source_path), str(pose_json_path))
    if not ok or not pose_json_path.exists():
        raise RuntimeError(f"Pose extraction failed for {source_path.name}")

    result = detect_pose_rules_from_json(
        pose_json_path,
        video_id=vid,
        include_retrieval=True,
        graph_file=config.KG_GRAPH_FILE,
        rag_db_dir=config.RAG_DB_DIR,
        movement=movement or config.DEFAULT_ANALYSIS_MOVEMENT,
        max_reps=config.DEFAULT_MAX_REPS if max_reps == -1 else max_reps,
    )
    result = _strip_frame_metrics(result)
    result["pose"] = build_pose_block(pose_json_path)
    result["source"] = "upload"
    return result


def _run_detector(
    payload: dict[str, Any],
    video_id: str,
    pose_json_path: Path,
    movement: str,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    # Deferred import: the detector drags in numpy/networkx; keep module import light.
    from src.pose.pose_rule_detector import detect_pose_rules_from_payload

    return detect_pose_rules_from_payload(
        payload,
        pose_json_path=pose_json_path,  # enables camera-view estimation (side-view knees_forward gating,
                                        # full confidence for knees_inward / excessive_forward_lean)
        video_id=video_id,
        include_retrieval=True,
        graph_file=config.KG_GRAPH_FILE,
        rag_db_dir=config.RAG_DB_DIR,
        movement=movement,
        max_reps=config.DEFAULT_MAX_REPS if max_reps == -1 else max_reps,
    )


def _has_detector(movement: str) -> bool:
    """Whether a rule detector is registered for ``movement``.

    Asked of the movement registry itself — the SAME source ``GET /api/movements`` advertises from
    — so what the studio offers and what this path can actually analyze are one set BY
    CONSTRUCTION. They used to be two: a hand-maintained ``{"Squat": ...}`` dict here versus the
    registry there. Registering Push-up and Overhead Press as detectors therefore advertised them
    to the studio while this path silently fell through to the ``analysis_pending`` skeleton —
    which carries no ``quality`` key, so the frontend's ``wasMeasured()`` reported a perfectly
    measurable clip as "no frame could be measured".
    """
    # Deferred like the detector import above: pulls in every movement module for its registration
    # side effects.
    from src.pose.movements import registry

    try:
        registry.get_detector(movement)
    except KeyError:
        return False
    return True


def analyze_pose_payload(
    payload: dict[str, Any],
    *,
    movement: str,
    video_id: str | None = None,
    pose_json_path: Path,
    max_reps: int | None = -1,
) -> dict[str, Any]:
    """Analyze a client-supplied pose JSON payload — no server-side MediaPipe.

    Routes by movement to its registered rule detector. Movements with no detector return a
    skeleton-only 'analysis pending' result AND NEVER WRITE ``pose_json_path`` — which is why
    ``store_artifacts`` uploads pose.json conditionally. The video is still stored by the caller.

    ``max_reps`` follows the same ``-1`` sentinel convention as ``analyze_video_file``.
    """
    vid = video_id or f"upload_{uuid.uuid4().hex[:12]}"
    pose_block = build_pose_block_from_payload(payload)
    if not _has_detector(movement):
        return {
            "video_id": vid,
            "source": "upload",
            "analysis_pending": True,
            "movement": movement,
            "detections": [],
            "retrievals": [],
            "pose": pose_block,
        }
    # Persist the client pose JSON so the detector can estimate camera view from it. Without a
    # path, detect_pose_rules_from_payload treats view as "unknown" and suppresses/downweights the
    # view-dependent squat faults (knees_forward on side view, knees_inward, excessive_forward_lean).
    pose_json_path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run_detector(payload, vid, pose_json_path, movement, max_reps=max_reps)
    result = _strip_frame_metrics(result)
    result["pose"] = pose_block
    result["source"] = "upload"
    return result


@dataclass(frozen=True)
class StagedUpload:
    """One in-flight upload: stored in the object store, staged on disk for the pipeline.

    ``prefix`` is the object-store key prefix holding every artifact for this upload, and is what
    ``videos.storage_key`` records. ``video_path`` and ``pose_path`` live in a temp directory that
    ``discard_stage`` removes once the analysis is done — they are scratch space, not storage.
    """

    video_id: str
    prefix: str
    video_path: Path
    pose_path: Path


def stage_upload(data: bytes, *, suffix: str, owner: str) -> StagedUpload:
    """Store the source video, then stage a temp copy the pose pipeline can open.

    THE OBJECT-STORE PUT HAPPENS FIRST AND IS ALLOWED TO RAISE. The raw clip is the one artifact
    that cannot be recomputed, and the put is fast next to the analysis, so discovering that
    storage is down before spending any CPU beats finishing an expensive analysis whose video
    cannot be kept. Callers map ``StorageError`` to a 503.

    The temp copy exists because ``process_video`` (OpenCV) and the detector's camera-view
    estimation both need a real filesystem path — bytes in memory are not enough.

    ``owner`` is the authenticated user's id, or ``"anon"`` for a demo upload.
    """
    video_id = f"upload_{uuid.uuid4().hex[:12]}"
    prefix = storage.upload_prefix(owner, video_id)
    storage.get_object_store().put(
        f"{prefix}/source", data, content_type=storage.video_content_type(suffix)
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{video_id}_"))
    video_path = tmp_dir / f"source{suffix}"
    video_path.write_bytes(data)
    return StagedUpload(
        video_id=video_id,
        prefix=prefix,
        video_path=video_path,
        pose_path=tmp_dir / "pose.json",
    )


def _put_artifact(staged: StagedUpload, name: str, data: bytes, content_type: str) -> int:
    """Upload one derived artifact, swallowing every failure. Returns the bytes STORED — 0 when
    the put failed. See ``store_artifacts``.

    Returning 0 rather than ``len(data)`` on failure is what keeps the storage quota honest:
    the caller adds this into ``videos.size_bytes``, and counting bytes that were never written
    would charge a user for space they do not occupy.

    ``except Exception`` is deliberate and matches ``store.persist_analysis``'s policy. A
    narrower ``except storage.StorageError`` would NOT hold the contract: ``LocalObjectStore.put``
    does real filesystem IO (``mkdir`` + ``write_bytes``) and raises ``OSError``, not
    ``StorageError`` — so on the dev/CI path a full disk would escape and sink a completed analysis.
    """
    try:
        storage.get_object_store().put(
            f"{staged.prefix}/{name}", data, content_type=content_type
        )
    except Exception:  # noqa: BLE001 — a derived artifact must never sink a completed analysis
        logger.exception("Failed to store %s for %s", name, staged.video_id)
        return 0
    return len(data)


def store_artifacts(staged: StagedUpload, *, thumbnail: bytes | None = None) -> int:
    """Best-effort upload of the derived artifacts. NEVER RAISES. Returns bytes ACTUALLY stored.

    Mirrors ``store.persist_analysis``'s policy: a storage hiccup is logged, but it must never
    discard an analysis that already cost a full pipeline run. The caller relies on that literally
    — it does not wrap this call — so every path here has to hold it, including the new return.

    The return value feeds ``videos.size_bytes`` and therefore the storage quota, so it counts
    only what landed: a failed put contributes 0, and an unreadable pose file contributes 0.

    ``pose.json`` is uploaded ONLY when one was actually produced. ``analyze_pose_payload``
    returns the ``analysis_pending`` skeleton without writing any pose JSON for a movement with
    no registered detector — which is most of the movement registry, not an edge case.
    """
    stored = 0
    if staged.pose_path.is_file():
        try:
            pose_bytes = staged.pose_path.read_bytes()
        except OSError:
            # ``is_file()`` and the read are not atomic, and the read raises OSError rather than
            # StorageError — so this needs its own guard, not the put's.
            logger.exception("Failed to read staged pose JSON for %s", staged.video_id)
        else:
            stored += _put_artifact(staged, "pose.json", pose_bytes, "application/json")
    if thumbnail:
        stored += _put_artifact(staged, "thumb.jpg", thumbnail, "image/jpeg")
    return stored


def discard_stage(staged: StagedUpload) -> None:
    """Remove the temp directory behind ``staged``. Idempotent and never raises."""
    shutil.rmtree(staged.video_path.parent, ignore_errors=True)
