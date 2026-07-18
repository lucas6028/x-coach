"""Complete test suite for the FastAPI web backend under ``backend/``.

Every backend section is exercised here:

- ``config``            -> path config, runtime-dir creation, env-driven concurrency cap.
- ``settings``          -> Supabase env settings + ``auth_configured`` + cached getter.
- ``auth``              -> bearer parsing, Supabase JWT verification, required/optional deps.
- ``services.analysis`` -> pose-block slimming, upload persistence, full upload pipeline.
- ``services.library``  -> split lookup, listing/ordering/filtering, precomputed analysis load.
- ``services.knowledge``-> KG / RAG passthrough to ``src/``.
- ``services.store``    -> user-scoped Supabase reads/writes (client mocked).
- ``routers.analyze``   -> upload endpoint (validation, success, 422 mapping, optional persist).
- ``routers.analyses``  -> per-user history list/fetch (auth required).
- ``routers.videos``    -> listing, analysis, pose, and video-file streaming endpoints.
- ``routers.knowledge`` -> graph / rag query endpoints + query validation.
- ``main``              -> health endpoint, store-presence + auth-configured reporting, startup.

The heavy ML pipeline (``src.pose.process_videos`` / ``src.pose.pose_rule_detector``) and the
knowledge retrieval (``src.knowledge.*``) are mocked so the API layer is tested in isolation,
exactly as the backend's deferred-import design intends.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app import auth, config
from backend.app import settings as app_settings
from backend.app.auth import CurrentUser, get_current_user, get_optional_user
from backend.app.main import app
from backend.app.services import analysis, knowledge, library, runtime_config, store
from backend.app.services import chat as chat_service


# --------------------------------------------------------------------------- helpers


def _pose_payload() -> dict:
    """A minimal pose JSON payload with one populated frame and one empty frame."""
    return {
        "metadata": {"fps": 24.0, "width": 640, "height": 480},
        "frames": [
            {
                "frame_index": 0,
                "landmarks": [
                    {"x": 0.123456789, "y": 0.987654321, "z": 0.5, "visibility": 0.91234}
                    for _ in range(33)
                ],
            },
            {"frame_index": 1, "landmarks": None},
        ],
    }


def _detection_payload(faults: list[str], *, view: str = "rear", retrievals=None) -> dict:
    data: dict = {
        "view": {"view_type": view},
        "detections": [{"fault_id": f, "start_s": 0.0, "end_s": 1.0} for f in faults],
        "frame_metrics": [{"i": 0, "knee_angle": 90.0}],
    }
    if retrievals is not None:
        data["retrievals"] = retrievals
    return data


class _TempConfigBase(unittest.TestCase):
    """Point all data/runtime config paths at an isolated temp tree per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos_dir = root / "videos"
        self.pose_json_dir = root / "pose_json"
        self.detections_dir = root / "detections"
        self.labels_dir = root / "labels"
        self.upload_dir = root / "uploads"
        self.upload_pose_dir = root / "upload_pose"
        self.kg_file = root / "kg.graphml"
        self.rag_dir = root / "rag_db"
        for d in (self.videos_dir, self.pose_json_dir, self.detections_dir, self.labels_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._patchers = [
            mock.patch.object(config, "VIDEOS_DIR", self.videos_dir),
            mock.patch.object(config, "POSE_JSON_DIR", self.pose_json_dir),
            mock.patch.object(config, "DETECTIONS_DIR", self.detections_dir),
            mock.patch.object(config, "LABELS_DIR", self.labels_dir),
            mock.patch.object(config, "UPLOAD_DIR", self.upload_dir),
            mock.patch.object(config, "UPLOAD_POSE_DIR", self.upload_pose_dir),
            mock.patch.object(config, "KG_GRAPH_FILE", self.kg_file),
            mock.patch.object(config, "RAG_DB_DIR", self.rag_dir),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._tmp.cleanup)
        for p in self._patchers:
            self.addCleanup(p.stop)

    # convenience writers -------------------------------------------------
    def write_detection(self, video_id: str, split: str, payload: dict) -> Path:
        path = self.detections_dir / split / f"{video_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_video(self, video_id: str, data: bytes = b"\x00\x00mp4") -> Path:
        path = self.videos_dir / f"{video_id}.mp4"
        path.write_bytes(data)
        return path

    def write_pose_json(self, video_id: str, split: str, payload: dict) -> Path:
        path = self.pose_json_dir / split / f"{video_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- config


class ConfigTests(unittest.TestCase):
    def test_ensure_runtime_dirs_creates_upload_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            up = Path(tmp) / "uploads"
            pj = Path(tmp) / "pose"
            with mock.patch.object(config, "UPLOAD_DIR", up), mock.patch.object(
                config, "UPLOAD_POSE_DIR", pj
            ):
                self.assertFalse(up.exists())
                config.ensure_runtime_dirs()
                self.assertTrue(up.is_dir())
                self.assertTrue(pj.is_dir())
                # Idempotent: a second call with the dirs present does not raise.
                config.ensure_runtime_dirs()
                self.assertTrue(up.is_dir())

    def test_repo_root_resolves_to_repository_root(self) -> None:
        # config lives at backend/app/config.py; repo root contains it.
        self.assertTrue((config.REPO_ROOT / "backend" / "app" / "config.py").exists())

    def test_split_names_default(self) -> None:
        self.assertEqual(config.SPLIT_NAMES, ("train", "val", "test"))

    def test_max_concurrent_analyses_reads_env(self) -> None:
        with mock.patch.dict(os.environ, {"XCOACH_MAX_CONCURRENT_ANALYSES": "7"}):
            reloaded = importlib.reload(config)
            try:
                self.assertEqual(reloaded.MAX_CONCURRENT_ANALYSES, 7)
            finally:
                importlib.reload(config)

    def test_max_concurrent_analyses_floors_at_one(self) -> None:
        with mock.patch.dict(os.environ, {"XCOACH_MAX_CONCURRENT_ANALYSES": "0"}):
            reloaded = importlib.reload(config)
            try:
                self.assertEqual(reloaded.MAX_CONCURRENT_ANALYSES, 1)
            finally:
                importlib.reload(config)

    def test_max_concurrent_analyses_default_present(self) -> None:
        self.assertGreaterEqual(config.MAX_CONCURRENT_ANALYSES, 1)


# --------------------------------------------------------------- services.analysis


class BuildPoseBlockTests(unittest.TestCase):
    def test_slims_landmarks_and_rounds(self) -> None:
        block = analysis.build_pose_block_from_payload(_pose_payload())
        self.assertEqual(block["fps"], 24.0)
        self.assertEqual(block["width"], 640)
        self.assertEqual(block["height"], 480)
        self.assertEqual(len(block["frames"]), 2)

        first = block["frames"][0]
        self.assertEqual(first["i"], 0)
        self.assertEqual(len(first["lm"]), 33)
        # x/y rounded to 5 places, visibility to 4; z dropped (triple only).
        self.assertEqual(first["lm"][0], [0.12346, 0.98765, 0.9123])
        self.assertEqual(len(first["lm"][0]), 3)

    def test_empty_frame_maps_to_none(self) -> None:
        block = analysis.build_pose_block_from_payload(_pose_payload())
        self.assertIsNone(block["frames"][1]["lm"])
        self.assertEqual(block["frames"][1]["i"], 1)

    def test_missing_metadata_uses_defaults(self) -> None:
        block = analysis.build_pose_block_from_payload({"frames": []})
        self.assertEqual(block, {"fps": 30.0, "width": 0, "height": 0, "frames": []})

    def test_zero_fps_falls_back_to_thirty(self) -> None:
        block = analysis.build_pose_block_from_payload({"metadata": {"fps": 0}, "frames": []})
        self.assertEqual(block["fps"], 30.0)

    def test_missing_frame_index_uses_running_length(self) -> None:
        payload = {
            "frames": [
                {"landmarks": None},
                {"landmarks": None},
            ]
        }
        block = analysis.build_pose_block_from_payload(payload)
        self.assertEqual([f["i"] for f in block["frames"]], [0, 1])

    def test_none_frames_list_yields_empty(self) -> None:
        block = analysis.build_pose_block_from_payload({"metadata": None, "frames": None})
        self.assertEqual(block["frames"], [])

    def test_build_pose_block_reads_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.json"
            path.write_text(json.dumps(_pose_payload()), encoding="utf-8")
            block = analysis.build_pose_block(path)
            self.assertEqual(block["fps"], 24.0)
            self.assertEqual(len(block["frames"]), 2)


class StripFrameMetricsTests(unittest.TestCase):
    def test_strips_frame_metrics_when_present(self) -> None:
        result = {"detections": [], "frame_metrics": [1, 2, 3]}
        out = analysis._strip_frame_metrics(result)
        self.assertNotIn("frame_metrics", out)
        self.assertIs(out, result)

    def test_strip_is_noop_without_frame_metrics(self) -> None:
        result = {"detections": []}
        out = analysis._strip_frame_metrics(result)
        self.assertEqual(out, {"detections": []})


class SaveUploadTests(_TempConfigBase):
    def test_persists_bytes_and_returns_id_and_path(self) -> None:
        video_id, dest = analysis.save_upload(b"hello-bytes", suffix=".mov")
        self.assertTrue(video_id.startswith("upload_"))
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_bytes(), b"hello-bytes")
        self.assertEqual(dest.suffix, ".mov")
        self.assertEqual(dest.parent, self.upload_dir)

    def test_default_suffix_is_mp4(self) -> None:
        _, dest = analysis.save_upload(b"x")
        self.assertEqual(dest.suffix, ".mp4")

    def test_ids_are_unique(self) -> None:
        id1, _ = analysis.save_upload(b"a")
        id2, _ = analysis.save_upload(b"b")
        self.assertNotEqual(id1, id2)


class AnalyzeVideoFileTests(_TempConfigBase):
    """Exercise the upload pipeline without importing the heavy pose stack.

    ``analyze_video_file`` lazily does ``from src.pose.process_videos import process_video`` (a
    module that pulls in OpenCV/MediaPipe). We inject a *fake* ``src.pose.process_videos`` into
    ``sys.modules`` so the deferred import binds to our stub — keeping these tests runnable on a
    lean CI runner with no ML libraries installed. ``pose_rule_detector`` is import-light, so it
    is patched directly.
    """

    def _patches(self, *, ok: bool = True, write: bool = True, payload: dict | None = None, detector_result=None):
        body = payload or _pose_payload()

        def fake_process(src: str, dst: str) -> bool:
            if write:
                Path(dst).write_text(json.dumps(body), encoding="utf-8")
            return ok

        fake_module = types.ModuleType("src.pose.process_videos")
        fake_module.process_video = fake_process  # type: ignore[attr-defined]
        module_patch = mock.patch.dict(sys.modules, {"src.pose.process_videos": fake_module})
        detect_patch = mock.patch(
            "src.pose.pose_rule_detector.detect_pose_rules_from_json",
            return_value=dict(detector_result) if detector_result is not None else {"detections": []},
        )
        return module_patch, detect_patch

    def _source(self, name: str = "clip.mp4") -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        source = self.upload_dir / name
        source.write_bytes(b"video")
        return source

    def test_full_pipeline_success(self) -> None:
        detector_result = {
            "detections": [{"fault_id": "knees_inward"}],
            "frame_metrics": [{"i": 0}],
            "retrievals": [{"fault_id": "knees_inward"}],
        }
        module_patch, detect_patch = self._patches(detector_result=detector_result)
        with module_patch, detect_patch as detect:
            result = analysis.analyze_video_file(self._source(), video_id="vid42")

        # frame_metrics stripped, slim pose block + source attached.
        self.assertNotIn("frame_metrics", result)
        self.assertEqual(result["source"], "upload")
        self.assertIn("pose", result)
        self.assertEqual(len(result["pose"]["frames"]), 2)
        # detector called with retrieval enrichment + configured stores.
        kwargs = detect.call_args.kwargs
        self.assertTrue(kwargs["include_retrieval"])
        self.assertEqual(kwargs["video_id"], "vid42")
        self.assertEqual(kwargs["graph_file"], self.kg_file)
        self.assertEqual(kwargs["rag_db_dir"], self.rag_dir)
        # analyze retrieval is scoped to the transitional squat-only movement — the branch's core
        # safety property; without it the multi-movement graph query would span all 16 movements.
        self.assertEqual(kwargs["movement"], config.DEFAULT_ANALYSIS_MOVEMENT)

    def test_raises_when_process_video_returns_false(self) -> None:
        module_patch, detect_patch = self._patches(ok=False)
        with module_patch, detect_patch as detect:
            with self.assertRaises(RuntimeError):
                analysis.analyze_video_file(self._source())
            detect.assert_not_called()

    def test_raises_when_pose_json_missing(self) -> None:
        # process_video reports success but writes no file -> RuntimeError.
        module_patch, detect_patch = self._patches(ok=True, write=False)
        with module_patch, detect_patch:
            with self.assertRaises(RuntimeError):
                analysis.analyze_video_file(self._source())

    def test_video_id_defaults_to_source_stem(self) -> None:
        module_patch, detect_patch = self._patches()
        with module_patch, detect_patch as detect:
            analysis.analyze_video_file(self._source("myclip.mp4"))
        self.assertEqual(detect.call_args.kwargs["video_id"], "myclip")


# ---------------------------------------------------------------- services.library


class LibraryHelperTests(_TempConfigBase):
    def test_split_of_finds_split(self) -> None:
        self.write_detection("clipA", "val", _detection_payload(["knees_inward"]))
        self.assertEqual(library._split_of("clipA"), "val")
        self.assertIsNone(library._split_of("missing"))

    def test_detection_path(self) -> None:
        self.write_detection("clipA", "test", _detection_payload([]))
        self.assertEqual(
            library.detection_path("clipA"), self.detections_dir / "test" / "clipA.json"
        )
        self.assertIsNone(library.detection_path("missing"))

    def test_pose_json_path(self) -> None:
        self.write_pose_json("clipA", "train", _pose_payload())
        self.assertEqual(
            library.pose_json_path("clipA"), self.pose_json_dir / "train" / "clipA.json"
        )
        self.assertIsNone(library.pose_json_path("missing"))

    def test_video_path(self) -> None:
        self.write_video("clipA")
        self.assertEqual(library.video_path("clipA"), self.videos_dir / "clipA.mp4")
        self.assertIsNone(library.video_path("missing"))

    def test_uploaded_video_path_resolves_exact_name(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "upload_abc.webm").write_bytes(b"v")
        self.assertEqual(
            library.uploaded_video_path("upload_abc"), self.upload_dir / "upload_abc.webm"
        )
        self.assertIsNone(library.uploaded_video_path("missing"))

    def test_uploaded_video_path_does_not_glob(self) -> None:
        # Regression for the glob-injection IDOR: a wildcard id must never expand to match a
        # real upload — only the exact requested id may resolve.
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "upload_secret.mp4").write_bytes(b"v")
        self.assertIsNone(library.uploaded_video_path("*"))
        self.assertIsNone(library.uploaded_video_path("upload_*"))
        self.assertIsNone(library.uploaded_video_path("upload_s*"))

    def test_path_resolvers_reject_unsafe_ids(self) -> None:
        # Even with matching files on disk, an unsafe id resolves to nothing.
        self.write_detection("clipA", "train", _detection_payload([]))
        self.write_pose_json("clipA", "train", _pose_payload())
        self.write_video("clipA")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "clipA.mp4").write_bytes(b"v")
        for bad in ("*", "upload_*", "a?b", "a[bc]", "..", "../etc", "a/b", "a\\b"):
            self.assertIsNone(library.detection_path(bad), bad)
            self.assertIsNone(library.pose_json_path(bad), bad)
            self.assertIsNone(library.video_path(bad), bad)
            self.assertIsNone(library.uploaded_video_path(bad), bad)


class IsSafeVideoIdTests(unittest.TestCase):
    def test_accepts_normal_ids(self) -> None:
        for ok in ("clipA", "upload_ab12cd34ef56", "subject-01_rep.2", "1", "a.b-c_d"):
            self.assertTrue(library.is_safe_video_id(ok), ok)

    def test_rejects_glob_traversal_and_separators(self) -> None:
        for bad in ("", "*", "?", "a*b", "a?b", "[abc]", "a/b", "a\\b", "..", "../x", "x\x00y"):
            self.assertFalse(library.is_safe_video_id(bad), bad)


class ListVideosTests(_TempConfigBase):
    def _seed(self) -> None:
        # clipA: has faults; clipB: clean; clipC: detection present but no video file.
        self.write_detection("clipA", "train", _detection_payload(["knees_inward", "knees_forward"]))
        self.write_video("clipA")
        self.write_detection("clipB", "train", _detection_payload([], view="front"))
        self.write_video("clipB")
        self.write_detection("clipC", "val", _detection_payload(["knees_inward"]))
        # no video for clipC

    def test_lists_only_entries_with_videos(self) -> None:
        self._seed()
        out = library.list_videos()
        ids = [it["video_id"] for it in out["items"]]
        self.assertIn("clipA", ids)
        self.assertIn("clipB", ids)
        self.assertNotIn("clipC", ids)  # no video file -> skipped
        self.assertEqual(out["total"], 2)

    def test_faulty_clips_sorted_first(self) -> None:
        self._seed()
        out = library.list_videos()
        # clipA (2 faults) before clipB (0 faults).
        self.assertEqual(out["items"][0]["video_id"], "clipA")
        self.assertEqual(out["items"][0]["fault_count"], 2)
        self.assertEqual(out["items"][0]["faults"], ["knees_forward", "knees_inward"])
        self.assertEqual(out["items"][0]["view_type"], "rear")
        self.assertEqual(out["items"][-1]["video_id"], "clipB")

    def test_fault_filter(self) -> None:
        self._seed()
        out = library.list_videos(fault="knees_forward")
        ids = [it["video_id"] for it in out["items"]]
        self.assertEqual(ids, ["clipA"])

    def test_fault_filter_no_match(self) -> None:
        self._seed()
        out = library.list_videos(fault="butt_wink")
        self.assertEqual(out["items"], [])

    def test_pagination(self) -> None:
        for i in range(5):
            self.write_detection(f"clip{i}", "train", _detection_payload(["knees_inward"]))
            self.write_video(f"clip{i}")
        page1 = library.list_videos(limit=2, offset=0)
        page2 = library.list_videos(limit=2, offset=2)
        self.assertEqual(page1["total"], 5)
        self.assertEqual(len(page1["items"]), 2)
        self.assertEqual(len(page2["items"]), 2)
        self.assertNotEqual(
            [it["video_id"] for it in page1["items"]],
            [it["video_id"] for it in page2["items"]],
        )

    def test_missing_detections_dir_returns_empty(self) -> None:
        # No split subdirs created -> graceful empty result.
        out = library.list_videos()
        self.assertEqual(out, {"total": 0, "items": []})

    def test_view_type_defaults_to_unknown(self) -> None:
        payload = {"detections": [{"fault_id": "knees_inward"}]}  # no "view" key
        self.write_detection("clipX", "train", payload)
        self.write_video("clipX")
        out = library.list_videos()
        self.assertEqual(out["items"][0]["view_type"], "unknown")


class GroundTruthLabelsTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        library._labels.cache_clear()
        self.addCleanup(library._labels.cache_clear)

    def test_returns_segments_for_matching_video(self) -> None:
        kf = self.labels_dir / "error_knees_forward.json"
        ki = self.labels_dir / "error_knees_inward.json"
        kf.write_text(json.dumps({"clipA": [[0.0, 1.0]]}), encoding="utf-8")
        ki.write_text(json.dumps({"clipB": [[2.0, 3.0]]}), encoding="utf-8")
        label_files = {"knees_forward": kf, "knees_inward": ki}
        with mock.patch.object(library, "_LABEL_FILES", label_files):
            library._labels.cache_clear()
            self.assertEqual(library.ground_truth_labels("clipA"), {"knees_forward": [[0.0, 1.0]]})
            self.assertEqual(library.ground_truth_labels("clipB"), {"knees_inward": [[2.0, 3.0]]})
            self.assertEqual(library.ground_truth_labels("nobody"), {})

    def test_missing_label_files_yield_empty(self) -> None:
        with mock.patch.object(
            library,
            "_LABEL_FILES",
            {"knees_forward": self.labels_dir / "nope.json"},
        ):
            library._labels.cache_clear()
            self.assertEqual(library.ground_truth_labels("clipA"), {})


class LoadAnalysisTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        library._labels.cache_clear()
        self.addCleanup(library._labels.cache_clear)

    def test_raises_when_no_detection(self) -> None:
        with self.assertRaises(FileNotFoundError):
            library.load_analysis("ghost")

    def test_enriches_retrieval_when_missing(self) -> None:
        self.write_detection("clipA", "train", _detection_payload(["knees_inward"]))
        self.write_pose_json("clipA", "train", _pose_payload())
        with mock.patch.object(
            library,
            "retrieve_contexts_for_detections",
            return_value=[{"fault_id": "knees_inward", "snippets": []}],
        ) as retr:
            result = library.load_analysis("clipA")
        retr.assert_called_once()
        self.assertEqual(result["retrievals"], [{"fault_id": "knees_inward", "snippets": []}])
        self.assertNotIn("frame_metrics", result)
        self.assertEqual(result["source"], "library")
        self.assertIn("pose", result)
        self.assertIn("ground_truth", result)
        # retrieval wired to the configured stores
        self.assertEqual(retr.call_args.kwargs["graph_file"], self.kg_file)
        self.assertEqual(retr.call_args.kwargs["rag_db_dir"], self.rag_dir)
        # library retrieval is likewise scoped to the squat-only movement.
        self.assertEqual(retr.call_args.kwargs["movement"], config.DEFAULT_ANALYSIS_MOVEMENT)

    def test_keeps_existing_retrievals(self) -> None:
        payload = _detection_payload(["knees_inward"], retrievals=[{"cached": True}])
        self.write_detection("clipA", "train", payload)
        with mock.patch.object(library, "retrieve_contexts_for_detections") as retr:
            result = library.load_analysis("clipA")
        retr.assert_not_called()
        self.assertEqual(result["retrievals"], [{"cached": True}])

    def test_no_retrieval_when_no_detections(self) -> None:
        self.write_detection("clipA", "train", _detection_payload([]))
        with mock.patch.object(library, "retrieve_contexts_for_detections") as retr:
            result = library.load_analysis("clipA")
        retr.assert_not_called()
        self.assertNotIn("retrievals", result)

    def test_pose_absent_omits_pose_block(self) -> None:
        self.write_detection("clipA", "train", _detection_payload([]))
        result = library.load_analysis("clipA")
        self.assertNotIn("pose", result)


# -------------------------------------------------------------- services.knowledge


class KnowledgeServiceTests(_TempConfigBase):
    def test_graph_context_passthrough(self) -> None:
        with mock.patch.object(
            knowledge, "retrieve_graph_context", return_value={"nodes": []}
        ) as rg:
            out = knowledge.graph_context("knees inward", hops=2, max_seeds=3)
        self.assertEqual(out, {"nodes": []})
        rg.assert_called_once_with(
            "knees inward", graph_file=self.kg_file, hops=2, max_seeds=3, movement=None
        )

    def test_graph_context_defaults(self) -> None:
        with mock.patch.object(knowledge, "retrieve_graph_context", return_value={}) as rg:
            knowledge.graph_context("q")
        self.assertEqual(rg.call_args.kwargs["hops"], 1)
        self.assertEqual(rg.call_args.kwargs["max_seeds"], 5)

    def test_rag_snippets_passthrough(self) -> None:
        with mock.patch.object(
            knowledge, "query_vector_db", return_value=[{"text": "do x"}]
        ) as qv:
            out = knowledge.rag_snippets("depth", top_k=3)
        self.assertEqual(out, {"query": "depth", "results": [{"text": "do x"}]})
        qv.assert_called_once_with("depth", db_dir=self.rag_dir, top_k=3)

    def test_movement_faults_passthrough(self) -> None:
        rows = [{"name": "Knee Valgus", "connectivity": 3}]
        with mock.patch.object(knowledge, "list_movement_faults", return_value=rows) as lf:
            out = knowledge.movement_faults("Overhead Press")
        self.assertEqual(out, rows)
        lf.assert_called_once_with(graph_file=self.kg_file, movement="Overhead Press")


# --------------------------------------------------------------------- routers


class AnalyzeRouterTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(app)

    def test_rejects_unsupported_suffix(self) -> None:
        resp = self.client.post(
            "/api/analyze", files={"file": ("clip.txt", b"data", "text/plain")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported", resp.json()["detail"])

    def test_rejects_empty_file(self) -> None:
        resp = self.client.post(
            "/api/analyze", files={"file": ("clip.mp4", b"", "video/mp4")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())

    def test_success_returns_analysis(self) -> None:
        fake_result = {"detections": [], "source": "upload", "pose": {"frames": []}}
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(analysis, "analyze_video_file", return_value=fake_result):
            resp = self.client.post(
                "/api/analyze", files={"file": ("clip.mp4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), fake_result)

    def test_pipeline_runtime_error_maps_to_422(self) -> None:
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(
            analysis, "analyze_video_file", side_effect=RuntimeError("boom")
        ):
            resp = self.client.post(
                "/api/analyze", files={"file": ("clip.mp4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.json()["detail"], "boom")

    def test_missing_file_is_422(self) -> None:
        resp = self.client.post("/api/analyze")
        self.assertEqual(resp.status_code, 422)

    def test_uppercase_suffix_accepted(self) -> None:
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(analysis, "analyze_video_file", return_value={"ok": True}):
            resp = self.client.post(
                "/api/analyze", files={"file": ("CLIP.MP4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)

    def test_anonymous_upload_does_not_persist(self) -> None:
        fake_result = {"detections": [], "source": "upload"}
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(
            analysis, "analyze_video_file", return_value=fake_result
        ), mock.patch.object(store, "persist_analysis") as persist:
            resp = self.client.post(
                "/api/analyze", files={"file": ("clip.mp4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        persist.assert_not_called()
        self.assertNotIn("analysis_id", resp.json())

    def test_authenticated_upload_persists_and_returns_id(self) -> None:
        app.dependency_overrides[get_optional_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        fake_result = {"detections": [], "source": "upload", "pose": {"frames": []}}
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(
            analysis, "analyze_video_file", return_value=fake_result
        ), mock.patch.object(store, "persist_analysis", return_value="analysis-1") as persist:
            resp = self.client.post(
                "/api/analyze", files={"file": ("clip.mp4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["analysis_id"], "analysis-1")
        kwargs = persist.call_args.kwargs
        self.assertEqual(kwargs["user_id"], "u1")
        self.assertEqual(kwargs["token"], "tok")
        self.assertEqual(kwargs["video_id"], "upload_abc")
        self.assertEqual(kwargs["source"], "upload")
        self.assertEqual(kwargs["filename"], "clip.mp4")

    def test_authenticated_upload_persist_failure_sets_id_none(self) -> None:
        app.dependency_overrides[get_optional_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        with mock.patch.object(
            analysis, "save_upload", return_value=("upload_abc", self.upload_dir / "u.mp4")
        ), mock.patch.object(
            analysis, "analyze_video_file", return_value={"detections": []}
        ), mock.patch.object(
            store, "persist_analysis", side_effect=RuntimeError("db down")
        ):
            resp = self.client.post(
                "/api/analyze", files={"file": ("clip.mp4", b"abcd", "video/mp4")}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["analysis_id"])


class VideosRouterTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(app)

    def test_list_videos_endpoint(self) -> None:
        self.write_detection("clipA", "train", _detection_payload(["knees_inward"]))
        self.write_video("clipA")
        resp = self.client.get("/api/videos")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["video_id"], "clipA")

    def test_list_videos_passes_query_params(self) -> None:
        with mock.patch.object(
            library, "list_videos", return_value={"total": 0, "items": []}
        ) as lv:
            self.client.get("/api/videos", params={"limit": 5, "offset": 10, "fault": "knees_inward"})
        lv.assert_called_once_with(limit=5, offset=10, fault="knees_inward")

    def test_get_analysis_success(self) -> None:
        self.write_detection("clipA", "train", _detection_payload([]))
        resp = self.client.get("/api/analysis/clipA")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["source"], "library")

    def test_get_analysis_404(self) -> None:
        resp = self.client.get("/api/analysis/ghost")
        self.assertEqual(resp.status_code, 404)

    def test_get_pose_success(self) -> None:
        self.write_pose_json("clipA", "train", _pose_payload())
        resp = self.client.get("/api/pose/clipA")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fps"], 24.0)
        self.assertEqual(len(resp.json()["frames"]), 2)

    def test_get_pose_404(self) -> None:
        resp = self.client.get("/api/pose/ghost")
        self.assertEqual(resp.status_code, 404)

    def test_get_video_file_from_library(self) -> None:
        self.write_video("clipA", data=b"\x00\x01mp4data")
        resp = self.client.get("/api/video-file/clipA")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"\x00\x01mp4data")
        self.assertEqual(resp.headers["content-type"], "video/mp4")

    def test_get_video_file_upload_fallback(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "upload_xyz.mp4").write_bytes(b"uploaded")
        resp = self.client.get("/api/video-file/upload_xyz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"uploaded")

    def test_get_video_file_404(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        resp = self.client.get("/api/video-file/ghost")
        self.assertEqual(resp.status_code, 404)

    def test_get_video_file_rejects_glob_wildcard(self) -> None:
        # Regression for the glob-injection IDOR: a wildcard id must not stream another user's
        # upload. Previously ``glob(f"{video_id}.*")`` let ``*`` match an arbitrary upload.
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "upload_victim.mp4").write_bytes(b"secret")
        for wid in ("*", "upload_*", "upload_v*"):
            resp = self.client.get(f"/api/video-file/{wid}")
            self.assertEqual(resp.status_code, 404, wid)
            self.assertNotIn(b"secret", resp.content)


class KnowledgeRouterTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(app)

    def test_graph_endpoint(self) -> None:
        with mock.patch.object(
            knowledge, "graph_context", return_value={"nodes": [1]}
        ) as gc:
            resp = self.client.get("/api/knowledge/graph", params={"query": "knees", "hops": 2})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"nodes": [1]})
        # The router now also threads the admin-tunable KG seed default through to the service.
        # `movement` defaults to None when the query param is omitted.
        gc.assert_called_once_with("knees", hops=2, max_seeds=5, movement=None)

    def test_graph_endpoint_forwards_movement(self) -> None:
        with mock.patch.object(
            knowledge, "graph_context", return_value={"nodes": []}
        ) as gc:
            resp = self.client.get(
                "/api/knowledge/graph", params={"query": "knee valgus", "movement": "Squat"}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gc.call_args.kwargs["movement"], "Squat")

    def test_graph_default_hops(self) -> None:
        with mock.patch.object(knowledge, "graph_context", return_value={}) as gc:
            self.client.get("/api/knowledge/graph", params={"query": "knees"})
        self.assertEqual(gc.call_args.kwargs["hops"], 1)

    def test_graph_requires_query(self) -> None:
        self.assertEqual(self.client.get("/api/knowledge/graph").status_code, 422)
        # empty query violates min_length=1
        resp = self.client.get("/api/knowledge/graph", params={"query": ""})
        self.assertEqual(resp.status_code, 422)

    def test_faults_endpoint(self) -> None:
        rows = [{"name": "Knee Valgus", "connectivity": 3}, {"name": "Bar Drift", "connectivity": 0}]
        with mock.patch.object(knowledge, "movement_faults", return_value=rows) as mf:
            resp = self.client.get("/api/knowledge/faults", params={"movement": "Overhead Press"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"movement": "Overhead Press", "faults": rows})
        mf.assert_called_once_with("Overhead Press")

    def test_faults_requires_movement(self) -> None:
        self.assertEqual(self.client.get("/api/knowledge/faults").status_code, 422)

    def test_rag_endpoint(self) -> None:
        with mock.patch.object(
            knowledge, "rag_snippets", return_value={"query": "depth", "results": []}
        ) as rs:
            resp = self.client.get("/api/knowledge/rag", params={"query": "depth", "top_k": 3})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"query": "depth", "results": []})
        rs.assert_called_once_with("depth", top_k=3)

    def test_rag_default_top_k(self) -> None:
        with mock.patch.object(knowledge, "rag_snippets", return_value={}) as rs:
            self.client.get("/api/knowledge/rag", params={"query": "depth"})
        self.assertEqual(rs.call_args.kwargs["top_k"], 5)

    def test_rag_requires_query(self) -> None:
        self.assertEqual(self.client.get("/api/knowledge/rag").status_code, 422)

    def test_graph_rejects_out_of_range_hops(self) -> None:
        # Unbounded / non-positive traversal depth is rejected before any retrieval runs.
        with mock.patch.object(knowledge, "graph_context", return_value={}) as gc:
            for hops in (0, -1, 4, 9999):
                resp = self.client.get(
                    "/api/knowledge/graph", params={"query": "knees", "hops": hops}
                )
                self.assertEqual(resp.status_code, 422, hops)
            gc.assert_not_called()

    def test_graph_accepts_hops_upper_bound(self) -> None:
        with mock.patch.object(knowledge, "graph_context", return_value={}) as gc:
            resp = self.client.get("/api/knowledge/graph", params={"query": "knees", "hops": 3})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(gc.call_args.kwargs["hops"], 3)

    def test_rag_rejects_out_of_range_top_k(self) -> None:
        with mock.patch.object(knowledge, "rag_snippets", return_value={}) as rs:
            for top_k in (0, -1, 51, 9999):
                resp = self.client.get(
                    "/api/knowledge/rag", params={"query": "depth", "top_k": top_k}
                )
                self.assertEqual(resp.status_code, 422, top_k)
            rs.assert_not_called()

    def test_rag_accepts_top_k_upper_bound(self) -> None:
        with mock.patch.object(knowledge, "rag_snippets", return_value={}) as rs:
            resp = self.client.get("/api/knowledge/rag", params={"query": "depth", "top_k": 50})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(rs.call_args.kwargs["top_k"], 50)


# ------------------------------------------------------------------------- main


class MainAppTests(_TempConfigBase):
    def setUp(self) -> None:
        super().setUp()
        self.client = TestClient(app)

    def test_health_all_stores_present(self) -> None:
        self.kg_file.write_text("graph", encoding="utf-8")
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["stores"]["labeled_videos"])
        self.assertTrue(body["stores"]["detections"])
        self.assertTrue(body["stores"]["kg_graph"])
        self.assertTrue(body["stores"]["rag_db"])

    def test_health_reports_missing_stores(self) -> None:
        # kg + rag absent (never created in temp tree).
        resp = self.client.get("/api/health")
        body = resp.json()
        self.assertFalse(body["stores"]["kg_graph"])
        self.assertFalse(body["stores"]["rag_db"])
        # videos + detections dirs were created in setUp.
        self.assertTrue(body["stores"]["labeled_videos"])

    def test_startup_creates_runtime_dirs(self) -> None:
        self.assertFalse(self.upload_dir.exists())
        with TestClient(app):  # triggers the startup event
            self.assertTrue(self.upload_dir.is_dir())
            self.assertTrue(self.upload_pose_dir.is_dir())

    def test_cors_headers_present(self) -> None:
        resp = self.client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"}
        )
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"), "http://localhost:5173"
        )

    def test_openapi_lists_all_routes(self) -> None:
        paths = self.client.get("/openapi.json").json()["paths"]
        for expected in (
            "/api/analyze",
            "/api/analyses",
            "/api/analyses/{analysis_id}",
            "/api/videos",
            "/api/analysis/{video_id}",
            "/api/pose/{video_id}",
            "/api/video-file/{video_id}",
            "/api/knowledge/graph",
            "/api/knowledge/rag",
            "/api/chat",
            "/api/health",
        ):
            self.assertIn(expected, paths)

    def test_health_reports_chat_models_and_default(self) -> None:
        resp = self.client.get("/api/health")
        body = resp.json()
        ids = body["chat_models"]  # a list of model-id strings
        self.assertTrue(ids)
        self.assertTrue(all(isinstance(m, str) for m in ids))
        # The default is always one of the offered models.
        self.assertIn(body["chat_default"], ids)

    def test_health_reports_auth_configured_true(self) -> None:
        with mock.patch(
            "backend.app.main.get_settings",
            return_value=types.SimpleNamespace(
                auth_configured=True, chat_configured=True, line_configured=True
            ),
        ):
            resp = self.client.get("/api/health")
        body = resp.json()
        self.assertTrue(body["auth_configured"])
        self.assertTrue(body["chat_configured"])
        self.assertTrue(body["line_configured"])

    def test_health_reports_auth_not_configured(self) -> None:
        with mock.patch(
            "backend.app.main.get_settings",
            return_value=types.SimpleNamespace(
                auth_configured=False, chat_configured=False, line_configured=False
            ),
        ):
            resp = self.client.get("/api/health")
        body = resp.json()
        self.assertFalse(body["auth_configured"])
        self.assertFalse(body["chat_configured"])
        self.assertFalse(body["line_configured"])


# ------------------------------------------------------------------------- settings


class SettingsTests(unittest.TestCase):
    def test_auth_configured_true_when_present(self) -> None:
        s = app_settings.Settings(
            supabase_url="https://x.supabase.co",
            supabase_anon_key="anon",
        )
        self.assertTrue(s.auth_configured)

    def test_auth_configured_false_when_any_missing(self) -> None:
        s = app_settings.Settings(
            supabase_url="https://x.supabase.co",
            supabase_anon_key="",
        )
        self.assertFalse(s.auth_configured)

    def test_chat_configured_tracks_llm_key(self) -> None:
        self.assertTrue(app_settings.Settings(llm_api_key="sk-or-123").chat_configured)
        self.assertFalse(app_settings.Settings(llm_api_key="").chat_configured)

    def test_get_settings_is_cached(self) -> None:
        app_settings.get_settings.cache_clear()
        self.addCleanup(app_settings.get_settings.cache_clear)
        self.assertIs(app_settings.get_settings(), app_settings.get_settings())


# ----------------------------------------------------------------------------- auth


def _auth_settings(*, configured: bool = True):
    return types.SimpleNamespace(
        auth_configured=configured,
        supabase_url="https://x.supabase.co",
        supabase_anon_key="anon-key",
    )


def _fake_supabase(*, user=None, raises: bool = False):
    """A fake ``supabase`` module whose ``create_client(...).auth.get_user`` is controllable."""
    module = types.ModuleType("supabase")

    def create_client(url, key):
        client = mock.Mock()
        if raises:
            client.auth.get_user.side_effect = RuntimeError("invalid token")
        else:
            client.auth.get_user.return_value = types.SimpleNamespace(user=user)
        return client

    module.create_client = create_client  # type: ignore[attr-defined]
    return module


class ExtractBearerTests(unittest.TestCase):
    def test_none_header_returns_none(self) -> None:
        self.assertIsNone(auth._extract_bearer(None))

    def test_single_token_returns_none(self) -> None:
        self.assertIsNone(auth._extract_bearer("Bearer"))

    def test_empty_token_returns_none(self) -> None:
        self.assertIsNone(auth._extract_bearer("Bearer    "))

    def test_wrong_scheme_returns_none(self) -> None:
        self.assertIsNone(auth._extract_bearer("Token abc"))

    def test_valid_bearer(self) -> None:
        self.assertEqual(auth._extract_bearer("Bearer abc.def"), "abc.def")

    def test_scheme_is_case_insensitive(self) -> None:
        self.assertEqual(auth._extract_bearer("bearer abc.def"), "abc.def")


class VerifyTests(unittest.TestCase):
    def test_valid_token_returns_user(self) -> None:
        fake = _fake_supabase(user=types.SimpleNamespace(id="user-42", email="a@b.c"))
        with mock.patch.object(auth, "get_settings", return_value=_auth_settings()), mock.patch.dict(
            sys.modules, {"supabase": fake}
        ):
            user = auth._verify("tok")
        self.assertEqual(user.id, "user-42")
        self.assertEqual(user.email, "a@b.c")
        self.assertEqual(user.token, "tok")

    def test_not_configured_is_503(self) -> None:
        with mock.patch.object(auth, "get_settings", return_value=_auth_settings(configured=False)):
            with self.assertRaises(HTTPException) as ctx:
                auth._verify("whatever")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_get_user_error_is_401(self) -> None:
        fake = _fake_supabase(raises=True)
        with mock.patch.object(auth, "get_settings", return_value=_auth_settings()), mock.patch.dict(
            sys.modules, {"supabase": fake}
        ):
            with self.assertRaises(HTTPException) as ctx:
                auth._verify("bad")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_no_user_is_401(self) -> None:
        fake = _fake_supabase(user=None)
        with mock.patch.object(auth, "get_settings", return_value=_auth_settings()), mock.patch.dict(
            sys.modules, {"supabase": fake}
        ):
            with self.assertRaises(HTTPException) as ctx:
                auth._verify("tok")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_user_without_id_is_401(self) -> None:
        fake = _fake_supabase(user=types.SimpleNamespace(id=None, email=None))
        with mock.patch.object(auth, "get_settings", return_value=_auth_settings()), mock.patch.dict(
            sys.modules, {"supabase": fake}
        ):
            with self.assertRaises(HTTPException) as ctx:
                auth._verify("tok")
        self.assertEqual(ctx.exception.status_code, 401)


class AuthDependencyTests(unittest.TestCase):
    def test_get_current_user_missing_token_is_401(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(authorization=None)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_current_user_valid(self) -> None:
        sentinel = CurrentUser(id="u9", token="tok")
        with mock.patch.object(auth, "_verify", return_value=sentinel) as verify:
            user = get_current_user(authorization="Bearer tok")
        self.assertEqual(user.id, "u9")
        verify.assert_called_once_with("tok")

    def test_get_optional_user_no_header_returns_none(self) -> None:
        self.assertIsNone(get_optional_user(authorization=None))

    def test_get_optional_user_valid(self) -> None:
        sentinel = CurrentUser(id="u9", token="tok")
        with mock.patch.object(auth, "_verify", return_value=sentinel):
            user = get_optional_user(authorization="Bearer tok")
        self.assertEqual(user.id, "u9")

    def test_get_optional_user_present_but_invalid_raises_401(self) -> None:
        with mock.patch.object(
            auth, "_verify", side_effect=HTTPException(status_code=401, detail="bad")
        ):
            with self.assertRaises(HTTPException) as ctx:
                get_optional_user(authorization="Bearer tok")
        self.assertEqual(ctx.exception.status_code, 401)


# ---------------------------------------------------------------------- services.store


class _Resp:
    def __init__(self, data=None, count=None) -> None:
        self.data = data
        self.count = count


class _FakeQuery:
    """Records chained PostgREST calls and returns a preset response on execute()."""

    def __init__(self, resp: _Resp) -> None:
        self._resp = resp
        self.inserted: dict | None = None
        self.upserted: dict | None = None
        self.range_args: tuple | None = None
        self.deleted = False

    def upsert(self, row, **kwargs):
        self.upserted = row
        return self

    def insert(self, row, **kwargs):
        self.inserted = row
        return self

    def delete(self, *a, **k):
        self.deleted = True
        return self

    def rpc(self, *a, **k):
        return self

    def select(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def range(self, start, end):
        self.range_args = (start, end)
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return self._resp


def _fake_client(resp: _Resp) -> tuple[mock.Mock, _FakeQuery]:
    query = _FakeQuery(resp)
    client = mock.Mock()
    client.table.return_value = query
    client.rpc.return_value = query  # so store.admin_list_users' .rpc(...).execute() is recorded too
    return client, query


class StoreSummarizeTests(unittest.TestCase):
    def test_summarize_promotes_fields(self) -> None:
        result = {
            "view": {"view_type": "rear"},
            "detections": [{"fault_id": "a"}, {"fault_id": "b"}],
            "pipeline_version": "rules-v3",
        }
        self.assertEqual(store._summarize(result), ("rear", 2, "rules-v3"))

    def test_summarize_handles_missing(self) -> None:
        self.assertEqual(store._summarize({}), (None, 0, None))


class StoreUserClientTests(unittest.TestCase):
    def test_user_client_auths_with_token(self) -> None:
        created: dict = {}
        fake_supabase = types.ModuleType("supabase")

        def fake_create_client(url, key):
            created["args"] = (url, key)
            client = mock.Mock()
            return client

        fake_supabase.create_client = fake_create_client  # type: ignore[attr-defined]
        fake_settings = types.SimpleNamespace(supabase_url="u", supabase_anon_key="k")
        with mock.patch.dict(sys.modules, {"supabase": fake_supabase}), mock.patch(
            "backend.app.settings.get_settings", return_value=fake_settings
        ):
            client = store._user_client("tok123")
        self.assertEqual(created["args"], ("u", "k"))
        client.postgrest.auth.assert_called_once_with("tok123")


class StorePersistTests(unittest.TestCase):
    def test_persist_inserts_and_returns_id(self) -> None:
        client, query = _fake_client(_Resp(data=[{"id": "analysis-1"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            aid = store.persist_analysis(
                token="t",
                user_id="u1",
                video_id="vid",
                source="upload",
                result={"view": {"view_type": "rear"}, "detections": [1]},
                filename="clip.mp4",
            )
        self.assertEqual(aid, "analysis-1")
        # Promoted columns landed on the analyses insert.
        self.assertEqual(query.inserted["user_id"], "u1")
        self.assertEqual(query.inserted["view_type"], "rear")
        self.assertEqual(query.inserted["fault_count"], 1)
        # The video row was upserted with the runtime storage key.
        self.assertEqual(query.upserted["video_id"], "vid")
        self.assertEqual(query.upserted["status"], "done")

    def test_persist_returns_empty_when_no_row(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            aid = store.persist_analysis(
                token="t", user_id="u1", video_id="vid", source="upload", result={}
            )
        self.assertEqual(aid, "")


class StoreListTests(unittest.TestCase):
    def test_list_returns_total_and_items(self) -> None:
        client, query = _fake_client(_Resp(data=[{"id": "a"}], count=3))
        with mock.patch.object(store, "_user_client", return_value=client):
            out = store.list_analyses(token="t", limit=10, offset=5)
        self.assertEqual(out, {"total": 3, "items": [{"id": "a"}]})
        self.assertEqual(query.range_args, (5, 14))

    def test_list_handles_empty(self) -> None:
        client, _ = _fake_client(_Resp(data=None, count=None))
        with mock.patch.object(store, "_user_client", return_value=client):
            out = store.list_analyses(token="t")
        self.assertEqual(out, {"total": 0, "items": []})

    def test_list_clamps_limit_and_offset(self) -> None:
        client, query = _fake_client(_Resp(data=[], count=0))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.list_analyses(token="t", limit=9999, offset=-5)
        # limit clamped to 200, offset floored to 0 -> range(0, 199).
        self.assertEqual(query.range_args, (0, 199))


class StoreDeleteTests(unittest.TestCase):
    def test_delete_all_returns_count_and_filters_by_user(self) -> None:
        client, query = _fake_client(_Resp(data=[{"id": "a"}, {"id": "b"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            n = store.delete_all_analyses(token="t", user_id="u1")
        self.assertEqual(n, 2)
        self.assertTrue(query.deleted)
        # The analyses, source video, and conversation rows are all cleared.
        self.assertEqual(client.table.call_count, 3)
        client.table.assert_any_call("analyses")
        client.table.assert_any_call("videos")
        client.table.assert_any_call("conversations")

    def test_delete_all_handles_empty(self) -> None:
        client, _ = _fake_client(_Resp(data=None))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(store.delete_all_analyses(token="t", user_id="u1"), 0)


class StoreGetTests(unittest.TestCase):
    def test_get_returns_row(self) -> None:
        client, _ = _fake_client(_Resp(data=[{"id": "a", "result": {}}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            row = store.get_analysis(token="t", analysis_id="a")
        self.assertEqual(row, {"id": "a", "result": {}})

    def test_get_returns_none_when_absent(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertIsNone(store.get_analysis(token="t", analysis_id="ghost"))


class StoreIsAdminTests(unittest.TestCase):
    def test_is_admin_true_when_role_row_present(self) -> None:
        client, query = _fake_client(_Resp(data=[{"user_id": "u1"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertTrue(store.is_admin(token="t", user_id="u1"))
        client.table.assert_called_with("user_roles")

    def test_is_admin_false_when_no_role_row(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertFalse(store.is_admin(token="t", user_id="u1"))


class GetAdminUserTests(unittest.TestCase):
    def test_passes_through_when_admin(self) -> None:
        me = CurrentUser(id="u1", token="tok")
        with mock.patch.object(store, "is_admin", return_value=True) as ia:
            result = auth.get_admin_user(user=me)
        self.assertIs(result, me)
        ia.assert_called_once_with(token="tok", user_id="u1")

    def test_raises_403_when_not_admin(self) -> None:
        me = CurrentUser(id="u1", token="tok")
        with mock.patch.object(store, "is_admin", return_value=False):
            with self.assertRaises(HTTPException) as ctx:
                auth.get_admin_user(user=me)
        self.assertEqual(ctx.exception.status_code, 403)


class AdminRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_status_reports_admin_true(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True) as ia:
            resp = self.client.get("/api/admin/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"is_admin": True})
        ia.assert_called_once_with(token="tok", user_id="u1")

    def test_status_reports_admin_false_for_non_admin(self) -> None:
        # A signed-in non-admin gets a truthful flag, NOT a 403 (status is only gated by auth).
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/status")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"is_admin": False})

    def test_status_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.get("/api/admin/status")
        self.assertEqual(resp.status_code, 401)


class StoreConversationTests(unittest.TestCase):
    def test_upsert_conversation_writes_the_thread(self) -> None:
        client, query = _fake_client(_Resp(data=[{"id": "c1"}]))
        msgs = [{"role": "user", "content": "why did my knees cave?"}]
        with mock.patch.object(store, "_user_client", return_value=client):
            store.upsert_conversation(
                token="t", user_id="u1", video_id="vid", messages=msgs, followups=["widen?"]
            )
        self.assertEqual(query.upserted["user_id"], "u1")
        self.assertEqual(query.upserted["video_id"], "vid")
        self.assertEqual(query.upserted["messages"], msgs)
        self.assertEqual(query.upserted["followups"], ["widen?"])
        client.table.assert_called_with("conversations")

    def test_upsert_conversation_defaults_followups_to_empty(self) -> None:
        # Omitting followups persists [] (a valid clear of the previous answer's chips).
        client, query = _fake_client(_Resp(data=[{"id": "c1"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.upsert_conversation(token="t", user_id="u1", video_id="vid", messages=[])
        self.assertEqual(query.upserted["followups"], [])

    def test_get_conversation_returns_messages_and_followups(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "drive knees out"},
        ]
        fups = ["Should I widen my stance?", "How low should I go?"]
        client, _ = _fake_client(_Resp(data=[{"messages": msgs, "followups": fups}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            out = store.get_conversation(token="t", video_id="vid")
        self.assertEqual(out, {"messages": msgs, "followups": fups})

    def test_get_conversation_none_when_no_thread(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertIsNone(store.get_conversation(token="t", video_id="ghost"))

    def test_get_conversation_defaults_null_fields_to_empty_lists(self) -> None:
        # A saved-but-empty thread (or a pre-followups row) reads back as empty lists, not None.
        client, _ = _fake_client(_Resp(data=[{"messages": None, "followups": None}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(
                store.get_conversation(token="t", video_id="vid"),
                {"messages": [], "followups": []},
            )


# ---------------------------------------------------------------- routers.analyses


class AnalysesRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_list_passes_params_and_returns_payload(self) -> None:
        with mock.patch.object(
            store, "list_analyses", return_value={"total": 1, "items": [{"id": "a"}]}
        ) as ls:
            resp = self.client.get("/api/analyses", params={"limit": 5, "offset": 2})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"total": 1, "items": [{"id": "a"}]})
        ls.assert_called_once_with(token="tok", limit=5, offset=2)

    def test_get_returns_analysis(self) -> None:
        with mock.patch.object(
            store, "get_analysis", return_value={"id": "a", "result": {"detections": []}}
        ) as ga:
            resp = self.client.get("/api/analyses/a")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], "a")
        ga.assert_called_once_with(token="tok", analysis_id="a")

    def test_get_missing_is_404(self) -> None:
        with mock.patch.object(store, "get_analysis", return_value=None):
            resp = self.client.get("/api/analyses/ghost")
        self.assertEqual(resp.status_code, 404)

    def test_delete_all_returns_count(self) -> None:
        with mock.patch.object(store, "delete_all_analyses", return_value=3) as da:
            resp = self.client.delete("/api/analyses")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"deleted": 3})
        da.assert_called_once_with(token="tok", user_id="u1")

    def test_delete_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.delete("/api/analyses")
        self.assertEqual(resp.status_code, 401)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.get("/api/analyses")
        self.assertEqual(resp.status_code, 401)


# ------------------------------------------------------------- routers.conversations


class ConversationsRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_put_saves_the_thread_with_followups(self) -> None:
        msgs = [
            {"role": "user", "content": "why did my knees cave?"},
            {"role": "assistant", "content": "drive knees out"},
        ]
        fups = ["Should I widen my stance?", "How low should I go?"]
        with mock.patch.object(store, "upsert_conversation") as up:
            resp = self.client.put(
                "/api/conversations/vid", json={"messages": msgs, "followups": fups}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"video_id": "vid", "messages": msgs, "followups": fups})
        up.assert_called_once_with(
            token="tok", user_id="u1", video_id="vid", messages=msgs, followups=fups
        )

    def test_put_defaults_followups_to_empty_when_omitted(self) -> None:
        # A PUT without `followups` is a valid clear, not a 422.
        msgs = [{"role": "user", "content": "hi"}]
        with mock.patch.object(store, "upsert_conversation") as up:
            resp = self.client.put("/api/conversations/vid", json={"messages": msgs})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["followups"], [])
        up.assert_called_once_with(
            token="tok", user_id="u1", video_id="vid", messages=msgs, followups=[]
        )

    def test_get_restores_the_thread(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        fups = ["Should I widen my stance?"]
        with mock.patch.object(
            store, "get_conversation", return_value={"messages": msgs, "followups": fups}
        ) as gc:
            resp = self.client.get("/api/conversations/vid")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"video_id": "vid", "messages": msgs, "followups": fups})
        gc.assert_called_once_with(token="tok", video_id="vid")

    def test_get_absent_thread_returns_empty(self) -> None:
        with mock.patch.object(store, "get_conversation", return_value=None):
            resp = self.client.get("/api/conversations/ghost")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"video_id": "ghost", "messages": [], "followups": []})

    def test_put_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.put("/api/conversations/vid", json={"messages": []})
        self.assertEqual(resp.status_code, 401)

    def test_get_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.get("/api/conversations/vid")
        self.assertEqual(resp.status_code, 401)


# ---------------------------------------------------------- services.runtime_config (P2 overrides)


class RuntimeConfigTests(unittest.TestCase):
    """The admin-override resolution layer: offline-safe, cached, patchable DB seam."""

    def setUp(self) -> None:
        runtime_config.clear_cache()
        self.addCleanup(runtime_config.clear_cache)

    def _configured(self):
        return types.SimpleNamespace(auth_configured=True)

    def test_empty_when_auth_not_configured(self) -> None:
        with mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace(auth_configured=False)
        ):
            self.assertEqual(runtime_config.get_overrides(), {})

    def test_empty_when_auth_settings_lacks_property(self) -> None:
        # A patched stand-in without ``auth_configured`` must not raise — it reads as "off".
        with mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace()
        ):
            self.assertEqual(runtime_config.get_overrides(), {})

    def test_empty_on_fetch_error(self) -> None:
        with mock.patch.object(app_settings, "get_settings", return_value=self._configured()), \
             mock.patch.object(runtime_config, "_fetch_rows", side_effect=RuntimeError("boom")):
            self.assertEqual(runtime_config.get_overrides(), {})

    def test_merges_rows_when_fetch_succeeds(self) -> None:
        rows = [{"key": "rag_top_k", "value": 9}, {"key": "llm_models", "value": ["a", "b"]}]
        with mock.patch.object(app_settings, "get_settings", return_value=self._configured()), \
             mock.patch.object(runtime_config, "_fetch_rows", return_value=rows):
            self.assertEqual(
                runtime_config.get_overrides(), {"rag_top_k": 9, "llm_models": ["a", "b"]}
            )

    def test_ttl_cache_hit_avoids_refetch(self) -> None:
        with mock.patch.object(app_settings, "get_settings", return_value=self._configured()), \
             mock.patch.object(
                 runtime_config, "_fetch_rows", return_value=[{"key": "kg_hops", "value": 2}]
             ) as fr:
            self.assertEqual(runtime_config.get_overrides(), {"kg_hops": 2})
            self.assertEqual(runtime_config.get_overrides(), {"kg_hops": 2})
            fr.assert_called_once()  # second read served from the TTL cache

    def test_clear_cache_forces_refetch(self) -> None:
        with mock.patch.object(app_settings, "get_settings", return_value=self._configured()), \
             mock.patch.object(
                 runtime_config, "_fetch_rows", return_value=[{"key": "kg_hops", "value": 2}]
             ) as fr:
            runtime_config.get_overrides()
            runtime_config.clear_cache()
            runtime_config.get_overrides()
            self.assertEqual(fr.call_count, 2)

    def test_fetch_rows_uses_anon_client_without_token(self) -> None:
        created: dict = {}

        def fake_create_client(url, key):
            created["args"] = (url, key)
            client = mock.Mock()
            client.table.return_value.select.return_value.execute.return_value = types.SimpleNamespace(
                data=[{"key": "rag_top_k", "value": 7}]
            )
            return client

        fake_supabase = types.ModuleType("supabase")
        fake_supabase.create_client = fake_create_client  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"supabase": fake_supabase}), mock.patch.object(
            app_settings,
            "get_settings",
            return_value=types.SimpleNamespace(supabase_url="u", supabase_anon_key="k"),
        ):
            rows = runtime_config._fetch_rows()
        self.assertEqual(created["args"], ("u", "k"))  # anon client, no postgrest.auth
        self.assertEqual(rows, [{"key": "rag_top_k", "value": 7}])


# ---------------------------------------------------------- settings getters (override-first)


class RuntimeSettingsGetterTests(unittest.TestCase):
    """Each getter returns the env/constant default with no override, and the override when present."""

    def setUp(self) -> None:
        runtime_config.clear_cache()
        self.addCleanup(runtime_config.clear_cache)

    def _no_overrides(self):
        return mock.patch.object(runtime_config, "get_overrides", return_value={})

    def _overrides(self, mapping):
        return mock.patch.object(runtime_config, "get_overrides", return_value=mapping)

    def test_chat_timeout(self) -> None:
        with self._no_overrides():
            self.assertEqual(app_settings.chat_timeout(), 60.0)
        with self._overrides({"chat_timeout": 90}):
            self.assertEqual(app_settings.chat_timeout(), 90.0)

    def test_followup_timeout(self) -> None:
        with self._no_overrides():
            self.assertEqual(app_settings.followup_timeout(), 15.0)
        with self._overrides({"followup_timeout": 8}):
            self.assertEqual(app_settings.followup_timeout(), 8.0)

    def test_rag_kg_defaults_and_overrides(self) -> None:
        with self._no_overrides():
            self.assertEqual(app_settings.rag_top_k_default(), 5)
            self.assertEqual(app_settings.kg_hops_default(), 1)
            self.assertEqual(app_settings.kg_seeds_default(), 5)
        with self._overrides({"rag_top_k": 9, "kg_hops": 2, "kg_seeds": 7}):
            self.assertEqual(app_settings.rag_top_k_default(), 9)
            self.assertEqual(app_settings.kg_hops_default(), 2)
            self.assertEqual(app_settings.kg_seeds_default(), 7)

    def test_chat_temperature_none_default_and_coerced(self) -> None:
        with self._no_overrides():
            self.assertIsNone(app_settings.chat_temperature())
        with self._overrides({"chat_temperature": "0.7"}):
            self.assertEqual(app_settings.chat_temperature(), 0.7)
        with self._overrides({"chat_temperature": "bad"}):
            self.assertIsNone(app_settings.chat_temperature())

    def test_coerce_helpers_fall_back_on_bad_values(self) -> None:
        with self._overrides({"chat_timeout": "x", "rag_top_k": "y"}):
            self.assertEqual(app_settings.chat_timeout(), 60.0)
            self.assertEqual(app_settings.rag_top_k_default(), 5)

    def test_getters_clamp_out_of_range_overrides(self) -> None:
        # F3: an out-of-band / direct-DB write can't drive an out-of-range value downstream — each
        # getter clamps to the same bound the PUT validator enforces (rather than rejecting).
        with self._overrides(
            {
                "rag_top_k": 999,
                "kg_hops": 99,
                "kg_seeds": 999,
                "chat_timeout": 5000,
                "followup_timeout": 5000,
                "chat_temperature": 9,
            }
        ):
            self.assertEqual(app_settings.rag_top_k_default(), 50)  # cap 50
            self.assertEqual(app_settings.kg_hops_default(), 3)  # cap 3
            self.assertEqual(app_settings.kg_seeds_default(), 20)  # cap 20
            self.assertEqual(app_settings.chat_timeout(), 300.0)  # cap 300
            self.assertEqual(app_settings.followup_timeout(), 300.0)  # cap 300
            self.assertEqual(app_settings.chat_temperature(), 2.0)  # cap 2
        with self._overrides(
            {
                "rag_top_k": 0,
                "kg_hops": 0,
                "kg_seeds": 0,
                "chat_timeout": -5,
                "followup_timeout": 0,
                "chat_temperature": -1,
            }
        ):
            self.assertEqual(app_settings.rag_top_k_default(), 1)  # floor 1
            self.assertEqual(app_settings.kg_hops_default(), 1)  # floor 1
            self.assertEqual(app_settings.kg_seeds_default(), 1)  # floor 1
            self.assertEqual(app_settings.chat_timeout(), 1.0)  # positive floor
            self.assertEqual(app_settings.followup_timeout(), 1.0)  # positive floor
            self.assertEqual(app_settings.chat_temperature(), 0.0)  # floor 0

    def test_chat_base_url(self) -> None:
        env = types.SimpleNamespace(
            llm_base_url="https://env.example.com/v1", llm_allowed_base_hosts=""
        )
        with self._no_overrides(), mock.patch.object(
            app_settings, "get_settings", return_value=env
        ):
            self.assertEqual(app_settings.chat_base_url(), "https://env.example.com/v1")
        # An allowlisted-host override (the env-default host is always allowed) is honoured.
        with self._overrides({"llm_base_url": "https://env.example.com/api/alt"}), mock.patch.object(
            app_settings, "get_settings", return_value=env
        ):
            self.assertEqual(app_settings.chat_base_url(), "https://env.example.com/api/alt")
        # An off-allowlist override falls back to the env default (the read-time SSRF/key-leak guard).
        with self._overrides({"llm_base_url": "https://evil.example.org/v1"}), mock.patch.object(
            app_settings, "get_settings", return_value=env
        ):
            self.assertEqual(app_settings.chat_base_url(), "https://env.example.com/v1")

    def test_base_url_allowed_accepts_builtin_and_env_hosts(self) -> None:
        env = types.SimpleNamespace(
            llm_base_url="https://env.example.com/v1", llm_allowed_base_hosts="extra.host.io"
        )
        with mock.patch.object(app_settings, "get_settings", return_value=env):
            # Built-in provider hosts.
            self.assertTrue(app_settings._base_url_allowed("https://openrouter.ai/api/v1"))
            self.assertTrue(app_settings._base_url_allowed("https://api.openai.com/v1"))
            self.assertTrue(app_settings._base_url_allowed("https://integrate.api.nvidia.com/v1"))
            # Env-default host + an LLM_ALLOWED_BASE_HOSTS entry (case-insensitive).
            self.assertTrue(app_settings._base_url_allowed("https://ENV.example.com/x"))
            self.assertTrue(app_settings._base_url_allowed("http://extra.host.io/y"))

    def test_base_url_allowed_rejects_offlist_and_bad_scheme(self) -> None:
        env = types.SimpleNamespace(
            llm_base_url="https://openrouter.ai/api/v1", llm_allowed_base_hosts=""
        )
        with mock.patch.object(app_settings, "get_settings", return_value=env):
            self.assertFalse(app_settings._base_url_allowed("https://evil.example.org/v1"))
            self.assertFalse(app_settings._base_url_allowed("http://127.0.0.1:8080/v1"))
            self.assertFalse(app_settings._base_url_allowed("http://localhost/v1"))
            self.assertFalse(app_settings._base_url_allowed("http://169.254.169.254/latest"))
            self.assertFalse(app_settings._base_url_allowed("ftp://openrouter.ai/v1"))
            self.assertFalse(app_settings._base_url_allowed("not-a-url"))
            self.assertFalse(app_settings._base_url_allowed(None))

    def test_chat_models_override_string_list_and_env_fallback(self) -> None:
        with self._overrides({"llm_models": "a,b,a"}):
            self.assertEqual(app_settings.chat_models(), ["a", "b"])  # deduped, ordered
        with self._overrides({"llm_models": ["x", "y"]}):
            self.assertEqual(app_settings.chat_models(), ["x", "y"])
        with self._no_overrides(), mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace(llm_models="e1,e2")
        ):
            self.assertEqual(app_settings.chat_models(), ["e1", "e2"])

    def test_followup_model_override(self) -> None:
        with self._overrides({"llm_followup_model": "fast/model"}):
            self.assertEqual(app_settings.followup_chat_model(), "fast/model")

    def test_allowed_upload_suffixes_default_override_and_malformed(self) -> None:
        default = (".mp4", ".mov", ".avi", ".mkv", ".webm")
        with self._no_overrides():
            self.assertEqual(app_settings.allowed_upload_suffixes(), default)
        with self._overrides({"allowed_upload_suffixes": [".mp4", ".GIF"]}):
            self.assertEqual(app_settings.allowed_upload_suffixes(), (".mp4", ".gif"))
        # An entry without a leading dot is dropped; an all-malformed list falls back to the default.
        with self._overrides({"allowed_upload_suffixes": ["mp4"]}):
            self.assertEqual(app_settings.allowed_upload_suffixes(), default)


# ---------------------------------------------------- chat body: temperature wiring (override)


class ChatTemperatureBodyTests(unittest.TestCase):
    """The completion body omits ``temperature`` by default and includes it when overridden."""

    def _request_json(self, temperature):
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(["data: [DONE]"])
        cm = mock.MagicMock()
        cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        with mock.patch.object(
            chat_service,
            "get_settings",
            return_value=types.SimpleNamespace(
                llm_api_key="sk", llm_base_url="https://openrouter.ai/api/v1"
            ),
        ), mock.patch.object(
            chat_service, "chat_base_url", return_value="https://openrouter.ai/api/v1"
        ), mock.patch.object(
            chat_service, "chat_temperature", return_value=temperature
        ), mock.patch.object(
            chat_service, "chat_timeout", return_value=60.0
        ), mock.patch("httpx.stream", return_value=cm) as stream:
            list(chat_service._stream_completion([{"role": "user", "content": "hi"}], "m"))
        _, kwargs = stream.call_args
        return kwargs["json"]

    def test_temperature_omitted_when_none(self) -> None:
        self.assertNotIn("temperature", self._request_json(None))

    def test_temperature_included_when_set(self) -> None:
        self.assertEqual(self._request_json(0.5)["temperature"], 0.5)


# ------------------------------------------------------ store: app_settings read/write seams


class StoreAppSettingsTests(unittest.TestCase):
    def test_get_app_settings_returns_key_value_map(self) -> None:
        client, _ = _fake_client(_Resp(data=[{"key": "rag_top_k", "value": 9}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            out = store.get_app_settings(token="t")
        self.assertEqual(out, {"rag_top_k": 9})
        client.table.assert_called_with("app_settings")

    def test_upsert_app_settings_writes_rows(self) -> None:
        client, query = _fake_client(_Resp(data=[{"key": "rag_top_k"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.upsert_app_settings(token="t", items={"rag_top_k": 9, "kg_hops": 2})
        self.assertEqual(
            query.upserted, [{"key": "rag_top_k", "value": 9}, {"key": "kg_hops", "value": 2}]
        )
        client.table.assert_called_with("app_settings")

    def test_upsert_app_settings_noop_on_empty(self) -> None:
        client, _ = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client) as uc:
            store.upsert_app_settings(token="t", items={})
        uc.assert_not_called()  # nothing to write -> no client built


# ------------------------------------------------ routers.admin: GET/PUT /api/admin/settings


class AdminSettingsRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        runtime_config.clear_cache()
        self.addCleanup(runtime_config.clear_cache)

    def test_get_settings_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/settings")
        self.assertEqual(resp.status_code, 403)

    def test_get_settings_returns_effective_and_defaults(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.get("/api/admin/settings")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("effective", body)
        self.assertIn("defaults", body)
        for group in ("llm", "rag_kg", "analyze"):
            self.assertIn(group, body["effective"])
            self.assertIn(group, body["defaults"])
        self.assertEqual(body["defaults"]["rag_kg"]["rag_top_k"], 5)
        # No secret is ever exposed in the payload.
        self.assertNotIn("llm_api_key", json.dumps(body))
        self.assertNotIn("supabase", json.dumps(body).lower())

    def test_put_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.put("/api/admin/settings", json={"rag_top_k": 8})
        self.assertEqual(resp.status_code, 403)

    def test_put_rejects_out_of_range_values(self) -> None:
        bad_payloads = [
            {"chat_temperature": 2.1},
            {"chat_timeout": 0},
            {"chat_timeout": 301},
            {"followup_timeout": 0},
            {"rag_top_k": 0},
            {"rag_top_k": 51},
            {"kg_hops": 4},
            {"kg_seeds": 21},
            {"llm_models": []},
            {"allowed_upload_suffixes": ["mp4"]},  # missing leading dot
            {"llm_base_url": "ftp://nope"},
            {"llm_base_url": "https://evil.example.org/v1"},  # off-allowlist host (F1)
        ]
        with mock.patch.object(store, "is_admin", return_value=True):
            for payload in bad_payloads:
                resp = self.client.put("/api/admin/settings", json=payload)
                self.assertEqual(resp.status_code, 422, msg=f"expected 422 for {payload}")

    def test_put_upserts_only_provided_keys_and_clears_cache(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "upsert_app_settings") as up, \
             mock.patch.object(runtime_config, "clear_cache") as cc, \
             mock.patch.object(runtime_config, "get_overrides", return_value={"rag_top_k": 8}):
            resp = self.client.put("/api/admin/settings", json={"rag_top_k": 8, "kg_hops": 2})
        self.assertEqual(resp.status_code, 200)
        up.assert_called_once_with(token="tok", items={"rag_top_k": 8, "kg_hops": 2})
        cc.assert_called_once()
        # The response reflects the (patched) new effective value.
        self.assertEqual(resp.json()["effective"]["rag_kg"]["rag_top_k"], 8)

    def test_put_accepts_and_normalizes_list_and_url_knobs(self) -> None:
        # Exercises the field validators' happy paths: models are trimmed, a suffix is lower-cased,
        # and a valid https base URL passes through.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "upsert_app_settings") as up, \
             mock.patch.object(runtime_config, "clear_cache"), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.put(
                "/api/admin/settings",
                json={
                    "llm_models": [" a/m ", "b/m"],
                    "llm_base_url": "https://openrouter.ai/api/alt",  # built-in allowlisted host
                    "allowed_upload_suffixes": [".MP4", ".gif"],
                },
            )
        self.assertEqual(resp.status_code, 200)
        up.assert_called_once_with(
            token="tok",
            items={
                "llm_models": ["a/m", "b/m"],
                "llm_base_url": "https://openrouter.ai/api/alt",
                "allowed_upload_suffixes": [".mp4", ".gif"],
            },
        )

    def test_put_rejects_empty_lists_via_validators(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True):
            for payload in ({"llm_models": []}, {"allowed_upload_suffixes": []}):
                resp = self.client.put("/api/admin/settings", json=payload)
                self.assertEqual(resp.status_code, 422, msg=f"expected 422 for {payload}")

    def test_put_tolerates_explicit_null_knobs(self) -> None:
        # An explicit null exercises the validators' ``v is None`` short-circuit and is simply ignored.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "upsert_app_settings") as up, \
             mock.patch.object(runtime_config, "clear_cache"), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.put(
                "/api/admin/settings",
                json={"llm_models": None, "llm_base_url": None, "allowed_upload_suffixes": None},
            )
        self.assertEqual(resp.status_code, 200)
        up.assert_called_once_with(
            token="tok",
            items={"llm_models": None, "llm_base_url": None, "allowed_upload_suffixes": None},
        )

    def test_put_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.put("/api/admin/settings", json={"rag_top_k": 8})
        self.assertEqual(resp.status_code, 401)

    def test_put_ignores_max_concurrent_analyses(self) -> None:
        # F5: max_concurrent_analyses is no longer a PUT field — it is silently ignored (never written),
        # not accepted as an override, and never rejected. Only the real knob (rag_top_k) is upserted.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "upsert_app_settings") as up, \
             mock.patch.object(runtime_config, "clear_cache"), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.put(
                "/api/admin/settings", json={"max_concurrent_analyses": 99, "rag_top_k": 8}
            )
        self.assertEqual(resp.status_code, 200)
        up.assert_called_once_with(token="tok", items={"rag_top_k": 8})  # no max_concurrent key

    def test_get_effective_includes_readonly_max_concurrent(self) -> None:
        # F5: the value stays present (read-only, env-sourced) under analyze in both effective+defaults.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.get("/api/admin/settings")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(
            body["effective"]["analyze"]["max_concurrent_analyses"], config.MAX_CONCURRENT_ANALYSES
        )
        self.assertEqual(
            body["defaults"]["analyze"]["max_concurrent_analyses"], config.MAX_CONCURRENT_ANALYSES
        )


# ------------------------------------------ store: admin_list_users / set_user_role (P3 seams)


class StoreAdminUsersTests(unittest.TestCase):
    def test_admin_list_users_calls_rpc_and_returns_data(self) -> None:
        rows = [{"id": "u1", "email": "a@x.com", "analyses_count": 3}]
        client, _ = _fake_client(_Resp(data=rows))
        with mock.patch.object(store, "_user_client", return_value=client):
            out = store.admin_list_users(token="t")
        self.assertEqual(out, rows)
        client.rpc.assert_called_once_with("admin_list_users")

    def test_admin_list_users_empty_when_no_data(self) -> None:
        client, _ = _fake_client(_Resp(data=None))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(store.admin_list_users(token="t"), [])

    def test_set_user_role_upserts_when_make_admin(self) -> None:
        client, query = _fake_client(_Resp(data=[{"user_id": "u2"}]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.set_user_role(token="t", user_id="u2", make_admin=True)
        self.assertEqual(query.upserted, {"user_id": "u2", "role": "admin"})
        self.assertFalse(query.deleted)
        client.table.assert_called_with("user_roles")

    def test_set_user_role_deletes_when_not_make_admin(self) -> None:
        client, query = _fake_client(_Resp(data=[]))
        with mock.patch.object(store, "_user_client", return_value=client):
            store.set_user_role(token="t", user_id="u2", make_admin=False)
        self.assertTrue(query.deleted)
        self.assertIsNone(query.upserted)

    def test_count_admins_returns_count(self) -> None:
        # Goes through the count_admins() SECURITY DEFINER RPC (not a table read), so the self-only
        # user_roles SELECT policy can't shrink the count to 1 for the acting admin.
        client, _ = _fake_client(_Resp(data=3))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(store.count_admins(token="t"), 3)
        client.rpc.assert_called_with("count_admins")

    def test_count_admins_zero_when_data_none(self) -> None:
        client, _ = _fake_client(_Resp(data=None))
        with mock.patch.object(store, "_user_client", return_value=client):
            self.assertEqual(store.count_admins(token="t"), 0)
        client.rpc.assert_called_with("count_admins")


# ------------------------------ routers.admin: /users, /users/{id}/role, /overview (P3 dashboard)


class AdminUsersRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        runtime_config.clear_cache()
        self.addCleanup(runtime_config.clear_cache)

    # -- GET /users -------------------------------------------------------------------------------
    def test_list_users_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/users")
        self.assertEqual(resp.status_code, 403)

    def test_list_users_returns_rows_for_admin(self) -> None:
        rows = [{"id": "u1", "email": "a@x.com", "analyses_count": 2, "is_admin": True}]
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "admin_list_users", return_value=rows) as lu:
            resp = self.client.get("/api/admin/users")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"users": rows})
        lu.assert_called_once_with(token="tok")

    # -- PUT /users/{id}/role ---------------------------------------------------------------------
    def test_set_role_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.put("/api/admin/users/u2/role", json={"make_admin": True})
        self.assertEqual(resp.status_code, 403)

    def test_set_role_rejects_self_demote(self) -> None:
        # An admin (id "u1") may not revoke their OWN admin role — anti-lockout guard returns 400.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "set_user_role") as sr:
            resp = self.client.put("/api/admin/users/u1/role", json={"make_admin": False})
        self.assertEqual(resp.status_code, 400)
        sr.assert_not_called()

    def test_set_role_allows_self_promote(self) -> None:
        # Re-granting one's own role is harmless (idempotent), so it is NOT blocked by the guard.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "set_user_role") as sr:
            resp = self.client.put("/api/admin/users/u1/role", json={"make_admin": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        sr.assert_called_once_with(token="tok", user_id="u1", make_admin=True)

    def test_set_role_success_for_other_user(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "set_user_role") as sr:
            resp = self.client.put("/api/admin/users/u2/role", json={"make_admin": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        sr.assert_called_once_with(token="tok", user_id="u2", make_admin=True)

    def test_set_role_rejects_removing_last_admin(self) -> None:
        # F4: revoking another admin's role when only one admin remains is refused (400, anti-lockout).
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "count_admins", return_value=1), \
             mock.patch.object(store, "set_user_role") as sr:
            resp = self.client.put("/api/admin/users/u2/role", json={"make_admin": False})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("last admin", resp.json()["detail"].lower())
        sr.assert_not_called()

    def test_set_role_allows_removal_when_multiple_admins(self) -> None:
        # F4: with more than one admin, revoking another admin's role succeeds.
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "count_admins", return_value=2), \
             mock.patch.object(store, "set_user_role") as sr:
            resp = self.client.put("/api/admin/users/u2/role", json={"make_admin": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        sr.assert_called_once_with(token="tok", user_id="u2", make_admin=False)

    # -- GET /overview ----------------------------------------------------------------------------
    def test_overview_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/overview")
        self.assertEqual(resp.status_code, 403)

    def test_overview_returns_health_flags_and_totals(self) -> None:
        users = [
            {"id": "u1", "analyses_count": 3},
            {"id": "u2", "analyses_count": 5},
            {"id": "u3", "analyses_count": None},  # null count coerces to 0
        ]
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch.object(store, "admin_list_users", return_value=users), \
             mock.patch.object(runtime_config, "get_overrides", return_value={}):
            resp = self.client.get("/api/admin/overview")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("auth_configured", "chat_configured", "chat_models", "chat_default", "stores"):
            self.assertIn(key, body)
        self.assertEqual(body["total_users"], 3)
        self.assertEqual(body["total_analyses"], 8)
        # No secret ever leaks into the dashboard payload.
        self.assertNotIn("llm_api_key", json.dumps(body))
        self.assertNotIn("supabase", json.dumps(body).lower())

    def test_users_endpoints_require_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        self.assertEqual(self.client.get("/api/admin/users").status_code, 401)
        self.assertEqual(self.client.get("/api/admin/overview").status_code, 401)
        self.assertEqual(
            self.client.put("/api/admin/users/u2/role", json={"make_admin": True}).status_code, 401
        )


if __name__ == "__main__":
    unittest.main()
