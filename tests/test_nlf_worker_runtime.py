"""Tests for the NLF worker's archive patcher, CLI helpers and runtime adapters.

The archive patcher, the CLI helpers (src/nlf_worker/cli.py) and the R2/Supabase adapters run
everywhere, including CI: the adapters import boto3/supabase lazily and take an injected fake
client. Anything needing the heavy GPU stack (torch, PyAV, OpenCV, pyrender) is guarded by
``pytest.importorskip`` inside the test itself, so a missing dependency skips that test only.
"""
from __future__ import annotations

import io
import math
import os
import pickle
import tempfile
import types
import unittest
import unittest.mock
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.nlf_worker import archive, cli, geometry, loop, result
from src.nlf_worker.runtime import r2_store, supabase_db

# ---------------------------------------------------------------------------------------
# archive.patch_to_fp32
# ---------------------------------------------------------------------------------------
_TORCH_PY = b"def forward(self):\n    x = predict_multi_same_weights(torch.to(crops_flat, 5), w)\n"
_DETECTOR_PY = b"    preds = (model).forward(torch.to(images, 5), )\n    return preds\n"
_WEIGHTS = bytes(range(256)) * 64
# Built at runtime so secret scanners (GitGuardian) don't read the fake sign-in as a leaked password.
_FAKE_CREDENTIAL = "-".join(("test", "only", "value"))


def _write_archive(path, torch_py=_TORCH_PY, detector_py=_DETECTOR_PY, include_detector=True):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("m/data/0"), _WEIGHTS)  # ZIP_STORED
        z.writestr("m/code/__torch__.py", torch_py, compress_type=zipfile.ZIP_DEFLATED)
        if include_detector:
            z.writestr(
                "m/code/__torch__/nlf/pt/multiperson/person_detector.py",
                detector_py,
                compress_type=zipfile.ZIP_DEFLATED,
            )


class PatchToFp32Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "stock.torchscript")
        self.dst = os.path.join(self.tmp.name, "fp32.torchscript")

    def _assert_no_output(self):
        self.assertFalse(os.path.exists(self.dst))
        self.assertFalse(os.path.exists(self.dst + ".partial"))

    def test_patches_both_casts_and_copies_everything_else(self):
        _write_archive(self.src)
        archive.patch_to_fp32(self.src, self.dst)
        with zipfile.ZipFile(self.dst) as z:
            torch_py = z.read("m/code/__torch__.py")
            detector_py = z.read("m/code/__torch__/nlf/pt/multiperson/person_detector.py")
            self.assertIn(b"torch.to(crops_flat, 6)", torch_py)
            self.assertNotIn(b"torch.to(crops_flat, 5)", torch_py)
            self.assertIn(b"torch.to(images, 6), )", detector_py)
            self.assertEqual(z.read("m/data/0"), _WEIGHTS)
            # Each entry keeps its own compression type.
            self.assertEqual(z.getinfo("m/data/0").compress_type, zipfile.ZIP_STORED)
            self.assertEqual(z.getinfo("m/code/__torch__.py").compress_type, zipfile.ZIP_DEFLATED)
        self.assertFalse(os.path.exists(self.dst + ".partial"))

    def test_missing_needle_raises_and_writes_nothing(self):
        _write_archive(self.src, detector_py=b"preds = (model).forward(torch.to(images, 6), )\n")
        with self.assertRaises(ValueError) as ctx:
            archive.patch_to_fp32(self.src, self.dst)
        self.assertIn("person_detector.py", str(ctx.exception))
        self.assertIn("found 0 time(s)", str(ctx.exception))
        self._assert_no_output()

    def test_needle_found_twice_raises(self):
        _write_archive(self.src, torch_py=_TORCH_PY + _TORCH_PY)
        with self.assertRaises(ValueError) as ctx:
            archive.patch_to_fp32(self.src, self.dst)
        self.assertIn("found 2 time(s)", str(ctx.exception))
        self._assert_no_output()

    def test_missing_entry_raises(self):
        _write_archive(self.src, include_detector=False)
        with self.assertRaises(ValueError):
            archive.patch_to_fp32(self.src, self.dst)
        self._assert_no_output()

    def test_failure_leaves_an_existing_destination_untouched(self):
        _write_archive(self.src, include_detector=False)
        with open(self.dst, "wb") as fh:
            fh.write(b"previous")
        with self.assertRaises(ValueError):
            archive.patch_to_fp32(self.src, self.dst)
        with open(self.dst, "rb") as fh:
            self.assertEqual(fh.read(), b"previous")


# ---------------------------------------------------------------------------------------
# geometry.turntable_framing (the renderer's pure framing math)
# ---------------------------------------------------------------------------------------
class TurntableFramingTests(unittest.TestCase):
    def test_tall_narrow_body_is_framed_by_height(self):
        v = np.array([[0.1, -0.9, 0.0], [-0.1, 0.8, 0.05]])
        f = geometry.turntable_framing([v])
        self.assertAlmostEqual(f["cy"], -0.05)
        self.assertAlmostEqual(f["mag"], 0.85 * 1.12)
        self.assertEqual(set(f), {"cy", "mag"})

    def test_wide_reach_widens_the_view_so_no_angle_clips(self):
        # Hands 0.9 m in front of the pelvis on a 1.0 m tall crouch.
        v = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.2, 0.9]])
        self.assertAlmostEqual(geometry.turntable_framing([v])["mag"], 0.9 * 1.05)

    def test_shared_framing_covers_the_union_of_all_key_frames(self):
        standing = np.array([[0.0, -0.95, 0.0], [0.0, 0.75, 0.0]])
        bottom_reaching = np.array([[0.0, -0.45, 0.0], [0.0, 0.7, 0.0], [0.0, 0.3, 0.85]])
        shared = geometry.turntable_framing([standing, bottom_reaching])
        self.assertAlmostEqual(shared["cy"], (-0.95 + 0.75) / 2)
        self.assertAlmostEqual(shared["mag"], max(0.85 * 1.12, 0.85 * 1.05))
        # Order-independent, and at least as large as either key frame framed alone.
        self.assertEqual(shared, geometry.turntable_framing([bottom_reaching, standing]))
        for body in (standing, bottom_reaching):
            self.assertGreaterEqual(shared["mag"], geometry.turntable_framing([body])["mag"])

    def test_no_meshes_raises(self):
        with self.assertRaises(ValueError):
            geometry.turntable_framing([])

    def test_guide_lines_are_joint_centre_heights(self):
        j = np.zeros((24, 3))
        j[geometry.L_HIP, 1], j[geometry.R_HIP, 1] = -0.04, -0.06
        j[geometry.L_KNEE, 1], j[geometry.R_KNEE, 1] = -0.4, -0.5
        lines = geometry.guide_line_heights(j)
        self.assertAlmostEqual(lines["hip_y"], -0.05)
        self.assertAlmostEqual(lines["knee_y"], -0.45)


# ---------------------------------------------------------------------------------------
# cli: arguments, dry-run job and stand-in adapters
# ---------------------------------------------------------------------------------------
class SegmentConversionTests(unittest.TestCase):
    def test_seconds_to_frames_at_30fps(self):
        self.assertEqual(
            cli.segments_to_rep_segments([(3.6, 25.2), [26.0, 27.5]]),
            [
                {"index": 0, "start_frame": 108, "end_frame": 756},
                {"index": 1, "start_frame": 780, "end_frame": 825},
            ],
        )

    def test_empty(self):
        self.assertEqual(cli.segments_to_rep_segments([]), [])

    def test_rejects_bad_segments(self):
        for bad in ([(5.0, 5.0)], [(6.0, 5.0)], [(-1.0, 2.0)], [(0.0, math.nan)]):
            with self.assertRaises(ValueError):
                cli.segments_to_rep_segments(bad)

    def test_dry_run_job_shape(self):
        job = cli.dry_run_job("C:/clips/my_clip.webm", "squat", [(3.6, 25.2)])
        self.assertEqual(job["storage_key"], "dryrun/my_clip")
        self.assertEqual(job["attempts"], 1)
        self.assertEqual(job["pose_fps"], 30.0)
        self.assertEqual(job["movement"], "squat")
        self.assertEqual(job["detections"], [])
        self.assertEqual(job["rep_segments"], [{"index": 0, "start_frame": 108, "end_frame": 756}])
        self.assertEqual(result.result_prefix(job["storage_key"], job["attempts"]), "dryrun/my_clip/nlf.v1.a1")


class ParseArgsTests(unittest.TestCase):
    def _error(self, argv):
        with self.assertRaises(SystemExit), unittest.mock.patch("sys.stderr", io.StringIO()):
            cli.parse_args(argv)

    def test_dry_run_with_segments(self):
        args = cli.parse_args(
            ["--dry-run", "clip.webm", "--out", "o", "--segment", "3.6", "25.2", "--segment", "26", "27"]
        )
        self.assertEqual(args.dry_run, "clip.webm")
        self.assertEqual(args.segment, [[3.6, 25.2], [26.0, 27.0]])
        self.assertEqual(args.movement, "squat")

    def test_defaults_to_the_loop(self):
        args = cli.parse_args([])
        self.assertFalse(args.once)
        self.assertIsNone(args.dry_run)
        self.assertEqual(args.env, "data/models/nlf/worker.env")

    def test_dry_run_needs_out(self):
        self._error(["--dry-run", "clip.webm"])

    def test_dry_run_rejects_a_bad_segment(self):
        self._error(["--dry-run", "clip.webm", "--out", "o", "--segment", "5", "4"])

    def test_out_and_segment_only_with_dry_run(self):
        self._error(["--out", "o"])
        self._error(["--once", "--segment", "1", "2"])


class DryRunDbTests(unittest.TestCase):
    def test_hands_out_the_job_once(self):
        db = cli.DryRunDb({"job_id": "j"})
        db.heartbeat()
        self.assertEqual(db.claim(), {"job_id": "j"})
        self.assertIsNone(db.claim())
        self.assertTrue(db.complete("j", "k", {"a": 1}))
        db.fail("j", "boom")
        self.assertEqual(db.heartbeats, 1)
        self.assertEqual(db.completed, [("j", "k", {"a": 1})])
        self.assertEqual(db.failed, [("j", "boom")])


class LocalDirStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = cli.LocalDirStore(self.tmp.name)

    def test_put_get_and_content_type(self):
        self.store.put("a/nlf.v1.a1/pose3d.json.gz", b"xyz", content_type="application/json")
        self.assertEqual(self.store.get("a/nlf.v1.a1/pose3d.json.gz"), b"xyz")
        self.assertTrue((Path(self.tmp.name) / "a" / "nlf.v1.a1" / "pose3d.json.gz").is_file())
        self.assertEqual(self.store.content_types["a/nlf.v1.a1/pose3d.json.gz"], "application/json")

    def test_delete_prefix_is_segment_exact(self):
        self.store.put("a/nlf.v1.a1/x.webp", b"1", content_type="image/webp")
        self.store.put("a/nlf.v1.a10/x.webp", b"2", content_type="image/webp")
        self.store.delete_prefix("a/nlf.v1.a1")
        self.assertFalse((Path(self.tmp.name) / "a" / "nlf.v1.a1").exists())
        self.assertEqual(self.store.get("a/nlf.v1.a10/x.webp"), b"2")
        self.assertEqual(list(self.store.content_types), ["a/nlf.v1.a10/x.webp"])

    def test_delete_missing_prefix_is_a_no_op(self):
        self.store.delete_prefix("nothing/here")

    def test_rejects_keys_that_escape_the_root(self):
        for key in ("../x", "a/../../x", "/abs", "C:/Windows/x", "a\\b", ""):
            with self.assertRaises(ValueError):
                self.store.put(key, b"", content_type="x")


class CopyFetcherTests(unittest.TestCase):
    def test_returns_a_copy_never_the_original(self):
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "clip.webm")
            with open(clip, "wb") as fh:
                fh.write(b"video")
            fetcher = cli.CopyFetcher(clip)
            path = fetcher.download("dryrun/clip/source", os.path.join(d, "tmp"))
            self.assertNotEqual(os.path.abspath(path), os.path.abspath(clip))
            self.assertTrue(path.endswith(".webm"))
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), b"video")
            os.remove(path)
            self.assertTrue(os.path.exists(clip))
            self.assertEqual(fetcher.keys, ["dryrun/clip/source"])


def _squat_joints(theta_deg):
    """(24, 3) camera-mm frame (y down): knee angle = 180 - theta_deg, shoulders above."""
    j = np.full((24, 3), np.nan)
    pelvis = np.array([0.0, 0.0, 2000.0])
    j[geometry.PELVIS] = pelvis
    j[geometry.NECK] = pelvis + [0.0, -600.0, 0.0]
    j[geometry.L_SHOULDER] = pelvis + [-150.0, -800.0, 0.0]
    j[geometry.R_SHOULDER] = pelvis + [150.0, -800.0, 0.0]
    shin = 400.0 * np.array([math.sin(math.radians(theta_deg)), math.cos(math.radians(theta_deg)), 0.0])
    for hip, knee, ankle, x in (
        (geometry.L_HIP, geometry.L_KNEE, geometry.L_ANKLE, -100.0),
        (geometry.R_HIP, geometry.R_KNEE, geometry.R_ANKLE, 100.0),
    ):
        j[hip] = pelvis + [x, 0.0, 0.0]
        j[knee] = pelvis + [x, 400.0, 0.0]
        j[ankle] = j[knee] + shin
    return j


class _Decoder:
    def __init__(self, n):
        self.n = n

    def iterate(self, path, sample_fps):
        assert os.path.exists(path)
        return [(i / sample_fps, np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(self.n)]


class _Model:
    def __init__(self, thetas):
        self.thetas = list(thetas)
        self.i = 0

    def infer(self, frames):
        out = []
        for _ in frames:
            theta = self.thetas[self.i]
            self.i += 1
            if theta is None:
                out.append({"joints3d": None, "joint_uncertainties": None, "vertices3d": None})
            else:
                out.append(
                    {
                        "joints3d": _squat_joints(theta),
                        "joint_uncertainties": np.full(24, 20.0),
                        "vertices3d": np.zeros((10, 3)),
                    }
                )
        return out


class _Renderer:
    def __init__(self):
        self.framings = []

    def render_strip(self, vertices_mm, joints_mm, basis, framing=None):
        self.framings.append(framing)
        return b"RIFF-fake-webp"


class DryRunEndToEndTests(unittest.TestCase):
    """``loop.run_one`` over the dry-run stand-ins, with fake compute."""

    def test_writes_results_locally_and_keeps_the_original_clip(self):
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "phone.webm")
            with open(clip, "wb") as fh:
                fh.write(b"video")
            out = os.path.join(d, "out")
            tmp = os.path.join(d, "tmp")
            job = cli.dry_run_job(clip, "squat", [(0.2, 0.8)])
            db, store = cli.DryRunDb(job), cli.LocalDirStore(out)
            # Standing, a set-up crouch outside the segment (t=0.1), then a rep bottoming at t=0.5.
            thetas = [5, 120, 5, 30, 60, 90, 60, 30, 5, None]
            decoder, model = cli.TimedDecoder(_Decoder(len(thetas))), cli.TimedModel(_Model(thetas))
            renderer = _Renderer()
            outcome = loop.run_one(
                db, store, cli.CopyFetcher(clip), decoder, model, cli.TimedRenderer(renderer),
                tmp_dir=tmp, batch=4, model_sha256="sha", torch_version="t",
            )
            self.assertEqual(outcome, "done", db.failed)
            self.assertTrue(os.path.exists(clip), "the dry-run must never delete the input clip")
            self.assertEqual(os.listdir(tmp), [], "the fetched copy is removed after the job")
            self.assertEqual((decoder.frames, model.frames, model.detected), (10, 10, 9))

            job_id, pose3d_key, meta = db.completed[0]
            self.assertEqual(pose3d_key, "dryrun/phone/nlf.v1.a1/pose3d.json.gz")
            self.assertEqual(len(meta["key_frames"]), 1)
            kf = meta["key_frames"][0]
            self.assertEqual((kf["kind"], kf["rep_index"]), ("rep_bottom", 0))
            self.assertAlmostEqual(kf["t_s"], 0.5)
            self.assertEqual(store.get(kf["strip_key"]), b"RIFF-fake-webp")
            self.assertEqual(store.content_types[kf["strip_key"]], "image/webp")
            self.assertEqual(len(renderer.framings), 1)
            self.assertEqual(set(renderer.framings[0]), {"cy", "mag"})  # passed through TimedRenderer

            pose3d = result.decode_pose3d(store.get(pose3d_key))
            rows = cli.describe_key_frames(pose3d, meta)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["knee_deg"], 90.0, places=4)
            # Hip 400 mm above the knee; the up axis estimated from the (5-degree shin) standing
            # frames tilts the body frame very slightly, hence the tolerance.
            self.assertAlmostEqual(rows[0]["hip_minus_knee_cm"], 40.0, delta=0.05)


class DescribeKeyFramesTests(unittest.TestCase):
    def test_hip_below_knee_is_negative_in_the_body_frame(self):
        j = _squat_joints(5)
        # Camera y points down: move both hips (and the pelvis) 20 mm BELOW the knees.
        for idx in (geometry.PELVIS, geometry.L_HIP, geometry.R_HIP):
            j[idx, 1] = j[geometry.L_KNEE, 1] + 20.0
        pose3d = result.build_pose3d(
            model_sha256="s", torch_version="t", sample_fps=10.0, sample_times_s=[0.0, 0.1],
            joints3d=[None, j], joint_uncertainties=[None, np.zeros(24)],
            up_axis=np.array([0.0, -1.0, 0.0]),
        )
        pose3d = result.decode_pose3d(result.encode_pose3d(pose3d))  # through JSON, as uploaded
        meta = {"key_frames": [{"t_s": 0.1, "kind": "rep_bottom", "rep_index": 2}]}
        (row,) = cli.describe_key_frames(pose3d, meta)
        self.assertEqual((row["index"], row["kind"], row["rep_index"]), (0, "rep_bottom", 2))
        self.assertAlmostEqual(row["hip_minus_knee_cm"], -2.0, places=6)
        self.assertAlmostEqual(row["knee_deg"], float(geometry.knee_angle(j)), places=6)


# ---------------------------------------------------------------------------------------
# cli: the poll loop and wrappers
# ---------------------------------------------------------------------------------------
class TimedDecoderTests(unittest.TestCase):
    def test_streams_without_buffering(self):
        pulled = []

        def source(path, fps):
            for i in range(3):
                pulled.append(i)
                yield i / fps, np.zeros((1, 1, 3), np.uint8)

        timed = cli.TimedDecoder(types.SimpleNamespace(iterate=source))
        it = timed.iterate("clip", 10.0)
        t, _ = next(it)
        self.assertEqual((t, pulled), (0.0, [0]))  # nothing read ahead
        self.assertEqual([t for t, _ in it], [0.1, 0.2])
        self.assertEqual(timed.frames, 3)
        self.assertGreaterEqual(timed.seconds, 0.0)


class ServeTests(unittest.TestCase):
    def _serve(self, outcomes, max_polls):
        seq = list(outcomes)
        sleeps, logs = [], []

        def run_once():
            o = seq.pop(0)
            if isinstance(o, BaseException):
                raise o
            return o

        cli.serve(run_once, 60, sleep=sleeps.append, log=logs.append, max_polls=max_polls)
        return sleeps, logs

    def test_polls_again_at_once_after_a_job_and_sleeps_when_idle(self):
        sleeps, _ = self._serve(["done", "failed", "orphaned", "idle"], 4)
        self.assertEqual(sleeps, [60])

    def test_an_exception_is_logged_and_waited_out(self):
        sleeps, logs = self._serve([ConnectionError("network down"), "idle"], 2)
        self.assertEqual(sleeps, [60, 60])
        self.assertIn("poll error: ConnectionError: network down", logs[0])

    def test_consecutive_idles_log_once(self):
        _, logs = self._serve(["idle", "idle", "idle", "done", "idle"], 5)
        self.assertEqual(len([m for m in logs if "idle" in m]), 2)
        self.assertEqual(len([m for m in logs if "done" in m]), 1)

    def test_ctrl_c_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self._serve(["idle", KeyboardInterrupt()], None)


class LoggingDbTests(unittest.TestCase):
    def test_logs_job_events_and_delegates(self):
        inner = cli.DryRunDb({"job_id": "j1", "attempts": 2, "movement": "Squat", "storage_key": "secret?"})
        logs = []
        db = cli.LoggingDb(inner, log=logs.append)
        db.heartbeat()
        self.assertEqual(db.claim()["job_id"], "j1")
        self.assertIsNone(db.claim())
        self.assertTrue(db.complete("j1", "k", {"key_frames": [{}, {}]}))
        db.fail("j1", "boom")
        self.assertEqual(inner.heartbeats, 1)
        self.assertEqual(inner.failed, [("j1", "boom")])
        self.assertEqual(
            logs,
            ["claimed job j1 (attempt 2, Squat)", "job j1: done, 2 key frame(s)", "job j1 failed: boom"],
        )


# ---------------------------------------------------------------------------------------
# runtime.r2_store, with a fake S3 client
# ---------------------------------------------------------------------------------------
class _FakePaginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.pages)


class _FakeS3:
    def __init__(self, pages=(), delete_errors=None, download_error=None):
        self.puts = []
        self.deletes = []
        self.paginator = _FakePaginator(list(pages))
        self.delete_errors = delete_errors
        self.download_error = download_error

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator

    def delete_objects(self, **kwargs):
        self.deletes.append(kwargs)
        return {"Errors": self.delete_errors} if self.delete_errors else {}

    def download_file(self, bucket, key, path):
        with open(path, "wb") as fh:
            fh.write(b"partial")
        if self.download_error:
            raise self.download_error
        with open(path, "wb") as fh:
            fh.write(f"{bucket}:{key}".encode())


class R2StoreTests(unittest.TestCase):
    def test_put_is_immutable_and_gzip_json_is_marked(self):
        s3 = _FakeS3()
        store = r2_store.R2Store(s3, "bkt")
        store.put("u/o/v/nlf.v1.a1/turntable/k00.webp", b"w", content_type="image/webp")
        store.put("u/o/v/nlf.v1.a1/pose3d.json.gz", b"g", content_type="application/json")
        webp, gz = s3.puts
        self.assertEqual(webp["Bucket"], "bkt")
        self.assertEqual(webp["ContentType"], "image/webp")
        self.assertEqual(webp["CacheControl"], "public, max-age=31536000, immutable")
        self.assertNotIn("ContentEncoding", webp)
        self.assertEqual(gz["ContentType"], "application/json")
        self.assertEqual(gz["ContentEncoding"], "gzip")
        self.assertEqual(gz["CacheControl"], "public, max-age=31536000, immutable")

    def test_delete_prefix_pages_with_a_trailing_slash(self):
        pages = [
            {"Contents": [{"Key": "u/o/v/nlf.v1.a1/a"}, {"Key": "u/o/v/nlf.v1.a1/b"}]},
            {},
            {"Contents": [{"Key": "u/o/v/nlf.v1.a1/c"}]},
        ]
        s3 = _FakeS3(pages)
        r2_store.R2Store(s3, "bkt").delete_prefix("u/o/v/nlf.v1.a1")
        self.assertEqual(s3.paginator.calls, [{"Bucket": "bkt", "Prefix": "u/o/v/nlf.v1.a1/"}])
        self.assertEqual(
            [[o["Key"] for o in d["Delete"]["Objects"]] for d in s3.deletes],
            [["u/o/v/nlf.v1.a1/a", "u/o/v/nlf.v1.a1/b"], ["u/o/v/nlf.v1.a1/c"]],
        )

    def test_delete_prefix_refuses_anything_but_an_nlf_result_folder(self):
        s3 = _FakeS3([{"Contents": [{"Key": "u/o/v/source"}]}])
        store = r2_store.R2Store(s3, "bkt")
        for prefix in ("u/o/v", "u/o/v/", "", "u/../nlf.v1.a1"):
            with self.assertRaises(ValueError):
                store.delete_prefix(prefix)
        self.assertEqual(s3.deletes, [])

    def test_per_key_delete_failures_raise(self):
        s3 = _FakeS3([{"Contents": [{"Key": "p/nlf.v1.a1/a"}]}], delete_errors=[{"Key": "p/nlf.v1.a1/a", "Code": "AccessDenied"}])
        with self.assertRaises(RuntimeError) as ctx:
            r2_store.R2Store(s3, "bkt").delete_prefix("p/nlf.v1.a1")
        self.assertIn("AccessDenied", str(ctx.exception))

    def test_fetcher_downloads_to_a_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "tmp")
            path = r2_store.R2Fetcher(_FakeS3(), "bkt").download("u/o/v/source", dest)
            self.assertEqual(os.path.dirname(path), dest)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), b"bkt:u/o/v/source")

    def test_fetcher_removes_a_partial_download(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OSError):
                r2_store.R2Fetcher(_FakeS3(download_error=OSError("reset")), "bkt").download("k", d)
            self.assertEqual(os.listdir(d), [])

    def test_make_s3_client_targets_the_account_endpoint(self):
        pytest.importorskip("boto3")
        client = r2_store.make_s3_client("acct123", "AKID", "SECRET")
        self.assertEqual(client.meta.endpoint_url, "https://acct123.r2.cloudflarestorage.com")
        self.assertEqual(client.meta.region_name, "auto")


# ---------------------------------------------------------------------------------------
# runtime.supabase_db, with a fake Supabase client
# ---------------------------------------------------------------------------------------
class _Session:
    def __init__(self, token, expires_at):
        self.access_token = token
        self.expires_at = expires_at


class _Resp:
    def __init__(self, data=None, session=None):
        self.data = data
        self.session = session


class _ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class _FakeAuth:
    def __init__(self, clock):
        self.clock = clock
        self.sign_ins = []
        self.refreshes = 0
        self.refresh_error = None

    def sign_in_with_password(self, creds):
        self.sign_ins.append(dict(creds))
        return _Resp(session=_Session(f"tok{len(self.sign_ins)}", self.clock() + 3600))

    def refresh_session(self):
        self.refreshes += 1
        if self.refresh_error:
            raise self.refresh_error
        return _Resp(session=_Session(f"ref{self.refreshes}", self.clock() + 3600))


class _Exec:
    def __init__(self, outcome):
        self.outcome = outcome

    def execute(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return _Resp(data=self.outcome)


class _FakeClient:
    def __init__(self, clock, outcomes=()):
        self.auth = _FakeAuth(clock)
        self.outcomes = list(outcomes)
        self.calls = []

    def rpc(self, fn, params):
        self.calls.append((fn, params))
        return _Exec(self.outcomes.pop(0) if self.outcomes else None)


class SupabaseDbTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000.0

    def _db(self, outcomes=()):
        client = _FakeClient(lambda: self.now, outcomes)
        db = supabase_db.SupabaseDb(
            "https://x.supabase.co", "anon", "w@example.com", _FAKE_CREDENTIAL, client=client, clock=lambda: self.now
        )
        return db, client

    def test_signs_in_once_then_reuses_the_session(self):
        db, client = self._db()
        db.heartbeat()
        db.heartbeat()
        self.assertEqual(client.auth.sign_ins, [{"email": "w@example.com", "password": _FAKE_CREDENTIAL}])
        self.assertEqual(client.auth.refreshes, 0)
        self.assertEqual(client.calls, [("nlf_worker_heartbeat", {}), ("nlf_worker_heartbeat", {})])

    def test_refreshes_near_expiry(self):
        db, client = self._db()
        db.heartbeat()
        self.now += 3600 - supabase_db.REFRESH_MARGIN_S + 1
        db.heartbeat()
        self.assertEqual(client.auth.refreshes, 1)
        self.assertEqual(len(client.auth.sign_ins), 1)

    def test_signs_in_again_when_refresh_fails(self):
        db, client = self._db()
        db.heartbeat()
        client.auth.refresh_error = RuntimeError("Invalid Refresh Token")
        self.now += 7200  # the PC slept past expiry
        db.heartbeat()
        self.assertEqual(client.auth.refreshes, 1)
        self.assertEqual(len(client.auth.sign_ins), 2)

    def test_auth_error_retries_once_after_a_new_sign_in(self):
        db, client = self._db([_ApiError("PGRST303", "JWT expired"), [{"job_id": "j1", "attempts": 1}]])
        self.assertEqual(db.claim(), {"job_id": "j1", "attempts": 1})
        self.assertEqual(len(client.auth.sign_ins), 2)
        self.assertEqual([c[0] for c in client.calls], ["nlf_claim_job", "nlf_claim_job"])

    def test_missing_worker_role_is_not_an_auth_error(self):
        db, client = self._db([_ApiError("42501", "not an nlf worker")])
        with self.assertRaises(_ApiError):
            db.heartbeat()
        self.assertEqual(len(client.auth.sign_ins), 1)
        self.assertEqual(len(client.calls), 1)

    def test_claim_returns_the_first_row_or_none(self):
        db, _ = self._db([[{"job_id": "j1"}], [], None])
        self.assertEqual(db.claim(), {"job_id": "j1"})
        self.assertIsNone(db.claim())
        self.assertIsNone(db.claim())

    def test_complete_is_true_only_for_an_unambiguous_true(self):
        db, client = self._db([True, [True], False, None, [], "true"])
        results = [db.complete("j1", "k", {"key_frames": []}) for _ in range(6)]
        self.assertEqual(results, [True, True, False, False, False, False])
        self.assertEqual(
            client.calls[0],
            ("nlf_complete_job", {"p_job_id": "j1", "p_result_key": "k", "p_meta": {"key_frames": []}}),
        )

    def test_fail_passes_the_error(self):
        db, client = self._db()
        db.fail("j1", "no frame had a detection")
        self.assertEqual(client.calls, [("nlf_fail_job", {"p_job_id": "j1", "p_error": "no frame had a detection"})])

    def test_repr_hides_credentials(self):
        db, _ = self._db()
        self.assertNotIn(_FAKE_CREDENTIAL, repr(db))
        self.assertNotIn("anon", repr(db))

    def test_is_auth_error(self):
        self.assertTrue(supabase_db.is_auth_error(_ApiError("PGRST301", "JWSError")))
        self.assertTrue(supabase_db.is_auth_error(_ApiError(401, "Unauthorized")))
        self.assertTrue(supabase_db.is_auth_error(_ApiError(None, "JWT expired")))
        self.assertFalse(supabase_db.is_auth_error(_ApiError("42501", "permission denied")))
        self.assertFalse(supabase_db.is_auth_error(ConnectionError("down")))


# ---------------------------------------------------------------------------------------
# heavy runtime adapters (skipped wherever their stack is missing, e.g. CI)
# ---------------------------------------------------------------------------------------
class NlfModelTests(unittest.TestCase):
    """CPU-only checks of the adapter's plumbing; the real inference is the GPU dry-run."""

    def setUp(self):
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        from src.nlf_worker.runtime import model

        self.model_mod = model
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_creates_the_fp32_archive_when_missing(self):
        src = os.path.join(self.tmp.name, "stock.torchscript")
        dst = os.path.join(self.tmp.name, "sub", "fp32.torchscript")
        _write_archive(src)
        m = self.model_mod.NlfModel(src, dst)
        self.assertTrue(os.path.exists(dst))
        import hashlib

        with open(dst, "rb") as fh:
            self.assertEqual(m.model_sha256, hashlib.sha256(fh.read()).hexdigest())
        self.assertIsInstance(m.torch_version, str)

    def test_batches_split_where_the_frame_size_changes(self):
        dst = os.path.join(self.tmp.name, "fp32.torchscript")
        Path(dst).write_bytes(b"already there")
        m = self.model_mod.NlfModel("unused-stock-path", dst)
        sizes = []
        m._infer_same_size = lambda frames: sizes.append(len(frames)) or [{"n": len(frames)}] * len(frames)
        a, b = np.zeros((4, 6, 3), np.uint8), np.zeros((6, 4, 3), np.uint8)
        out = m.infer([a, a, b, b, b, a])
        self.assertEqual(sizes, [2, 3, 1])
        self.assertEqual(len(out), 6)
        self.assertEqual(m.infer([]), [])


class DecoderTests(unittest.TestCase):
    def test_fit_short_side_downscales_only(self):
        pytest.importorskip("cv2")
        pytest.importorskip("av")
        from src.nlf_worker.runtime.decoder import fit_short_side

        self.assertEqual(fit_short_side(np.zeros((1280, 2560, 3), np.uint8), 720).shape, (720, 1440, 3))
        self.assertEqual(fit_short_side(np.zeros((2160, 3840, 3), np.uint8), 720).shape, (720, 1280, 3))
        small = np.zeros((480, 640, 3), np.uint8)
        self.assertIs(fit_short_side(small, 720), small)

    def test_samples_a_synthetic_clip_by_stream_time(self):
        av = pytest.importorskip("av")
        pytest.importorskip("cv2")
        from src.nlf_worker.runtime.decoder import PyAvDecoder

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.mp4")
            with av.open(path, "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
                for i in range(60):  # 2 s at 30 fps
                    img = np.full((240, 320, 3), (i * 4) % 256, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            samples = list(PyAvDecoder(short_side=120).iterate(path, 10.0))
        times = [t for t, _ in samples]
        self.assertEqual(len(samples), 20)
        np.testing.assert_allclose(times, np.arange(20) / 10.0, atol=1e-6)
        self.assertEqual(samples[0][1].shape, (120, 160, 3))
        self.assertEqual(samples[0][1].dtype, np.uint8)


class TurntableRendererTests(unittest.TestCase):
    def test_renders_a_strip_of_tiles(self):
        pytest.importorskip("pyrender")
        pytest.importorskip("trimesh")
        cv2 = pytest.importorskip("cv2")
        from src.nlf_worker.runtime.renderer import TurntableRenderer

        # A tetrahedron "body" spanning 1.6 m, in camera mm (y down), plus a matching skeleton.
        verts = np.array([[0, -900, 2000], [-150, 700, 2000], [150, 700, 2000], [0, 700, 2150]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        joints = _squat_joints(5)
        with tempfile.TemporaryDirectory() as d:
            pkl = os.path.join(d, "faces.pkl")
            with open(pkl, "wb") as fh:
                pickle.dump({"f": faces}, fh, protocol=2)
            renderer = TurntableRenderer(pkl, tile_px=64, angles=8)
            try:
                data = renderer.render_strip(verts, joints, geometry.body_basis(np.array([0.0, -1.0, 0.0])))
                with self.assertRaises(ValueError):
                    renderer.render_strip(None, joints, np.eye(3))
            finally:
                renderer.close()
        strip = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(strip.shape, (64, 8 * 64, 3))
        self.assertLess(strip[:, :64].mean(), 250)  # something was drawn on the white tile

    def test_a_shared_framing_keeps_the_scale_across_key_frames(self):
        pytest.importorskip("pyrender")
        pytest.importorskip("trimesh")
        from src.nlf_worker.runtime.renderer import TurntableRenderer

        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        tall = np.array([[0, -900, 2000], [-150, 700, 2000], [150, 700, 2000], [0, 700, 2150]], dtype=float)
        short = tall.copy()
        short[0, 1] = -100.0  # the same shape 0.8 m shorter (a crouch)
        joints = _squat_joints(5)
        basis = geometry.body_basis(np.array([0.0, -1.0, 0.0]))

        def drawn_rows(tiles):
            body = (tiles[0][:, :, 0].astype(int) - tiles[0][:, :, 2].astype(int)) > 25
            rows = np.where(body.any(axis=1))[0]
            return rows.max() - rows.min() + 1

        with tempfile.TemporaryDirectory() as d:
            pkl = os.path.join(d, "faces.pkl")
            with open(pkl, "wb") as fh:
                pickle.dump({"f": faces}, fh, protocol=2)
            renderer = TurntableRenderer(pkl, tile_px=128, angles=4)
            try:
                bodies = [geometry.to_body_frame(v, joints[geometry.PELVIS], basis) for v in (tall, short)]
                shared = geometry.turntable_framing(bodies)
                tall_rows = drawn_rows(renderer.render_tiles(tall, joints, basis, shared))
                short_shared = drawn_rows(renderer.render_tiles(short, joints, basis, shared))
                short_alone = drawn_rows(renderer.render_tiles(short, joints, basis))
            finally:
                renderer.close()
        # Alone, the short body is blown up to fill the tile; with the shared framing it keeps its
        # true size relative to the tall one (0.8 m vs 1.6 m of mesh height, so about half).
        self.assertGreater(short_alone, 0.8 * tall_rows)
        self.assertAlmostEqual(short_shared / tall_rows, 0.5, delta=0.08)


if __name__ == "__main__":
    unittest.main()
