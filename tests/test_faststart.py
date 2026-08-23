"""Tests for the streaming remux (``backend.app.services.faststart``).

Two halves. The atom/EBML inspection is tested against hand-built byte strings — it is pure
parsing, and a synthetic file pins the exact layouts (moov-last, moov-first, 64-bit sizes,
truncation) that a real sample would only cover by accident. The remux itself is tested with
``subprocess.run`` mocked, because what matters is the DEGRADATION CONTRACT — every failure path
returns the caller's original bytes — and a real ffmpeg cannot be made to fail on demand. One
end-to-end case runs the real binary when it is present, so the mocked half cannot drift from a
command ffmpeg would reject.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from backend.app.services import faststart


def _atom(atom_type: bytes, payload: bytes = b"") -> bytes:
    """One 32-bit-size top-level atom."""
    return struct.pack(">I", 8 + len(payload)) + atom_type + payload


def _atom64(atom_type: bytes, payload: bytes = b"") -> bytes:
    """The same atom with a 64-bit extended size, which large ``mdat`` boxes really use."""
    return struct.pack(">I", 1) + atom_type + struct.pack(">Q", 16 + len(payload)) + payload


MOOV_LAST = _atom(b"ftyp", b"isom") + _atom(b"mdat", b"\x00" * 64) + _atom(b"moov", b"\x00" * 16)
MOOV_FIRST = _atom(b"ftyp", b"isom") + _atom(b"moov", b"\x00" * 16) + _atom(b"mdat", b"\x00" * 64)


class NeedsFaststartTests(unittest.TestCase):
    def test_moov_after_mdat_needs_the_move(self) -> None:
        """The case this whole module exists for: every phone camera writes this layout, and a
        browser cannot render one frame of it until the last byte has arrived."""
        self.assertTrue(faststart.needs_faststart(MOOV_LAST))

    def test_moov_before_mdat_is_already_streamable(self) -> None:
        self.assertFalse(faststart.needs_faststart(MOOV_FIRST))

    def test_a_sixty_four_bit_mdat_is_skipped_by_its_real_size(self) -> None:
        """A >4GB-capable ``mdat`` puts its length after the type. Reading the 32-bit field as the
        size would walk into the middle of the media and mis-report the atom order."""
        data = _atom(b"ftyp", b"isom") + _atom64(b"mdat", b"\x00" * 64) + _atom(b"moov")
        self.assertTrue(faststart.needs_faststart(data))

    def test_a_trailing_size_zero_atom_terminates_the_scan(self) -> None:
        """Size 0 means "to end of file"; it is legal and always last."""
        data = _atom(b"ftyp", b"isom") + struct.pack(">I", 0) + b"mdat" + b"\x00" * 64
        self.assertTrue(faststart.needs_faststart(data))

    def test_garbage_is_not_reported_as_needing_a_remux(self) -> None:
        """No ``mdat`` was seen, so there is nothing to move ahead of — and, crucially, parsing
        junk must not raise into the upload path."""
        self.assertFalse(faststart.needs_faststart(b"not an mp4 at all"))

    def test_a_truncated_header_stops_rather_than_raising(self) -> None:
        self.assertFalse(faststart.needs_faststart(_atom(b"ftyp", b"isom") + b"\x00\x00"))

    def test_a_truncated_sixty_four_bit_header_stops_rather_than_raising(self) -> None:
        """The extended size is read past the type, so a file cut mid-header would unpack short."""
        data = _atom(b"ftyp", b"isom") + struct.pack(">I", 1) + b"mdat" + b"\x00\x00\x00"
        self.assertFalse(faststart.needs_faststart(data))

    def test_the_scan_is_bounded_so_a_pathological_file_cannot_pin_a_worker(self) -> None:
        """Thousands of tiny top-level boxes are not a real MP4, but they ARE a cheap way to make
        this loop walk a whole upload. The cap ends the scan; a file whose ``moov`` was never
        reached is treated as not needing the move, and simply stays as it is."""
        data = _atom(b"ftyp", b"isom") * (faststart._MAX_ATOMS_SCANNED + 10) + _atom(b"mdat")
        self.assertFalse(faststart.needs_faststart(data))

    def test_a_size_smaller_than_its_own_header_stops_the_scan(self) -> None:
        """A corrupt size would otherwise leave the offset stuck or moving backwards."""
        data = _atom(b"ftyp", b"isom") + struct.pack(">I", 4) + b"mdat"
        self.assertFalse(faststart.needs_faststart(data))


class NeedsWebmIndexTests(unittest.TestCase):
    def test_a_live_muxed_segment_has_no_index(self) -> None:
        """``MediaRecorder`` writes the unknown-size marker because it does not know the length
        while recording — the same defect that makes ``video.duration`` report Infinity."""
        data = b"\x1a\x45\xdf\xa3" + b"\x18\x53\x80\x67" + faststart._EBML_UNKNOWN_SIZE
        self.assertTrue(faststart.needs_webm_index(data))

    def test_a_file_muxed_segment_is_left_alone(self) -> None:
        data = b"\x1a\x45\xdf\xa3" + b"\x18\x53\x80\x67" + b"\x01\x00\x00\x00\x00\x0f\x42\x40"
        self.assertFalse(faststart.needs_webm_index(data))


class OptimizeDispatchTests(unittest.TestCase):
    """Which inputs reach ffmpeg at all. A needless remux costs a subprocess and a full rewrite of
    the clip on every upload, so "already fine" must short-circuit before ``shutil.which``."""

    def test_an_already_faststart_mp4_never_spawns_ffmpeg(self) -> None:
        with mock.patch.object(faststart.subprocess, "run") as run:
            self.assertIs(faststart.optimize_for_streaming(MOOV_FIRST, ".mp4"), MOOV_FIRST)
        run.assert_not_called()

    def test_an_unhandled_container_is_passed_through(self) -> None:
        data = b"\x00" * 64
        with mock.patch.object(faststart.subprocess, "run") as run:
            self.assertIs(faststart.optimize_for_streaming(data, ".avi"), data)
        run.assert_not_called()

    def test_the_suffix_check_is_case_insensitive(self) -> None:
        """Phones hand back ``.MOV`` often enough that a case-sensitive check would silently skip
        exactly the uploads that need this most."""
        with mock.patch.object(faststart, "_remux", return_value=b"x" * 2048) as remux:
            faststart.optimize_for_streaming(MOOV_LAST, ".MOV")
        remux.assert_called_once()
        self.assertEqual(remux.call_args.args[1], ".mov")

    def test_mp4_asks_for_faststart_and_a_stream_copy(self) -> None:
        with mock.patch.object(faststart, "_remux", return_value=b"x" * 2048) as remux:
            faststart.optimize_for_streaming(MOOV_LAST, ".mp4")
        self.assertEqual(remux.call_args.args[2], ["-movflags", "+faststart"])

    def test_webm_needs_no_flag_because_writing_to_a_file_writes_the_index(self) -> None:
        data = b"\x1a\x45\xdf\xa3" + faststart._EBML_UNKNOWN_SIZE
        with mock.patch.object(faststart, "_remux", return_value=b"x" * 2048) as remux:
            faststart.optimize_for_streaming(data, ".webm")
        self.assertEqual(remux.call_args.args[2], [])


class RemuxDegradationTests(unittest.TestCase):
    """Every failure returns the ORIGINAL bytes. A clip that streams poorly is worth having; an
    upload that 500s because a codec was unusual is not."""

    def setUp(self) -> None:
        self.data = MOOV_LAST

    def test_a_missing_ffmpeg_keeps_the_original(self) -> None:
        """ffmpeg ships in backend/Dockerfile but is not guaranteed on a development machine."""
        with mock.patch.object(faststart.shutil, "which", return_value=None):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_a_nonzero_exit_keeps_the_original(self) -> None:
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 1, b"", b"unsupported codec"),
        ):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_a_timeout_keeps_the_original(self) -> None:
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1),
        ):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_an_unlaunchable_binary_keeps_the_original(self) -> None:
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run", side_effect=OSError("Exec format error")
        ):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_a_success_that_wrote_no_output_file_keeps_the_original(self) -> None:
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0, b"", b""),
        ):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_an_implausibly_tiny_output_keeps_the_original(self) -> None:
        """A header-only file is the shape of a silent failure: ffmpeg exits 0 having written a
        container with no media. Storing it would replace a playable clip with an unplayable one."""

        def write_stub(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"\x00" * 10)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run", side_effect=write_stub
        ):
            self.assertIs(faststart.optimize_for_streaming(self.data, ".mp4"), self.data)

    def test_a_real_output_replaces_the_original(self) -> None:
        remuxed = b"\x01" * 4096

        def write_output(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(remuxed)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run", side_effect=write_output
        ):
            self.assertEqual(faststart.optimize_for_streaming(self.data, ".mp4"), remuxed)

    def test_the_command_copies_streams_and_never_re_encodes(self) -> None:
        """Load-bearing beyond speed: SkeletonOverlay maps landmarks onto ``currentTime`` and the
        detections carry frame indices from the original file, so a re-encode that resampled
        timestamps would drift the skeleton off the body with nothing failing loudly."""
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 1, b"", b""),
        ) as run:
            faststart.optimize_for_streaming(self.data, ".mp4")
        cmd = run.call_args.args[0]
        self.assertIn("-c", cmd)
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertIn("+faststart", cmd)

    def test_the_remux_is_bounded_by_a_timeout(self) -> None:
        """A stream copy is I/O bound; anything long is a hang, and an unbounded wait would pin an
        analysis worker slot (``MAX_CONCURRENT_ANALYSES``) indefinitely."""
        with mock.patch.object(faststart.shutil, "which", return_value="ffmpeg"), mock.patch.object(
            faststart.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 1, b"", b""),
        ) as run:
            faststart.optimize_for_streaming(self.data, ".mp4")
        self.assertEqual(run.call_args.kwargs["timeout"], faststart.REMUX_TIMEOUT_SECONDS)


class IsStreamableTests(unittest.TestCase):
    """The predicate the backfill needs to tell "nothing to do" from "the remux failed" —
    ``optimize_for_streaming`` returns the caller's bytes in both cases."""

    def test_a_moov_last_mp4_is_not_streamable(self) -> None:
        self.assertFalse(faststart.is_streamable(MOOV_LAST, ".mp4"))

    def test_a_moov_first_mp4_is_streamable(self) -> None:
        self.assertTrue(faststart.is_streamable(MOOV_FIRST, ".mp4"))

    def test_a_live_muxed_webm_is_not_streamable(self) -> None:
        data = b"\x1a\x45\xdf\xa3" + faststart._EBML_UNKNOWN_SIZE
        self.assertFalse(faststart.is_streamable(data, ".webm"))

    def test_a_container_this_module_does_not_touch_counts_as_streamable(self) -> None:
        """"Nothing to fix here" and "this is fine" are the same answer to every caller."""
        self.assertTrue(faststart.is_streamable(b"\x00" * 64, ".avi"))


class _FakeStore:
    """An in-memory object store with the maintenance surface (``iter_keys`` / ``get``)."""

    def __init__(self, objects: dict[str, tuple[bytes, str]]) -> None:
        self.objects = dict(objects)
        self.puts: list[str] = []
        self.raise_on_put: set[str] = set()

    def iter_keys(self, prefix: str):
        return iter(sorted(k for k in self.objects if k.startswith(f"{prefix}/")))

    def get(self, key: str):
        return self.objects.get(key)

    def put(self, key: str, data: bytes, *, content_type: str) -> None:
        if key in self.raise_on_put:
            raise RuntimeError("R2 down")
        self.puts.append(key)
        self.objects[key] = (data, content_type)


class BackfillTests(unittest.TestCase):
    """The half of the fix that reaches clips ALREADY in the bucket. Without it the upload-path
    remux only helps videos that do not exist yet, which is nobody's history."""

    def setUp(self) -> None:
        self.remuxed = b"\x02" * 4096
        patcher = mock.patch.object(faststart, "_remux", return_value=self.remuxed)
        self.remux = patcher.start()
        self.addCleanup(patcher.stop)

    def test_rewrites_a_moov_last_clip_in_place_under_its_own_key(self) -> None:
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4")})
        report = faststart.backfill(store)
        self.assertEqual(store.puts, ["uploads/u1/v1/source"])
        self.assertEqual(store.objects["uploads/u1/v1/source"], (self.remuxed, "video/mp4"))
        self.assertEqual(report.rewritten, 1)

    def test_leaves_an_already_streamable_clip_untouched(self) -> None:
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_FIRST, "video/mp4")})
        report = faststart.backfill(store)
        self.assertEqual(store.puts, [])
        self.assertEqual(report.already_streamable, 1)
        self.remux.assert_not_called()

    def test_a_second_pass_rewrites_nothing(self) -> None:
        """Rerunnable by design: a deploy step that is unsafe to repeat will be repeated anyway."""
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4")})
        faststart.backfill(store)
        self.remux.return_value = MOOV_FIRST  # what a real remux would have produced
        store.objects["uploads/u1/v1/source"] = (MOOV_FIRST, "video/mp4")
        second = faststart.backfill(store)
        self.assertEqual(second.rewritten, 0)
        self.assertEqual(second.already_streamable, 1)

    def test_ignores_the_pose_json_and_thumbnail_beside_the_clip(self) -> None:
        """They share the prefix and are not video; remuxing a JSON file is not a no-op, it is a
        wasted download and an ffmpeg invocation per analysis in the bucket."""
        store = _FakeStore(
            {
                "uploads/u1/v1/source": (MOOV_LAST, "video/mp4"),
                "uploads/u1/v1/pose.json": (b"{}", "application/json"),
                "uploads/u1/v1/thumb.jpg": (b"\xff\xd8", "image/jpeg"),
            }
        )
        report = faststart.backfill(store)
        self.assertEqual(report.scanned, 1)
        self.assertEqual(store.puts, ["uploads/u1/v1/source"])

    def test_skips_a_content_type_it_cannot_act_on_rather_than_guessing(self) -> None:
        """The source object deliberately carries no extension, so ContentType is the only signal
        for what the file is. Guessing ``.mp4`` for an unknown type would hand ffmpeg a lie."""
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/x-matroska")})
        report = faststart.backfill(store)
        self.assertEqual((report.skipped, report.rewritten), (1, 0))
        self.assertEqual(store.puts, [])

    def test_a_content_type_with_parameters_still_resolves(self) -> None:
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4; codecs=avc1")})
        report = faststart.backfill(store)
        self.assertEqual(report.rewritten, 1)

    def test_an_object_deleted_mid_run_is_skipped_not_fatal(self) -> None:
        """The listing can move under the pass — a user may delete an analysis while it runs."""
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4")})
        del store.objects["uploads/u1/v1/source"]
        store.objects["uploads/u1/v1/source"] = (MOOV_LAST, "video/mp4")
        store.get = lambda key: None  # type: ignore[assignment]
        report = faststart.backfill(store)
        self.assertEqual((report.scanned, report.skipped), (1, 1))

    def test_a_clip_ffmpeg_cannot_remux_is_reported_as_failed_not_as_already_fine(self) -> None:
        """The distinction the whole report hangs on: ``optimize_for_streaming`` returns the
        original bytes for BOTH "nothing to do" and "the remux failed", so counting unchanged
        bytes as success would report a clean sweep over a bucket it fixed nothing in."""
        self.remux.side_effect = lambda data, suffix, extra: data
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4")})
        report = faststart.backfill(store)
        self.assertEqual((report.failed, report.already_streamable, report.rewritten), (1, 0, 0))
        self.assertEqual(report.failed_keys, ["uploads/u1/v1/source"])
        self.assertEqual(store.puts, [])

    def test_a_failed_write_leaves_the_original_and_does_not_end_the_pass(self) -> None:
        store = _FakeStore(
            {
                "uploads/u1/v1/source": (MOOV_LAST, "video/mp4"),
                "uploads/u2/v2/source": (MOOV_LAST, "video/mp4"),
            }
        )
        store.raise_on_put = {"uploads/u1/v1/source"}
        report = faststart.backfill(store)
        self.assertEqual(report.failed_keys, ["uploads/u1/v1/source"])
        self.assertEqual(report.rewritten, 1)
        self.assertEqual(store.objects["uploads/u1/v1/source"], (MOOV_LAST, "video/mp4"))

    def test_a_dry_run_remuxes_but_never_writes(self) -> None:
        """It skips only the PUT, so its counts are the counts a real run will produce — including
        which clips ffmpeg cannot handle, which a cheaper dry run could not know."""
        store = _FakeStore({"uploads/u1/v1/source": (MOOV_LAST, "video/mp4")})
        report = faststart.backfill(store, dry_run=True)
        self.assertEqual(report.rewritten, 1)
        self.assertEqual(store.puts, [])
        self.assertEqual(store.objects["uploads/u1/v1/source"], (MOOV_LAST, "video/mp4"))
        self.remux.assert_called_once()

    def test_the_prefix_narrows_the_pass_to_one_user(self) -> None:
        store = _FakeStore(
            {
                "uploads/u1/v1/source": (MOOV_LAST, "video/mp4"),
                "uploads/u2/v2/source": (MOOV_LAST, "video/mp4"),
            }
        )
        report = faststart.backfill(store, prefix="uploads/u1")
        self.assertEqual(report.scanned, 1)
        self.assertEqual(store.puts, ["uploads/u1/v1/source"])

    def test_a_live_muxed_webm_upload_is_rewritten_too(self) -> None:
        data = b"\x1a\x45\xdf\xa3" + faststart._EBML_UNKNOWN_SIZE
        store = _FakeStore({"uploads/u1/v1/source": (data, "video/webm")})
        report = faststart.backfill(store)
        self.assertEqual(report.rewritten, 1)
        self.assertEqual(self.remux.call_args.args[1], ".webm")


_DEMO_CLIP = Path(__file__).resolve().parents[1] / "data" / "demo" / "32979_1.mp4"


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
@unittest.skipUnless(_DEMO_CLIP.is_file(), "the demo clip is not checked out")
class RealFfmpegTests(unittest.TestCase):
    """One end-to-end pass over a real moov-last clip, so the mocked tests above cannot drift from
    a command line ffmpeg actually accepts."""

    def test_a_real_moov_last_clip_comes_back_streamable(self) -> None:
        original = _DEMO_CLIP.read_bytes()
        self.assertTrue(faststart.needs_faststart(original), "fixture is no longer moov-last")
        out = faststart.optimize_for_streaming(original, ".mp4")
        self.assertNotEqual(out, original)
        self.assertFalse(faststart.needs_faststart(out))
        # A stream copy moves the index, it does not shrink the media. A wildly smaller file would
        # mean ffmpeg dropped streams rather than relocating the header.
        self.assertGreater(len(out), len(original) * 0.9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
