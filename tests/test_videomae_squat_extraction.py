from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.video.videomae_feature_extraction import (
    ClipRequest,
    build_requests,
    iter_requests,
    sample_clip_starts,
    save_feature_bundle,
    select_chunk,
)
from src.video.videomae_pooling import LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM, build_provenance


def write_split(split_dir: Path, split_name: str, video_ids: list[str]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{split_name}_keys.json").write_text(json.dumps(video_ids), encoding="utf-8")


class SampleClipStartsTests(unittest.TestCase):
    def test_four_clips_span_the_whole_video(self) -> None:
        # 16 frames at stride 2 covers 31 frames, so the last start is 120 - 31 = 89.
        self.assertEqual(sample_clip_starts(120, 16, 2, 4), [0, 29, 59, 89])

    def test_single_clip_is_centred(self) -> None:
        self.assertEqual(sample_clip_starts(120, 16, 2, 1), [44])

    def test_video_shorter_than_one_clip_collapses_to_zero(self) -> None:
        """The frame reader pads by repeating, so a short video is legal, not an error."""
        self.assertEqual(sample_clip_starts(20, 16, 2, 4), [0, 0, 0, 0])

    def test_unreadable_frame_count_still_yields_a_start(self) -> None:
        self.assertEqual(sample_clip_starts(0, 16, 2, 4), [0])


class BuildRequestsTests(unittest.TestCase):
    def test_requests_carry_their_split_and_find_nested_videos(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos"
            (video_root / "nested").mkdir(parents=True)
            (video_root / "a.mp4").write_bytes(b"")
            (video_root / "nested" / "b.mp4").write_bytes(b"")
            write_split(root / "Splits", "train", ["a"])
            write_split(root / "Splits", "val", ["b"])

            requests = build_requests(video_root, root / "Splits", ["train", "val"])

            self.assertEqual([(r.video_id, r.split) for r in requests], [("a", "train"), ("b", "val")])
            self.assertEqual(requests[1].video_path.name, "b.mp4")

    def test_missing_videos_are_reported_and_skipped(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos"
            video_root.mkdir(parents=True)
            (video_root / "a.mp4").write_bytes(b"")
            write_split(root / "Splits", "test", ["a", "gone"])

            requests = build_requests(video_root, root / "Splits", ["test"])

            self.assertEqual([r.video_id for r in requests], ["a"])


class SelectChunkTests(unittest.TestCase):
    def make_requests(self, count: int) -> list[ClipRequest]:
        return [ClipRequest(video_id=str(i), split="train", video_path=Path(f"{i}.mp4")) for i in range(count)]

    def test_chunks_partition_the_work_exactly_once(self) -> None:
        requests = self.make_requests(10)
        chunks = [select_chunk(requests, 3, index) for index in range(3)]

        covered = [request.video_id for chunk in chunks for request in chunk]
        self.assertEqual(sorted(covered, key=int), [r.video_id for r in requests])
        self.assertEqual(len(covered), len(set(covered)))

    def test_single_chunk_is_the_whole_list(self) -> None:
        requests = self.make_requests(4)
        self.assertEqual(select_chunk(requests, 1, 0), requests)

    def test_out_of_range_chunk_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_chunk(self.make_requests(4), 2, 2)


class SaveFeatureBundleTests(unittest.TestCase):
    def test_bundle_stores_both_pooling_stacks_and_the_variant_provenance(self) -> None:
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "train" / "a.npz"
            request = ClipRequest(video_id="a", split="train", video_path=Path("videos/a.mp4"))
            bundle = {
                f"clip_features_{LEGACY_FIRST_TOKEN}": np.zeros((4, 8), dtype=np.float32),
                f"clip_features_{MEAN_POOL_FC_NORM}": np.ones((4, 8), dtype=np.float32),
                "clip_starts": np.asarray([0, 1, 2, 3], dtype=np.int32),
                "total_frames": np.asarray(120, dtype=np.int32),
            }
            provenance = build_provenance(
                model_name="m",
                clip_length=16,
                frame_stride=2,
                num_clips=4,
                transformers_version="5.5.0",
                variant="person_crop",
            )

            save_feature_bundle(output_path, request, bundle, provenance)

            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(str(data["video_id"]), "a")
                self.assertEqual(str(data["split"]), "train")
                self.assertEqual(data[f"clip_features_{MEAN_POOL_FC_NORM}"].shape, (4, 8))
                self.assertEqual(str(data["provenance_variant"]), "person_crop")
                self.assertEqual(str(data["provenance_num_clips"]), "4")

    def test_limit_truncates_the_work_list(self) -> None:
        requests = [ClipRequest(video_id=str(i), split="train", video_path=Path("x.mp4")) for i in range(5)]
        self.assertEqual(len(list(iter_requests(requests, 2))), 2)
        self.assertEqual(len(list(iter_requests(requests, None))), 5)


if __name__ == "__main__":
    unittest.main()
