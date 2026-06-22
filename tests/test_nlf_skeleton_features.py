from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rehab24.nlf_skeleton_features import (
    build_features,
    raw_npz_name,
    repetition_feature,
)
from src.rehab24.skeleton_features import SUMMARY_SIZE

# 24 SMPL joints; 3D block = J*3*2 (pos+vel) channels, 2D block = J*2*2, each x SUMMARY_SIZE stats.
DIM_3D = 24 * 3 * 2 * SUMMARY_SIZE
DIM_2D = 24 * 2 * 2 * SUMMARY_SIZE


def _synth(frames: int = 60):
    """Synthetic SMPL-24 joints: finite shoulder/hip spans so normalization is well-defined."""
    rng = np.random.RandomState(0)
    j3 = (rng.randn(frames, 24, 3).astype("float32") * 50.0 + 1000.0)
    j2 = (rng.randn(frames, 24, 2).astype("float32") * 20.0 + 500.0)
    return j3, j2


class RawNpzNameTests(unittest.TestCase):
    def test_matches_kernel_out_name_convention(self):
        # The local builder must reconstruct exactly the kernel's per-video filename.
        self.assertEqual(raw_npz_name("Ex1/PM_000-Camera17-30fps.mp4"), "Ex1__PM_000-Camera17-30fps.npz")
        self.assertEqual(
            raw_npz_name("Ex4/PM_027-Camera18-30fps-transposed.mp4"),
            "Ex4__PM_027-Camera18-30fps-transposed.npz",
        )


class RepetitionFeatureTests(unittest.TestCase):
    def test_feature_dims_and_finite(self):
        j3, j2 = _synth()
        f_3d2d = repetition_feature(j3, j2, 1, 60, "3d2d")
        f_3d = repetition_feature(j3, j2, 1, 60, "3d")
        self.assertEqual(f_3d.shape[0], DIM_3D)
        self.assertEqual(f_3d2d.shape[0], DIM_3D + DIM_2D)
        self.assertTrue(np.isfinite(f_3d2d).all())

    def test_interpolates_dropped_detection(self):
        # A NaN frame inside the rep (a missed detection) must be filled, not propagated.
        j3, j2 = _synth()
        j3[5] = np.nan
        j2[5] = np.nan
        feat = repetition_feature(j3, j2, 1, 60, "3d2d")
        self.assertTrue(np.isfinite(feat).all())

    def test_segment_slicing_uses_frame_bounds(self):
        # Distinct motion before/after the rep window must not leak into the feature.
        j3, j2 = _synth()
        j3[:20] += 5000.0  # frames outside rep [30,60] should be ignored
        a = repetition_feature(j3, j2, 30, 60, "3d")
        b = repetition_feature(j3[20:], j2[20:], 10, 40, "3d")
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


class BuildFeaturesTests(unittest.TestCase):
    def _write_manifest(self, path: Path, video_path: str, reps):
        fields = ["sample_id", "split", "video_id", "exercise_id", "person_id",
                  "first_frame", "last_frame", "camera", "correctness", "video_path"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, (a, b, corr) in enumerate(reps, 1):
                w.writerow({
                    "sample_id": f"Ex6_PM_000_rep{i}_cam17", "split": "train", "video_id": "PM_000",
                    "exercise_id": "6", "person_id": "1", "first_frame": a, "last_frame": b,
                    "camera": "cam17", "correctness": corr, "video_path": video_path,
                })

    def test_end_to_end_writes_bundles(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw"
            raw.mkdir()
            vpath = "Ex6/PM_000-Camera17-30fps.mp4"
            j3, j2 = _synth()
            np.savez_compressed(
                raw / raw_npz_name(vpath), smpl3d=j3, smpl3d_np=j3 + 1.0, smpl2d=j2,
                unc=np.zeros((60, 24), "float32"), ndet=np.ones(60, "int16"),
            )
            manifest = td / "manifest.csv"
            self._write_manifest(manifest, vpath, [(1, 30, 1), (31, 60, 0)])
            out = td / "feat"
            written, missing = build_features(raw, manifest, out, "parametric", "3d2d", overwrite=True)

            self.assertEqual(written, 2)
            self.assertEqual(missing, [])
            bundles = sorted(out.rglob("*.npz"))
            self.assertEqual(len(bundles), 2)
            with np.load(bundles[0]) as d:
                self.assertEqual(int(d["video_feature"].shape[0]), DIM_3D + DIM_2D)
                self.assertTrue(np.isfinite(d["video_feature"]).all())
                self.assertIn(int(d["correctness"]), (0, 1))

    def test_missing_raw_video_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw"
            raw.mkdir()  # intentionally empty: no per-video npz
            manifest = td / "manifest.csv"
            self._write_manifest(manifest, "Ex6/PM_000-Camera17-30fps.mp4", [(1, 30, 1)])
            written, missing = build_features(raw, manifest, td / "feat", "parametric", "3d2d", overwrite=True)
            self.assertEqual(written, 0)
            self.assertEqual(missing, ["Ex6/PM_000-Camera17-30fps.mp4"])

    def test_nonparam_source_selects_other_array(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw"
            raw.mkdir()
            vpath = "Ex6/PM_000-Camera17-30fps.mp4"
            j3, j2 = _synth()
            np.savez_compressed(
                raw / raw_npz_name(vpath), smpl3d=j3, smpl3d_np=j3 + 100.0, smpl2d=j2,
                unc=np.zeros((60, 24), "float32"), ndet=np.ones(60, "int16"),
            )
            manifest = td / "manifest.csv"
            self._write_manifest(manifest, vpath, [(1, 60, 1)])
            par, _ = build_features(raw, manifest, td / "par", "parametric", "3d", overwrite=True)
            nph, _ = build_features(raw, manifest, td / "nph", "nonparam", "3d", overwrite=True)
            self.assertEqual((par, nph), (1, 1))
            # parametric and nonparam differ by a constant offset -> identical after root-centering.
            with np.load(next((td / "par").rglob("*.npz"))) as a, np.load(next((td / "nph").rglob("*.npz"))) as b:
                np.testing.assert_allclose(a["video_feature"], b["video_feature"], rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
