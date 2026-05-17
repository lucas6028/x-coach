from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_classifier_errors import (
    build_error_rows,
    collect_predictions,
    infer_seed,
    summarize_errors,
    top_error_rows,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pose_feature(path: Path, quality_values: list[float], valid_frame_ratio: float) -> None:
    feature_names = np.asarray(
        [
            "quality_pose_detected_ratio",
            "quality_valid_lower_body_ratio",
            "quality_bottom_frame_ratio",
        ]
    )
    np.savez_compressed(
        path,
        video_feature=np.asarray(quality_values, dtype=np.float32),
        feature_names=feature_names,
        valid_frame_ratio=np.asarray([valid_frame_ratio], dtype=np.float32),
    )


class AnalyzeClassifierErrorsTests(unittest.TestCase):
    def test_infer_seed_from_prediction_filename(self) -> None:
        self.assertEqual(infer_seed(Path("combined_seed12_predictions.csv")), "12")

    def test_collect_predictions_joins_view_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            predictions_dir = root / "predictions"
            pose_dir = root / "pose_features"
            view_path = root / "view_metadata.csv"

            write_csv(
                predictions_dir / "combined_seed1_predictions.csv",
                [
                    "label_mode",
                    "split",
                    "video_id",
                    "label",
                    "probability",
                    "fixed_0_5_prediction",
                    "selected_threshold_prediction",
                    "selected_threshold",
                ],
                [
                    {
                        "label_mode": "combined",
                        "split": "test",
                        "video_id": "a",
                        "label": 0,
                        "probability": 0.8,
                        "fixed_0_5_prediction": 1,
                        "selected_threshold_prediction": 1,
                        "selected_threshold": 0.6,
                    }
                ],
            )
            write_csv(
                view_path,
                ["split", "video_id", "view_type", "view_confidence"],
                [{"split": "test", "video_id": "a", "view_type": "rear", "view_confidence": 0.75}],
            )
            (pose_dir / "test").mkdir(parents=True)
            write_pose_feature(pose_dir / "test" / "a.npz", [0.9, 0.8, 0.3], valid_frame_ratio=0.7)

            rows = collect_predictions(
                predictions_dir=predictions_dir,
                view_metadata_path=view_path,
                pose_feature_dir=pose_dir,
                split="test",
                threshold_kind="selected_threshold",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].seed, "1")
            self.assertEqual(rows[0].view_type, "rear")
            self.assertAlmostEqual(rows[0].quality_valid_lower_body_ratio, 0.8, places=6)
            self.assertAlmostEqual(rows[0].valid_frame_ratio, 0.7, places=6)

    def test_build_errors_and_top_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = self._fixture_predictions(Path(tmp_dir))
            errors = build_error_rows(rows)

            self.assertEqual([row.error_type for row in errors], ["false_positive", "false_negative"])
            self.assertAlmostEqual(errors[0].confidence_margin, 0.2, places=6)
            self.assertAlmostEqual(errors[1].confidence_margin, 0.3, places=6)

            top = top_error_rows(errors, top_n=1)

            self.assertEqual(len(top), 2)
            self.assertEqual({row.error_type for row in top}, {"false_positive", "false_negative"})

    def test_summarize_errors_uses_label_specific_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = self._fixture_predictions(Path(tmp_dir))
            errors = build_error_rows(rows)

            summary = summarize_errors(rows, errors)
            all_fp = next(
                row
                for row in summary
                if row["label_mode"] == "combined" and row["view_type"] == "ALL" and row["error_type"] == "false_positive"
            )
            all_fn = next(
                row
                for row in summary
                if row["label_mode"] == "combined" and row["view_type"] == "ALL" and row["error_type"] == "false_negative"
            )

            self.assertEqual(all_fp["eligible_count"], 2.0)
            self.assertEqual(all_fp["n_errors"], 1.0)
            self.assertEqual(all_fp["error_rate"], 0.5)
            self.assertEqual(all_fn["eligible_count"], 2.0)
            self.assertEqual(all_fn["n_errors"], 1.0)
            self.assertEqual(all_fn["error_rate"], 0.5)

    def _fixture_predictions(self, root: Path):
        predictions_dir = root / "predictions"
        pose_dir = root / "pose_features"
        view_path = root / "view_metadata.csv"
        prediction_rows = [
            ("a", 0, 0.8, 1),
            ("b", 1, 0.3, 0),
            ("c", 0, 0.2, 0),
            ("d", 1, 0.9, 1),
        ]
        write_csv(
            predictions_dir / "combined_seed1_predictions.csv",
            [
                "label_mode",
                "split",
                "video_id",
                "label",
                "probability",
                "fixed_0_5_prediction",
                "selected_threshold_prediction",
                "selected_threshold",
            ],
            [
                {
                    "label_mode": "combined",
                    "split": "test",
                    "video_id": video_id,
                    "label": label,
                    "probability": probability,
                    "fixed_0_5_prediction": prediction,
                    "selected_threshold_prediction": prediction,
                    "selected_threshold": 0.6,
                }
                for video_id, label, probability, prediction in prediction_rows
            ],
        )
        write_csv(
            view_path,
            ["split", "video_id", "view_type", "view_confidence"],
            [
                {"split": "test", "video_id": video_id, "view_type": "rear", "view_confidence": 0.8}
                for video_id, *_ in prediction_rows
            ],
        )
        (pose_dir / "test").mkdir(parents=True)
        for index, (video_id, *_rest) in enumerate(prediction_rows):
            write_pose_feature(
                pose_dir / "test" / f"{video_id}.npz",
                [1.0, 0.5 + index * 0.1, 0.25],
                valid_frame_ratio=0.4 + index * 0.1,
            )

        return collect_predictions(
            predictions_dir=predictions_dir,
            view_metadata_path=view_path,
            pose_feature_dir=pose_dir,
            split="test",
            threshold_kind="selected_threshold",
        )


if __name__ == "__main__":
    unittest.main()
