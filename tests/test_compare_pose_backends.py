from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_pose_backends import classifier_metric_rows, markdown_table, rule_metric_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ComparePoseBackendsTests(unittest.TestCase):
    def test_classifier_metric_rows_average_test_selected_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "experiment_summary.csv"
            fieldnames = [
                "label_mode",
                "seed",
                "split",
                "threshold_kind",
                "balanced_accuracy",
                "macro_f1",
                "recall",
                "specificity",
                "f1",
            ]
            write_csv(
                path,
                fieldnames,
                [
                    {
                        "label_mode": "combined",
                        "seed": 1,
                        "split": "test",
                        "threshold_kind": "selected_threshold",
                        "balanced_accuracy": 0.6,
                        "macro_f1": 0.5,
                        "recall": 0.7,
                        "specificity": 0.5,
                        "f1": 0.8,
                    },
                    {
                        "label_mode": "combined",
                        "seed": 2,
                        "split": "test",
                        "threshold_kind": "selected_threshold",
                        "balanced_accuracy": 0.8,
                        "macro_f1": 0.7,
                        "recall": 0.9,
                        "specificity": 0.7,
                        "f1": 0.6,
                    },
                    {
                        "label_mode": "combined",
                        "seed": 1,
                        "split": "val",
                        "threshold_kind": "selected_threshold",
                        "balanced_accuracy": 1.0,
                        "macro_f1": 1.0,
                        "recall": 1.0,
                        "specificity": 1.0,
                        "f1": 1.0,
                    },
                ],
            )

            rows = classifier_metric_rows("mmpose", path)
            balanced_accuracy = next(row for row in rows if row.metric == "balanced_accuracy")
            balanced_accuracy_std = next(row for row in rows if row.metric == "balanced_accuracy_std")

            self.assertAlmostEqual(balanced_accuracy.value, 0.7, places=6)
            self.assertAlmostEqual(balanced_accuracy_std.value, 0.1, places=6)
            self.assertEqual(balanced_accuracy.detail, "mean over 2 seed(s)")

    def test_rule_metric_rows_reads_only_all_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rule_metrics.csv"
            fieldnames = [
                "class_id",
                "view_type",
                "n",
                "precision",
                "recall",
                "f1",
                "specificity",
                "balanced_accuracy",
                "mean_segment_iou",
            ]
            write_csv(
                path,
                fieldnames,
                [
                    {
                        "class_id": "knees_inward",
                        "view_type": "ALL",
                        "n": 10,
                        "precision": 0.4,
                        "recall": 0.5,
                        "f1": 0.45,
                        "specificity": 0.6,
                        "balanced_accuracy": 0.55,
                        "mean_segment_iou": 0.2,
                    },
                    {
                        "class_id": "knees_inward",
                        "view_type": "rear",
                        "n": 5,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "specificity": 1.0,
                        "balanced_accuracy": 1.0,
                        "mean_segment_iou": 1.0,
                    },
                ],
            )

            rows = rule_metric_rows("mediapipe", path)
            metrics = {row.metric: row.value for row in rows}

            self.assertEqual({row.label for row in rows}, {"knees_inward"})
            self.assertAlmostEqual(metrics["balanced_accuracy"], 0.55, places=6)
            self.assertNotIn(1.0, metrics.values())

    def test_markdown_table_pivots_backends_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "experiment_summary.csv"
            write_csv(
                path,
                [
                    "label_mode",
                    "seed",
                    "split",
                    "threshold_kind",
                    "balanced_accuracy",
                    "macro_f1",
                    "recall",
                    "specificity",
                    "f1",
                ],
                [
                    {
                        "label_mode": "combined",
                        "seed": 1,
                        "split": "test",
                        "threshold_kind": "selected_threshold",
                        "balanced_accuracy": 0.6,
                        "macro_f1": 0.0,
                        "recall": 0.0,
                        "specificity": 0.0,
                        "f1": 0.0,
                    }
                ],
            )
            rows = classifier_metric_rows("mediapipe", path)
            rows += [
                type(rows[0])(
                    backend="mmpose",
                    metric_group="classifier",
                    label="combined",
                    metric="balanced_accuracy",
                    value=0.7,
                    detail="",
                    source="",
                )
            ]

            markdown = markdown_table(rows, left_backend="mediapipe", right_backend="mmpose")

            self.assertIn("| classifier | combined | balanced_accuracy | 0.6000 | 0.7000 | 0.1000 |", markdown)


if __name__ == "__main__":
    unittest.main()
