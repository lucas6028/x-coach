"""Unit tests for the torch-free helpers in src/video/video_level_error_classification.py.

Covers JSON loaders, error-interval labelling, frame->video aggregation, split label
summaries, sample assembly, feature indexing, and the metric utilities (sigmoid,
compute_binary_metrics, find_best_threshold). The model/training paths are out of scope.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.video.video_level_error_classification import (
    Sample,
    aggregate_frame_labels_to_video,
    build_samples,
    compute_binary_metrics,
    find_best_threshold,
    index_feature_paths,
    label_from_error_intervals,
    load_json_list,
    load_json_mapping,
    sigmoid,
    summarize_task_labels,
)


class JsonLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, payload) -> Path:
        path = self.tmp / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_load_json_list_coerces_str(self):
        self.assertEqual(load_json_list(self._write([1, "b"])), ["1", "b"])

    def test_load_json_list_rejects_object(self):
        with self.assertRaises(ValueError):
            load_json_list(self._write({"a": 1}))

    def test_load_json_mapping_coerces_keys(self):
        self.assertEqual(load_json_mapping(self._write({1: "x"})), {"1": "x"})

    def test_load_json_mapping_rejects_list(self):
        with self.assertRaises(ValueError):
            load_json_mapping(self._write([1, 2]))


class LabelFromIntervalsTests(unittest.TestCase):
    def test_non_empty_interval_list_is_positive(self):
        self.assertEqual(label_from_error_intervals([[10, 20]]), 1)

    def test_empty_interval_list_is_negative(self):
        self.assertEqual(label_from_error_intervals([]), 0)

    def test_non_list_is_negative(self):
        self.assertEqual(label_from_error_intervals(None), 0)


class AggregateFrameLabelsTests(unittest.TestCase):
    def test_takes_max_label_per_video(self):
        frame_labels = {"vidA_0": 0, "vidA_1": 1, "vidB_0": 0}
        self.assertEqual(
            aggregate_frame_labels_to_video(frame_labels),
            {"vidA": 1, "vidB": 0},
        )

    def test_bool_values_coerced_to_int(self):
        self.assertEqual(aggregate_frame_labels_to_video({"vidC_0": True}), {"vidC": 1})

    def test_non_numeric_value_treated_as_zero(self):
        self.assertEqual(aggregate_frame_labels_to_video({"vidD_0": "oops"}), {"vidD": 0})


class SummarizeTaskLabelsTests(unittest.TestCase):
    def test_counts_positives_and_negatives_per_split(self):
        split_video_ids = {"train": ["v1", "v2", "v3"], "test": ["v4"]}
        task_labels = {"knees_forward": {"v1": 1, "v2": 0, "v4": 1}}  # v3 has no label
        summary = summarize_task_labels(split_video_ids, task_labels)
        self.assertEqual(
            summary["knees_forward"]["train"],
            {"available_videos": 2, "positives": 1, "negatives": 1},
        )
        self.assertEqual(
            summary["knees_forward"]["test"],
            {"available_videos": 1, "positives": 1, "negatives": 0},
        )


class BuildSamplesTests(unittest.TestCase):
    def test_splits_into_samples_and_missing_buckets(self):
        feature_index = {"v1": Path("f1.npz"), "v3": Path("f3.npz")}
        labels = {"v1": 1, "v2": 0}
        samples, missing_labels, missing_features = build_samples(
            ["v1", "v2", "v3"], labels, feature_index
        )
        self.assertEqual(samples, [Sample(video_id="v1", feature_path=Path("f1.npz"), label=1)])
        self.assertEqual(missing_labels, ["v3"])  # no label entry
        self.assertEqual(missing_features, ["v2"])  # labelled but no feature file


class IndexFeaturePathsTests(unittest.TestCase):
    def test_indexes_npz_by_stem(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "sub").mkdir()
        np.savez(tmp / "x.npz", video_feature=np.zeros(3))
        np.savez(tmp / "sub" / "y.npz", video_feature=np.zeros(3))
        index = index_feature_paths(tmp)
        self.assertEqual(set(index.keys()), {"x", "y"})
        self.assertEqual(index["y"].name, "y.npz")


class SigmoidTests(unittest.TestCase):
    def test_zero_maps_to_half(self):
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5)

    def test_saturates_at_extremes(self):
        self.assertAlmostEqual(float(sigmoid(np.array([100.0]))[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sigmoid(np.array([-100.0]))[0]), 0.0, places=6)


class BinaryMetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        probabilities = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 0, 1, 0])
        metrics = compute_binary_metrics(probabilities, labels, threshold=0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_empty_inputs_return_zeros(self):
        metrics = compute_binary_metrics(np.array([]), np.array([]))
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["threshold"], 0.5)


class FindBestThresholdTests(unittest.TestCase):
    def test_separable_scores_reach_perfect_f1(self):
        probabilities = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        threshold, metrics = find_best_threshold(probabilities, labels)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertLessEqual(threshold, 0.8)

    def test_empty_inputs_default_threshold(self):
        threshold, metrics = find_best_threshold(np.array([]), np.array([]))
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
