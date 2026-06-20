from __future__ import annotations

import unittest

import numpy as np

from src.pose.rtmpose_pose_extraction import (
    LANDMARK_COUNT,
    coco_wholebody_to_mediapipe_landmarks,
    prediction_instances,
    select_primary_instance,
)


class RTMPosePoseExtractionTests(unittest.TestCase):
    def test_coco_wholebody_adapter_outputs_mediapipe_33_landmarks(self) -> None:
        keypoints = np.zeros((133, 2), dtype=np.float32)
        scores = np.ones(133, dtype=np.float32)
        for index in range(133):
            keypoints[index] = [float(index), float(index * 2)]

        keypoints[17] = [40.0, 100.0]
        keypoints[18] = [60.0, 100.0]
        scores[17] = 0.2
        scores[18] = 0.8
        scores[19] = 0.3

        landmarks = coco_wholebody_to_mediapipe_landmarks(
            keypoints,
            scores,
            width=100,
            height=200,
        )

        self.assertEqual(len(landmarks), LANDMARK_COUNT)
        self.assertAlmostEqual(landmarks[23]["x"], 0.11, places=6)
        self.assertAlmostEqual(landmarks[23]["y"], 0.11, places=6)
        self.assertAlmostEqual(landmarks[29]["x"], 0.19, places=6)
        self.assertAlmostEqual(landmarks[29]["visibility"], 0.3, places=6)
        self.assertAlmostEqual(landmarks[31]["x"], 0.56, places=6)
        self.assertAlmostEqual(landmarks[31]["y"], 0.5, places=6)
        self.assertAlmostEqual(landmarks[31]["visibility"], 0.8, places=6)
        self.assertEqual(landmarks[1]["visibility"], 0.0)

    def test_prediction_instances_accepts_batched_inferencer_shape(self) -> None:
        result = {
            "predictions": [
                [
                    {"keypoints": [[0.0, 0.0]], "bbox_score": 0.2},
                    {"keypoints": [[1.0, 1.0]], "bbox_score": 0.9},
                ]
            ]
        }

        instances = prediction_instances(result)
        selected = select_primary_instance(instances)

        self.assertEqual(len(instances), 2)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["bbox_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
