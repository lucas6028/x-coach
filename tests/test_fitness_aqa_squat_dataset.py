"""Video-level squat labels: span -> binary reduction and split disjointness."""

import json
import tempfile
import unittest
from pathlib import Path

from src.fitness_aqa import squat_dataset as sq


class TestSquatDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "Labels").mkdir(parents=True)
        (root / "Splits").mkdir(parents=True)
        json.dump({"v1": [[0.5, 1.0]], "v2": [], "v3": [[2.0, 3.0], [4.0, 5.0]]},
                  open(root / "Labels" / "error_knees_forward.json", "w"))
        json.dump({"v1": [], "v2": [[1.0, 2.0]], "v3": []},
                  open(root / "Labels" / "error_knees_inward.json", "w"))
        for split, keys in (("train", ["v1"]), ("val", ["v2"]), ("test", ["v3"])):
            json.dump(keys, open(root / "Splits" / f"{split}_keys.json", "w"))
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_span_to_binary(self):
        kf = sq.load_binary_labels("knees_forward", self.root)
        self.assertEqual(kf, {"v1": 1, "v2": 0, "v3": 1})

    def test_combined_is_union(self):
        comb = sq.load_combined_labels(self.root)
        self.assertEqual(comb["v1"], 1)  # forward only
        self.assertEqual(comb["v2"], 1)  # inward only
        self.assertEqual(comb["v3"], 1)  # forward only

    def test_combined_zero_when_both_clean(self):
        json.dump({"vx": []}, open(self.root / "Labels" / "error_knees_forward.json", "w"))
        json.dump({"vx": []}, open(self.root / "Labels" / "error_knees_inward.json", "w"))
        self.assertEqual(sq.load_combined_labels(self.root)["vx"], 0)

    def test_split_of_maps_every_key(self):
        so = sq.split_of(self.root)
        self.assertEqual(so, {"v1": "train", "v2": "val", "v3": "test"})

    def test_all_labels_has_three_faults(self):
        al = sq.all_labels(self.root)
        self.assertEqual(set(al), {"knees_forward", "knees_inward", "combined"})

    def test_unknown_fault_rejected(self):
        with self.assertRaises(ValueError):
            sq.load_spans("knees_sideways", self.root)


if __name__ == "__main__":
    unittest.main()
