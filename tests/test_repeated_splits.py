from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.video.repeated_splits import (
    Fold,
    make_folds,
    stratified_folds,
    verify_folds,
    write_all_folds,
)


def corpus(n: int = 200, positive_rate: float = 0.3) -> tuple[list[str], dict[str, int]]:
    ids = [f"v{index:04d}" for index in range(n)]
    labels = {video_id: int(index < n * positive_rate) for index, video_id in enumerate(ids)}
    return ids, labels


class StratifiedFoldTests(unittest.TestCase):
    def test_folds_partition_the_ids(self) -> None:
        ids, labels = corpus(100)
        folds = stratified_folds(ids, labels, 5, random.Random(0))
        flat = [video_id for fold in folds for video_id in fold]
        self.assertEqual(sorted(flat), sorted(ids))
        self.assertEqual(len(flat), len(set(flat)))

    def test_every_fold_carries_a_similar_positive_rate(self) -> None:
        """A fold that drew no positives makes balanced accuracy undefined on it."""
        ids, labels = corpus(300, positive_rate=0.2)
        folds = stratified_folds(ids, labels, 5, random.Random(1))
        rates = [sum(labels[v] for v in fold) / len(fold) for fold in folds]
        self.assertLess(max(rates) - min(rates), 0.05)

    def test_the_uneven_remainder_does_not_always_land_on_fold_zero(self) -> None:
        """Dealing every label from fold 0 would concentrate remainders there and
        skew that fold's positive rate on small or unbalanced corpora."""
        ids, labels = corpus(23, positive_rate=0.3)
        folds = stratified_folds(ids, labels, 5, random.Random(2))
        sizes = [len(fold) for fold in folds]
        self.assertLessEqual(max(sizes) - min(sizes), 2)


class MakeFoldsTests(unittest.TestCase):
    def test_each_repeat_tests_every_video_exactly_once(self) -> None:
        """This is what makes a repeat an out-of-fold score over the whole corpus."""
        ids, labels = corpus(200)
        folds = make_folds(ids, labels, n_repeats=3, n_folds=5)

        for repeat in (1, 2, 3):
            tested = [v for fold in folds if fold.repeat == repeat for v in fold.test]
            self.assertEqual(sorted(tested), sorted(ids))

    def test_no_video_appears_in_two_splits_of_one_fold(self) -> None:
        ids, labels = corpus(200)
        for fold in make_folds(ids, labels, n_repeats=2, n_folds=5):
            self.assertEqual(set(fold.train) & set(fold.test), set())
            self.assertEqual(set(fold.val) & set(fold.test), set())
            self.assertEqual(set(fold.train) & set(fold.val), set())

    def test_training_size_stays_close_to_the_fixed_split(self) -> None:
        """The arms have to stay comparable to the archived fixed-split numbers, so a
        fold must not train on dramatically less data than the historical 1136."""
        ids, labels = corpus(1623)
        folds = make_folds(ids, labels, n_repeats=1, n_folds=5)
        train_sizes = [len(fold.train) for fold in folds]
        self.assertTrue(all(1050 <= size <= 1180 for size in train_sizes), train_sizes)

    def test_folds_are_reproducible_from_the_seed(self) -> None:
        ids, labels = corpus(200)
        first = make_folds(ids, labels, n_repeats=2, seed=7)
        second = make_folds(ids, labels, n_repeats=2, seed=7)
        self.assertEqual([f.test for f in first], [f.test for f in second])

    def test_adding_a_repeat_leaves_the_earlier_repeats_untouched(self) -> None:
        """Otherwise extending the run silently invalidates the results already
        reported for repeats 1..n."""
        ids, labels = corpus(200)
        three = make_folds(ids, labels, n_repeats=3, seed=7)
        five = make_folds(ids, labels, n_repeats=5, seed=7)
        self.assertEqual([f.test for f in three], [f.test for f in five[: len(three)]])

    def test_an_unlabeled_id_is_refused(self) -> None:
        ids, labels = corpus(50)
        labels.pop(ids[0])
        with self.assertRaises(ValueError):
            make_folds(ids, labels)

    def test_a_single_fold_is_refused(self) -> None:
        ids, labels = corpus(50)
        with self.assertRaises(ValueError):
            make_folds(ids, labels, n_folds=1)


class VerifyFoldsTests(unittest.TestCase):
    def test_a_video_in_train_and_test_is_refused(self) -> None:
        """Leakage inflates every arm equally, so no downstream comparison exposes it."""
        folds = [Fold(repeat=1, fold=0, train=["a", "b"], val=[], test=["b"])]
        with self.assertRaises(ValueError):
            verify_folds(folds, ["a", "b"], n_repeats=1)

    def test_a_repeat_that_never_tests_a_video_is_refused(self) -> None:
        folds = [Fold(repeat=1, fold=0, train=["b"], val=[], test=["a"])]
        with self.assertRaises(ValueError) as ctx:
            verify_folds(folds, ["a", "b"], n_repeats=1)
        self.assertIn("never tests", str(ctx.exception))

    def test_a_repeat_that_tests_a_video_twice_is_refused(self) -> None:
        folds = [
            Fold(repeat=1, fold=0, train=[], val=[], test=["a"]),
            Fold(repeat=1, fold=1, train=[], val=[], test=["a"]),
        ]
        with self.assertRaises(ValueError) as ctx:
            verify_folds(folds, ["a"], n_repeats=1)
        self.assertIn("more than once", str(ctx.exception))


class WriteFoldsTests(unittest.TestCase):
    def test_key_files_land_in_the_layout_the_classifier_reads(self) -> None:
        ids, labels = corpus(100)
        folds = make_folds(ids, labels, n_repeats=1, n_folds=5)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_all_folds(folds, root)

            self.assertEqual(manifest["n_folds"], 5)
            for fold in folds:
                for split_name in ("train", "val", "test"):
                    path = root / fold.name / f"{split_name}_keys.json"
                    self.assertTrue(path.exists())
                    self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), list)
            self.assertTrue((root / "folds.json").exists())


if __name__ == "__main__":
    unittest.main()
