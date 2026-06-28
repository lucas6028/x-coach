"""Unit tests for the EgoExo-Fitness E3 agreement metrics (verified against hand calcs)."""
from __future__ import annotations

import math
import unittest

from src.egoexo.agreement import (
    krippendorff_alpha,
    mean_abs_pairwise_diff,
    pairwise_agreement,
)


class KrippendorffNominalTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertEqual(krippendorff_alpha([[1, 1], [0, 0], [1, 1]], "nominal"), 1.0)

    def test_one_agreed_one_split_unit_is_zero(self):
        # Hand calc: units [A,A] and [A,B] -> alpha = 0 (see module/derivation).
        self.assertAlmostEqual(krippendorff_alpha([["A", "A"], ["A", "B"]], "nominal"), 0.0, places=9)

    def test_single_rating_units_are_ignored(self):
        # The lone [1] unit carries no agreement info; result equals the perfect pair's alpha.
        self.assertEqual(krippendorff_alpha([[1], [0, 0]], "nominal"), 1.0)

    def test_no_pairable_data_is_nan(self):
        self.assertTrue(math.isnan(krippendorff_alpha([[1], [0]], "nominal")))


class KrippendorffOrdinalTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertEqual(krippendorff_alpha([[1, 1], [5, 5]], "ordinal"), 1.0)

    def test_ordinal_penalizes_far_disagreement_more(self):
        # Same structure, but a 1-vs-3 disagreement should yield lower alpha than 1-vs-2.
        near = krippendorff_alpha([[1, 1], [3, 3], [1, 2]], "ordinal")
        far = krippendorff_alpha([[1, 1], [3, 3], [1, 3]], "ordinal")
        self.assertGreater(near, far)
        self.assertAlmostEqual(near, 0.7778, places=3)
        self.assertAlmostEqual(far, 0.4444, places=3)


class PairwiseTests(unittest.TestCase):
    def test_exact_and_tolerant_agreement(self):
        units = [[3, 3, 4]]  # pairs: (3,3) (3,4) (3,4) -> 1/3 exact, 3/3 within-1
        self.assertAlmostEqual(pairwise_agreement(units, tol=0), 1 / 3, places=9)
        self.assertEqual(pairwise_agreement(units, tol=1), 1.0)

    def test_mean_abs_pairwise_diff(self):
        self.assertAlmostEqual(mean_abs_pairwise_diff([[1, 4]]), 3.0, places=9)

    def test_ignores_singletons(self):
        self.assertTrue(math.isnan(pairwise_agreement([[5]], tol=0)))


if __name__ == "__main__":
    unittest.main()
