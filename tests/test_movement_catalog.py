"""The sixteen-movement catalog, and the invariants that keep it honest against the registry.

``catalog.py`` writes its names out rather than deriving them from the detector modules -- that is
deliberate, and the reason is in its docstring (deriving would re-create the import edge
``registry.py`` refuses for the two unregistered movements). This file is the price of that choice:
without it, the catalog and the registry could drift silently.
"""

from __future__ import annotations

import unittest

from src.pose.movements import catalog, registry


class CatalogContentTests(unittest.TestCase):
    def test_holds_exactly_sixteen_movements(self) -> None:
        """Sixteen designed movements is where the programme closed (registry.py's closing note)."""
        self.assertEqual(len(catalog.CATALOG), 16)

    def test_names_are_unique(self) -> None:
        # A duplicate would silently shorten the effective catalog and give one movement two slots
        # in every menu built from it.
        self.assertEqual(len(set(catalog.CATALOG)), len(catalog.CATALOG))

    def test_groups_partition_the_catalog(self) -> None:
        grouped = catalog.LOWER_BODY + catalog.UPPER_BODY + catalog.CORE + catalog.FULL_BODY
        self.assertEqual(grouped, catalog.CATALOG)


class CatalogVersusRegistryTests(unittest.TestCase):
    def test_every_registered_detector_is_in_the_catalog(self) -> None:
        """The registry's fourteen are a SUBSET of the catalog's sixteen.

        This is the drift guard, and it catches more than an addition: renaming a detector without
        renaming its catalog entry fails here too, which no amount of derivation would have caught.
        """
        registered = {d.name for d in registry.list_detectors()}
        self.assertTrue(
            registered <= set(catalog.CATALOG),
            f"registered but not in the catalog: {sorted(registered - set(catalog.CATALOG))}",
        )

    def test_the_two_unregistered_movements_are_catalog_only(self) -> None:
        """Jumping Jacks and High Knee are plannable but not analysable.

        Their detectors exist and are tested but are deliberately never registered (every rule is
        permanently silent or withdrawn). If one of them ever IS registered, this test fails and
        whoever did it gets to decide what that means for the plan UI's "manual tick only" state.
        """
        registered = {d.name for d in registry.list_detectors()}
        self.assertEqual(set(catalog.CATALOG) - registered, {"Jumping Jacks", "High Knee"})


class CanonicalMovementTests(unittest.TestCase):
    def test_resolves_exact_names(self) -> None:
        self.assertEqual(catalog.canonical_movement("Overhead Press"), "Overhead Press")

    def test_is_case_and_whitespace_insensitive(self) -> None:
        # Matches how the registry resolves a movement, so a name arriving through the plan API and
        # the same name arriving through the analyze API canonicalize identically.
        self.assertEqual(catalog.canonical_movement("  push-UP "), "Push-up")

    def test_unknown_movement_is_none(self) -> None:
        self.assertIsNone(catalog.canonical_movement("Burpee"))

    def test_blank_and_none_are_none_not_errors(self) -> None:
        # Callers are validating user input and want one "not a movement" answer, not two.
        self.assertIsNone(catalog.canonical_movement(""))
        self.assertIsNone(catalog.canonical_movement(None))

    def test_is_catalog_movement_agrees_with_canonical(self) -> None:
        self.assertTrue(catalog.is_catalog_movement("squat"))
        self.assertFalse(catalog.is_catalog_movement("Burpee"))
