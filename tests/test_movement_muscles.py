"""Drift guard for ``src/pose/movements/muscles.py`` against its frontend source of truth.

``frontend/src/lib/movementDetail.ts`` is the AUTHORED table (see ``muscles.py``'s module
docstring for why the backend keeps its own written-out copy instead of importing TypeScript).
This file parses that source file straight off disk -- regex over its literal object, no Node
required, matching how ``muscles.py`` describes the guard -- and asserts the two tables agree,
muscle group by muscle group, for all sixteen movements and all sixteen ``Muscle`` keys.

The second half of the file unit-tests ``coverage()`` itself: ranking, the primary-beats-secondary
rule, the gap checklist, unknown-movement skipping, and the empty-input case.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.pose.movements import muscles

REPO_ROOT = Path(__file__).resolve().parents[1]
MOVEMENT_DETAIL_TS = REPO_ROOT / "frontend" / "src" / "lib" / "movementDetail.ts"


def _parse_muscle_union(text: str) -> list[str]:
    """The sixteen quoted values of the ``Muscle`` union type, in declaration order."""
    match = re.search(r'export type Muscle =\s*((?:\|\s*"[^"]+"\s*)+);', text)
    if not match:
        raise AssertionError("could not find `export type Muscle = ...;` in movementDetail.ts")
    return re.findall(r'"([^"]+)"', match.group(1))


def _parse_movement_detail(text: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """Every movement's ``primary``/``secondary`` arrays, keyed by its catalog name.

    Isolates the ``MOVEMENT_DETAIL`` object literal (between its own ``export const`` and the next
    one, ``export const movementDetail =``, which immediately follows it in the file), then walks
    the object's top-level keys -- each is either a bare identifier (``Squat:``) or a quoted string
    (``"Leg Abduction":``) at 2-space indent, immediately followed by ``{`` -- and pulls the
    ``primary``/``secondary`` array literal out of the text between one key and the next.
    """
    start = text.index("export const MOVEMENT_DETAIL")
    end = text.index("export const movementDetail =")
    block = text[start:end]

    key_pattern = re.compile(r'^  (?:"([^"]+)"|([A-Za-z][\w-]*)):\s*\{', re.MULTILINE)
    keys = list(key_pattern.finditer(block))
    if not keys:
        raise AssertionError("could not find any movement keys in MOVEMENT_DETAIL")
    bounds = [m.start() for m in keys] + [len(block)]

    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for i, key_match in enumerate(keys):
        name = key_match.group(1) or key_match.group(2)
        chunk = block[key_match.end() : bounds[i + 1]]
        primary_match = re.search(r"primary:\s*\[([^\]]*)\]", chunk)
        secondary_match = re.search(r"secondary:\s*\[([^\]]*)\]", chunk)
        if primary_match is None or secondary_match is None:
            raise AssertionError(f"movement {name!r} is missing a primary/secondary array")
        out[name] = {
            "primary": tuple(re.findall(r'"([^"]+)"', primary_match.group(1))),
            "secondary": tuple(re.findall(r'"([^"]+)"', secondary_match.group(1))),
        }
    return out


class DriftGuardTests(unittest.TestCase):
    """These fail the moment the Python table and the TypeScript source disagree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts_text = MOVEMENT_DETAIL_TS.read_text(encoding="utf-8")
        cls.ts_muscles = _parse_muscle_union(cls.ts_text)
        cls.ts_movements = _parse_movement_detail(cls.ts_text)

    def test_parsed_sixteen_muscle_keys_from_the_frontend(self) -> None:
        # A sanity check on the parser itself, so a regex that silently returns [] cannot make the
        # comparisons below vacuously pass.
        self.assertEqual(len(self.ts_muscles), 16)

    def test_parsed_sixteen_movements_from_the_frontend(self) -> None:
        self.assertEqual(len(self.ts_movements), 16)

    def test_muscles_tuple_matches_the_frontend_union_exactly(self) -> None:
        self.assertEqual(list(muscles.MUSCLES), self.ts_muscles)

    def _assert_agrees_with_frontend(self, table: dict) -> None:
        """The whole comparison, in one place so the corruption test below can re-run THIS and not
        a hand-written echo of it. A guard whose falsification test asserts something weaker than
        the guard itself proves nothing about the guard."""
        self.assertEqual(set(table), set(self.ts_movements))
        for name, ts_entry in self.ts_movements.items():
            with self.subTest(movement=name):
                self.assertEqual(table[name]["primary"], ts_entry["primary"])
                self.assertEqual(table[name]["secondary"], ts_entry["secondary"])

    def test_every_movement_muscles_matches_the_frontend_table(self) -> None:
        self._assert_agrees_with_frontend(muscles.MOVEMENT_MUSCLES)

    def test_a_corrupted_entry_makes_this_guard_fail(self) -> None:
        # Proves the guard has teeth by running the REAL comparison against a corrupted copy of the
        # table (the module object and the source file are both left alone) and requiring it to
        # raise. Without this, a comparison that silently degraded to a no-op would still "pass".
        corrupted = {name: dict(entry) for name, entry in muscles.MOVEMENT_MUSCLES.items()}
        corrupted["Squat"] = {"primary": ("chest",), "secondary": ()}
        with self.assertRaises(AssertionError):
            self._assert_agrees_with_frontend(corrupted)

    def test_a_missing_movement_makes_this_guard_fail(self) -> None:
        # The other way the table can drift: the frontend gains a movement the backend never got.
        short = {name: entry for name, entry in muscles.MOVEMENT_MUSCLES.items() if name != "Row"}
        with self.assertRaises(AssertionError):
            self._assert_agrees_with_frontend(short)


# ------------------------------------------------------------------------------------- coverage()


class CoverageTests(unittest.TestCase):
    def test_empty_input_returns_the_full_gap_checklist(self) -> None:
        result = muscles.coverage([])
        self.assertEqual(result, {"primary": [], "secondary": [], "gaps": list(muscles._GAP_CHECKLIST)})

    def test_single_movement_primary_and_secondary_split(self) -> None:
        result = muscles.coverage(["Squat"])
        self.assertEqual(set(result["primary"]), {"quads", "glutes"})
        self.assertEqual(set(result["secondary"]), {"hamstrings", "lowerBack", "abs", "calves"})

    def test_primary_beats_secondary_when_a_muscle_is_both(self) -> None:
        # Deadlift's secondary includes quads; Squat/Lunge make quads primary elsewhere in the set.
        # A muscle that is primary anywhere in the set must never also appear in `secondary`.
        result = muscles.coverage(["Squat", "Deadlift"])
        self.assertIn("quads", result["primary"])
        self.assertNotIn("quads", result["secondary"])

    def test_ranking_is_times_primary_desc_then_declaration_order(self) -> None:
        # Squat + Lunge: quads and glutes are BOTH primary twice each -- tied on times_primary at 2,
        # so the tiebreak is MUSCLES' declaration order: glutes (index 10) sorts before quads (11).
        result = muscles.coverage(["Squat", "Lunge"])
        self.assertEqual(result["primary"], ["glutes", "quads"])
        # secondary: nothing here is ever a prime mover, so all four tie at zero and fall straight
        # to declaration order -- abs (5), lowerBack (9), hamstrings (12), calves (15). How often
        # each merely SUPPORTS is deliberately not a tiebreak.
        self.assertEqual(result["secondary"], ["abs", "lowerBack", "hamstrings", "calves"])

    def test_a_supporting_count_never_promotes_a_muscle_over_an_equally_driven_one(self) -> None:
        # Squat + Push-up + Sit-up drive quads, glutes, chest, triceps and abs once EACH, and abs is
        # additionally a supporting muscle twice (squat, push-up). Counting that used to put abs at
        # the head of the list. The frontend's `rank` pins the identical order.
        result = muscles.coverage(["Squat", "Push-up", "Sit-up"])
        self.assertEqual(result["primary"], ["chest", "triceps", "abs", "glutes", "quads"])

    def test_gap_list_is_checklist_order_not_muscles_order(self) -> None:
        # A single-movement plan trains almost nothing on the checklist; whatever remains must come
        # back in _GAP_CHECKLIST's own order (quads, hamstrings, glutes, chest, upperBack, lats,
        # shoulders, abs), not MUSCLES' declaration order.
        result = muscles.coverage(["Bicep Curl"])  # primary biceps, secondary forearms -- neither
        # is on the checklist, so every checklist entry is a gap, in checklist order.
        self.assertEqual(result["gaps"], list(muscles._GAP_CHECKLIST))

    def test_gap_list_excludes_covered_groups(self) -> None:
        # Squat (quads/glutes primary; hamstrings/abs secondary) + Push-up (chest primary;
        # shoulders/abs secondary) + Row (lats/upperBack primary) between them touch all eight
        # checklist groups -- proving a secondary-only hit ("gap" means neither primary nor
        # secondary, not "no primary") clears a group same as a primary one.
        result = muscles.coverage(["Squat", "Push-up", "Row"])
        self.assertEqual(result["gaps"], [])

    def test_gap_list_leaves_an_uncovered_group_in(self) -> None:
        # Bicep Curl (primary biceps, secondary forearms) touches none of the eight checklist
        # groups, so adding it alongside Squat+Push-up+Row changes nothing about the gap list --
        # but dropping Row (the only mover of lats/upperBack) reopens exactly those two gaps.
        result = muscles.coverage(["Squat", "Push-up"])
        self.assertEqual(result["gaps"], ["upperBack", "lats"])

    def test_unknown_movement_is_skipped_not_raised_on(self) -> None:
        result = muscles.coverage(["Squat", "Burpee", "", None])  # type: ignore[list-item]
        self.assertEqual(set(result["primary"]), {"quads", "glutes"})

    def test_movement_names_are_canonicalized(self) -> None:
        # Case/whitespace-insensitive, same as catalog.canonical_movement.
        result = muscles.coverage(["  squat ", "PUSH-up"])
        self.assertIn("quads", result["primary"])
        self.assertIn("chest", result["primary"])

    def test_repeated_movements_do_not_change_the_result(self) -> None:
        once = muscles.coverage(["Squat"])
        repeated = muscles.coverage(["Squat", "Squat", "Squat"])
        self.assertEqual(once, repeated)


if __name__ == "__main__":
    unittest.main()
