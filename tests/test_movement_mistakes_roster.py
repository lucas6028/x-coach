"""The frontend's common-mistakes roster, checked against the detectors it claims to mirror.

``frontend/src/lib/movementMistakes.ts`` lists, per movement, the faults that movement's rule
detector can report -- fault_id and kg_query -- plus authored prose for each. It has to write the
list out: the browser cannot import ``src/pose/movements``. This file is the price of that, and it
is the same bargain ``catalog.py`` + ``test_movement_catalog.py`` already struck for the sixteen
movement names.

Without it the two drift silently and in the worst direction: a rule added to a detector simply
never appears on the page, and a rule renamed or re-pointed at a different KG node leaves a card
whose "causes / risks / cues" expansion quietly retrieves the wrong concept. Both are invisible in
the browser and both fail here.

The detector side is read by AST rather than by importing and running anything, because the
metadata lives in ``build_detection(...)`` keyword arguments inside rule function bodies -- the
detector dataclass carries its rules as bare callables, so there is nothing to enumerate at
runtime short of feeding each detector a video.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MOVEMENTS_DIR = REPO_ROOT / "src" / "pose" / "movements"
ROSTER_TS = REPO_ROOT / "frontend" / "src" / "lib" / "movementMistakes.ts"

# Movement name -> detector module, in the registration order at the bottom of registry.py. Written
# out for the same reason catalog.py writes its names out: importing the modules to derive this
# would re-create the import edge registry.py refuses for the two unregistered movements.
REGISTERED: tuple[tuple[str, str], ...] = (
    ("Squat", "squat"),
    ("Overhead Press", "overhead_press"),
    ("Push-up", "pushup"),
    ("Lunge", "lunge"),
    ("Deadlift", "deadlift"),
    ("Row", "row"),
    ("Band Pull Apart", "band_pull_apart"),
    ("Bicep Curl", "bicep_curl"),
    ("Arm Abduction", "arm_abduction"),
    ("Arm VW", "arm_vw"),
    ("Sit-up", "situp"),
    ("Shoulder Bridge", "shoulder_bridge"),
    ("Leg Abduction", "leg_abduction"),
    ("Torso Twist", "torso_twist"),
)


def _detector_roster(module: str) -> list[tuple[str, str]]:
    """Every (fault_id, kg_query) a module's rules can build, in source order.

    A fault built at more than one call site (squat's knees_forward has two, one per view) counts
    once, at its first appearance -- the page shows one card per fault, not one per branch.
    """
    tree = ast.parse((MOVEMENTS_DIR / f"{module}.py").read_text(encoding="utf-8"))

    # Module-level string constants, so `kg_query=LUNGE_DEPTH_KG_QUERY` resolves to its value.
    constants: dict[str, str] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                constants[target.id] = node.value.value

    def literal(node: ast.expr | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        return None

    roster: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "build_detection":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        fault_id = literal(keywords.get("fault_id"))
        kg_query = literal(keywords.get("kg_query"))
        if fault_id is None or kg_query is None:
            raise AssertionError(
                f"{module}.py line {node.lineno}: build_detection has a fault_id or kg_query this "
                "test cannot resolve to a string literal. Keep both a literal or a module-level "
                "constant, or this guard goes blind on that rule."
            )
        if fault_id not in seen:
            seen.add(fault_id)
            roster.append((fault_id, kg_query))
    return roster


def _authored_roster() -> dict[str, list[tuple[str, str]]]:
    """Parse the TS module's (movement -> [(id, kgQuery)]) table.

    A regex, not a parser, and that is why the TS file keeps one rigid shape: every entry is a
    `mistake("<id>", "<kgQuery>",` call opening a movement's array literal. If someone reformats
    that file into a shape this cannot read, the counts stop matching and the test says so -- it
    does not silently pass on an empty parse (see test_parse_found_every_movement).
    """
    source = ROSTER_TS.read_text(encoding="utf-8")
    body = source[source.index("export const MOVEMENT_MISTAKES") :]

    roster: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    # A movement key opens an array: `Squat: [` or `"Overhead Press": [`.
    key_pattern = re.compile(r'^\s{2}(?:"([^"]+)"|([A-Za-z][A-Za-z0-9]*)):\s*\[')
    call_pattern = re.compile(r'^\s*mistake\(\s*$')
    for index, line in enumerate(body.splitlines()):
        key = key_pattern.match(line)
        if key:
            current = key.group(1) or key.group(2)
            roster[current] = []
            continue
        if current and call_pattern.match(line):
            rest = body.splitlines()[index + 1 : index + 3]
            ids = [re.search(r'"([^"]*)"', part) for part in rest]
            if len(ids) != 2 or not all(ids):
                raise AssertionError(f"unparseable mistake(...) call near line {index}")
            roster[current].append((ids[0].group(1), ids[1].group(1)))
    return roster


class RosterParityTests(unittest.TestCase):
    """The page's list IS the detector's list, per movement and in the detector's own order."""

    def setUp(self) -> None:
        self.authored = _authored_roster()

    def test_parse_found_every_movement(self) -> None:
        # Guards the guard: a regex that matched nothing would make every assertion below vacuous.
        self.assertEqual(
            sorted(self.authored), sorted(name for name, _ in REGISTERED),
            "movementMistakes.ts does not list exactly the fourteen registered movements",
        )

    def test_ids_and_kg_queries_match_the_detectors(self) -> None:
        for movement, module in REGISTERED:
            with self.subTest(movement=movement):
                self.assertEqual(
                    self.authored[movement],
                    _detector_roster(module),
                    f"{movement}: movementMistakes.ts and src/pose/movements/{module}.py disagree "
                    "about which faults exist, what they are called, or which KG node they "
                    "retrieve. Update the TS entry (id and kgQuery are copied verbatim).",
                )

    def test_fault_ids_are_globally_unique(self) -> None:
        # The id is the join to a detection, and detections carry no movement of their own on the
        # fault card -- two movements sharing one id would make a card ambiguous.
        seen: dict[str, str] = {}
        for movement, entries in self.authored.items():
            for fault_id, _ in entries:
                self.assertNotIn(
                    fault_id, seen, f"{fault_id} is used by both {seen.get(fault_id)} and {movement}"
                )
                seen[fault_id] = movement

    def test_every_registered_movement_has_at_least_one_mistake(self) -> None:
        # A registered detector with no card would render an "authored nothing" empty state on a
        # movement the app will happily analyse -- the one combination that reads as a bug.
        for movement, _ in REGISTERED:
            with self.subTest(movement=movement):
                self.assertTrue(self.authored[movement])
