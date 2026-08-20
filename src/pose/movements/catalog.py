"""The canonical movement catalog: every movement this project models, analysable or not.

This is a DIFFERENT list from ``registry.list_detectors()`` and the difference is the point.
The registry holds the fourteen movements a video can be analysed against; the catalog holds all
sixteen the programme designed, including ``Jumping Jacks`` and ``High Knee``, whose detectors
exist and are tested but are deliberately never registered (every rule of theirs is permanently
silent or withdrawn -- see the closing block of ``registry.py``).

Features that let a user NAME a movement without analysing one -- training plans being the first
-- need the sixteen. Features that run a detector need the fourteen. Asking the registry for the
sixteen would mean importing the two unregistered modules, which is exactly the import edge
``registry.py`` refuses: registering is a side effect of importing there, so a future edit that
adds a ``registry.register(...)`` call inside ``jumping_jacks.py`` would silently make it
analysable through this module's back door.

So the names are written out rather than derived, and ``tests/test_movement_catalog.py`` is what
keeps the two lists honest: it asserts the catalog is exactly sixteen names and that every
registered detector's name appears in it. That check also catches a RENAMED registry entry, which
no amount of derivation would.

Order is body region then the registry's own order within a region, matching how
``frontend/src/lib/movements.ts`` groups the same sixteen for the movement menu.
"""

from __future__ import annotations

LOWER_BODY: tuple[str, ...] = (
    "Squat",
    "Lunge",
    "Deadlift",
    "Leg Abduction",
    "Shoulder Bridge",
)

UPPER_BODY: tuple[str, ...] = (
    "Push-up",
    "Overhead Press",
    "Row",
    "Bicep Curl",
    "Band Pull Apart",
    "Arm Abduction",
    "Arm VW",
)

CORE: tuple[str, ...] = ("Sit-up", "Torso Twist")

FULL_BODY: tuple[str, ...] = ("Jumping Jacks", "High Knee")

CATALOG: tuple[str, ...] = LOWER_BODY + UPPER_BODY + CORE + FULL_BODY

# Case-insensitive lookup to the canonical spelling, built once. Mirrors how the registry resolves
# a movement (``registry.get_detector`` lowercases its key), so a plan item posted as "push-up"
# canonicalizes to "Push-up" here exactly as an analysis request would there -- one spelling ends
# up stored, whichever surface the name arrived through.
_BY_LOWER: dict[str, str] = {name.lower(): name for name in CATALOG}


def canonical_movement(movement: str | None) -> str | None:
    """Return the catalog's spelling of ``movement``, or ``None`` if it is not a catalog movement.

    ``None``/blank input answers ``None`` rather than raising: callers are validating user-supplied
    input and want one "not a movement" answer, not two.
    """
    if not movement:
        return None
    return _BY_LOWER.get(movement.strip().lower())


def is_catalog_movement(movement: str | None) -> bool:
    """Whether ``movement`` names a catalog movement (case- and whitespace-insensitive)."""
    return canonical_movement(movement) is not None
