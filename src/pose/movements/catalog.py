"""The canonical movement catalog: every movement this project models, analysable or not.

This is a DIFFERENT list from ``registry.list_detectors()``. The catalog holds every movement
the programme designed; the registry holds the ones a video can be analysed against. Since
2026-09-26 the two contain the same sixteen names -- ``Jumping Jacks`` and ``High Knee`` were
catalog-only until their first live rules shipped (notes/egoexo-silent-rules-full-archive.md) --
but they stay separate lists: features that let a user NAME a movement (training plans) read this
one, features that run a detector read the registry, and a future movement can again be designed
before it is analysable.

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
