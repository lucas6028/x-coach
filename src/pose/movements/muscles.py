"""The canonical backend muscle table: which muscles each of the sixteen catalog movements trains.

``frontend/src/lib/movementDetail.ts`` is the AUTHORED source of this data -- a person's
transcription, per movement, of its real primary/secondary muscle groups, alongside the prose the
movement detail page renders. Backend features that need to reason about muscle balance (the Lumen
plan agent, answering "這週哪裡沒練到") need the same lists without importing TypeScript or running
a Node build step at request time, so the table is written out here rather than derived -- the same
call ``catalog.py``'s own docstring makes for the sixteen movement names themselves: a written-out
copy is strictly cheaper to keep honest with one drift-guard test than any derivation machinery
would be to keep honest with itself, and this project already accepts that trade for the names.

``tests/test_movement_muscles.py`` is that drift guard: it parses ``movementDetail.ts``'s literal
``MOVEMENT_DETAIL`` and ``Muscle`` union straight off disk (regex over static literals, no Node
required) and asserts this module's ``MOVEMENT_MUSCLES``/``MUSCLES`` agree with them exactly, for
all sixteen movements and all sixteen muscle keys. If you edit muscles on either side, run that test.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.pose.movements.catalog import canonical_movement

# The sixteen muscle keys, in the SAME declaration order as the frontend `Muscle` union
# (frontend/src/lib/movementDetail.ts). Order matters beyond readability: it is the final tiebreak
# in `coverage`'s ranking, which is what makes that ranking stable and language-independent.
MUSCLES: tuple[str, ...] = (
    "shoulders",
    "chest",
    "biceps",
    "triceps",
    "forearms",
    "abs",
    "obliques",
    "upperBack",
    "lats",
    "lowerBack",
    "glutes",
    "quads",
    "hamstrings",
    "hipFlexors",
    "adductors",
    "calves",
)

# Transcribed EXACTLY from `MOVEMENT_DETAIL` in frontend/src/lib/movementDetail.ts -- do not invent
# or "improve" an entry here; edit the frontend file first and re-copy. Keyed by the catalog's
# canonical spelling (src/pose/movements/catalog.py), same sixteen names in the same order.
MOVEMENT_MUSCLES: dict[str, dict[str, tuple[str, ...]]] = {
    "Squat": {
        "primary": ("quads", "glutes"),
        "secondary": ("hamstrings", "lowerBack", "abs", "calves"),
    },
    "Lunge": {
        "primary": ("quads", "glutes"),
        "secondary": ("hamstrings", "abs", "calves"),
    },
    "Deadlift": {
        "primary": ("glutes", "hamstrings"),
        "secondary": ("lowerBack", "upperBack", "lats", "forearms", "quads"),
    },
    "Leg Abduction": {
        "primary": ("glutes",),
        "secondary": ("obliques", "lowerBack", "abs"),
    },
    "Shoulder Bridge": {
        "primary": ("glutes", "hamstrings"),
        "secondary": ("lowerBack", "abs"),
    },
    "Push-up": {
        "primary": ("chest", "triceps"),
        "secondary": ("shoulders", "abs", "obliques"),
    },
    "Overhead Press": {
        "primary": ("shoulders", "triceps"),
        "secondary": ("upperBack", "abs", "lowerBack"),
    },
    "Row": {
        "primary": ("lats", "upperBack"),
        "secondary": ("biceps", "forearms", "lowerBack"),
    },
    "Bicep Curl": {
        "primary": ("biceps",),
        "secondary": ("forearms",),
    },
    "Band Pull Apart": {
        "primary": ("upperBack", "shoulders"),
        "secondary": ("lats",),
    },
    "Arm Abduction": {
        "primary": ("shoulders",),
        "secondary": ("upperBack",),
    },
    "Arm VW": {
        "primary": ("upperBack", "shoulders"),
        "secondary": ("lats",),
    },
    "Sit-up": {
        "primary": ("abs",),
        "secondary": ("obliques", "hipFlexors"),
    },
    "Torso Twist": {
        "primary": ("obliques",),
        "secondary": ("abs", "lowerBack"),
    },
    "Jumping Jacks": {
        "primary": ("shoulders", "calves"),
        "secondary": ("quads", "glutes", "adductors", "abs"),
    },
    "High Knee": {
        "primary": ("hipFlexors", "quads"),
        "secondary": ("calves", "abs", "glutes"),
    },
}

# The major groups a balanced week should hit, checked in THIS order (specs/muscles-brief.md's
# shared vocabulary) when `coverage` builds its gap list. Deliberately not all sixteen -- nobody
# needs "forearms" or "adductors" flagged as a hole in their week.
_GAP_CHECKLIST: tuple[str, ...] = (
    "quads",
    "hamstrings",
    "glutes",
    "chest",
    "upperBack",
    "lats",
    "shoulders",
    "abs",
)


def coverage(movements: Iterable[str]) -> dict[str, list[str]]:
    """The shared-vocabulary coverage summary for a set of plan items' movements.

    For each movement (after canonicalizing through ``catalog.canonical_movement`` -- unknown or
    uncanonical names are SKIPPED, not raised on, since a plan may briefly hold a name the model
    just proposed) every one of its `primary` muscles counts as primary for the whole set, and every
    `secondary` muscle counts as secondary UNLESS the same muscle is already primary somewhere in
    the set (a muscle that is primary for one movement and secondary for another is primary for the
    week -- "otherwise secondary" in the brief's wording).

    Both `primary` and `secondary` are ranked by (times seen as primary desc, then `MUSCLES`' own
    declaration order) -- ties broken by a fixed, language-independent order rather than dict
    iteration order, so the same plan always renders the same way.

    The times-seen-as-SECONDARY count is deliberately NOT a tiebreak: it let a stabiliser outrank a
    prime mover. A day of Squat + Push-up + Sit-up drives quads, glutes, chest, triceps and abs once
    each, but abs is also secondary twice -- counting that put abs at the head of a list that answers
    "what drives this session". Kept identical to `frontend/src/lib/planMuscles.ts`'s `rank`.

    `gaps` lists the `_GAP_CHECKLIST` groups that are neither primary nor secondary anywhere in the
    set, in checklist order.

    Returns ``{"primary": [...], "secondary": [...], "gaps": [...]}``, muscle KEYS only (e.g.
    "quads") -- no counts, no prose. Empty input (or input that resolves to no known movements)
    returns all three lists empty except `gaps`, which lists the full checklist.
    """
    times_primary: dict[str, int] = dict.fromkeys(MUSCLES, 0)
    times_secondary: dict[str, int] = dict.fromkeys(MUSCLES, 0)

    for raw in movements:
        name = canonical_movement(raw)
        if name is None:
            continue
        entry = MOVEMENT_MUSCLES.get(name)
        if entry is None:
            continue
        for muscle in entry["primary"]:
            times_primary[muscle] += 1
        for muscle in entry["secondary"]:
            times_secondary[muscle] += 1

    def _rank(muscle: str) -> tuple[int, int]:
        return (-times_primary[muscle], MUSCLES.index(muscle))

    primary = sorted((m for m in MUSCLES if times_primary[m] > 0), key=_rank)
    secondary = sorted(
        (m for m in MUSCLES if times_primary[m] == 0 and times_secondary[m] > 0), key=_rank
    )
    gaps = [m for m in _GAP_CHECKLIST if times_primary[m] == 0 and times_secondary[m] == 0]

    return {"primary": primary, "secondary": secondary, "gaps": gaps}
