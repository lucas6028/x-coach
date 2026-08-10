from __future__ import annotations

from src.pose.movements.base import MovementDetector

_REGISTRY: dict[str, MovementDetector] = {}


def register(detector: MovementDetector) -> None:
    _REGISTRY[detector.name.lower()] = detector


def get_detector(movement: str | None) -> MovementDetector:
    key = (movement or "Squat").lower()
    if key not in _REGISTRY:
        raise KeyError(f"No detector registered for movement {movement!r}")
    return _REGISTRY[key]


def list_detectors() -> list[MovementDetector]:
    """Every registered detector, in registration order.

    Registration order is the import order at the bottom of this module (Squat, Overhead
    Press, Push-up, Lunge, Deadlift, Row, Band Pull Apart, Bicep Curl, Arm Abduction, Arm VW,
    Sit-up, Shoulder Bridge, Leg Abduction, Torso Twist) -- deterministic, and it puts the
    validated detector first without encoding a UI
    preference in the ML layer. Backs GET /api/movements, which is why the frontend needs no
    hand-maintained list of analyzable movements.
    """
    return list(_REGISTRY.values())


# Import movement modules for their registration side effects.
from src.pose.movements import squat  # noqa: E402,F401
from src.pose.movements import overhead_press  # noqa: E402,F401
from src.pose.movements import pushup  # noqa: E402,F401
from src.pose.movements import lunge  # noqa: E402,F401
from src.pose.movements import deadlift  # noqa: E402,F401
from src.pose.movements import row  # noqa: E402,F401
from src.pose.movements import band_pull_apart  # noqa: E402,F401
from src.pose.movements import bicep_curl  # noqa: E402,F401
from src.pose.movements import arm_abduction  # noqa: E402,F401
from src.pose.movements import arm_vw  # noqa: E402,F401
from src.pose.movements import situp  # noqa: E402,F401
from src.pose.movements import shoulder_bridge  # noqa: E402,F401
from src.pose.movements import leg_abduction  # noqa: E402,F401
from src.pose.movements import torso_twist  # noqa: E402,F401
# NO `jumping_jacks` IMPORT, AND THAT IS DELIBERATE. `src/pose/movements/jumping_jacks.py` exists,
# is tested, and defines `JUMPING_JACKS_DETECTOR` -- but every one of its rules is permanently
# silent or withdrawn, so registering it would offer users an analysis that can never report a
# fault. See that module's closing block; design spec
# docs/superpowers/specs/2026-08-10-jumping-jacks-detector-design.md section 7.4.
#
# NO `high_knee` IMPORT EITHER, FOR THE SAME REASON AND ON THE SAME EVIDENCE STANDARD.
# `src/pose/movements/high_knee.py` exists, is tested, and defines `HIGH_KNEE_DETECTOR` -- one rule
# permanently silent, four withdrawn. Two of the four withdrawals are measurements rather than
# arguments: the trunk rules' reference axis (the support limb) sits 6.4-14.2 deg off the trunk
# during normal marching, against thresholds of 10-15 deg, and three SIMULTANEOUS cameras disagree
# about pelvic obliquity by 1.9-12.9 deg against a 5-8 deg threshold. See that module's closing
# block; design spec docs/superpowers/specs/2026-08-10-high-knee-detector-design.md section 7.
#
# That makes 16 movements designed and 14 registered, which is where the programme closes.
