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
    Sit-up, Shoulder Bridge, Leg Abduction) -- deterministic, and it puts the validated detector
    first without encoding a UI
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
