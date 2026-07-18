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


# Import movement modules for their registration side effects.
from src.pose.movements import squat  # noqa: E402,F401
