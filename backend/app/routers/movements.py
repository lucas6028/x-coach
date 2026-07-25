"""The analyzable-movement catalog, derived from the pose detector registry.

Public, like /api/knowledge/*. The /movements page itself is behind RequireAuth, but /app is
the anonymous public demo and needs this list to render its movement selector and to validate
a ?movement= URL parameter BEFORE enabling the dropzone -- otherwise a hand-typed movement
costs the user a full upload to discover a 400.

The list is derived from src/pose/movements/registry.py rather than restated here, so
registering a fourth detector surfaces it in the UI with no backend or frontend edit.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["movements"])


@router.get("/movements")
def movements() -> dict:
    """Every registered detector: canonical name plus whether its rules are validated against
    labeled data. The frontend renders unvalidated movements with a Beta tag."""
    # Imported lazily: the registry pulls in the detector modules (numpy), and the API layer is
    # tested without the heavy ML stack installed. Matches services/analysis.py's deferred-import
    # rationale.
    from src.pose.movements import registry

    return {
        "movements": [
            {"name": d.name, "validated": d.validated} for d in registry.list_detectors()
        ]
    }
