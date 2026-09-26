"""Pytest fixtures shared across the whole ``tests/`` suite (backend + ML).

Autouse, so every test gets a hermetic ``model_catalog`` (see
``backend/app/services/model_catalog.py``): its process-wide cache/dead-marks are reset before AND
after each test, and its background-refresh seam (``_start_refresh``) is stubbed to a no-op so no
test can accidentally spawn a real network thread. With the catalog always "unknown" and nothing
dead-marked, ``model_catalog.is_available`` fails open for every model on every pre-existing test —
which is the whole point of this fixture: it makes this feature's addition invisible to every test
that isn't specifically about it.

The import is wrapped in ``try/except ImportError`` so this file loads even when the (optional)
backend web dependencies aren't installed — ``conftest.py`` is collected before any test file runs,
and a hard import here would break collection for the ML-only pipelines under this same ``tests/``
directory that never touch the backend at all.
"""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_model_catalog():
    try:
        from backend.app.services import model_catalog
    except ImportError:
        yield
        return

    model_catalog._reset()
    with mock.patch.object(model_catalog, "_start_refresh", lambda: None):
        yield
    model_catalog._reset()
