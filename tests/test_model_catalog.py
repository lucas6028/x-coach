"""Tests for ``backend/app/services/model_catalog.py`` — live LLM model availability.

Covers, in order: the network-parsing seam (``_fetch_catalog``, both OpenRouter's and NVIDIA NIM's
``/models`` shapes), ``refresh()`` success/failure semantics (a failure keeps the previous good ids
and never blocks), TTL/staleness/backoff driving ``_start_refresh`` only when configured, the
base-URL-change invalidation rule, single-flight, dead-mark TTL (and that a catalog refresh never
clears one), fail-open availability + the four ``model_status`` outcomes, the ``settings.py``
getters this module feeds (``available_chat_models`` / ``default_chat_model`` /
``resolve_chat_model`` / ``followup_chat_model`` / ``configured_followup_model``), the transport
fallback through the REAL ``services.chat._stream_raw_chunks`` (410/404-without-tools mark dead and
retry; 404-with-tools does not), the ``done`` frame reporting the effective model, ``/api/health``'s
filtered list and non-blocking guarantee, and the new admin endpoint.

``tests/conftest.py`` resets ``model_catalog`` and stubs ``_start_refresh`` to a no-op before/after
EVERY test in the suite, so a test here that wants the REAL single-flight/background-thread
behaviour must explicitly opt back in via ``_REAL_START_REFRESH`` (captured at import time, before
that fixture ever runs) or by patching ``_start_refresh`` back to it for the duration of the test.
"""

from __future__ import annotations

import json
import threading
import time
import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from backend.app import settings as app_settings
from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.services import chat as chat_service
from backend.app.services import model_catalog, runtime_config, store

# Captured before any test runs (module import happens before pytest collects fixtures), so this is
# genuinely the real implementation regardless of what ``tests/conftest.py`` stubs it to per-test.
_REAL_START_REFRESH = model_catalog._start_refresh


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def _cm_for(status: int | None = None, lines: list[str] | None = None) -> mock.MagicMock:
    """One fake ``httpx.stream(...)`` context manager: either a status failure or a line stream."""
    fake_resp = mock.Mock()
    if status is not None:
        fake_resp.raise_for_status.side_effect = _http_status_error(status)
    else:
        fake_resp.raise_for_status.return_value = None
        fake_resp.iter_lines.return_value = iter(lines or [])
    cm = mock.MagicMock()
    cm.__enter__.return_value = fake_resp
    cm.__exit__.return_value = False
    return cm


# ----------------------------------------------------------------------------------- _fetch_catalog


class FetchCatalogTests(unittest.TestCase):
    def test_parses_openrouter_shape_with_expiration_dates(self) -> None:
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "data": [
                {"id": "deepseek/deepseek-v4-flash", "expiration_date": None},
                {"id": "openai/gpt-oss-120b", "expiration_date": "2026-09-03"},
            ]
        }
        with mock.patch("httpx.get", return_value=fake_resp) as get:
            ids = model_catalog._fetch_catalog("https://openrouter.ai/api/v1")
        self.assertEqual(
            ids,
            {"deepseek/deepseek-v4-flash": None, "openai/gpt-oss-120b": "2026-09-03"},
        )
        args, _ = get.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/models")

    def test_parses_nim_shape_with_no_expiration_field(self) -> None:
        # NIM's schema has no expiration_date at all -- every id maps to None, not an error.
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "data": [{"id": "meta/llama2-70b", "object": "model", "created": 1, "owned_by": "meta"}]
        }
        with mock.patch("httpx.get", return_value=fake_resp):
            ids = model_catalog._fetch_catalog("https://integrate.api.nvidia.com/v1")
        self.assertEqual(ids, {"meta/llama2-70b": None})

    def test_sends_bearer_header_when_a_key_is_configured(self) -> None:
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {"data": []}
        with mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace(llm_api_key="sk-test")
        ), mock.patch("httpx.get", return_value=fake_resp) as get:
            model_catalog._fetch_catalog("https://openrouter.ai/api/v1")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(kwargs["timeout"], 5.0)

    def test_omits_auth_header_when_no_key_is_configured(self) -> None:
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {"data": []}
        with mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace(llm_api_key="")
        ), mock.patch("httpx.get", return_value=fake_resp) as get:
            model_catalog._fetch_catalog("https://x/v1")
        _, kwargs = get.call_args
        self.assertEqual(kwargs["headers"], {})

    def test_raises_on_a_bad_status(self) -> None:
        fake_resp = mock.Mock()
        fake_resp.raise_for_status.side_effect = _http_status_error(503)
        with mock.patch("httpx.get", return_value=fake_resp):
            with self.assertRaises(httpx.HTTPStatusError):
                model_catalog._fetch_catalog("https://p/v1")

    def test_tolerates_a_non_dict_or_missing_data_list(self) -> None:
        for payload in ({"data": "not-a-list"}, {}, {"data": [1, 2, "x", {"no_id": True}]}):
            fake_resp = mock.Mock()
            fake_resp.raise_for_status.return_value = None
            fake_resp.json.return_value = payload
            with mock.patch("httpx.get", return_value=fake_resp):
                self.assertEqual(model_catalog._fetch_catalog("https://p/v1"), {})


# ------------------------------------------------------------------------------------------ refresh


class RefreshTests(unittest.TestCase):
    def test_success_populates_the_cache_and_clears_any_error(self) -> None:
        with mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ), mock.patch.object(
            model_catalog, "_fetch_catalog", return_value={"m/a": None, "m/b": "2026-01-01"}
        ):
            model_catalog.refresh()
            self.assertEqual(model_catalog.catalog_ids(), {"m/a": None, "m/b": "2026-01-01"})
            state = model_catalog.catalog_state()
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["model_count"], 2)
        self.assertIsNone(state["error"])
        self.assertIsNotNone(state["checked_at"])

    def test_failure_keeps_the_previous_good_ids_and_sets_error(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}):
                model_catalog.refresh()
            with mock.patch.object(
                model_catalog, "_fetch_catalog", side_effect=RuntimeError("boom")
            ):
                model_catalog.refresh()  # must not raise
            self.assertEqual(model_catalog.catalog_ids(), {"m/a": None})  # kept
            state = model_catalog.catalog_state()
        self.assertEqual(state["status"], "error")
        self.assertIn("boom", state["error"])

    def test_failure_on_a_brand_new_base_url_has_nothing_of_its_own_to_keep(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://old/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}):
                model_catalog.refresh()
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://new/v1"):
            with mock.patch.object(
                model_catalog, "_fetch_catalog", side_effect=RuntimeError("boom")
            ):
                model_catalog.refresh()
            self.assertIsNone(model_catalog.catalog_ids())

    def test_unresolvable_base_url_sets_an_error_without_raising(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value=None):
            model_catalog.refresh()  # must not raise
            self.assertIsNone(model_catalog.catalog_ids())


# --------------------------------------------------------------------------- staleness / backoff / TTL


class StalenessTests(unittest.TestCase):
    def test_a_fresh_cache_never_triggers_a_refresh(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}), \
                 mock.patch.object(model_catalog, "_now", return_value=1000.0):
                model_catalog.refresh()
            with mock.patch.object(model_catalog, "_now", return_value=1000.0), \
                 mock.patch.object(
                     app_settings, "get_settings",
                     return_value=types.SimpleNamespace(chat_configured=True),
                 ), mock.patch.object(model_catalog, "_start_refresh") as sr:
                ids = model_catalog.catalog_ids()
        self.assertEqual(ids, {"m/a": None})
        sr.assert_not_called()

    def test_stale_cache_triggers_a_refresh_only_when_chat_is_configured(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}), \
                 mock.patch.object(model_catalog, "_now", return_value=1000.0):
                model_catalog.refresh()

            past_ttl = 1000.0 + model_catalog._CATALOG_SUCCESS_TTL_S + 1

            with mock.patch.object(model_catalog, "_now", return_value=past_ttl), \
                 mock.patch.object(
                     app_settings, "get_settings",
                     return_value=types.SimpleNamespace(chat_configured=False),
                 ), mock.patch.object(model_catalog, "_start_refresh") as sr:
                model_catalog.catalog_ids()
            sr.assert_not_called()

            with mock.patch.object(model_catalog, "_now", return_value=past_ttl), \
                 mock.patch.object(
                     app_settings, "get_settings",
                     return_value=types.SimpleNamespace(chat_configured=True),
                 ), mock.patch.object(model_catalog, "_start_refresh") as sr:
                model_catalog.catalog_ids()
            sr.assert_called_once()

    def test_a_missing_catalog_never_fetches_when_chat_is_not_configured(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"), \
             mock.patch.object(
                 app_settings, "get_settings",
                 return_value=types.SimpleNamespace(chat_configured=False),
             ), mock.patch.object(model_catalog, "_start_refresh") as sr:
            self.assertIsNone(model_catalog.catalog_ids())
        sr.assert_not_called()

    def test_a_failed_refresh_backs_off_before_retrying(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(
                model_catalog, "_fetch_catalog", side_effect=RuntimeError("boom")
            ), mock.patch.object(model_catalog, "_now", return_value=1000.0):
                model_catalog.refresh()

            settings_on = types.SimpleNamespace(chat_configured=True)
            with mock.patch.object(model_catalog, "_now", return_value=1000.0 + 60), \
                 mock.patch.object(app_settings, "get_settings", return_value=settings_on), \
                 mock.patch.object(model_catalog, "_start_refresh") as sr:
                model_catalog.catalog_ids()  # inside the 120s failure backoff window
            sr.assert_not_called()

            with mock.patch.object(model_catalog, "_now", return_value=1000.0 + 130), \
                 mock.patch.object(app_settings, "get_settings", return_value=settings_on), \
                 mock.patch.object(model_catalog, "_start_refresh") as sr:
                model_catalog.catalog_ids()  # past it
            sr.assert_called_once()


class BaseUrlChangeTests(unittest.TestCase):
    def test_switching_base_url_treats_the_old_ids_as_absent(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://a/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}):
                model_catalog.refresh()
            self.assertEqual(model_catalog.catalog_ids(), {"m/a": None})

        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://b/v1"), \
             mock.patch.object(model_catalog, "_start_refresh"):
            self.assertIsNone(model_catalog.catalog_ids())
            self.assertEqual(model_catalog.catalog_state()["status"], "unknown")


class SingleFlightTests(unittest.TestCase):
    """Exercises the REAL ``_start_refresh`` (``tests/conftest.py`` stubs it per-test by default)."""

    def test_a_second_call_is_a_no_op_while_one_is_already_running(self) -> None:
        with model_catalog._state_lock:
            model_catalog._refreshing = True
        try:
            with mock.patch("threading.Thread") as thread_cls:
                _REAL_START_REFRESH()
            thread_cls.assert_not_called()
        finally:
            with model_catalog._state_lock:
                model_catalog._refreshing = False

    def test_spawns_a_daemon_thread_targeting_refresh_when_none_is_running(self) -> None:
        with mock.patch("threading.Thread") as thread_cls:
            instance = thread_cls.return_value
            _REAL_START_REFRESH()
        thread_cls.assert_called_once()
        _, kwargs = thread_cls.call_args
        self.assertIs(kwargs["target"], model_catalog.refresh)
        self.assertTrue(kwargs["daemon"])
        instance.start.assert_called_once()
        with model_catalog._state_lock:
            model_catalog._refreshing = False  # tidy up: the mocked Thread never really ran


class NonBlockingTests(unittest.TestCase):
    def test_catalog_ids_never_blocks_even_while_a_refresh_is_slow(self) -> None:
        released = threading.Event()

        def slow_fetch(base_url: str) -> dict[str, str | None]:
            released.wait(timeout=2)
            return {"a/m": None}

        with mock.patch.object(
            model_catalog, "_start_refresh", _REAL_START_REFRESH
        ), mock.patch.object(
            model_catalog, "_fetch_catalog", side_effect=slow_fetch
        ), mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ), mock.patch.object(
            app_settings, "get_settings", return_value=types.SimpleNamespace(chat_configured=True)
        ):
            start = time.monotonic()
            ids = model_catalog.catalog_ids()  # empty cache -> kicks off a background refresh
            elapsed = time.monotonic() - start
        released.set()
        time.sleep(0.05)  # let the background thread finish so it doesn't leak past this test
        self.assertIsNone(ids)  # nothing was cached yet -- the read never waited for the fetch
        self.assertLess(elapsed, 0.5)


# --------------------------------------------------------------------------------------- dead marks


class DeadMarkTests(unittest.TestCase):
    def test_marked_within_ttl_is_reported_unavailable(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"), \
             mock.patch.object(model_catalog, "_now", return_value=1000.0):
            model_catalog.mark_unavailable("dead/model", "HTTP 410")
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"), \
             mock.patch.object(
                 model_catalog, "_now", return_value=1000.0 + model_catalog._DEAD_MARK_TTL_S - 1
             ):
            self.assertTrue(model_catalog.is_marked_unavailable("dead/model"))

    def test_mark_expires_after_its_ttl(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"), \
             mock.patch.object(model_catalog, "_now", return_value=1000.0):
            model_catalog.mark_unavailable("dead/model", "HTTP 410")
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"), \
             mock.patch.object(
                 model_catalog, "_now", return_value=1000.0 + model_catalog._DEAD_MARK_TTL_S + 1
             ):
            self.assertFalse(model_catalog.is_marked_unavailable("dead/model"))

    def test_mark_is_scoped_to_the_base_url_it_was_set_under(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://a/v1"):
            model_catalog.mark_unavailable("m", "HTTP 410")
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://b/v1"):
            self.assertFalse(model_catalog.is_marked_unavailable("m"))

    def test_an_empty_dead_dict_never_resolves_a_base_url(self) -> None:
        # The common case (nothing ever marked dead) must not touch settings at all -- this is what
        # keeps is_available/is_marked_unavailable safe against an incomplete settings stand-in.
        with mock.patch.object(
            model_catalog, "_current_base_url", side_effect=AssertionError("must not be called")
        ):
            self.assertFalse(model_catalog.is_marked_unavailable("anything"))

    def test_a_catalog_refresh_never_clears_a_dead_mark(self) -> None:
        # NIM's own /models listing keeps listing models that 404 -- trusting a refresh to clear a
        # dead mark would silently resurrect exactly the failure this module exists to prevent.
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            model_catalog.mark_unavailable("m/dead", "HTTP 410")
            with mock.patch.object(
                model_catalog, "_fetch_catalog", return_value={"m/dead": None, "m/ok": None}
            ):
                model_catalog.refresh()
            self.assertTrue(model_catalog.is_marked_unavailable("m/dead"))
            self.assertFalse(model_catalog.is_available("m/dead"))  # even though the catalog lists it


# --------------------------------------------------------------------- availability / model_status


class AvailabilityTests(unittest.TestCase):
    def test_fails_open_when_the_catalog_is_unknown(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            self.assertTrue(model_catalog.is_available("anything"))
            self.assertEqual(
                model_catalog.model_status("anything"),
                {"status": "unknown", "expires": None, "detail": None},
            )

    def test_available_when_listed(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(
                model_catalog, "_fetch_catalog", return_value={"m/a": "2027-01-01"}
            ):
                model_catalog.refresh()
            self.assertTrue(model_catalog.is_available("m/a"))
            self.assertEqual(
                model_catalog.model_status("m/a"),
                {"status": "available", "expires": "2027-01-01", "detail": None},
            )

    def test_not_listed_when_the_catalog_is_known_but_lacks_it(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"m/a": None}):
                model_catalog.refresh()
            self.assertFalse(model_catalog.is_available("m/ghost"))
            self.assertEqual(
                model_catalog.model_status("m/ghost"),
                {"status": "not_listed", "expires": None, "detail": None},
            )

    def test_unavailable_when_dead_marked_even_if_the_catalog_still_lists_it(self) -> None:
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(
                model_catalog, "_fetch_catalog", return_value={"m/dead": "2099-01-01"}
            ):
                model_catalog.refresh()
            model_catalog.mark_unavailable("m/dead", "HTTP 410")
            self.assertFalse(model_catalog.is_available("m/dead"))
            self.assertEqual(
                model_catalog.model_status("m/dead"),
                {"status": "unavailable", "expires": "2099-01-01", "detail": "HTTP 410"},
            )

    def test_an_unlisted_routing_suffix_is_judged_by_its_base_id(self) -> None:
        # OpenRouter lists ``:free`` variants as ids of their own but never lists routing suffixes
        # like ``:nitro`` — a working ``m/a:nitro`` must not drop out of the picker.
        catalog = {"m/a": "2027-01-01", "m/b:free": None, "m/c": None}
        with mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value=catalog):
                model_catalog.refresh()
            self.assertTrue(model_catalog.is_available("m/a:nitro"))
            self.assertEqual(
                model_catalog.model_status("m/a:nitro"),
                {"status": "available", "expires": "2027-01-01", "detail": None},
            )
            # A listed-kind variant that is missing is gone, even though its base model is listed.
            self.assertFalse(model_catalog.is_available("m/c:free"))
            self.assertTrue(model_catalog.is_available("m/b:free"))
            # A suffix on a base the catalog doesn't know is not rescued.
            self.assertFalse(model_catalog.is_available("m/ghost:nitro"))
            model_catalog.mark_unavailable("m/a:nitro", "HTTP 404")
            self.assertEqual(
                model_catalog.model_status("m/a:nitro"),
                {"status": "unavailable", "expires": "2027-01-01", "detail": "HTTP 404"},
            )


# ---------------------------------------------------------------------------- settings.py integration


class SettingsIntegrationTests(unittest.TestCase):
    def _overrides(self, mapping: dict) -> mock._patch:
        return mock.patch.object(runtime_config, "get_overrides", return_value=mapping)

    def test_available_chat_models_excludes_dead_and_not_listed(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m", "c/m"]}), mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            with mock.patch.object(model_catalog, "_fetch_catalog", return_value={"b/m": None}):
                model_catalog.refresh()
            self.assertEqual(app_settings.available_chat_models(), ["b/m"])

    def test_available_chat_models_is_never_empty(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m"]}), mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            model_catalog.mark_unavailable("b/m", "HTTP 410")
            self.assertEqual(app_settings.available_chat_models(), ["a/m", "b/m"])

    def test_default_chat_model_skips_a_dead_first_entry(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m"]}), mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            self.assertEqual(app_settings.default_chat_model(), "b/m")

    def test_resolve_chat_model_rejects_a_dead_requested_model(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m"]}), mock.patch.object(
            model_catalog, "_current_base_url", return_value="https://p/v1"
        ):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            self.assertEqual(app_settings.resolve_chat_model("a/m"), "b/m")
            self.assertEqual(app_settings.resolve_chat_model("b/m"), "b/m")
            self.assertEqual(app_settings.resolve_chat_model(None), "b/m")

    def test_followup_chat_model_falls_back_when_the_pin_is_dead(self) -> None:
        with self._overrides(
            {"llm_models": ["a/m", "b/m"], "llm_followup_model": "a/m"}
        ), mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            self.assertEqual(app_settings.followup_chat_model(), "b/m")

    def test_followup_chat_model_uses_the_pin_when_available(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m"], "llm_followup_model": "b/m"}):
            self.assertEqual(app_settings.followup_chat_model(), "b/m")

    def test_configured_followup_model_is_raw_and_unaffected_by_a_dead_mark(self) -> None:
        # Acceptance criterion: the admin edit form must show the CONFIGURED value regardless of
        # live availability -- this is what backs _effective_settings().
        with self._overrides(
            {"llm_models": ["a/m", "b/m"], "llm_followup_model": "a/m"}
        ), mock.patch.object(model_catalog, "_current_base_url", return_value="https://p/v1"):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            self.assertEqual(app_settings.configured_followup_model(), "a/m")
            self.assertEqual(app_settings.followup_chat_model(), "b/m")  # the two now diverge

    def test_configured_followup_model_blank_falls_back_to_the_first_raw_model(self) -> None:
        with self._overrides({"llm_models": ["a/m", "b/m"], "llm_followup_model": "  "}):
            self.assertEqual(app_settings.configured_followup_model(), "a/m")


# ------------------------------------------------------------------------------- transport fallback


class TransportFallbackTests(unittest.TestCase):
    def _patches(self, models: list[str]):
        return mock.patch.object(
            runtime_config,
            "get_overrides",
            return_value={"llm_models": models, "llm_base_url": "https://openrouter.ai/api/v1"},
        )

    def _settings_patch(self):
        return mock.patch.object(
            chat_service, "get_settings", return_value=types.SimpleNamespace(llm_api_key="sk-test")
        )

    def test_410_marks_dead_and_falls_back_to_the_next_model(self) -> None:
        # ``httpx.stream`` is called with the SAME (mutated-in-place) ``json`` dict on every
        # attempt, so ``mock``'s recorded ``call_args_list`` would show the FINAL value for every
        # past call too -- the model actually sent on each attempt is captured live instead.
        cm_dead = _cm_for(status=410)
        cm_ok = _cm_for(lines=['data: {"choices":[{"delta":{"content":"hi"}}]}', "data: [DONE]"])
        responses = iter([cm_dead, cm_ok])
        sent_models: list[str] = []

        def fake_stream(method, url, *, headers, json, timeout):
            sent_models.append(json["model"])
            return next(responses)

        with self._patches(["dead/model", "ok/model"]), self._settings_patch(), mock.patch(
            "httpx.stream", side_effect=fake_stream
        ) as stream:
            chunks = list(
                chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "dead/model")
            )
            self.assertTrue(model_catalog.is_marked_unavailable("dead/model"))
        self.assertEqual(chunks, [{"choices": [{"delta": {"content": "hi"}}]}])
        self.assertEqual(stream.call_count, 2)
        self.assertEqual(sent_models, ["dead/model", "ok/model"])

    def test_404_without_tools_marks_dead_and_falls_back(self) -> None:
        cm_dead = _cm_for(status=404)
        cm_ok = _cm_for(lines=["data: [DONE]"])
        with self._patches(["dead/model", "ok/model"]), self._settings_patch(), mock.patch(
            "httpx.stream", side_effect=[cm_dead, cm_ok]
        ) as stream:
            list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "dead/model"))
            self.assertTrue(model_catalog.is_marked_unavailable("dead/model"))
        self.assertEqual(stream.call_count, 2)

    def test_404_with_tools_raises_and_does_not_mark_dead(self) -> None:
        cm_dead = _cm_for(status=404)
        with self._patches(["maybe/model", "ok/model"]), self._settings_patch(), mock.patch(
            "httpx.stream", return_value=cm_dead
        ) as stream:
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(
                    chat_service._stream_raw_chunks(
                        [{"role": "user", "content": "hi"}],
                        "maybe/model",
                        extra_body={"tools": [{"type": "function"}], "tool_choice": "auto"},
                    )
                )
            self.assertEqual(ctx.exception.status, 404)
            self.assertFalse(model_catalog.is_marked_unavailable("maybe/model"))
        stream.assert_called_once()  # no retry -- a 404 with tools is ambiguous, not a dead signal

    def test_all_candidates_dead_eventually_raises(self) -> None:
        with self._patches(["a/model", "b/model"]), self._settings_patch():
            model_catalog.mark_unavailable("a/model", "HTTP 410")
            model_catalog.mark_unavailable("b/model", "HTTP 410")
            cm_dead = _cm_for(status=410)
            with mock.patch("httpx.stream", return_value=cm_dead):
                with self.assertRaises(chat_service._LLMError):
                    list(
                        chat_service._stream_raw_chunks(
                            [{"role": "user", "content": "hi"}], "a/model"
                        )
                    )

    def test_a_dead_marked_model_is_substituted_before_the_first_request(self) -> None:
        cm_ok = _cm_for(lines=["data: [DONE]"])
        with self._patches(["dead/model", "ok/model"]), self._settings_patch():
            model_catalog.mark_unavailable("dead/model", "HTTP 410")
            with mock.patch("httpx.stream", return_value=cm_ok) as stream:
                list(
                    chat_service._stream_raw_chunks(
                        [{"role": "user", "content": "hi"}], "dead/model"
                    )
                )
            stream.assert_called_once()
            self.assertEqual(stream.call_args.kwargs["json"]["model"], "ok/model")

    def test_a_non_dead_4xx_still_raises_immediately_with_no_retry(self) -> None:
        # A plain 400 (unrelated to model availability) must behave exactly as before this feature:
        # one attempt, no dead-mark, the status preserved on the raised error.
        cm_bad = _cm_for(status=400)
        with self._patches(["m"]), self._settings_patch(), mock.patch(
            "httpx.stream", return_value=cm_bad
        ) as stream:
            with self.assertRaises(chat_service._LLMError) as ctx:
                list(chat_service._stream_raw_chunks([{"role": "user", "content": "hi"}], "m"))
            self.assertFalse(model_catalog.is_marked_unavailable("m"))
        self.assertEqual(ctx.exception.status, 400)
        stream.assert_called_once()


class DoneFrameEffectiveModelTests(unittest.TestCase):
    def _one_round(self, model: str):
        def fake_turn(messages, model, *, timeout=None, tools=None):
            yield "hi"
            yield chat_service._Turn(text="hi", tool_calls=[], finish_reason="stop")

        with mock.patch.object(chat_service, "_stream_turn", side_effect=fake_turn):
            return list(
                chat_service._run_tool_loop(
                    messages=[{"role": "user", "content": "hi"}],
                    system="sys",
                    model=model,
                    tools=None,
                    dispatch=lambda name, args: chat_service._ToolResult(text="", sources=[]),
                )
            )

    def test_done_frame_reports_the_fallback_model_when_dead_marked(self) -> None:
        with mock.patch.object(
            model_catalog, "is_marked_unavailable", return_value=True
        ), mock.patch.object(
            chat_service, "available_chat_models", return_value=["fallback/model"]
        ):
            frames = self._one_round("dead/model")
        done = frames[-1]
        self.assertTrue(done.startswith("event: done"))
        data = json.loads(done.split("data: ", 1)[1])
        self.assertEqual(data["model"], "fallback/model")

    def test_done_frame_is_byte_identical_with_no_substitution(self) -> None:
        with mock.patch.object(model_catalog, "is_marked_unavailable", return_value=False):
            frames = self._one_round("fine/model")
        self.assertEqual(frames[-1], chat_service._sse("done", {"model": "fine/model"}))


# ------------------------------------------------------------------------------------- /api/health


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_models_is_filtered_by_availability(self) -> None:
        with mock.patch.object(
            runtime_config,
            "get_overrides",
            return_value={"llm_models": ["a/m", "b/m"], "llm_base_url": "https://openrouter.ai/api/v1"},
        ):
            model_catalog.mark_unavailable("a/m", "HTTP 410")
            resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["chat_models"], ["b/m"])
        self.assertEqual(body["chat_default"], "b/m")


# ------------------------------------------------------------------------- admin: GET /api/admin/llm/models


class AdminLlmModelsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/llm/models")
        self.assertEqual(resp.status_code, 403)

    def test_contract_shape_and_role_merge(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            runtime_config,
            "get_overrides",
            return_value={
                "llm_models": ["deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"],
                "llm_followup_model": "openai/gpt-oss-120b",
                "llm_base_url": "https://openrouter.ai/api/v1",
            },
        ):
            model_catalog.mark_unavailable("openai/gpt-oss-120b", "HTTP 410")
            resp = self.client.get("/api/admin/llm/models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        self.assertEqual(body["base_url"], "https://openrouter.ai/api/v1")
        self.assertIn(body["catalog"]["status"], ("ok", "error", "unknown"))
        self.assertEqual(body["effective_default"], "deepseek/deepseek-v4-flash")
        self.assertEqual(body["effective_followup"], "deepseek/deepseek-v4-flash")  # pin is dead

        entries = {m["id"]: m for m in body["models"]}
        self.assertEqual(len(body["models"]), 2)  # no duplicate entry for the followup pin
        self.assertEqual(entries["deepseek/deepseek-v4-flash"]["roles"], ["default"])
        self.assertEqual(entries["openai/gpt-oss-120b"]["roles"], ["option", "followup"])
        self.assertEqual(entries["openai/gpt-oss-120b"]["status"], "unavailable")
        self.assertEqual(entries["openai/gpt-oss-120b"]["detail"], "HTTP 410")
        self.assertNotIn("llm_api_key", json.dumps(body))

    def test_followup_pin_outside_llm_models_gets_its_own_entry(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            runtime_config,
            "get_overrides",
            return_value={
                "llm_models": ["deepseek/deepseek-v4-flash"],
                "llm_followup_model": "xiaomi/mimo-v2.5",
                "llm_base_url": "https://openrouter.ai/api/v1",
            },
        ):
            resp = self.client.get("/api/admin/llm/models")
        body = resp.json()
        entries = {m["id"]: m for m in body["models"]}
        self.assertEqual(len(body["models"]), 2)
        self.assertEqual(entries["xiaomi/mimo-v2.5"]["roles"], ["followup"])

    def test_refresh_true_runs_a_synchronous_refresh(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            runtime_config, "get_overrides", return_value={"llm_models": ["a/m"]}
        ), mock.patch.object(model_catalog, "refresh") as refresh_mock:
            resp = self.client.get("/api/admin/llm/models?refresh=true")
        self.assertEqual(resp.status_code, 200)
        refresh_mock.assert_called_once()

    def test_refresh_omitted_never_calls_refresh(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), mock.patch.object(
            runtime_config, "get_overrides", return_value={"llm_models": ["a/m"]}
        ), mock.patch.object(model_catalog, "refresh") as refresh_mock:
            resp = self.client.get("/api/admin/llm/models")
        self.assertEqual(resp.status_code, 200)
        refresh_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
