"""Unit tests for the LINE Login bridge (services/line_auth + routers/auth_line).

Mirrors the conventions of ``tests/test_backend.py``: unittest.TestCase classes, the
``supabase`` package faked through ``sys.modules`` (it is not installed in CI), external
HTTP (LINE's verify endpoint) mocked at the ``httpx.post`` seam, and FastAPI routes
exercised through ``TestClient`` with ``get_settings`` patched.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import line_auth


def _settings(**overrides) -> types.SimpleNamespace:
    """A lightweight Settings stand-in with the fields line_auth reads."""
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_channel_id": "1234567890",
        "line_login_configured": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _verify_response(status_code: int = 200, payload: dict | None = None) -> mock.Mock:
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


_CLAIMS = {
    "sub": "Uabc123DEF",
    "name": "小明",
    "picture": "https://profile.line-scdn.net/pic",
    "email": "user@example.com",
}


class SyntheticEmailTests(unittest.TestCase):
    def test_lowercases_and_wraps_sub(self) -> None:
        self.assertEqual(line_auth.synthetic_email("Uabc123DEF"), "line_uabc123def@line.invalid")

    def test_strips_whitespace(self) -> None:
        self.assertEqual(line_auth.synthetic_email("  Uxyz  "), "line_uxyz@line.invalid")

    def test_deterministic(self) -> None:
        self.assertEqual(line_auth.synthetic_email("U1"), line_auth.synthetic_email("U1"))


class VerifyLineIdTokenTests(unittest.TestCase):
    def test_valid_token_returns_claims(self) -> None:
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.object(
            line_auth.httpx, "post", return_value=_verify_response(200, dict(_CLAIMS))
        ) as post:
            claims = line_auth.verify_line_id_token("tok.tok.tok")
        self.assertEqual(claims["sub"], "Uabc123DEF")
        # The channel id must be sent as the audience check.
        _, kwargs = post.call_args
        self.assertEqual(kwargs["data"], {"id_token": "tok.tok.tok", "client_id": "1234567890"})
        self.assertEqual(post.call_args[0][0], line_auth.LINE_VERIFY_URL)

    def test_non_200_is_401(self) -> None:
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.object(
            line_auth.httpx, "post", return_value=_verify_response(400)
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.verify_line_id_token("expired")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_sub_is_401(self) -> None:
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.object(
            line_auth.httpx, "post", return_value=_verify_response(200, {"name": "no-sub"})
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.verify_line_id_token("odd")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_non_dict_payload_is_401(self) -> None:
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.object(
            line_auth.httpx, "post", return_value=_verify_response(200, [])
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.verify_line_id_token("odd")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_network_error_is_502(self) -> None:
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.object(
            line_auth.httpx, "post", side_effect=httpx.ConnectError("boom")
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.verify_line_id_token("tok")
        self.assertEqual(ctx.exception.status_code, 502)


class EnsureUserTests(unittest.TestCase):
    def test_creates_user_with_metadata_and_confirmed_email(self) -> None:
        admin = mock.Mock()
        line_auth._ensure_user(admin, dict(_CLAIMS))
        admin.auth.admin.create_user.assert_called_once_with(
            {
                "email": "line_uabc123def@line.invalid",
                "email_confirm": True,
                "user_metadata": {
                    "full_name": "小明",
                    "avatar_url": "https://profile.line-scdn.net/pic",
                    "line_sub": "Uabc123DEF",
                    "line_email": "user@example.com",
                },
            }
        )

    def test_absent_optional_claims_are_omitted(self) -> None:
        admin = mock.Mock()
        line_auth._ensure_user(admin, {"sub": "U1"})
        payload = admin.auth.admin.create_user.call_args[0][0]
        self.assertEqual(payload["user_metadata"], {"line_sub": "U1"})

    def test_duplicate_email_error_is_swallowed(self) -> None:
        admin = mock.Mock()
        admin.auth.admin.create_user.side_effect = Exception("email already registered")
        line_auth._ensure_user(admin, {"sub": "U1"})  # must not raise


def _fake_supabase_module(anon_auth: mock.Mock) -> types.ModuleType:
    """A fake ``supabase`` package whose ``create_client`` returns an anon client stub."""
    module = types.ModuleType("supabase")

    def create_client(url: str, key: str):  # noqa: ARG001 — signature parity
        client = mock.Mock()
        client.auth = anon_auth
        return client

    module.create_client = create_client  # type: ignore[attr-defined]
    return module


def _link_response(hashed_token: str | None) -> types.SimpleNamespace:
    properties = types.SimpleNamespace(hashed_token=hashed_token) if hashed_token is not None else None
    return types.SimpleNamespace(properties=properties)


def _otp_response(access: str | None, refresh: str | None) -> types.SimpleNamespace:
    session = types.SimpleNamespace(access_token=access, refresh_token=refresh)
    return types.SimpleNamespace(session=session)


class MintSessionTests(unittest.TestCase):
    def test_happy_path_returns_tokens(self) -> None:
        admin = mock.Mock()
        admin.auth.admin.generate_link.return_value = _link_response("hashed-123")
        anon_auth = mock.Mock()
        anon_auth.verify_otp.return_value = _otp_response("acc", "ref")
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(anon_auth)}
        ):
            session = line_auth._mint_session(admin, "line_u1@line.invalid")
        self.assertEqual(session, {"access_token": "acc", "refresh_token": "ref"})
        admin.auth.admin.generate_link.assert_called_once_with(
            {"type": "magiclink", "email": "line_u1@line.invalid"}
        )
        anon_auth.verify_otp.assert_called_once_with(
            {"type": "magiclink", "token_hash": "hashed-123"}
        )

    def test_missing_hashed_token_is_502(self) -> None:
        admin = mock.Mock()
        admin.auth.admin.generate_link.return_value = _link_response(None)
        with self.assertRaises(line_auth.LineAuthError) as ctx:
            line_auth._mint_session(admin, "e@line.invalid")
        self.assertEqual(ctx.exception.status_code, 502)

    def test_missing_properties_is_502(self) -> None:
        admin = mock.Mock()
        admin.auth.admin.generate_link.return_value = types.SimpleNamespace(properties=None)
        with self.assertRaises(line_auth.LineAuthError) as ctx:
            line_auth._mint_session(admin, "e@line.invalid")
        self.assertEqual(ctx.exception.status_code, 502)

    def test_missing_session_tokens_is_502(self) -> None:
        admin = mock.Mock()
        admin.auth.admin.generate_link.return_value = _link_response("hashed")
        anon_auth = mock.Mock()
        anon_auth.verify_otp.return_value = _otp_response("acc", None)
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(anon_auth)}
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth._mint_session(admin, "e@line.invalid")
        self.assertEqual(ctx.exception.status_code, 502)


class LoginWithLineTests(unittest.TestCase):
    def test_orchestrates_verify_ensure_mint(self) -> None:
        admin = mock.Mock()
        with mock.patch.object(
            line_auth, "verify_line_id_token", return_value=dict(_CLAIMS)
        ) as verify, mock.patch.object(
            line_auth, "_admin_client", return_value=admin
        ), mock.patch.object(
            line_auth, "_ensure_user"
        ) as ensure, mock.patch.object(
            line_auth, "_mint_session", return_value={"access_token": "a", "refresh_token": "r"}
        ) as mint:
            session = line_auth.login_with_line("tok")
        self.assertEqual(session, {"access_token": "a", "refresh_token": "r"})
        verify.assert_called_once_with("tok")
        ensure.assert_called_once_with(admin, dict(_CLAIMS))
        mint.assert_called_once_with(admin, "line_uabc123def@line.invalid")

    def test_line_auth_error_passes_through(self) -> None:
        with mock.patch.object(
            line_auth, "verify_line_id_token", return_value=dict(_CLAIMS)
        ), mock.patch.object(line_auth, "_admin_client", return_value=mock.Mock()), mock.patch.object(
            line_auth, "_ensure_user"
        ), mock.patch.object(
            line_auth, "_mint_session", side_effect=line_auth.LineAuthError(502, "nope")
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.login_with_line("tok")
        self.assertEqual(ctx.exception.detail, "nope")

    def test_unexpected_supabase_error_becomes_502(self) -> None:
        with mock.patch.object(
            line_auth, "verify_line_id_token", return_value=dict(_CLAIMS)
        ), mock.patch.object(line_auth, "_admin_client", return_value=mock.Mock()), mock.patch.object(
            line_auth, "_ensure_user"
        ), mock.patch.object(
            line_auth, "_mint_session", side_effect=RuntimeError("supabase down")
        ):
            with self.assertRaises(line_auth.LineAuthError) as ctx:
                line_auth.login_with_line("tok")
        self.assertEqual(ctx.exception.status_code, 502)


class ClientBuilderTests(unittest.TestCase):
    """The two thin client builders: right key to the right constructor arg."""

    def test_admin_client_uses_service_role_key(self) -> None:
        captured: dict[str, str] = {}
        module = types.ModuleType("supabase")

        def create_client(url: str, key: str):
            captured.update(url=url, key=key)
            return mock.Mock()

        module.create_client = create_client  # type: ignore[attr-defined]
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": module}
        ):
            line_auth._admin_client()
        self.assertEqual(captured, {"url": "https://proj.supabase.co", "key": "service-key"})

    def test_anon_client_uses_anon_key(self) -> None:
        captured: dict[str, str] = {}
        module = types.ModuleType("supabase")

        def create_client(url: str, key: str):
            captured.update(url=url, key=key)
            return mock.Mock()

        module.create_client = create_client  # type: ignore[attr-defined]
        with mock.patch.object(line_auth, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": module}
        ):
            line_auth._anon_client()
        self.assertEqual(captured, {"url": "https://proj.supabase.co", "key": "anon-key"})


_VALID_BODY = {"id_token": "x" * 32}


class LineLoginRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_unconfigured_is_503(self) -> None:
        from backend.app.routers import auth_line as router_module

        with mock.patch.object(
            router_module, "get_settings", return_value=_settings(line_login_configured=False)
        ):
            resp = self.client.post("/api/auth/line", json=_VALID_BODY)
        self.assertEqual(resp.status_code, 503)

    def test_happy_path_returns_session(self) -> None:
        from backend.app.routers import auth_line as router_module

        with mock.patch.object(
            router_module, "get_settings", return_value=_settings()
        ), mock.patch.object(
            line_auth, "login_with_line", return_value={"access_token": "a", "refresh_token": "r"}
        ) as login:
            resp = self.client.post("/api/auth/line", json=_VALID_BODY)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"access_token": "a", "refresh_token": "r"})
        login.assert_called_once_with(_VALID_BODY["id_token"])

    def test_line_auth_error_maps_to_http_status(self) -> None:
        from backend.app.routers import auth_line as router_module

        with mock.patch.object(
            router_module, "get_settings", return_value=_settings()
        ), mock.patch.object(
            line_auth, "login_with_line", side_effect=line_auth.LineAuthError(401, "bad token")
        ):
            resp = self.client.post("/api/auth/line", json=_VALID_BODY)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["detail"], "bad token")

    def test_gateway_error_maps_to_502(self) -> None:
        from backend.app.routers import auth_line as router_module

        with mock.patch.object(
            router_module, "get_settings", return_value=_settings()
        ), mock.patch.object(
            line_auth, "login_with_line", side_effect=line_auth.LineAuthError(502, "LINE unreachable")
        ):
            resp = self.client.post("/api/auth/line", json=_VALID_BODY)
        self.assertEqual(resp.status_code, 502)

    def test_too_short_id_token_is_422(self) -> None:
        resp = self.client.post("/api/auth/line", json={"id_token": "short"})
        self.assertEqual(resp.status_code, 422)

    def test_missing_id_token_is_422(self) -> None:
        resp = self.client.post("/api/auth/line", json={})
        self.assertEqual(resp.status_code, 422)


class LineSettingsTests(unittest.TestCase):
    """line_login_configured over the real Settings class (env-driven)."""

    def _settings_obj(self, **env):
        from backend.app.settings import Settings

        base = {
            "supabase_url": "https://p.supabase.co",
            "supabase_anon_key": "anon",
            "line_channel_id": "123",
            "supabase_service_role_key": "svc",
        }
        base.update(env)
        return Settings(_env_file=None, **base)

    def test_configured_when_all_present(self) -> None:
        self.assertTrue(self._settings_obj().line_login_configured)

    def test_not_configured_without_channel_id(self) -> None:
        self.assertFalse(self._settings_obj(line_channel_id="").line_login_configured)

    def test_not_configured_without_service_role_key(self) -> None:
        self.assertFalse(self._settings_obj(supabase_service_role_key="").line_login_configured)

    def test_not_configured_without_base_auth(self) -> None:
        self.assertFalse(self._settings_obj(supabase_url="").line_login_configured)


class HealthLineFlagTests(unittest.TestCase):
    def test_health_reports_line_login_configured(self) -> None:
        from backend.app import main as main_module

        stand_in = types.SimpleNamespace(
            auth_configured=True, chat_configured=False, line_login_configured=True
        )
        with mock.patch.object(main_module, "get_settings", return_value=stand_in):
            resp = TestClient(app).get("/api/health")
        self.assertTrue(resp.json()["line_login_configured"])

    def test_health_defaults_line_flag_false_on_lightweight_settings(self) -> None:
        from backend.app import main as main_module

        stand_in = types.SimpleNamespace(auth_configured=False, chat_configured=False)
        with mock.patch.object(main_module, "get_settings", return_value=stand_in):
            resp = TestClient(app).get("/api/health")
        self.assertFalse(resp.json()["line_login_configured"])


class CorsEnvTests(unittest.TestCase):
    def test_extra_origins_appended_from_env(self) -> None:
        import importlib
        import os

        from backend.app import config

        with mock.patch.dict(
            os.environ, {"XCOACH_CORS_ORIGINS": "https://a.ngrok-free.app, https://b.example"}
        ):
            reloaded = importlib.reload(config)
            try:
                self.assertIn("https://a.ngrok-free.app", reloaded.CORS_ORIGINS)
                self.assertIn("https://b.example", reloaded.CORS_ORIGINS)
                self.assertIn("http://localhost:5173", reloaded.CORS_ORIGINS)
            finally:
                importlib.reload(config)

    def test_no_env_keeps_dev_origins_only(self) -> None:
        from backend.app import config

        self.assertEqual(
            [o for o in config.CORS_ORIGINS if not o.startswith("http://")],
            [o for o in config.CORS_ORIGINS if "5173" not in o],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
