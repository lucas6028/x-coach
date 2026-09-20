"""Unit tests for services/line_bot.push -- the daily job's LINE push helper.

Mirrors tests/test_backend_admin_line.py: unittest.TestCase, LINE's push endpoint mocked at the
httpx.post seam, get_settings patched to a lightweight stand-in.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import httpx

from backend.app.services import line_bot

_VALID_ID = "U" + "a1b2c3d4e5f6" + "0" * 20  # 'U' + 32 lowercase hex


class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _settings(token: str = "chan-token") -> types.SimpleNamespace:
    return types.SimpleNamespace(line_messaging_access_token=token)


class LinePushTests(unittest.TestCase):
    def test_sends_expected_request(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings("chan-token")), \
             mock.patch.object(line_bot.httpx, "post", return_value=_FakeResp(200)) as post:
            result = line_bot.push(_VALID_ID, "hello")
        self.assertTrue(result)
        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], line_bot.LINE_PUSH_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer chan-token")
        self.assertEqual(
            kwargs["json"],
            {"to": _VALID_ID, "messages": [{"type": "text", "text": "hello"}]},
        )

    def test_2xx_returns_true(self) -> None:
        for status in (200, 201, 202, 299):
            with mock.patch.object(line_bot, "get_settings", return_value=_settings()), \
                 mock.patch.object(line_bot.httpx, "post", return_value=_FakeResp(status)):
                self.assertTrue(line_bot.push(_VALID_ID, "hi"), msg=str(status))

    def test_non_2xx_returns_false(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot.httpx, "post", return_value=_FakeResp(400)):
            self.assertFalse(line_bot.push(_VALID_ID, "hi"))

    def test_httpx_error_returns_false(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot.httpx, "post", side_effect=httpx.ConnectError("no route")):
            self.assertFalse(line_bot.push(_VALID_ID, "hi"))

    def test_malformed_id_refused_without_request(self) -> None:
        bad_ids = [
            "",
            "not-a-line-id",
            "U" + "a" * 31,          # too short
            "U" + "a" * 33,          # too long
            "U" + "A" * 32,          # uppercase hex not allowed
            "X" + "a" * 32,          # wrong prefix
            None,
        ]
        for bad in bad_ids:
            with mock.patch.object(line_bot, "get_settings", return_value=_settings()), \
                 mock.patch.object(line_bot.httpx, "post") as post:
                self.assertFalse(line_bot.push(bad, "hi"), msg=repr(bad))
            post.assert_not_called()

    def test_never_raises(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), \
             mock.patch.object(line_bot.httpx, "post", side_effect=httpx.TimeoutException("slow")):
            try:
                result = line_bot.push(_VALID_ID, "hi")
            except Exception as exc:  # noqa: BLE001
                self.fail(f"push() raised {exc!r}")
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
