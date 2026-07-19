# LINE Messaging API 訓練摘要 Bot 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓使用者在 LINE 官方帳號聊天室點圖文選單（或傳關鍵字），收到自己的 x-coach 訓練摘要。

**Architecture:** LINE 平台把事件 POST 到新的 `/api/line/webhook`；router 驗簽後交給
`services/line_bot`，它以 service_role 呼叫一支 SECURITY DEFINER 的 Postgres 函式
`line_training_summary(p_line_sub)`，用 LINE userId（＝ LINE Login 的 `sub`）直接對到
`auth.users`，聚合出摘要後用 LINE reply API 回一則文字訊息。前端零改動。

**Tech Stack:** FastAPI、httpx、supabase-py（service_role client）、Postgres（SECURITY
DEFINER function）、pytest（unittest.TestCase + `TestClient`）。

設計來源：`docs/superpowers/specs/2026-07-20-line-messaging-api-design.md`

## Global Constraints

- **Python 直譯器一律 `.venv\Scripts\python.exe`**（本機沒有 `python` on PATH）；所有指令從 repo root 執行。
- **測試一律 scope 到 `tests/`**：`.venv\Scripts\python.exe -m pytest tests/`，絕不用裸 `pytest`。
- **CI backend coverage gate 95%**：`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` 必須通過。
- 測試風格：`unittest.TestCase` 類別、`supabase` 套件用 `sys.modules` 假造（CI 未安裝）、
  外部 HTTP 在 `httpx.post` 這個接縫 mock、FastAPI 路由用 `TestClient` 且 patch `get_settings`。
  參考 `tests/test_backend_line_auth.py`。
- 未設定環境變數時端點回 **503**（比照 `/api/auth/line`、`/api/chat`）。
- **驗簽通過後一律回 200**：LINE 把非 2xx 視為 webhook 失敗，事件已消費，重試無意義。
- log **不得**記錄 LINE userId 全值或摘要內容。
- 回覆訊息一律**繁體中文**。
- `LINE_CHANNEL_ID`（Login channel）與本次的 Messaging channel secret／token 是**不同 channel**
  的憑證，不可混用。

---

## File Structure

| 檔案 | 狀態 | 責任 |
|---|---|---|
| `backend/app/settings.py` | 修改 | 新增 3 個 Messaging 設定欄位 + `line_messaging_configured` |
| `.env.example` | 修改 | 對應的環境變數說明 |
| `db/migrations/20260720000000_line_training_summary.sql` | 新增 | SECURITY DEFINER 摘要函式 + 權限 |
| `backend/app/services/line_bot.py` | 新增 | 驗簽、RPC 取摘要、訊息格式化、事件處理、reply API |
| `backend/app/routers/line_webhook.py` | 新增 | 薄 HTTP 層：raw bytes → 驗簽 → service → 200 |
| `backend/app/main.py` | 修改 | `include_router(line_webhook.router)` |
| `tests/test_backend_line_webhook.py` | 新增 | 上述全部的單元測試 |
| `docs/line-login-liff-setup.md` | 修改 | LINE console 手動設定步驟 |
| `docs/line-login-liff-evaluation.md` | 修改 | 修正 §3.3「pairwise 每個 channel」的錯誤 |

---

### Task 1: 設定值與環境變數

**Files:**
- Modify: `backend/app/settings.py`（LINE 區塊，`line_channel_id` 附近）
- Modify: `.env.example`
- Test: `tests/test_backend_line_webhook.py`（新建）

**Interfaces:**
- Consumes: 既有 `Settings.auth_configured`、`Settings.supabase_service_role_key`
- Produces: `Settings.line_messaging_channel_secret: str`、`Settings.line_messaging_access_token: str`、
  `Settings.line_liff_id: str`、`Settings.line_messaging_configured: bool`（property）

> 註：`line_liff_id` 是設計文件之外的小追加——回覆訊息裡的「打開 x-coach」連結需要
> LIFF id，而後端目前不知道它。留空時連結那行會被省略，不影響其他功能。

- [ ] **Step 1: 寫失敗測試**

建立 `tests/test_backend_line_webhook.py`：

```python
"""Unit tests for the LINE Messaging API bot (services/line_bot + routers/line_webhook).

Mirrors ``tests/test_backend_line_auth.py``: unittest.TestCase classes, the ``supabase``
package faked through ``sys.modules`` (it is not installed in CI), external HTTP (LINE's
reply endpoint) mocked at the ``httpx.post`` seam, and FastAPI routes exercised through
``TestClient`` with ``get_settings`` patched.
"""

from __future__ import annotations

import unittest

from backend.app.settings import Settings


class MessagingConfiguredTests(unittest.TestCase):
    def _settings(self, **overrides) -> Settings:
        values = {
            "supabase_url": "https://proj.supabase.co",
            "supabase_anon_key": "anon-key",
            "supabase_service_role_key": "service-key",
            "line_messaging_channel_secret": "secret",
            "line_messaging_access_token": "token",
        }
        values.update(overrides)
        return Settings(**values)

    def test_configured_when_all_present(self) -> None:
        self.assertTrue(self._settings().line_messaging_configured)

    def test_not_configured_without_secret(self) -> None:
        self.assertFalse(self._settings(line_messaging_channel_secret="").line_messaging_configured)

    def test_not_configured_without_access_token(self) -> None:
        self.assertFalse(self._settings(line_messaging_access_token="").line_messaging_configured)

    def test_not_configured_without_service_role_key(self) -> None:
        self.assertFalse(self._settings(supabase_service_role_key="").line_messaging_configured)

    def test_not_configured_without_supabase(self) -> None:
        self.assertFalse(self._settings(supabase_url="").line_messaging_configured)

    def test_liff_id_defaults_to_empty(self) -> None:
        # Assert the declared default, not an instance: a real repo-root .env with
        # LINE_LIFF_ID set would otherwise make this pass/fail by machine.
        self.assertEqual(Settings.model_fields["line_liff_id"].default, "")
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: FAIL — `pydantic` 拒絕未知欄位或 `AttributeError: 'Settings' object has no attribute 'line_messaging_configured'`

- [ ] **Step 3: 實作設定**

在 `backend/app/settings.py` 的 `line_channel_id` / `supabase_service_role_key` 之後加入：

```python
    # LINE Messaging API bot (the official account chat room). A SEPARATE channel from the
    # Login channel above — its own secret (webhook signature) and access token (reply API).
    # Both channels must live under the SAME LINE provider: that is what makes the webhook's
    # source.userId identical to the Login ID token's ``sub`` (which line_auth stores in
    # user_metadata.line_sub), so the bot can find the account with no binding flow.
    # Leave unset to keep POST /api/line/webhook disabled (503). See services/line_bot.
    line_messaging_channel_secret: str = ""
    line_messaging_access_token: str = ""
    # LIFF app id, used only to build the "open x-coach" deep link in bot replies. Optional:
    # when blank the link line is omitted.
    line_liff_id: str = ""
```

在 `line_login_configured` property 之後加入：

```python
    @property
    def line_messaging_configured(self) -> bool:
        """True when the bot can verify webhook signatures, reply, and read the summary.

        The service_role key is required because the webhook has no user JWT: it reads the
        summary through the ``line_training_summary`` SECURITY DEFINER function, which is
        granted to service_role only.
        """
        return bool(
            self.line_messaging_channel_secret
            and self.line_messaging_access_token
            and self.supabase_service_role_key
            and self.auth_configured
        )
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 更新 `.env.example`**

在既有的 `LINE_CHANNEL_ID` / `SUPABASE_SERVICE_ROLE_KEY` 區塊之後追加：

```
# LINE Messaging API bot (official account chat) — optional; leave unset to keep
# POST /api/line/webhook disabled (503). This is a DIFFERENT channel from LINE_CHANNEL_ID
# above, and it MUST be created under the same LINE provider as the Login channel — that is
# what makes the webhook's userId equal the Login ID token's `sub`.
# LINE_MESSAGING_CHANNEL_SECRET: Messaging API channel → Basic settings → Channel secret.
# LINE_MESSAGING_ACCESS_TOKEN:   Messaging API channel → Messaging API → Channel access token.
# LINE_LIFF_ID: optional; only used to build the "open x-coach" link in bot replies.
LINE_MESSAGING_CHANNEL_SECRET=
LINE_MESSAGING_ACCESS_TOKEN=
LINE_LIFF_ID=
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/settings.py .env.example tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): add Messaging API settings and configured gate"
```

---

### Task 2: Webhook 簽章驗證

**Files:**
- Create: `backend/app/services/line_bot.py`
- Test: `tests/test_backend_line_webhook.py`（追加）

**Interfaces:**
- Consumes: `Settings.line_messaging_channel_secret`（Task 1）
- Produces: `line_bot.verify_signature(raw_body: bytes, signature: str | None) -> bool`、
  模組常數 `line_bot.LINE_REPLY_URL: str`

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_backend_line_webhook.py` 追加（連同 import）：

```python
import base64
import hashlib
import hmac
import types
from unittest import mock

from backend.app.services import line_bot


def _settings(**overrides) -> types.SimpleNamespace:
    """A lightweight Settings stand-in with the fields line_bot reads."""
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_messaging_channel_secret": "chan-secret",
        "line_messaging_access_token": "chan-token",
        "line_liff_id": "1234567890-Abcdefgh",
        "line_messaging_configured": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _sign(body: bytes, secret: str = "chan-secret") -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


class VerifySignatureTests(unittest.TestCase):
    def test_valid_signature_passes(self) -> None:
        body = b'{"events":[]}'
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertTrue(line_bot.verify_signature(body, _sign(body)))

    def test_tampered_body_fails(self) -> None:
        signature = _sign(b'{"events":[]}')
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b'{"events":[1]}', signature))

    def test_missing_signature_fails(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b"{}", None))

    def test_wrong_secret_fails(self) -> None:
        body = b"{}"
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(body, _sign(body, "other-secret")))

    def test_empty_secret_fails(self) -> None:
        body = b"{}"
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(line_messaging_channel_secret="")
        ):
            self.assertFalse(line_bot.verify_signature(body, _sign(body)))

    def test_non_ascii_signature_fails(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertFalse(line_bot.verify_signature(b"{}", "不是-base64-簽章"))
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.app.services.line_bot'`

- [ ] **Step 3: 建立 `backend/app/services/line_bot.py`**

```python
"""LINE Messaging API bot: answer "my training summary" inside the LINE chat room.

The companion to ``services/line_auth`` (which signs LINE users in). Both channels live
under the SAME LINE provider, so the webhook's ``source.userId`` is byte-identical to the
Login ID token's ``sub`` that ``line_auth`` stored in ``user_metadata.line_sub`` — that is
what lets the bot resolve the account with no binding flow (LINE docs: a user ID is issued
per *provider*, not per channel).

The webhook has no user JWT, so it cannot use ``services/store`` (every call there runs as
the user with RLS). Instead it calls ONE SECURITY DEFINER function,
``public.line_training_summary(p_line_sub)``, granted to service_role only — the smallest
possible widening of the "backend never touches data with service_role" posture.

``httpx`` is a top-level import (as in ``line_auth``); the ``supabase`` import is deferred so
the module stays light and unit tests can fake the package without it installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

# LINE's reply endpoint. Replies are free and need no push quota, but the reply token is
# single-use and expires ~1 minute after the event — never retry a failed reply.
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"


def verify_signature(raw_body: bytes, signature: str | None) -> bool:
    """Whether ``signature`` is LINE's HMAC-SHA256 of ``raw_body`` under our channel secret.

    Must be computed over the RAW request bytes — re-serialising the parsed JSON would change
    whitespace/key order and never match. Compared with ``compare_digest`` so a wrong
    signature leaks no timing information.
    """
    secret = get_settings().line_messaging_channel_secret
    if not secret or not signature:
        return False
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/line_bot.py tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): verify LINE webhook signatures over raw body"
```

---

### Task 3: 摘要 RPC（migration + service 讀取）

**Files:**
- Create: `db/migrations/20260720000000_line_training_summary.sql`
- Modify: `backend/app/services/line_bot.py`
- Test: `tests/test_backend_line_webhook.py`（追加）

**Interfaces:**
- Consumes: `Settings.supabase_url`、`Settings.supabase_service_role_key`
- Produces: `line_bot.summary_for_line_user(line_user_id: str) -> dict | None`，回傳形狀為
  `{"total": int, "latest": {"created_at": str, "view_type": str, "fault_count": int} | None,
  "top_faults": [{"id": str, "name": str, "count": int}, ...]}`；查無此 LINE 使用者時回 `None`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_backend_line_webhook.py` 追加（連同 `import sys`）：

```python
import sys


def _fake_supabase_module(client: mock.Mock) -> types.ModuleType:
    """A fake ``supabase`` package whose ``create_client`` returns ``client``."""
    module = types.ModuleType("supabase")

    def create_client(url: str, key: str):  # noqa: ARG001 — signature parity
        return client

    module.create_client = create_client  # type: ignore[attr-defined]
    return module


def _rpc_client(data) -> mock.Mock:
    client = mock.Mock()
    client.rpc.return_value.execute.return_value = types.SimpleNamespace(data=data)
    return client


_SUMMARY = {
    "total": 12,
    "latest": {"created_at": "2026-07-19T13:03:11.5+00:00", "view_type": "side", "fault_count": 3},
    "top_faults": [
        {"id": "knees_inward", "name": "Knees Inward / Knee Valgus", "count": 7},
        {"id": "shallow_depth", "name": "Shallow Depth", "count": 5},
    ],
}


class SummaryForLineUserTests(unittest.TestCase):
    def test_returns_rpc_payload(self) -> None:
        client = _rpc_client(dict(_SUMMARY))
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            result = line_bot.summary_for_line_user("Uabc123")
        self.assertEqual(result["total"], 12)
        client.rpc.assert_called_once_with(
            "line_training_summary", {"p_line_sub": "Uabc123"}
        )

    def test_null_payload_returns_none(self) -> None:
        client = _rpc_client(None)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            self.assertIsNone(line_bot.summary_for_line_user("Uabc123"))

    def test_unexpected_payload_shape_returns_none(self) -> None:
        client = _rpc_client([1, 2, 3])
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.dict(
            sys.modules, {"supabase": _fake_supabase_module(client)}
        ):
            self.assertIsNone(line_bot.summary_for_line_user("Uabc123"))
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -k Summary -v`
Expected: FAIL — `AttributeError: module 'backend.app.services.line_bot' has no attribute 'summary_for_line_user'`

- [ ] **Step 3a: 寫 migration**

建立 `db/migrations/20260720000000_line_training_summary.sql`：

```sql
-- LINE Messaging API bot: one read-only, tightly-scoped entry point for the webhook.
--
-- The webhook is authenticated by LINE's request signature, not by a Supabase JWT, so it has
-- no user token and cannot go through the usual RLS-scoped path (services/store). Rather than
-- letting the backend read tables with service_role, we expose exactly ONE SECURITY DEFINER
-- function that takes a LINE user id and returns only that user's aggregate summary.
--
-- Resolution works because the Messaging API channel and the LINE Login channel live under the
-- same LINE provider: LINE issues a user id per provider, so the webhook's source.userId equals
-- the Login ID token's `sub`, which services/line_auth stored in user_metadata.line_sub.
--
-- Access control is the GRANT: service_role only. anon/authenticated cannot call it at all, so
-- no signed-in user can pass someone else's `sub` and read their data.

create or replace function public.line_training_summary(p_line_sub text)
returns jsonb
language plpgsql
security definer
-- Pin name resolution so the definer body can't be hijacked by a caller-controlled search_path.
set search_path = public, auth
as $$
declare
    v_user_id uuid;
    v_total   bigint;
    v_latest  jsonb;
    v_top     jsonb;
begin
    if p_line_sub is null or length(p_line_sub) = 0 then
        return null;
    end if;

    select u.id into v_user_id
    from auth.users u
    where u.raw_user_meta_data ->> 'line_sub' = p_line_sub
    limit 1;

    -- No x-coach account for this LINE user: the bot turns this into a "sign in first" reply.
    if v_user_id is null then
        return null;
    end if;

    select count(*) into v_total
    from public.analyses a
    where a.user_id = v_user_id;

    select to_jsonb(x) into v_latest
    from (
        select a.created_at, a.view_type, a.fault_count
        from public.analyses a
        where a.user_id = v_user_id
        order by a.created_at desc
        limit 1
    ) x;

    -- Top 3 faults by how often they were detected across every analysis. Grouped by the
    -- stable fault_id (the backend maps it to a localised label); the English fault_name rides
    -- along as a fallback for ids the backend doesn't know yet.
    select coalesce(
        jsonb_agg(jsonb_build_object('id', t.id, 'name', t.name, 'count', t.cnt)),
        '[]'::jsonb
    ) into v_top
    from (
        select
            d ->> 'fault_id'      as id,
            min(d ->> 'fault_name') as name,
            count(*)              as cnt
        from public.analyses a
        cross join lateral jsonb_array_elements(
            coalesce(a.result -> 'detections', '[]'::jsonb)
        ) as d
        where a.user_id = v_user_id
        group by d ->> 'fault_id'
        order by count(*) desc, d ->> 'fault_id'
        limit 3
    ) t;

    return jsonb_build_object('total', v_total, 'latest', v_latest, 'top_faults', v_top);
end;
$$;

-- Least privilege: revoke the implicit PUBLIC execute grant (and the roles that inherit it),
-- then grant only to service_role, which only the backend holds.
revoke all on function public.line_training_summary(text) from public;
revoke all on function public.line_training_summary(text) from anon;
revoke all on function public.line_training_summary(text) from authenticated;
grant execute on function public.line_training_summary(text) to service_role;
```

- [ ] **Step 3b: 實作 `summary_for_line_user`**

在 `backend/app/services/line_bot.py` 的 `verify_signature` 之後加入：

```python
from typing import Any

# The SECURITY DEFINER function granted to service_role (see the matching migration). It is
# the ONLY data the bot can reach: one user's aggregate summary, keyed by their LINE user id.
SUMMARY_RPC = "line_training_summary"


def _service_client() -> Any:
    """Build a service_role Supabase client (needed to call the summary RPC).

    Deferred import, as in ``line_auth._admin_client``: keeps the module light and lets unit
    tests fake the ``supabase`` package without it installed.
    """
    from supabase import create_client  # deferred heavy import (gotrue/postgrest/...)

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def summary_for_line_user(line_user_id: str) -> dict[str, Any] | None:
    """Return this LINE user's training summary, or ``None`` if they have no x-coach account.

    The RPC returns SQL NULL for an unknown ``sub``; PostgREST surfaces that as ``None``. Any
    other shape is treated as "unknown" rather than trusted.
    """
    response = _service_client().rpc(SUMMARY_RPC, {"p_line_sub": line_user_id}).execute()
    data = getattr(response, "data", None)
    return data if isinstance(data, dict) else None
```

（把 `from typing import Any` 併到檔案頂端既有的 import 區。）

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（15 passed）

- [ ] **Step 5: Commit**

```bash
git add db/migrations/20260720000000_line_training_summary.sql backend/app/services/line_bot.py tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): read the training summary via a service_role-only RPC"
```

---

### Task 4: 訊息格式化

**Files:**
- Modify: `backend/app/services/line_bot.py`
- Test: `tests/test_backend_line_webhook.py`（追加）

**Interfaces:**
- Consumes: Task 3 的摘要形狀、`Settings.line_liff_id`
- Produces: `line_bot.format_summary(summary: dict) -> str`、`line_bot.unbound_message() -> str`、
  `line_bot.empty_message() -> str`、`line_bot.help_message() -> str`

- [ ] **Step 1: 寫失敗測試**

追加：

```python
class FormatSummaryTests(unittest.TestCase):
    def _format(self, summary: dict, **setting_overrides) -> str:
        with mock.patch.object(
            line_bot, "get_settings", return_value=_settings(**setting_overrides)
        ):
            return line_bot.format_summary(summary)

    def test_full_summary_has_counts_faults_and_link(self) -> None:
        text = self._format(dict(_SUMMARY))
        self.assertIn("累積分析：12 次", text)
        # 2026-07-19T13:03Z is 21:03 in UTC+8.
        self.assertIn("2026-07-19 21:03", text)
        self.assertIn("側面", text)
        self.assertIn("3 個問題", text)
        self.assertIn("1. 膝蓋內夾 ×7", text)
        self.assertIn("2. 深度不足 ×5", text)
        self.assertIn("https://liff.line.me/1234567890-Abcdefgh", text)

    def test_unknown_fault_id_falls_back_to_english_name(self) -> None:
        summary = {
            "total": 1,
            "latest": None,
            "top_faults": [{"id": "brand_new_fault", "name": "Brand New Fault", "count": 2}],
        }
        self.assertIn("Brand New Fault ×2", self._format(summary))

    def test_no_faults_omits_the_fault_section(self) -> None:
        text = self._format({"total": 2, "latest": None, "top_faults": []})
        self.assertNotIn("最常出現的問題", text)
        self.assertIn("累積分析：2 次", text)

    def test_blank_liff_id_omits_the_link(self) -> None:
        text = self._format(dict(_SUMMARY), line_liff_id="")
        self.assertNotIn("liff.line.me", text)

    def test_unknown_view_type_and_bad_timestamp_degrade_gracefully(self) -> None:
        summary = {
            "total": 1,
            "latest": {"created_at": "not-a-date", "view_type": "weird", "fault_count": None},
            "top_faults": [],
        }
        text = self._format(summary)
        self.assertIn("未知時間", text)
        self.assertIn("未知", text)
        self.assertIn("0 個問題", text)

    def test_missing_keys_do_not_raise(self) -> None:
        self.assertIn("累積分析：0 次", self._format({}))


class StaticMessageTests(unittest.TestCase):
    def test_unbound_message_points_at_line_sign_in(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            text = line_bot.unbound_message()
        self.assertIn("LINE 登入", text)
        self.assertIn("https://liff.line.me/1234567890-Abcdefgh", text)

    def test_empty_message_mentions_no_records(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertIn("還沒有分析紀錄", line_bot.empty_message())

    def test_help_message_lists_the_keyword(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            self.assertIn("摘要", line_bot.help_message())
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -k "Format or StaticMessage" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'format_summary'`

- [ ] **Step 3: 實作格式化**

在 `backend/app/services/line_bot.py` 追加（並把 `datetime` import 併到頂端）：

```python
from datetime import datetime, timedelta, timezone

# Replies are read in Taiwan; fix the display offset rather than depending on a tz database
# (zoneinfo needs the `tzdata` package on Windows, which is not in the lean CI dependency set).
_DISPLAY_TZ = timezone(timedelta(hours=8))

# Localised labels, kept in step with the frontend's i18n keys (`fault.*`, `view.*` in
# frontend/src/lib/i18n.tsx) so the chat room and the web app name the same thing the same way.
FAULT_LABELS: dict[str, str] = {
    "knees_inward": "膝蓋內夾",
    "knees_forward": "膝蓋前移",
    "shallow_depth": "深度不足",
    "excessive_forward_lean": "軀幹過度前傾",
    "heel_rise": "腳跟離地",
    "butt_wink": "骨盆後傾",
    "asymmetric_shift": "左右不對稱",
}
VIEW_LABELS: dict[str, str] = {
    "front": "正面",
    "front_oblique": "正面斜角",
    "side": "側面",
    "rear": "背面",
    "rear_oblique": "背面斜角",
    "left": "左側",
    "right": "右側",
    "unknown": "未知",
}


def _liff_link() -> str:
    """The deep link into the LIFF app, or "" when no LIFF id is configured."""
    liff_id = (getattr(get_settings(), "line_liff_id", "") or "").strip()
    return f"https://liff.line.me/{liff_id}" if liff_id else ""


def _format_time(raw: Any) -> str:
    """Render a PostgREST ISO timestamp in UTC+8, or "未知時間" if it can't be parsed."""
    if not isinstance(raw, str) or not raw:
        return "未知時間"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "未知時間"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")


def _fault_label(fault: dict[str, Any]) -> str:
    """Localised fault name, falling back to the English name for ids we don't know yet."""
    fault_id = str(fault.get("id") or "")
    return FAULT_LABELS.get(fault_id) or str(fault.get("name") or fault_id or "未知問題")


def _with_link(lines: list[str], call_to_action: str) -> str:
    link = _liff_link()
    if link:
        lines += ["", f"{call_to_action} 👉 {link}"]
    return "\n".join(lines)


def format_summary(summary: dict[str, Any]) -> str:
    """Render one training summary as a single LINE text message.

    Every field is defensive: the RPC shape is trusted, but a malformed row must degrade into
    a readable message rather than raise inside a webhook that has already been acknowledged.
    """
    lines = ["📊 你的訓練摘要", "", f"累積分析：{int(summary.get('total') or 0)} 次"]

    latest = summary.get("latest")
    if isinstance(latest, dict):
        view = VIEW_LABELS.get(str(latest.get("view_type") or "unknown"), "未知")
        faults = int(latest.get("fault_count") or 0)
        lines.append(
            f"最近一次：{_format_time(latest.get('created_at'))}"
            f"（{view}視角，偵測到 {faults} 個問題）"
        )

    top = [f for f in (summary.get("top_faults") or []) if isinstance(f, dict)]
    if top:
        lines += ["", "最常出現的問題"]
        for rank, fault in enumerate(top, start=1):
            lines.append(f"{rank}. {_fault_label(fault)} ×{int(fault.get('count') or 0)}")

    return _with_link(lines, "打開 x-coach 看完整報告")


def unbound_message() -> str:
    """Reply for a LINE user with no matching x-coach account."""
    return _with_link(
        ["還沒有找到你的 x-coach 帳號。", "請先用 LINE 登入 x-coach，之後就能在這裡查詢訓練摘要。"],
        "前往登入",
    )


def empty_message() -> str:
    """Reply for a known user who has no analyses yet."""
    return _with_link(
        ["你還沒有分析紀錄。", "上傳一支深蹲影片做第一次分析，之後就能在這裡看到摘要。"],
        "開始分析",
    )


def help_message() -> str:
    """Reply for text we don't recognise."""
    return _with_link(
        ["傳「摘要」或點下方選單，就能看到你的訓練摘要。"],
        "打開 x-coach",
    )
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（24 passed）

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/line_bot.py tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): format the training summary as a chat reply"
```

---

### Task 5: 事件處理與 reply API

**Files:**
- Modify: `backend/app/services/line_bot.py`
- Test: `tests/test_backend_line_webhook.py`（追加）

**Interfaces:**
- Consumes: Task 3 的 `summary_for_line_user`、Task 4 的四支訊息函式
- Produces: `line_bot.handle_events(payload: dict) -> list[dict[str, str]]`（每個元素為
  `{"reply_token": str, "text": str}`）、`line_bot.reply(reply_token: str, text: str) -> None`、
  常數 `line_bot.SUMMARY_KEYWORDS: frozenset[str]`

- [ ] **Step 1: 寫失敗測試**

追加（連同 `import httpx`）：

```python
import httpx


def _text_event(text: str, user_id: str = "Uabc123", reply_token: str = "rt-1") -> dict:
    return {
        "type": "message",
        "replyToken": reply_token,
        "source": {"type": "user", "userId": user_id},
        "message": {"type": "text", "id": "m1", "text": text},
    }


class HandleEventsTests(unittest.TestCase):
    def _handle(self, payload: dict, summary_return=..., summary_side_effect=None) -> list[dict]:
        patcher = mock.patch.object(
            line_bot,
            "summary_for_line_user",
            side_effect=summary_side_effect,
            **({} if summary_return is ... else {"return_value": summary_return}),
        )
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), patcher:
            return line_bot.handle_events(payload)

    def test_keyword_returns_the_summary(self) -> None:
        replies = self._handle({"events": [_text_event("我的訓練摘要")]}, summary_return=dict(_SUMMARY))
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["reply_token"], "rt-1")
        self.assertIn("累積分析：12 次", replies[0]["text"])

    def test_short_keyword_and_whitespace_and_case_are_normalised(self) -> None:
        replies = self._handle({"events": [_text_event("  Summary  ")]}, summary_return=dict(_SUMMARY))
        self.assertIn("你的訓練摘要", replies[0]["text"])

    def test_unknown_account_returns_the_sign_in_reply(self) -> None:
        replies = self._handle({"events": [_text_event("摘要")]}, summary_return=None)
        self.assertIn("還沒有找到你的 x-coach 帳號", replies[0]["text"])

    def test_zero_analyses_returns_the_empty_reply(self) -> None:
        replies = self._handle(
            {"events": [_text_event("摘要")]},
            summary_return={"total": 0, "latest": None, "top_faults": []},
        )
        self.assertIn("還沒有分析紀錄", replies[0]["text"])

    def test_unknown_text_returns_help(self) -> None:
        replies = self._handle({"events": [_text_event("你好")]}, summary_return=dict(_SUMMARY))
        self.assertIn("傳「摘要」", replies[0]["text"])

    def test_non_text_and_non_message_events_are_ignored(self) -> None:
        payload = {
            "events": [
                {"type": "follow", "replyToken": "rt", "source": {"userId": "U1"}},
                {
                    "type": "message",
                    "replyToken": "rt",
                    "source": {"userId": "U1"},
                    "message": {"type": "sticker", "id": "s1"},
                },
                "not-a-dict",
            ]
        }
        self.assertEqual(self._handle(payload, summary_return=dict(_SUMMARY)), [])

    def test_events_without_reply_token_or_user_id_are_skipped(self) -> None:
        payload = {
            "events": [
                _text_event("摘要", reply_token=""),
                {"type": "message", "replyToken": "rt", "source": {}, "message": {"type": "text", "text": "摘要"}},
            ]
        }
        self.assertEqual(self._handle(payload, summary_return=dict(_SUMMARY)), [])

    def test_missing_events_key_returns_nothing(self) -> None:
        self.assertEqual(self._handle({}, summary_return=dict(_SUMMARY)), [])

    def test_rpc_failure_falls_back_to_an_apology_and_keeps_other_events(self) -> None:
        payload = {"events": [_text_event("摘要", reply_token="rt-1"), _text_event("你好", reply_token="rt-2")]}
        replies = self._handle(payload, summary_side_effect=RuntimeError("db down"))
        self.assertEqual(len(replies), 2)
        self.assertIn("暫時查不到", replies[0]["text"])
        self.assertIn("傳「摘要」", replies[1]["text"])


class ReplyTests(unittest.TestCase):
    def test_posts_a_text_message_with_the_bearer_token(self) -> None:
        response = mock.Mock(status_code=200)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", return_value=response
        ) as post:
            line_bot.reply("rt-1", "嗨")
        self.assertEqual(post.call_args[0][0], line_bot.LINE_REPLY_URL)
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer chan-token")
        self.assertEqual(
            kwargs["json"],
            {"replyToken": "rt-1", "messages": [{"type": "text", "text": "嗨"}]},
        )

    def test_non_200_is_swallowed(self) -> None:
        response = mock.Mock(status_code=400)
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", return_value=response
        ):
            line_bot.reply("rt-1", "嗨")  # must not raise

    def test_network_error_is_swallowed(self) -> None:
        with mock.patch.object(line_bot, "get_settings", return_value=_settings()), mock.patch.object(
            line_bot.httpx, "post", side_effect=httpx.ConnectError("boom")
        ):
            line_bot.reply("rt-1", "嗨")  # must not raise
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -k "HandleEvents or Reply" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'handle_events'`

- [ ] **Step 3: 實作**

在 `backend/app/services/line_bot.py` 追加（並把 `import httpx` 併到頂端）：

```python
import httpx

_REPLY_TIMEOUT_S = 10.0

# Text that means "show me my summary". The rich-menu button is configured as a *message*
# action sending "我的訓練摘要", so the menu and typed keywords share one code path. Compared
# after stripping and lower-casing (lower-casing is a no-op for the Chinese entries).
SUMMARY_KEYWORDS = frozenset({"我的訓練摘要", "摘要", "訓練", "紀錄", "summary"})


def _reply_text_for(line_user_id: str) -> str:
    """Decide what to say to this user, degrading to an apology if the lookup fails."""
    try:
        summary = summary_for_line_user(line_user_id)
    except Exception:  # noqa: BLE001 — a webhook must answer, never propagate.
        logger.exception("LINE bot: training-summary lookup failed")
        return "暫時查不到你的訓練摘要，請稍後再試一次。"
    if summary is None:
        return unbound_message()
    if int(summary.get("total") or 0) == 0:
        return empty_message()
    return format_summary(summary)


def handle_events(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Turn a webhook payload into the replies to send.

    Returns planned replies instead of sending them so the decision logic stays a pure-ish
    function (one mocked seam) and the router owns all the I/O. Non-text and malformed events
    are skipped silently: LINE delivers many event types we don't answer.
    """
    replies: list[dict[str, str]] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("type") != "text":
            continue
        reply_token = event.get("replyToken")
        source = event.get("source")
        line_user_id = source.get("userId") if isinstance(source, dict) else None
        if not reply_token or not line_user_id:
            continue

        text = str(message.get("text") or "").strip().lower()
        answer = _reply_text_for(str(line_user_id)) if text in SUMMARY_KEYWORDS else help_message()
        replies.append({"reply_token": str(reply_token), "text": answer})
    return replies


def reply(reply_token: str, text: str) -> None:
    """Send one text message back through LINE's reply API; failures are logged, never raised.

    The reply token is single-use and expires ~1 minute after the event, so a failed reply is
    not retried — and the webhook has already been acknowledged either way.
    """
    settings = get_settings()
    try:
        response = httpx.post(
            LINE_REPLY_URL,
            headers={"Authorization": f"Bearer {settings.line_messaging_access_token}"},
            json={"replyToken": reply_token, "messages": [{"type": "text", "text": text}]},
            timeout=_REPLY_TIMEOUT_S,
        )
    except httpx.HTTPError:
        logger.warning("LINE bot: reply request failed")
        return
    if response.status_code != 200:
        # Never log the body or the token — it can carry user-identifying content.
        logger.warning("LINE bot: reply rejected with status %s", response.status_code)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（36 passed）

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/line_bot.py tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): route keyword events to summary replies"
```

---

### Task 6: Webhook 端點

**Files:**
- Create: `backend/app/routers/line_webhook.py`
- Modify: `backend/app/main.py:15-25`（import 區）與 `backend/app/main.py:41-49`（`include_router` 區）
- Test: `tests/test_backend_line_webhook.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `verify_signature`、Task 5 的 `handle_events` 與 `reply`、
  Task 1 的 `line_messaging_configured`
- Produces: `POST /api/line/webhook`

- [ ] **Step 1: 寫失敗測試**

追加：

```python
import json

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import line_webhook


class WebhookRouteTests(unittest.TestCase):
    def _post(self, body: dict, *, signature: str | None = ..., settings=None):
        raw = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if signature is ...:
            headers["X-Line-Signature"] = _sign(raw)
        elif signature is not None:
            headers["X-Line-Signature"] = signature
        with mock.patch.object(
            line_webhook, "get_settings", return_value=settings or _settings()
        ), mock.patch.object(line_bot, "get_settings", return_value=settings or _settings()):
            with TestClient(app) as client:
                return client.post("/api/line/webhook", content=raw, headers=headers)

    def test_unconfigured_is_503(self) -> None:
        response = self._post({"events": []}, settings=_settings(line_messaging_configured=False))
        self.assertEqual(response.status_code, 503)

    def test_bad_signature_is_400(self) -> None:
        response = self._post({"events": []}, signature="wrong")
        self.assertEqual(response.status_code, 400)

    def test_missing_signature_is_400(self) -> None:
        response = self._post({"events": []}, signature=None)
        self.assertEqual(response.status_code, 400)

    def test_valid_event_replies_and_returns_200(self) -> None:
        with mock.patch.object(
            line_bot, "handle_events", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
        ), mock.patch.object(line_bot, "reply") as reply:
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
        reply.assert_called_once_with("rt-1", "嗨")

    def test_non_text_event_returns_200_without_replying(self) -> None:
        with mock.patch.object(line_bot, "reply") as reply:
            response = self._post({"events": [{"type": "follow", "replyToken": "rt"}]})
        self.assertEqual(response.status_code, 200)
        reply.assert_not_called()

    def test_malformed_json_after_valid_signature_is_still_200(self) -> None:
        raw = b"not json"
        headers = {"X-Line-Signature": _sign(raw), "Content-Type": "application/json"}
        with mock.patch.object(
            line_webhook, "get_settings", return_value=_settings()
        ), mock.patch.object(line_bot, "get_settings", return_value=_settings()):
            with TestClient(app) as client:
                response = client.post("/api/line/webhook", content=raw, headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_reply_failure_is_still_200(self) -> None:
        with mock.patch.object(
            line_bot, "handle_events", return_value=[{"reply_token": "rt-1", "text": "嗨"}]
        ), mock.patch.object(line_bot, "reply", side_effect=RuntimeError("boom")):
            response = self._post({"events": [_text_event("摘要")]})
        self.assertEqual(response.status_code, 200)
```

- [ ] **Step 2: 執行測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -k Webhook -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.app.routers.line_webhook'`

- [ ] **Step 3a: 建立 router**

建立 `backend/app/routers/line_webhook.py`：

```python
"""POST /api/line/webhook — the LINE Messaging API bot's event sink.

The thin HTTP layer over ``services/line_bot`` (which documents the whole flow). Two things
are specific to this router and worth stating here:

* The handler is ``async`` and reads ``await request.body()`` because the signature must be
  computed over the RAW bytes — FastAPI's parsed model would not round-trip byte-identically.
* Once the signature checks out, it ALWAYS answers 200. LINE treats a non-2xx as a webhook
  failure, and the event has already been consumed, so a retry would only duplicate replies.

Returns 503 when the bot isn't configured, mirroring the rest of the API.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.services import line_bot
from backend.app.settings import get_settings

router = APIRouter(prefix="/api/line", tags=["line"])

logger = logging.getLogger(__name__)


@router.post("/webhook")
async def line_webhook(request: Request) -> dict[str, bool]:
    settings = get_settings()
    if not getattr(settings, "line_messaging_configured", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The LINE bot is not configured on the server.",
        )

    raw_body = await request.body()
    if not line_bot.verify_signature(raw_body, request.headers.get("x-line-signature")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid LINE signature.",
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))
        for planned in line_bot.handle_events(payload):
            line_bot.reply(planned["reply_token"], planned["text"])
    except Exception:  # noqa: BLE001 — a signed event is acknowledged no matter what.
        logger.exception("LINE bot: webhook handling failed")
    return {"ok": True}
```

- [ ] **Step 3b: 掛上 router**

`backend/app/main.py`：在 import 區的 `knowledge,` 之後加入 `line_webhook,`（維持字母序），
並在 `app.include_router(auth_line.router)` 之後加入：

```python
app.include_router(line_webhook.router)
```

- [ ] **Step 4: 執行測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_line_webhook.py -v`
Expected: PASS（43 passed）

- [ ] **Step 5: 跑整套測試與 coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Expected: 全綠（既有測試不受影響）

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: 通過。若未達標，先看報告中 `line_bot.py` / `line_webhook.py` 未覆蓋的行，補測試——
**不要**調低門檻。

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/line_webhook.py backend/app/main.py tests/test_backend_line_webhook.py
git commit -m "feat(line-bot): add the /api/line/webhook endpoint"
```

---

### Task 7: 文件

**Files:**
- Modify: `docs/line-login-liff-setup.md`
- Modify: `docs/line-login-liff-evaluation.md`（§3 第 3 點的 `sub` 說明）

**Interfaces:**
- Consumes: Task 1–6 的最終設定名稱與端點路徑
- Produces: 無程式介面

- [ ] **Step 1: 修正評估文件的錯誤事實**

在 `docs/line-login-liff-evaluation.md` 中，把描述 ID token `sub` 為
「**pairwise**（每個 channel 一組）」的那一行改為：

```
   - ID token 簽章：**ES256**；`sub` 為 **pairwise（每個 provider 一組，同 provider 下的
     channel 共用同一個 user ID）**——這正是 Messaging API bot 能直接用 webhook 的
     `source.userId` 對到登入帳號的原因，見
     https://developers.line.biz/en/docs/messaging-api/getting-user-ids/
```

- [ ] **Step 2: 在設定手冊加一節**

在 `docs/line-login-liff-setup.md` 末尾追加：

````markdown
## LINE Messaging API bot（聊天室查訓練摘要）

> 硬前提：Messaging API channel 必須建在**與 LINE Login channel 相同的 Provider** 底下。
> LINE 的 user ID 是每個 provider 一組，同 provider 下的 channel 共用同一個值——這是 bot
> 能用 webhook 的 `source.userId` 直接對到登入帳號、不需要綁定流程的唯一原因。

1. LINE Developers Console → 選到現有 Login channel 所屬的 Provider → **Create a new channel**
   → **Messaging API**（會同時建立一個 LINE 官方帳號）。
2. **Basic settings** → 複製 **Channel secret** → 填入 `.env` 的 `LINE_MESSAGING_CHANNEL_SECRET`。
3. **Messaging API** 分頁 → **Channel access token (long-lived)** → Issue → 填入
   `LINE_MESSAGING_ACCESS_TOKEN`。
4. 同一分頁 → **Webhook URL** 填 `https://<ngrok-host>/api/line/webhook` → 開啟 **Use webhook**
   → 按 **Verify**（後端要先啟動；未設定環境變數會回 503）。
5. 在 LINE Official Account Manager 關閉「自動回應訊息」與「加入好友的歡迎訊息」，
   否則會與 bot 的回覆打架。
6. **圖文選單（Rich menu）**：LINE Official Account Manager → 建立圖文選單 → 動作型別選
   **傳送文字**，文字填 `我的訓練摘要`。
   ⚠️ 不可選「連結（URI/LIFF）」——那不會觸發 webhook，也就拿不到 `replyToken`。
7. `.env` 的 `LINE_LIFF_ID` 填既有 LIFF app id（可留空，只是回覆訊息會少一個連結）。
8. 在 Supabase SQL editor 執行 `db/migrations/20260720000000_line_training_summary.sql`。

### 手動驗證

- 用 **service_role** key 呼叫 `select public.line_training_summary('<你的 LINE sub>')`：
  回傳 `{"total": ..., "latest": ..., "top_faults": [...]}`，數字與 x-coach History 頁一致。
- 用 **anon** key 呼叫同一支函式：應被 Postgres 拒絕（permission denied）。
- 沒登入過 x-coach 的 LINE 帳號敲 bot：得到引導登入的訊息，且 Supabase 的
  `auth.users` **沒有**多出任何一列。
````

- [ ] **Step 3: Commit**

```bash
git add docs/line-login-liff-setup.md docs/line-login-liff-evaluation.md
git commit -m "docs: document the LINE Messaging API bot setup"
```

---

## 完成後的驗收（對應 spec §10）

1. `.venv\Scripts\python.exe -m pytest tests/` 全綠。
2. `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` 通過。
3. ngrok 對外，點圖文選單收到自己的訓練摘要，數字與 History 頁一致。
4. 未登入過的 LINE 帳號敲 bot → 引導登入訊息，且沒有新增 `auth.users` 列。
5. anon key 呼叫 `line_training_summary` 被拒。
6. 竄改過的 request body 打 webhook → 400。
