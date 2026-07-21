# Admin LINE 狀態 + 推播額度面板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 admin 後台加一個唯讀面板,顯示 LINE 連線狀態(登入橋接 + Messaging bot)與本月 push 訊息已用/上限/剩餘,數字直接代理自 LINE 官方 quota API。

**Architecture:** 新 service `line_quota.py` 代理 LINE 的 `/message/quota` + `/quota/consumption`(唯一可測試接縫 + 60s TTL 快取);新端點 `GET /api/admin/line/status` 組裝連線旗標 + quota;前端在 `AdminOverview` 以獨立 fetch 渲染一組 LINE 卡片。全程唯讀、永不回傳密鑰。

**Tech Stack:** FastAPI + httpx(後端)、pydantic-settings、React 18 + TS + vitest(前端)、unittest(後端測試)。

## Global Constraints

- Python 直譯器一律 `.venv\Scripts\python.exe`(倉庫根目錄執行);後端測試 `.venv\Scripts\python.exe -m pytest tests/`。
- 後端 coverage gate:`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`(CI 強制 95%)。
- 前端所有 yarn/vitest 指令 cwd 必須是 `frontend/`;測試 `yarn test`、覆蓋率 `yarn test:coverage`。
- **面板永不讀寫任何密鑰**:LINE channel access token 只在後端 server-side 用來呼叫 LINE;回應只含非密鑰的 `channel_id`。
- 防禦性風格比照 `services/line_bot.reply`:任何 LINE 呼叫失敗都吞掉、絕不 raise/500。
- LINE `/message/quota` 的 `value` 是帳號在 LINE OA Manager 自設的每月上限;`type:"none"` 時無「剩餘」可算,只顯示已用 + 提示。
- 對外可見字串一律走既有 i18n `t(...)`,en 與 zh-Hant 兩份字典都要加 key。

---

## File Structure

- **Create** `backend/app/services/line_quota.py` — 代理 LINE quota/consumption API,組裝 `{type,used,[value,remaining]}`,60s TTL 快取。唯一對外函式 `fetch_quota()`。
- **Modify** `backend/app/routers/admin.py` — 新增 `GET /api/admin/line/status`(import `line_quota`)。
- **Create** `tests/test_backend_admin_line.py` — `line_quota` 單元測試 + 端點測試。
- **Modify** `frontend/src/api.ts` — 新增 `LineQuota`/`LineStatus` 型別與 `api.getLineStatus()`。
- **Modify** `frontend/src/lib/i18n.tsx` — 新增 `admin.line.*` key(en + zh-Hant)。
- **Modify** `frontend/src/pages/admin/AdminOverview.tsx` — 新增 `LineSection` 元件並渲染。
- **Modify** `frontend/src/test/pages.admin.test.tsx` — mock `getLineStatus`,新增 LINE 區塊測試。

---

## Task 1: `line_quota` service — 代理 LINE 額度 + 快取

**Files:**
- Create: `backend/app/services/line_quota.py`
- Test: `tests/test_backend_admin_line.py`

**Interfaces:**
- Consumes: `backend.app.settings.get_settings`(讀 `line_messaging_access_token`)。
- Produces:
  - `fetch_quota() -> dict[str, Any] | None` — 回 `{"type":"limited","used":int,"value":int,"remaining":int}` 或 `{"type":"none","used":int}` 或 `None`(未設 token / 任何失敗)。
  - `clear_cache() -> None` — 清 TTL 快取。
  - 常數 `LINE_QUOTA_URL`、`LINE_CONSUMPTION_URL`、模組層 `httpx`、`get_settings`(供測試 patch)。

- [ ] **Step 1: 寫失敗測試(單元)**

在新檔 `tests/test_backend_admin_line.py`:

```python
"""Unit tests for services/line_quota + GET /api/admin/line/status.

Mirrors tests/test_backend_line_webhook.py: unittest.TestCase, external HTTP (LINE's quota
endpoints) mocked at the httpx.get seam, get_settings patched to a lightweight stand-in, and
the FastAPI route exercised via TestClient with dependency_overrides + store.is_admin patched.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from backend.app.auth import CurrentUser, get_current_user
from backend.app.main import app
from backend.app.services import line_quota, store
from backend.app.settings import Settings


class _FakeResp:
    """A stand-in httpx.Response: .raise_for_status() optionally raises, .json() returns payload."""

    def __init__(self, payload, *, ok: bool = True) -> None:
        self._payload = payload
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            raise httpx.HTTPStatusError("bad", request=mock.Mock(), response=mock.Mock())

    def json(self):
        return self._payload


def _stub_settings(token: str = "chan-token") -> types.SimpleNamespace:
    return types.SimpleNamespace(line_messaging_access_token=token)


class LineQuotaFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        line_quota.clear_cache()
        self.addCleanup(line_quota.clear_cache)

    def _run(self, quota_payload, consumption_payload, *, token="chan-token"):
        responses = [_FakeResp(quota_payload), _FakeResp(consumption_payload)]
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings(token)), \
             mock.patch.object(line_quota.httpx, "get", side_effect=responses) as g:
            return line_quota.fetch_quota(), g

    def test_limited_computes_remaining(self) -> None:
        result, _ = self._run({"type": "limited", "value": 200}, {"totalUsage": 12})
        self.assertEqual(result, {"type": "limited", "used": 12, "value": 200, "remaining": 188})

    def test_none_type_omits_value_and_remaining(self) -> None:
        result, _ = self._run({"type": "none"}, {"totalUsage": 5})
        self.assertEqual(result, {"type": "none", "used": 5})

    def test_remaining_never_negative(self) -> None:
        result, _ = self._run({"type": "limited", "value": 100}, {"totalUsage": 150})
        self.assertEqual(result["remaining"], 0)

    def test_missing_token_returns_none_without_calling_line(self) -> None:
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings("")), \
             mock.patch.object(line_quota.httpx, "get") as g:
            self.assertIsNone(line_quota.fetch_quota())
        g.assert_not_called()

    def test_non_200_returns_none(self) -> None:
        responses = [_FakeResp({"type": "limited", "value": 200}, ok=False)]
        with mock.patch.object(line_quota, "get_settings", return_value=_stub_settings()), \
             mock.patch.object(line_quota.httpx, "get", side_effect=responses):
            self.assertIsNone(line_quota.fetch_quota())

    def test_malformed_consumption_returns_none(self) -> None:
        result, _ = self._run({"type": "limited", "value": 200}, {"totalUsage": "oops"})
        self.assertIsNone(result)

    def test_ttl_cache_hits_avoid_second_line_call(self) -> None:
        result1, g = self._run({"type": "limited", "value": 200}, {"totalUsage": 12})
        # Second call within TTL must NOT re-hit LINE (httpx.get called twice total for the first read).
        with mock.patch.object(line_quota.httpx, "get") as g2:
            result2 = line_quota.fetch_quota()
        self.assertEqual(result1, result2)
        g2.assert_not_called()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py::LineQuotaFetchTests -q`
Expected: FAIL — `ModuleNotFoundError: backend.app.services.line_quota`。

- [ ] **Step 3: 寫最小實作**

Create `backend/app/services/line_quota.py`:

```python
"""Read the LINE Messaging API push-message quota + this month's consumption (read-only).

Companion to services/line_bot: the bot SENDS (reply, which is free and uncounted); this module
READS the account's push quota so the admin panel can show "used / limit / remaining". The numbers
therefore reflect ONLY push/multicast/broadcast usage — exactly what LINE bills.

Two LINE endpoints, both authorised with the Messaging channel access token (server-side only;
never exposed to the browser):
    GET /v2/bot/message/quota             -> {"type": "none"} | {"type": "limited", "value": N}
    GET /v2/bot/message/quota/consumption -> {"totalUsage": N}

Defensive throughout (mirrors line_bot.reply): ANY failure — no token, network error, non-200,
malformed shape — returns None so the panel degrades to "unavailable" rather than raising. A short
process-wide TTL cache keeps admin refreshes from hammering LINE (its quota endpoints rate-limit).

``httpx`` is a top-level import (as in line_bot); ``get_settings`` is read through the module
namespace so tests can patch it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)

LINE_QUOTA_URL = "https://api.line.me/v2/bot/message/quota"
LINE_CONSUMPTION_URL = "https://api.line.me/v2/bot/message/quota/consumption"
_QUOTA_TIMEOUT_S = 10.0
_TTL_SECONDS = 60.0

# Process-wide snapshot: last fetched result (may be None) and when it was fetched.
_cache: dict[str, Any] | None = None
_cache_at: float = 0.0
_cache_valid: bool = False  # distinguishes "cached None" from "never fetched".


def _safe_int(value: Any) -> int | None:
    """int() or None — the LINE payload is trusted but not guaranteed well-formed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(url: str, token: str) -> dict[str, Any]:
    """GET a LINE quota endpoint; return its JSON dict. Raises on non-200 or non-dict payload."""
    resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_QUOTA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected LINE quota payload")
    return data


def _fetch() -> dict[str, Any] | None:
    """One live read of both endpoints -> the assembled quota dict, or None on any failure."""
    token = get_settings().line_messaging_access_token
    if not token:
        return None
    try:
        quota = _get(LINE_QUOTA_URL, token)
        consumption = _get(LINE_CONSUMPTION_URL, token)
    except Exception:  # noqa: BLE001 — any failure means "unavailable"; never propagate.
        logger.warning("LINE quota: read failed")
        return None

    used = _safe_int(consumption.get("totalUsage"))
    if used is None:
        return None
    result: dict[str, Any] = {
        "type": "limited" if quota.get("type") == "limited" else "none",
        "used": used,
    }
    if result["type"] == "limited":
        value = _safe_int(quota.get("value"))
        if value is None:
            return None
        result["value"] = value
        result["remaining"] = max(0, value - used)
    return result


def fetch_quota() -> dict[str, Any] | None:
    """Push-quota snapshot ({"type","used",[value,remaining]}) or None, served from a 60s TTL cache."""
    global _cache, _cache_at, _cache_valid
    now = time.monotonic()
    if _cache_valid and (now - _cache_at) < _TTL_SECONDS:
        return _cache
    _cache = _fetch()
    _cache_at = now
    _cache_valid = True
    return _cache


def clear_cache() -> None:
    """Invalidate the TTL cache so the next fetch_quota re-reads (used by tests)."""
    global _cache, _cache_at, _cache_valid
    _cache = None
    _cache_at = 0.0
    _cache_valid = False
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py::LineQuotaFetchTests -q`
Expected: PASS(7 passed）。

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/line_quota.py tests/test_backend_admin_line.py
git commit -m "feat(admin): line_quota service proxies LINE push-quota API"
```

---

## Task 2: `GET /api/admin/line/status` 端點

**Files:**
- Modify: `backend/app/routers/admin.py`(imports + 新 route,加在 `admin_overview` 之後)
- Test: `tests/test_backend_admin_line.py`(新增端點測試類)

**Interfaces:**
- Consumes: `line_quota.fetch_quota()`(Task 1)、`settings.get_settings()` 的 `line_messaging_configured`/`line_login_configured`/`line_channel_id`、`auth.get_admin_user`。
- Produces: `GET /api/admin/line/status` → JSON `{messaging_configured, login_configured, channel_id, quota, quota_error}`,其中 `quota` 是 Task 1 的 dict 或 `null`,`quota_error` 為 `"unreachable"` 或 `null`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_backend_admin_line.py` 末端追加:

```python
def _settings(**overrides) -> Settings:
    """A real Settings whose LINE properties compute correctly (mirrors the webhook test helper)."""
    values = {
        "supabase_url": "https://proj.supabase.co",
        "supabase_anon_key": "anon-key",
        "supabase_service_role_key": "service-key",
        "line_channel_id": "2010629653",
        "line_messaging_channel_secret": "secret",
        "line_messaging_access_token": "token",
    }
    values.update(overrides)
    return Settings(**values)


class AdminLineStatusRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)
        line_quota.clear_cache()
        self.addCleanup(line_quota.clear_cache)

    def _get(self, *, quota, settings_obj):
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_quota, "fetch_quota", return_value=quota):
            return self.client.get("/api/admin/line/status")

    def test_configured_with_limited_quota(self) -> None:
        quota = {"type": "limited", "used": 12, "value": 200, "remaining": 188}
        resp = self._get(quota=quota, settings_obj=self._settings())
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["messaging_configured"])
        self.assertTrue(body["login_configured"])
        self.assertEqual(body["channel_id"], "2010629653")
        self.assertEqual(body["quota"], quota)
        self.assertIsNone(body["quota_error"])

    def test_configured_but_line_unreachable_sets_error(self) -> None:
        resp = self._get(quota=None, settings_obj=self._settings())
        body = resp.json()
        self.assertIsNone(body["quota"])
        self.assertEqual(body["quota_error"], "unreachable")

    def test_not_configured_skips_line_and_has_no_error(self) -> None:
        settings_obj = self._settings(line_messaging_access_token="")  # -> messaging_configured False
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_quota, "fetch_quota") as fq:
            resp = self.client.get("/api/admin/line/status")
        body = resp.json()
        self.assertFalse(body["messaging_configured"])
        self.assertIsNone(body["quota"])
        self.assertIsNone(body["quota_error"])
        fq.assert_not_called()  # unconfigured => never call LINE

    def test_response_carries_no_secret(self) -> None:
        resp = self._get(quota={"type": "none", "used": 3}, settings_obj=self._settings())
        blob = resp.text
        self.assertNotIn("token", blob)   # access token / channel secret never serialised
        self.assertNotIn("secret", blob)
        self.assertNotIn("service-key", blob)

    def test_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.get("/api/admin/line/status")
        self.assertEqual(resp.status_code, 403)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop override -> real dependency runs
        resp = self.client.get("/api/admin/line/status")
        self.assertEqual(resp.status_code, 401)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py::AdminLineStatusRouteTests -q`
Expected: FAIL — 404(route 尚未存在)。

- [ ] **Step 3: 寫最小實作**

在 `backend/app/routers/admin.py` 的 import 區,把 service import 改為含 `line_quota`:

```python
from backend.app.services import line_quota, runtime_config, store
```

在 `admin_overview` 函式之後(檔案末端)新增:

```python
@router.get("/line/status")
def admin_line_status(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """LINE connection status + this month's push-message quota (admin-only; read-only).

    Never returns a secret: the channel access token is used server-side (in ``line_quota``) to
    read LINE's quota endpoints, and the channel secret / service_role key are never touched.
    ``channel_id`` is the non-secret LINE Login channel id, surfaced only so an admin can confirm
    which channel is wired. When messaging isn't configured we skip the LINE call entirely; when it
    is configured but the read fails, ``quota`` is ``None`` and ``quota_error`` flags it.
    """
    s = settings.get_settings()
    quota = line_quota.fetch_quota() if s.line_messaging_configured else None
    return {
        "messaging_configured": s.line_messaging_configured,
        "login_configured": s.line_login_configured,
        "channel_id": s.line_channel_id,
        "quota": quota,
        "quota_error": "unreachable" if (s.line_messaging_configured and quota is None) else None,
    }
```

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: PASS(全部,含 Task 1 的 7 個 + 端點 6 個)。

- [ ] **Step 5: 後端覆蓋率關卡**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS(≥95%)。若 `line_quota` 有未覆蓋行,補對應測試再跑。

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/admin.py tests/test_backend_admin_line.py
git commit -m "feat(admin): GET /api/admin/line/status endpoint"
```

---

## Task 3: 前端 LINE 狀態卡片

**Files:**
- Modify: `frontend/src/api.ts`(型別 + `getLineStatus`)
- Modify: `frontend/src/lib/i18n.tsx`(`admin.line.*` key,en + zh-Hant)
- Modify: `frontend/src/pages/admin/AdminOverview.tsx`(`LineSection` 元件 + 渲染)
- Test: `frontend/src/test/pages.admin.test.tsx`

**Interfaces:**
- Consumes: `GET /api/admin/line/status`(Task 2)。
- Produces:
  - `api.getLineStatus(): Promise<LineStatus>`。
  - 型別 `LineStatus { messaging_configured, login_configured, channel_id, quota: LineQuota | null, quota_error: "unreachable" | null }`,`LineQuota { type: "limited" | "none"; used: number; value?: number; remaining?: number }`。

- [ ] **Step 1: 寫失敗測試**

先在 `frontend/src/test/pages.admin.test.tsx` 的 import 加 `type LineStatus`,並在 `SAMPLE_OVERVIEW` 之後新增樣本 + 在 `beforeEach` mock:

```tsx
// (import 區)
import {
  api,
  type AdminOverview,
  type AdminSettingsResponse,
  type AdminUserRow,
  type LineStatus,
} from "../api";

const SAMPLE_LINE_STATUS: LineStatus = {
  messaging_configured: true,
  login_configured: true,
  channel_id: "2010629653",
  quota: { type: "limited", used: 12, value: 200, remaining: 188 },
  quota_error: null,
};
```

在 `beforeEach` 內(既有三個 spyOn 之後)加:

```tsx
  vi.spyOn(api, "getLineStatus").mockResolvedValue(SAMPLE_LINE_STATUS);
```

在 `describe("AdminOverview", ...)` 區塊內新增測試:

```tsx
  it("renders the LINE quota cards when messaging is configured with a limit", async () => {
    renderAdmin("/admin");
    expect(await screen.findByText("Push used this month")).toBeInTheDocument();
    expect(screen.getByText("12 / 200")).toBeInTheDocument();
    expect(screen.getByText("Free remaining")).toBeInTheDocument();
    expect(screen.getByText("188")).toBeInTheDocument();
  });

  it("shows a dash and the no-cap note when the account has no monthly limit", async () => {
    vi.spyOn(api, "getLineStatus").mockResolvedValue({
      ...SAMPLE_LINE_STATUS,
      quota: { type: "none", used: 7 },
    });
    renderAdmin("/admin");
    expect(await screen.findByText("7")).toBeInTheDocument(); // used, no "/ limit"
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(
      screen.getByText(/No monthly limit set in LINE Official Account Manager/i)
    ).toBeInTheDocument();
  });

  it("shows the unreachable note when LINE can't be reached for quota", async () => {
    vi.spyOn(api, "getLineStatus").mockResolvedValue({
      ...SAMPLE_LINE_STATUS,
      quota: null,
      quota_error: "unreachable",
    });
    renderAdmin("/admin");
    expect(await screen.findByText("Couldn't reach LINE for quota.")).toBeInTheDocument();
  });

  it("does not break the overview when the LINE status fetch fails", async () => {
    vi.spyOn(api, "getLineStatus").mockRejectedValue(new Error("500 boom"));
    renderAdmin("/admin");
    // Main overview still renders; the LINE section simply renders nothing.
    expect(await screen.findByText("Total users")).toBeInTheDocument();
    expect(screen.queryByText("Push used this month")).not.toBeInTheDocument();
  });
```

- [ ] **Step 2: 跑測試確認失敗**

Run(cwd=`frontend/`): `yarn test src/test/pages.admin.test.tsx`
Expected: FAIL — `api.getLineStatus is not a function` / 型別 `LineStatus` 不存在。

- [ ] **Step 3: api.ts 加型別與方法**

在 `frontend/src/api.ts` 的 `AdminOverview` interface 之後新增:

```ts
// LINE connection status + this month's push-message quota (admin-only; read-only). No secret here.
export interface LineQuota {
  type: "limited" | "none";
  used: number;
  value?: number; // present only when type === "limited"
  remaining?: number; // present only when type === "limited"
}
export interface LineStatus {
  messaging_configured: boolean;
  login_configured: boolean;
  channel_id: string;
  quota: LineQuota | null;
  quota_error: "unreachable" | null;
}
```

在 `api` 物件內、`getAdminOverview` 之後新增:

```ts
  // LINE connection status + push-quota usage (admin-only). Auth header auto-attached.
  getLineStatus: () => getJSON<LineStatus>("/api/admin/line/status"),
```

- [ ] **Step 4: i18n 加 key(兩份字典)**

在 `frontend/src/lib/i18n.tsx` 的 en 字典 `"admin.overview.totalAnalyses"` 那行之後插入:

```ts
  "admin.line.title": "LINE",
  "admin.line.loginBridge": "LINE login bridge",
  "admin.line.bot": "LINE bot",
  "admin.line.pushUsed": "Push used this month",
  "admin.line.remaining": "Free remaining",
  "admin.line.unreachable": "Couldn't reach LINE for quota.",
  "admin.line.noCapNote":
    "No monthly limit set in LINE Official Account Manager, so remaining can't be shown. Set a monthly message limit there to track your free allowance.",
```

在 zh-Hant 字典 `"admin.overview.totalAnalyses"` 對應行之後插入:

```ts
  "admin.line.title": "LINE",
  "admin.line.loginBridge": "LINE 登入橋接",
  "admin.line.bot": "LINE Bot",
  "admin.line.pushUsed": "本月推播已用",
  "admin.line.remaining": "剩餘免費額度",
  "admin.line.unreachable": "無法取得 LINE 額度。",
  "admin.line.noCapNote":
    "未在 LINE Official Account Manager 設定每月上限,無法計算剩餘。請在該後台設定每月訊息上限,即可追蹤免費額度。",
```

> 註:若 en 字典沒有現成的 `admin.overview.totalAnalyses` 錨點行,改插在該 en 字典 `admin.overview.*` 區塊尾端即可;zh-Hant 同理。key 名稱兩份必須一致。

- [ ] **Step 5: AdminOverview 加 LineSection**

在 `frontend/src/pages/admin/AdminOverview.tsx`:

(a) import 補上 icon 與型別:

```tsx
import {
  Brain,
  ChatCircleText,
  Database,
  Gauge,
  ShieldCheck,
  SlidersHorizontal,
  Users,
  WarningCircle,
} from "@phosphor-icons/react";
import { api, type AdminOverview as AdminOverviewData, type LineStatus } from "../../api";
```

(b) 在主元件 return 的最外層 `</div>` 之前、overview grid 之後,渲染 `<LineSection />`:

```tsx
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {/* ...既有的 OverviewCard... */}
      </div>
      <LineSection />
    </div>
  );
}
```

(c) 在檔案末端(`OverviewCard` 之後)新增 `LineSection`:

```tsx
// LINE connection status + push-quota, fetched independently so a slow/failed LINE call never
// blocks or breaks the main overview. On error the section renders nothing (the page stays intact).
function LineSection() {
  const { t } = useI18n();
  const [data, setData] = useState<LineStatus | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  useEffect(() => {
    let active = true;
    api
      .getLineStatus()
      .then((res) => {
        if (!active) return;
        setData(res);
        setStatus("ready");
      })
      .catch(() => active && setStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  if (status !== "ready" || !data) return null;

  const q = data.quota;
  const limited = q?.type === "limited";

  return (
    <div className="mt-8">
      <div className="flex items-center gap-2">
        <ChatCircleText size={18} weight="duotone" className="text-primary" />
        <h2 className="text-sm font-semibold text-content">{t("admin.line.title")}</h2>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <OverviewCard
          icon={<ShieldCheck size={16} weight="duotone" />}
          label={t("admin.line.loginBridge")}
          value={
            data.login_configured
              ? t("admin.overview.configured")
              : t("admin.overview.notConfigured")
          }
          ok={data.login_configured}
        />
        <OverviewCard
          icon={<ChatCircleText size={16} weight="duotone" />}
          label={t("admin.line.bot")}
          value={
            data.messaging_configured
              ? t("admin.overview.configured")
              : t("admin.overview.notConfigured")
          }
          ok={data.messaging_configured}
        />
        {data.quota_error === "unreachable" ? (
          <div className="rounded-2xl border border-border-dark bg-surface-dark p-4">
            <p className="text-xs text-muted">{t("admin.line.unreachable")}</p>
          </div>
        ) : q ? (
          <>
            <OverviewCard
              icon={<Gauge size={16} weight="duotone" />}
              label={t("admin.line.pushUsed")}
              value={limited ? `${q.used} / ${q.value}` : String(q.used)}
            />
            <OverviewCard
              icon={<SlidersHorizontal size={16} weight="duotone" />}
              label={t("admin.line.remaining")}
              value={limited ? String(q.remaining) : "—"}
            />
          </>
        ) : null}
      </div>
      {q && !limited ? <p className="mt-3 text-xs text-muted">{t("admin.line.noCapNote")}</p> : null}
    </div>
  );
}
```

- [ ] **Step 6: 跑測試確認通過**

Run(cwd=`frontend/`): `yarn test src/test/pages.admin.test.tsx`
Expected: PASS(既有 + 4 個新測試)。

- [ ] **Step 7: 前端覆蓋率 + build**

Run(cwd=`frontend/`): `yarn test:coverage`
Expected: PASS。
Run(cwd=`frontend/`): `yarn build`
Expected: build 成功(TS 無型別錯)。

- [ ] **Step 8: Commit**

```bash
git add frontend/src/api.ts frontend/src/lib/i18n.tsx frontend/src/pages/admin/AdminOverview.tsx frontend/src/test/pages.admin.test.tsx
git commit -m "feat(admin): LINE status + push-quota cards on the overview page"
```

---

## Self-Review

**Spec coverage:**
- §3.1 line_quota service → Task 1 ✅
- §3.2 端點 + quota_error 邏輯 → Task 2 ✅
- §3.3 前端 api/i18n/UI → Task 3 ✅
- §4 錯誤處理總表(未設定/unreachable/none/limited/非admin)→ Task 2 測試 + Task 3 測試全數涵蓋 ✅
- §5 測試(後端 unit+端點、前端各情境、覆蓋率關卡)→ 各 Task 的測試步驟 ✅
- §6 YAGNI(不自建計數、不編輯密鑰)→ 未納入任何 Task ✅

**Placeholder scan:** 無 TBD/TODO;所有 code step 皆含完整程式碼與確切指令。i18n 錨點行有 fallback 說明。

**Type consistency:** `fetch_quota`/`clear_cache`/`LINE_QUOTA_URL`(Task 1)在 Task 2 一致使用;`LineStatus`/`LineQuota`/`getLineStatus`(Task 3 定義)在測試與元件一致;`quota_error` 值 `"unreachable"` 前後端一致;`type: "limited"|"none"` 前後端一致。

**發現並修正:** 端點測試 `test_response_carries_no_secret` 斷言回應不含 `token`/`secret`/`service-key`——`channel_id` 用純數字 `2010629653` 不含這些子字串,斷言安全。
