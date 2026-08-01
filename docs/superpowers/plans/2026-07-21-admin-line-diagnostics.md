# Admin LINE 診斷面板擴充 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有 LINE admin 面板上加 bot info、webhook 健康(含手動測試按鈕)、昨日 reply/push 發送數三塊唯讀資訊,全部代理自 LINE Messaging API。

**Architecture:** 把 `line_quota.py` 改名 `line_admin.py` 並泛化其快取為 key 化;新增 3 個唯讀 fetch(bot info / webhook / delivery)併進 `GET /api/admin/line/status`;webhook 實測(有副作用的 POST)獨立為 `POST /api/admin/line/webhook-test` + 前端按鈕。全程唯讀、只用 Messaging token server-side、回應不含密鑰。

**Tech Stack:** FastAPI + httpx(後端)、pydantic-settings、React 18 + TS + vitest(前端)、unittest(後端測試)。

## Global Constraints

- Python 直譯器一律 `.venv\Scripts\python.exe`(倉庫根執行);後端測試 `.venv\Scripts\python.exe -m pytest tests/...`。
- 後端 coverage gate:`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`(CI 強制 95%)。
- 前端所有 yarn/vitest 指令 cwd 必須是 `frontend/`;`yarn test` / `yarn test:coverage` / `yarn build`(build 嚴格,unused import/local 會失敗)。
- **永不讀寫或回傳任何密鑰**:Messaging channel access token 只在後端 server-side 用;回應只含非密鑰欄位。
- 防禦性風格比照 `services/line_bot.reply`:任何 LINE 呼叫失敗都吞掉(logged)、回 `None`、絕不 raise/500。
- 唯讀讀取走 60s TTL 快取;`webhook/test` 是有副作用的 POST,**不快取、只在按鈕點擊時呼叫**。
- `delivery` 抓昨天(UTC+8);LINE 對缺資料回 `status!="ready"` → 該值 `None`。
- 回應 JSON 用 snake_case(`display_name`/`basic_id`/`status_code`…),前端型別對齊之。
- 對外可見字串走既有 i18n `t(...)`,en 與 zh-Hant 兩份都要加、key 名一致。

---

## File Structure

- **Rename** `backend/app/services/line_quota.py` → `backend/app/services/line_admin.py` — 泛化快取 + 新增 fetch。
- **Modify** `backend/app/routers/admin.py` — import 改名;`/line/status` 擴充;新增 `POST /line/webhook-test`。
- **Modify** `tests/test_backend_admin_line.py` — 改名引用 + 新測試。
- **Modify** `frontend/src/api.ts` — 新型別 + 擴充 `LineStatus` + `testLineWebhook`。
- **Modify** `frontend/src/lib/i18n.tsx` — 新 `admin.line.*` key(en + zh-Hant)。
- **Modify** `frontend/src/pages/admin/AdminOverview.tsx` — `LineSection` 加三張卡 + 測試按鈕。
- **Modify** `frontend/src/test/pages.admin.test.tsx` — 新測試 + 更新樣本。

---

## Task 1: 改名 `line_quota` → `line_admin` + 泛化快取(行為不變)

**Files:**
- Rename: `backend/app/services/line_quota.py` → `backend/app/services/line_admin.py`
- Modify: `backend/app/routers/admin.py`(import + 呼叫)
- Test: `tests/test_backend_admin_line.py`(改名引用)

**Interfaces:**
- Produces: 模組 `backend.app.services.line_admin`,公開 `fetch_quota() -> dict|None`、`clear_cache() -> None`、helper `_get(url, token)`、`_safe_int(value)`、常數 `LINE_QUOTA_URL`/`LINE_CONSUMPTION_URL`/`_TIMEOUT_S`/`_TTL_SECONDS`、key 化快取 `_cached(key, producer)`。行為與改名前 100% 相同。

- [ ] **Step 1: 改名檔案 + 更新引用(先讓測試紅)**

先只改測試與 admin.py 的引用,讓測試因找不到 `line_admin` 而紅:

```bash
git mv backend/app/services/line_quota.py backend/app/services/line_admin.py
```

在 `backend/app/routers/admin.py`:把 `from backend.app.services import line_quota, runtime_config, store` 改為 `line_admin`,並把 `line_quota.fetch_quota()` 改為 `line_admin.fetch_quota()`。

在 `tests/test_backend_admin_line.py`:將檔內所有 `line_quota` 替換為 `line_admin`(import、`line_admin.clear_cache()`、`mock.patch.object(line_admin, ...)`、`mock.patch.object(line_admin.httpx, "get", ...)`、`line_admin.fetch_quota()` 等)。

- [ ] **Step 2: 跑測試確認紅**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: FAIL —(此時 `line_admin.py` 內部尚未泛化,但檔已存在;若已可 import 則測試可能綠。真正的紅點是下一步的 `_cached` 尚未存在——若本步已綠,直接進 Step 3 重構並保持綠。）

- [ ] **Step 3: 泛化 `line_admin.py` 的快取 + 改名內部符號**

編輯 `backend/app/services/line_admin.py`。(a) 更新模組 docstring 首段為:

```python
"""Read the LINE Messaging API for the admin panel (read-only).

Companion to services/line_bot (which SENDS): this module READS account state for the admin
diagnostics panel — push-message quota, bot info, webhook health, and daily delivery counts.
Every read uses the Messaging channel access token server-side only (never exposed to the browser).

Defensive throughout (mirrors line_bot.reply): ANY failure — no token, network error, non-200,
malformed shape — returns None so the panel degrades to "unavailable" rather than raising. Read-only
lookups share a keyed 60s TTL cache so admin refreshes don't hammer LINE (its endpoints rate-limit).

``httpx`` is a top-level import (as in line_bot); ``get_settings`` is read through the module
namespace so tests can patch it.
"""
```

(b) 把 `_QUOTA_TIMEOUT_S = 10.0` 改名為 `_TIMEOUT_S = 10.0`,並更新 `_get` 內的引用(`timeout=_TIMEOUT_S`)。

(c) 用 key 化快取取代原本的 `_cache`/`_cache_at`/`_cache_valid` 三個全域與 `fetch_quota`/`clear_cache`。將原 `def _fetch()` 改名為 `def _fetch_quota()`(函式體不變)。新的快取區塊:

```python
_TTL_SECONDS = 60.0

# Keyed process-wide TTL cache: values may be None (a real "unavailable" result worth caching for the
# window); ``key in _cache_at`` distinguishes cached-None from never-fetched.
_cache: dict[str, dict[str, Any] | None] = {}
_cache_at: dict[str, float] = {}


def _cached(key: str, producer) -> dict[str, Any] | None:
    """Serve ``key`` from the 60s TTL cache, calling ``producer()`` on miss/expiry."""
    now = time.monotonic()
    if key in _cache_at and (now - _cache_at[key]) < _TTL_SECONDS:
        return _cache[key]
    result = producer()
    _cache[key] = result
    _cache_at[key] = now
    return result


def fetch_quota() -> dict[str, Any] | None:
    """Push-quota snapshot ({"type","used",[value,remaining]}) or None, served from the 60s TTL cache."""
    return _cached("quota", _fetch_quota)


def clear_cache() -> None:
    """Invalidate the whole TTL cache so the next read re-fetches (used by tests)."""
    _cache.clear()
    _cache_at.clear()
```

- [ ] **Step 4: 跑測試確認全綠**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: PASS(既有 quota + 端點測試全過,行為未變)。

- [ ] **Step 5: 覆蓋率關卡**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS(≥95%)。若因 `_cached`/新結構有未覆蓋分支,補一個「同 key 第二次命中快取」的測試(見 Task 2 也會加)。

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/line_admin.py backend/app/routers/admin.py tests/test_backend_admin_line.py
git commit -m "refactor(admin): rename line_quota -> line_admin, keyed TTL cache"
```

---

## Task 2: bot info / webhook / delivery fetchers + `/line/status` 擴充

**Files:**
- Modify: `backend/app/services/line_admin.py`(新常數、`_token`、`_yesterday_yyyymmdd`、三個 `_fetch_*` + cached wrappers)
- Modify: `backend/app/routers/admin.py`(`/line/status` 加三個 key)
- Test: `tests/test_backend_admin_line.py`

**Interfaces:**
- Consumes: `line_admin._get`、`_safe_int`、`_cached`、`get_settings`(Task 1)。
- Produces:
  - `fetch_bot_info() -> dict|None` → `{"display_name","basic_id","premium_id","chat_mode","mark_as_read_mode"}`
  - `fetch_webhook() -> dict|None` → `{"endpoint": str, "active": bool}`
  - `fetch_delivery() -> dict|None` → `{"date": str(yyyymmdd), "reply": int|None, "push": int|None}`
  - `_yesterday_yyyymmdd() -> str`(測試 patch 點)、`_token() -> str`
  - `GET /api/admin/line/status` 回應新增 `bot_info`/`webhook`/`delivery`(configured 時各為 dict|None,未設定時皆 None)。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_backend_admin_line.py` 追加(沿用檔內既有的 `_FakeResp`、`mock`、`httpx` import):

```python
class LineBotInfoWebhookDeliveryTests(unittest.TestCase):
    def setUp(self) -> None:
        line_admin.clear_cache()
        self.addCleanup(line_admin.clear_cache)

    def _patch_get(self, responses):
        return mock.patch.object(line_admin.httpx, "get", side_effect=responses)

    def _settings_stub(self, token="chan-token"):
        return mock.patch.object(line_admin, "get_settings",
                                 return_value=types.SimpleNamespace(line_messaging_access_token=token))

    def test_bot_info_happy(self) -> None:
        payload = {"displayName": "x-coach", "basicId": "@xcoach", "premiumId": None,
                   "chatMode": "bot", "markAsReadMode": "auto"}
        with self._settings_stub(), self._patch_get([_FakeResp(payload)]):
            self.assertEqual(line_admin.fetch_bot_info(), {
                "display_name": "x-coach", "basic_id": "@xcoach", "premium_id": None,
                "chat_mode": "bot", "mark_as_read_mode": "auto"})

    def test_bot_info_no_token_returns_none(self) -> None:
        with self._settings_stub(token=""), self._patch_get([]) as g:
            self.assertIsNone(line_admin.fetch_bot_info())
        g.assert_not_called()

    def test_bot_info_non_200_returns_none(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({}, ok=False)]):
            self.assertIsNone(line_admin.fetch_bot_info())

    def test_webhook_happy(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"endpoint": "https://x/api/line/webhook", "active": True})]):
            self.assertEqual(line_admin.fetch_webhook(), {"endpoint": "https://x/api/line/webhook", "active": True})

    def test_webhook_missing_endpoint_returns_none(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"active": True})]):
            self.assertIsNone(line_admin.fetch_webhook())

    def test_delivery_ready_counts(self) -> None:
        with self._settings_stub(), \
             mock.patch.object(line_admin, "_yesterday_yyyymmdd", return_value="20260720"), \
             self._patch_get([_FakeResp({"status": "ready", "success": 12}),
                              _FakeResp({"status": "ready", "success": 3})]):
            self.assertEqual(line_admin.fetch_delivery(), {"date": "20260720", "reply": 12, "push": 3})

    def test_delivery_unready_yields_none_counts(self) -> None:
        with self._settings_stub(), \
             mock.patch.object(line_admin, "_yesterday_yyyymmdd", return_value="20260720"), \
             self._patch_get([_FakeResp({"status": "unready"}), _FakeResp({"status": "unready"})]):
            self.assertEqual(line_admin.fetch_delivery(), {"date": "20260720", "reply": None, "push": None})

    def test_readonly_cache_hits_avoid_second_call(self) -> None:
        with self._settings_stub(), self._patch_get([_FakeResp({"endpoint": "https://x", "active": False})]):
            first = line_admin.fetch_webhook()
        with mock.patch.object(line_admin.httpx, "get") as g2:
            second = line_admin.fetch_webhook()
        self.assertEqual(first, second)
        g2.assert_not_called()
```

在既有 `AdminLineStatusRouteTests` 追加(class 內)三個 key 的擴充驗證。先確認檔頭已 import `line_admin`;`_settings()` helper 已存在:

```python
    def test_status_includes_bot_info_webhook_delivery(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=self._settings()), \
             mock.patch.object(line_admin, "fetch_quota", return_value={"type": "none", "used": 1}), \
             mock.patch.object(line_admin, "fetch_bot_info", return_value={"display_name": "x", "basic_id": "@x", "premium_id": None, "chat_mode": "bot", "mark_as_read_mode": "auto"}), \
             mock.patch.object(line_admin, "fetch_webhook", return_value={"endpoint": "https://x", "active": True}), \
             mock.patch.object(line_admin, "fetch_delivery", return_value={"date": "20260720", "reply": 4, "push": 0}):
            body = self.client.get("/api/admin/line/status").json()
        self.assertEqual(body["bot_info"]["chat_mode"], "bot")
        self.assertTrue(body["webhook"]["active"])
        self.assertEqual(body["delivery"]["reply"], 4)

    def test_status_not_configured_nulls_new_keys_and_skips_line(self) -> None:
        settings_obj = self._settings(line_messaging_access_token="")
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=settings_obj), \
             mock.patch.object(line_admin, "fetch_bot_info") as bi, \
             mock.patch.object(line_admin, "fetch_webhook") as wh, \
             mock.patch.object(line_admin, "fetch_delivery") as dl:
            body = self.client.get("/api/admin/line/status").json()
        self.assertIsNone(body["bot_info"])
        self.assertIsNone(body["webhook"])
        self.assertIsNone(body["delivery"])
        bi.assert_not_called(); wh.assert_not_called(); dl.assert_not_called()
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'fetch_bot_info'` / 端點回應缺 key。

- [ ] **Step 3: 實作 fetchers**

在 `backend/app/services/line_admin.py`,先擴充 import:

```python
from datetime import datetime, timedelta, timezone
```

在常數區加:

```python
LINE_BOT_INFO_URL = "https://api.line.me/v2/bot/info"
LINE_WEBHOOK_ENDPOINT_URL = "https://api.line.me/v2/bot/channel/webhook/endpoint"
LINE_DELIVERY_REPLY_URL = "https://api.line.me/v2/bot/message/delivery/reply"
LINE_DELIVERY_PUSH_URL = "https://api.line.me/v2/bot/message/delivery/push"

# Delivery counts are per-day and only complete the day after, so we always read YESTERDAY.
_DISPLAY_TZ = timezone(timedelta(hours=8))  # LINE OA account timezone (Taiwan), matches services/line_bot.
```

在檔案（`_fetch_quota` 之後）加:

```python
def _token() -> str:
    """The Messaging channel access token (empty string when unconfigured)."""
    return get_settings().line_messaging_access_token


def _yesterday_yyyymmdd() -> str:
    """Yesterday in the OA timezone as ``yyyymmdd`` — the newest day LINE has complete delivery data for."""
    return (datetime.now(_DISPLAY_TZ) - timedelta(days=1)).strftime("%Y%m%d")


def _fetch_bot_info() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    try:
        data = _get(LINE_BOT_INFO_URL, token)
    except Exception:  # noqa: BLE001 — any failure means "unavailable"; never propagate.
        logger.warning("LINE bot info: read failed")
        return None
    return {
        "display_name": str(data.get("displayName") or ""),
        "basic_id": str(data.get("basicId") or ""),
        "premium_id": data.get("premiumId") if isinstance(data.get("premiumId"), str) else None,
        "chat_mode": str(data.get("chatMode") or ""),
        "mark_as_read_mode": str(data.get("markAsReadMode") or ""),
    }


def _fetch_webhook() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    try:
        data = _get(LINE_WEBHOOK_ENDPOINT_URL, token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE webhook endpoint: read failed")
        return None
    endpoint = data.get("endpoint")
    if not isinstance(endpoint, str):
        return None
    return {"endpoint": endpoint, "active": bool(data.get("active"))}


def _fetch_delivery() -> dict[str, Any] | None:
    token = _token()
    if not token:
        return None
    date = _yesterday_yyyymmdd()
    try:
        reply = _get(f"{LINE_DELIVERY_REPLY_URL}?date={date}", token)
        push = _get(f"{LINE_DELIVERY_PUSH_URL}?date={date}", token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE delivery: read failed")
        return None
    # ``success`` is present only when status == "ready"; _safe_int -> None otherwise.
    return {"date": date, "reply": _safe_int(reply.get("success")), "push": _safe_int(push.get("success"))}


def fetch_bot_info() -> dict[str, Any] | None:
    """OA display name / basic id / chat mode, or None. 60s TTL cache."""
    return _cached("bot_info", _fetch_bot_info)


def fetch_webhook() -> dict[str, Any] | None:
    """Configured webhook endpoint + active flag, or None. 60s TTL cache."""
    return _cached("webhook", _fetch_webhook)


def fetch_delivery() -> dict[str, Any] | None:
    """Yesterday's reply/push delivery counts, or None. 60s TTL cache."""
    return _cached("delivery", _fetch_delivery)
```

- [ ] **Step 4: 擴充 `/line/status` 端點**

在 `backend/app/routers/admin.py` 的 `admin_line_status` 內,改為(沿用既有 docstring,補一句涵蓋新 key):

```python
    s = settings.get_settings()
    configured = s.line_messaging_configured
    quota = line_admin.fetch_quota() if configured else None
    return {
        "messaging_configured": configured,
        "login_configured": s.line_login_configured,
        "channel_id": s.line_channel_id,
        "quota": quota,
        "quota_error": "unreachable" if (configured and quota is None) else None,
        "bot_info": line_admin.fetch_bot_info() if configured else None,
        "webhook": line_admin.fetch_webhook() if configured else None,
        "delivery": line_admin.fetch_delivery() if configured else None,
    }
```

- [ ] **Step 5: 跑測試確認通過 + 覆蓋率**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: PASS。
Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS(≥95%;若某 `_fetch_*` 有未覆蓋分支補測試)。

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/line_admin.py backend/app/routers/admin.py tests/test_backend_admin_line.py
git commit -m "feat(admin): LINE bot info + webhook + delivery in /line/status"
```

---

## Task 3: `POST /api/admin/line/webhook-test`

**Files:**
- Modify: `backend/app/services/line_admin.py`(`_post` + `test_webhook`)
- Modify: `backend/app/routers/admin.py`(新端點)
- Test: `tests/test_backend_admin_line.py`

**Interfaces:**
- Consumes: `line_admin._token`、`_safe_int`、`httpx`(Task 1/2)。
- Produces:
  - `test_webhook() -> dict|None` → `{"success": bool, "status_code": int|None, "reason": str|None, "detail": str|None}`,transport 失敗回 None。**不快取。**
  - `POST /api/admin/line/webhook-test`(admin-only)→ `{"result": dict|None, "error": "not_configured"|"unreachable"|None}`。

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_backend_admin_line.py` 追加:

```python
class LineWebhookTestTests(unittest.TestCase):
    def _settings_stub(self, token="chan-token"):
        return mock.patch.object(line_admin, "get_settings",
                                 return_value=types.SimpleNamespace(line_messaging_access_token=token))

    def test_success_result(self) -> None:
        payload = {"success": True, "statusCode": 200, "reason": "OK", "detail": "200"}
        with self._settings_stub(), mock.patch.object(line_admin.httpx, "post", return_value=_FakeResp(payload)):
            self.assertEqual(line_admin.test_webhook(),
                             {"success": True, "status_code": 200, "reason": "OK", "detail": "200"})

    def test_transport_failure_returns_none(self) -> None:
        with self._settings_stub(), mock.patch.object(line_admin.httpx, "post", return_value=_FakeResp({}, ok=False)):
            self.assertIsNone(line_admin.test_webhook())

    def test_no_token_returns_none(self) -> None:
        with self._settings_stub(token=""), mock.patch.object(line_admin.httpx, "post") as p:
            self.assertIsNone(line_admin.test_webhook())
        p.assert_not_called()


class AdminLineWebhookTestRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u1", token="tok")
        self.addCleanup(app.dependency_overrides.clear)

    def test_not_configured(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings(line_messaging_access_token="")):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": None, "error": "not_configured"})

    def test_unreachable(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings()), \
             mock.patch.object(line_admin, "test_webhook", return_value=None):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": None, "error": "unreachable"})

    def test_success(self) -> None:
        result = {"success": True, "status_code": 200, "reason": "OK", "detail": "200"}
        with mock.patch.object(store, "is_admin", return_value=True), \
             mock.patch("backend.app.settings.get_settings", return_value=_settings()), \
             mock.patch.object(line_admin, "test_webhook", return_value=result):
            body = self.client.post("/api/admin/line/webhook-test").json()
        self.assertEqual(body, {"result": result, "error": None})

    def test_forbidden_for_non_admin(self) -> None:
        with mock.patch.object(store, "is_admin", return_value=False):
            resp = self.client.post("/api/admin/line/webhook-test")
        self.assertEqual(resp.status_code, 403)

    def test_requires_auth(self) -> None:
        app.dependency_overrides.clear()
        resp = self.client.post("/api/admin/line/webhook-test")
        self.assertEqual(resp.status_code, 401)
```

（`_settings` 是檔案內既有的 module-level helper;若 `LineWebhookTestTests` 需要 `types`,檔頭已 import。）

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: FAIL — `AttributeError: ... 'test_webhook'` / POST 路由 404。

- [ ] **Step 3: 實作 `_post` + `test_webhook`**

在 `backend/app/services/line_admin.py`,常數區加:

```python
LINE_WEBHOOK_TEST_URL = "https://api.line.me/v2/bot/channel/webhook/test"
```

在 `_get` 之後加 `_post`,並在 fetchers 之後加 `test_webhook`:

```python
def _post(url: str, token: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST to a LINE endpoint; return its JSON dict. Raises on non-200 or non-dict payload."""
    resp = httpx.post(url, headers={"Authorization": f"Bearer {token}"}, json=json or {}, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected LINE payload")
    return data


def test_webhook() -> dict[str, Any] | None:
    """Actively ask LINE to POST a test event to the configured webhook; report the outcome.

    Has a SIDE EFFECT (LINE delivers a test event to the webhook), so this is never called on a plain
    status read — only on an explicit admin action. NOT cached. Returns None on any transport failure;
    ``success`` reflects whether the webhook itself answered LINE with 200.
    """
    token = _token()
    if not token:
        return None
    try:
        data = _post(LINE_WEBHOOK_TEST_URL, token)
    except Exception:  # noqa: BLE001
        logger.warning("LINE webhook test: request failed")
        return None
    return {
        "success": bool(data.get("success")),
        "status_code": _safe_int(data.get("statusCode")),
        "reason": data.get("reason") if isinstance(data.get("reason"), str) else None,
        "detail": data.get("detail") if isinstance(data.get("detail"), str) else None,
    }
```

- [ ] **Step 4: 新增端點**

在 `backend/app/routers/admin.py`,於 `admin_line_status` 之後加:

```python
@router.post("/line/webhook-test")
def admin_line_webhook_test(user: CurrentUser = Depends(get_admin_user)) -> dict:
    """Ask LINE to POST a test event to the configured webhook and report the result (admin-only).

    This has a side effect (LINE delivers a test event), so it is a distinct POST triggered only by an
    explicit admin action — never by the status read. Returns no secret.
    """
    s = settings.get_settings()
    if not s.line_messaging_configured:
        return {"result": None, "error": "not_configured"}
    result = line_admin.test_webhook()
    return {"result": result, "error": "unreachable" if result is None else None}
```

- [ ] **Step 5: 跑測試 + 覆蓋率**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend_admin_line.py -q`
Expected: PASS。
Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/line_admin.py backend/app/routers/admin.py tests/test_backend_admin_line.py
git commit -m "feat(admin): POST /api/admin/line/webhook-test"
```

---

## Task 4: 前端 — bot info / webhook(含測試按鈕)/ delivery 卡片

**Files:**
- Modify: `frontend/src/api.ts`(型別 + 擴充 `LineStatus` + `testLineWebhook`)
- Modify: `frontend/src/lib/i18n.tsx`(新 key,en + zh-Hant)
- Modify: `frontend/src/pages/admin/AdminOverview.tsx`(`LineSection` 三張卡 + 按鈕)
- Test: `frontend/src/test/pages.admin.test.tsx`

**Interfaces:**
- Consumes: `GET /api/admin/line/status`(擴充,Task 2)、`POST /api/admin/line/webhook-test`(Task 3)。
- Produces: `api.testLineWebhook()`、型別 `LineBotInfo`/`LineWebhook`/`LineDelivery`/`LineWebhookTestResult`/`LineWebhookTestResponse`,`LineStatus` 加三個非可選欄位 `bot_info`/`webhook`/`delivery`。

- [ ] **Step 1: 寫失敗測試**

在 `frontend/src/test/pages.admin.test.tsx`:

(a) 引入型別:`import { api, type AdminOverview, type AdminSettingsResponse, type AdminUserRow, type LineStatus } from "../api";`(已有 `LineStatus`)。

(b) 把既有的 `SAMPLE_LINE_STATUS` 擴充為含新欄位:

```tsx
const SAMPLE_LINE_STATUS: LineStatus = {
  messaging_configured: true,
  login_configured: true,
  channel_id: "2010629653",
  quota: { type: "limited", used: 12, value: 200, remaining: 188 },
  quota_error: null,
  bot_info: { display_name: "x-coach", basic_id: "@xcoach", premium_id: null, chat_mode: "bot", mark_as_read_mode: "auto" },
  webhook: { endpoint: "https://x-coach.app/api/line/webhook", active: true },
  delivery: { date: "20260720", reply: 4, push: 3 },
};
```

(c) 既有那個「not configured」的完整字面物件測試,補上三個新欄位:在 `quota: null, quota_error: null,` 之後加 `bot_info: null, webhook: null, delivery: null,`。

(d) 新增測試(放在 `describe("AdminOverview", ...)` 內):

```tsx
  it("renders the bot info, webhook, and delivery cards", async () => {
    renderAdmin("/admin");
    expect(await screen.findByText("x-coach")).toBeInTheDocument();
    expect(screen.getByText("@xcoach")).toBeInTheDocument();
    expect(screen.getByText("Webhook")).toBeInTheDocument();
    expect(screen.getByText("Replies yesterday")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument(); // reply count
    expect(screen.getByText("3")).toBeInTheDocument(); // push count
  });

  it("warns when the bot is in chat mode (webhook won't receive events)", async () => {
    vi.spyOn(api, "getLineStatus").mockResolvedValue({
      ...SAMPLE_LINE_STATUS,
      bot_info: { display_name: "x-coach", basic_id: "@xcoach", premium_id: null, chat_mode: "chat", mark_as_read_mode: "auto" },
    });
    renderAdmin("/admin");
    expect(await screen.findByText(/won't receive message events/i)).toBeInTheDocument();
  });

  it("shows 'not ready yet' when delivery counts are unavailable", async () => {
    vi.spyOn(api, "getLineStatus").mockResolvedValue({
      ...SAMPLE_LINE_STATUS,
      delivery: { date: "20260720", reply: null, push: null },
    });
    renderAdmin("/admin");
    expect((await screen.findAllByText("Not ready yet")).length).toBeGreaterThan(0);
  });

  it("runs the webhook test and shows a reachable result", async () => {
    const testFn = vi.spyOn(api, "testLineWebhook").mockResolvedValue({
      result: { success: true, status_code: 200, reason: "OK", detail: "200" },
      error: null,
    });
    renderAdmin("/admin");
    fireEvent.click(await screen.findByRole("button", { name: "Test webhook" }));
    await waitFor(() => expect(testFn).toHaveBeenCalled());
    expect(await screen.findByText(/Reachable \(200\)/i)).toBeInTheDocument();
  });

  it("shows an error when the webhook test can't reach LINE", async () => {
    vi.spyOn(api, "testLineWebhook").mockResolvedValue({ result: null, error: "unreachable" });
    renderAdmin("/admin");
    fireEvent.click(await screen.findByRole("button", { name: "Test webhook" }));
    expect(await screen.findByText("Couldn't reach LINE.")).toBeInTheDocument();
  });
```

(e) 在 `beforeEach` 既有 `vi.spyOn(api, "getLineStatus")...` 之後加預設 stub:

```tsx
  vi.spyOn(api, "testLineWebhook").mockResolvedValue({
    result: { success: true, status_code: 200, reason: "OK", detail: "200" }, error: null,
  });
```

- [ ] **Step 2: 跑測試確認失敗**

Run(cwd=`frontend/`): `yarn test src/test/pages.admin.test.tsx`
Expected: FAIL — `api.testLineWebhook is not a function` / 型別缺欄位 / 找不到卡片文字。

- [ ] **Step 3: api.ts 型別 + 方法**

在 `frontend/src/api.ts`,把 `LineStatus` interface 擴充並在其後新增型別:

```ts
export interface LineBotInfo {
  display_name: string;
  basic_id: string;
  premium_id: string | null;
  chat_mode: string; // "bot" | "chat" (string; chat mode means the webhook gets no message events)
  mark_as_read_mode: string;
}
export interface LineWebhook {
  endpoint: string;
  active: boolean;
}
export interface LineDelivery {
  date: string; // yyyymmdd the counts are for (yesterday, OA timezone)
  reply: number | null; // null when LINE's data for that day isn't ready
  push: number | null;
}
export interface LineWebhookTestResult {
  success: boolean;
  status_code: number | null;
  reason: string | null;
  detail: string | null;
}
export interface LineWebhookTestResponse {
  result: LineWebhookTestResult | null;
  error: "not_configured" | "unreachable" | null;
}
```

並在既有 `LineStatus` interface 內、`quota_error` 之後加三行:

```ts
  bot_info: LineBotInfo | null;
  webhook: LineWebhook | null;
  delivery: LineDelivery | null;
```

在 `api` 物件內、`getLineStatus` 之後加:

```ts
  // Ask LINE to POST a test event to the webhook and report the outcome (admin-only). Side-effecting,
  // so it is a POST triggered only by the admin's explicit click — never on a status read.
  async testLineWebhook(): Promise<LineWebhookTestResponse> {
    const res = await fetch("/api/admin/line/webhook-test", {
      method: "POST",
      headers: { ...(await authHeader()) },
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for /api/admin/line/webhook-test`);
    return (await res.json()) as LineWebhookTestResponse;
  },
```

- [ ] **Step 4: i18n key(兩份字典)**

在 `frontend/src/lib/i18n.tsx` 的 en 字典 `admin.line.*` 區塊(`admin.line.noCapNote` 之後)插入:

```ts
  "admin.line.oaName": "Official account",
  "admin.line.chatModeWarn": "Chat mode — the webhook won't receive message events",
  "admin.line.webhook": "Webhook",
  "admin.line.webhookActive": "Active",
  "admin.line.webhookInactive": "Inactive",
  "admin.line.webhookTest": "Test webhook",
  "admin.line.webhookTesting": "Testing…",
  "admin.line.webhookReachable": "Reachable ({code})",
  "admin.line.webhookFailed": "Failed ({reason})",
  "admin.line.webhookTestError": "Couldn't reach LINE.",
  "admin.line.replyYesterday": "Replies yesterday",
  "admin.line.pushYesterday": "Pushes yesterday",
  "admin.line.deliveryUnready": "Not ready yet",
```

在 zh-Hant 字典對應位置插入(key 名一致):

```ts
  "admin.line.oaName": "官方帳號",
  "admin.line.chatModeWarn": "聊天模式 — webhook 不會收到訊息事件",
  "admin.line.webhook": "Webhook",
  "admin.line.webhookActive": "運作中",
  "admin.line.webhookInactive": "未啟用",
  "admin.line.webhookTest": "測試 webhook",
  "admin.line.webhookTesting": "測試中…",
  "admin.line.webhookReachable": "連得到（{code}）",
  "admin.line.webhookFailed": "失敗（{reason}）",
  "admin.line.webhookTestError": "無法連到 LINE。",
  "admin.line.replyYesterday": "昨日回覆",
  "admin.line.pushYesterday": "昨日推播",
  "admin.line.deliveryUnready": "資料尚未就緒",
```

> 註:英文測試斷言用 en 值(`"Test webhook"`、`"Replies yesterday"`、`"Webhook"`、`/won't receive message events/`、`/Reachable \(200\)/`、`"Couldn't reach LINE."`、`"Not ready yet"`);渲染時 `I18nProvider` 預設 en。

- [ ] **Step 5: `AdminOverview.tsx` 擴充 `LineSection`**

在 `frontend/src/pages/admin/AdminOverview.tsx`:

(a) import 補型別與 icon(檔案已 import 一組 phosphor icon 與 `useState`/`useEffect`):

```tsx
import { api, type AdminOverview as AdminOverviewData, type LineStatus, type LineWebhookTestResponse } from "../../api";
```

在既有 phosphor import 清單加 `Robot`、`Plugs`、`PaperPlaneTilt`(若某名稱在安裝版本不存在,改用 `ChatCircleText`/`Plug`/`PaperPlane` 等等值 icon,並在報告註明)。

(b) 在 `LineSection` 既有 quota 區塊之後(`</div>` 收掉 grid 之前或另起 grid),加入三張卡與按鈕。將 `LineSection` return 內、既有 quota grid 之後補一段:

```tsx
      {data.bot_info ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Robot size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.oaName")}</span>
          </div>
          <p className="mt-2 text-base font-semibold text-content">{data.bot_info.display_name}</p>
          <p className="text-xs text-muted">{data.bot_info.basic_id}</p>
          {data.bot_info.chat_mode !== "bot" ? (
            <p className="mt-2 text-xs font-medium text-danger">{t("admin.line.chatModeWarn")}</p>
          ) : null}
        </div>
      ) : null}

      {data.webhook ? (
        <div className="mt-3 rounded-2xl border border-border-dark bg-surface-dark p-4">
          <div className="flex items-center gap-1.5 text-faint">
            <Plugs size={16} weight="duotone" />
            <span className="text-xs font-medium">{t("admin.line.webhook")}</span>
          </div>
          <p className="mt-2 truncate text-sm text-content" title={data.webhook.endpoint}>
            {data.webhook.endpoint}
          </p>
          <p className={`text-xs font-medium ${data.webhook.active ? "text-secondary" : "text-danger"}`}>
            {data.webhook.active ? t("admin.line.webhookActive") : t("admin.line.webhookInactive")}
          </p>
          <button
            type="button"
            onClick={runWebhookTest}
            disabled={testing}
            className="mt-3 rounded-lg border border-border-dark px-3 py-1.5 text-xs font-medium text-content disabled:opacity-50"
          >
            {testing ? t("admin.line.webhookTesting") : t("admin.line.webhookTest")}
          </button>
          {testMsg ? <p className="mt-2 text-xs text-muted">{testMsg}</p> : null}
        </div>
      ) : null}

      {data.delivery ? (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <OverviewCard
            icon={<PaperPlaneTilt size={16} weight="duotone" />}
            label={t("admin.line.replyYesterday")}
            value={data.delivery.reply === null ? t("admin.line.deliveryUnready") : String(data.delivery.reply)}
          />
          <OverviewCard
            icon={<PaperPlaneTilt size={16} weight="duotone" />}
            label={t("admin.line.pushYesterday")}
            value={data.delivery.push === null ? t("admin.line.deliveryUnready") : String(data.delivery.push)}
          />
        </div>
      ) : null}
```

(c) 在 `LineSection` 元件內(`useEffect` 之後、return 之前)加 webhook 測試的 state 與 handler:

```tsx
  const [testing, setTesting] = useState(false);
  const [testMsg, setTestMsg] = useState<string | null>(null);

  async function runWebhookTest() {
    setTesting(true);
    setTestMsg(null);
    let res: LineWebhookTestResponse;
    try {
      res = await api.testLineWebhook();
    } catch {
      setTesting(false);
      setTestMsg(t("admin.line.webhookTestError"));
      return;
    }
    setTesting(false);
    if (!res.result) {
      setTestMsg(t("admin.line.webhookTestError"));
    } else if (res.result.success) {
      setTestMsg(t("admin.line.webhookReachable", { code: res.result.status_code ?? 0 }));
    } else {
      setTestMsg(t("admin.line.webhookFailed", { reason: res.result.reason ?? "" }));
    }
  }
```

- [ ] **Step 6: 跑測試 + build + 覆蓋率**

Run(cwd=`frontend/`): `yarn test src/test/pages.admin.test.tsx`
Expected: PASS(既有 + 5 新測試;注意 `lib.liff`/`lib.supabase` 的既有 .env flake 與本檔無關)。
Run(cwd=`frontend/`): `yarn build`
Expected: 乾淨(TS 無 unused/型別錯)。
Run(cwd=`frontend/`): `yarn test:coverage`
Expected: PASS(門檻之上)。

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.ts frontend/src/lib/i18n.tsx frontend/src/pages/admin/AdminOverview.tsx frontend/src/test/pages.admin.test.tsx
git commit -m "feat(admin): LINE bot info / webhook test / delivery cards"
```

---

## Self-Review

**Spec coverage:**
- §3.1 改名 line_admin + 引用更新 → Task 1 ✅
- §3.2 泛化快取 `_cached` → Task 1 ✅
- §3.3 fetch_bot_info/webhook/delivery + `_yesterday_yyyymmdd`/`_token` → Task 2 ✅;test_webhook + `_post` → Task 3 ✅
- §3.4 `/status` 三新 key → Task 2 ✅;`POST /webhook-test` 三態 → Task 3 ✅
- §3.5 前端型別/testLineWebhook/三張卡/按鈕/i18n → Task 4 ✅
- §4 錯誤處理總表 → Task 2/3 後端測試 + Task 4 前端測試(unready/unreachable/chat mode/not-configured)✅
- §5 測試 + 覆蓋率關卡 → 各 Task 步驟 ✅
- §6 YAGNI(不加 insight/圖表/寫入/LIFF)→ 未納入任何 Task ✅

**Placeholder scan:** 無 TBD/TODO;所有 code step 皆含完整程式碼與確切指令;icon 名稱附有「不存在則換等值 icon」的具體 fallback。

**Type consistency:** 後端組裝 dict 的 snake_case key(`display_name`/`basic_id`/`chat_mode`/`status_code`/`reply`/`push`)與前端 interface 完全對齊;`test_webhook`/`fetch_bot_info`/`fetch_webhook`/`fetch_delivery`/`_cached`/`_token`/`_yesterday_yyyymmdd` 在 Task 2/3 定義、Task 3/4 一致使用;`LineWebhookTestResponse.error` 值 `"not_configured"|"unreachable"|null` 前後端一致;`testLineWebhook` 回傳型別一致。

**發現並修正:** Task 4 明確處理了「擴充 `LineStatus` 後,既有測試的 not-configured 完整字面物件會因缺三個必填欄位而 TS 編譯失敗」——Step 1(c) 要求補 `bot_info:null, webhook:null, delivery:null`,避免 build 破。
