# Admin LINE 診斷面板擴充(bot info / webhook 健康 / 每日發送數)— 設計 spec

**日期**:2026-07-21
**分支**:`feat/admin-line-diagnostics`(開自 `main`,PR #45 已 merge)
**狀態**:設計已核可,待實作

## 1. 目標

在既有的 LINE admin 面板(PR #45:連線狀態 + push 額度)上,再加三塊唯讀診斷資訊,全部代理自 LINE Messaging API、只用 access token server-side、回應不含密鑰:

1. **Bot info**(`GET /v2/bot/info`)— OA 顯示名稱、`@basicId`、chatMode,確認接的是正確帳號,並揪出 `chatMode:"chat"` 這個「webhook 收不到訊息」的常見雷。
2. **Webhook 健康**(`GET /v2/bot/channel/webhook/endpoint` + 按鈕觸發的 `POST /v2/bot/channel/webhook/test`)— 顯示 webhook URL 與 `active`,並讓 admin 手動實測 LINE 打得通不通。
3. **每日發送數**(`GET /v2/bot/message/delivery/{reply,push}`)— 顯示「昨日回覆 N 則 / 昨日推播 N 則」,補上 quota API 算不到的 **reply** 數。

## 2. 關鍵前提(設計依據)

- **這些與現有 `line_quota` 同模式**:Messaging channel access token、server-side、唯讀、任何失敗降級為 `null`(不 raise)、60s TTL 快取。
- **`delivery` 是每日彙總、隔天才完整**:LINE 對「今天」回 `status:"unready"`,所以面板抓**昨天**(UTC+8),並回傳該日期讓 admin 知道是哪天。非即時計數。
- **`webhook/test` 是有副作用的 POST**(會主動送一個測試事件到 webhook)→ **不在載入時呼叫**,做成獨立端點 + 按鈕,admin 點了才打。
- **`bot/info` 的 chatMode**:`"chat"`(手動聊天)時 webhook 不收訊息事件 → 這張卡幫使用者看出來。

## 3. 架構

### 3.1 模組改名 `line_quota.py` → `line_admin.py`

模組責任已從「讀 quota」擴大為「為 admin 面板讀 LINE Messaging API」,沿用既有名稱會名不符實。趁 PR #45 已 merge、尚無外部依賴,改名並泛化。

**改名要更新的所有引用點:**
- `backend/app/routers/admin.py`:`from backend.app.services import line_quota, ...` → `line_admin`,以及 `line_quota.fetch_quota()` 呼叫。
- `tests/test_backend_admin_line.py`:`from backend.app.services import line_quota` → `line_admin`,及所有 `line_quota.` 前綴、`mock.patch.object(line_quota, ...)`。
- 文件字串內對 "quota" 的敘述改為涵蓋新責任。
- `scripts/run_backend_coverage.py` 的 `_DEFAULT_TESTS` 已列 `tests/test_backend_admin_line.py`(檔名不變)→ 不需改。

### 3.2 泛化快取(支援多個 fetch)

現況是單一 quota 專用快取。改為 key 化的 60s TTL 快取,讓四個唯讀讀取各自快取、`clear_cache()` 一次清空:

```python
_TTL_SECONDS = 60.0
_cache: dict[str, dict[str, Any] | None] = {}
_cache_at: dict[str, float] = {}

def _cached(key: str, producer) -> dict[str, Any] | None:
    """Serve `key` from a 60s TTL cache; `key in _cache_at` distinguishes cached-None from never-fetched."""
    now = time.monotonic()
    if key in _cache_at and (now - _cache_at[key]) < _TTL_SECONDS:
        return _cache[key]
    result = producer()
    _cache[key] = result
    _cache_at[key] = now
    return result

def clear_cache() -> None:
    _cache.clear()
    _cache_at.clear()
```

(取代現有的 `_cache`/`_cache_at`/`_cache_valid` 三個全域;dict membership 取代 `_cache_valid`。)

### 3.3 `line_admin.py` 新增函式

共用 helper:保留 `_safe_int`、`_get(url, token)`;新增 `_post(url, token, json=None)`(比照 `_get`,`raise_for_status` + 回 dict)。所有 `_fetch_*` 先讀 token,無 token 回 `None`;呼叫包 try/except → `None`(logged)。

- `_fetch_quota()`:現有 `_fetch` 改名。`fetch_quota()` = `_cached("quota", _fetch_quota)`(行為不變)。
- `_fetch_bot_info()` → `GET /v2/bot/info` → 取 `{"display_name","basic_id","premium_id","chat_mode","mark_as_read_mode"}`(缺欄位給 `None`/`""`)。`fetch_bot_info()` = `_cached("bot_info", _fetch_bot_info)`。
- `_fetch_webhook()` → `GET /v2/bot/channel/webhook/endpoint` → `{"endpoint": str, "active": bool}`(`active` 強制 `bool()`)。`fetch_webhook()` = `_cached("webhook", ...)`。
- `_fetch_delivery()` → 呼叫兩支 `GET /v2/bot/message/delivery/reply?date=` 與 `/push?date=`,date = `_yesterday_yyyymmdd()`。回 `{"date": "yyyymmdd", "reply": int|None, "push": int|None}`;LINE 回 `status!="ready"` 或缺 `success` 時該值為 `None`(用 `_safe_int(resp.get("success"))`)。`fetch_delivery()` = `_cached("delivery", ...)`。
- `_yesterday_yyyymmdd()` → `(datetime.now(_DISPLAY_TZ) - timedelta(days=1)).strftime("%Y%m%d")`,`_DISPLAY_TZ = timezone(timedelta(hours=8))`。獨立小函式,測試可 patch 它取得定值。
- `test_webhook() -> dict | None`(**不快取**)→ `POST /v2/bot/channel/webhook/test` → `{"success": bool, "status_code": int|None, "reason": str|None, "detail": str|None}`;transport 失敗回 `None`。

新增常數:`LINE_BOT_INFO_URL`、`LINE_WEBHOOK_ENDPOINT_URL`、`LINE_WEBHOOK_TEST_URL`、`LINE_DELIVERY_REPLY_URL`、`LINE_DELIVERY_PUSH_URL`。

### 3.4 端點

**擴充 `GET /api/admin/line/status`**(admin-only,既有):當 `messaging_configured` 時,除 `quota` 外一併呼叫 `fetch_bot_info()`/`fetch_webhook()`/`fetch_delivery()`,新增三個 key(各自 `null` on 失敗/未設定):

```jsonc
{
  "messaging_configured": true,
  "login_configured": true,
  "channel_id": "2010629653",
  "quota": { ... } ,            // 不變
  "quota_error": null,          // 不變
  "bot_info": { "display_name": "x-coach", "basic_id": "@xxx", "premium_id": null,
                "chat_mode": "bot", "mark_as_read_mode": "auto" },
  "webhook":  { "endpoint": "https://.../api/line/webhook", "active": true },
  "delivery": { "date": "20260720", "reply": 12, "push": 0 }
}
```

**新增 `POST /api/admin/line/webhook-test`**(admin-only):

```jsonc
// messaging 未設定:
{ "result": null, "error": "not_configured" }
// LINE 呼叫失敗:
{ "result": null, "error": "unreachable" }
// 成功(不論 webhook 本身通不通,success 反映實測結果):
{ "result": { "success": true, "status_code": 200, "reason": "OK", "detail": "200" }, "error": null }
```

兩端點皆**永不回傳密鑰**。

### 3.5 前端

`frontend/src/api.ts`:
- `LineStatus` 加 `bot_info: LineBotInfo | null`、`webhook: LineWebhook | null`、`delivery: LineDelivery | null`;新增這三個 interface 與 `LineWebhookTestResponse`/`LineWebhookTestResult`。
- 新增 `api.testLineWebhook()` → `POST /api/admin/line/webhook-test`(比照 `updateAdminSettings` 帶 auth header 的 POST 寫法)。

`frontend/src/pages/admin/AdminOverview.tsx`(`LineSection`,沿用 `OverviewCard`):
- **Bot info 卡**:`bot_info` 存在時顯示 `display_name`(主值)+ 副行 `@basic_id`;若 `chat_mode !== "bot"` 顯示警示樣式 + 提示「聊天模式,webhook 不收訊息」。
- **Webhook 卡**:`endpoint`(截斷)+ `active` 綠/紅;卡內「測試 webhook」按鈕 → `testLineWebhook()`,本地 state `idle|testing|done`,結果 inline 顯示 `success` + `status_code`/`reason`(失敗顯示 `error`)。
- **每日發送卡**:`昨日回覆 {reply} 則`、`昨日推播 {push} 則`;`reply`/`push` 為 `null` 時顯示「資料尚未就緒」;卡註明日期。
- 這些卡只在 `bot_info`/`webhook`/`delivery` 各自非 `null` 時渲染;整個 LINE 區塊維持既有「fetch 失敗渲染 null、不拖垮主 overview」行為。
- en + zh-Hant i18n 補 `admin.line.*` key(名稱兩份一致)。

## 4. 錯誤處理總表

| 情況 | 後端 | 前端 |
|---|---|---|
| messaging 未設定 | 三個 key 皆 `null`;webhook-test → `error:"not_configured"` | 只顯示既有連線狀態卡 |
| 某支 LINE 呼叫失敗 | 該 key `null`(其他不受影響) | 該卡不渲染 |
| `delivery` LINE 回 unready | `reply`/`push` 為 `null` | 「資料尚未就緒」 |
| webhook-test transport 失敗 | `result:null, error:"unreachable"` | 按鈕結果區顯示「無法連到 LINE」 |
| webhook 實測不通 | `result:{success:false, status_code:...}` | 顯示 success=false + 狀態碼 |
| 非 admin | 兩端點皆 403 | AdminLayout 既有 gating |

## 5. 測試

**後端**(擴充 `tests/test_backend_admin_line.py`;patch `line_admin` 各 fetch 的 `httpx` 接縫;`_yesterday_yyyymmdd` patch 成定值):
- `fetch_bot_info`/`fetch_webhook`/`fetch_delivery` 各:happy(組裝正確)、缺 token → `None`、非 200 → `None`、malformed → `None`。
- `fetch_delivery`:`status:"ready"` 帶 success → int;`status:"unready"`/缺 success → `None`;date 用 patch 定值驗證組進 query。
- `test_webhook`:success=true/false 組裝;transport 失敗 → `None`。
- key 化快取:同 key 第二次命中不重打;`clear_cache` 後重打。
- `GET /line/status` 擴充:configured 時含三新 key 且不含密鑰;未設定時三 key 皆 null 且不呼叫 LINE。
- `POST /line/webhook-test`:not_configured / unreachable / 成功三態;非 admin → 403;未登入 → 401。
- 既有 quota 測試改 `line_admin` 引用後仍全過。
- 覆蓋率關卡 `--fail-under 95`。

**前端**(擴充 `pages.admin.test.tsx`;cwd=`frontend/`):
- bot_info 卡渲染 display_name/@basic_id;`chat_mode:"chat"` 顯示警示。
- webhook 卡 active 綠/紅;點「測試 webhook」→ mock `testLineWebhook` resolve success → 顯示結果;reject/error → 顯示失敗。
- delivery 卡顯示 reply/push;`null` → 「資料尚未就緒」。
- 各新 key 為 `null` 時該卡不渲染;LINE fetch 失敗不影響主 overview(既有行為不回歸)。
- `yarn build` clean、`yarn test:coverage` 過。

## 6. 明確不做(YAGNI)

- 不加 followers/demographic insight(帳號規模門檻、資料延遲,等長大再說)。
- 不做歷史趨勢/圖表,只顯示當前快照與昨日單日數。
- 不在面板做任何**寫入/發送**(push/multicast/broadcast);webhook-test 是唯一的主動呼叫,且唯讀語意(只驗證連線)。
- 不列 LIFF app(認證方式不同,不成比例)。
- 不代設 LINE OA Manager 的每月上限(承 PR #45)。

## 7. 影響檔案

- 改名:`backend/app/services/line_quota.py` → `line_admin.py`(泛化 + 新 fetch)
- 修改:`backend/app/routers/admin.py`(import 改名;`/line/status` 擴充;新增 `POST /line/webhook-test`)
- 修改:`frontend/src/api.ts`(型別 + `testLineWebhook`)
- 修改:`frontend/src/pages/admin/AdminOverview.tsx`(三張新卡 + 測試按鈕)
- 修改:`frontend/src/lib/i18n.tsx`(新 key,en + zh-Hant)
- 修改:`tests/test_backend_admin_line.py`(改名引用 + 新測試)
- 修改:`frontend/src/test/pages.admin.test.tsx`(新測試)
