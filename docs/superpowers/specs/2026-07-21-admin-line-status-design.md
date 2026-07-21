# Admin LINE 連線狀態 + 推播額度面板 — 設計 spec

**日期**：2026-07-21
**分支**：`feat/admin-line-status`（開自 `main`）
**狀態**：設計已核可,待實作

## 1. 目標

在 admin 後台新增一個**唯讀**面板,讓管理者一眼看出:

1. **LINE 連線狀態** — LINE 登入橋接(LIFF→Supabase)與 LINE Messaging Bot 是否已正確設定。
2. **本月推播已用次數** — LINE 計費的 push 訊息本月消耗量。
3. **免費剩餘次數** — 每月上限扣掉已用(僅當帳號有設定上限時)。

面板**永不讀寫任何密鑰**:access token 只在後端 server-side 用來呼叫 LINE,絕不回傳給瀏覽器;唯一顯示的識別碼是非密鑰的 `channel_id`。

## 2. 關鍵前提(設計依據)

- **目前 bot 只用 Reply API,回覆訊息免費且不計入額度**(見 `services/line_bot.py` `LINE_REPLY_URL` 註解)。因此 push 消耗量現在會是 0;此面板是為了未來加推播、以及即時監控計費而預先到位。
- **「發送次數/剩餘次數」的唯一權威來源是 LINE 官方 quota API**,不是我們自己的 DB。我們代理呼叫並顯示 LINE 回的數字(這也是 LINE 實際計費的數字)。
- **LINE `/message/quota` 的 `value` 是帳號擁有者在 LINE Official Account Manager 後台自訂的「每月訊息上限 cap」,不等於免費方案的固定額度本身。** 若未設定上限,LINE 回 `{"type": "none"}`,此時**沒有「剩餘」可算**,面板只顯示「本月已用」並附提示。要讓「剩餘免費次數」有意義,帳號擁有者須在 LINE OA Manager 把每月上限設成其免費額度(如 200)。此為 LINE 端設定,程式無法代設,只在 UI 提示。

## 3. 架構(方案 A:獨立端點 + 獨立 service 模組)

理由:LINE 額度需對外發即時 HTTP 呼叫(有延遲、會失敗),不可塞進現有 `/api/admin/overview`(該端點刻意保持純本地、secret-free、保證不失敗)。獨立後,LINE 呼叫慢/失敗時不會拖垮或弄壞 Overview 頁。

### 3.1 新 service 模組 `backend/app/services/line_quota.py`

代理 LINE quota API 的**單一可測試接縫**(比照 `line_bot`/`line_auth` 隔離 httpx 的既有寫法)。`httpx` top-level import。

LINE 端點(皆需 `Authorization: Bearer {line_messaging_access_token}`):

- `GET https://api.line.me/v2/bot/message/quota`
  → `{"type": "none"}` 或 `{"type": "limited", "value": <int>}`
- `GET https://api.line.me/v2/bot/message/quota/consumption`
  → `{"totalUsage": <int>}`

函式:

```python
LINE_QUOTA_URL = "https://api.line.me/v2/bot/message/quota"
LINE_CONSUMPTION_URL = "https://api.line.me/v2/bot/message/quota/consumption"
_QUOTA_TIMEOUT_S = 10.0
_TTL_SECONDS = 60.0  # 短 TTL 快取,避免 admin 重整時狂打 LINE 被 rate-limit

def fetch_quota() -> dict | None:
    """回傳 {"type","value"?,"used","remaining"?} 或 None(未設定/呼叫失敗)。

    - messaging 未設定 → 直接回 None(呼叫端另外回報 configured 旗標)。
    - 任一 LINE 呼叫失敗 / 非 200 / 形狀不符 → 回 None,絕不 raise。
    - type=="limited" 時才算 remaining = max(0, value - used);type=="none" 時 remaining 省略。
    - 走 60s 程序內 TTL 快取(仿 runtime_config 的 _cache/_cache_at/clear_cache 模式)。
    """
```

防禦性原則(比照 `line_bot`):任何例外都吞掉回 `None`,面板降級為「無法取得額度」而非爆錯。數值以 best-effort `int()` 解析,非預期型別視為失敗。

### 3.2 新端點 `GET /api/admin/line/status`(加在 `routers/admin.py`)

- 由 `get_admin_user` 把關(非 admin → 403)。
- 組裝回應:
  - `messaging_configured` / `login_configured` ← `settings.get_settings()` 的 `line_messaging_configured` / `line_login_configured` property。
  - `channel_id` ← `settings.line_channel_id`(非密鑰)。
  - `quota` ← `line_quota.fetch_quota()`(未設定/失敗為 `null`)。
  - `quota_error` ← 當 `messaging_configured` 為 true 但 `quota` 為 `null` 時給 `"unreachable"`,否則 `null`。
- **回應永不含 access token / channel secret / service_role key。**

回應形狀:

```jsonc
{
  "messaging_configured": true,
  "login_configured": true,
  "channel_id": "2010629653",
  "quota": {
    "type": "limited",     // 或 "none"
    "value": 200,          // 僅 type=limited
    "used": 12,
    "remaining": 188       // 僅 type=limited
  },
  "quota_error": null       // 或 "unreachable"
}
```

`type=="none"` 時 `quota` 例:`{"type":"none","used":12}`(無 `value`/`remaining`)。

### 3.3 前端

- `frontend/src/api.ts`:新增 `getLineStatus()` 與對應型別 `LineStatus`。
- `AdminOverview.tsx`:在現有 overview 卡片格下方,新增一個 **LINE 區塊**,以**獨立 `useEffect` fetch**(與主 overview 分開,互不拖累;各自的 loading/error 狀態)。
- 卡片(沿用現有 `OverviewCard`):
  - `LINE 登入橋接` — `login_configured` → 已設定/未設定(ok tone)。
  - `LINE Bot` — `messaging_configured` → 已設定/未設定(ok tone)。
  - `本月推播已用` — `quota.type=="limited"` 顯示 `used / value`(如 `12 / 200`);`type=="none"` 顯示 `used`(如 `12`)。
  - `剩餘免費額度` — 僅 `type=="limited"` 顯示 `remaining`;`type=="none"` 顯示 `—` 並在區塊附提示「未在 LINE 後台設定每月上限,無法計算剩餘」。
- `quota_error === "unreachable"`:LINE 卡片區改顯示淡色提示「無法取得 LINE 額度」,不顯示數字。
- `messaging_configured === false`:額度卡片不顯示(或顯示未設定),只留連線狀態卡片。
- 所有可見字串走既有 i18n(`t(...)`),新增對應 key。

## 4. 錯誤處理總表

| 情況 | 後端行為 | 前端顯示 |
|---|---|---|
| messaging 未設定 | `quota:null, quota_error:null, messaging_configured:false` | 只顯示連線狀態卡(未設定) |
| LINE 呼叫失敗/timeout/非200 | `quota:null, quota_error:"unreachable"` | 「無法取得 LINE 額度」淡色提示 |
| `type:"none"`(未設上限) | `quota:{type,used}`(無 remaining) | 已用顯示數字,剩餘顯示 `—` + 提示 |
| `type:"limited"` | `quota:{type,value,used,remaining}` | 已用 `used/value`、剩餘 `remaining` |
| 非 admin 呼叫端點 | 403 | AdminLayout 既有 gating |

## 5. 測試

**後端**(`tests/test_backend_admin_line.py`,patch `line_quota` 的 LINE 呼叫接縫;比照現有 admin/line 測試風格,`.venv\Scripts\python.exe -m pytest`):

- messaging+limited → `remaining == value - used` 正確算出。
- `type=="none"` → `quota` 無 `remaining`。
- LINE 呼叫拋錯/非200 → `quota:null` + `quota_error:"unreachable"`,端點不 500。
- messaging 未設定 → 不呼叫 LINE、`messaging_configured:false`、`quota:null`、`quota_error:null`。
- 回應**不含**任何密鑰欄位(斷言 access token/secret 不出現)。
- 非 admin → 403;未登入 → 401/相應。
- TTL 快取:第二次呼叫命中快取不重打 LINE(patch 呼叫計數驗證)。

**前端**(`frontend/src/test/pages.admin.test.tsx` 擴充,cwd=`frontend/`,`yarn test`):

- ready + limited → 顯示 `used/value` 與 remaining。
- `type:"none"` → 剩餘顯示 `—` + 提示。
- `quota_error:"unreachable"` → 顯示「無法取得 LINE 額度」。
- messaging 未設定 → 不顯示額度卡。
- LINE fetch 失敗 → 該區降級,不影響主 overview 卡片渲染。

**CI 對齊**:後端 coverage gate 95%(`scripts/run_backend_coverage.py --fail-under 95`);前端 `yarn test:coverage`。

## 6. 明確不做(YAGNI)

- 不自建 reply 計數(使用者已選:只追蹤官方 push 額度)。
- 不做歷史趨勢/圖表,只顯示當前快照。
- 不在後台可**編輯** LINE 設定(密鑰維持純 `.env`;此面板純唯讀,見前次結論)。
- 不代設 LINE OA Manager 的每月上限(LINE 端設定,無 API 可代寫;只在 UI 提示)。

## 7. 影響檔案

- 新增:`backend/app/services/line_quota.py`
- 修改:`backend/app/routers/admin.py`(新端點)
- 修改:`frontend/src/api.ts`(`getLineStatus` + 型別)
- 修改:`frontend/src/pages/admin/AdminOverview.tsx`(LINE 區塊)
- 修改:`frontend/src/lib/i18n.tsx`(新 key)
- 新增:`tests/test_backend_admin_line.py`
- 修改:`frontend/src/test/pages.admin.test.tsx`
