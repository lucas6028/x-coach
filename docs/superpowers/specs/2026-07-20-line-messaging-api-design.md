# LINE Messaging API：在聊天室查詢個人訓練摘要

Status: **設計已確認，待實作** · Created 2026-07-20 ·
前置：LINE Login + LIFF 已實作（`backend/app/services/line_auth.py`、`frontend/src/lib/liff.ts`）

在既有的 LINE Login / LIFF 之上串接 **Messaging API**，讓使用者在 LINE 官方帳號的聊天室裡，
用圖文選單或關鍵字取得自己的 x-coach 訓練摘要。

---

## 1. 目標與範圍

**要做的**：使用者在 LINE 聊天室點圖文選單按鈕（或傳關鍵字）→ bot 回一則文字訊息，內容是
該使用者的訓練摘要（總分析次數、最近一次分析、最常出現的 fault）。

**明確不做（YAGNI）**：

- push message（分析完成主動推播）— 這次一律走免費且在時限內的 reply
- 在 LINE 內接現有 LLM `/api/chat` 對話層 — 回應時限與成本是另一個題目
- 顯式帳號綁定流程 — 靠同 provider 共用 userId（見 §2）
- `getProfile` 顯示 LINE 名稱／頭像 — 不是這次要的「個人資訊」
- 用 API 建立 Rich Menu — LINE console 手動設定
- `follow` / `unfollow` 事件處理（加好友歡迎訊息）— 已評估，這次刻意不做
- 正式 HTTPS 部署 — 開發階段沿用 ngrok

前端**零改動**。

---

## 2. 樞紐事實：同一個 Provider 底下，userId 與 Login 的 `sub` 是同一個值

LINE 官方文件（[Get user IDs](https://developers.line.biz/en/docs/messaging-api/getting-user-ids/)）：

> If the provider is the same, the user ID is the same regardless of the channel type.
> User IDs are issued different values for each provider, even for the same user.

亦即 user ID 是 **per provider** 唯一，而非 per channel。因此 Messaging API webhook 收到的
`events[].source.userId`，與 LINE Login ID token 的 `sub`（`line_auth.py` 已寫進
`user_metadata.line_sub`）**是同一個字串**，前提是兩個 channel 建在同一個 Provider 底下。

> 修正：`docs/line-login-liff-evaluation.md` §3.3 寫 `sub` 是「pairwise（每個 channel 一組）」
> 並不精確——pairwise 的粒度是 provider。實作時一併修正該行。

**這是整份設計的硬前提**：Messaging API channel 必須建在現有 LINE Login channel 所屬的
同一個 Provider 底下，否則 webhook 永遠對不到帳號。

---

## 3. 外部設定（LINE Developers Console，手動）

1. 在現有 Login channel 的 **同一個 Provider** 下新建 **Messaging API channel**（＝一個 LINE 官方帳號）。
2. Webhook URL 設為 `https://<ngrok-host>/api/line/webhook`；開啟 "Use webhook"；
   關閉「自動回應訊息」與「歡迎訊息」（避免與 bot 回覆打架）。
3. 建立 Rich Menu，一顆按鈕，action 型別為 **message**（送出文字「我的訓練摘要」）。
   *不可*用 URI／LIFF action——那不會觸發 webhook，也就沒有 `replyToken`。
4. 取得 **Channel secret**（驗簽用）與 **Channel access token**（呼叫 reply API 用）。

完成後補進 `docs/line-login-liff-setup.md`。

---

## 4. 設定值（`backend/app/settings.py`）

比照既有 `line_login_configured` 的形狀新增：

```python
line_messaging_channel_secret: str = ""
line_messaging_access_token: str = ""

@property
def line_messaging_configured(self) -> bool:
    """True when the webhook can both verify signatures and reply."""
    return bool(
        self.line_messaging_channel_secret
        and self.line_messaging_access_token
        and self.auth_configured
        and self.supabase_service_role_key
    )
```

未設定時 webhook 回 503，與 `/api/auth/line`、`/api/chat` 的既有行為一致。
`.env.example` 同步新增這兩個變數與說明註解。

註：`LINE_CHANNEL_ID` 是 **Login** channel 的 id（ID token audience），與這裡的 Messaging
channel 是兩個不同的 channel，兩組 secret 不可混用。

---

## 5. 資料存取：SECURITY DEFINER RPC

webhook 沒有使用者的 JWT，而 `services/store.py` 每支函式都以使用者 token 走 RLS。
評估過三條路（每則訊息重新 mint session、對映表＋service_role 讀表、SECURITY DEFINER RPC），
選定 **RPC**：

- 一次往返，不需新表，**沒有回填問題**（既有 LINE 使用者立刻可用）
- service_role 的資料權限被收斂成「一支明確定義的唯讀函式」，而非整個 schema 通行，
  最貼近本專案「後端不用 service_role 碰資料」的既有姿態
- 被否決的「每則訊息 mint session」有硬傷：`generate_link` magiclink 預設會自動建帳號，
  沒登入過的人敲 bot 就會被建出 `auth.users` 列；且每則訊息 2 次 Admin 往返並受 rate limit

### Migration：`db/migrations/<ts>_line_training_summary.sql`

```sql
create or replace function public.line_training_summary(p_line_sub text)
returns jsonb
language sql
security definer
set search_path = public, auth
as $$ ... $$;

revoke all on function public.line_training_summary(text) from public, anon, authenticated;
grant execute on function public.line_training_summary(text) to service_role;
```

行為：

1. 以 `auth.users.raw_user_meta_data->>'line_sub' = p_line_sub` 找出使用者；找不到回 `null`。
2. 找到則回傳 `jsonb`：
   - `total`：該使用者的 `analyses` 總筆數
   - `latest`：最近一筆的 `created_at`、`view_type`、`fault_count`（無資料時為 null）
   - `top_faults`：對 `result->'detections'` 展開後依 `fault_name` 聚合的前 3 名
     `[{name, count}, ...]`

`security definer` + 明確 `search_path` + 只 grant `service_role`：一般使用者（anon /
authenticated）無法呼叫，無法用別人的 `sub` 查別人的資料。

---

## 6. 元件

| 檔案 | 職責 |
|---|---|
| `backend/app/services/line_bot.py` | 以純函式為主：`verify_signature(raw_body, header)`、`summary_for_line_user(line_user_id)`（呼 RPC）、`format_summary(data)`（組中文訊息）、`reply(reply_token, text)`（打 LINE reply API）、`handle_events(payload)`（關鍵字比對＋決定回覆內容） |
| `backend/app/routers/line_webhook.py` | 薄 HTTP 層：讀 **raw bytes** → 驗簽 → 解析 → 交給 service → 一律回 200 |
| `db/migrations/<ts>_line_training_summary.sql` | §5 的 RPC |
| `backend/app/main.py` | 多一行 `include_router(line_webhook.router)` |

切分理由：驗簽、摘要格式化、關鍵字比對都是可獨立單測的純函式；只有 `reply` 與
`summary_for_line_user` 碰外部 I/O，測試時只需 mock 這兩處。

---

## 7. 資料流

```
使用者點圖文選單 / 傳「摘要」
  → LINE 平台 POST /api/line/webhook
       headers: X-Line-Signature
       body:    events[].{type, message.text, replyToken, source.userId}
  → 未設定 (line_messaging_configured = False) → 503
  → verify_signature(raw_bytes, header) 失敗 → 400
  → 非 text message 事件 → 忽略，200
  → 關鍵字未命中 → 回一則使用說明
  → 關鍵字命中 → rpc("line_training_summary", {"p_line_sub": userId})
       ├─ null      → 「請先在 x-coach 用 LINE 登入」＋ LIFF 連結
       ├─ total = 0 → 「還沒有分析紀錄」＋ LIFF 連結
       └─ 有資料    → format_summary → POST /v2/bot/message/reply
```

觸發關鍵字（大小寫與前後空白正規化後比對）：`我的訓練摘要`、`摘要`、`訓練`、`紀錄`、
`summary`。圖文選單按鈕送出的文字就是其中之一，所以選單與關鍵字共用同一條路徑。

回覆訊息（單則 text，繁體中文）範例：

```
📊 你的訓練摘要

累積分析：12 次
最近一次：2026-07-19 21:03（側面視角，偵測到 3 個問題）

最常出現的問題
1. 膝蓋內夾 ×7
2. 深度不足 ×5
3. 軀幹前傾過多 ×3

打開 x-coach 看完整報告 👉 https://liff.line.me/<LIFF_ID>
```

---

## 8. 錯誤處理

| 情況 | 行為 |
|---|---|
| 未設定 secret / access token | 503，不做任何事 |
| `X-Line-Signature` 缺失或不符 | 400。`hmac.compare_digest` 常數時間比對，且比對 **raw bytes**（不是 re-serialize 後的 JSON） |
| 驗簽通過後的任何內部錯誤 | 記 log、**仍回 200**。LINE 把非 2xx 視為 webhook 失敗，且事件已消費，重試無意義 |
| 非 text message 事件（sticker/image/join…） | 忽略，200 |
| 關鍵字未命中 | 回簡短使用說明 |
| RPC 查無對應使用者 | 回「請先用 LINE 登入 x-coach」＋ LIFF 連結 |
| 有帳號但零筆分析 | 回「還沒有分析紀錄」＋ LIFF 連結 |
| reply API 失敗／逾時 | `httpx` timeout 10s（同 `line_auth`）；失敗記 log 後回 200，**不重試**（`replyToken` 只能用一次、1 分鐘內有效） |
| log 內容 | 只記事件型別與處理結果；**不記 userId 全值、不記摘要內容** |

一個 webhook request 可能含多個 event；逐一處理，單一 event 失敗不影響其他 event。

---

## 9. 測試策略

新增 `tests/test_backend_line_webhook.py`，沿用 `tests/test_backend_line_auth.py` 的既有作法
（monkeypatch `get_settings`、假 supabase client、`TestClient`）：

- `verify_signature`：正確簽章通過／竄改 body 失敗／缺 header 失敗／空 secret 失敗
- router：未設定 → 503；壞簽章 → 400；非 text 事件 → 200 且不呼叫 reply；
  未知關鍵字 → 回使用說明
- `summary_for_line_user`：RPC 回 `None`／回 `total = 0`／回正常資料 三條路徑
- `format_summary` 純函式：top-3 截斷、只有 1 筆分析、無 fault、時間格式
- RPC 拋例外 → 回 200 且不拋出

SQL 函式本身不進 pytest（Python 端 mock RPC 回傳），改以 migration 內註解說明權限意圖，
並在 `docs/line-login-liff-setup.md` 列出手動驗證步驟（用 service_role 呼叫回正確資料、
用 anon key 呼叫被拒）。

CI 的 95% backend coverage gate 必須通過：
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

---

## 10. 驗收條件

1. `.venv\Scripts\python.exe -m pytest tests/` 全綠，coverage gate ≥ 95%。
2. ngrok 對外時，用 LINE 官方帳號點圖文選單，聊天室收到自己的訓練摘要，數字與
   x-coach History 頁一致。
3. 尚未用 LINE 登入過的帳號敲 bot，得到引導登入的訊息，且**沒有**建立任何 `auth.users` 列。
4. 用 anon key 直接呼叫 `line_training_summary` 被 Postgres 拒絕。
5. 竄改過的 request body 打 webhook 得到 400。
