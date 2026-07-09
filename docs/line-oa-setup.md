# LINE 官方帳號（LINE OA）AI 教練設定指南

讓使用者可以在 LINE 上，接續自己在 x-coach 網頁做的深蹲分析，繼續向 AI 教練追問。

本文涵蓋：LINE Developer Console 的 channel 設定、後端環境變數、資料庫 migration、webhook 串接，
以及連結（linking）流程與測試方式。

> **這個整合在做什麼？**
> LINE 使用者用 LINE userId 辨識，webhook 進來時 **沒有 Supabase 登入 token**。所以流程設計成：
> 已登入的網頁使用者先產生一組「連結碼」（把該次分析的 grounding 快照一起存起來），LINE 使用者把
> 連結碼傳給 bot 完成綁定，之後每則訊息就用同一套 grounded prompt 回答。webhook 只碰兩張 LINE 專用
> 資料表（`line_link_codes`、`line_bindings`），用 Supabase 的 **service_role** 金鑰存取；這兩張表
> 的 RLS 開啟但**不給 anon/authenticated 任何 policy**，等於只有後端能碰。

---

## 架構總覽

```
[網頁 已登入使用者]
      │  1. 點「連結 LINE」→ POST /api/line/link-code  (帶 buildChatContext 的分析快照)
      ▼
[後端]  產生一次性連結碼，寫入 line_link_codes（含 grounding 快照，15 分鐘後過期）
      │  回傳連結碼，例如 ABC234
      ▼
[使用者] 切到 LINE，把「連結 ABC234」傳給官方帳號
      ▼
[LINE 平台] ── webhook ──▶ POST /api/line/webhook  (帶 X-Line-Signature)
      ▼
[後端]  驗簽 → 兌換連結碼 → 建立 line_bindings（綁定 LINE userId ↔ 分析快照）→ 回覆「已連結」
      ▼
[使用者] 之後直接用文字提問 → 後端用 answer_once（同一套 grounded prompt）產生回覆 → LINE reply API 推回
```

涉及的後端檔案：

| 檔案 | 職責 |
|------|------|
| `backend/app/routers/line.py` | `POST /api/line/webhook`（驗簽、事件分派、背景回覆）+ `POST /api/line/link-code`（產生連結碼） |
| `backend/app/services/line_client.py` | `X-Line-Signature` HMAC 驗證 + LINE reply/push API |
| `backend/app/services/line_store.py` | service_role 存取兩張 LINE 表（連結碼、綁定） |
| `backend/app/services/chat.py` | `answer_once()` — 非串流的 grounded 回覆（給 LINE 用） |
| `db/migrations/20260709000000_line_bindings.sql` | 建立 `line_link_codes` + `line_bindings` |

---

## 步驟 1：建立 Messaging API channel

你已經開好 LINE OA 帳號，接下來要讓它能收發訊息（Messaging API）。

1. 到 [LINE Developers Console](https://developers.line.biz/console/) 用 LINE 帳號登入。
2. 選（或建立）一個 **Provider**（例如你的團隊/專案名稱）。
3. 建立一個 **Messaging API channel**：
   - 若你的 OA 是在 [LINE Official Account Manager](https://manager.line.biz/) 建立的，到該 OA 的
     **設定 → Messaging API → 啟用 Messaging API**，並綁定到上面的 Provider。啟用後這個 channel 就會
     出現在 Developers Console 裡。
   - 也可以直接在 Developers Console 內用 **Create a Messaging API channel** 建立。
4. 建好後，在 Developers Console 打開這個 channel，準備取用下面兩個機密值。

---

## 步驟 2：取得三組機密值

| 環境變數 | 從哪裡拿 |
|----------|----------|
| `LINE_CHANNEL_SECRET` | channel → **Basic settings** 分頁 → **Channel secret** |
| `LINE_CHANNEL_ACCESS_TOKEN` | channel → **Messaging API** 分頁 → **Channel access token (long-lived)** → **Issue** |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase 專案 → **Project Settings → API → `service_role` secret** |

> ⚠️ **三組都是伺服器端機密，絕不能放進前端或 commit 進 repo。** 特別是 `service_role` 金鑰會
> 繞過 Supabase RLS，本專案只在 `services/line_store.py` 用它存取兩張 LINE 表，不要挪作他用。

---

## 步驟 3：設定後端環境變數

在 repo 根目錄的 `.env`（gitignored）加上（可參考 `.env.example`）：

```bash
# 前提：Supabase 與 LLM 已設定好（LINE 教練要靠這兩者才能運作）
SUPABASE_URL=https://<你的專案>.supabase.co
SUPABASE_ANON_KEY=...
LLM_API_KEY=...            # 見 backend/README.md § Conversational coaching

# LINE 整合
LINE_CHANNEL_SECRET=...
LINE_CHANNEL_ACCESS_TOKEN=...
SUPABASE_SERVICE_ROLE_KEY=...
```

只要 `LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`、`SUPABASE_SERVICE_ROLE_KEY`、`SUPABASE_URL`
與一組可用的 `LLM_API_KEY` 全部到位，`GET /api/health` 會回報 `"line_configured": true`。少任何一項，
webhook 會直接回 200 但不處理、`POST /api/line/link-code` 回 503（避免半殘狀態）。

---

## 步驟 4：套用資料庫 migration

在 Supabase 建立 LINE 用的兩張表（作法同 `db/migrations/README.md`）：

- **Dashboard**：Supabase 專案 → SQL Editor → 貼上 `db/migrations/20260709000000_line_bindings.sql` → Run。
- **CLI**：`psql "$SUPABASE_DB_URL" -f db/migrations/20260709000000_line_bindings.sql`

套用後在 Table Editor 確認 `line_link_codes`、`line_bindings` 存在、RLS 已啟用，且 **policy 數為 0**
（這是刻意的——只有 service_role 能碰）。

---

## 步驟 5：把 webhook URL 設給 LINE

1. 後端要能被 LINE 公開存取。本機開發可用通道工具（例如 `ngrok http 8000`）取得對外 HTTPS 網址。
2. 在 channel → **Messaging API** 分頁：
   - **Webhook URL** 填：`https://<你的網域>/api/line/webhook`
   - 開啟 **Use webhook**。
   - 按 **Verify**：後端已設定好的話會回 200（`line_configured` 為 false 時也會回 200，但不會處理訊息）。
3. 同一分頁把 LINE 預設的自動回覆關掉，避免和 bot 搶話：
   - **Auto-reply messages**：停用。
   - **Greeting messages**：可留可停（本 bot 在使用者加好友的 `follow` 事件會自己送一則歡迎與連結指引）。

---

## 步驟 6：連結流程（linking）

LINE webhook 沒有使用者登入資訊，所以要由**已登入的網頁使用者**先產生連結碼。

**後端 API（已就緒）：** `POST /api/line/link-code`，需帶 `Authorization: Bearer <supabase access token>`，
body 是網頁 `buildChatContext(analysis)` 產生的 grounding 快照：

```bash
curl -X POST https://<你的網域>/api/line/link-code \
  -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"context": { /* buildChatContext(analysis) 的輸出 */ }}'
# → {"code": "ABC234", "expires_in_minutes": 15}
```

使用者拿到碼後，在 LINE 對 bot 傳：

- `連結 ABC234`（或 `綁定 ABC234` / `link ABC234`），或
- 直接貼上 `ABC234`（bot 也認得純代碼）

bot 兌換成功後回「✅ 已連結你的分析」，之後所有文字訊息都會用該次分析的 grounding 回答。重新分析後
再產生新碼、再傳一次即可切換到新的分析（會重置對話串）。

> **前端整合（尚未做，屬下一步）：** 目前後端端點已完成，但網頁上的「連結 LINE」按鈕還沒接。要做的話，
> 在 CoachTray/分析頁加一顆按鈕，呼叫上面的 `/api/line/link-code`（用 `buildChatContext(analysis)`），
> 把回傳的 `code` 顯示給使用者，或組成 `https://line.me/R/oaMessage/@<你的OA id>/?連結%20ABC234`
> 這種深連結讓使用者一點就把代碼帶進 LINE 輸入框。

---

## 測試與驗證

**後端單元測試**（不需真的打 LINE / LLM / Supabase；都以 mock 取代）：

```bash
# 本機開發環境（Windows 見 CLAUDE.md，用 .venv\Scripts\python.exe）
python -m pytest tests/test_line_endpoint.py
```

涵蓋：簽章驗證、連結碼兌換（有效／未知／過期）、綁定使用者提問→grounded 回覆→存檔、未綁定引導、
LLM 失敗的優雅退場、以及 webhook 路由的驗簽與背景排程。

**線上煙霧測試（smoke test）：**

1. `GET /api/health` → 確認 `"line_configured": true`。
2. LINE Developers Console 按 **Verify** → 200。
3. 用 LINE 加官方帳號好友 → 應收到歡迎訊息。
4. 未連結時傳一句提問 → 應收到「請先產生連結碼」的引導。
5. 依步驟 6 產生碼、傳給 bot → 收到「已連結」→ 再提問 → 收到 grounded 回覆。

---

## 設計備註與已知限制

- **同步呼叫 LLM、背景回覆。** webhook 收到事件後先回 200 給 LINE（避免 LINE 重送導致重複回覆），
  再在 FastAPI `BackgroundTasks` 裡呼叫 LLM 並用 reply token 回覆。reply token 有效期有限，正式環境
  若 LLM 偶爾很慢，可改成先回一則「思考中…」再用 push API 補上答案，或改接工作佇列。
- **service_role 的取捨。** 這是本專案唯一使用 service_role 的路徑，因為 bot 是 server-to-server、
  webhook 沒有使用者 JWT。爆炸半徑靠紀律限制：`line_store` 只查兩張 `line_*` 表；分析內容是在連結時
  由已登入使用者快照進 binding，webhook 不去讀 `analyses`/`conversations`。
- **多 worker。** 綁定狀態存在 Supabase，不是行程內記憶體，所以 `uvicorn --workers N` 沒問題。
- **連結碼壽命。** 一次性、15 分鐘過期（`line_store._CODE_TTL`），兌換後即刪除。過期而未兌換的碼可
  依 `expires_at` 定期清掃。
- **訊息長度。** LINE 文字上限 5000 字，`answer_once` 截到 4900 字以策安全（教練回覆通常遠短於此）。
