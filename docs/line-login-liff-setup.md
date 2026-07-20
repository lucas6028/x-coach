# LINE Login + LIFF 設定手冊(Runbook)

Status: **程式碼已實作,待外部服務設定** · Created 2026-07-15 ·
決策背景見 `line-login-liff-evaluation.md`

程式碼側(前端 LIFF 啟動、`signInWithLine`、後端 `/api/auth/line` 橋接、`/liff/diag` 檢測頁)
已全部就緒;本文件是把它接上真實 LINE channel 與 Supabase 專案的操作步驟。

---

## 0. 架構速覽:一個橋接,兩種進入點

**web 與 LINE App 內走同一條路**:都用 LIFF 拿到 LINE ID token,再 `POST /api/auth/line`
交給後端橋接驗證並鑄造「真正的 Supabase session」。因此同一個 LINE 使用者不論從哪進來
**都是同一個 Supabase 帳號**,RLS、歷史紀錄、聊天、管理員機制完全共用:

| 情境 | 路徑 | 程式碼 |
|---|---|---|
| **一般網頁**(含 LIFF 外部瀏覽器) | `liff.login()` 導去 LINE 登入 → 返回後 `liff.getIDToken()` → `POST /api/auth/line` → `setSession` | `frontend/src/lib/auth.tsx` `signInWithLine()` + auto-login effect(返回後在原頁完成交換) |
| **LINE App 內(LIFF)** | 已登入,直接 `liff.getIDToken()` → `POST /api/auth/line` → `setSession` | 同上;開啟即自動登入 |

> **為什麼不用 Supabase Custom OIDC 接 LINE?** LINE 的 ID token 用 **HS256**(以 channel
> secret 為 HMAC key)簽章,但 Supabase 的 custom OIDC provider 依 LINE discovery 只接受
> **ES256**(對 JWKS 公鑰驗證),兩者結構上不相容 —— web 登入會回
> `Error getting user profile from external provider`(GoTrue:`id token signed with
> unsupported algorithm, expected ["ES256"] got "HS256"`)。後端橋接改用 LINE 官方 verify
> endpoint,不受簽章演算法影響,所以兩條路都走它。

後端橋接細節(合成 email 設計、service_role 只用於鑄造 session)見
`backend/app/services/line_auth.py` 檔頭註解。

---

## 1. LINE Developers Console

1. 到 [developers.line.biz](https://developers.line.biz/console/) 建立(或使用既有)**Provider**。
2. 建立 **LINE Login channel**:
   - **Channel ID** → 填入根目錄 `.env` 的 `LINE_CHANNEL_ID`(後端驗 ID token 的 audience)。
   - **Channel secret** → 橋接不需要(用不到 Supabase Custom OIDC 了);保管好即可。
   - Scopes:勾選 **openid**、**profile**。(**email** 需另外向 LINE 申請審核,先不用)
   - **Callback URL**:channel 要求至少填一個,填前端網址即可
     (dev 填 ngrok 網址,見 §4)。LIFF 的登入導向由 LINE 經 `liff.line.me` 自動處理,
     **不需要**再填 Supabase 的 `/auth/v1/callback`。
3. 在同一個 channel 底下建立 **LIFF app**(LIFF tab → Add):
   - **Endpoint URL**:HTTPS 前端網址 —— dev 先填 ngrok 網址(見 §4),正式部署後改。
   - **Size**:`Full`(全螢幕,最像 App)。
   - **Scopes**:`openid`、`profile`(沒有 openid 就拿不到 ID token,橋接會失敗)。
   - 產生的 **LIFF ID**(形如 `1234567890-Abcdefgh`)→ 填 `frontend/.env` 的 `VITE_LIFF_ID`。

## 2. Supabase Dashboard

> **不需要** Custom OIDC provider —— web 與 App 內都走橋接(見 §0 的 HS256/ES256 說明)。
> 若你之前照舊版手冊建了名為 `line` 的 custom provider,可以直接刪掉,已無任何程式碼引用。

1. **service_role key(LIFF 橋接,唯一必要設定)**:Project Settings → API → `service_role`
   → 填入根目錄 `.env` 的 `SUPABASE_SERVICE_ROLE_KEY`。
   ⚠️ 只給後端、只被 `services/line_auth` 用來建使用者與產生一次性登入連結;
   絕不放進 `frontend/`、絕不 commit。

橋接用 `generate_link` + `verify_otp` 鑄造 session,不經過瀏覽器 OAuth 導向,
所以 **不需要**在 Authentication → URL Configuration 設 redirect URLs。

## 3. 環境變數總表

| 檔案 | 變數 | 用途 |
|---|---|---|
| `.env`(根目錄) | `LINE_CHANNEL_ID` | 後端驗 LINE ID token 的 audience |
| `.env` | `SUPABASE_SERVICE_ROLE_KEY` | LIFF 橋接鑄造 session(僅此用途) |
| `.env` | `XCOACH_CORS_ORIGINS`(選) | 跨域直連 API 時的額外 CORS 來源;走 Vite proxy 則不需要 |
| `frontend/.env` | `VITE_LIFF_ID` | 前端 `liff.init` 用;未設則整個 LIFF 行為關閉、僅剩 web 路徑 |

兩者未設時的行為:`/api/auth/line` 回 503、前端不載入 LIFF SDK —— 一切退回現況,不會壞。

## 4. Dev:用 ngrok 提供 HTTPS(LIFF 硬性要求)

```
# 1. 後端(repo 根目錄)
uvicorn backend.app.main:app --reload --port 8000
# 2. 前端(cwd = frontend/)
yarn dev
# 3. 隧道(免費方案每次網址會變)
ngrok http 5173
```

- 把 ngrok 給的 `https://xxxx.ngrok-free.app` 填到 LIFF 的 **Endpoint URL**
  (每次重開 ngrok 要更新)與 Supabase 的 redirect URLs。
- `vite.config.ts` 已加 `server.allowedHosts`(`.ngrok-free.app` 等),不用再改。
- SPA 經 ngrok → Vite → `/api` proxy → 後端,全程**同源**,CORS 不會擋。

## 5. Phase 0:真機檢測(先做這步再決定相機範圍)

用手機 LINE 開啟 LIFF 網址(`https://liff.line.me/<LIFF_ID>`),導到 **`/liff/diag`**:

1. **執行環境**表:`liff.init` 應為 `ok`、`isInClient` 為 `true`、`ID token` 為 `present`。
2. **登入 Session**:若後端橋接已設定,開頁應已自動登入(顯示 `line_<sub>@line.invalid`)。
3. **即時相機**(快篩):按「測試相機」——
   - `ok` → 相機打得開,續做步驟 4 才能下結論。
   - `timeout` / `unsupported`(iOS 已知問題)→ LIFF 內即時相機不可用,步驟 4 免做;
     遊戲頁會自動顯示「改用外部瀏覽器」提示(`camera.liffHint`),主流程的影片上傳不受影響。
   - `denied` → 單純權限被拒,再試並允許即可。
4. **即時姿態(相機 + MediaPipe)**(決定性測試):按「測試相機＋姿態」——
   相機通了**不代表**遊戲可玩:WebView 裡的 WASM/WebGL 是獨立的失敗點
   (模型下載、GPU shader 編譯可凍住主執行緒數十秒、推論 FPS 掉到不可玩)。
   此測試跑的是與遊戲完全相同的鏈路(同一個 `createPoseLandmarker`、同樣的相機參數),
   量測 5 秒後回報:
   - `model` / `warmup` 毫秒數:模型載入與首次推論(shader 編譯)時間。
   - `fps`:**≥15 可玩;8–14 勉強;<8 不可玩**(頁面會直接給結論)。
   - `landmarks: no` → 鏈路正常但沒拍到人,對準自己再測。
   - 失敗時會顯示卡在哪一段(`camera` / `video` / `model` / `warmup` / `detect`)。
5. **影片檔案拍攝**:點檔案輸入,確認能否直接開相機錄影(上傳分析流程的替代路徑)。

請在 **iOS 與 Android 各測一台**,以步驟 4 的 FPS 結論決定要不要把即時相機功能留在 LIFF 內。

## 6. 已知設計取捨(troubleshooting 前先讀)

- **合成 email**:LIFF 橋接建立的使用者 auth email 是 `line_<sub>@line.invalid`
  (deterministic、永不可達、不會撞真實帳號)。管理員列表會看到這種地址;
  真名/頭像在 `user_metadata`(`full_name` / `avatar_url`),前端顯示正常。
- **provider-scoped sub**:LINE 的使用者 ID(OIDC `sub` / Messaging API 的 `source.userId`)
  是「每個 provider 一組」,同一個 provider 底下所有 channel(LINE Login、LIFF、Messaging
  API bot)共用同一個值 —— 不是「每個 channel 一組」。dev 與 prod **若掛在同一個 provider**
  下,會共用同一組使用者資料(帳號不會分家,但也會**混在一起**);要讓 dev/prod 互相隔離,
  必須各自建一個獨立的 **Provider**,而不是在同一個 provider 底下開不同 channel。
  web 與 App 內都走同一條橋接、以 `sub` 衍生同一個合成 email,所以**只要 LIFF app 掛在
  與 `LINE_CHANNEL_ID` 相同的 LINE Login channel 底下**,同一人兩種進入點就是同一個帳號。
- **一次性 reload 保險**:LIFF 內 ID token 過期時會 `liff.login()` 重載一次;同一瀏覽器
  session 內只會嘗試一次(`sessionStorage` 旗標),不會無限迴圈。
- **登出後不會被自動再登入**:auto-login 每次載入只嘗試一次。

## 7. 驗證

```
# 後端(容器/CI 環境用 python3;Windows 本機用 .venv\Scripts\python.exe)
python3 -m pytest tests/test_backend_line_auth.py -q
python3 scripts/run_backend_coverage.py --fail-under 95
# 前端(cwd = frontend/)
yarn test
yarn test:coverage
```

## 8. LINE Messaging API bot(聊天室查訓練摘要)

> 硬前提:Messaging API channel 必須建在**與 LINE Login channel 相同的 Provider** 底下。
> LINE 的 user ID 是每個 provider 一組,同 provider 下的 channel 共用同一個值 —— 這是 bot
> 能用 webhook 的 `source.userId` 直接對到登入帳號、不需要綁定流程的唯一原因。

1. LINE Developers Console → 選到現有 Login channel 所屬的 Provider → **Create a new channel**
   → **Messaging API**(會同時建立一個 LINE 官方帳號)。
2. **Basic settings** → 複製 **Channel secret** → 填入 `.env` 的 `LINE_MESSAGING_CHANNEL_SECRET`。
3. **Messaging API** 分頁 → **Channel access token (long-lived)** → Issue → 填入
   `LINE_MESSAGING_ACCESS_TOKEN`。
4. 同一分頁 → **Webhook URL** 填 `https://<ngrok-host>/api/line/webhook` → 開啟 **Use webhook**
   → 按 **Verify**(後端要先啟動;未設定環境變數會回 503)。
5. 在 LINE Official Account Manager 關閉「自動回應訊息」與「加入好友的歡迎訊息」,
   否則會與 bot 的回覆打架。
6. **圖文選單(Rich menu)**:LINE Official Account Manager → 建立圖文選單 → 動作型別選
   **傳送文字**,文字填 `我的訓練摘要`。
   ⚠️ 不可選「連結(URI/LIFF)」—— 那不會觸發 webhook,也就拿不到 `replyToken`。
7. `.env` 的 `LINE_LIFF_ID` 填既有 LIFF app id(可留空,只是回覆訊息會少一個連結)。
8. 在 Supabase SQL editor 執行 `db/migrations/20260720000000_line_training_summary.sql`。

### 手動驗證

- 用 **service_role** key 呼叫 `select public.line_training_summary('<你的 LINE sub>')`:
  回傳 `{"total": ..., "latest": ..., "top_faults": [...]}`,數字與 x-coach History 頁一致。
- 用 **anon** key 呼叫同一支函式:應被 Postgres 拒絕(permission denied)。
- 沒登入過 x-coach 的 LINE 帳號敲 bot:得到引導登入的訊息,且 Supabase 的
  `auth.users` **沒有**多出任何一列。

## 9. LIFF App Shell(LINE App 內的分頁式介面,2026-07-20)

在 LINE App 內,SPA 改渲染 `LiffAppShell`(底部分頁:分析 / 我的紀錄 / 小遊戲 / 設定,對應
路由 `/app`、`/history`、`/games`、`/settings`),取代一般網頁的 navbar + sidebar。
偵測邏輯在 `frontend/src/lib/liffContext.tsx`(`LiffProvider` / `useLiffContext()`);
`AppLayout` 依 `isInClient` 分流,套用這層 shell 換裝時頁面元件不需要知道 LIFF 的存在——
但 `Landing.tsx`、`Login.tsx`、`Games.tsx` 為了各自的理由(in-client 導向、相機提示)仍直接
呼叫 `useLiffContext()`。

需要的 console 設定:

- LIFF app **Endpoint URL** → 網站根目錄 `https://<host>/`,**不要**填 `/app`:LIFF 的
  URL 轉發是把深連結多出來的路徑接在 Endpoint URL「既有的完整路徑」後面,不是只接在
  origin 後面——若填 `/app`,`https://liff.line.me/{liffId}/history` 會解析成
  `https://<host>/app/history`,但 SPA 沒有這條路由。填根目錄後,一般啟動(不帶額外路徑)
  仍會落在工作室:in-client 時 `Landing.tsx` 的 `isInClient` 分支會把 `/` 導向 `/app`。
- LIFF app **Size** → `Full`(沿用 §1 既有設定)。
- 不需要新增 scope。`liff.sendMessages()` / share target picker 是之後的階段才會用到,
  屆時才需要 `chat_message.write` 權限與 share-target-picker 開關。

圖文選單(Rich menu)的項目可以深連結到 `/app`、`/history`、`/games`、`/settings` 中的
任一路由——填根目錄後這些深連結會直接解析成對應的 SPA 路由。
