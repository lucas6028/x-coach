# LINE Login + LIFF 設定手冊(Runbook)

Status: **程式碼已實作,待外部服務設定** · Created 2026-07-15 ·
決策背景見 `line-login-liff-evaluation.md`

程式碼側(前端 LIFF 啟動、`signInWithLine`、後端 `/api/auth/line` 橋接、`/liff/diag` 檢測頁)
已全部就緒;本文件是把它接上真實 LINE channel 與 Supabase 專案的操作步驟。

---

## 0. 架構速覽:兩條登入路徑,一個結果

兩條路都終結於「真正的 Supabase session」,所以 RLS、歷史紀錄、聊天、管理員機制完全共用:

| 情境 | 路徑 | 程式碼 |
|---|---|---|
| **一般網頁**(含 LIFF 外部瀏覽器) | Supabase Custom OIDC redirect(provider `custom:line`) | `frontend/src/lib/auth.tsx` `signInWithLine()` 後半 |
| **LINE App 內(LIFF)** | `liff.getIDToken()` → `POST /api/auth/line` → 後端驗證並鑄造 session → `setSession` | 同上前半 + `backend/app/services/line_auth.py`;開啟即自動登入(`auth.tsx` auto-login effect) |

後端橋接細節(合成 email 設計、service_role 只用於鑄造 session)見
`backend/app/services/line_auth.py` 檔頭註解。

---

## 1. LINE Developers Console

1. 到 [developers.line.biz](https://developers.line.biz/console/) 建立(或使用既有)**Provider**。
2. 建立 **LINE Login channel**:
   - **Channel ID** → 填入根目錄 `.env` 的 `LINE_CHANNEL_ID`(後端驗 ID token 的 audience)。
   - **Channel secret** → 之後給 Supabase Custom OIDC 用。
   - Scopes:勾選 **openid**、**profile**。(**email** 需另外向 LINE 申請審核,先不用)
   - **Callback URL**:填 Supabase 的 OAuth callback
     `https://<你的專案>.supabase.co/auth/v1/callback`(web 路徑用)。
3. 在同一個 channel 底下建立 **LIFF app**(LIFF tab → Add):
   - **Endpoint URL**:HTTPS 前端網址 —— dev 先填 ngrok 網址(見 §4),正式部署後改。
   - **Size**:`Full`(全螢幕,最像 App)。
   - **Scopes**:`openid`、`profile`(沒有 openid 就拿不到 ID token,橋接會失敗)。
   - 產生的 **LIFF ID**(形如 `1234567890-Abcdefgh`)→ 填 `frontend/.env` 的 `VITE_LIFF_ID`。

## 2. Supabase Dashboard

1. **Custom OIDC provider(web 路徑)**:Authentication → Sign In / Providers →
   Custom OAuth/OIDC → 新增,**名稱填 `line`**(前端寫死呼叫 `custom:line`):
   - 模式選 **OIDC auto-discovery**,Issuer:`https://access.line.me`
     (discovery 文件在 `https://access.line.me/.well-known/openid-configuration`)。
   - Client ID / Client secret:LINE Login channel 的 Channel ID / Channel secret。
   - Scopes:`openid profile`。
   - 免費方案最多 3 個 custom provider,夠用。
2. **service_role key(LIFF 橋接)**:Project Settings → API → `service_role` →
   填入根目錄 `.env` 的 `SUPABASE_SERVICE_ROLE_KEY`。
   ⚠️ 只給後端、只被 `services/line_auth` 用來建使用者與產生一次性登入連結;
   絕不放進 `frontend/`、絕不 commit。
3. **Redirect URLs**:Authentication → URL Configuration → 把前端網址(ngrok / 正式域名)
   加進 allowed redirect URLs,否則 web 路徑登入完會被擋。

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
- **pairwise sub**:LINE 的 `sub` 是「每個 channel 一組」。dev 與 prod 用不同 channel
  會產生**不同帳號**;正式上線前就決定好 channel,別讓使用者資料分家。
  同理,web 路徑(custom OIDC)與 LIFF 橋接**必須用同一個 LINE Login channel**,
  否則同一人兩條路登入會變兩個帳號。
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
