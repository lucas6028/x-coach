# LINE Login + LIFF 導入評估

Status: **已決策並實作（方案 A + B 並行）；外部服務設定見 `line-login-liff-setup.md`** ·
Created 2026-07-15 · Updated 2026-07-16

評估「在 x-coach 導入 LINE Login 作為登入方式，並以 LINE LIFF 讓前端能在 LINE App 內執行、
取得類似原生 App 的體驗」的可行性、架構選項、風險與落地計畫。

> **2026-07-16 決策**：完整 LIFF App 化、LIFF／相機不可用時 fallback 回網頁；dev 先用
> ngrok；接受方案 B 的 service_role。§8 的 Phase 1（方案 A 前端）、Phase 3（方案 B 後端
> 橋接 `/api/auth/line`）與 Phase 0 的檢測頁（`/liff/diag`）＋ 相機 timeout fallback 均已
> 實作；剩餘手動步驟是 Phase 0 真機驗證與 LINE／Supabase 控制台設定（見設定手冊）。

本文結論建立在實際程式碼（`frontend/src/lib/auth.tsx`、`frontend/src/api.ts`、
`backend/app/auth.py`、`db/migrations/*`）與 LINE / Supabase 官方文件之上，非泛論。

---

## 0. 一分鐘摘要（TL;DR）

- **可行，且切點乾淨。** 目前身分完全由 Supabase Auth 掌管，後端無狀態；只要 LINE 使用者
  最終能拿到「一個真正的 Supabase session（真的 `auth.users` 列 + 真的 Supabase JWT）」，
  現有的 RLS、`user_roles` 管理員機制、`_verify` token 驗證、`api.ts` bearer 附帶邏輯
  **全部不用改**。
- **「LINE Login」與「LIFF」是兩件可分開的事。** 前者是「多一個登入提供者」；後者是
  「把 SPA 放進 LINE 內執行的通路 / App 化外殼」。可以只做前者，也可以兩者一起做。
- **Supabase 原生不支援 LINE**（`signInWithOAuth` / `signInWithIdToken` 的 provider 都是
  固定 enum，不含 LINE）。**但** Supabase 已推出 **Custom OIDC Provider**，而 LINE Login v2.1
  是完整 OIDC 提供者（`https://access.line.me/.well-known/openid-configuration`、ES256、JWKS），
  因此可用 `custom:line` 接上，**後端幾乎零改動**（方案 A，建議優先）。
- **本專案最大的專屬風險：LIFF 內建瀏覽器在 iOS 上 `getUserMedia`（即時相機）有已知失效問題。**
  x-coach 的即時相機功能（SixSeven、FruitNinja、即時姿態）依賴它。上傳影片分析（主流程）
  多半不受影響，但「在 LINE 內開相機」需要先用真機做 PoC 驗證，不能假設可行。
- **建議路線：** 先做方案 A（Web 版 LINE Login，投入最小、風險最低）→ 驗證 LIFF 內相機是否可用
  → 若要 LINE 內無縫登入再加方案 B 的後端橋接。詳見 §8 分階段計畫。

---

## 1. 現況：目前的登入架構（改動基準）

**前端（`frontend/`）**
- 只用 `@supabase/supabase-js`，session 存在瀏覽器（`persistSession` / `autoRefreshToken` /
  `detectSessionInUrl`，見 `frontend/src/lib/supabase.ts:15-24`）。
- `AuthProvider`（`frontend/src/lib/auth.tsx`）以 `onAuthStateChange` 把 session 灌進 React；
  已有三種登入：email/password 與 **Google OAuth**（`signInWithOAuth({provider:"google"})`，
  `auth.tsx:124-130`）。
- `api.ts` 每次請求從 `supabase.auth.getSession()` 取 `access_token`，加 `Authorization: Bearer`
  （`frontend/src/api.ts:296-311`）。
- Supabase env 未設定時 `supabase === null`，App 進入匿名 demo 模式。

**後端（`backend/`）— 無狀態**
- `backend/app/auth.py:_verify` 是「token → 使用者」的**唯一**入口：呼叫
  `create_client(url, anon_key).auth.get_user(token)`（即 Supabase `GET /auth/v1/user`），
  **不在本地驗簽**，因此任何簽章方案都通用（`auth.py:45-82`）。
- 三個 dependency：`get_current_user`（必須登入）、`get_optional_user`（demo 上傳可匿名）、
  `get_admin_user`（多一層 `store.is_admin`）。
- **後端從不持有 service_role key**，一律用「使用者自己的 JWT」打 Postgres，讓 RLS 當最後防線
  （`.env.example:7-8`、`store._user_client` → `postgrest.auth(token)`）。

**資料庫 / RLS（`db/migrations/`）**
- 使用者表就是 Supabase 的 `auth.users`，本專案不自建 users 表。
- `videos` / `analyses` / `conversations` 皆 `user_id uuid references auth.users(id)`，
  RLS `auth.uid() = user_id`（`20260620000000_init_videos_analyses.sql:87-102`）。
- 管理員：`public.user_roles`（PK = `auth.users.id`）+ `is_admin(auth.uid())` SECURITY DEFINER
  函式；`admin_list_users()` 直接讀 `auth.users`（`20260713000000_admin_roles.sql`、`...000200`）。

> **兩個關鍵接縫**：
> 1. **Token 驗證接縫** — `auth._verify`：必須產出 `CurrentUser(id, token, email)`，其中 `id` 是
>    `auth.users.id`（UUID）、`token` 是 Postgres RLS 認得的 JWT。
> 2. **RLS token 接縫** — `store._user_client`：把同一顆 token 交給 Postgres，policy 檢查
>    `auth.uid() = user_id`。
>
> 只要 LINE 流程的產物是「真正的 Supabase session」，這兩個接縫都不用動——這是整份評估的樞紐。

---

## 2. 兩個要分清楚的功能

| | **A. LINE Login（認證）** | **B. LIFF（App 化執行環境）** |
|---|---|---|
| 是什麼 | 多一個「用 LINE 登入」的身分提供者 | 把 SPA 掛成 LIFF app，在 LINE 內建瀏覽器開啟 |
| 受惠對象 | 一般網頁使用者也受惠 | 從 LINE 官方帳號 / 圖文選單進來的使用者 |
| App 化能力 | 無（只是登入按鈕） | 全螢幕、圖文選單入口、`liff.sendMessages` 回傳訊息到聊天室、`shareTargetPicker`、`closeWindow`、深連結 |
| 是否相依 | 可獨立做 | 通常搭配 LINE Login（LIFF 內已登入 LINE → 可無縫換身分） |

可以只做 A（風險最低）；要「在 LINE 裡像 App 一樣用」才需要 B。兩者一起做時，LIFF 內因使用者
已登入 LINE，可做到「幾乎無登入畫面」的體驗。

---

## 3. 關鍵技術限制與事實（決策依據）

1. **Supabase 原生不支援 LINE。**
   - `signInWithIdToken` 只接受 `google | apple | azure | facebook | kakao`。
   - `signInWithOAuth` 的 provider 也是固定 enum（github、google、kakao… 不含 line）。
   - ⇒ 前端**無法**直接用原生 API 接 LINE。（先前草稿若寫 `signInWithOAuth({provider:'line'})`
     或 `signInWithIdToken({provider:'line'})` 皆為誤，實務上會被拒。）

2. **但 Supabase 有 Custom OIDC Provider。** 可在 Dashboard 加入任何標準 OIDC/OAuth2 provider，
   之後用 `signInWithOAuth({ provider: 'custom:line' })` 呼叫，行為與內建 provider 相同
   （產生 `auth.users` 列、發真的 Supabase JWT、RLS 照常）。Free plan 最多 3 個 custom provider。

3. **LINE Login v2.1 是完整 OIDC 提供者**，可直接被 OIDC 自動探索接上：
   - discovery：`https://access.line.me/.well-known/openid-configuration`
   - issuer：`https://access.line.me`；authorize：`access.line.me/oauth2/v2.1/authorize`
   - token：`api.line.me/oauth2/v2.1/token`；JWKS：`api.line.me/oauth2/v2.1/certs`
   - ID token 簽章：**ES256**（非對稱）；`sub` 為 **pairwise**（每個 channel 一組）
   - claims 含 `name`、`picture`（Supabase 會映到 `user_metadata`，前端 `profile.ts` 直接吃得到）

4. **LIFF 執行環境限制：**
   - LIFF endpoint URL 必須是 **HTTPS**；`liff.init()` 只保證在 endpoint URL 及其**子路徑**有效。
   - 外部瀏覽器情境需 `withLoginOnExternalBrowser: true` 才會自動觸發 `liff.login()`。
   - `liff.getIDToken()` 需要 **`openid`** scope；要拿 **email** 需 `email` scope，而 email 權限
     **要向 LINE 申請審核**（需附畫面）。沒有 email 時，LINE 使用者可能是 null email。
   - 目前專案**沒有正式 HTTPS 部署**，`CORS_ORIGINS` 寫死 localhost（`config.py:41-44`）——
     這是 LIFF 的前置門檻。

5. **⚠️ 本專案專屬風險：LIFF 內建瀏覽器在 iOS 上 `getUserMedia` 有已知失效**
   （社群回報：LIFF 內呼叫 `navigator.mediaDevices.getUserMedia` 不回傳也不報錯，Safari 正常）。
   x-coach 的 **SixSeven / FruitNinja / 即時姿態** 都靠即時相機；若在 LINE 內失效，這些功能會壞。
   影片**檔案上傳**分析（主分析流程）多半不受影響，但仍須實測 `<input type=file capture>` 行為。
   → **必須先用真機（尤其 iOS）做相機 PoC**，再決定 LIFF 是否涵蓋即時相機功能。

---

## 4. 導入方案比較

### 方案 A — Supabase Custom OIDC（`custom:line`）〔建議優先〕

在 Supabase Dashboard 以 OIDC 自動探索加入 LINE（issuer `https://access.line.me`），前端新增一支
`signInWithLine()`，內容幾乎是 Google 那段的翻版。

- **後端 / DB / RLS / 管理員：零改動**（產物是真正的 Supabase session）。
- 前端改動極小：`auth.tsx` 加一個 action、`Login.tsx` 加一顆 LINE 按鈕與 SVG。
- Web 版 LINE Login 立刻可用；LIFF 內也可用同一顆 redirect（LINE 內建瀏覽器已登入 LINE，通常一鍵同意）。
- **未知點 / 需驗證**：LINE 的 OIDC 探索與 Supabase Custom OIDC 的相容性（pairwise sub、nonce、
  LINE 對 scope 格式的要求）需要一次接線驗證。custom OIDC 是較新功能，建議先開一個測試 channel 試通。
- **成本**：LINE Developers channel 設定 + Supabase Dashboard 設定 + 一次驗證。

### 方案 B — 後端橋接：驗 LINE token → 用 Admin API 換發 Supabase session〔LIFF 無縫登入最佳〕

適合「LIFF 內完全無登入畫面」：LIFF 拿 `liff.getIDToken()` → POST 到新端點 `/api/auth/line` →
後端驗 LINE ID token（LINE `verify` 端點或本地 JWKS 驗簽）→ 以 Supabase **Admin API**
find-or-create `auth.users`、再 mint session（`generateLink` magiclink → `verifyOtp` 取 access/refresh）
→ 回傳給前端 `supabase.auth.setSession(...)`。

- 之後 downstream 同樣**全部不變**（一樣是真的 Supabase JWT）。
- **代價**：後端得持有 **`SUPABASE_SERVICE_ROLE_KEY`**——這與現行「後端絕不用 service_role」的
  刻意安全姿態相衝突，需要謹慎隔離（只在這支端點用、不外洩、不進前端）。
- 新增：`/api/auth/line` 端點、LINE token 驗證、Admin 換發 session、對應測試。
- **好處**：LIFF 內體驗最順（無 redirect、無登入畫面）；也能自訂 email 缺失時的處理。

### 方案 C — Supabase Third-Party Auth（直接信任 LINE 的 JWT）〔不建議〕

把 LINE 設為受信任的第三方 issuer，前端直接拿 LINE token 當 Supabase bearer。

- 會**打斷**現有模型：`client.auth.get_user()` 不認 LINE token（需重寫 `_verify`）；RLS 的
  `auth.uid()` 變成 LINE 的 `sub` 而非 UUID；`user_roles` 對 `auth.users` 的外鍵、`admin_list_users()`
  讀 `auth.users` 都會失效（LINE 使用者不在 `auth.users`）。
- 對「這套已經長好的 codebase」而言改動面最大、風險最高，**不建議**。

### 對照表

| 面向 | A. Custom OIDC | B. 後端橋接 | C. Third-Party Auth |
|---|---|---|---|
| 後端改動 | 幾乎零 | 新增 1 端點 | 大（重寫驗證 + RLS 模型） |
| 需 service_role | 否 | **是** | 視作法 |
| 產生真 `auth.users` 列 | 是 | 是 | 否 |
| RLS / 管理員機制 | 不動 | 不動 | 破壞 |
| LIFF 內無縫登入 | 中（redirect 一次） | **最佳（無畫面）** | 佳 |
| Web 版 LINE Login | **佳** | 需另接 redirect | 佳 |
| 主要未知數 | LINE↔Supabase OIDC 相容性 | service_role 安全隔離 | 全面改寫 |
| 建議 | ✅ 首選 | ✅ 追加（要極致 LIFF 體驗時） | ❌ |

> 註：A 與 B 可並存——Web 用 A 的 redirect、LIFF 內用 B 的靜默換發，兩者都落到同一個 `auth.users`
> 使用者（以 LINE `sub` 對應），資料自然打通。

---

## 5. LIFF 執行層評估（App 化體驗）

**做得到的 App 化能力**
- 從 LINE **官方帳號 / 圖文選單**一鍵進入，全螢幕開啟，像原生 App。
- LIFF 內已登入 LINE ⇒ 可做到近乎無登入畫面（搭配方案 B 最順）。
- `liff.sendMessages()` 把分析結果 / 教練回饋推回聊天室；`shareTargetPicker` 分享；
  `liff.closeWindow()`、深連結、`liff.isInClient()` 分流。

**前置需求（目前缺）**
- 前端需**正式 HTTPS 部署**（LIFF endpoint）；後端 `CORS_ORIGINS` 要加上該正式來源。
- 加入 `@line/liff` SDK（`package.json` 目前沒有），App 啟動時 `liff.init({ liffId })`。
- 新增 `VITE_LIFF_ID`（沿用 `VITE_` 慣例）到 `frontend/.env.example`。
- SPA 用 `BrowserRouter`，client route 需落在 LIFF endpoint URL 子路徑下（相容，但要留意）。

**風險（務必先驗）**
- **即時相機（§3.5）**：iOS LIFF 內 `getUserMedia` 可能失效 → 即時相機遊戲/姿態在 LINE 內可能不可用。
  緩解：對即時相機功能改走 external browser / LINE MINI App，或在 LIFF 內偵測不到相機時降級提示。
- MediaPipe WASM/WebGL 在內建 WebView 效能較差，需實測。
- email 需 LINE 審核；未取得時的使用者建檔策略要先定。

---

## 6. 需要改動的清單（依方案）

**LINE Developers Console（A、B 共同）**
- 建 Provider 與 **LINE Login channel**；設定 callback / redirect URL（Supabase callback 或本站）。
- 需要 LIFF 時：在該 channel 下建 **LIFF app**（設定 endpoint URL、size、scope `openid`（＋`profile`／視需要 `email`）。

**Supabase**
- 方案 A：Dashboard → Auth → 加 Custom OIDC provider（issuer `https://access.line.me`）。
- 方案 B：專案設定取得 **service_role key**（僅後端用）；確認 Admin API 可用。

**前端（`frontend/`）**
- `package.json`：需 LIFF 時加 `@line/liff`。
- `frontend/.env.example`：加 `VITE_LIFF_ID`（LIFF）／視情況 LINE channel id。
- `src/lib/auth.tsx`：新增 `signInWithLine()`（A 為 `signInWithOAuth({provider:'custom:line'})`；
  B 為呼叫 `/api/auth/line` 後 `setSession`），掛進 `AuthValue` 與 `useMemo`。
- `src/pages/Login.tsx`：加 LINE 按鈕 + 品牌 SVG（比照 Google 段 `Login.tsx:222-230`）。
- （LIFF）新增啟動時 `liff.init` 與 `isInClient()` 分流；即時相機頁做降級處理。
- `src/pages/Settings.tsx`：provider 標籤加 `line`（可選）。
- 對應 vitest 測試（比照現有 `pages.Login.test.tsx`、`lib.auth.test.tsx`）。

**後端（`backend/`）**
- 方案 A：僅 `config.py` 的 `CORS_ORIGINS` 加正式/ LIFF 來源；其餘不動。
- 方案 B：新增 `/api/auth/line`（驗 LINE token + Admin 換發 session）、`settings.py` 加 LINE/
  service_role env、對應 pytest；務必守 95% coverage gate。

---

## 7. 風險與注意事項

- **相機（最高優先）**：先做 iOS 真機 PoC，再決定 LIFF 是否涵蓋即時相機。
- **service_role（方案 B）**：打破現行「零 service_role」姿態；只在單一端點使用、嚴禁進前端 bundle。
- **email 缺失**：LINE email 需審核；未取得時 `auth.users.email` 可能為 null，會影響
  `admin_list_users` 顯示與第一位管理員以 email seed 的邏輯（現以 `lucas60303@gmail.com` seed）。
- **Custom OIDC 相容性（方案 A）**：LINE 的 pairwise sub / nonce / scope 需一次接線驗證，勿假設即通。
- **部署缺口**：無正式 HTTPS 部署與 prod CORS，是 LIFF 的硬前提。
- **既有 demo/匿名模式**：LINE 登入是「加法」，需與 email/password、Google、匿名 demo 並存不衝突。

---

## 8. 建議的分階段落地計畫

1. **Phase 0 — 相機 PoC（0.5–1 天）**：最小 LIFF app 在 iOS/Android 真機測 `getUserMedia` 與檔案上傳。
   結果決定 LIFF 範圍（是否含即時相機）。
2. **Phase 1 — Web 版 LINE Login（方案 A，約 1 天＋設定）**：Supabase Custom OIDC 接 LINE，前端加按鈕。
   後端零改動。先讓「用 LINE 登入」在一般網頁可用並驗證通。
3. **Phase 2 — LIFF 外殼（視 Phase 0 結果，2–4 天）**：部署 HTTPS、加 `@line/liff`、`liff.init`、
   圖文選單入口、prod CORS；即時相機依 Phase 0 決定涵蓋或降級。
4. **Phase 3 — LIFF 內無縫登入（方案 B，選配，2–3 天）**：加 `/api/auth/line` 後端橋接與 service_role，
   讓 LINE 內完全免登入畫面；加測試補齊 coverage。

**工作量粗估**：只做 Phase 1（Web LINE Login）約 1 天可見成果；完整 LIFF App 化（Phase 0–3）
約 1 週上下，變數主要在相機相容性與正式部署。

---

## 9. 待你確認的問題

1. 這次目標是「多一個 LINE 登入選項」就好，還是要**完整的 LINE 內 App 化（LIFF）體驗**？
2. 前端有無**正式 HTTPS 部署**環境（LIFF 的硬前提）？目前 repo 只有 dev server + 寫死 localhost 的 CORS。
3. 是否需要在 LINE 內使用**即時相機**功能（SixSeven/FruitNinja/即時姿態）？這決定要不要為相機做特別處理。
4. 是否接受在**方案 B** 讓後端持有 `service_role key`（以換取 LIFF 內無縫登入）？或偏好維持零 service_role、
   接受方案 A 的一次 redirect？
5. 是否需要向 LINE 申請 **email 權限**（審核流程），或可接受先不取 email。
