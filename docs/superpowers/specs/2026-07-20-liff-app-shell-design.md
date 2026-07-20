# LIFF App Shell — 設計

Status: **設計已核准，待實作** · Created 2026-07-20

在現有 SPA 上加一層「LIFF 感知的外殼」，讓使用者在 LINE App 內開啟時，UI 看起來、操作
起來像原生 app：底部 tab bar 取代桌面 sidebar 與行銷式 header，網頁專屬 chrome 隱藏，
全螢幕 + safe-area。前置背景見 `docs/line-login-liff-evaluation.md`（§5 App 化能力、
§8 Phase 2）與 `docs/line-login-liff-setup.md`。

---

## 1. 範圍

**做（本 spec）** — 外殼本身：

- `useLiffContext()` 情境偵測 hook
- `LiffAppShell` 元件：極簡標題列 + 底部 tab bar + 100dvh + safe-area
- 隱藏網頁專屬 chrome（Landing、登入按鈕、Header/Sidebar）
- LIFF 進入點落在 `/app`
- 遊戲的相機**預先**降級提示

**不做（另開 spec）** — LIFF 原生能力：`liff.sendMessages()`、`shareTargetPicker`、
`liff.closeWindow()`、rich menu 深連結。這些在外殼存在後都很薄，且需要先在 LINE console
開 `chat_message.write` scope 與 share target picker，屬於外部設定而非程式碼。

**已經做完、本 spec 不重做：**

- 相機降級邏輯 — `lib/camera.ts` 的 `CameraError` / timeout wrapper 已存在，且
  `SixSeven.tsx:255` 與 `FruitNinja.tsx:330` 皆已 catch 並在 LIFF 內顯示
  `camera.liffHint`。本 spec 只加「進入前的預先提示」文案。
- `viewport-fit=cover` — `frontend/index.html:5` 已設定，`env(safe-area-inset-*)` 因此
  真的會回傳非零值（沒有這行的話 inset 工作會是靜默 no-op）。
- LIFF 內靜默登入（評估文件方案 B）— `lib/auth.tsx` 的 auto-login effect 已上線，所以
  「LIFF 內隱藏登入按鈕」純粹是 UI 決定，不需要動認證。
- `h-[100dvh]` — `AppLayout` 與 `LiffDiag` 都已使用，沿用同一寫法。

---

## 2. 架構：姊妹外殼，不是分支

`AppLayout` 頂端做**一次**分流：

```
AppLayout(props)
  ├─ isInClient → <LiffAppShell title=…>{children}</LiffAppShell>
  └─ 否則       → 現有的 Header + Sidebar + main（一行不動）
```

**每個頁面（App / History / Games / Settings / Movements）都不用改** —— 它們已經全部
經由 `AppLayout` 渲染，分流發生在它們看不到的地方。

考慮過但否決的替代方案：

- **在 `AppLayout` 內部加 in-client 分支** —— `AppLayout` 已經同時承載 desktop rail 與
  mobile drawer 兩套 layout，再疊第三套會讓它失焦。
- **在 `main.tsx` 用不同 route element 包** —— 要改 12 條 route，且 games 那類自行傳
  `initialSidebarOpen` 的頁面會斷掉。

---

## 3. 元件

### 3.1 `frontend/src/lib/liffContext.tsx`

```ts
interface LiffContextValue {
  ready: boolean;      // 真實的 isInClient() 是否已解析完成
  isInClient: boolean; // 目前採用的值（ready 前是樂觀猜測）
}
export function useLiffContext(): LiffContextValue
export function LiffProvider({ children }: { children: ReactNode }): JSX.Element
```

**為什麼需要 provider 而不是直接呼叫 `isInLiffClient()`：** `lib/liff.ts` 的
`isInLiffClient()` 是 **async**（它 await `initLiff()`），layout 沒辦法同步讀。Provider
解析一次、全 app 共用。

**Pending 視窗處理（設計核心）：** 從 LINE redirect 回來時 init 要 ~1–1.5s。若這段期間
預設渲染網頁 chrome，使用者會看到錯誤外殼一閃。採**同步樂觀猜測**：

```ts
function guessInClient(): boolean {
  if (!isLiffConfigured()) return false;
  const ua = navigator.userAgent || "";
  if (/\bLine\//i.test(ua)) return true;
  const q = window.location.search;
  return q.includes("liff.state") || q.includes("liff-referrer");
}
```

初始 state = `guessInClient()`；effect 內 `await initLiff()` 取得 `liff.isInClient()` 的
真值後覆寫並 `setReady(true)`。猜錯時外殼會換一次（罕見且無害）；猜對時（幾乎總是）零閃爍、
零白畫面。

**必須沿用既有的快取 promise。** `main.tsx` 已在 bootstrap 呼叫 `void initLiff()`，
`initLiff()` 內部 memoize 了 `liffPromise`。Provider 呼叫同一支函式，因此不會第二次 init。

**未設定時的行為：** `VITE_LIFF_ID` 未設 → `{ ready: true, isInClient: false }`，與
`lib/liff.ts` 現有的「沒設就 no-op、退回純網頁」姿態一致。init 失敗（`initLiff()` 回
`null`）同樣落到 `isInClient: false`。

**掛載位置：** `main.tsx` 的 `AuthProvider` **外層**（`LiffProvider > AuthProvider >
I18nProvider > Routes`）。理由：Landing 與 Login 的分流都需要讀它，而它們不在 auth 之下。

### 3.2 `frontend/src/components/LiffAppShell.tsx`

```
<div class="min-h-[100dvh] flex flex-col bg-background-dark text-content">
  <header>  頁名（title prop），無品牌行銷、無登入按鈕、無 sidebar 觸發  </header>
  <main class="flex-1 min-h-0 overflow-y-auto">{children}</main>
  <nav>     底部 tab bar（fixed 於底、safe-area 內縮）                  </nav>
</div>
```

四個 tab（沿用現有 i18n key 與 Sidebar 的 phosphor icon）：

| Tab | 路由 | Active 判定 |
|---|---|---|
| 分析 | `/app` | `pathname === "/app"` |
| 歷史 | `/history` | `pathname === "/history"` |
| 遊戲 | `/games` | `/games` \| `/67` \| `/ninja`（比照 `Sidebar.tsx:44`）|
| 設定 | `/settings` | `pathname === "/settings"` |

**標題列文字：** `AppLayout` 的 `title` prop 直接沿用（History / Games / Settings /
Movements / 兩個遊戲頁都已經在傳）。只有 `App.tsx` 傳的是 `analysis` 而非 `title` ——
in-client 時標題列顯示分析 tab 的標籤（`nav.newAnalysis` 之類的既有 key），不顯示 web 版
Header 的 session 狀態 pill。

**遊戲頁：** `SixSeven` / `FruitNinja` 也走 `AppLayout`，因此在 LINE 內同樣帶著 tab bar
—— 這是刻意的，離開遊戲不需要先找返回鍵。它們傳的 `initialSidebarOpen` 在 app 外殼下被
忽略（無 sidebar 可言）。

**Safe area：** tab bar 用 `pb-[env(safe-area-inset-bottom)]`；`main` 補一段等同
「tab bar 高度 + inset」的下留白，避免內容被蓋住。

**Movements 與 Admin：** 不佔 tab 位，但路由照常可達 —— 僅限 rich menu 或在 app 外開啟的
直接連結；in-client 時 app 內沒有連到這兩個路由的連結（Sidebar 的對應連結只在網頁版渲染）。
這是刻意的：四個 tab 是拇指可及的主要動線，不是完整的網站地圖。

### 3.3 隱藏網頁專屬 chrome

- **Header / Sidebar** —— 由 §2 的分流自然消失，不需要額外程式碼。
- **Landing** —— `Landing` 頂端 `if (isInClient) return <Navigate to="/app" replace />`。
  LIFF endpoint URL 設定在網站根目錄（見 §7、`docs/line-login-liff-setup.md` §9），所以
  一般啟動每次都會先落在這裡；這行不是保險用的邊角處理，而是每次 in-client 啟動都要走的
  導向路徑。
- **Login** —— in-client 時同樣 `<Navigate to="/app" replace />`。靜默登入已在背景跑，
  再顯示登入按鈕只會讓人以為沒登入。
- **語言 / 主題切換** —— in-client 時 Header 不存在，所以本來就沒地方放；移進 `Settings`
  頁（web 版的 Header 保留現狀不動）。

### 3.4 相機預先提示

`Games` 頁在 `isInClient` 為真時，於遊戲卡片上顯示一行提示：在 LINE 內相機可能無法使用，
可用 LINE 右上選單的「用其他瀏覽器開啟」。純文案（新增一個 i18n key），**不改任何相機
邏輯** —— 失敗後的降級路徑已經存在且已測試。

---

## 4. 錯誤處理

單一原則，與 `lib/liff.ts` 現有姿態一致：**任何 LIFF 側的失敗都退回「純網頁」**。
`initLiff()` 已經 catch 了 init 失敗並回 `null`；provider 把 `null` 讀成
`isInClient: false`，於是 app 渲染既有的 web 外殼。沒有新的失敗模式，也沒有 LIFF 專屬的
錯誤畫面。

---

## 5. 測試（vitest，`frontend/src/test/`）

- `lib.liffContext.test.tsx` — 樂觀猜測（UA / `liff.state`）、非同步修正、`VITE_LIFF_ID`
  未設、`initLiff()` 回 `null`。用既有的 `_resetLiffForTests()` 隔離。
- `components.LiffAppShell.test.tsx` — 四個 tab 都在、依 pathname 高亮（含 `/67` →
  遊戲）、safe-area class 存在。
- `components.AppLayout.liff.test.tsx` — in-client 時不渲染 Sidebar / Header；web 時
  渲染既有外殼（回歸保護）。
- `pages.Games.liffHint.test.tsx` — in-client 顯示相機提示，web 不顯示。

Landing / Login 的 in-client 導向可併入既有的 page 測試檔。CI 的前端關卡是
`yarn test:coverage`（cwd = `frontend/`）。

---

## 6. 檔案清單

**新增**

- `frontend/src/lib/liffContext.tsx`
- `frontend/src/components/LiffAppShell.tsx`
- 上述四個測試檔

**修改**

- `frontend/src/main.tsx` — 掛 `LiffProvider`
- `frontend/src/components/AppLayout.tsx` — 頂端分流（~4 行）
- `frontend/src/landing/Landing.tsx` — in-client 導向
- `frontend/src/pages/Login.tsx` — in-client 導向
- `frontend/src/pages/Settings.tsx` — 收納語言 / 主題切換
- `frontend/src/pages/Games.tsx` — 相機預先提示
- i18n 檔 — 四個 tab 標籤 + 相機提示（zh / en 皆需）

**不動：** `lib/liff.ts`、`lib/auth.tsx`、`lib/camera.ts`、`SixSeven.tsx`、
`FruitNinja.tsx`、任何後端程式碼。

---

## 7. 外部設定（非程式碼，實作後需手動完成）

- LINE Developers console：LIFF app 的 endpoint URL 指向**網站根目錄** `https://<host>/`，
  size 設 `Full`。

  > 2026-07-20 修正：本節原本寫 `/app`，是錯的。LIFF 的深連結是把額外路徑接在 endpoint
  > 的**完整路徑**後面，所以 endpoint 設 `/app` 時 `liff.line.me/{id}/history` 會解析成
  > `/app/history` —— 這條路由不存在（`main.tsx` 是扁平路由、沒有 catch-all），圖文選單
  > 會壞掉三個入口。改用根目錄後四個分頁都能深連結，而 §3.3 的「in-client 的 `/` 轉
  > `/app`」剛好讓一般啟動仍然落在工作室。權威說明見
  > `docs/line-login-liff-setup.md` §9。
- 本 spec 不需要新的 scope；`sendMessages` / share target picker 所需的 scope 屬於後續
  的原生能力 spec。
