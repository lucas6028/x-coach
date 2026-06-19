我來先了解一下你目前專案的實際狀態，這樣才能給出貼合 x-coach 的建議，而不是泛泛而談。

已經有 `frontend/`（React + Vite + TS + Tailwind）和 `backend/`（FastAPI）了。我看一下這兩個目前的實作程度，才知道是「從零設計」還是「在現有基礎上擴充」。

現在很清楚了 — 已經有完整的 FastAPI 後端 + React 前端，目前是 demo 等級。我看一下後端目前怎麼跑分析、怎麼存結果，這決定了登入/儲存/擴充要怎麼接。

我已經掌握現況了。先講結論：**骨架是對的，但目前是 demo 級，離「多使用者 + 歷史紀錄」有三個關鍵缺口**，下面逐項分析。

先載入繪圖規範，待會給你一張目標架構圖。

The guidance file is one giant JSON line. Let me extract the SVG boilerplate directly.

已經掌握全部現況。先給結論再逐項分析。

## 現況評估：骨架是對的，但有三個硬缺口

你已經有 **FastAPI 後端 + React/Vite/TS/Tailwind 前端**，方向完全正確、不需要重寫。但要支援「登入 + 歷史紀錄」，目前是 demo 級實作，有三個必須先補的洞（我從程式碼確認）：

1. **沒有資料庫、結果不落地** — [analyze.py](backend/app/routers/analyze.py) 算完直接回傳，不存；library 清單直接讀本機 `data/` 資料夾。要有歷史，就一定要有 DB。
2. **分析是同步阻塞** — `analyze` 是 `async def`，卻在裡面直接呼叫 `process_video()`（MediaPipe，CPU 密集）。這會卡死整個 event loop，**只要 2 人同時上傳就互卡**，長影片直接 timeout。
3. **影片整包進記憶體 + 存本機磁碟** — `await file.read()` 把整支影片載進 RAM，寫到 repo 下的 `data/runtime/`。多人/多機一定 OOM 且檔案不共享。

下面對你五個問題逐一給建議與取捨。

---

### 1. 框架與技術
- **後端維持 FastAPI** — 整條 ML pipeline 都是 Python，沒有更好的選擇。
- **認證（這是新的關鍵決定）**，兩條路：
  - **託管式（推薦給研究原型）**：**Supabase**（Postgres + Auth + Storage + Row-Level Security 一次到位、免費額度大、仍是純 SQL）或 Clerk / Auth0。把密碼雜湊、信箱驗證、OAuth、重設密碼這些「容易出安全漏洞」的工作外包出去。
  - **自管（純 Python）**：`fastapi-users` + JWT，註冊/登入/OAuth/重設密碼開箱即用。想完全不依賴外部服務時選它。
  - **Token 策略**：短效 access JWT + refresh token，放 **httpOnly cookie**（不要 localStorage，防 XSS）。
- **前端要補**：路由（React Router）、Auth context + 受保護路由、「我的紀錄」儀表板頁；把手刻的 [api.ts](frontend/src/api.ts) 換成 **TanStack Query**（分析是非同步任務，需要快取 + 輪詢狀態）。

### 2. 資料儲存（多型儲存，分工明確）
核心原則：**影片不要進資料庫，分析結果不要丟掉。**
- **PostgreSQL**（結構化、唯一真實來源）：`users`、`videos`（含 `storage_key`、`status`: pending/processing/done/failed）、`analyses`。分析結果是巢狀文件 → 用一個 `analyses.result JSONB` 存整包，再把 `view_type`、`fault_count` 提升成欄位方便列表/篩選（GIN index 可查進 JSONB）。日後 RAG 要換真 embedding 也能用 **pgvector** 併進同一個 DB。
- **物件儲存**（重二進位）：原始影片、pose JSON、縮圖/疊圖。推薦 **Cloudflare R2**（影片 = 大量 egress，R2 egress 免費）；S3 / GCS / Supabase Storage 亦可。
  - **關鍵模式：presigned URL 直傳** — 前端跟後端要簽名 URL → 影片**直接傳到物件儲存（不經過 FastAPI）** → 再通知後端。這同時解掉現在「整包讀進 RAM」的問題。
- **Redis**：任務佇列 broker + 狀態快取 + rate-limit。
- RAG/KG 的離線 `HashEmbeddingBackend` + graphml 維持唯讀、隨 app 出貨即可，每個 worker 載一次。

### 3. 部署（把 Web 層和 ML 運算層拆開 — 最重要的一步）
- **Web/API 層**（無狀態 FastAPI）：認證、CRUD、發簽名 URL、派工、回結果。水平擴展、便宜。
- **Worker 層**（ML pipeline）：從佇列拿任務，跑 pose + 規則 + RAG，結果寫回 Postgres + 物件儲存。MediaPipe 是 CPU；未來 VideoMAE 是 GPU。
- 中間用 **Celery + Redis**（ML 業界標準）。
- **階段一（最便宜、最快上線）**：Docker 打包，丟 **Railway / Render / Fly.io**（同一 repo 開 web + worker + 託管 Redis + 託管 Postgres），前端放 Vercel / Cloudflare Pages，影片放 R2。約 **$20–50/月**就有可用的多人 + 歷史服務。
- **GPU 不要 24h 常開**（貴又閒置）：用 **serverless GPU**（**Modal** 最適合，Python 原生、可縮到 0；或 Replicate / RunPod），Celery worker 需要時才呼叫它跑 VideoMAE。
- **階段二（規模化）**：容器移到 ECS/GKE，GPU worker 獨立 autoscaling node pool，用 KEDA 依佇列長度擴縮。先別從這開始。
- ⚠️ 要先把 `src/` / backend 那套「相對 repo root + 讀寫本機 `data/`」改成 **env 設定 + 儲存抽象層**（dev 用本機、prod 用 R2/S3），否則無法容器化多副本跑。

### 4. 高流量
- **把重運算移出 request（第一優先）**：上傳 → 立刻回 job id → 前端輪詢 `GET /api/jobs/{id}` 或 SSE/WebSocket → 完成取結果。這招同時修掉現在卡 event loop 的 bug，也是「2 人就掛」與「扛得住尖峰」的分水嶺。
- 無狀態 Web 層水平擴展（JWT/Redis，無機器黏著）。
- Worker 依**佇列長度**自動擴縮；CPU/GPU 分池；每 worker 限併發（ML 吃記憶體，模型開機載一次、不要每次請求載）。
- 影片走 **CDN + 簽名 URL**，永不經過 FastAPI。
- DB：**連線池**（PgBouncer / 平台 pooler — 多 worker × 連線數很快打爆 Postgres）；`(user_id, created_at)` 加索引；必要時讀副本。
- 快取常用讀取（使用者最近紀錄、library 清單）。
- **每人 rate limit + 上傳配額**（分析很貴，要保護）；**去重**：對影片做 hash，同人同片直接回快取結果、不重算。
- 真正的瓶頸是 **成本（GPU 秒數）**，不是 RPS。scale-to-zero + 結果快取 + 去重才是讓帳單可控的關鍵。

### 5. 其他技術問題
- **隱私 / 個資（最容易低估，但很重要）**：這是可辨識人體/臉部的影片，屬敏感個資（台灣 PDPA / GDPR）。
  - 傳輸 TLS、靜態加密（物件儲存 SSE）；每個 video/analysis 綁 `user_id` 並在**每個 endpoint 強制檢查**（Supabase 可用 RLS）；影片一律用**有時效的簽名 URL**，bucket 不可公開。
  - 同意條款 + 保存期限 + 刪帳號要連物件儲存的影片一起清；log 不要記影片內容或簽名 URL。
- **檔案驗證/正規化**：現在只信副檔名 → 要驗真實 MIME/codec、限大小/長度，先用 ffmpeg 轉統一格式 + 修正方向再處理。
- **長任務狀態機**：queued/processing/done/failed、重試退避、毒任務 dead-letter、timeout；失敗清楚回報（如「未偵測到人」）。
- **可重現性**：每筆分析存下 pipeline 版本與當時的規則閾值（你的 detector 閾值可調），規則改了還能重跑。
- **可觀測性**：結構化 log、Sentry、佇列長度/任務延遲/GPU 使用率指標、health/readiness probe。
- **Migration / 設定**：一開始就上 **Alembic**；設定走 **pydantic-settings + env**，金鑰不進 repo。

---

### 建議優先順序（務實路線）
- **P0｜止血**：修同步阻塞 — 至少先把 `analyze` 包進 `run_in_executor` / 背景任務。
- **P1｜核心功能**：Postgres + 認證 + `videos`/`analyses` 表 + 「我的紀錄」頁（先單機、本機存檔也行）。
- **P2｜非同步化**：物件儲存 + presigned 直傳 + Celery/Redis 佇列 + job 狀態輪詢。
- **P3｜規模化**：拆 web/worker 部署、CDN、GPU serverless、rate limit/配額、可觀測性。

一句話總結:**留著 FastAPI + React,新增的是「認證 + Postgres(metadata) + 物件儲存(影片) + 非同步佇列」這四件事**,而最該先動的不是登入,是把分析從同步阻塞改成背景任務(P0),否則加了使用者反而更快被一兩個並發上傳拖垮。

要我接著動手哪一塊?常見的起手式:
- **設計資料表 schema**(`users` / `videos` / `analyses`,含 JSONB 與索引)— 歷史紀錄的地基
- **P0 非阻塞改造**(把 `analyze` 包進 executor / 背景任務)— 最小改動、立刻止血
- **認證選型 + 落地**(Supabase vs `fastapi-users`,含前端受保護路由)

或你想先把整套寫成一份正式的技術規劃文件也可以。