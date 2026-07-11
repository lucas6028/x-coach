# TODO

> **2026-07-07 全面盤點**：已對照 repo 實際程式碼逐項核實（非憑印象）。
> 現況：規則式深蹲教練 + GraphRAG + grounded LLM chat 已全部上線；
> VideoMAE / pose 分類器完成 research baseline（未接入 app）；
> contrastive learning、fusion、3D 路由尚未開始。
> 總路線圖見文末「Pipeline 與 Agent Harness 路線圖」，
> 設計細節見 `docs/ai-coach-pipeline-and-agent-harness.md`。

## 目標定義

- [x] 明確定義目前 `Squat` 任務主軸為（`研究計畫.md` Phase 1 已載明）：
  - `error detection`
  - `error classification`
  - `temporal localization`
  - `contrastive representation learning`
- [ ] 避免將目前資料描述成完整 `AQA score regression`（計畫書措辭需再檢查一輪）
- [x] 將計畫書中的敘述微調為（`研究計畫.md` 已含）：
  - 使用預訓練 `VideoMAE V2` 作為時空特徵提取器
  - 搭配對比學習強化正常動作與錯誤動作的表徵可分性
  - 融合 `MediaPipe` 幾何特徵提升可解釋性

## 資料集現況整理

> `notes/dataset-summary.md` 只有質性描述；下列數字尚未正式記錄進 notes/docs。

- [ ] 確認已可直接使用的標註與切分：
  - `data/Squat/Labeled_Dataset/Splits/train_keys.json`
  - `data/Squat/Labeled_Dataset/Splits/val_keys.json`
  - `data/Squat/Labeled_Dataset/Splits/test_keys.json`
- [ ] 記錄目前資料規模：
  - `Labeled_Dataset/videos`: 1739 支影片
  - split union: 1623 支有正式 train/val/test key
  - `Unlabeled_Dataset/videos`: 4970 支影片
- [ ] 記錄標註類型：
  - `error_knees_forward.json`: 影片 key 對應錯誤時間區段
  - `error_knees_inward.json`: 影片 key 對應錯誤時間區段
  - `labels_shallow_depth.json`: 片段/局部 label
- [ ] 記錄標註格式：
  - `[video_id]: [error_start_time, error_end_time]`
  - 無錯誤時為空陣列
- [ ] 記錄影片基本特性：
  - 多數影片約 `30 FPS`
  - 長度約 `3 秒`
  - 解析度約 `480x600`

## 資料可行性結論

- [ ] 在文件中明確寫出：
  - 目前資料集可以進行訓練、驗證、測試
  - 可做 `VideoMAE V2` 特徵提取
  - 可做自監督 / 弱監督對比學習
  - 可做監督式錯誤辨識與片段定位
- [ ] 在文件中補充限制：
  - 沒有完整連續品質分數
  - 不適合直接做傳統 `AQA regression`
  - 類別不平衡，特別是 `knees_inward`
  - 標註粒度不完全一致

## 模型設計

### Stage 1: VideoMAE V2 特徵提取 — ✅ 大致完成（research，見 `notes/videomae_classifier_experiment_summary.md`）

- [x] 決定使用預訓練 `VideoMAE V2` 作為 backbone
- [x] 決定先採用：
  - frozen backbone（特徵提取路線，見 `src/video/`）
- [ ] 決定 clip 長度（`16` vs `32 frames`）— 實驗設定需回寫成正式決策
- [x] 決定特徵輸出方式：CLS token（rehab24 分支同款）

### Stage 2: Contrastive Learning — ⏸ 未開始（2026-07-07 盤點：repo 無任何 contrastive 實作）

- [ ] 使用 `Unlabeled_Dataset/videos` 做 domain adaptation
- [ ] 設計正樣本對：
  - 同一影片不同 temporal crop
  - 同一影片不同 augmentation
- [ ] 設計負樣本對：
  - 不同影片 clip
  - 不同錯誤型態 clip
- [ ] 評估 loss 選項：
  - `InfoNCE`
  - `Supervised Contrastive Loss`
  - `classification loss + contrastive loss`

### Stage 3: Supervised Downstream Task — ✅ 完成（video-level，`src/video/video_level_error_classification.py`）

- [x] 建立影片級 / clip 級任務：
  - `knees_forward`
  - `knees_inward`
  - `shallow_depth`
- [x] 決定任務形式：binary classification（video-level；temporal localization 未做，見 Stage 3.5）
- [x] 建立 classification head
- [x] 建立 threshold tuning 流程（`find_best_threshold()` 在 val set 上掃描，非固定 0.5）
- [ ] （新增）Stage 3.5：temporal localization —— VideoMAE 路線尚無 segment IoU / frame-level 定位；目前時間定位只靠規則偵測器

### Stage 4: Pose / Geometry Fusion — 🔶 一半（pose-only baseline 已做，fusion 未跑）

- [x] 保留 `MediaPipe` 或 pose estimation 模組（production 規則偵測器即是）
- [x] 計算幾何特徵（規則偵測器已算）：
  - 膝角
  - 髖角
  - 軀幹傾角
  - 膝蓋與腳尖相對偏移
- [ ] 設計融合方式（`notes/pose_only_classifier_experiment_summary.md` 明列為 next step，未執行）：
  - late fusion
  - feature concatenation + MLP
- [x] 將幾何特徵用於可解釋輸出與結果驗證（FaultCard 的 evidence 欄位）

## 即時性規劃 — ⏸ 未開始（設計文件 §2.4 已納入「即時 webcam 模式」為遠期項）

- [ ] 明確區分兩種系統模式：
  - 離線分析模式
  - 準即時回饋模式
- [ ] 在文件中說明 `VideoMAE` 不適合逐幀超低延遲即時回饋
- [ ] 若要做準即時，考慮：
  - 小型 backbone
  - 較短 clip
  - 較低解析度
  - sliding window inference
- [ ] 系統角色分工：
  - `MediaPipe` 負責即時幾何警示
  - `VideoMAE` 負責高品質時空判斷

## 如何辨識 VideoMAE 的結果

- [x] 明確區分 `VideoMAE` 原始輸出是特徵向量，不是人類可讀文字
- [x] 決定下游辨識方式：接 classification head（已實作）
- [ ] 滑動視窗做時間片段辨識（未做，同 Stage 3.5）
- [x] 建立輸出格式（由規則 pipeline 產出，VideoMAE 路線尚未接入）：
  - 錯誤類型
  - 發生時間點
  - 信心分數
  - 對應建議
- [ ] 加入特徵空間視覺化（未做）：
  - `t-SNE`
  - `UMAP`

## 驗證指標

### 影片級二元分類 — ✅ 已在 baseline 實驗回報

- [x] 計算：
  - `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
  - `Precision = TP / (TP + FP)`
  - `Recall = TP / (TP + FN)`
  - `F1 = 2 * Precision * Recall / (Precision + Recall)`

### 多類別 / 多標籤

- [ ] 回報：
  - `Accuracy`
  - `Macro F1`
  - `Weighted F1`
- [ ] 若為多標籤任務，優先觀察：
  - per-class F1
  - `micro F1`
  - `macro F1`

### 時間片段定位 — ⏸ 未開始（VideoMAE 路線）

- [ ] 若做 temporal localization，計算：
  - `segment IoU`
  - frame-level `Precision / Recall / F1`
  - 視情況加入 `mAP@IoU`

### 類別不平衡處理

- [ ] 不只看 `Accuracy`
- [ ] 對 `knees_inward` 額外回報：
  - `Precision`
  - `Recall`
  - `F1`
  - `PR-AUC` 或 `ROC-AUC`
- [x] 在 validation set 上調整 threshold，而不是固定 `0.5`

## 實驗設計

- [ ] 設計 baseline：
  - [x] only `VideoMAE`（`notes/videomae_classifier_experiment_summary.md`）
  - [x] only pose / geometry（`notes/pose_only_classifier_experiment_summary.md`，已互相對照）
  - [ ] `VideoMAE + contrastive`
  - [ ] `VideoMAE + pose`
  - [ ] `VideoMAE + pose + contrastive`
- [ ] 設計 ablation study：
  - 是否使用 unlabeled pretraining
  - 是否使用 supervised contrastive loss
  - 不同 clip 長度
  - 不同 fusion 方法
- [ ] 規劃最終 test set 僅用於最後一次評估

## 最終成果形式 — ✅ v1 完成（規則式；VideoMAE 版尚未接入 app）

- [x] 完成一個可輸入深蹲影片的模型（規則偵測器 5 faults + KG/RAG）
- [x] 輸出內容至少包含：
  - 錯誤類型
  - 錯誤發生時間區段
  - 信心分數
  - 結構化回饋
- [x] 規劃 demo 介面（Studio 已上線）：
  - 左側影片 / 骨架顯示
  - 時間軸錯誤標記
  - 右側顯示觀察 / 原因 / 建議
- [x] 結合 RAG 產出（FaultCard cause→risk→fix 因果階梯 + grounded chat）：
  - 觀察
  - 生物力學原因
  - 可執行糾正建議

## 建議先做的最小可行版本

- [x] 第一步先完成 `Squat` 的 `knees_inward` 二元分類
- [x] 第二步加入 `knees_forward`
- [x] 第三步加入 `shallow_depth`
- [ ] 第四步加入 contrastive learning
- [ ] 第五步加入 pose fusion
- [x] 第六步再考慮 RAG 與教練式文字回饋（GraphRAG + SSE chat + followup chips 已上線）

## 系統與部署：使用者登入 + 歷史紀錄

> 把現有 demo（`backend/` FastAPI + `frontend/` React）做成多使用者、能保存影片與分析歷史的服務。
> 分階段推進：先止血，再加功能，最後規模化。

### P0：止血（同步阻塞）— ✅ 已完成 2026-06-19

- [x] `/api/analyze` 由同步阻塞改為非阻塞：阻塞 pipeline 丟到 worker thread（`run_in_threadpool`），event loop 不再被單一分析卡死
- [x] `asyncio.Semaphore` 限制同時分析數，避免併發上傳打爆 CPU/RAM
- [x] 新增 `MAX_CONCURRENT_ANALYSES` 環境變數（`config.py`，預設 2，per-process）
- [x] 排隊前 `del data` 釋放影片 buffer，避免 semaphore 變成記憶體放大器
- [x] `analysis.py` 延後載入 `src.pose`（MediaPipe/torch）→ web 啟動不載 ML、API 層可在無 ML 環境測試
- [x] 新增 `tests/test_analyze_endpoint.py`（契約不變 / 跑在 worker thread / 併發有上限）；本機 6 passed

### P1：核心功能（登入 + 歷史地基）— ✅ 大致完成（Supabase 路線）

- [x] 認證選型並落地：**Supabase Auth**（取代自管 fastapi-users）
  - [ ] ~~access JWT + refresh token 放 httpOnly cookie~~ → 現況為 supabase-js 預設 localStorage（`frontend/src/lib/supabase.ts`）；換 httpOnly cookie 需自訂 storage，列為後續強化
- [x] PostgreSQL schema（Supabase migration，取代 Alembic）：
  - [x] `users`（Supabase auth 內建）
  - [x] `videos`（`storage_key`、`status`，upsert on user_id+video_id）
  - [x] `analyses`（`result JSONB` 整包 + 提升 `view_type`/`fault_count` 為欄位）
  - [x] `conversations`（chat 訊息 JSONB + followups，計畫外新增）
- [x] 分析結果落地：`/api/analyze` 算完寫入 DB（登入者 best-effort persist）
- [x] 前端：React Router + Auth context + 受保護路由 +「我的紀錄」（History 頁）
- [ ] 前端資料層改用 TanStack Query（未採用；現況 supabase-js + 自製 fetch，運作正常，視痛點再決定）
- [x] 設定改用 pydantic-settings + env（`backend/app/settings.py`）

### P2：非同步化 + 儲存 — ⏸ 未開始（= 新設計文件的 P0「非同步分析佇列」，優先級提高）

- [ ] 物件儲存（建議 Cloudflare R2）：原始影片、pose JSON、縮圖
- [ ] presigned URL 直傳：影片不經過 FastAPI（解掉 `await file.read()` 整支進 RAM）
- [ ] Celery + Redis 佇列：上傳→回 job id→worker 處理→輪詢/SSE 取結果
- [ ] job 狀態機：queued/processing/done/failed、重試退避、dead-letter、timeout
- [ ] 去重：影片 hash，同人同片回快取、不重算

### P3：規模化 + 維運 — ⏸ 未開始

- [ ] 拆 web / worker 部署（Docker；先 Railway/Render/Fly.io，之後 ECS/GKE）
- [ ] GPU 用 serverless（Modal/Replicate/RunPod，可縮到 0），VideoMAE / direct-3D 在此跑
- [ ] worker 依佇列長度自動擴縮（KEDA）；CPU/GPU 分池
- [ ] CDN + 簽名 URL 提供影片
- [ ] DB 連線池（PgBouncer）、`(user_id, created_at)` 索引、必要時讀副本
- [ ] 每人 rate limit + 上傳配額
- [ ] 可觀測性：結構化 log、Sentry、佇列/延遲/GPU 指標、health/readiness probe

### 橫切議題（越早處理越省事）

- [ ] 隱私 / 個資（PDPA・GDPR）：影片屬敏感個資 — 靜態加密、每筆綁 `user_id`、簽名 URL、刪帳號連物件儲存一起清、log 不記影片內容/URL（帳號刪除目前是 Settings 頁 stub，未接線）
- [ ] 檔案驗證：驗真實 MIME/codec（別只信副檔名）、限大小/長度、ffmpeg 正規化方向與格式
- [x] 可重現性（一半）：每筆分析已存 `pipeline_version`
  - [ ] 補存當時規則閾值 snapshot（detector 閾值可調）
- [ ] 重構 repo-root / `sys.path` 耦合 → 儲存抽象層（dev 用本機、prod 用 R2/S3）

## Demo

- [ ] Line ChatBot
- [ ] QR Code demo, real time interaction
- [ ] 語音回饋（composer 已預留 UI 槽位，功能未做）
- [ ] 動作偵測、分類（= 多動作 movement ID，見路線圖 P2）
- [x] LLM follow up questions (options)（followup chips 已上線，pinned 快速模型）
- [ ] 健身菜單客製化，可用 LLM 進行修改（= 路線圖 P2 的 `make_drill_plan` 工具）

## Pipeline 與 Agent Harness 路線圖（2026-07-06 設計，詳見 `docs/ai-coach-pipeline-and-agent-harness.md`）

> 產品主線改依此路線推進；上面各節屬研究支線或被此路線圖涵蓋。

### P0：地基

- [ ] KG 切換 `squat_kg_v2` → `sports_kg_v3` + movement-aware 檢索參數（backend `config.py` 改指向 + 回歸測試）
- [ ] Rep 切分 + per-rep metrics（膝/髖角度極值切 rep；決定 frame_metrics 保留策略——建議存 per-rep 摘要）
- [ ] 非同步分析佇列（= 上方系統 P2 的 Celery/Redis 項）

### P1：Agent 最小可用

- [ ] chat 升級為 tool-calling 迴圈（OpenRouter function calling；先 3 個工具：`get_analysis` / `kg_query` / `rag_search`）
- [ ] Critic-lite：回答送出前做 grounding 檢查（FAIL → 重生成一次 → 降級為 FaultCard 模板直出）
- [ ] SSE 加 `tool_start` / `tool_result` 事件，前端顯示「正在查知識圖譜…」
- [ ] 工具呼叫 trace 存進 conversations JSONB（可重播、可稽核）

### P2：多動作 + 記憶

- [ ] Lunge rule pack（RAG 文件已備；KG 抽取待跑——需 API key）
- [ ] `compare_analyses` / `list_user_history` 工具 + 進步追蹤（跨分析記憶）
- [ ] Drill library + `make_drill_plan` 工具（fault → KG `CORRECTED_BY` → 矯正課表）
- [ ] 動作識別（movement ID）輕量分類器，自動載入對應 rule pack
- [ ] 各個動作，相機擺放最適合的位置、角度

### P3：感知升級

- [ ] 深度類 cue 的 direct-3D 路由（NLF 類模型，GPU；先做「深入分析」按鈕的非同步重分析）
- [ ] RAG 換神經 embedding（hash-BoW → sentence-transformer 類本地模型，介面不變）
- [ ] VideoMAE 融合分類器接入 app（補規則抓不到的「順不順」缺陷）

### 評估迴路（隨 P1 起步）

- [ ] golden-set 規則回歸（已標註影片，CI 跑 rule pack diff）
- [ ] RAGAS-style faithfulness 離線抽樣評估
- [ ] Critic grounding score 線上入庫
