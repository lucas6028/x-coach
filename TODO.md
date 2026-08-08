# TODO

> **2026-08-08 全面盤點**：對照 repo 實際程式碼與已合併 PR 逐項核實（前次盤點 2026-07-07，
> 之間 549 個 commit）。每個被翻成 ✅ 的項目都附 commit / PR / 路徑作為證據。
>
> 現況摘要：
>
> - **規則偵測器 7/16 動作**：squat、push-up、overhead press、lunge、deadlift、row、
>   band pull apart（`src/pose/movements/`，registry 驅動；`/api/movements` 由 registry 導出，
>   新增偵測器不需改前端）。PR #47 #48 #51 #53 #54。
> - **逐 rep 偵測已上線**（`src/pose/rep_segmentation.py`，PR #49）；RS-SP2「只密集抽取
>   要評分的 rep」仍在 **PR #50（未合併）**。
> - **chat 已升級為 tool-calling 迴圈**（PR #57）：`get_analysis` / `kg_query` / `rag_search`
>   ＋ SSE tool 事件 ＋ tool trace 落庫。**Critic-lite 仍未做**。
> - **物件儲存上線**（PR #55，Cloudflare R2 + presigned 直傳）。
> - **LINE 全線上線**：Login/LIFF（#37 #43）、訓練摘要 bot（#41）、admin 診斷面板（#45 #46）。
> - **前端全面改版**（#58 #59 #60 #62）：studio、側邊 rail、My Records、動作圖書館、新 X mark。
> - **仍未動**：contrastive learning、VideoMAE↔pose fusion、非同步佇列（Celery/Redis）、
>   Critic-lite、temporal localization（VideoMAE 路線）。
>
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

> ⚠️ **2026-07-29 論文範圍決策尚未回寫到本節與下方「模型設計 / 實驗設計」**：
> 已決定寫**一篇**論文（非碩論），主軸是「可解釋性強迫使用可引用的固定閾值 ⇒ MPJPE 不是對的
> 目標」，統合 Fit3D 與 Fitness-AQA 兩條線。這會重新界定下面 Stage 2/4 與實驗設計的優先序，
> 但屬於**計畫改寫**而非狀態更新，故此處只留標記，待決定後再改。
> 角度整理在 `paper` 分支的 `notes/paper_angles.md`（commit `a9f27386`，尚未進 main）。

## 資料集現況整理

> `notes/dataset-summary.md` 是跨資料集的質性 survey（含 EgoExo-Fitness 標註量表）；
> 下列 Squat 數字目前只零星出現在實驗筆記中
> （例如 `notes/rtmpose_result_analysis_and_backend_comparison.md` 記了 `n=1623`），
> 尚未正式整理成一份資料集規格文件。

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

> `src/video/` 自 2026-07-07 起 **0 個 commit**——VideoMAE 這條線自上次盤點後沒有推進。

### Stage 1: VideoMAE V2 特徵提取 — ✅ 大致完成（research，見 `notes/videomae_classifier_experiment_summary.md`）

- [x] 決定使用預訓練 `VideoMAE V2` 作為 backbone
- [x] 決定先採用：
  - frozen backbone（特徵提取路線，見 `src/video/`）
- [ ] 決定 clip 長度（`16` vs `32 frames`）— 實驗設定需回寫成正式決策
- [x] 決定特徵輸出方式：CLS token（rehab24 分支同款）

### Stage 2: Contrastive Learning — ⏸ 未開始（2026-08-08 複查：`src/` 與 `scripts/` 全 repo 無 `contrastive` / `InfoNCE` 實作）

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
- [ ] Stage 3.5：temporal localization —— VideoMAE 路線仍無 segment IoU / frame-level 定位；
  時間定位目前只靠規則偵測器，且自 PR #49 起是 **per-rep** 而非整支影片一次判定
  （`src/pose/rep_segmentation.py`）

### Stage 4: Pose / Geometry Fusion — 🔶 一半（pose-only baseline 已做，VideoMAE↔pose fusion 未跑）

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
- [x] **（相鄰但不同的問題）3D pose 模型之間的 cue-level 融合已被否證**：
  `notes/model_fusion_gate_results.md` — 直接平均多個姿態模型的 cue 會失敗（誤差相關 0.55–0.78，
  表面「增益」其實是 bias 互抵）。**這不推翻上面的 VideoMAE＋pose 特徵融合**（不同層級、
  不同模態），該項仍為未執行。

## 即時性規劃 — ⏸ 未開始（設計文件 §2.4 已納入「即時 webcam 模式」為遠期項）

> 2026-08-08 複查：`scripts/`、`notes/`、`src/` 沒有任何 `nlf_s`（S backbone）實驗產物，
> Exp 1–3 三項全部未開始。唯一相關的既有結果是瀏覽器端 MediaPipe 擷取
> （PR #44、#56：client-side pose capture + 推論最佳化），走的是完全不同的路。

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
- [ ] Exp 1(先做,最便宜,無相依):nlf_s_multi 過現有 Fit3D 框架 + 在這台機器上實測 CPU latency。回答「深度恢復在 S backbone 下還在不在、CPU 多快」。⚠️ 版本對齊:你們跑的是 v0.2.0 nlf_l_multi,配對就用 v0.2.0 nlf_s_multi(v0.2.2 修了 detect_smpl_batched 的 translation/2D 投影 bug,S-vs-L delta 要同版本才公平)。
- [ ] Exp 2(延遲槓桿):nlf_s_crop + 便宜 box(MediaPipe 偵測/每 N 幀追蹤),量 CPU latency 與 Fit3D 上的準確度保留。
- [ ] Exp 3(1–2 有前景才做):接一個非 NLF 的輕量架構(ROMP 或 HybrIK)當第二點交叉驗證,寫 adapter 到同 npz。
- [ ] 訓一個 CPU 學生(輸入 MediaPipe 2D+world 或裁切影像,目標 NLF depth),直接把「NLF 的深度恢復」蒸餾進一個 CPU 模型——這比找現成輕量模型更精準命中「便宜地拿回 NLF 深度」。列為 stretch,因為工程量較大。

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

### 規則偵測器的標註驗證 — 🔶 起步（2026-08-08 新增：這一節原本不存在）

- [x] 第一個對照人工標註驗證的偵測器：**Lunge**（PR #51，
  `notes/lunge-rule-validation.md`）——REHAB24-6 `Ex5`，174 個人工標註 rep、8 位受試者、
  兩台正交攝影機。這是本 repo **第一個**被真實標註檢驗過的偵測器。
  - 結果：出貨用的 lead-leg cue **在三維下就是錯的**（不只是投影損失）；
    規格書自己定義的另一半替代量測從單目 2D 拿到 0.959/0.894。
  - 注意：**沒有任何閾值因此被調整**，`LUNGE_DETECTOR.validated` 仍為 `False`。
- [ ] Squat / Overhead Press / Push-up / Deadlift / Row 五個偵測器仍是
  「spec-derived、UNVALIDATED」，前端以 Beta tag 標示（`/api/movements` 的 `validated` 欄位）
- [ ] 把 `validated` 從「人工判斷」變成「有標註集撐腰」：每個動作至少一組標註資料 + 回歸腳本

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

### 已完成的其他實驗線（2026-08-08 新增：這些原本在本文件裡沒有位置）

> 全部是「深度瓶頸」主線的延伸，成果都在 `notes/`。

- [x] Fit3D 深度瓶頸系列：view-dependence、depth recovery、decision fidelity、
  model comparison、2D-vs-3D 分解、sparse-depth（`notes/fit3d_*_summary.md`，PR #15）
- [x] Fit3D「盲點」系列（皆為否證結果）：axial rotation、bar geometry、uncertainty
  （`notes/fit3d_{axial_rotation,bar_geometry,uncertainty}_summary.md`）
- [x] 模型融合閘門實驗（`notes/model_fusion_plan.md` + `notes/model_fusion_gate_results.md`，
  `src/fit3d/model_fusion.py`，34 tests）
- [x] 相機擺位掃描（`notes/camera-placement-hypothesis.md`）：13 個 Fit3D 動作的虛擬相機方位掃描；
  **oblique 不是最準的**，sagittal 在 sagittal cue 上 13/13 全勝，oblique 只是雙平面折衷（11/13），
  代價是 +0.03–0.10 的判決翻轉率
- [ ] Fitness-AQA 淺蹲下游測試（深度通道 **冗餘**，偵測器品質才是主導）——
  程式在 **`feat/fitness-aqa-squat-depth` 分支，尚未進 main**
  （`src/fitness_aqa/`、`scripts/fitness_aqa/`）：決定要合併還是留在研究分支

## 最終成果形式 — ✅ v1 完成（規則式；VideoMAE 版尚未接入 app）

- [x] 完成一個可輸入深蹲影片的模型（規則偵測器 5 faults + KG/RAG）
- [x] 輸出內容至少包含：
  - 錯誤類型
  - 錯誤發生時間區段
  - 信心分數
  - 結構化回饋
- [x] 規劃 demo 介面（Studio 已上線，並於 PR #58/#59/#60 全面改版）：
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
- [x] 第六步再考慮 RAG 與教練式文字回饋（GraphRAG + SSE chat + followup chips 已上線，
  並於 PR #57 升級為 tool-calling）

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
  - [x] 追加 **LINE Login + LIFF** 第二條登入路徑（PR #37、#43）
  - [ ] ~~access JWT + refresh token 放 httpOnly cookie~~ → 現況為 supabase-js 預設 localStorage（`frontend/src/lib/supabase.ts`）；換 httpOnly cookie 需自訂 storage，列為後續強化
- [x] PostgreSQL schema（Supabase migration，取代 Alembic）：
  - [x] `users`（Supabase auth 內建）
  - [x] `videos`（`storage_key`、`status`，upsert on user_id+video_id）
  - [x] `analyses`（`result JSONB` 整包 + 提升 `view_type`/`fault_count`/`movement` 為欄位）
  - [x] `conversations`（chat 訊息 JSONB + followups + **tool records**，計畫外新增）
- [x] 分析結果落地：`/api/analyze` 算完寫入 DB（登入者 best-effort persist）
- [x] 前端：React Router + Auth context + 受保護路由 +「我的紀錄」（History 頁，
  PR #59 重建為 My Records，含手機版 funnel 篩選、單筆刪除）
- [ ] 前端資料層改用 TanStack Query（未採用；現況 supabase-js + 自製 fetch，運作正常，視痛點再決定）
- [x] 設定改用 pydantic-settings + env（`backend/app/settings.py`）
- [x] **（計畫外）admin console**：使用者監看、runtime settings 覆寫層、LINE 診斷面板
  （PR #35、#45、#46；`backend/app/routers/admin.py`、`frontend/src/pages/admin/`）

### P2：非同步化 + 儲存 — 🔶 一半（儲存做完，佇列還沒）

- [x] 物件儲存（Cloudflare R2）：原始影片、pose JSON、縮圖（PR #55，`backend/app/services/storage.py`）
- [x] presigned URL 直傳：影片不經過 FastAPI（解掉 `await file.read()` 整支進 RAM）
- [ ] Celery + Redis 佇列：上傳→回 job id→worker 處理→輪詢/SSE 取結果
  （`config.py:41` 與 `routers/analyze.py:24` 都還標著「P0 stop-gap until the Celery/Redis
  worker queue lands」——semaphore 仍是目前唯一的背壓機制）
- [ ] job 狀態機：queued/processing/done/failed、重試退避、dead-letter、timeout
  （`store.py` 目前只寫死 `status: "done"`）
- [ ] 去重：影片 hash，同人同片回快取、不重算（backend 無任何 video hash）

### P3：規模化 + 維運 — 🔶 起步

- [ ] 拆 web / worker 部署（Docker；先 Railway/Render/Fly.io，之後 ECS/GKE）
  → **containerise 的第一步在 PR #61（`feat/docker`）未合併**，main 上還沒有任何 Dockerfile
- [ ] GPU 用 serverless（Modal/Replicate/RunPod，可縮到 0），VideoMAE / direct-3D 在此跑
- [ ] worker 依佇列長度自動擴縮（KEDA）；CPU/GPU 分池
- [ ] CDN + 簽名 URL 提供影片（R2 presigned 已有，CDN 未接）
- [ ] DB 連線池（PgBouncer）、`(user_id, created_at)` 索引、必要時讀副本
- [ ] 每人 rate limit + 上傳配額
- [ ] 可觀測性：結構化 log、Sentry、佇列/延遲/GPU 指標、health/readiness probe
- [ ] 加入 Pub/Sub, 處理高流量

### 橫切議題（越早處理越省事）

- [ ] 隱私 / 個資（PDPA・GDPR）：影片屬敏感個資 — 靜態加密、每筆綁 `user_id`、簽名 URL、
  刪帳號連物件儲存一起清、log 不記影片內容/URL
  （帳號刪除仍是 stub：`frontend/src/components/settings/AccountPane.tsx:92` 只有一列 UI，
  後端沒有對應端點）
- [ ] 檔案驗證：驗真實 MIME/codec（別只信副檔名）、限大小/長度、ffmpeg 正規化方向與格式
- [x] 可重現性（一半）：每筆分析已存 `pipeline_version`（`backend/app/services/store.py`）
  - [ ] 補存當時規則閾值 snapshot（detector 閾值可調，`analysis.py` 目前不存 threshold）
- [x] 儲存抽象層（dev 用本機、prod 用 R2/S3）— PR #55 的 `services/storage.py`
  - [ ] repo-root / `sys.path` 耦合本身仍未重構

## Demo

- [ ] Line ChatBot — 剩下 LLM 對話
- [ ] QR Code demo, real time interaction
- [ ] 語音回饋（composer 已預留 UI 槽位，功能未做）
- [x] **（計畫外）互動 mini-games**：`/games` hub、`/67` 手勢計數、`/ninja` Fruit Ninja
  （PR #27、#29、#30），皆用瀏覽器端 MediaPipe，含每局熱量估算
  - [ ] 其餘 game 分支（Pose Duel #26、Meme Blaster #25、Pose Match Rush #24）仍未合併
- [ ] 動作偵測、分類（= 多動作 movement ID，見路線圖 P2）
- [x] LLM follow up questions (options)（followup chips 已上線，pinned 快速模型）
- [ ] 健身菜單客製化，可用 LLM 進行修改（= 路線圖 P2 的 `make_drill_plan` 工具）
- [ ] 新增運動科學、運動力學及 mocap 相關知識筆記的頁面

## Pipeline 與 Agent Harness 路線圖（2026-07-06 設計，詳見 `docs/ai-coach-pipeline-and-agent-harness.md`）

> 產品主線改依此路線推進；上面各節屬研究支線或被此路線圖涵蓋。

### P0：地基 — 🔶 兩項完成，佇列未動

- [x] KG 切換 `squat_kg_v2` → `sports_kg_v3` + movement-aware 檢索參數
  （`config.py` 指向 v3；`movement` 參數貫穿 services/router；
  `/api/knowledge/graph?movement=` + `/api/knowledge/faults?movement=`；commit `23d1089f` + `3ea1f1fd`）
  - ⚠️ 修正前次盤點的錯誤敘述：commit `835afbf2` 的 **Explore 頁與 MovementSelector 已被移除**
    （commit `9058abe9`，使用者指定）。現況是扁平的 `/movements` 動作圖書館頁
    （`frontend/src/pages/Movements.tsx`，PR #60 依 exercise_library 參考稿重建），
    **app 內沒有 KG 瀏覽器**；`api.movementFaults` 與後端 faults 端點保留。
- [x] Rep 切分 + per-rep metrics — `src/pose/rep_segmentation.py`（PR #49）：
  以單一 1-D 訊號 + 遲滯門檻切 rep，規則逐 rep 執行（預設 3 rep 取首/中/末）
  - [ ] RS-SP2（粗掃找 rep → 只對選中的 rep 密集量測，客戶端擁有 rep 邊界）仍在 **PR #50 未合併**
  - [ ] frame_metrics 保留策略仍未定案（建議存 per-rep 摘要）
- [ ] 非同步分析佇列（= 上方系統 P2 的 Celery/Redis 項）

### P1：Agent 最小可用 — 🔶 四項中三項完成

- [x] chat 升級為 tool-calling 迴圈（PR #57）：OpenRouter function calling，
  三個工具 `get_analysis` / `kg_query` / `rag_search`，`backend/app/services/chat.py`
  含 bounded loop、串流 tool-call 重組、與「檢索到的知識是參考資料、不是對這支影片的觀察」
  的誠實性規則
- [ ] Critic-lite：回答送出前做 grounding 檢查（FAIL → 重生成一次 → 降級為 FaultCard 模板直出）
  — **仍未實作**（`frontend/src/lib/grounding.ts` 是送出前組 grounding blob 的建構器，
  不是回答後的查核器；`chat.py` 內無 critic）
- [x] SSE 加 tool 事件，前端顯示工具狀態（`frontend/src/components/ToolRunList.tsx`、
  CoachTray 具名狀態列、pending dots、來源折疊為可點擊計數）
- [x] 工具呼叫 trace 存進 conversations JSONB（可重播、可稽核；含 per-tool sources）

### P2：多動作 + 記憶 — 🔶 偵測器 7/16，其餘未動

- [x] **多動作規則偵測器（原「Lunge rule pack」已被更大的工程取代）**：
  registry 驅動的 per-movement 偵測器，端到端接進 web app（PR #47、#48、#51、#53、#54）
  - 已上線 7 個：`squat`、`pushup`、`overhead_press`、`lunge`、`deadlift`、`row`、
    `band_pull_apart`（`src/pose/movements/`；`GET /api/movements` 直接由 registry 導出，
    新增一個偵測器不需改前端）
  - 皆帶 Beta tag（`validated=False`），唯一有標註驗證的是 Lunge（見上方「規則偵測器的標註驗證」）
  - 部分規則被**證明無法實作**並明白記錄（Row 第 5 條、Deadlift 撤回一條、
    Band Pull Apart 的 scapular retraction 條恆為 silent）
  - [ ] 其餘 9 個動作的 rule pack 未做
- [ ] `compare_analyses` / `list_user_history` 工具 + 進步追蹤（跨分析記憶）
- [ ] Drill library + `make_drill_plan` 工具（fault → KG `CORRECTED_BY` → 矯正課表）
- [ ] 動作識別（movement ID）輕量分類器，自動載入對應 rule pack
  （目前仍由使用者在 studio 下拉選單指定動作）
- [x] 各個動作，相機擺放最適合的位置、角度 — `notes/camera-placement-hypothesis.md`：
  虛擬相機方位掃描（13 個 Fit3D 動作）。結論：**oblique 不是最準的**；
  sagittal 在 sagittal cue 上 13/13 全勝，oblique 只是雙平面折衷（11/13），代價 +0.03–0.10 判決翻轉
  - [ ] 把這個結論變成產品面的拍攝指引（app 內拍攝提示目前沒有依動作給角度建議）
- [ ] **（待多動作偵測器補齊後）全面泛化「分析流程」文案**：前端仍多處把流程說成深蹲專屬。
  前次盤點的「~26 處」是 PR #58/#59/#60 改版**前**的數字，**已作廢**。
  2026-08-08 人工判讀（非 grep 計數）`frontend/src/lib/i18n.tsx`，需要改的英文字串約 **9 條**：
  `chat.intro`、`tier.lite.hint`、`landing.hero.sub`、`landing.cta.title`、`landing.showcase.sub`、
  `auth` 登入說明、`history.emptyHint`、`history.startCta`、Fruit Ninja 說明中的
  「x-coach's squat analysis」。（`landing.showcase.squat.*` 與範例教練語句中的 goblet squat
  屬合理保留，不算。）另有硬字串散在 `App.tsx`、`UploadDropzone.tsx`、`CaptureStudio.tsx`、
  `StudioMobile.tsx`、`DemoIntro.tsx`、`history/HistoryStats.tsx` 等，尚未逐一清點。
  注意 `landing.hero.sub` 目前寫「squat, push-up or overhead-press」——偵測器已到 7 個，
  這句本身也過期了。
  使用者 2026-07-17 指定：未來採「完全泛化」而非加註解（en + zhHant 皆需更新；
  品牌／landing 文案已於 commit `4d64e659` 泛化為 multi-exercise）。
- [ ] **（待多動作分析上線後）修正 landing showcase 過度宣稱**：
  `landing.showcase.sub`（「reads the rest of the library, on real footage」）與
  `landing.showcase.title`（「One pipeline, the whole movement library.」）仍在
  （`frontend/src/lib/i18n.tsx`、`frontend/src/landing/MovementShowcase.tsx`），
  但 showcase 的 push-up/high-knee/sit-up clip 只有 MediaPipe pose tracking、無 fault 偵測。
  偵測器已到 7 個，落差比原本小但仍存在（16 個動作的知識庫 vs 7 個動作的分析）。
- [ ] 許多動作的錯誤需要脊椎及其他 MediaPipe 無法定義的點，思考其他解決方法。
- [ ] 許多錯誤沒有對應到 Knowledge Graph 的節點。
  - Band Pull Apart 具體案例（2026-08-09）：`Bent Elbows` 節點存在但 connectivity 0
    （沒有 cause / risk / correction），`trunk_extension_compensation` 則完全沒有對應節點。
    兩者都是 `scripts/knowledge/stub_general_movements_v3.py:80-87` 的一行修正，
    但 graphml 已 gitignore，重新產生屬部署步驟。
- [ ] `src/pose/rep_segmentation.py` 的 `DEFAULT_MIN_REP_SECONDS = 0.4`（30fps 下 12 幀）對
  Band Pull Apart 未曾用真實影片驗證過節奏，若真實 clip 每下快於 0.4 秒會整段被丟成雜訊，
  退回 whole-clip fallback——與 High Knee 需要 `min_rep_seconds` override 是同一類問題。

### P3：感知升級 — ⏸ 未開始（但研究面已備妥依據）

- [ ] 深度類 cue 的 direct-3D 路由（NLF 類模型，GPU；先做「深入分析」按鈕的非同步重分析）
  — 研究面證據已齊（Fit3D 深度恢復、decision fidelity、sparse-depth 三份 notes），
  但**產品端一行都還沒接**
- [ ] RAG 換神經 embedding（hash-BoW → sentence-transformer 類本地模型，介面不變）
- [ ] VideoMAE 融合分類器接入 app（補規則抓不到的「順不順」缺陷）

### 評估迴路（隨 P1 起步）— ⏸ 未開始

- [ ] golden-set 規則回歸（已標註影片，CI 跑 rule pack diff）
  — 素材已有一份：Lunge 用的 REHAB24-6 `Ex5` 174 rep（`scripts/rehab24/validate_lunge_rules.py`），
  但尚未進 CI
- [ ] RAGAS-style faithfulness 離線抽樣評估
- [ ] Critic grounding score 線上入庫（前提是 Critic-lite 先做出來）
