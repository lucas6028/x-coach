# TODO

> 2026-08-20 精簡版。只留現況與未完成項；細節見 `notes/`、`docs/`、各 PR。
> 前次完整盤點：2026-08-08（git history 可查）。

## 現況（已完成，不再列）

- **規則偵測器 16/16 動作設計完成、14 個註冊上線**（`src/pose/movements/`，registry 驅動；
  Jumping Jacks / High Knee 刻意不註冊——零條 live 規則）。全部 `validated=False`（Beta）。
- **逐 rep 偵測**（PR #49）、**tool-calling chat**（PR #57）、**R2 物件儲存**（PR #55）、
  **LINE Login/LIFF/bot/admin 面板**（#37 #41 #43 #45 #46）、**訓練菜單**（PR #80）、
  **Docker + Azure Container Apps 上線**（PR #82）、**前端改版**（#58–#60 #62 #83 #84）。
- **研究線已結案**：Fit3D 深度瓶頸系列＋三個盲點否證、模型融合否證、相機擺位掃描、
  VideoMAE Stage A（pooling 修正後 0.657，通過）/ Stage B（retention 失敗、fusion 無增益，
  不進 Stage C）/ REHAB24 framing 三臂（「差的是人不是框」）。
- Squat 資料集規模、標註格式、baseline 指標（VideoMAE-only、pose-only）皆已在 `notes/` 記錄。

## 論文

- [ ] 2026-07-29 決定：**一篇**論文，主軸「可解釋性強迫使用可引用固定閾值 ⇒ MPJPE 不是對的目標」，
  統合 Fit3D 與 Fitness-AQA；E5 是唯一缺口（`notes/paper_angles.md`）。
- [ ] 文件層：寫明資料集可做/不可做（非完整 AQA regression、`knees_inward` 不平衡、標註粒度不一）。

## 研究支線（未開始，優先序依論文決定）

- [ ] Contrastive learning（Stage 2）：repo 內無任何實作。
- [ ] VideoMAE ↔ pose 特徵融合（late fusion / concat+MLP）；Stage B 結果已降低其價值。
- [ ] VideoMAE temporal localization（segment IoU / frame-level P/R）。
- [ ] 特徵空間視覺化（t-SNE / UMAP）。
- [ ] 即時模式：nlf_s 系列實驗（Exp 1–3）與 CPU 蒸餾學生，全部未開始。
- [ ] Fitness-AQA 淺蹲下游測試程式仍在 `feat/fitness-aqa-squat-depth`，決定合併或留研究分支。

## 規則偵測器：待驗證與已知缺陷

驗證（標註資料在手、沒人跑）：
- [ ] **Arm Abduction** ← REHAB24-6 Ex1（178 下，單手變體；只有 trunk-lean 規則講得上話，出貨 12° 門檻 0/178 觸發）。
- [ ] **Arm VW** ← REHAB24-6 Ex2（208 下，雙手；只有 `vw_loss_of_elevation` 有訊號，AUC 0.735）。
- [ ] **Shoulder Bridge** ← EgoExo-Fitness 77 個評分 action；缺 `frames_open` 的 `.ac` 分割（實際 21 parts / ~66 GB）。
- [ ] 把 `validated` 變成「有標註集＋回歸腳本」；golden-set 回歸進 CI（Lunge Ex5 腳本已有）。

框架層缺陷（刻意不在規則內修）：
- [ ] **靜止片段會以滿分 fire 所有 incomplete-ROM 規則**：`segment_reps` 百分位閾值是 scale-free，
  0.4° 抖動切成 3 rep。修法二選一：noise floor，或把 `fallback` 接進 `RuleContext`。
  `tests/test_situp.py::test_a_motionless_clip_fires_this_rule_at_full_severity` 釘住，修好會轉紅。
- [ ] **`angle_degrees` 無號、對 180° 對稱**：>180° 的規則永遠不 fire，<160° 規則會對「拱過頭」反向提示。
  兩種補號法在真實影片都失敗。待盤點 registry 內依賴角度方向的規則。
- [ ] **`view_estimation.py` 在站姿受試者也系統性反轉**（Leg Abduction 以 Ex4 `cam17_orientation` 實測）。
  待盤點所有 gate/discount 在 view 標籤上的規則（至少 squat `knees_inward`、arm_abduction 兩條 frontal）。
- [ ] `arm_abd_lr_asymmetry` / `ohp_asymmetric_press` 是否需 view gate（Ex2 證明斜角**製造**不對稱，arm_vw 已 gate）。
- [ ] `band_pull_apart.rule_shrugging` 的 shoulder-ear gap confound 未在自己資料量過（方向上應較小）。
- [ ] citation 與資料方向相反的 `Pelvic Drop` 類規則，其餘動作待查。
- [ ] Bicep Curl ROM 門檻貼邊（1/40、0/40）、伸展判定受 rep trimming 影響——只記錄不調（no-threshold-tuning）。

## KG 缺口（改 `scripts/knowledge/stub_general_movements_v3.py`，graphml gitignore ⇒ 重生成是部署步驟）

- [ ] Band Pull Apart `Bent Elbows` 無連結、`trunk_extension_compensation` 無節點。
- [ ] Bicep Curl `Elbow Drift Forward` 無連結。
- [ ] Arm Abduction / Arm VW 無 asymmetry 節點（現用泛用 `Muscle Imbalance`，刻意保留薄卡片）。
- [ ] **Sit-up：圖講完整仰臥起坐、spec 講捲腹**，交集為零——要先決定 app 出哪一種。
- [ ] Shoulder Bridge `No Segmental Spinal Articulation` dangling；`Pelvic Drop` 無節點。

## 系統 / 產品

- [ ] **非同步佇列**（Celery/Redis；job 狀態機、重試、timeout）——semaphore 仍是唯一背壓。
- [ ] 影片 hash 去重。
- [ ] Critic-lite：回答送出前 grounding 檢查（FAIL → 重生成 → 降級 FaultCard）；之後 grounding score 入庫。
- [ ] `compare_analyses` / `list_user_history` 工具＋進步追蹤。
- [ ] Drill library + `make_drill_plan`（fault → KG `CORRECTED_BY`）；菜單 LLM 客製化。
- [ ] 動作識別分類器自動選 rule pack（目前 studio 下拉手選）。
- [ ] 拍攝指引：把「sagittal 13/13 勝、oblique 是折衷」變成 app 內依動作的角度提示。
- [ ] 深度 cue 的 direct-3D 非同步重分析（研究證據齊，產品端零行）。
- [ ] RAG 換神經 embedding；VideoMAE 分類器接入 app。
- [ ] 前端：`FAULT_LANDMARKS`（`frontend/src/lib/pose.ts`）只涵蓋 squat 5 個 fault，其餘偵測器不 highlight；
  i18n 仍有 ~9 條 squat 專屬文案；landing showcase 過度宣稱。
- [ ] 橫切：帳號刪除端點（前端是 stub）、檔案 MIME/codec 驗證、規則閾值 snapshot 入庫、
  rate limit / 配額、可觀測性（Sentry、probe）、httpOnly cookie session、TanStack Query（視痛點）。

## Demo

- [ ] Line ChatBot — 剩下 LLM 對話
- [ ] QR Code demo, real time interaction
- [ ] 語音回饋（composer 已預留 UI 槽位，功能未做）
- [x] 互動 mini-games：`/games` hub、`/67` 手勢計數、`/ninja` Fruit Ninja（PR #27 #29 #30）
  - [ ] 其餘 game 分支（Pose Duel #26、Meme Blaster #25、Pose Match Rush #24）仍未合併
- [ ] 動作偵測、分類（= 多動作 movement ID，見上方「系統 / 產品」）
- [x] LLM follow up questions (options)（followup chips 已上線）
- [ ] 健身菜單客製化，可用 LLM 進行修改（= `make_drill_plan` 工具）
- [ ] 新增運動科學、運動力學及 mocap 相關知識筆記的頁面
