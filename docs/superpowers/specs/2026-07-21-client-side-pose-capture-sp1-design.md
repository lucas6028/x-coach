# Client 端姿態擷取（SP1）— 設計

Status: **設計已核准，待實作** · Created 2026-07-21

把 MediaPipe 姿態擷取從伺服器搬到使用者裝置（瀏覽器），並讓使用者能**即時錄影**或
**上傳影片**，錄影時即時顯示骨架。這是一個更大願景（C）的第一個、也是唯一先做的子專案：
「即時層」與「正式分析」解耦，之後可漸進換模型、加即時教練、升級 3D。

前置研究框架見 `project-overview.md`；姿態與規則偵測既有實作見 `src/pose/`；
既有 client-side MediaPipe（遊戲用）見 `frontend/src/components/poseLandmarker.ts`。

---

## 0. 大局與拆解（本 spec 只做 SP1 的 squat）

完整願景（代號 C）：**MediaPipe 當即時層**（顯示骨架 + 之後讓 Lumen 即時糾錯），
**正式分析是另一套**（依 fault 類型分流：2D 線索走 MediaPipe、depth/flexion 走 3D）。
它橫跨數個獨立子系統，拆成：

| | 子專案 | 內容 | 狀態 |
|---|---|---|---|
| **SP1** | Client 擷取層（動作無關） | 即時錄影／上傳、Lite 預設 + complexity 選擇器、即時骨架、pose JSON + 影片送後端；分析接**現有 squat 規則偵測器**，後端留可抽換 seam | **本 spec** |
| **SP2** | Per-movement 分析（16+ 動作） | 依 70 條規則 spec 逐動作建規則偵測器，插進 SP1 的後端 seam | 使用者另外進行中 |
| **SP3** | Lumen 即時糾錯 | 錄影時即時教練提示，只講 2D 可靠 fault（valgus/軀幹前傾），depth/flexion 閉嘴 | 後續 |
| **SP4** | 3D 正式分析升級 | 伺服器 3D 模型處理 depth/flexion，依 fault 類型與 2D 線索合併 | 後續 |

**擷取層是動作無關的**（33 點骨架與動作無關），所以 SP1 的擷取管線與 UI 一次就能服務全部
16+ 動作；**分析才是逐動作的**（SP2），SP1 只把 squat 這條端到端打通。後端端點從第一天就
**帶 `movement` 參數**，squat 以外的動作路由到「分析建置中」，讓 SP2 純粹是往 registry 加 key。

### 依 fault 類型分流的研究依據（供 SP3/SP4，SP1 不實作）

專案自己的 Fit3D 實驗已收斂：valgus、軀幹前傾在 2D 就可靠（甚至完美 2D ≈ 真實 2D）；
depth、flexion 是投影失真、只有 3D 能修（`notes/` fit3d 三個 finding）。所以未來 Lumen
即時層對 depth/flexion 閉嘴、交給 3D 層 —— 矛盾**由建構消除**（Lumen 從不對它看不見的線索
下判定）。SP1 先記錄此原則，不實作。

---

## 1. 範圍

**做（本 spec）**

- **選動作入口**：延伸現有 `/movements` 選單，Squat 卡片進入新的擷取畫面；其餘動作維持
  inert「Soon」。
- **擷取畫面 · 兩種模式**：上傳（沿用 `UploadDropzone`）與即時錄影（鏡頭預覽 + Lite 即時
  骨架疊圖 + `MediaRecorder` 錄成 blob）。
- **共用離線抽取管線**：錄影/上傳都把 blob 逐幀跑 MediaPipe，產出與 `process_videos.py`
  **一字不差**的 pose JSON。
- **complexity 選擇器**：Lite/Full/Heavy，只作用於「分析抽取 tier」；`localStorage` 記住。
  即時疊圖恆為 Lite。預設 tier 由 §7 驗證任務決定。
- **新後端端點** `POST /api/analyze/pose`：收 `movement` + pose JSON + 影片檔，
  跑 **movement-keyed 分析 strategy registry**（squat = 現有 `detect_pose_rules_from_json`），
  存影片、登入者 persist、回傳同樣的 `Analysis`。
- **前後端測試** + 守 CI 覆蓋率（後端 95%）。

**不做（另開 spec / 別的子專案）**

- SP2 per-movement 偵測器（使用者另外做）。
- SP3 即時 Lumen 糾錯、SP4 3D 分析升級。
- 伺服器端 3D 模型、非 MediaPipe 模型。
- 為 squat 以外動作接分析（後端只保留 seam 與「建置中」回應）。

**已經存在、本 spec 不重做**

- Client-side `PoseLandmarker`（`poseLandmarker.ts`，`@mediapipe/tasks-vision@0.10.35` 已是
  production 依賴）、相機取得與 hang 處理（`lib/camera.ts`）、逐幀等待（`lib/videoFrame.ts`）、
  能力偵測（`pages/LiffDiag.tsx`、`lib/poseProbe.ts`）。SP1 複用這些，不重寫。
- 伺服器端 squat 規則偵測器 `src/pose/pose_rule_detector.py` 與其 KG/RAG 檢索 —— 完全不動，
  只是改由新端點以 pose JSON 直接呼叫（跳過 `process_video`）。
- 前端 `Analysis` / `PoseBlock` / `PoseFrame` 型別（`api.ts`）與骨架疊圖
  （`SkeletonOverlay.tsx` + `lib/pose.ts`）—— 回傳形狀不變，沿用。

---

## 2. 架構：一條擷取管線，兩個輸入源

```
選動作 (/movements → Squat)
        │
        ▼
   擷取畫面
   ├─ 上傳模式 ── File ─────────────┐
   └─ 錄影模式 ── 鏡頭預覽 + Lite    │
                 即時疊圖 (視覺)     │
                 + MediaRecorder ── blob ┐
                                         ▼
                        ┌──────────────────────────────┐
                        │  共用離線抽取管線             │
                        │  blob → <video> 逐幀           │
                        │  → PoseLandmarker(分析 tier)   │
                        │  → 累積 33 點 landmarks         │
                        │  → 組 pose JSON (既有 schema)   │
                        └──────────────────────────────┘
                                         │ pose JSON + movement + 影片檔
                                         ▼
                        POST /api/analyze/pose
                        → movement-keyed strategy registry
                        → squat: detect_pose_rules_from_json
                        → 存影片 / persist / build_pose_block
                                         │ Analysis
                                         ▼
                        既有結果畫面（影片 + timeline + faults + 疊圖）
```

**設計要點**

- **即時疊圖與分析抽取解耦**。即時疊圖（錄影當下）純視覺、效能優先、恆用 Lite、掉幀無所謂；
  分析用抽取是**錄完之後離線**從 blob 重跑，不需即時，tier 可選。這讓「即時 Lite 順」與
  「分析準」兩個目標互不打架。
- **錄影與上傳共用同一條分析路徑**。兩者最後都是「一段 blob → 抽取 → pose JSON」，frame index
  對齊 blob 對應的影片，疊圖不漂移。錄影的即時骨架不餵分析（避免 FPS 掉幀/不一致污染判定）。
- **後端 seam 是可抽換的分析 strategy，不是硬接偵測器**。端點形狀是
  `{movement, pose JSON, 影片} → strategy(movement).analyze(...)`，squat strategy 是今天的
  規則偵測器；SP4 換 3D 模型是「換 strategy」而非重寫端點。

考慮過但否決：

- **在舊 `/api/analyze` 上加 `pose_json` 選填欄位** —— 舊端點語意是「給影片、伺服器抽」，
  硬塞會讓它同時承載兩種相斥流程；新端點語意乾淨。舊端點保留（給非瀏覽器客戶端/退路），
  前端改用新端點。
- **錄影時邊錄邊累積 landmarks 當分析輸入** —— 即時 FPS 會掉幀且與錄下影片幀率不一致，疊圖
  需 timestamp 對齊且分析精度可能下降；離線重抽換來與上傳路徑完全一致、frame index 對齊。
- **裝置不支援就偷偷退回伺服器抽** —— 違反「純 client」決定，且會讓兩條抽取路徑（Lite vs
  Heavy、client vs server）產生不一致判定。改為優雅報錯（§5）。

---

## 3. 前端元件

所有新檔在 `frontend/src`：

- **擷取畫面容器**（新頁/新元件，掛在 Squat 入口）：持有「模式（上傳/錄影）」與 complexity
  選擇器狀態，把產出的 blob + pose JSON 交給既有 `runUpload` 等價流程（改呼叫新端點）。
- **錄影元件**（新）：`camera.ts` 取 stream → `<video>` 預覽 → 疊一層 Lite 即時骨架
  （複用 `poseProbe`/`poseLandmarker` 的 detect loop）→ `MediaRecorder` 錄製，停止得 blob。
  `MediaRecorder` 是本專案首次使用，需處理 mimeType 選擇（`video/webm` / iOS `video/mp4`）
  與 iOS/Safari 相容。
- **上傳**：沿用 `UploadDropzone`，取得 File 即視為 blob 進管線。
- **抽取管線**（新，`lib/` 純函式）：`blob → PoseJson`。以 `<video>` +
  `requestVideoFrameCallback`（複用 `videoFrame.ts` 概念）逐幀 seek/decode，對每幀呼叫
  `PoseLandmarker.detectForVideo`，將 `result.landmarks[0]` 與 `result.worldLandmarks[0]`
  序列化成 §4 的 schema。回報進度（0..1）供進度條。
- **complexity 選擇器**（新，小元件）：Lite/Full/Heavy；值存 `localStorage`；只決定抽取
  管線用哪個模型 `.task`。
- **`poseLandmarker.ts` 參數化**：目前硬寫 `pose_landmarker_lite`；改成 `createPoseLandmarker(tier)`
  依 tier 載 `lite`/`full`/`heavy` 的 `.task`（即時疊圖傳 `lite`，抽取管線傳選擇器的值）。
  保留既有遊戲呼叫端相容（預設 `lite`）。

**單元可測性**：抽取管線設計成純函式 `(frames, tier) → PoseJson`（把 PoseLandmarker 以介面
注入），schema 序列化可獨立於瀏覽器測。錄影/相機那層薄、以整合層對待。

---

## 4. 資料契約（與現有 schema 對齊）

client 抽取管線輸出必須**逐欄位等同** `src/pose/process_videos.py` 寫出的 pose JSON：

```
{
  "metadata": { "fps": float, "width": int, "height": int, "total_frames": int },
  "frames": [
    {
      "frame_index": int,
      "landmarks":       [ {"x","y","z","visibility"} × 33 ] | null,
      "world_landmarks": [ {"x","y","z","visibility"} × 33 ] | null
    }, ...
  ]
}
```

- `x,y` 正規化 `[0,1]`；33 點 MediaPipe Pose 拓撲順序。
- 同時序列化 `landmarks`（含 `z`）與 `world_landmarks`，確保 `pose_rule_detector.py` 的
  `landmarks_to_array()`（需 ≥33 點、讀 `x/y/z/visibility`）完全不動。
- Tasks-Vision 的 `visibility`/`presence` 與 legacy `solutions.pose` 的 `visibility` 欄位對齊，
  取 `visibility`。
- `metadata.fps/width/height` 取自 `<video>`（`videoWidth`/`videoHeight`、錄影時取實際幀率或
  回報的 `fps`）。無偵測到人時該幀 `landmarks: null`（與現有一致，偵測器會 drop）。

後端回傳的 `Analysis` 形狀不變；`build_pose_block_from_payload` 從收到的 pose JSON 產生疊圖用
的 slim block（`{i, lm:[[x,y,visibility]×33]}`），前端型別 `PoseBlock` 不改。

---

## 5. 錯誤處理與退路

**維持「純 client、無伺服器抽取 fallback」決定。** 不因裝置不支援就偷偷改送伺服器。

- **無鏡頭 / `getUserMedia` 失敗 / hang**：複用 `camera.ts` 的 `CameraError`/timeout；錄影模式
  顯示錯誤與提示，**上傳模式仍可用**。
- **MediaPipe 初始化失敗**（WASM/GPU/CDN 取不到 `.task`）：進擷取畫面前以 `poseProbe`/
  `LiffDiag` 那套做一次能力偵測；不支援時明確告知「此裝置無法在瀏覽器內分析」，不 silent
  退伺服器。
- **抽取中斷 / 全片無人**：回報「未偵測到人體，請確認全身入鏡」；不送空 pose JSON 進分析。
- **長影片**：抽取顯示進度條；影片 > 30 秒給柔性提醒（可續，但提示較慢）。soft cap，不硬砍。
- **上傳/端點失敗**：沿用現有 `runUpload` 的錯誤呈現。

---

## 6. 後端

- **新端點** `POST /api/analyze/pose`（`backend/app/routers/analyze.py` 或新 router）：
  multipart 收 `movement`（str）、`pose`（JSON）、`file`（影片）。auth 選填，沿用現有 bearer。
  跑在既有 `asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)` 界限內。
- **分析 strategy registry**（新，`backend/app/services/`）：`movement → strategy`。
  `AnalysisStrategy.analyze(pose_json, video_path, ...) → Analysis`。
  - `squat` strategy = 呼叫既有 `detect_pose_rules_from_json`（**跳過 `process_video`**）
    + `build_pose_block_from_payload`。
  - 未註冊的 movement → 回「分析建置中」結構化回應（存影片、可回骨架，但無 faults）。
    SP2 只需往 registry 註冊新 strategy。
- **影片儲存 / persist**：沿用 `analyze.py` 既有邏輯（存檔、登入者 `store.persist_analysis`）。
- **舊 `/api/analyze` 保留不動**（退路 / 非瀏覽器客戶端）；前端改打新端點。
- **驗證輸入**：pose JSON 幀數/landmark 數基本檢查；影片副檔名沿用 admin 允許清單。

---

## 7. Lite 預設的驗證任務（SP1 第一步，gate 預設）

「Lite≈Full≈Heavy」的既有實驗（`notes/rehab24_correctness_experiment_summary.md:195-252`）
量的是**跨 6 復健動作 pooled 的下游二元 correctness 分類器 bal_acc**（Lite 0.663 / Full 0.686 /
Heavy 0.665，non-monotonic、作者自稱雜訊），單一 2 人 split ——**既非 squat 規則偵測器判定、
亦非 keypoint 誤差**。因此「squat 分析預設 Lite」目前是未驗證假設。

**任務**：在 squat 上跨 complexity 0/1/2 跑
`scripts/pose/evaluate_pose_rule_detection.py`（或等價的 squat-only 判定一致性檢查），
比較每個 fault 的 pass/fail 判定是否隨 tier 翻動。

- **判定不翻** → 分析抽取預設 **Lite**（即使用者原意），並在本 spec 與程式註記引用此驗證。
- **判定會翻** → 分析抽取預設 **Heavy**（與現有產線一致、零回退），complexity 選擇器仍讓
  使用者自行降階。

即時疊圖不受此影響（恆 Lite）。此任務先做，結果決定實作計畫裡 complexity 選擇器的預設值。

---

## 8. 測試

- **前端（vitest，cwd=`frontend/`）**：抽取管線純函式（給定 mock landmarks → 正確 schema、
  frame_index、null 幀處理）；complexity 選擇器持久化；`poseLandmarker` tier→`.task` 對映；
  錄影/相機層以整合層薄測。
- **後端（pytest，scope `tests/`）**：新端點 happy path（squat pose JSON → faults）、
  registry 路由（未知 movement → 建置中回應）、影片儲存/persist、輸入驗證；守
  `scripts/run_backend_coverage.py --fail-under 95`。
- **對齊 CI**（`.github/workflows/ci.yml`）本地跑過再宣稱通過。

---

## 9. 交付順序（供 writing-plans 展開）

1. §7 驗證任務 → 定 complexity 預設。
2. 後端：strategy registry + `POST /api/analyze/pose` + squat strategy + 測試。
3. 前端：`poseLandmarker` 參數化 + 抽取管線純函式 + 測試。
4. 前端：擷取畫面（上傳模式先接新端點端到端）。
5. 前端：錄影模式（相機 + Lite 即時疊圖 + MediaRecorder）。
6. 前端：complexity 選擇器 + `/movements` Squat 入口接線。
7. 錯誤處理與能力偵測收尾；全套測試 + 覆蓋率。
