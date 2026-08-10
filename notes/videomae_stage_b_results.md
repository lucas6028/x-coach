# Stage B 結果:Fitness-AQA corrected-pooling squat fault detection

對應計畫:[`videomae_dataset_validation_plan.md`](videomae_dataset_validation_plan.md) Stage B。
前一階段:[`videomae_stage_a_results.md`](videomae_stage_a_results.md)(通過,建議進入本階段)。

---

## 0. 事前登錄(pre-registration)

**本節在任何 Stage B 結果產生前寫定並提交。** Stage A 學到的第一課是:四臂設計會同時
給出三個 delta,等看到數字再挑一個當主結果,等於用 best-of-three 冒充單一假設檢定。
Stage B 更嚴重——計畫列了兩種 fusion(early / late)乘上三個 label mode,共六個候選
delta,再加上兩個控制組。以下在跑任何實驗前先鎖定。

### 0.1 主結果

| 項目 | 值 |
| --- | --- |
| **Primary arm** | **Calibrated late fusion**:normalized pose-only 與 corrected VideoMAE-only 各自訓練,於 validation 上校準後合併機率 |
| **分母(paired 比較的基準)** | **Normalized pose-only**,本機重抽特徵重跑,歷史值 `0.635 ± 0.010` |
| **Primary label mode** | `combined`(與 0.635 這個數字對應的同一個 label mode)|
| **Primary metric** | test split、selected-threshold balanced accuracy,5 個 seed(1–5)的平均 |
| Seeds | `1, 2, 3, 4, 5`——沿用歷史 pose-only 與 VideoMAE-only 的同一組 seed |
| 超參數 | `run_videomae_experiment_grid.py` 全部預設值(epochs 20 / batch 32 / lr 3e-4 / hidden 128 / dropout 0.4 / weight decay 0.01 / patience 5 / threshold objective `balanced_accuracy`),與產生 0.635 的那次執行逐項相同 |

**為什麼 primary 選 late fusion 而不是 early concat**(此判斷與 Stage B 任何結果無關,
純粹來自已完成的 Stage A 與維度事實):

1. Stage A §2.5 已經量到 early concat 在**三個** skeleton backbone 上都只收斂到兩個
   branch 中較強的那一個,沒有一次超越它。那是「這個 fusion 方法測不到互補性」的直接
   證據,不是猜測。
2. 計畫本身把 calibrated late fusion / gating 排進 Stage B,理由寫得很明白:降低弱 RGB
   branch 稀釋 pose signal 的風險。把它當 secondary,等於預先放棄計畫想測的東西。
3. 維度不對等會重演:VideoMAE 是 768 維,pose-only 特徵維度由 `pose_feature_extraction`
   決定,兩者 concat 後正規化能處理尺度、處理不了容量。

Early fusion 仍然會跑,但列為 **secondary/diagnostic**,其 delta 不作為保留決策的依據。

### 0.2 完整實驗矩陣與各臂角色

| 臂 | 角色 | 用來回答 |
| --- | --- | --- |
| `legacy_first_token_max` VideoMAE-only | 重現歷史抽取 | 分母是否可信(對照歷史 `0.555`,範圍 `0.532–0.584`)|
| `mean_pool_fc_norm_mean` VideoMAE-only | corrected standalone | task-specific 是否有獨立訊號 |
| Normalized pose-only | **主要可部署 baseline** | Stage B 的分母 |
| **Calibrated late fusion** | **primary** | 保留門檻的唯一判準 |
| Regularized early fusion | secondary | 與 Stage A 的 early concat 結論是否一致 |
| Person-crop 控制 | 捷徑檢查 | 訊號是否在身體上 |
| Background-only 控制 | 捷徑檢查 | 訊號是否只是場景/視角 |

Corrected VideoMAE 的 token pooling 與 clip aggregation 組合**沿用 Stage A 的 primary**
(`mean_pool_fc_norm` + `mean`),不在 Stage B 重新挑選;四種組合的特徵都會物化出來,
但只有 primary 進入決策,其餘僅供診斷。

### 0.3 分母門檻(必須先過,才准看 fusion 的數字)

`data/Squat/Labeled_Dataset/pose_features` 已不存在於本機,`pose_json/` 只剩 5 個檔案
(且那 5 個是 view-estimation regression corpus 的一部分,已先移到暫存區,由本次重抽
還原)。**0.635 因此不是封存數字,而是必須重新導出的分母。**

- 重抽方式與歷史完全相同:`src/pose/process_videos.py`(MediaPipe Pose,
  `model_complexity=2`)→ `pose_feature_extraction` → grid runner 加 `--normalize-features`。
- **判準:重抽後的 normalized pose-only combined balanced accuracy 必須落在
  `0.635 ± 0.010` 內。** 落在區間內,分母成立,所有 delta 以 0.635 這條線陳述。
- 若落在區間外:**不得默默吸收進 delta**。改以重抽值為分母,並同時列出兩個數字與差距。
- 附帶檢核:`tests/test_view_regression_corpus.py` 會用重抽出的那 5 個 test pose JSON
  對照凍結的 view baseline。verdict 若不動,就是重抽 recipe 與歷史一致的獨立證據。

### 0.4 保留門檻(直接引用計畫,不加碼也不放寬)

將 VideoMAE 保留為 production 預設 input 的條件:

1. Fusion 相對 normalized pose-only 的平均 `Δ balanced accuracy ≥ +0.02`。
2. Paired 95% CI 下限大於 0。
3. Recall 或 specificity 任一不得惡化超過 `0.03`。
4. 改善在多數 seeds 中方向一致。
5. Person-crop / background 控制後仍保留效果。

未達門檻但 VideoMAE-only 優於 null ⇒ 保留為研究或選配 branch,不作 production 預設。

### 0.5 統計方法與其已知限制

**Paired 95% CI 的重抽單位是「影片」。** 計畫寫的是 participant-level bootstrap,但計畫
自己的資料表也寫了 Fitness-AQA「沒有可靠 participant mapping」。已驗證的資料事實:
1623 個 labeled id 對應 **1623 個相異前綴**(`32903_8` 的前綴是 `32903`),也就是每個
樣本各自來自不同來源影片,沒有 clip-within-video 的群集結構可用;train/val/test 的前綴
交集皆為 0。因此:

- 重抽單位 = test split 的 244 支影片,paired(兩臂使用同一組重抽索引),2000 次。
- **誠實的限制:同一位健身者可能出現在多支影片中而無從得知,這個 CI 因此可能偏樂觀。**
  沒有任何方法能在這個資料集上修正它,只能陳述。

**Recall / specificity 的 ±0.03 護欄以「5 個 seed 的平均」判定,並同時報告跨 seed 全距。**
歷史 normalized pose-only 的 recall 是 `0.717 ± 0.101`、specificity `0.553 ± 0.111`——
單一 seed 上的 0.03 差距完全在雜訊內,拿單 seed 判護欄等於擲骰子。

**Threshold 與 checkpoint 都只由 validation 決定。** 已核對
`src/video/videomae_video_classifier.py`:`find_best_threshold` 吃的是 val 機率,
checkpoint 也以 val 目標挑選,test 只在最後評估。歷史的 0.635 與 0.555 同屬這個慣例,
所以兩邊可比。

### 0.6 三個必須連同結果一起讀的前提

1. **Legacy 是重新抽取,不是封存數字。** 舊的 Fitness-AQA VideoMAE 特徵不在本機,
   transformers 版本也已不同。§0.2 的 legacy 臂就是為了確認重現落在 `0.532–0.584` 內。
2. **兩個 fusion 的輸入特徵完全相同**,差別只在合併方式;pose branch 一律使用
   train-fold 統計的 normalization,不與未 normalization 的舊 baseline `0.581` 比較。
3. **控制組的變體影片是新編碼的**。person-crop 與 background-only 由同一組 pose box
   產生,frame count 補齊到來源 header,確保三個變體的 clip start 完全一致——否則「控制
   組掉了幾分」會混進「取樣到不同畫格」這個無關因素。

### 0.7 停止條件

若 corrected VideoMAE-only 在 Fitness-AQA 上仍接近隨機(≤ 0.55),則不再往 fusion 與
EgoExo-Fitness 推進,並依計畫的結果解讀表記為「REHAB24-6 成功、Fitness-AQA 失敗:
VideoMAE 含一般動作品質訊號,但不適合目前 squat fault endpoint」。

---

## 1. 方法與實作

### 1.1 修正的是同一個 pooling,這次在第二個抽取器上

計畫第 1 步要求修正**兩條** VideoMAE extractor。Stage A 修了 REHAB24-6 那條;
`src/video/videomae_feature_extraction.py` 直到本階段仍原封不動地帶著同一個缺陷:
`last_hidden_state[:, 0, :]`(VideoMAE 沒有 CLS token,這是第一個 tubelet 的左上角
patch)加上對 clip 取 `max`。它的預設路徑還指向 `data/Squat/...`,而資料集早已搬到
`data/Fitness-AQA/Squat/`。

修正後與 Stage A 對齊:兩種 token pooling 由**同一次 forward** 算出、存下 per-clip
stack、每個 `.npz` 蓋 provenance、輸出到新目錄。

### 1.2 新增的共用元件

| 模組 | 用途 |
| --- | --- |
| `src/video/videomae_backbone.py` | fc_norm 載入與 default-init 防呆、單次 forward 的雙 pooling、`resolve_device` |
| `src/video/videomae_materialize.py` | 與資料集無關的物化步驟(REHAB24-6 與 Fitness-AQA 共用)|
| `src/video/videomae_audit.py` | 與資料集無關的稽核;Fitness-AQA 由 split-key JSON 驅動 |
| `src/video/squat_dataset.py` | 資料集路徑與 split/label 讀取,純 stdlib |
| `src/video/squat_video_variants.py` | person-crop 與 background-only 變體影片 |

`resolve_device` 把 Stage A 用一次失敗執行換來的教訓寫進程式:Kaggle 會配給 sm_60 的
P100,而該映像的 torch 只內建 sm_70 以上 kernel,錯誤發生在**模型載入之後**的第一個
conv3d。改為載入前先用真正的 conv3d 探測,失敗即退回 CPU。

### 1.3 控制組變體的兩個設計選擇

- **person-crop 用正方形而非貼身框**:`VideoMAEImageProcessor` 會把短邊縮到 224 再
  center crop 224,貼身的高瘦框會被裁掉腳——正是 Stage A 在 cam18 上必須列為 caveat 的
  那個預處理不對稱。
- **background-only 用「每像素時間中位數」補洞,而不是塗灰**:塗一個灰色矩形仍然把
  受試者的位置與大小畫在畫面上,控制組可能因為讀到那個矩形而「通過」。中位數在靜態
  相機上就是空場景;手持相機上是糊的,這點必須在解讀時說明。
- 每支影片只用**一個** box(跨影格取聯集),避免逐影格框帶入原本不存在的鏡頭運動。

### 1.4 執行紀錄

*(待填:pose 重抽、變體影片、三個 Kaggle kernel 的實際執行數據)*

---

## 2. 結果

*(待填。在 §0.3 的分母門檻通過前,本節不得寫入任何 fusion 數字。)*
