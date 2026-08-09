# VideoMAE 資料集驗證計畫

## 目的

本計畫要回答的不是「VideoMAE classifier 能不能跑」，而是：

> 修正特徵提取方式後，VideoMAE 是否含有可跨人物泛化的動作品質訊號，並能在 pose features 之外為 x-coach 提供額外價值？

建議採用三階段驗證：

1. **REHAB24-6**：受控環境下驗證 VideoMAE representation 是否有效。
2. **Fitness-AQA**：驗證它是否能改善 x-coach 的 squat fault detection。
3. **EgoExo-Fitness**：驗證它能否支援細粒度、可解釋的 coaching feedback。

不建議直接捨棄 Fitness-AQA，也不建議一開始就投入 EgoExo-Fitness。先在最乾淨且 pipeline 已完整的 REHAB24-6 排除 representation 問題，再逐步增加任務與資料分布難度。

## 重要前置問題：現有 pooling 不正確

目前兩條 VideoMAE extractor 都將 `last_hidden_state[:, 0, :]` 視為 CLS embedding：

- [`src/video/videomae_feature_extraction.py`](../src/video/videomae_feature_extraction.py)
- [`src/rehab24/videomae_features.py`](../src/rehab24/videomae_features.py)

但目前使用的 Hugging Face VideoMAE 實作沒有可代表整段影片的 CLS token；classification path 預設會對所有 patch tokens 做 mean pooling，再套用 `fc_norm`。因此既有特徵實際上很可能只是第一個 patch token，而不是完整 clip representation。

這代表舊實驗只能視為 `legacy_first_token` baseline，不能用來斷言 VideoMAE 本身沒有訊號。

修正後應至少支援兩種明確命名的模式：

- `legacy_first_token`：重現舊方法，僅用於 paired comparison。
- `mean_pool_fc_norm`：對所有 tokens mean pooling，並套用 pretrained classification head 的 `fc_norm`。

所有新特徵必須輸出到新目錄，避免 extractor 因檔案已存在而沿用舊 cache。

## 資料集分工

| Dataset | 驗證角色 | 優點 | 主要限制 | 本機狀態 |
| --- | --- | --- | --- | --- |
| **REHAB24-6** | 第一階段：representation debugging | 10 位受試者、1,072 repetitions、雙 RGB 視角、binary correctness、2D/3D GT skeleton；可做 subject-wise LOSO | 受試者少、實驗室環境，不能單獨證明產品泛化 | 130 支 RGB 影片與 manifest 完整，pose/skeleton baselines 已建立 |
| **Fitness-AQA** | 第二階段：task-specific replication | 直接對應 in-the-wild squat form errors，最接近目前 x-coach endpoint | 沒有可靠 participant mapping；視角、背景與類別不平衡可能造成混淆 | 1,739 支影片、1,623 個 split samples；本機沒有 VideoMAE cache |
| **EgoExo-Fitness** | 第三階段：product-value validation | technical-keypoint verification、1–5 quality score、自然語言評語與同步 ego/exo views，最接近可解釋教練 | TKV 稀疏且有標註一致性問題；完整 RGB frames 與新 pipeline 成本較高 | manifest/splits/labels 已有；frames archive 不完整，現階段無法重抽 VideoMAE |

官方資料來源：

- [REHAB24-6 Zenodo](https://zenodo.org/records/13305826)
- [Fitness-AQA official repository](https://github.com/ParitoshParmar/Fitness-AQA)
- [EgoExo-Fitness official repository](https://github.com/iSEE-Laboratory/EgoExo-Fitness)
- [EgoExo-Fitness ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3057_ECCV_2024_paper.php)

## Stage A：REHAB24-6 representation validation

### 研究問題

修正 pooling 後，VideoMAE 是否能學到跨人物泛化的 correctness signal？

### 實驗矩陣

所有設定必須使用完全相同的 repetition samples、subject-wise LOSO folds、clip sampling、classifier capacity 與 random seeds。

| 設定 | 用途 |
| --- | --- |
| `legacy_first_token` VideoMAE-only | 重現舊 extraction，隔離 pooling 修正效果 |
| `mean_pool_fc_norm` VideoMAE-only | 驗證修正後的 frozen VideoMAE representation |
| Skeleton-only | 幾何／運動學對照與可達上限 |
| Skeleton + corrected VideoMAE | 檢查 RGB context 是否提供增量資訊 |
| Shuffled labels / permuted VideoMAE features | 建立 null control，排除 classifier 偶然擬合 |

既有舊基線為 VideoMAE LOSO balanced accuracy `0.536 ± 0.044`，但因使用錯誤 pooling，只能作為 legacy reference。Vicon skeleton LOSO 約為 `0.702 ± 0.078`。

### 評估方式

- Primary metric：subject-wise LOSO balanced accuracy。
- Secondary metrics：macro F1、recall、specificity。
- 報告每位受試者的 paired delta，不將 folds 或 seeds 錯當成獨立樣本。
- 同時依 exercise 與 camera/view 分層，確認提升不是單一動作或單一視角造成。
- 使用 participant-level paired bootstrap confidence interval，或明確報告每位受試者改善方向。

### Stage A 通過條件

進入 Stage B 前，至少應滿足：

1. Corrected VideoMAE 相對 legacy 的 paired balanced-accuracy delta 為正。
2. 改善出現在多數 held-out subjects，而非只由單一受試者拉高平均。
3. Corrected VideoMAE 明顯優於 shuffled/permuted null。
4. 效果不只存在於單一背景、camera 或 exercise。

若 corrected VideoMAE 在 REHAB24-6 仍接近隨機，應停止擴大到更昂貴的 EgoExo-Fitness，優先檢查 clip sampling、temporal aggregation、模型選擇或是否應放棄這條 feature branch。

## Stage B：Fitness-AQA task-specific validation

### 研究問題

即使 corrected VideoMAE 含有一般性的 correctness signal，它是否能改善 x-coach 關心的 squat form error detection？

### 公平基線

Fitness-AQA 既有結果：

- 舊 VideoMAE-only combined balanced accuracy：平均約 `0.555`，範圍 `0.532–0.584`。
- Train-only normalization 後的 pose-only combined balanced accuracy：`0.635 ± 0.010`。

因此 fusion 必須比較 `pose + corrected VideoMAE` 與 **normalized pose-only 0.635**，不能與未 normalization 的舊 pose baseline `0.581` 比較。

### 實驗矩陣

| 設定 | 用途 |
| --- | --- |
| Corrected VideoMAE-only | 確認 task-specific standalone signal |
| Normalized pose-only | 目前主要可部署 baseline |
| Regularized early fusion | 最簡單的 feature concat 對照 |
| Calibrated late fusion / gating | 降低弱 RGB branch 稀釋 pose signal 的風險 |
| Background-only / person-crop challenge | 檢查 VideoMAE 是否只學到場景與視角 |

所有 normalization 統計只能由 training fold 計算；checkpoint 與 threshold 只能由 validation set 選擇，test set 在設計鎖定後只評估一次。

### 建議保留門檻

將 VideoMAE 保留為預設 input 的建議條件：

- Fusion 相對 normalized pose-only 的平均 `Δ balanced accuracy ≥ +0.02`。
- Paired 95% confidence interval 下限大於 0。
- Recall 或 specificity 任一不得惡化超過 `0.03`。
- 改善在多數 seeds／resamples 中方向一致。
- Person-crop、view-stratified 或背景控制後仍保留效果。

若 VideoMAE-only 優於 null，但 fusion 未達門檻，應將其保留為研究或選配 branch，不作為 production default input。

## Stage C：EgoExo-Fitness product-value validation

只有 Stage A 與 Stage B 都通過後，才建議投入 EgoExo-Fitness。

EgoExo-Fitness 提供 technical-keypoint verification、quality score 與自然語言評語，適合把問題從單一 correctness classification 推進到：

1. 每個技術要點是否通過的 multi-label prediction。
2. Pose-only、VideoMAE-only 與 fusion 的 per-criterion AP/F1 比較。
3. Ego、exo 與多視角 fusion 的 paired comparison。
4. 以預測失敗的 technical keypoints 作為 RAG／LLM coaching feedback 的 grounding。

Primary metric 應使用 macro per-criterion average precision 或 F1，而不是整體 accuracy，避免大量稀疏或多數類 criterion 掩蓋真正的錯誤辨識能力。

## 結果解讀

| 結果 | 可支持的結論 | 後續決策 |
| --- | --- | --- |
| REHAB24-6 失敗 | 修正 pooling 後仍沒有可靠的跨人物 correctness signal | 停止擴大資料集，重新評估 VideoMAE branch |
| REHAB24-6 成功、Fitness-AQA 失敗 | VideoMAE 含一般動作品質訊號，但不適合目前 squat fault endpoint | 不作 production input；可保留研究用途 |
| REHAB24-6 與 Fitness-AQA 都成功，但 fusion 無增益 | VideoMAE 有獨立訊號，但 pose 已捕捉同一資訊 | 保留選配或診斷 branch，不作預設輸入 |
| 兩個資料集都成功且 fusion 穩定增益 | VideoMAE 提供 pose 之外的 RGB context | 進入 EgoExo-Fitness 細粒度驗證 |
| 效果只存在於背景／視角 | 模型學到 dataset shortcut，而非動作品質 | 判定為 no-go |

## 建議執行順序

1. 修正兩條 VideoMAE extractor，加入 `legacy_first_token` 與 `mean_pool_fc_norm` 模式。
2. 在 `.npz` 保存 provenance：model name、pooling、clip length、stride、num clips、Transformers version。
3. 加入 feature audit：覆蓋率、duplicate stem、維度、dtype、NaN/Inf 與 split 對應。
4. 在 REHAB24-6 各抽少量 samples 做 CPU smoke test。
5. 使用 GPU 產生完整 REHAB24-6 legacy 與 corrected features。
6. 執行 subject-wise LOSO 與 paired comparison。
7. Stage A 通過後，在 Fitness-AQA 重抽 corrected features並執行 pose/fusion ablation。
8. Stage B 通過後，補齊 EgoExo-Fitness frames並建立 multi-label TKV pipeline。

## 最終建議

現階段應將 **REHAB24-6 作為第一個 corrected-VideoMAE 驗證資料集**，因為它的 subject-wise 設計、相對平衡的 correctness labels、雙 RGB 視角與 GT skeleton 能提供最乾淨的 representation test。

Fitness-AQA 應保留為第二階段的 task-specific replication；EgoExo-Fitness 則作為通過前兩階段後的產品級、可解釋 feedback 驗證。

建議順序為：

> **REHAB24-6 corrected-pooling LOSO → Fitness-AQA squat-fault replication → EgoExo-Fitness interpretable coaching validation**
