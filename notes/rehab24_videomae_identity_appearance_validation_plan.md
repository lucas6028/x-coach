# REHAB24-6：VideoMAE 的 correctness 訊號能否在同一人物、同一錄影內成立？

> **事前登錄結論規則：先做身份／外觀控制，不直接進 fusion。Primary test 以同一人物、同一 exercise、同一錄影中的 correct/incorrect repetitions 排序能力為準；只有這個檢驗顯示穩定外觀無法解釋全部訊號，才允許進入 fusion。**
>
> 狀態：pre-registration draft；尚未讀取或計算本計畫的模型預測結果  
> 登錄日期：2026-08-23  
> Dataset：REHAB24-6  
> Primary visual arm：`full_frame_letterbox`，corrected VideoMAE pooling `mean_pool_fc_norm_mean`

## 1. 為什麼現在不是先做 fusion

[framing 結果](rehab24_videomae_framing_results.md)顯示，四個含人物的 VideoMAE arms 都落在相近區間：`full_frame_letterbox` balanced accuracy 為 `0.6612 ± 0.0567`，`person_crop` 為 `0.6603 ± 0.0467`，`full_frame` 為 `0.6505 ± 0.0585`，Kaggle full-frame 為 `0.6506 ± 0.0548`。相對地，`background_only` 為 `0.5074 ± 0.0304`，`box_geometry` 為 `0.5075 ± 0.0133`，`n_frames` 為 `0.4786 ± 0.0316`。

這些結果把訊號定位到人物區域，但還不能分辨 VideoMAE 使用的是：

- repetition 之間會改變的動作、姿勢或時間資訊；或
- 人物身份、體型、衣著等穩定外觀，配合各錄影的 correctness base rate。

目前 REHAB24-6 的 early fusion 也沒有顯示穩定互補性：RTMPose 從 `0.571` 到 `0.628`，MediaPipe 從 `0.634` 到 `0.661`，NLF 則從 `0.668` 到 `0.657`。因此下一個 claim-changing experiment 是排除身份／外觀捷徑，而不是再增加一組 fusion 組合。

## 2. REHAB24-6 能控制什麼，不能控制什麼

REHAB24-6 適合做強的**資料集內部控制**。同一 source session 內有同一人物、衣著、體型、背景、camera setup 與 exercise，但包含多個 correct 與 incorrect repetitions。若模型在這個範圍內仍能排序 correctness，任何在整段錄影中固定不變的身份／外觀特徵都不足以單獨解釋結果。

REHAB24-6 本身不能檢驗衣著、場地或不同日期的外部泛化，因為沒有同一人物換衣、換房間、換 session 的完整反事實設計。這一層需要日後重錄或外部資料集，不能由本計畫宣稱解決。

### 2.1 僅使用 labels 的 feasibility audit

以下只由 `data/REHAB24-6/processed/manifest.csv` 的 metadata 與 labels 計數，不含模型預測，因此不是 outcome analysis：

| 範圍 | rows | 人物 | 雙相機 sessions | mixed-label sessions | mixed-label rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部資料，含 P10 | 2,144 | 10 | 65 | 62 | 2,072 |
| Primary，排除 P10 | 2,128 | 9 | 64 | 61 | 2,056 |

Primary 的 61 個 mixed-label sessions 分布為 Ex1 `12`、Ex2 `12`、Ex3 `8`、Ex4 `12`、Ex5 `8`、Ex6 `9`。九位 primary subjects 都至少有五個 mixed-label sessions，因此可先在 session 內計算 AUC，再以 subject 為獨立統計單位。

## 3. 預先登錄的研究問題與假設

### 3.1 Primary question：同一錄影內是否仍有 correctness 排序訊號？

固定同一 source session 後，人物身份、體型、衣著、背景、exercise 與錄影條件不變。Primary question 是 `full_frame_letterbox` 的 out-of-fold probability 是否仍能把 correct repetitions 排在 incorrect repetitions 之前。

- **Observed statistic**：九位受試者的 subject-macro within-session ROC-AUC 平均值。
- **Null**：在每個 source session 內，以 repetition 為單位置換 correctness label；兩個 camera rows 使用同一個置換後 label，並保留該 session 的正負樣本數。
- **Alternative**：observed subject-macro AUC 高於這個 within-session permutation null。

這個檢驗針對「整段錄影中固定不變的外觀／身份是否足以解釋分類」；它不直接分辨動態、時間順序與單幀姿勢。

### 3.2 Confirmatory control：appearance-only 是否仍可預測 correctness？

建立 `canonical_frame_repeat` arm：對每個 source camera video，以不讀取 repetition labels 的固定規則選整段影片的時間中點 frame，將同一張 frame 重複成 16-frame clip，並把同一份 source-video embedding 指派給該影片的所有 repetitions。

這個 arm 保留人物身份、體型、衣著、背景、camera、exercise 與 session-level appearance，但移除 repetition-specific motion 與姿勢差異。它測量穩定外觀加上 session-level class prior 能取得多少 LOSO 表現。

此 arm 是 supporting negative control，不是 primary endpoint。低分只能表示這個具體 appearance-only construction 沒有顯示足夠訊號，不能當作「外觀完全沒有影響」的等價性證明。

## 4. Primary analysis 的精確統計單位

### 4.1 Sample、repetition、session 與 subject

- `sample`：單一 repetition × camera row。
- `repetition`：同一 `exercise_id + video_id + repetition_number` 的 cam17/cam18 pair。
- `session`：同一 `exercise_id + video_id`；這是同一人物、同一 exercise、同一錄影 pair。
- `subject`：獨立推論單位；primary 為 P1–P9，P10 只做 sensitivity analysis。

Camera rows、repetitions、sessions 與三個 random seeds 都不得當成獨立的 `n`。Primary inferential `n = 9` subjects。

### 4.2 OOF probabilities 與 aggregation

對 seeds `42, 7, 1234` 各自重跑既有 9-fold LOSO classifier，且只新增保存 test-fold out-of-fold probabilities：

1. 每個 seed、每個 sample 恰有一筆 OOF probability；模型訓練時不得看見 test subject。
2. 每個 repetition 先平均 cam17 與 cam18 probability，得到一個 dual-view repetition score。
3. 每個 seed 在每個 mixed-label session 計算 ROC-AUC。Single-class sessions 不能計算 AUC，依登錄規則排除，不改用其他 metric 補位。
4. 每個 session 的三個 seed AUC 取平均；seed 不視為額外樣本。
5. 每位 subject 對自己的 mixed-label session AUC 取等權平均。
6. 九位 subject scores 再取 mean、sample standard deviation 與範圍，得到 primary statistic。

不先跨 seeds 平均 raw probability，因為不同 seed 的 probability scale 可能不同。AUC 先在 seed 內計算，再聚合。

### 4.3 Primary permutation test

- 執行 `10,000` 次 Monte Carlo permutations，另加 observed statistic。
- 每次在每個 session 內置換 repetition labels；同一置換同時套用到兩個 cameras 與三個 seeds。
- 置換保留每個 session 的 class count、session membership 與所有 OOF scores。
- 每次依 4.2 的完整 hierarchy 重算 subject-macro statistic。
- One-sided Monte Carlo p-value 使用 `(1 + null >= observed 的次數) / (10,000 + 1)`。
- Primary 顯著水準固定為 `α = 0.05`，不因 secondary analyses 調整 primary 門檻。

同時報告九個 subject AUC、mean ± sample SD、range、高於 `0.5` 的 subject 數，以及以 subject 為 resampling unit 的 95% bootstrap interval。Bootstrap interval 是不確定性描述；permutation test 才是 primary inference。

### 4.4 Secondary analyses

以下不改變 primary conclusion，全部標為 secondary：

- cam17 與 cam18 分開計算同一套 within-session AUC；
- 各 exercise 分層 AUC；
- P10-inclusive sensitivity analysis；
- 九個 subject AUC 對 `0.5` 的 exact paired Wilcoxon sensitivity test；
- 全體 OOF balanced accuracy 與既有 framing result 的一致性檢查。

若 secondary p-values 成組呈現，報 raw 與 Holm-corrected p-values；不得挑選有利 camera 或 exercise 取代 overall primary result。

## 5. Appearance-only arm 的分析

`canonical_frame_repeat` 使用與 `full_frame_letterbox` 相同的 VideoMAE checkpoint、processor、pooling、classifier、LOSO splits、validation-subject selection、seeds 與 threshold objective。唯一改變是輸入 clip construction。

報告項目：

1. P1–P9 subject-macro balanced accuracy、macro-F1、recall 與 specificity；
2. 三 seeds 先在 subject 內平均，再以九位 subjects 報 mean ± sample SD；
3. 與 `full_frame_letterbox` 的九個 paired subject deltas；
4. exact paired Wilcoxon p-value 與每位 subject 方向；
5. P10-inclusive sensitivity analysis。

`canonical_frame_repeat` 的多個 repetition rows 共享完全相同的 source-video embedding，這是刻意的 negative control。推論單位仍是 subject，不能把重複 embeddings 當成增加樣本量。

## 6. 執行前必須通過的 gates

### 6.1 OOF completeness gate

- `full_frame_letterbox` 每個 seed 都必須涵蓋 primary manifest 的全部 `2,128` rows，sample IDs 不重複、不缺漏。
- OOF file 必須保存 `sample_id, person_id, exercise_id, video_id, repetition_number, camera, label, seed, test_subject, probability`。
- label、test subject 與 fold mapping 必須和 frozen manifest/split 邏輯一致。
- probabilities 必須全部 finite；任何 test-subject leakage 都使整個 experiment 無效。

### 6.2 Pairing 與 grouping gate

- 每個 repetition 必須恰有 cam17、cam18 各一列，兩列 labels 必須一致。
- Primary 應得到 64 sessions，其中 61 個 mixed-label sessions、2,056 mixed-label rows。
- 若實作後的 denominator 不吻合，先停止並修正 grouping；不得靜默 drop rows 後繼續分析。

### 6.3 Appearance construction gate

- Canonical frame index 僅由 source video frame count 決定，不得讀取 repetition boundaries 或 labels。
- 同一 source camera video 的所有 repetitions 必須有 bit-identical embedding。
- P1–P9 應只有 128 個 unique source camera videos；含 P10 sensitivity 應為 130 個。
- 任何 decode failure 都 fail closed；不得回退成 repetition-specific clip 或 full-frame baseline。

### 6.4 Frozen-analysis gate

本 note、grouping code、permutation seed 與 output schema 在讀取 OOF outcome 前先 commit 或保存 hash。看到 primary AUC 後，不得改 session definition、P10 規則、camera aggregation、seed aggregation、permutation sidedness 或 decision thresholds；任何變更都要在 results note 標為 plan deviation，並同時保留原登錄分析。

## 7. 預先鎖定的判讀與停止規則

`0.55 AUC` 是本專案的 practical decision threshold，不是普遍科學門檻，也不是等價界線。

| 結果模式 | 預先登錄判讀 | 下一步 |
| --- | --- | --- |
| Primary mean AUC `≥ 0.55`、至少 `6/9` subjects AUC `> 0.5`，且 permutation `p < 0.05` | 穩定身份、衣著、體型、背景與 session 條件不足以解釋全部 VideoMAE correctness 訊號；存在 repetition-varying visual signal | 完成 appearance-only control 後，才可進一個 preregistered fusion gate |
| Primary mean AUC `< 0.55`，或 permutation `p ≥ 0.05`，或少於 `6/9` subjects 高於 `0.5` | **undetermined**；不能主張模型在同一人物／錄影內辨識 correctness | 不跑 fusion；先完成 appearance-only 與錯誤／base-rate audit |
| Primary 接近 `0.5`，但 global balanced accuracy 仍約為既有 `0.6612` | global 成績主要可能來自人物／session 間差異或 class priors | 停止 fusion，優先重設資料或蒐集跨衣著／跨 session 資料 |
| Primary 通過，且 appearance-only 明顯弱於 full model | 兩個控制方向一致，支持 repetition-specific visual content | 允許 fusion gate |
| Primary 通過，但 appearance-only subject-macro BA `≥ 0.55` | repetition-specific signal 與 appearance/session shortcut 可能同時存在 | 在 fusion 前先做去捷徑／重新平衡；不得只報較好的 fusion 分數 |
| 兩項控制方向衝突或信賴區間很寬 | **undetermined** | 檢查 grouping、class priors 與 model calibration，不進 fusion |

不顯著結果一律寫成 **undetermined**，不得寫成「沒有差異」或「等價」。

## 8. Fusion 的條件式後續計畫

只有第 7 節的 primary pass 且 appearance-only 沒有觸發 shortcut gate，才開啟一個新的 fusion pre-registration。候選限制為目前最強 pose arm NLF 與 `full_frame_letterbox` 的 calibrated late fusion；不先掃大量 fusion weights。

Fusion pre-registration 至少要鎖定：validation-only calibration、單一 primary fusion rule、subject-level paired endpoint、相對於較強單模態 NLF 的 improvement gate，以及沒有 improvement 時停止。身份／外觀控制的結果不得因 fusion 成績較高而被覆寫。

## 9. 實作與 artifacts 計畫

建議新增：

- `src/rehab24/videomae_identity_control.py`：讀取或產生 OOF predictions、驗證 pair/session 結構、計算 hierarchical statistic 與 permutations。
- `scripts/rehab24/videomae_identity_control.py`：薄 CLI wrapper。
- `src/rehab24/videomae_appearance_only.py`：label-blind canonical-frame extraction 與 provenance。
- 對現有 classifier runner 做最小修改，保存每個 test sample 的 probability；不得改模型選擇或 evaluation recipe。

預計 artifacts：

```text
data/REHAB24-6/processed/videomae_identity_control/oof_seed42.csv
data/REHAB24-6/processed/videomae_identity_control/oof_seed7.csv
data/REHAB24-6/processed/videomae_identity_control/oof_seed1234.csv
data/REHAB24-6/processed/videomae_identity_control/within_session_summary.json
data/REHAB24-6/processed/videomae_identity_control/permutation_null.npz
data/REHAB24-6/processed/videomae_raw_canonical_frame_repeat/
data/REHAB24-6/processed/videomae_identity_control/appearance_only_summary.json
```

## 10. 這個實驗不支持什麼

即使 primary 通過，本實驗仍不支持以下主張：

- 不能證明模型使用時間順序；同一錄影內的單幀姿勢差異也可能產生 AUC。這需要 repetition-specific static-frame 與 temporal-shuffle controls。
- 不能證明模型對換衣、換場地、換日期或新相機具有 robustness。
- 不能證明身份／外觀完全沒有貢獻；它只檢驗整段 session 中固定的外觀是否足以解釋結果。
- 不能把低 appearance-only 分數解讀成與 chance 等價，除非另做有足夠 power 的 equivalence design。
- 不能把相關性結果寫成因果機制，也不能由 REHAB24-6 推廣到其他族群或復健情境。

## 11. Reproduce（實作後的預定介面）

先只保存 frozen `full_frame_letterbox` OOF predictions，再做 primary analysis；不要先查看 subgroup outcomes：

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py audit
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze --permutations 10000 --permutation-seed 20260823
```

Primary 完成後再建立 supporting control：

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py extract-appearance --variant canonical_frame_repeat
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py evaluate-appearance --seeds 42 7 1234
```

Results note 應寫到 `notes/rehab24_videomae_identity_appearance_results.md`，逐項列出 gates、任何 plan deviations、primary subject-level 結果、permutation inference、appearance-only 結果、fusion go/no-go，以及本計畫第 10 節的限制。

