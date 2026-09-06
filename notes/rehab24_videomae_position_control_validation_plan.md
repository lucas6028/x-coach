# REHAB24-6：VideoMAE 的 within-session 訊號，有多少是「這一下發生在錄影的什麼位置」？

> **事前登錄結論規則：先把「錄影位置」這條路關掉，再談任何 fusion 或「模型看得懂動作」的宣稱。**
> Primary test 是：把 VideoMAE 特徵裡**能線性預測「類內位置」的方向**移除之後，同一段錄影內
> 把做對的 rep 排在做錯的前面的能力（within-session ROC-AUC）還剩多少。只有這個檢驗通過，
> [取景實驗](rehab24_videomae_framing_results.md)的「訊號來自畫面裡的人」和
> [身份控制](rehab24_videomae_identity_appearance_results.md)的 0.8741 才能繼續當結論用。
>
> 狀態：pre-registration draft；本計畫的 primary / mechanism / bound 三組分析**都還沒有跑**
> 登錄日期：2026-09-06
> Dataset：REHAB24-6，P1–P9 primary，P10 只做敏感度
> 特徵：`full_frame_letterbox`（整幀補成正方形、人和背景都在），corrected pooling `mean_pool_fc_norm_mean`
> 分類器、fold、seed（42 / 7 / 1234）、驗證受試者、門檻目標：全部沿用 `videomae_stage_a.run_arm`，一行不改

---

## 0. 為什麼是這個實驗，不是別的

身份控制的結果留下一個沒關上的門，而且這個門比原本以為的大。以下數字**已經看過了**
（2026-09-06 的探索分析，腳本在 §9 列出，全部只吃已存在的 out-of-fold 機率與 `Segmentation.csv`，
沒有重新訓練任何模型）。它們是本計畫的動機，**不是**本計畫的結果；本計畫的 confirmatory
分析全部是還沒跑的東西。

| 已觀察到的事 | 數字 | 出處 |
| --- | ---: | --- |
| 「這是第幾下」單獨當分數，within-session AUC | **0.1761**（反向讀 ≡ 0.8239） | 身份控制 note §7.1，事後控制 |
| 同一條規則、一個門檻、LOSO subject-macro balanced accuracy | **0.7309 ± 0.0481**，9/9 > 0.5 | 本次探索，`Segmentation.csv` only |
| 取景實驗最高分的 arm `full_frame_letterbox`，同一指標同一協定 | 0.6612 ± 0.0567 | 取景 note 結果表 |
| 位置平衡後（同一段錄影內「做對排前」與「做對排後」兩半各給 0.5 權重）的模型 AUC | **0.8497**（未平衡、同 34 段：0.8487；位置規則從 0.7309 掉到恆等 0.5000） | 本次探索，34 段可算 |
| 只取「做錯的排在做對的前面」那 984 對（位置規則在此 = 0.0000） | 模型 **0.8223** | 本次探索 |

這些數字合起來說了兩件互相拉扯的事：

1. **位置是一條真的捷徑，強到在取景實驗自己的指標上就蓋過頭條。** 一個整數、一個門檻，
   balanced accuracy 0.7309；整個 VideoMAE pipeline 0.6612。取景 note 那句「訊號來自畫面裡的人，
   不是背景」排除了背景，沒排除位置。
2. **在標籤有交錯的 34 段錄影上，位置平衡幾乎不動模型的分數**（0.8487 → 0.8497），
   位置指向相反的配對上模型仍有 0.8223。位置**不是**那個訊號的來源——至少在這 34 段上不是。

第 2 點不能推到全部資料，理由是結構性的：**61 段裡有 27 段是「前面全對、後面全錯」的單一分界**
（身份控制 note §7.1：交替區塊數中位數 3，27 段只有 2），那裡不存在「做對排在做錯後面」的配對，
位置平衡統計**構造上算不出來**。而正是這 27 段，位置幾乎完全決定標籤。

所以還缺的不是再一個配對統計，而是一個**能覆蓋全部 61 段的介入**：直接從特徵裡拿掉位置，
看排序能力剩多少。這就是本計畫的 primary。

## 1. 名詞說明

| 名詞 | 意思 |
| --- | --- |
| **rep / repetition** | 一次動作。REHAB24-6 每支影片包含多次 rep，每次有自己的做對／做錯標籤。 |
| **session** | 同一 `exercise_id + video_id`，也就是同一人、同一動作、同一段錄影（兩台相機各一支影片）。 |
| **位置（position）** | `Segmentation.csv` 的 `repetition_number`：這一下是該段錄影的第幾次 rep。 |
| **類內位置（within-class position）** | 在同一段錄影、同一個標籤的 rep 裡，這一下排第幾。例：某段錄影 12 個做對的 rep，類內位置就是 1–12 之間。本計畫要拿掉的是**這個**，不是原始位置——原因見 §3.1。 |
| **within-session AUC** | 只在同一段錄影內部比較：隨機抽一個做對的 rep 和一個做錯的 rep，模型給做對那個較高分的機率。整段錄影不變的東西（人、衣服、背景、相機）對它零貢獻。 |
| **subject-macro** | 每段錄影先算一個數，同一位受試者的錄影平均，再取九位受試者的平均。推論單位 n = 9。 |
| **LOSO** | leave-one-subject-out：輪流拿一位受試者當測試集，模型訓練時看不到那個人。 |
| **drift（漂移）** | 任何沿著錄影時間軸單調變化、又出現在畫面裡的東西：燈光、相機沉降、站位、疲勞、流汗、衣服位移。這些和位置同向，是模型「讀到位置」在像素層面的唯一可能載體。 |
| **probe（探針）** | 一個從凍結特徵預測某個變數的簡單線性模型，用來問「特徵裡有沒有這個資訊」。 |

## 2. 只用標籤與 metadata 算得出的可行性表（不含任何模型輸出）

| 範圍 | sessions | 標籤交錯（可做位置平衡） | 單一分界（位置平衡算不出） | (做對, 做錯) 配對 | 其中做對排後面 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary，P1–P9，mixed-label | 61 | 34 | 27 | 4,391 | 984 |

- 九位受試者在 34 段交錯錄影裡每人至少一段（本次探索的位置平衡統計 n_subjects = 9）。
- 類內位置在全部 61 段都有定義（單一分界的錄影裡，做對那一塊內部仍有先後）。
- 兩個 drift 代理變數的來源：`data/REHAB24-6/processed/box_geometry_features/`（已存在，12 維，
  含人物方框 `x0, y0, x1, y1, 面積`）；每個 rep 的平均亮度（**新算**，一次 CPU 解碼）。

## 3. 事前登錄的研究問題

### 3.1 Primary：拿掉「類內位置」方向之後，within-session 排序能力剩多少？

**為什麼是類內位置，不是原始位置。** 原始位置和標籤在錄影內幾乎是同一件事（AUC 0.1761）。
一個能預測原始位置的特徵方向，同時就是一個能預測標籤的方向；把它拿掉，就算模型看的完全是動作，
AUC 也會掉。那樣的下降無法判讀。類內位置把標籤先固定住：在做對的 rep 裡分早晚、在做錯的 rep 裡
分早晚。能預測類內位置的方向是 drift 的方向，**一階上與標籤正交**。拿掉它，掉的才是 drift。

**介入。** 在每一個 LOSO fold 內：

1. 只用**訓練受試者**的樣本，目標變數 = 類內位置歸一到 [0, 1]（該類只有一個 rep 的取 0.5）。
2. 特徵標準化後（沿用 `FoldConfig.normalize_features`），ridge 回歸得到方向 β，取單位向量 β̂。
3. 對該 fold 的**所有**樣本（訓練與測試）做 x' = x − (x·β̂) β̂。測試受試者的樣本從未參與 β 的擬合。
4. 把 x' 交給 `run_arm`，其餘一切不變。

Primary 移除 **k = 1** 個方向。k ∈ {4, 16}（逐次擬合、移除、再擬合）是 secondary，
用來看「線性 drift 子空間有多寬」。

**Observed statistic / null / alternative** 與身份控制計畫的主檢驗完全相同，這裡重述一次：
cam17 與 cam18 的機率先平均成一個 rep 分數；每個 seed 在每段 mixed-label 錄影算 ROC-AUC；
三個 seed 在錄影內平均；受試者對自己的錄影等權平均；九位受試者取 mean ± sample SD。
Null：每段錄影內部置換 rep 標籤、保留該段正負數、同一置換套到兩台相機與三顆 seed，
10,000 次，`p = (1 + #null ≥ observed) / 10,001`，α = 0.05。

### 3.2 Mechanism：VideoMAE 特徵到底編不編碼類內位置？

Primary 是介入；這一組回答「介入拿掉的東西原本存不存在」。

- LOSO ridge probe：訓練受試者上擬合「特徵 → 類內位置」，在測試受試者的每段錄影內算
  預測值與真實類內位置的 Spearman ρ（**在類內算**，做對與做錯各算一次再平均），
  subject-macro 中位數與平均。
- Null：每段錄影內部、每個標籤類內置換位置，10,000 次。雙尾。
- 同一個 probe 也跑在**殘差化後**的特徵上：這是 §6.3 的正控制 gate，不是結果。

### 3.3 Bound：看得見的 drift 有多強？

零參數，直接拿 metadata / 幾何量當分數，套 §3.1 完全相同的 within-session 統計：

| 代理變數 | 來源 | 抓的是 |
| --- | --- | --- |
| 方框 `x0`、`y0`（左上角） | box_geometry 已存在 | 站位漂移 |
| 方框面積 | box_geometry 已存在 | 前後距離漂移 |
| 每個 rep 的平均亮度 | 新算 | 燈光／曝光漂移 |

兩台相機的值先平均。**雙尾** p：反向預測和正向預測一樣算捷徑（身份控制 note §7.1 的教訓）。
「informative」的判準沿用該 note：|AUC − 0.5| > 0.15。

### 3.4 Secondary（不改變 primary 結論）

- 殘差化 arm（k = 1）的**位置平衡** within-session AUC，34 段。這是本次探索 0.8497 的複製，
  但用的是**新**訓練的分類器與殘差化特徵，所以不是同一個數字的重報。
- 殘差化 arm 的 LOSO subject-macro balanced accuracy，與 `full_frame_letterbox` 0.6612 ± 0.0567
  做九位受試者的 paired delta 與 exact Wilcoxon。這是取景實驗那句話能不能保住的直接檢驗。
- 殘差化 arm 在 **27 段單一分界錄影**上的 within-session AUC（只報，不與 34 段對比：兩堆錄影不同）。
- 用**原始位置**而非類內位置做殘差化的 naïve 版本：預期會過度移除（§3.1 的理由），報出來
  讓讀者看到差多少。
- k ∈ {4, 16}。
- P10-inclusive sensitivity。

多組 p 值報 raw 與 Holm-corrected；不得挑選有利的 k 或子集取代 primary。

## 4. 統計單位（與身份控制計畫相同）

sample = rep × camera；repetition = cam17/cam18 一對；session = 同一 `exercise_id + video_id`；
**subject 是唯一的推論單位，n = 9**。相機、rep、錄影、seed 都不是 n。

## 5. 事前登錄的預期（讓讀者能判斷結果意不意外）

本次探索給了一個 prior：在 34 段交錯錄影上位置平衡只動了 +0.001，位置反向的配對上仍有 0.8223，
兩個方向一致指向殘餘 **0.02–0.07**。若 27 段單一分界錄影的行為與 34 段相同，primary 應落在
**0.80–0.86**。若 primary 掉到 0.80 以下，代表單一分界的那 27 段裡模型對 drift 的依賴比
交錯錄影重——這正是本計畫要能偵測的情形。

## 6. 執行前必須通過的 gates

### 6.1 特徵與 OOF 完整性

- 三個 seed 各 2,128 筆 primary rows，sample_id 不重複不缺漏，機率全部 finite，
  `person_id == test_subject` 每一列成立（探索腳本已在既有 OOF 上驗過，新跑要再驗）。
- 每個 rep 恰有 cam17、cam18 各一列，標籤一致；mixed-label sessions = 61。

### 6.2 Fold 純度

- 每個 fold 的 β 只用該 fold 訓練受試者擬合；記錄每個 fold 的 β hash，十個 fold 必須**互不相同**
  （相同代表擬合時混進了全部資料）。
- 測試受試者的樣本在 β 擬合的輸入裡出現 → 整個實驗無效。

### 6.3 正控制：介入真的拿掉了東西

- §3.2 的 probe 跑在殘差化後（k = 1）的特徵上，subject-macro 類內 Spearman 中位數必須落在
  該 probe 的 null 95% 區間內。否則介入失敗，primary 不得判讀。
- 若 probe 在**殘差化前**的特徵上就已經落在 null 內（特徵根本不線性編碼類內位置），
  primary 介入是空操作；此時結論改由 §3.2 的 bound 承擔（見 §7 第四列），不得把 primary 的
  高分寫成「介入後仍成立」。

### 6.4 重現 gate

同一套新程式碼、`k = 0`（不殘差化）跑 `full_frame_letterbox`，within-session AUC 必須重現
**0.8741**（seed、fold、config 全固定，應到小數第四位相同），null 平均 0.5002 ± 0.0211。
另外，探索腳本的三個數字（位置平衡 0.8497、反向配對 0.8223、同 34 段未平衡 0.8487）
必須由正式模組在既有 OOF 上重現到小數第四位，否則模組有 bug。

### 6.5 Frozen-analysis

本 note、殘差化程式、probe 定義、置換種子（`20260906`）、判讀表在讀取任何新 OOF 之前 commit。
看到 primary 之後不得改 k、類內位置的定義、session 定義、相機／seed 聚合方式、單雙尾。
任何變更都在 results note 標為 plan deviation 並保留原登錄分析。

## 7. 預先鎖定的判讀表

`0.55` 沿用本專案的 practical threshold；`0.80` 來自 §5 的探索 prior（0.8741 − 0.07）。
兩個門檻都不是等價界線。

| 結果模式 | 事前登錄判讀 | 下一步 |
| --- | --- | --- |
| Primary（k=1）AUC **≥ 0.80**、≥ 6/9 受試者 > 0.5、p < 0.05；且 §6.3 兩個 probe 條件成立（殘差化前顯著、殘差化後在 null 內） | 線性 drift 方向不是 within-session 訊號的來源；位置這條路（線性）關閉。取景 note 的「來自畫面裡的人」與身份 note 的 0.8741 **保留**，並附「線性 drift 已排除、殘餘 ≤ 0.8741 − primary」的界 | 允許開 fusion pre-registration（沿用身份控制計畫的條件：只比 NLF 與 `full_frame_letterbox` 的 calibrated late fusion、validation-only calibration、單一 fusion rule、以 subject 為配對單位、相對 NLF 沒有改善就停） |
| Primary 落在 **[0.55, 0.80)**，p < 0.05 | 訊號有一部分與 drift 同方向；報「份額 = (0.8741 − primary) / (0.8741 − 0.5)」，不得寫成「主要來自動作」 | 不進 fusion；先做 k = 4, 16 看子空間寬度，再決定是否需要非線性移除 |
| Primary **< 0.55** 或 p ≥ 0.05 或 < 6/9 受試者 > 0.5 | **undetermined**：不能主張模型在同一段錄影內辨識動作品質；取景 note 的頭條句**撤回**，改寫為「訊號來自畫面裡的人，且與錄影位置無法分離」 | 停止 REHAB24-6 上的 VideoMAE 主張；下一步是資料層面（重錄成交錯協定）而非模型層面 |
| §3.2 probe 在殘差化前就落在 null 內 | 特徵不線性編碼類內位置；像素層面的位置路徑受此 bound。**Primary 不判讀**（介入為空操作） | 報 bound；非線性路徑仍開放，寫進「不支持」 |
| §3.3 任一 drift 代理 \|AUC − 0.5\| > 0.15 且雙尾 p < 0.05 | 指名那個 drift 是看得見的；它是 §3.1 拿掉的東西的候選載體 | 寫進機制段落；不改 primary 判讀 |
| Primary 通過但 secondary 的 27 段單一分界 AUC 明顯低於 34 段 | 只報，不推論（兩堆錄影不同） | 寫進「不支持」 |

p ≥ 0.05 一律寫成 **undetermined**，不寫「沒有差異」、不寫「等價」。

## 8. 這個實驗不支持什麼（就算 primary 通過）

- **只移除線性方向。** 非線性地編碼 drift 的路徑不在本計畫的檢驗範圍。k = 4, 16 只是線性子空間更寬，
  仍是線性。
- **類內位置 ≠ 位置。** 若某種 drift 在做對／做錯切換的瞬間跳變（例如教練喊停調整燈光），
  它與類內位置無關、與標籤完全共線，本設計看不到它。這種 drift 在 27 段單一分界的錄影裡
  無法與動作區分，任何分析都不行——這是資料的限制，只能重錄。
- **不檢驗 rep 內部的時間順序。** 「位置」是 rep 在錄影中的先後，不是 16 幀 clip 內的順序；
  temporal-shuffle 控制仍然沒做（身份控制 note 的「不支持」清單已列為未測，本計畫不覆蓋）。
- **不檢驗跨日、換衣、換場地的泛化。**
- **不能把 probe 落在 null 內讀成「特徵完全不含位置」**——那是等價主張，需要 power。
- **0.7309 vs 0.6612 不是對等比較。** 一參數規則 vs 完整 pipeline，n = 9、非配對、未檢定；
  它的功能是「證明捷徑強到足以蓋過頭條」，不是「位置贏 VideoMAE 0.07」。

## 9. 實作與 artifacts

新增（把本次探索的 scratchpad 腳本 `posctl3.py` / `posctl4.py` / `loso.py` 收進正式模組並加測試）：

- `src/rehab24/videomae_position_control.py`：類內位置目標、fold 內 ridge 殘差化、probe、
  drift 代理、位置平衡統計；`midranks / build_sessions / observed_statistic / permutation_null`
  直接從 `videomae_identity_control.py` 匯入，不重寫。
- `scripts/rehab24/videomae_position_control.py`：薄 CLI。
- `videomae_stage_a.run_arm` 加一個**可選**參數 `feature_transform_factory`（吃該 fold 的
  訓練 sample_ids，回傳作用在 (train, test) 特徵上的 callable），預設 None、行為不變；
  這是唯一動到既有 runner 的地方，現有測試必須全綠。
- `tests/rehab24/test_videomae_position_control.py`：合成資料上驗證
  (a) 殘差化後 probe 為 0，(b) 純位置捷徑的合成特徵殘差化後 AUC 掉到 0.5，
  (c) 純動作的合成特徵殘差化後 AUC 不變，(d) 位置平衡統計在位置規則上恆為 0.5。

預計 artifacts：

```text
data/REHAB24-6/processed/videomae_position_control/oof_k0_seed{42,7,1234}.csv      # 重現 gate
data/REHAB24-6/processed/videomae_position_control/oof_k1_seed{42,7,1234}.csv      # primary
data/REHAB24-6/processed/videomae_position_control/oof_k{4,16}_seed*.csv           # secondary
data/REHAB24-6/processed/videomae_position_control/oof_naive_k1_seed*.csv          # 原始位置版本
data/REHAB24-6/processed/videomae_position_control/fold_betas.json                 # 每 fold β hash
data/REHAB24-6/processed/videomae_position_control/probe_summary.json
data/REHAB24-6/processed/videomae_position_control/drift_proxies.json
data/REHAB24-6/processed/videomae_position_control/luminance_per_sample.csv
data/REHAB24-6/processed/videomae_position_control/within_session_summary.json
data/REHAB24-6/processed/videomae_position_control/permutation_null.npz
```

計算需求：全部 CPU、`.venv`。ridge 是 2,128 × 768；`run_arm` 的 MLP 在 CPU 上每 seed 十折
數分鐘；亮度解碼 128 支影片一次。不需要 GPU、不需要重抽 VideoMAE 特徵。

## 10. Reproduce（實作後的預定介面）

先跑 gate，再跑 primary，最後才看 secondary：

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py replicate-exploratory   # §6.4 後半
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 0 --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 0 --permutations 10000 --permutation-seed 20260906   # 必須 = 0.8741
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 0            # §6.3 前半
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 1 --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 1            # §6.3 後半
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 1 --permutations 10000 --permutation-seed 20260906   # primary
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py drift-proxies --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 4 16 --naive --seeds 42 7 1234
```

Results note 寫到 `notes/rehab24_videomae_position_control_results.md`，逐項列出 §6 五個 gate、
plan deviations、primary 九位受試者的數字、置換推論、probe 前後、drift 代理、§7 判讀表命中哪一列，
以及 §8 的限制。取景 note 與身份控制 note 依 §7 的結果**同時更新**：不是只改一份。
