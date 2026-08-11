# VideoMAE person-crop 驗證計畫(Stage B 後續)

## 目的

本計畫要回答的不是「VideoMAE 有沒有用」——Stage B 已經以事前登錄的方式回答過:
以 full-frame 特徵做 fusion 沒有增益,判定為不作 production 預設 input
([`videomae_stage_b_results.md`](videomae_stage_b_results.md) §3,**本計畫不更動該判定**)。

要回答的是 Stage B 結束時浮出、而原計畫沒有預期到的兩件事:

> 1. **person-crop 是所有臂裡最強的**(0.666),而且 pose + person-crop 的 late fusion
>    是整個計畫中**唯一一次超過兩個 branch 各自表現**的 fusion(0.682,post-hoc)。
> 2. **但 person-crop 同時改變了四件事**,「移除背景」只是其中一件,無法分離。

因此:**先分離混淆,再決定要不要重新事前登錄一輪 fusion。** 順序不能反——在不知道
+0.026 到底來自背景還是取景之前,任何以 person-crop 為主的設計都建立在未分離的混淆上。

## 前置事實(全部已實測,不是推論)

| 事實 | 值 | 來源 |
| --- | --- | --- |
| person-crop VideoMAE-only | 0.6662 ± 0.0117 | Stage B §2.6 |
| full-frame VideoMAE-only | 0.6401 ± 0.0211 | Stage B §2.6 |
| pose-only(分母)| 0.6504 ± 0.0118 | Stage B §2.1 |
| pose + person-crop late fusion(post-hoc)| 0.682 ± 0.016,Δ +0.032,CI [−0.018, +0.085] | Stage B §4.1 |
| **非正方形影片** | **53%**(480×600)| §4.3 |
| **full-frame 路徑下身體被 centre crop 截掉的影片** | **38%**,p90 截掉 14.6% | §4.3 |
| 身體佔 224² 輸入的面積 | full-frame 34.0% → person-crop 43.8%(線性 1.14×)| §4.3 |
| 人框幾何單獨(zero-parameter,無像素)| 0.5777 ± 0.0230 | Stage B §2.6 |
| 純錄影格式(解析度/長寬比)| 0.4979 ± 0.0157(隨機)| Stage B §2.6 |

## 混淆拆解

person-crop 相對 full-frame 同時動了四個因子:

| 因子 | full-frame | person-crop |
| --- | --- | --- |
| **F1 背景** | 有 | 無 |
| **F2 身體完整性** | 38% 的影片被 centre crop 截掉 | 完整(letterbox 使 centre crop 成為 no-op)|
| **F3 身體有效解析度** | 34.0% 面積 | 43.8% 面積 |
| **F4 幾何洩漏** | 人框位置/大小隨影片變動 | 人被置中,但 letterbox 比例仍編碼框的長寬比 |

---

## Stage B1:混淆分離(必做,且必須先做)

### 研究問題

person-crop 相對 full-frame 的 +0.026,是 **F1(移除背景)** 造成的,還是
**F2/F3(看到完整且更大的身體)** 造成的?

### 實驗矩陣

全部為 VideoMAE-only、`mean_pool_fc_norm` + `mean`、seeds 1–5、與 Stage B 完全相同的
split 與超參數。前兩列已有,後三列為新抽取。

| 臂 | F1 背景 | F2 完整身體 | 定義 | 狀態 |
| --- | --- | --- | --- | --- |
| `full_frame` | 有 | 否(38% 被截)| 原始畫面直接進 processor | 已有 0.640 |
| **`full_frame_letterbox`** | **有** | **是** | 整張畫面 letterbox 成正方形,**不裁切** | **新** |
| **`person_crop_centercrop`** | **無** | **否** | 裁到人框,**不 letterbox**,讓 processor 自己 centre crop | **新** |
| `person_crop` | 無 | 是 | 裁到人框 + letterbox 補正方 | 已有 0.666 |
| **`person_crop_fixed_scale`** | 無 | 是 | 裁到人框後,把**框高正規化到畫布的固定比例**(消掉 F3/F4 的尺度成分)| **新** |

前四列構成 **F1 × F2 的 2×2**,第五列處理 F3/F4。

### 評估方式

- Primary metric 與 Stage B 相同:test split、val 選出的 threshold、balanced accuracy、
  5 seeds 平均。
- **主效果以 2×2 的邊際差計算**,而不是只看兩個角落:
  - F1(背景)主效果 = 平均(無背景兩列)− 平均(有背景兩列)
  - F2(完整身體)主效果 = 平均(完整兩列)− 平均(被截兩列)
  - 交互作用一併報告。
- 每一臂都**必須報告 F3**(身體佔 224² 的面積中位數),因為它無法完全與 F1/F2 分離,
  只能量化後陳述。
- Paired 95% CI 用 Stage B 同一套 video-level bootstrap(`late_fusion.paired_bootstrap_delta`)。

### 判讀規則(在看到數字前寫定)

| 結果 | 可支持的結論 | 對 Stage B2 的影響 |
| --- | --- | --- |
| F2 主效果大、F1 主效果 ≈ 0 | +0.026 來自「終於看到完整的人」,與背景無關 | **B2 改為以 `full_frame_letterbox` 為 primary**;person-crop 的故事撤回 |
| F1 主效果大、F2 主效果 ≈ 0 | 背景確實是干擾 | B2 依原構想以 person-crop 為 primary |
| 兩者都大 | 兩個因素都真實 | B2 以 `person_crop_fixed_scale`(同時具備兩者且尺度受控)為 primary |
| 兩者都 ≈ 0 | 0.666 與 0.640 的差距是雜訊 | **不進 B2**,結案 |

`person_crop_fixed_scale` 若明顯低於 `person_crop`,代表原本的增益有一部分來自尺度/
幾何洩漏,該差額必須從任何 fusion 增益中扣掉後再陳述。

### 成本

三個新臂 × 一個 Kaggle kernel,CPU fallback 下每個 3.4–5.7 h,可並行(GPU session
上限 2,第三個設 `enable_gpu: false` 走 CPU queue)。分類器訓練每臂約 5 分鐘。
不需要新的上傳:box 由既有 manifest 提供,新的變體只是 `apply_variant` 的新分支。

---

## Stage B2:事前登錄的 fusion 重測(視 B1 結果決定是否進行)

### 研究問題

在 B1 選出的**乾淨定義**下,pose + VideoMAE 的 late fusion 是否真的超過 pose-only,
且達到計畫原本的保留門檻?

### 為什麼需要重測而不是沿用 §4.1 的 0.682

Stage B §4.1 的 +0.032 是**看到事前登錄的 primary 失敗之後才換的變體**,屬於
best-of-N。它可以作為投入的理由,不能作為結論。B2 的全部意義在於用一份新的事前登錄
把它重跑一次。

### 事前登錄項目(在跑任何 B2 實驗前寫定並提交)

| 項目 | 值 |
| --- | --- |
| Primary arm | calibrated late fusion(pose + B1 選出的 VideoMAE 變體)|
| 分母 | normalized pose-only,**沿用 Stage B 重抽值 0.650**,不再重抽 |
| Fusion 三個自由度 | 沿用 Stage B §0.1:Platt 校準、等權平均、seed 對 seed |
| Secondary | early fusion、VideoMAE-only |
| 保留門檻 | 沿用計畫原本的五條,不加碼不放寬 |

### 檢定力:244 支 test 影片不夠

Stage B 的 +0.032 對應 CI [−0.018, +0.085],寬度約 ±0.05。要讓下限大於 0,樣本量
大致需要 2–3 倍。三個選項,**B2 必須在事前登錄裡挑定一個**:

| 選項 | 做法 | 代價 |
| --- | --- | --- |
| **A. Repeated video-level splits(建議)** | 對 1623 支影片做 5 次 5-fold,每支影片都會在某一折成為 test | 偏離歷史固定 split,**必須同時報告固定 split 的數字**以保持與 0.650 / 0.555 可比 |
| B. 納入其他動作 | 加入 Fitness-AQA 的 OHP 與 BarbellRow | 換了任務,不再是同一個 endpoint |
| C. 不動 | 承認檢定力不足 | 條件 2 幾乎必然不通過 |

選 A 時,分母必須在**同一組 repeated splits** 上重算,不能拿固定 split 的 0.650 去比。

### 通過條件

沿用計畫原本的五條(Δ ≥ +0.02、CI 下限 > 0、護欄 ≤ 0.03、多數 seed 同向、控制組後
仍保留)。第五條在此階段的具體形式:**在 B1 的 F1/F2 分解下,增益必須歸屬於一個
已識別的因子**,而不是「換了一個變體就變好了」。

---

## Stage B3:泛化(視 B2 通過與否決定)

若 B2 通過,才值得問「這是不是只有深蹲成立」。Fitness-AQA 的 OHP 與 BarbellRow
本機已有(見記憶 `fitness-aqa-depth-finding`),可用同一條 pipeline 重跑。
若 B2 不通過,本計畫在 B2 結案。

---

## 結果解讀

| 結果 | 可支持的結論 | 後續決策 |
| --- | --- | --- |
| B1 顯示增益來自 F2(取景/截斷)| 這是**預處理缺陷的修正**,不是 RGB 表徵的價值 | 修 processor 的 centre crop(對所有臂),重跑 full-frame;VideoMAE 的地位不變 |
| B1 顯示增益來自 F1(背景)、B2 通過 | VideoMAE 在**移除場景後**確實提供 pose 之外的資訊 | 可考慮作為選配 input;進 B3 |
| B1 顯示增益來自 F1、B2 未通過 | 方向對但證據不足 | 保留為研究 branch,記錄所需樣本量 |
| B1 顯示兩者皆 ≈ 0 | 0.666 vs 0.640 是雜訊 | 結案,Stage B 的判定即為最終判定 |
| 任何一階段顯示增益隨 `box_geometry` 同步移動 | 仍是幾何洩漏而非影像內容 | 判定為 no-go |

## 建議執行順序

1. 在 `apply_variant` 加三個新分支(`full_frame_letterbox`、`person_crop_centercrop`、
   `person_crop_fixed_scale`),各自附測試。
2. 三個 Kaggle kernel 並行抽取,`audit_videomae_features` 作為硬 gate。
3. **逐臂驗證變換確實生效**:與 full-frame 特徵比對,identical 數必須為 0
   (唯一合法例外是人框覆蓋整張畫面的影片)。這是 Stage B 學到的教訓——
   有一次 51% 的控制組樣本其實未經變換。
4. 跑 B1 的 2×2,寫下 F1/F2 主效果與交互作用。
5. 依 B1 的判讀規則決定是否進 B2;**若進,先提交事前登錄再跑實驗**。
6. B2 通過才考慮 B3。

## 最終建議

**先做 B1,而且只做 B1。** 它是三個 kernel、約半天的機器時間,卻決定了後面所有投入的
方向——甚至可能把結論從「VideoMAE 需要移除背景」翻轉成「processor 的 centre crop
一直在切掉受試者」,那會是一個**對所有現有 VideoMAE 數字都適用的預處理修正**,
比 fusion 本身更有價值。

在 B1 之前不要投入 B2,也不要重新抽取任何以 person-crop 為基礎的特徵。
