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
| **`full_frame_letterbox`** | **有** | **是** | 整張畫面 letterbox 成正方形,**不裁切** | **已實作** |
| **`person_crop_centercrop`** | **無** | **否** | 裁到人框,**不 letterbox**,讓 processor 自己 centre crop | **已實作** |
| `person_crop` | 無 | 是 | 裁到人框 + letterbox 補正方 | 已有 0.666 |
| ~~`person_crop_fixed_scale`~~ | — | — | **撤回,見下** | 不做 |

四列構成 **F1 × F2 的 2×2**。兩個新臂共用 `person_crop` 的同一個 expanded box,
差別只在 letterbox;`full_frame_letterbox` 不吃 box,由 `BOX_VARIANTS` 明確排除。

### `person_crop_fixed_scale` 撤回的理由

本計畫原本要用第五臂消掉尺度洩漏。實作前先量了 1623 支影片的框幾何,結論是這一臂
**對 99% 的影片是 no-op**:

| 量 | p10 | 中位數 | p90 |
| --- | --- | --- | --- |
| 框高 / letterbox 畫布邊長 | 1.000 | 1.000 | 1.000 |
| 框寬 / letterbox 畫布邊長(= 框長寬比)| 0.351 | 0.438 | 0.600 |

99.0%(1606/1623)的框是直立的,所以 `person_crop` 的畫布邊長就等於框高——**受試者
的高度早就被釘在畫布上了**,而 processor 一律縮放到 224,整體均勻縮放對模型不可見。
把框高正規化到固定比例,等於什麼都沒做。

替代方案(非等比壓成固定正方形)被否決:那會把長寬比換成身形變形,而變形是所有其他
臂都沒有的混淆,一旦掉分無法區分「幾何洩漏是真的」與「模型不喜歡被壓扁的人」——
正是 B1 存在要修的那種問題。

`person_crop` 之後殘存的幾何通道其實是**框的長寬比**,而它的上界已經是免費的:
`box_geometry`(零參數、無像素)0.578 vs `person_crop` 0.666,像素在框幾何之外
多給約 **+0.09**。這個界限直接寫進 B1 的結果,不需要再跑一臂。

### 事前量測:每一臂實際餵給模型的畫面

由 `src/video/variant_framing.py` + `scripts/video/run_variant_framing_report.py`
算出(純幾何,不解碼影片、不跑模型)。它重現了 §4.3 已發表的兩個數字——full-frame
身體面積 0.340、截斷率 37.8%——所以這張表和既有結果同源。

全部 1623 支:

| 臂 | 身體佔 224² (p10/中位/p90) | 框保留率 | 頭部裁掉 p90 | 腳部裁掉 p90 | 截斷率 | 與 full_frame 相同 |
| --- | --- | --- | --- | --- | --- | --- |
| `full_frame` | 0.182 / **0.340** / 0.552 | 1.000 | 3.8% | 11.8% | **37.8%** | — |
| `full_frame_letterbox` | 0.122 / **0.270** / 0.472 | 1.000 | 0% | 0% | 0% | **768** |
| `person_crop_centercrop` | 1.000 / 1.000 / 1.000 | **0.438** | 32.5% | 32.5% | 99.9% | 0 |
| `person_crop` | 0.350 / **0.438** / 0.600 | 1.000 | 0% | 0% | 0% | 0 |

只看 `full_frame` 真的被截的 613 支:

| 臂 | 身體佔 224² | 框保留率 | 頭部裁掉 p90 | 腳部裁掉 p90 |
| --- | --- | --- | --- | --- |
| `full_frame` | 0.402 | 0.880 | 10.0% | **13.4%** |
| `full_frame_letterbox` | 0.280 | 1.000 | 0% | 0% |

這張表在看到任何準確率之前就改變了三件事:

**一、`full_frame_letterbox` 的「相同數」預期是 768,不是 0。** 47.3% 的影片本來就是
正方形,它們的 full-frame 輸入從來沒被 centre crop 動過,這一臂沒有東西可以還原。
執行順序第 3 步的驗證門檻因此**事前登錄為 768**;要求 0 會被「修好」成一個壞掉的臂。
F2 的對比實際上由另外 855 支扛。

**二、F2 的操弄比預期弱,而且和 F3 反向。** 在真的被截的 613 支裡,full-frame 中位
只丟掉 12% 的框面積;換來的代價是身體面積從 0.402 掉到 0.280(相對 −30%)。所以
`full_frame_letterbox` 掉分**不能**讀成「F2 不重要」,它同樣符合「F3 的損失蓋過 F2 的
增益」。面積不是 F2 的正確嚴重度指標——腳部 p90 裁掉 13.4% 才是,因為深蹲深度判在
踝關節,而被裁掉的那一端偏偏是腳(11.8% vs 頭部 3.8%)。

**三、2×2 在 F3 上不平衡,交互作用非零是設計出來的。** 四格的身體面積分別是 0.270、
0.340、0.438、1.000,而 F2 的操弄強度在兩列差了約 4 倍(有背景那列丟 13% 身高,
無背景那列兩端各丟 32.5%)。因此**邊際主效果不是乾淨的估計**,必須連同這張表一起讀。

### 評估方式

- Primary metric 與 Stage B 相同:test split、val 選出的 threshold、balanced accuracy、
  5 seeds 平均。
- **Primary 讀出是單一因子的成對對比,不是邊際主效果。** 上面第三點的量測顯示邊際差
  同時混了 F3;最乾淨的一格是 F3 移動最小的那一格:

  | 對比 | 固定 | 變動 | ΔF3 | 地位 |
  | --- | --- | --- | --- | --- |
  | `full_frame_letterbox` − `full_frame` | 背景有 | 身體完整性 | −0.122 | **primary** |
  | `person_crop` − `person_crop_centercrop` | 背景無 | 身體完整性 | −0.562 | 佐證 |
  | `person_crop` − `full_frame_letterbox` | 身體完整 | 背景 | +0.168 | F1 的最佳讀數 |
  | `person_crop_centercrop` − `full_frame` | 身體被截 | 背景 | +0.660 | 混淆最重,僅報告 |

- 邊際主效果與交互作用仍然全部報告,但標註為受 F3 汙染。
- **Primary 對比在 613 支被截子集上成對計算**(事前登錄)。全體 1623 的邊際同時報告,
  但它被 768 支不可能移動的影片稀釋,不是主要讀數。
- 每一臂都必須報告 F3,因為它無法完全與 F1/F2 分離,只能量化後陳述。
- Paired 95% CI 用 Stage B 同一套 video-level bootstrap(`late_fusion.paired_bootstrap_delta`)。

### 判讀規則(在看到數字前寫定)

| 結果 | 可支持的結論 | 對 Stage B2 的影響 |
| --- | --- | --- |
| F2 主效果大、F1 主效果 ≈ 0 | +0.026 來自「終於看到完整的人」,與背景無關 | **B2 改為以 `full_frame_letterbox` 為 primary**;person-crop 的故事撤回 |
| F1 主效果大、F2 主效果 ≈ 0 | 背景確實是干擾 | B2 依原構想以 person-crop 為 primary |
| 兩者都大 | 兩個因素都真實 | B2 以 `person_crop` 為 primary,並把 `box_geometry` 的 0.578 當成必須超過的地板 |
| 兩者都 ≈ 0 | 0.666 與 0.640 的差距是雜訊 | **不進 B2**,結案 |
| `full_frame_letterbox` 掉分 | **不可讀為「F2 不重要」** | 先確認是否 F3(0.402→0.280)蓋過 F2;若是,B1 對 F2 不做結論 |

幾何洩漏不再另設一臂:`person_crop` 殘存的通道是框長寬比,其上界由既有的
`box_geometry` 0.578 給出,像素在其之上多給 +0.09。任何 fusion 增益都要在這個
地板之上陳述。

### 成本

兩個新臂 × 一個 Kaggle kernel,CPU fallback 下每個 3.4–5.7 h,可並行。分類器訓練
每臂約 5 分鐘。不需要新的上傳:box 由既有 `videos_person_crop/manifest.json` 提供,
新變體只是 `apply_variant` 的分支,在記憶體中套用。

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

1. ~~在 `apply_variant` 加新分支~~ **已完成**:`full_frame_letterbox` 與
   `person_crop_centercrop` 已實作並附測試,`person_crop_fixed_scale` 依上面的量測撤回。
   `full_frame_letterbox` 在 `apply_variant` 裡必須答在 `box is None` 提早返回**之前**
   ——它永遠拿不到 box,若落到那條分支就會回傳未變換的影片、產出一個與 `full_frame`
   位元相同卻沉默的臂,正是 51% 那件事的同一類失效。已有測試釘住。
2. ~~事前量測每一臂的取景~~ **已完成**,見上面的 F3 表。
3. 兩個 Kaggle kernel 並行抽取,`audit_videomae_features` 作為硬 gate。
4. **逐臂驗證變換確實生效**:與 full-frame 特徵比對。
   `person_crop_centercrop` 的 identical 數必須為 **0**;
   `full_frame_letterbox` 的 identical 數必須恰好是 **768**(已正方形的影片)。
5. 跑 B1 的四臂,先寫下 primary 對比(613 支子集),再寫邊際主效果與交互作用。
6. 依 B1 的判讀規則決定是否進 B2;**若進,先提交事前登錄再跑實驗**。
7. B2 通過才考慮 B3。

## 最終建議

**先做 B1,而且只做 B1。** 它是兩個 kernel、約半天的機器時間,卻決定了後面所有投入的
方向——可能把結論從「VideoMAE 需要移除背景」翻轉成「processor 的 centre crop 一直在
切掉受試者」,那會是一個**對所有現有 VideoMAE 數字都適用的預處理修正**,比 fusion
本身更有價值。

但實作後的量測也把這個期望值調低了:centre crop 只動到 37.8% 的影片,在這些影片上
中位只丟 12% 的框面積,而還原它要付出 30% 的相對縮小。所以「預處理 bug」這條線
現在的先驗比寫這份計畫時弱,B1 仍值得做,但要準備好它給出的是一個小效果或一個
被 F3 蓋掉的效果。

在 B1 之前不要投入 B2,也不要重新抽取任何以 person-crop 為基礎的特徵。
