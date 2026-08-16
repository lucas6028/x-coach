# REHAB24-6 VideoMAE framing 結果(P0 letterbox + P1 background control)

對應事前登錄:[`rehab24_videomae_framing_validation_plan.md`](rehab24_videomae_framing_validation_plan.md)
(commit `32daa5a1`,**在任何 framing accuracy 產生前寫定**)。
執行日期:2026-08-15 / 16。已完成 **P0 `full_frame_letterbox`** 與
**P1 `background_only` + `n_frames`**;`person_crop`(P2)尚未抽取。

---

## 0. 兩句話結論

**(P0 framing)** `full_frame_letterbox` 確實修好了一個**實際存在、且已量到**的預處理缺陷
——cam18 有 **100%** 的樣本被 center crop 切掉身體,中位數切掉身高的 **22.3%**,而且切的是
**下緣**(腳踝,正是深蹲判定的位置)——但**沒有換到可測得的 accuracy 增益**:
paired Δ = **+0.011 ± 0.056**(9 折,5/9 同向,exact p = 0.570,區間 [−0.084, +0.097])。
依 §8 事前判讀規則,這是 **practically small point estimate + undetermined**,
**不得宣稱 equivalence**,也不得寫成「完整身體沒有用」。
同時得到一個明確的正面答案:**Stage A 的 0.657 不是 cam18 裁腳造成的假象**——
換成完全不裁切的 framing 後仍是 0.661,兩者重疊。

**(P1 negative control)** **把人體移除後,訊號幾乎全部消失。**
`background_only` 只有 **0.5074 ± 0.0304**(高於 chance 僅 +0.007),與
`box_geometry`(0.5075)**完全打平**(Δ = −0.0000),而 `n_frames` 甚至在 chance 之下
(0.4786)。三個非人體 arm 全部貼在 0.5,對照 `full_frame` 的 0.6505。
**在這個資料集內,0.65 不是場景像素、不是片段長度、也不是人物框幾何。**
這與 Fitness-AQA 完全相反(該資料集 background-only 0.6238、clip length 0.6139),
理由也很清楚:REHAB24-6 只有一個實驗室與兩臺固定相機,「場景」無從辨識。

---

## 1. 事前登錄與實際執行的差異

只有一項,且與結果無關,在看到任何 accuracy 前就決定並記錄:

**Baseline 改為本機重抽,而非沿用既有的 `videomae_raw/`。** 計畫 §2 寫的是「保留既有
baseline」。既有 bundle 來自 Kaggle kernel、transformers **5.0.0**;本機為 **5.5.0**。
以相同程式碼在本機重跑 8 個樣本,相對 L2 差 **9.4e-04**(cosine 0.9999996)。數值很小,
但它只會落在**其中一臂**上,等於把環境差異放進 measured delta 裡。因此兩臂都在本機、
同一個 venv 重抽,並把封存的 Kaggle 臂當作**第三臂一起評分**做重現檢查(見 §5)。
它**不是** secondary comparison:重現檢查不是 framing 假說,不進 Holm family。

事後看,這個保守決定**沒有改變任何結論**(本機 0.6505 vs Kaggle 0.6506),但它把
「環境沒有影響」從假設變成量到的數字。

其餘條件全部照 §4.3 未動:模型、clip 長度 16 / stride 2 / 每 rep 4 clips、相同
`clip_starts`、`mean_pool_fc_norm` + `mean`、相同 fold 與 threshold objective、相同標籤。

---

## 2. 抽取執行紀錄

| 項目 | 值 |
| --- | --- |
| 環境 | 本機 `.venv-cuda`,GPU,torch 2.13.0+cu126 / transformers 5.5.0 |
| 平行度 | 3 workers(`--num-chunks 3`),manifest round-robin 切成互斥集合 |
| 耗時 | letterbox 3h45m、full_frame 3h29m(CPU 單 worker 推估需 ~5.5h/臂) |
| 產出 | 各 **2144 / 2144** bundles,worker exit status 全 0 |
| 裝置差異 | CPU vs GPU 相對 L2 **4.1e-06**(cosine 1.00000000)——比 library 版本差異小 ~230 倍 |

兩臂**必須同 venv**:provenance 記錄 transformers 版本但**不記錄裝置**,所以裝置混用
事後無法偵測。中途曾以 CPU 抽出 402 個 letterbox bundle,全數刪除重抽。

---

## 3. Gates(全部在看到 accuracy 前通過)

### 3.1 Geometry gate — 這一節本身就是一個結果

`data/REHAB24-6/processed/videomae_framing_geometry.json`,2144 樣本、130 支影片,
純幾何運算,不解碼、不載模型。

| | cam17 (1920×1080) | cam18 (1080×1920) |
| --- | ---: | ---: |
| `full_frame` 截斷率 | 75.3% | **100.0%** |
| 中位 **下緣**損失(佔身高) | 0.000 | **0.223** |
| 中位上緣損失 | 0.000 | 0.046 |
| `full_frame_letterbox` 截斷率 | **0.0%** | **0.0%** |
| 中位 body area:`full_frame` → `letterbox` | 0.682 → 0.245 | 0.950 → 0.415 |

計畫 §1.2 的疑慮完全成立,而且比預期更極端:cam18 **每一個**樣本都被切,切的是腳;
cam17 只掉左右兩側(上下損失皆為 0)。

### 3.1.1 「100% 截斷」到底是什麼意思——逐幀複核

上表的框是**整支影片的 union box**,而且外擴了 15% margin。因此「截斷」嚴格說是
「該影片中人體**曾經到達**的區域有一部分落在 crop window 外」,不等於「每一幀都被切」。
兩者差多少,直接用 mocap 2D 骨架逐幀量(crop window 為畫面正中央的 1080×1080):

| | cam17 | cam18 |
| --- | ---: | ---: |
| 外擴框截斷率(上表所用) | 75.3% | 100.0% |
| **原始 landmark 框**截斷率 | **43.8%** | **100.0%** |
| 有任一幀掉關節的影片數 | 35 / 65 | **65 / 65** |
| 每支影片掉關節的幀數比例(min / 中位 / max) | 0% / **0.8%** / 49.6% | **100% / 100% / 100%** |
| 最常被切掉的關節 | RHand_end 5%、RHand 3% | **LToeBase 99%、LFoot 98%、RToeBase 98%、RFoot 97%** |

**cam18 三種量法完全一致:每一支影片、幾乎每一幀,腳掌與腳趾都在模型看不到的地方。**
這是本次最硬的一個事實,且與模型無關。

**cam17 則必須修正說法。** 75.3% 是外擴框的數字,其中很大一部分只切到 margin;改用原始
landmark 框是 43.8%,而逐幀來看中位數只有 **0.8%** 的幀掉關節,掉的是**手**不是腳。
之所以有這個落差,是因為框取整支影片的 union——只要全片中有一幀手伸出畫面邊緣,該影片
的**所有** repetition 都會被記為截斷。因此 cam17 的真實情況是「偶爾在畫面邊緣切到手」,
不是「少了四分之一個人」。兩臺相機的對比比上表看起來更極端,不是更輕微。

**同一張表也記下 letterbox 的代價**:body area 中位數 0.786 → 0.290。這一欄是在看到
accuracy 前就寫下的,正是為了讓 §7 的判讀不能事後選擇。

### 3.2 Pairing gate

2144 / 2144 全數比對:sample id、split、`clip_starts`、rep metadata、dtype、finiteness
全同,provenance 各自單一且宣告不同 variant,**bit-identical 樣本 0 個**(REHAB24-6 兩個
相機都非正方形,容忍度為零)。

### 3.3 Feature audit

兩個 materialized dir 皆 PASS:2144 樣本、dim 768、float32、4 clips/sample、全 finite、
provenance 單一、split 與 manifest 相符。

---

## 4. Primary 結果

種子 42 / 7 / 1234;**先對每位受試者跨種子平均,再做 9 個 paired deltas**(§7.2),
不把 9×3 當成 27 個獨立樣本。

| Arm | balanced accuracy(9 折,排除 P10) | macro-F1 | recall | specificity |
| --- | ---: | ---: | ---: | ---: |
| `full_frame`(本機重抽) | 0.6505 ± 0.0585 | 0.6354 | 0.6656 | 0.6355 |
| **`full_frame_letterbox`** | **0.6612 ± 0.0567** | 0.6469 | 0.7103 | 0.6121 |

**Primary:`full_frame_letterbox − full_frame` = +0.0107 ± 0.0562**
5/9 受試者同向,範圍 [−0.084, +0.097],exact paired Wilcoxon **p = 0.5703**。

| 受試者 | full_frame | letterbox | Δ |
| --- | ---: | ---: | ---: |
| P1 | 0.6389 | 0.6119 | −0.0269 |
| P2 | 0.7341 | 0.6812 | −0.0529 |
| P3 | 0.5129 | 0.5797 | +0.0668 |
| P4 | 0.6785 | 0.5949 | −0.0836 |
| P5 | 0.6327 | 0.6497 | +0.0170 |
| P6 | 0.6989 | 0.7380 | +0.0392 |
| P7 | 0.6257 | 0.6804 | +0.0547 |
| P8 | 0.6571 | 0.7541 | +0.0970 |
| P9 | 0.6757 | 0.6609 | −0.0149 |

依 §8:`abs(Δ) < 0.02` 且區間寬 → **practically small point estimate / undetermined**。
n=9 檢定力低,**p ≥ 0.05 一律寫成 undetermined,不是「沒有差異」**。

唯一方向性的變化是 operating point:recall 0.666 → 0.710、specificity 0.636 → 0.612。
letterbox 讓模型更傾向判「正確」,但 balanced accuracy 幾乎不動。

---

## 5. 重現檢查(QC,非假說)

| Arm | seed 42 | seed 7 | seed 1234 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| `kaggle_full_frame`(封存) | 0.6573 | 0.6426 | 0.6520 | 0.6506 |
| `full_frame`(本機重抽) | 0.6594 | 0.6474 | 0.6448 | 0.6505 |

Kaggle 臂 seed 42 為 **0.6573**,與 Stage A 已發表的 **0.657** 相符;本機重抽平均
0.6505 對上 0.6506,差 **0.0001**。**本機重抽成立**,§1 的偏離不影響 delta 的分母。

---

## 6. 分層(機制檢查,不取代 primary)

### 6.1 Camera — 方向對,量級不足

| Camera | full_frame | letterbox | Δ | 同向 | exact p |
| --- | ---: | ---: | ---: | :---: | ---: |
| cam17 | 0.6419 | 0.6438 | +0.0020 | 5/9 | 0.8203 |
| cam18 | 0.6591 | 0.6786 | **+0.0194** | 5/9 | 0.4961 |

事前指定的機制是「增益應主要出現在 cam18」。**方向符合**(cam18 的 Δ 是 cam17 的 ~10 倍,
而 cam18 正是 100% 被裁腳的那一臺),但兩者都未達顯著,且 cam18 的 Δ 仍在 0.02 實務帶
邊緣。這只能算**與機制一致的弱訊號**,不足以宣稱機制成立。

### 6.2 Exercise — 六個動作全部回報,無任何 stratum 被排除

| 動作 | Δ | 同向 | exact p | §8 判讀 |
| --- | ---: | :---: | ---: | --- |
| arm abduction | +0.0360 | 4/9 | 0.3125 | size 夠但方向不一致 |
| arm VW | −0.0097 | 5/9 | 0.7344 | practically small |
| leg abduction | −0.0476 | 2/9 | 0.1641 | practical loss |
| leg lunge | +0.0512 | 4/7 | 0.4688 | practical gain |
| squats | +0.0207 | 4/9 | 0.8438 | size 夠但方向不一致 |
| table push-ups | +0.0158 | 4/7 | 1.0000 | practically small |

方向不一致、無一顯著,符合「n=9、六個 stratum、未校正」下的雜訊樣態。
**不得挑 leg lunge 或 squats 當結論**;§7.4 明文禁止以最好看的分層取代 overall。

### 6.3 Sensitivity:含 P10 的 10 折

Δ = +0.0396 ± 0.1018(6/10,p = 0.3223)。看似比 primary 大,但 P10 只有 16 個樣本,
balanced accuracy 在該折幾乎沒有意義——這正是它被列為 sensitivity 而非 primary 的原因。
**主結論以 9 折為準。**

---

## 7. 判讀:兩種解釋,本設計分不開

letterbox 修好了截斷(0% vs cam18 的 100%),卻沒換到 accuracy。可能是:

- **(a) 完整身體對這個分類任務不重要** —— 模型本來就沒在用腳踝資訊;或
- **(b) F2 的增益被 F3 的損失抵銷** —— 補成正方形後人體在 224×224 內縮小,
  body area 中位數 0.786 → 0.290(cam18 0.950 → 0.415)。完整但更小,對上不完整但更大。

**這兩者本設計無法分離**,而且這一點在 §3.1 的 geometry report 中已事先寫下,不是事後
找的理由。要分開需要第三種 framing(例如把 crop window 重新置中於人體、不縮放),
計畫並未登錄該臂。

---

## 7A. P1 Negative control:移除人體後還剩多少?

計畫 §2 把這一臂定義成「相對三個 floor 來讀」,而不是單看它的絕對分數。三個 floor 全部
在同一次 run、同樣的 fold 與 seed 下評分。

### 7A.1 六臂總表(seeds 42/7/1234,9 折,排除 P10)

| Arm | 內容 | bal_acc | 高於 chance |
| --- | --- | ---: | ---: |
| `full_frame_letterbox` | 人 + 場景,完整身體 | **0.6612 ± 0.0567** | +0.161 |
| `kaggle_full_frame` | (QC:封存 Kaggle 版) | 0.6506 ± 0.0548 | +0.151 |
| `full_frame` | 人 + 場景,cam18 裁腳 | 0.6505 ± 0.0585 | +0.151 |
| `box_geometry` | 12 個數字,無像素 | 0.5075 ± 0.0133 | +0.008 |
| **`background_only`** | **場景,人體被塗掉** | **0.5074 ± 0.0304** | **+0.007** |
| `n_frames` | 1 個數字(片段長度) | 0.4786 ± 0.0316 | −0.021 |

### 7A.2 §7.3 指定的兩個負控制比較(secondary,Holm 校正)

| 比較 | Δ | 同向 | raw p | Holm p | 判讀 |
| --- | ---: | :---: | ---: | ---: | --- |
| `background_only − n_frames` | +0.0289 | 5/9 | 0.2031 | 0.4062 | undetermined |
| `background_only − box_geometry` | **−0.0000** | 4/9 | 0.5703 | 0.5703 | 完全打平 |

### 7A.3 讀法

依 §8 的事前規則:「background-only ≤ n_frames/box_geometry → **未量到額外場景像素訊號**」。
本次結果比這更乾淨——`background_only` 不只沒有超過 floor,它**本身就在 chance 上**
(+0.007,std 0.030)。把人體塗掉之後,VideoMAE 從 0.65 掉到 0.51。

三個非人體 arm 互相一致,這件事本身也有意義:

- `background_only` 與 `box_geometry` 差 **0.0000**。`background_only` 唯一無法消除的洩漏
  就是那個矩形的位置與大小(`squat_video_variants` docstring 明載),而 `box_geometry`
  正是「只有矩形位置與大小」的 arm。兩者打平,等於直接量到**該洩漏也沒有攜帶訊號**。
- `n_frames` 落在 chance **之下**(0.4786)。片段長度在 REHAB24-6 完全不是捷徑,與
  Fitness-AQA 的 0.6139 形成強烈對比([[clip-length-shortcut]] 的結論**不外推**到本資料集)。

**與既有紀錄的一個小差異**:`box_geometry` 舊紀錄為 0.5316 ± 0.0274
(`correctness_loso_box_geometry.json`,單一 seed),本次三 seed 平均為 0.5075 ± 0.0133。
兩次的 `FoldConfig` 逐欄相同,差異純粹來自 seed。一個 chance 附近的 control 光換 seed 就能
移動 ~0.024,這正是 §7.2 要求跨 seed 平均的理由。兩個數字都在 chance 附近,結論不變。

### 7A.4 這一臂的內建限制(在看到分數前就成立)

- 遮罩覆蓋 cam17 畫面的 **45.2%**、cam18 的 **69.5%**。cam18 只剩約三成像素,
  所以「場景」在本設計中本來就所剩不多——這降低了偵測場景捷徑的敏感度。
- 框貼到畫面邊緣時(cam18 的 `x0 = 0`),水平內插退化成單邊拉伸。
- 因此「沒量到場景訊號」的正確讀法是**在這個遮罩定義下**,不是「背景絕對無訊號」。

---

## 8. 可以說 / 不能說

**可以說:**

- Stage A 的 **0.657 對「不裁腳的 framing」是穩健的**(0.6612 vs 0.6505),
  它不是 cam18 預處理缺陷造成的產物。這是本次最紮實的結論。
- cam18 的 center-crop 截斷是**真實且普遍**的:65/65 支影片、幾乎每一幀,腳掌與腳趾
  都在 crop window 外(中位切掉 22.3% 身高,切的是下緣)。三種獨立量法一致,與模型無關。
- cam17 的截斷**輕微**:逐幀中位數僅 0.8% 的幀掉關節,且掉的是手。因此本次的
  primary Δ 幾乎完全由 cam18 這一側的修正驅動——這也是 §6.1 只在 cam18 看到
  ~10 倍大 Δ 的原因。
- 在本設計的檢定力下,**修正該截斷帶來的 correctness 分類增益測不到**
  (點估計 +0.011,低於 0.02 實務帶)。

**不能說:**

- ❌「完整身體不影響 correctness 分類」——p = 0.570 是 undetermined,不是 equivalence;
  區間 [−0.084, +0.097] 寬到同時容得下實務上有意義的正負效果。
- ❌「letterbox 沒用,所以可以繼續用 `full_frame`」——(b) 的抵銷解釋同樣未被排除。
- ❌ 任何以 cam18 或單一動作的正向 Δ 作為結論的說法。

**P1 之後新增可以說的:**

- **在 REHAB24-6 內,0.65 的訊號不是背景像素、不是片段長度、也不是人物框幾何。**
  三個非人體 arm 全部落在 chance(0.479 / 0.507 / 0.508),對照 full_frame 的 0.6505。
  這是本計畫對「該分數是否為資料集捷徑」最直接的一次回答,且方向是正面的。
- `background_only` 與 `box_geometry` 打平(Δ = −0.0000)⇒ 遮罩矩形的位置/大小洩漏
  **本身不帶訊號**,這個已知限制被量到了,不必再當成未知風險。
- [[clip-length-shortcut]](Fitness-AQA 上 clip length = 0.6139)**不外推**到 REHAB24-6:
  這裡的 `n_frames` 在 chance 之下。

**P1 之後仍然不能說:**

- ❌「模型在新場景、新醫院、消費者手機影片上不會用背景捷徑」——本資料集只有一個實驗室、
  兩臺固定相機,**沒有場景變異可供辨識**,所以這裡測不到捷徑是預期中的,而不是外部效度的證據。
- ❌「背景完全不含資訊」——遮罩已覆蓋 cam18 的 69.5%,剩下的場景本來就很少。
- ❌ 以 `background_only` ≈ chance 反推「模型看的一定是動作品質」。它排除的是
  場景/長度/框幾何,**不排除人體外觀**(體型、衣著、個人特徵)。要分離那一層需要
  `person_crop`(P2)以及本計畫未涵蓋的 identity control。

---

## 9. 下一步

**已完成**:P0 `full_frame_letterbox`、P1 `background_only`、`n_frames` duration floor。

**剩下 `person_crop`(P2)**,計畫 §2 的 secondary:`person_crop − full_frame_letterbox`。
兩臂都保留完整身體,差別在有無場景,是本設計中最接近「背景有無」的對比。

需要注意的是,P1 的結果已經**削弱了 P2 的邊際價值**:既然 `background_only` 已落在 chance,
「場景帶有訊號」這條路基本上已經關閉,`person_crop` 主要能回答的變成
「**縮小並移除場景後,人體訊號是否仍保留**」——即 §8 表格最後兩列的 body-scale/aspect 效應,
而那正好也是 P0 無法與 F2 分離的那個因子。若要繼續,建議把它讀成**對 F3(body scale)的
探測**,而不是再一次的背景控制。

抽取成本與本次相同(~2h/臂,GPU 3 workers);pixel transform、box resolver、gates 與
report runner 都已就緒,`--variant person_crop --box-index ...` 即可。

---

## 10. 產物

```text
data/REHAB24-6/processed/videomae_boxes.json               # 固定人物框(每支影片一個)
data/REHAB24-6/processed/videomae_framing_geometry.json    # geometry gate
data/REHAB24-6/processed/videomae_framing_pairing.json     # pairing gate
data/REHAB24-6/processed/videomae_framing_audit.json       # feature audit
data/REHAB24-6/processed/videomae_framing_seed{42,7,1234}.json
data/REHAB24-6/processed/videomae_framing_summary.json     # P0 三臂
data/REHAB24-6/processed/n_frames_features/                # duration floor(由 box control 切出)
data/REHAB24-6/processed/videomae_framing_pairing_background_only.json
data/REHAB24-6/processed/videomae_framing_audit_background_only.json
data/REHAB24-6/processed/videomae_framing_p1_seed{42,7,1234}.json
data/REHAB24-6/processed/videomae_framing_p1_summary.json  # §7A 全部數字的來源(六臂)
```

重跑指令與 gate 順序:`scripts/rehab24/README.md`「Framing arms」。
