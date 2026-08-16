# Letterbox 結果:REHAB24-6 VideoMAE framing 第一臂

對應事前登錄:[`rehab24_videomae_framing_validation_plan.md`](rehab24_videomae_framing_validation_plan.md)
(commit `32daa5a1`,**在任何 framing accuracy 產生前寫定**)。
執行日期:2026-08-15 / 16。本 note 只報告 P0 primary arm;`background_only`(P1)與
`person_crop`(P2)尚未抽取。

---

## 0. 一句話結論

`full_frame_letterbox` 確實修好了一個**實際存在、且已量到**的預處理缺陷——cam18 有
**100%** 的樣本被 center crop 切掉身體,中位數切掉身高的 **22.3%**,而且切的是**下緣**
(腳踝,正是深蹲判定的位置)——但**沒有換到可測得的 accuracy 增益**:
paired Δ = **+0.011 ± 0.056**(9 折,5/9 同向,exact p = 0.570,區間 [−0.084, +0.097])。
依 §8 事前判讀規則,這是 **practically small point estimate + undetermined**,
**不得宣稱 equivalence**,也不得寫成「完整身體沒有用」。

同時得到一個明確的正面答案:**Stage A 的 0.657 不是 cam18 裁腳造成的假象**。
換成完全不裁切的 framing 後仍是 0.661,兩者重疊。

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

## 8. 可以說 / 不能說

**可以說:**

- Stage A 的 **0.657 對「不裁腳的 framing」是穩健的**(0.6612 vs 0.6505),
  它不是 cam18 預處理缺陷造成的產物。這是本次最紮實的結論。
- cam18 的 center-crop 截斷是**真實且普遍**的(100% 樣本、中位切掉 22.3% 身高、切下緣),
  這一點由幾何直接量到,與模型無關。
- 在本設計的檢定力下,**修正該截斷帶來的 correctness 分類增益測不到**
  (點估計 +0.011,低於 0.02 實務帶)。

**不能說:**

- ❌「完整身體不影響 correctness 分類」——p = 0.570 是 undetermined,不是 equivalence;
  區間 [−0.084, +0.097] 寬到同時容得下實務上有意義的正負效果。
- ❌「letterbox 沒用,所以可以繼續用 `full_frame`」——(b) 的抵銷解釋同樣未被排除。
- ❌ 任何以 cam18 或單一動作的正向 Δ 作為結論的說法。
- ❌ 任何關於 0.657 是否含**背景/場景捷徑**的說法——那需要 `background_only`,尚未執行。

---

## 9. 下一步(依計畫優先序)

1. **`n_frames` duration-only control(P0,近乎免費)**:純 manifest 運算,不需抽取。
   `box_geometry` 已有 0.528 ± 0.028,但 duration 需單獨拆出來當 floor。
2. **`background_only`(P1)**:若要在論文中主張 0.66 是人體動作品質而非場景捷徑,
   這一臂是必要的。抽取成本與本次相同(~3.5h/臂,GPU 3 workers)。
3. **`person_crop`(P2)**:完成 body/background 完整拆解。

三個 arm 的 pixel transform、box resolver、gates 與 report runner **都已實作並測試完成**,
只差抽取:`--variant background_only --box-index ...` 即可,box index 已建好(130 支影片)。

---

## 10. 產物

```text
data/REHAB24-6/processed/videomae_boxes.json               # 固定人物框(每支影片一個)
data/REHAB24-6/processed/videomae_framing_geometry.json    # geometry gate
data/REHAB24-6/processed/videomae_framing_pairing.json     # pairing gate
data/REHAB24-6/processed/videomae_framing_audit.json       # feature audit
data/REHAB24-6/processed/videomae_framing_seed{42,7,1234}.json
data/REHAB24-6/processed/videomae_framing_summary.json     # 本 note 全部數字的來源
```

重跑指令與 gate 順序:`scripts/rehab24/README.md`「Framing arms」。
