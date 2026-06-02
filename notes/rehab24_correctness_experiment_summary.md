# REHAB24-6 Correctness 分類器實驗摘要

## 背景

REHAB24-6 pipeline 以 repetition 為單位，訓練一個輕量分類器判斷單次深蹲是否「正確」（correctness 二元分類）。
本次比較三種特徵來源在相同訓練/評估流程下的表現：

- **Skeleton only** —— 本地抽取的骨架特徵。
- **VideoMAE only** —— VideoMAE 影片層級時空特徵。
- **Skeleton + VideoMAE fuse** —— 兩者融合特徵。

評估流程沿用既有設定：固定 seed、最終測試前重新載入 validation 最佳 checkpoint、回報 always-positive / always-negative baseline，並以 **balanced_accuracy** 作為 threshold 選擇目標。
資料切分為 subject-wise（train / val / test 受試者不重疊），類別大致平衡（正類約 53%）。

## 測試集核心結果（以 validation 選出的 selected threshold 為準）

| 設定 | threshold | balanced_acc | accuracy | macro_f1 | recall | specificity | precision |
|---|---|---|---|---|---|---|---|
| **Skeleton only** | 0.354 | **0.723** | **0.729** | **0.724** | 0.808 | 0.638 | 0.719 |
| VideoMAE only | 0.600 | 0.544 | 0.530 | 0.517 | 0.342 | 0.746 | 0.607 |
| Skeleton + VideoMAE fuse | 0.188 | 0.683 | 0.695 | 0.679 | 0.861 | 0.504 | 0.666 |

## 過擬合程度（train@0.5 → test selected）

| 設定 | Train bal_acc (0.5) | Test bal_acc (selected) | 落差 |
|---|---|---|---|
| Skeleton only | 0.823 | 0.723 | **0.10** |
| VideoMAE only | 0.888 | 0.544 | **0.34** ⚠️ |
| Fuse | 0.903 | 0.683 | **0.22** ⚠️ |

## 主要發現

1. **Skeleton-only 是唯一能泛化、且三者最佳的設定。**
   測試 balanced_acc 0.723、macro_f1 0.724，train→test 落差最小（0.10）；驗證集挑出的閾值 0.354 也能順利轉移到測試集（val bal_acc 0.809 → test 0.723）。深蹲 correctness 本質由關節角度/運動學決定，骨架特徵直接命中訊號，結果合理。

2. **VideoMAE-only 基本沒學到 correctness，測試集近乎隨機。**
   訓練 bal_acc 0.888 但測試僅 0.544（≈ 隨機水準），落差 0.34，嚴重過擬合。在 subject-wise split 下，VideoMAE 很可能記住的是受試者外觀/身份而非動作正確性，換人即崩。閾值也不穩：驗證挑了 0.600，測試 recall 直接塌到 0.342。此特徵單獨使用不可信。

3. **融合反而拖累，不是加分。**
   Fuse 測試 bal_acc 0.683 < Skeleton 單獨的 0.723。加入不泛化的 VideoMAE 後，train bal_acc 衝到最高 0.903 但測試掉回 0.683（落差 0.22），典型「強特徵被弱特徵稀釋」。挑出的閾值 0.188 極低，把 recall 推到 0.861 但 specificity 只剩 0.504，幾乎在亂判負類。

4. **閾值選擇策略偏脆弱。**
   以 balanced_accuracy 在驗證集選閾值，對 Skeleton 能轉移，但對 VideoMAE / Fuse 在 val→test 分布偏移下失效；VideoMAE 的 selected threshold（test bal_acc 0.544）甚至比 fixed-0.5（0.559）更差。

## 建議的下一步

1. 以 **Skeleton-only 作為 correctness 主基線**，目前唯一站得住的結果。
2. **先解決 VideoMAE 過擬合再談融合**：加強正則化（dropout / weight decay）、降維（PCA / 線性探針）、確認 VideoMAE 特徵是否做了 subject-wise 正規化，並複查 split 是否真的 subject-disjoint。
3. **改進融合方式**：以 late fusion / gating 取代早期 concat，或先凍結 skeleton 分支再小幅引入 video 分支，避免壞特徵主導。
4. **報告口徑**：以 test selected-threshold 為準，並附上 train→test 落差佐證泛化，不要只報訓練或固定 0.5 的數字。

## 備註

- 原始 log 中三個區塊的 test selected-threshold 行各重複列印一次（數值一致），疑為 log 寫了兩遍，不影響結論。
- 資料來源：`result.md`（三組設定的完整 train/val/test metrics）。
