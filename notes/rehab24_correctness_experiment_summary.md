# REHAB24-6 Correctness 分類器實驗摘要

## 背景

REHAB24-6 pipeline 以 repetition 為單位，訓練一個輕量分類器判斷單次深蹲是否「正確」（correctness 二元分類）。
本次比較三種特徵來源在相同訓練/評估流程下的表現：

- **Skeleton only** —— REHAB24-6 內建的 **Vicon 光學動捕**骨架特徵（高保真 ground truth）。
- **VideoMAE only** —— VideoMAE 影片層級時空特徵。
- **Skeleton + VideoMAE fuse** —— 兩者融合特徵。
- **MediaPipe skeleton（後續補做）** —— 從 RGB 影片用 **MediaPipe Pose 單目估計**的 33-landmark 骨架，套用與 Vicon 相同的幾何特徵流程。用來量化「便宜的單目估計骨架」離「昂貴的 Vicon 動捕」差多少。

評估流程沿用既有設定：固定 seed、最終測試前重新載入 validation 最佳 checkpoint、回報 always-positive / always-negative baseline，並以 **balanced_accuracy** 作為 threshold 選擇目標。
資料切分為 subject-wise（train / val / test 受試者不重疊），類別大致平衡（正類約 53%）。

## 測試集核心結果（以 validation 選出的 selected threshold 為準）

| 設定 | threshold | balanced_acc | accuracy | macro_f1 | recall | specificity | precision |
|---|---|---|---|---|---|---|---|
| **Skeleton only（Vicon mocap）** | 0.354 | **0.723** | **0.729** | **0.724** | 0.808 | 0.638 | 0.719 |
| MediaPipe skeleton（單目估計） | 0.326 | 0.665 | 0.667 | 0.665 | 0.695 | 0.634 | 0.685 |
| VideoMAE only | 0.600 | 0.544 | 0.530 | 0.517 | 0.342 | 0.746 | 0.607 |
| Skeleton + VideoMAE fuse | 0.188 | 0.683 | 0.695 | 0.679 | 0.861 | 0.504 | 0.666 |

## LOSO 交叉驗證（10 受試者輪流當 test，較可信的標尺）

固定切分只測 2 位受試者（P8、P9），單一數字高變異。改用 Leave-One-Subject-Out：每位受試者各當一次 test，其餘 9 人中取 1 人當 val（早停＋選閾值，保持 subject-disjoint），其餘 8 人訓練；每折重訓一個分類器。

| 特徵來源 | 固定切分 test bal_acc | **LOSO 9 折 mean±std** | LOSO pooled | 每折範圍 |
|---|---|---|---|---|
| **Vicon 動捕骨架** | 0.723 | **0.702 ± 0.078** | 0.722 | 0.569–0.816 |
| **MediaPipe 估計骨架** | 0.665 | **0.633 ± 0.055** | 0.642 | 0.527–0.724 |
| VideoMAE | 0.544 | **0.536 ± 0.044** | 0.563 | 0.504–0.653 |

（9 折＝排除只有 16 個 sample 的 P10；含 P10 的 10 折相近：Vicon 0.712±0.080、MediaPipe 0.645±0.063、VideoMAE 0.550±0.061。pooled＝把 10 折每位受試者各被 held-out 一次的預測串起來算單一數字。）

**LOSO 帶來的修正：**

1. **排序穩固**：Vicon > MediaPipe > VideoMAE 在每位受試者上都成立，不是固定切分剛好抽到 P8/P9 的假象。
2. **固定切分偏樂觀約 0.03–0.06**：Vicon 0.723→0.702、MediaPipe 0.665→0.633、VideoMAE 0.544→0.536，但都在 1 個 std 內。
3. **±0.08 的折間 std ＞ 想衝的進步幅度**：單一 2-受試者 test 的雜訊比「0.72→0.78」還大，單一切分量不出真假；**後續任何改動都應以 LOSO mean±std 判定**。
4. **P5 對所有特徵都接近隨機**（Vicon 0.569 / MediaPipe 0.527 / VideoMAE 0.507），連高保真 Vicon 都救不了 → 指向該受試者 label 模糊或動作非典型，是資料天花板而非模型容量問題，值得單獨檢視。
5. **Vicon→MediaPipe 差距穩定在 ~0.07**，與固定切分的 ~0.06 一致：單目估計取回約九成判別力的結論可信。

產物：`data/REHAB24-6/processed/correctness_loso_{vicon,mediapipe,videomae}.json`（含每折明細）。執行：`python scripts/rehab24/loso_cross_validation.py --feature-dir <feature_dir>`。

## 過擬合程度（train@0.5 → test selected，固定切分）

| 設定 | Train bal_acc (0.5) | Test bal_acc (selected) | 落差 |
|---|---|---|---|
| Skeleton only（Vicon mocap） | 0.823 | 0.723 | **0.10** |
| MediaPipe skeleton（單目估計） | 0.845 | 0.665 | **0.18** |
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

5. **單目 MediaPipe 估計骨架就能取回大部分 Vicon 的判別力，且遠勝 VideoMAE。**
   測試 bal_acc 0.665、macro_f1 0.665，只比昂貴的 Vicon 光學動捕（0.723）低約 **6 個百分點**，卻比 VideoMAE（0.544）高出 **12 點**，也與 Fuse（0.683）相當。這說明 correctness 的判別訊號主要落在**幾何/運動學結構**上，而便宜的單目姿態估計已能捕捉大部分——不一定要 Vicon 動捕。train→test 落差 0.18，介於 Vicon（0.10）與 VideoMAE（0.34）之間：單目估計引入的雜訊（抖動、遮擋、深度模糊）讓泛化略降，但骨架特徵本身仍穩健。閾值轉移也良好（val 0.779 → test 0.665）。
   - **研究意涵**：對「可部署的單鏡頭深蹲教練」這條線而言，這是正面證據——RGB→pose→規則/分類的幾何路線可行，VideoMAE 那種端到端影像嵌入在 subject-wise 下不可靠。Vicon 與 MediaPipe 之間那 6 點的差距，標示了「單目估計品質」的改進空間（多視角、時間平滑、更強的 pose backbone、3D-lifting）。
   - **注意**：MediaPipe 特徵維度 2970（3D world + 2D image 通道較多），Vicon 為 2340；分類器動態讀取維度，不影響比較，但兩者非同一特徵空間，差距解讀以「同流程、同 split 下的判別力」為準。

## 建議的下一步

1. 以 **Skeleton-only（Vicon）作為 correctness 上限基線**，**MediaPipe 單目估計骨架作為可部署基線**——兩者都站得住，VideoMAE 不可單獨採信。後續單目品質改進（多視角融合、時間平滑、更強 backbone、2D→3D lifting）以縮小那 6 點差距為目標。
2. **先解決 VideoMAE 過擬合再談融合**：加強正則化（dropout / weight decay）、降維（PCA / 線性探針）、確認 VideoMAE 特徵是否做了 subject-wise 正規化，並複查 split 是否真的 subject-disjoint。
3. **改進融合方式**：以 late fusion / gating 取代早期 concat，或先凍結 skeleton 分支再小幅引入 video 分支，避免壞特徵主導。
4. **報告口徑**：以 test selected-threshold 為準，並附上 train→test 落差佐證泛化，不要只報訓練或固定 0.5 的數字。

## 備註

- 原始 log 中三個區塊的 test selected-threshold 行各重複列印一次（數值一致），疑為 log 寫了兩遍，不影響結論。
- 資料來源：`result.md`（Vicon / VideoMAE / Fuse 三組設定的完整 train/val/test metrics）；MediaPipe 一組為後續以 `scripts/rehab24/extract_mediapipe_skeleton_features.py` 抽取（complexity=2、stride=1 全保真，130 支影片）後，用 `train_correctness_classifier.py --feature-dir .../mediapipe_skeleton_features` 訓練所得（feature_dim=2970，train=1434 / val=212 / test=498）。
