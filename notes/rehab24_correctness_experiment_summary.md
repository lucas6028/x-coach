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
| MediaPipe skeleton（單目估計，pseudo-3D） | 0.326 | 0.665 | 0.667 | 0.665 | 0.695 | 0.634 | 0.685 |
| RTMPose skeleton（單目估計，**2D-only**） | 0.443 | 0.562 | 0.568 | 0.560 | 0.658 | 0.466 | 0.585 |
| VideoMAE only | 0.600 | 0.544 | 0.530 | 0.517 | 0.342 | 0.746 | 0.607 |
| Skeleton + VideoMAE fuse | 0.188 | 0.683 | 0.695 | 0.679 | 0.861 | 0.504 | 0.666 |

## LOSO 交叉驗證（10 受試者輪流當 test，較可信的標尺）

固定切分只測 2 位受試者（P8、P9），單一數字高變異。改用 Leave-One-Subject-Out：每位受試者各當一次 test，其餘 9 人中取 1 人當 val（早停＋選閾值，保持 subject-disjoint），其餘 8 人訓練；每折重訓一個分類器。

| 特徵來源 | 固定切分 test bal_acc | **LOSO 9 折 mean±std** | LOSO pooled | 每折範圍 |
|---|---|---|---|---|
| **Vicon 動捕骨架** | 0.723 | **0.702 ± 0.078** | 0.722 | 0.569–0.816 |
| **MediaPipe 估計骨架（pseudo-3D）** | 0.665 | **0.633 ± 0.055** | 0.642 | 0.527–0.724 |
| **RTMPose 估計骨架（2D-only）** | 0.562 | **0.570 ± 0.051** | 0.580 | 0.509–0.667 |
| **HRNet-w48 估計骨架（2D-only）** | —（未跑固定切分） | **0.575 ± 0.075** | — | 0.414–0.647 |
| VideoMAE | 0.544 | **0.536 ± 0.044** | 0.563 | 0.504–0.653 |

（9 折＝排除只有 16 個 sample 的 P10；含 P10 的 10 折相近：Vicon 0.712±0.080、MediaPipe 0.645±0.063、RTMPose 0.583±0.062、VideoMAE 0.550±0.061。pooled＝把 10 折每位受試者各被 held-out 一次的預測串起來算單一數字。）

**LOSO 帶來的修正：**

1. **排序穩固**：Vicon > MediaPipe > VideoMAE 在每位受試者上都成立，不是固定切分剛好抽到 P8/P9 的假象。
2. **固定切分偏樂觀約 0.03–0.06**：Vicon 0.723→0.702、MediaPipe 0.665→0.633、VideoMAE 0.544→0.536，但都在 1 個 std 內。
3. **±0.08 的折間 std ＞ 想衝的進步幅度**：單一 2-受試者 test 的雜訊比「0.72→0.78」還大，單一切分量不出真假；**後續任何改動都應以 LOSO mean±std 判定**。
4. **P5 對所有特徵都接近隨機**（Vicon 0.569 / MediaPipe 0.527 / VideoMAE 0.507），連高保真 Vicon 都救不了 → 指向該受試者 label 模糊或動作非典型，是資料天花板而非模型容量問題，值得單獨檢視。
5. **Vicon→MediaPipe 差距穩定在 ~0.07**，與固定切分的 ~0.06 一致：單目估計取回約九成判別力的結論可信。
6. **更強的 2D backbone（RTMPose）但「只保留 2D」反而退步到 0.570 ± 0.051**，落在 MediaPipe（0.633）與 VideoMAE（0.536）之間，比 MediaPipe 低 ~0.06。**這不是 backbone 變差，而是丟掉了深度通道**：repo 的 RTMPose 路線是 2D-only（1188-d），MediaPipe 是 pseudo-3D（2970-d，含 BlazePose world depth）。深蹲 correctness 的核心訊號之一是垂直/深度位移（蹲多深），RTMPose 的 2D 精度增益補不回失去的 3D 結構。**推論：MediaPipe 的 world-depth 通道帶真訊號**；要讓更強 backbone 發揮，須搭配 2D→3D lifting 把深度補回（見 brief R5），或做「MediaPipe-2D-only vs RTMPose-2D-only」對照以隔離 backbone 與深度兩個因素。每折一致（P3/P7/P8 近隨機，與其他特徵同樣難），結論可信非雜訊。

7. **更強的 backbone（HRNet-w48）在「同為 2D-only」下與 RTMPose 打平，差距在雜訊內。** 用 **paired LOSO**（每折同一 train/val/test 切分＋同 seed 同時訓練兩組特徵，逐折配對抵銷受試者本身難度）直接對照 HRNet-2D vs RTMPose-2D：9 折（排除 P10）RTMPose 0.570 ± 0.051 vs HRNet 0.575 ± 0.075，**配對 Δ = +0.005 ± 0.071**，6/9 折為正，**Wilcoxon p = 0.734（undetermined）**；再排除資料天花板 P5 後 Δ = +0.015（p = 0.461），仍穩在雜訊內。**這正是第 6 點預期的結果**：把 backbone 從 RTMPose 換到更強的 HRNet-w48，只要仍停在 2D-only，就補不回丟掉的深度通道——兩者（~0.57）都低於 MediaPipe pseudo-3D（0.633）。結論回報為「within noise / undetermined」，**不是「HRNet 較差」**。HRNet 的價值在於它乾淨的 2D 是 **2D→3D lifting（R5）的好輸入**，而非當作 drop-in 的精度升級。

產物：`data/REHAB24-6/processed/correctness_loso_{vicon,mediapipe,rtmpose,videomae}.json`（含每折明細）、`correctness_rtmpose_fixed.json`（固定切分）、`correctness_loso_hrnet_vs_rtmpose.json`（HRNet-w48 vs RTMPose 配對對照）。執行：`python scripts/rehab24/loso_cross_validation.py --feature-dir <feature_dir>`；配對對照 `python scripts/rehab24/loso_hrnet_vs_rtmpose.py`。HRNet-w48 特徵在 Kaggle GPU 上抽取（見 `notebooks/rehab24_hrnet_colab.ipynb`，feature_dim=1188，2144 reps，與 RTMPose 2D-only 同特徵空間）。

## 各動作（6 exercises）拆解（pooled LOSO，每筆樣本各被 held-out 一次）

把 LOSO pooled 預測按 `exercise_id` 分桶，用每折 val 選出的閾值二值化後算各動作的 balanced_accuracy。`n` 為該動作全部樣本（含 2 機位，故約為 repetition 數 ×2）。

| Ex | 動作 | n | pos% | Vicon | MediaPipe (pseudo-3D) | RTMPose (2D-only) | HRNet-w48 (2D-only) | VideoMAE |
|---|---|---|---|---|---|---|---|---|
| Ex1 | arm abduction | 356 | 51% | **0.787** | 0.646 | 0.518 | 0.505 | 0.583 |
| Ex2 | arm VW | 416 | 45% | **0.703** | 0.557 | 0.548 | 0.500 | 0.534 |
| Ex3 | table push-ups | 214 | 49% | **0.657** | 0.620 | 0.519 | 0.450 | 0.514 |
| Ex4 | leg abduction | 420 | 57% | **0.733** | 0.686 | 0.688 | 0.631 | 0.606 |
| Ex5 | leg lunge | 348 | 45% | 0.498 | 0.510 | 0.473 | 0.539 | 0.445 |
| Ex6 | squats | 390 | 69% | 0.650 | **0.714** | 0.665 | 0.662 | 0.545 |

**各動作觀察：**

1. **Ex6 squats：MediaPipe（0.714）與 RTMPose（0.665）雙雙追平甚至超越 Vicon（0.650）。** 深蹲 correctness 主要由垂直/深度位移（蹲多深）決定，單目正好擅長——這對 x-coach 的「可部署單鏡頭深蹲教練」是最直接的正面證據：在真正的 squat 上，便宜的單目並不輸給昂貴動捕。
2. **Ex5 leg lunge：所有特徵都近隨機（Vicon 0.498 / MediaPipe 0.510 / RTMPose 0.473 / VideoMAE 0.445），連 Vicon 都救不了。** 這是**動作層級的資料天花板**，與「P5 受試者近隨機」是同型問題（label 模糊或動作判準不一致），且 lunge 很可能正是拖累 P5 的動作之一。值得單獨檢視 lunge 的標註。
3. **RTMPose 2D-only 的退步集中在上肢動作**：Ex1 arm abduction（0.518）、Ex3 table push-ups（0.519）、Ex2 arm VW（0.548）幾乎崩到隨機；但**腿部動作 Ex4 leg abduction（0.688）、Ex6 squats（0.665）與 MediaPipe 相當不掉分**。→ 修正前述 R2 結論：丟深度的傷害**高度依動作別**——上肢動作的判別訊號沿視軸（深度方向）走，2D 投影最吃虧；腿部/蹲類動作的訊號在影像平面內就看得到，2D-only 不受影響。
4. **Ex1 arm abduction：Vicon 一枝獨秀（0.787），單目大幅落後**（MediaPipe −0.14、RTMPose −0.27）。手臂外展的正確性靠細緻 3D 臂位，單目（尤其 2D-only）最難捕捉——這條最該靠 R5 lifting 或多視角補深度。
5. **VideoMAE 每個動作都 ≤ 0.61、多數近隨機**，與整體失敗一致，無單一動作例外可救。
6. **HRNet-w48（2D-only）逐動作幾乎處處貼齊或略低於 RTMPose，沒有藏在整體 wash 底下的單一動作增益。** 關鍵的 **squat（Ex6）打成平手：HRNet 0.662 ≈ RTMPose 0.665**——把 backbone 換到更強的 HRNet，在 x-coach 最在意的動作上完全沒有加分，兩者同樣卡在 MediaPipe pseudo-3D（0.714）之下，瓶頸是深度不是 backbone。上肢動作（Ex1–3）兩者都近隨機、HRNet 微低，但 n 小（214–416、含 2 機位）落在雜訊內。Ex4 leg abduction HRNet −0.057 是唯一非平凡的退步（但 0.688 本就是 RTMPose 唯一強的非蹲類格）；Ex5 leg lunge HRNet +0.066 落在**資料天花板動作**（連 Vicon 0.498 都近隨機）上，同屬雜訊。**逐動作確認了配對 LOSO 的整體結論：更強的 2D backbone 在任何單一動作上都沒有恢復判別訊號。**

**研究意涵**：correctness 不是單一難度——**動作別決定了單目能逼近 mocap 多少**。x-coach 聚焦的 squat（Ex6）恰是單目最強的一格；若產品擴及上肢動作（arm abduction/VW、push-ups），單目缺口會放大，深度補償（lifting/多視角）才有迫切性。後續所有改進應**同時報整體與分動作**，避免 squat 的好被 lunge 的天花板稀釋。

產物：`data/REHAB24-6/processed/correctness_loso_per_exercise.json`（4 來源）、`correctness_loso_per_exercise_hrnet.json`（HRNet-w48 vs RTMPose）。執行：`python scripts/rehab24/loso_per_exercise.py`（4 來源）；HRNet 對照 `python scripts/rehab24/loso_per_exercise.py --sources rtmpose=rtmpose_skeleton_features hrnet=hrnet_w48_skeleton_features --summary-output data/REHAB24-6/processed/correctness_loso_per_exercise_hrnet.json`。（RTMPose 欄位完全重現原表，確認折切分確定性、可逐欄並列。）

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
