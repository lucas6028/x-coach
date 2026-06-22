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

## 2D→3D Lifting 實驗：深度可否從單目恢復（R5）

第 6/7 點留下的假設：2D-only backbone（RTMPose/HRNet）追不上 MediaPipe pseudo-3D，是因為**丟了深度通道**；若用 2D→3D lifting 把深度補回，應能逼近含深度的設定。本實驗用三條互補路線直接檢驗「深度訊號能否從單目 2D 恢復」。

每條路線都訓練一個**不看 correctness 標籤**的 lifter（純幾何映射），再用 lifted-3D 重建骨架特徵跑同一套 LOSO，因此下游 LOSO 仍是乾淨的泛化估計。每條都做「**同一 2D 來源、只換 3D block**」的對照，隔離「lifted 3D 帶來的增益」。各路線 feature_dim 不同（關節佈局不同），故重點是**同路線內 lifted vs 2D 的 delta** 與**跨路線排序**，不是絕對值並列。

### 三條路線（9 折 LOSO，排除 P10；mean±std / pooled）

| 路線 | 設定 | feat_dim | 9 折 mean±std | pooled | lifter recon MSE |
|---|---|---|---|---|---|
| **S1 MediaPipe 空間** | mediapipe pseudo-3D（上界，既有） | 2970 | **0.633 ± 0.055** | 0.642 | — |
| | lifted3d（自訓 TCN，由 MP-2D 抬升） | 2970 | 0.621 ± 0.041 | 0.627 | 0.039 |
| | mp2d（MP image 2D-only，下界） | 1188 | 0.607 ± 0.059 | 0.601 | — |
| **S2 Vicon 空間** | Vicon 真實 3D mocap（上界，既有） | 2340 | **0.702 ± 0.078** | 0.722 | — |
| | lifted3d_vicon（自訓 TCN，由 Vicon-2D 抬升） | 2340 | 0.566 ± 0.050 | 0.595 | 0.170 |
| | vicon2d（Vicon 投影 2D-only，下界） | 936 | 0.583 ± 0.055 | 0.590 | — |
| **S3 預訓練 lifter** | vp3d_lifted（VideoPose3D 抬升 + COCO-2D） | 1530 | 0.635 ± 0.067 | 0.652 | — |
| | vp3d_2d（COCO-17 image 2D-only，下界） | 612 | 0.631 ± 0.068 | 0.622 | — |

### 核心結論：lifting 能補回「偽深度」，補不回「真深度」

1. **S1：lifted3d（0.621）落在 mp2d（0.607）與 mediapipe（0.633）之間**，約恢復一半（且整段落差本來就只有 0.026）。recon MSE 僅 0.039——因為 lifting 的監督目標 MediaPipe world 本身就是 BlazePose 的**單目學習估計**，其「深度」大半是 2D 可導出的，所以 lifter 能模仿大半。

2. **S2（決定性）：lifted3d_vicon（0.566）≈ vicon2d（0.583），甚至略低（在雜訊內），且都遠低於真實 Vicon 3D（0.702）。** 抬升從單一視角的 Vicon-2D**完全沒補回**那 +0.12 的真深度增益。recon MSE 0.170（是 S1 的 4 倍）——真實 mocap 深度是貨真價實的離面（out-of-plane）訊號，**單目 2D 推不出來**。

3. **S3（最強反證）：連在海量 H36M mocap 上預訓練的 VideoPose3D，lifted（0.635）也只比自己的 2D（0.631）高 +0.004（9 折，雜訊內；pooled +0.030）。** 升級 lifter 容量／資料量改變不了結論：**深度增益不是 lifter 不夠強，而是訊號根本不在單目 2D 裡。** lifted 3D 幾何健全（骨長 CV 1.5–3%、股骨≈脛骨），排除了「抬升壞掉」的可能。

4. **修正 R5 假設**：第 6/7 點期待「lifting 把深度補回就能逼近含深度設定」——**只對偽深度成立**。MediaPipe 之所以贏 2D-only backbone，主要不是它的 BlazePose world 帶了多少真深度（mp2d→mediapipe 只差 0.026），而是**它的 2D 本身就比 RTMPose/HRNet 的 2D 更能判別**（見第 5 點）。真深度只有 Vicon 那種量測級 3D 才有，而那條路單鏡頭補不回來。

5. **側發現：2D 品質才是單目這條線的主槓桿。** 各 2D-only 下界排序：vp3d_2d（COCO-17 MediaPipe 2D）**0.631** ≳ mp2d（MediaPipe-33 2D）0.607 > HRNet-2D 0.575 ≈ RTMPose-2D 0.570。MediaPipe 系的 2D 全面優於 RTMPose/HRNet 的 2D；且最好的 2D-only（0.631）已逼近 MediaPipe pseudo-3D（0.633）。**這正是「MediaPipe-2D-only vs RTMPose-2D-only」對照所缺的一塊**：MediaPipe 的優勢主要來自 2D 而非深度。

6. **產品意涵（x-coach 單鏡頭深蹲教練）**：靠 2D→3D lifting 把單目推向 mocap 水準是**死路**——無論自訓 TCN 或 SOTA 預訓練 lifter 都補不回真深度。能真正加訊號的只有 (a) **更好的 2D**（squat 上好 2D ≈ pseudo-3D 已夠用），或 (b) **真 3D 量測**（多視角／深度感測）。先前「需要 lifting 補深度」的方向應降權，「拿最好的單目 2D」應升權。

產物：`correctness_loso_{lifted3d,mp2d,lifted3d_vicon,vicon2d,vp3d_lifted,vp3d_2d}.json`（各含每折明細）、`lift_2d_to_3d_metrics.json`、`lift_2d_to_3d_vicon_metrics.json`、`lift_2d_to_3d_vp3d_metrics.json`。執行：
- S1：`python scripts/rehab24/lift_2d_to_3d.py` → LOSO 各 feature_dir。
- S2：`python scripts/rehab24/lift_2d_to_3d_vicon.py` → LOSO。
- S3：`python scripts/rehab24/lift_2d_to_3d_pretrained.py`（需先 clone VideoPose3D + 下載 `pretrained_h36m_detectron_coco.bin` 至 `third_party/VideoPose3D`）→ LOSO。
S1/S2 在本機 CPU 訓練（各早停 ~47/75 epoch）；S3 VideoPose3D 為輕量 TCN，本機 CPU 推論 130 支影片約數分鐘。**MotionBERT（重型 transformer）尚未跑**——預期與 VideoPose3D 同結論，若要確認需上 Kaggle GPU。

## 直接 image→3D（NLF）：lifting 補不回的深度，直接回歸補得回一部分（promising 但 n=9 未達顯著）

lifting 實驗結論是「單目 **2D→3D lifting** 補不回真深度」。但 lifting 只吃 2D 幾何；**直接 image→3D**（NLF, Neural Localizer Fields, NeurIPS'24）直接從像素回歸 metric 3D + SMPL，能用到 appearance／shading／人體先驗——這些是 lifter 看不到的。問題：直接回歸能否補回 lifting 補不回的深度、逼近 mocap？

**方法**：NLF `nlf_l_multi.torchscript` 在 Kaggle P100 跑 `detect_smpl_batched`（**半解析度 960×540**，~104 ms/frame，與全解析度逐關節差僅 **1.8 mm**；2 個 kernel 並行各 ~3.5h，130 支影片×雙鏡頭，每個 rep 幀 **100% 偵測**，取面積最大框＝病人）→ SMPL-24 3D（mm，相機座標）＋ 2D。套用與其他來源**相同**的 normalize（root＝骨盆、肩寬/髖寬尺度）→velocity→time-summary 特徵管線與**相同 9 折 LOSO**，只換 3D 來源。

### 結果（9 折 LOSO，排除 P10；mean±std / pooled）

| 設定 | dim | 9 折 bal_acc | pooled |
|---|---|---|---|
| **NLF parametric 3d2d** | 2160 | **0.668 ± 0.044** | **0.692** |
| NLF parametric 3d-only | 1296 | 0.652 ± 0.046 | 0.678 |
| NLF nonparam 3d2d | 2160 | 0.646 ± 0.069 | 0.678 |
| (對照) MediaPipe pseudo-3D | 2970 | 0.634 | 0.642 |
| (對照) vp3d_lifted | 1530 | 0.635 | 0.652 |
| (對照) vp3d_2d（最佳 2D-only） | 612 | 0.631 | 0.622 |
| (天花板) Vicon mocap | — | ~0.702 | — |

**配對 LOSO（同折同 seed，逐折配對抵銷受試者難度）**：
- NLF vs MediaPipe：d = **+0.034 ± 0.063**，6/9 折正，**Wilcoxon p = 0.203（undetermined）**；減 P5 後 d=+0.024（p≈0.38）。
- NLF vs vp3d_lifted：d = **+0.033 ± 0.084**，6/9 折正，**p = 0.301（undetermined）**；減 P5 後 d=+0.023（p≈0.55）。

### 核心結論

1. **點估計上 NLF 是目前最強的單目路線**（0.668）：比最佳單目（pseudo-3D 0.634 / vp3d_lifted 0.635）高 +0.033–0.034、pooled 高 ~+0.05，把「單目→mocap」缺口（0.634→~0.702）補回**約一半**。
2. **但配對顯著性 undetermined**（Wilcoxon p=0.20–0.30）：REHAB24-6 只有 **9 個可用受試者**、跨受試者變異大（折 delta −0.11~+0.17），+0.034 的均值撐不到顯著。**這不是雜訊歸零的 wash**——對比 hrnet-vs-rtmpose（d=+0.005、p=0.73），NLF 是**一致為正、約 7 倍大**的正向趨勢，且在 baseline 崩到近隨機的折（P8 +0.15、P5 +0.11）贏最多，只是 n=9 撐不到統計顯著。
3. **增益來自回歸的「深度」，不只是 NLF 的 2D 較乾淨**：3D-only 對照（**0.652**）就已超過最佳單目（0.634）。且 parametric（0.668）> nonparam（0.646）→ SMPL 身體先驗（時間穩定、解剖約束）有額外幫助。
4. **修正 lifting 結論**：「lifting 是死路」仍成立（純 2D 幾何補不回深度）；但**直接 image→3D ≠ lifting**——它靠像素 appearance＋學到的人體先驗，補回 lifting 補不回的一部分真深度。**深度訊號「在影像裡」，只是「不在 2D 幾何裡」**，所以判別關鍵是「用什麼把深度從像素取出」，而非「能不能取出」。
5. **各動作拆解佐證「補深度」機制**（pooled LOSO per exercise）：NLF **在 6 個動作全數 ≥ MediaPipe**（一致方向，補上 subject 層 n=9 的不足），且**增益集中在上肢／離面深度大的動作**——table push-ups **+0.106**、arm VW **+0.076**、arm abduction +0.033（判別關鍵在手臂/軀幹的離面深度，單目 2D 最難）；偏面內的下肢動作增益較小（leg abduction +0.013、lunge +0.019、squats +0.024，單目 2D 本就接近夠用）。對比 vp3d_lifted：NLF 在 arm VW 大勝（0.635 vs 0.489，+0.146），跨動作更穩健（lifted 在複雜手臂動作崩盤）。**「增益正落在最該補深度的動作」這個分布，是 NLF 補回真深度的機制證據**，與 lifting 實驗第 71 點（arm abduction 單目缺口最大）呼應。
6. **產品意涵（x-coach 單鏡頭教練）**：直接 image→3D（NLF 這類）是比 lifting 更有前景的補深度路線——尤其產品若擴及上肢動作，缺口大、NLF 補得多。但 subject 層增益尚未達統計顯著，須以 (a) **第二個直接 3D 模型（HMR2.0/4DHumans）**佐證收斂、(b) 更多受試者、或 (c) 真 3D 量測 來確認。

產物：`correctness_loso_nlf_{parametric_3d2d,parametric_3d,nonparam_3d2d}.json`、配對 `correctness_loso_nlf_vs_{mediapipe,vp3dlift}.json`、`correctness_loso_per_exercise_nlf.json`。重現：Kaggle 抽取 → `python -m src.rehab24.nlf_skeleton_features --raw-dir data/REHAB24-6/processed/nlf_raw3d` → `python -m src.rehab24.loso_cross_validation --feature-dir …nlf_parametric_3d2d_skeleton_features`。原始每影片 3D 存 `data/REHAB24-6/processed/nlf_raw3d/`（半解析度抽取），特徵管線細節見 `src/rehab24/nlf_skeleton_features.py`（7 個單元測試）。

## 第二個直接 image→3D 模型交叉驗證（HMR2.0 / 4DHumans）

NLF 的 subject 層增益（+0.034）方向一致但 n=9 撐不到顯著，第 6 點留下的待辦是「拿**第二個獨立的直接 image→3D 模型**佐證收斂」。選 HMR2.0（4DHumans，ViT-based SMPL 回歸）——與 NLF 架構不同（NLF 是 localizer-field 點回歸、HMR2.0 是參數化 SMPL transformer），若兩者在同樣的動作上同向補回深度，就是機制證據而非單一模型的偶然。

**抽取差異（關鍵 confound）**：HMR2.0 用內建偵測器逐幀偵測＋取最大框，但 REHAB24-6 部分機位/幀偵測失敗，**每 rep 平均僅 ~75% 幀有偵測**（NLF 是 100%），缺幀以線性插補補回。這 25% 插補幀引入的雜訊會系統性壓低 HMR——比較時須記住 HMR 吃了 NLF 沒有的 localization 雜訊虧。

### LOSO 結果（9 折，排除 P10；mean±std / pooled）

| 設定 | dim | 9 折 bal_acc | pooled |
|---|---|---|---|
| NLF parametric 3d-only（對照，乾淨偵測 100%） | 1296 | 0.652 ± 0.046 | 0.678 |
| **HMR parametric 3d-only**（公平數字） | 1296 | **0.643 ± 0.058** | 0.635 |
| HMR parametric 3d2d（2D 受 crop-plane 雜訊污染） | 2160 | 0.623 ± 0.070 | 0.645 |
| (對照) MediaPipe pseudo-3D | 2970 | 0.634 | 0.642 |

**配對 LOSO（HMR-3d-only vs MediaPipe，同折同 seed）**：Δ = **+0.009 ± 0.047**，5/9 折正，**Wilcoxon p = 0.734（undetermined）**；**再排除 P5 後 Δ 翻成 −0.002（p = 0.945）**——HMR 那點微小優勢幾乎全來自資料天花板受試者 P5（該折 +0.092），抽掉就歸零。

### 核心結論：方向一致但被偵測率與弱 localization 雙重 confound，**未達 NLF 期望的乾淨佐證**

1. **HMR 的 2D block 反而扣分（3d2d 0.623 < 3d-only 0.643）**，與 NLF 相反（NLF 3d2d 0.668 > 3d-only 0.652）。差別在 2D 來源品質：NLF 給的是乾淨的影像座標 2D，HMR 只有 crop-plane 重投影 2D（受偵測框抖動污染）。故 **HMR 的公平數字是 3d-only = 0.643**，後續比較一律用 3d-only 對 3d-only。

2. **整體上 HMR 僅 +0.009 over MediaPipe（p=0.73），且去掉 P5 即歸零**——遠弱於 NLF（+0.034，去 P5 仍 +0.024）。這**不是雜訊歸零的 wash 也不是反證**，但**確實不是當初想要的乾淨收斂**：兩個直接 3D 模型同向（都 ≥ baseline），但 HMR 的訊號被 25% 插補幀＋弱 2D 壓到雜訊內。

3. **唯一穩固的交叉佐證：table push-ups。** 逐動作（pooled LOSO，3d-only 對照）兩個獨立直接-3D 模型在**離面深度最大的上肢撐體動作**上都大幅補分——**HMR +0.096、NLF +0.119**，方向與幅度都收斂。架構迥異的兩個 image→3D 回歸器在同一個最該補深度的動作上各自獨立 +~0.10，**這是「直接回歸補回離面深度」機制的最強單點證據**，比 subject 層的均值更難用偶然解釋。

   | Ex | 動作 | n | MediaPipe | HMR-3d-only（Δ） | NLF-3d-only（Δ） |
   |---|---|---|---|---|---|
   | Ex1 | arm abduction | 356 | 0.652 | 0.611（−0.041） | 0.621（−0.031） |
   | Ex2 | arm VW | 416 | 0.559 | 0.537（−0.022） | 0.629（**+0.070**） |
   | Ex3 | table push-ups | 214 | 0.620 | 0.716（**+0.096**） | 0.739（**+0.119**） |
   | Ex4 | leg abduction | 420 | 0.686 | 0.676（−0.010） | 0.699（+0.013） |
   | Ex5 | leg lunge | 348 | 0.510 | 0.561（+0.051） | 0.484（−0.026） |
   | Ex6 | squats | 390 | 0.713 | 0.746（**+0.033**） | 0.692（−0.021） |

   （此表 NLF 欄為 **3d-only**，故與上文 NLF 章節「6 動作全數 ≥ MediaPipe」的表不同——那張表用的是 NLF **3d2d**，含 NLF 乾淨的影像 2D；要與 HMR-3d-only 公平對照才改用 3d-only。）

4. **兩模型分歧處正好被 HMR 的偵測 confound 解釋**：NLF 在 **arm VW 大補（+0.070）但 HMR 反退（−0.022）**——arm VW 是快速上肢揮動，最吃逐幀 localization 精度，25% 插補幀正好打在這裡，HMR 補不回反而被插補雜訊拖累。squats 則相反（HMR +0.033、NLF −0.021），但兩者都落在 MediaPipe 2D 本就夠用的面內動作、差異小。leg lunge 的 HMR +0.051 落在資料天花板動作（連 Vicon 0.498 都近隨機）→ 與 P5 同屬雜訊。**扣掉這些 confound/天花板格，剩下乾淨可信的就是 push-ups 的雙模型收斂。**

5. **對 NLF 結論的影響**：HMR 交叉驗證**沒有推翻也沒有乾淨確認** NLF。它提供 (a) 方向一致（同向 ≥ baseline）、(b) push-ups 上的機制收斂（兩個直接-3D 模型獨立 +~0.10 於最離面的動作）兩項弱佐證；但 subject 層因 75% 偵測率＋弱 2D 而被壓到雜訊內，無法當作 NLF +0.034 的獨立確認。**誠實的口徑：NLF 仍是目前最強的單目路線，HMR 的角色是「機制上呼應、統計上未確認」。**

6. **要乾淨確認，兩條路**：(a) **重抽 HMR**——把偵測率從 75% 拉到接近 100%（換更穩的偵測器或人工框追蹤），消除插補雜訊後重跑配對，看 arm VW 是否也翻正、整體是否逼近 NLF；(b) 釜底抽薪——**更多受試者**才能讓 +0.03 級別的真增益撐到顯著（n=9 是根本瓶頸，見 NLF 第 2 點）。在這之前，直接 image→3D 的結論維持「promising、機制有雙模型微弱呼應、但 subject 層未達顯著」。

產物：`correctness_loso_hmr_parametric_{3d2d,3d}.json`、配對 `correctness_loso_hmr3d_vs_mediapipe.json`、逐動作 `correctness_loso_per_exercise_hmr.json`（含 hmr3d / nlf3d / mediapipe / vp3dlift 四欄，皆 3d-only 同流程）。重現：Kaggle 抽取 HMR2.0/4DHumans 原始 3D → `python -m src.rehab24.nlf_skeleton_features`（同一特徵管線，--raw-dir 指向 hmr_raw3d）→ `python -m src.rehab24.loso_cross_validation --feature-dir …hmr_parametric_3d_skeleton_features`；配對 `python -m src.rehab24.paired_loso`；逐動作 `python scripts/rehab24/loso_per_exercise.py --sources mediapipe=mediapipe_skeleton_features hmr3d=hmr_parametric_3d_skeleton_features nlf3d=nlf_parametric_3d_skeleton_features vp3dlift=vp3d_lifted_skeleton_features --summary-output data/REHAB24-6/processed/correctness_loso_per_exercise_hmr.json`。

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

1. 以 **Skeleton-only（Vicon）作為 correctness 上限基線**，**MediaPipe 單目估計骨架作為可部署基線**——兩者都站得住，VideoMAE 不可單獨採信。**縮小那 6 點差距的方向已修正**（見「2D→3D Lifting 實驗」）：**2D→3D lifting 是死路**（自訓 TCN 與 SOTA 預訓練 VideoPose3D 都補不回真深度），應降權；改以「**拿最好的單目 2D**」（MediaPipe 系 2D > RTMPose/HRNet 2D）＋（若可行）**真 3D 量測（多視角／深度感測）** 為主線。
2. **先解決 VideoMAE 過擬合再談融合**：加強正則化（dropout / weight decay）、降維（PCA / 線性探針）、確認 VideoMAE 特徵是否做了 subject-wise 正規化，並複查 split 是否真的 subject-disjoint。
3. **改進融合方式**：以 late fusion / gating 取代早期 concat，或先凍結 skeleton 分支再小幅引入 video 分支，避免壞特徵主導。
4. **報告口徑**：以 test selected-threshold 為準，並附上 train→test 落差佐證泛化，不要只報訓練或固定 0.5 的數字。

## 備註

- 原始 log 中三個區塊的 test selected-threshold 行各重複列印一次（數值一致），疑為 log 寫了兩遍，不影響結論。
- 資料來源：`result.md`（Vicon / VideoMAE / Fuse 三組設定的完整 train/val/test metrics）；MediaPipe 一組為後續以 `scripts/rehab24/extract_mediapipe_skeleton_features.py` 抽取（complexity=2、stride=1 全保真，130 支影片）後，用 `train_correctness_classifier.py --feature-dir .../mediapipe_skeleton_features` 訓練所得（feature_dim=2970，train=1434 / val=212 / test=498）。
