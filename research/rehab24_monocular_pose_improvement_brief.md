# REHAB24-6 單目姿態品質改進簡報

> 工程改進簡報 — 聚焦「縮小 Vicon→MediaPipe 那 ~6–7 個百分點的判別力差距」。
> 依據：`notes/rehab24_correctness_experiment_summary.md`（LOSO：Vicon 0.702±0.078、MediaPipe 0.633±0.055、VideoMAE 0.536±0.044）。
> 產製日期：2026-06-12。文獻佐證見文末「來源」。本簡報由 AI 輔助研究工具協助整理。

---

## 1. 研究問題與定位

實驗已確立一個穩固結論：**深蹲 correctness 的判別訊號主要落在幾何／運動學結構上**，骨架路線（Vicon、MediaPipe）在 subject-wise 下都能泛化，而 VideoMAE 端到端嵌入會記住受試者身份、換人即崩。因此可部署的單鏡頭教練應走 **RGB → pose → 幾何特徵 → 規則／分類**，而非影像嵌入。

剩下的工程問題收斂成一句話：

> 在「同流程、同 split」前提下，把便宜的單目估計骨架（LOSO 0.633）往昂貴的 Vicon 動捕（LOSO 0.702）推近多少、用什麼方法、成本多少？

這 ~0.07 的差距，summary 已歸因到**單目估計引入的雜訊（抖動、遮擋、深度模糊）**——train→test 落差 MediaPipe 0.18 介於 Vicon 0.10 與 VideoMAE 0.34 之間，骨架特徵本身穩健，崩的不是表示法而是輸入品質。所以改進方向就是**降低估計雜訊**，而非換模型家族。

**先講上限**：這條線的天花板大約是 Vicon 的 0.702，而且其中 P5 對所有特徵都接近隨機（資料天花板，非模型問題）。所以單目改進的「可回收空間」實際只有 ~0.07，且被 ±0.08 的折間 std 蓋住。**任何單一改動的真實效益都小於評估雜訊**——這決定了下面的驗證紀律（§4），也意味著要靠多個低成本改動疊加，而不是賭單一銀彈。

---

## 2. 文獻定錨：單目到底能逼近 mocap 多少

- **單目 markerless 對 mocap 的關節「角度」一致性可以很好**，即使關節「位置」誤差不小。Ergo 系統在過頭深蹲對 marker-based mocap 的時間序列關節角 R²=0.88–0.99、峰值角 ICC=0.75–1.0（Scientific Reports 2024）。這支持：correctness 用的是角度/運動學特徵，單目有機會逼近 mocap。
- **但 3D 深度仍是弱點**：markerless 對 mocap 的每關節位置誤差，2D 影像平面內約 72–122 mm，含深度的 3D 達 146–249 mm（Scientific Reports 2025 臨床評估）。**深度模糊是單目骨架最大的系統性誤差來源**——直接呼應 summary 的歸因。
- **2D 估計本身已具臨床可用度**：多篇報告 2D 投影膝屈曲角的準確度「等於或優於目視檢查」，與 3D mocap 有中到強相關。

**推論**：MediaPipe 的 3D world landmark 深度通道很可能是雜訊主要載體。改進策略應**強化時間一致性與 2D 品質、並謹慎處理深度**（要嘛用更好的 lifting，要嘛降低深度通道權重）。

---

## 3. 改進清單（依「效益/成本比」排序）

每項標注：**做法 → 文獻佐證 → 預期效益 → 實作成本 → 改哪裡 → 風險**。
效益用「LOSO bal_acc 的方向性增量」表達，且都需用 §4 紀律驗證（不是保證值）。

### ⭐ R1. 時間平滑（最高 CP 值，先做）
- **做法**：在 `mediapipe_skeleton_features.py` 的 `interpolate_missing` 之後、`add_velocity` 之前，對每個 joint/channel 沿時間軸做平滑。最低成本：One-Euro filter 或 Savitzky–Golay；進階：掛 **SmoothNet**（plug-and-play、temporal-only、跨估計器/模態可轉移，ECCV 2022）。
- **文獻佐證**：SmoothNet 在 5 個資料集 × 11 個 backbone 上同時降低抖動並提升困難幀準確度，且明確設計為「貼在現有估計器後面」。
- **預期效益**：直接打到 summary 點名的「抖動」雜訊。velocity 特徵對逐幀抖動特別敏感（微分放大噪聲），平滑對運動學特徵的收益通常高於對原始座標。估計 +0.01～0.03。
- **成本**：低。One-Euro/Savitzky–Golay 純 numpy/scipy，半天內可加；SmoothNet 需一個預訓練權重與一次前向，1–2 天。
- **改哪裡**：`src/rehab24/mediapipe_skeleton_features.py`（新增 `smooth_series` step）。Vicon 路線**不要**套（mocap 已乾淨，平滑反而抹細節）。
- **風險**：過度平滑會吃掉深蹲底部的轉折（correctness 訊號就在轉折），需調截止頻率並用 LOSO 掃。

### ⭐ R2. 換更強的 2D backbone（基礎建設已在）
- **做法**：repo 已有 `extract_rtmpose_skeleton_features.py`／`rtmpose_skeleton_features.py`（COCO-WholeBody → MediaPipe 33-landmark 對齊）。把 backbone 換成 **RTMPose-m/l** 或 **ViTPose**，重抽一份特徵，丟同一個分類器比 LOSO。
- **文獻佐證**：基準上 RTMPose/ViTPose 的 OKS 明顯優於 MediaPipe（肩、肘、腕分佈更集中）；ViTPose 為 COCO SOTA，RTMPose-m 75.8% AP 且可即時。MediaPipe 偏向速度而非精度。
- **預期效益**：更準的 2D = 更乾淨的角度特徵。這是「把估計品質往 mocap 推」最直接的一刀。估計 +0.02～0.04，且與 R1 可疊加。
- **成本**：中。基礎已存在，主要是裝 rtmlib 環境 + GPU 抽 130 支影片 + 確認 landmark 對齊正確。
- **改哪裡**：沿用 `rtmpose_skeleton_features.py`；驗證 COCO→BlazePose 對齊（尤其髖/肩 scale pair）後 `train_correctness_classifier.py --feature-dir .../rtmpose_skeleton_features`。
- **風險**：COCO-WholeBody 與 BlazePose 的關節定義不完全等價（如髖中心、足部），對齊誤差會抵銷精度收益——對齊驗證是成敗關鍵。

### R3. 信賴度感知的插補與加權（低成本清理）
- **做法**：MediaPipe 每個 landmark 有 visibility/presence 分數。目前 `interpolate_missing` 只處理 NaN。改成：(a) 低信賴度幀視為缺失再插補；(b) 在 `summarize_time_series` 用信賴度加權統計，讓遮擋幀少貢獻。
- **文獻佐證**：TTA/robustness 文獻一致顯示「以信賴樣本輔助、降低不可靠樣本權重」能提升穩健度；遮擋是 markerless 已知的位置誤差來源。
- **預期效益**：打到「遮擋」這一塊（summary 三大雜訊之一）。估計 +0.005～0.02。
- **成本**：低。純特徵端邏輯，半天～1 天。
- **改哪裡**：`mediapipe_skeleton_features.py`（`interpolate_missing`、`summarize_time_series`）。
- **風險**：閾值設太高會把太多幀當缺失，反而增加插補偏差；需掃 visibility 閾值。

### R4. 測試時增強（TTA）— 水平翻轉平均
- **做法**：抽特徵時，對每幀同時跑「原圖」與「水平鏡像」兩次姿態估計，鏡像結果交換左右關節後與原圖平均。
- **文獻佐證**：水平翻轉 TTA 是姿態估計標準做法，能降低預測變異、對小幅姿勢偏移更穩健，且不增訓練成本。
- **預期效益**：小而穩，+0.005～0.015。深蹲常有左右不對稱的估計偏差，翻轉平均可校正。
- **成本**：低，但**抽特徵時間 ×2**。
- **改哪裡**：兩個 skeleton 抽取模組的逐幀估計迴圈。
- **風險**：左右 landmark index 必須正確交換，否則平均會破壞姿態。

### R5. 2D→3D lifting 取代原始深度通道（中高成本，最大潛在天花板）
- **做法**：丟掉 MediaPipe 的 3D world 深度，改用 **時間 2D→3D lifting**（VideoPose3D 的 dilated temporal conv，或現代的 MotionBERT 類）把乾淨的 2D 軌跡抬到 3D，再算幾何特徵。
- **文獻佐證**：VideoPose3D 在 Human3.6M 用 2D 軌跡的時間卷積，較前 SOTA 降 6 mm MPJPE（−11%），且其半監督 back-projection 可用未標記影片。臨床基準也指出「現有單目 3D 深度估計仍不準」——所以用**多幀時間 lifting** 比 MediaPipe 的**逐幀**深度更可靠。
- **預期效益**：直接攻擊最大系統誤差（深度，3D 位置誤差 146–249 mm）。潛在天花板最高（+0.02～0.05），但也最不確定。
- **成本**：高。需接 lifting 模型、處理座標系/單位對齊、可能要微調。1–2 週。
- **改哪裡**：新模組 `src/rehab24/lifted_skeleton_features.py`（2D 來源建議用 R2 的強 backbone，串成「強 2D → lifting → 幾何特徵」）。
- **風險**：lifting 自身會引入誤差（depth ambiguity 仍在），summary 文獻也警告「lifting 過程不可避免引入額外誤差」。**必須與「只用乾淨 2D（丟深度）」的消融對照**——有可能單純丟掉雜訊深度通道就贏過保留 MediaPipe 原始深度。

### R6. 多視角融合（條件性，視資料而定）
- **做法**：若 REHAB24-6 manifest 內同一 repetition 有多台同步相機，做多視角三角測量或多視角 2D 融合，逼近 mocap 幾何。
- **文獻佐證**：單目最大殘差就是深度；多視角直接解掉深度模糊，是 markerless 逼近 mocap 的已知路徑。
- **預期效益**：高，但**僅在「可部署＝單鏡頭」約束放寬時成立**——否則違背產品定位。建議只當「上限探針」：量出「多視角能拿回多少」，標示單目改進的理論空間。
- **成本**：中高，且**先要確認資料是否多視角**（查 `build_manifest.py` / 原始資料夾）。
- **風險**：與單鏡頭部署目標衝突；只適合做為診斷基線，不是部署方案。

---

## 4. 評估紀律（每個改動都必須遵守）

summary 第 3 點已立規矩：**±0.08 折間 std ＞ 想衝的進步幅度**。上面每項預期效益（多在 +0.01～0.04）都**小於**單折雜訊。因此：

1. **一律用 LOSO 9 折 mean±std 判定**，禁止只看固定切分（P8/P9）——固定切分偏樂觀 0.03–0.06，會把雜訊當成果。沿用 `scripts/rehab24/loso_cross_validation.py`。
2. **配對比較**：同一組受試者折，base vs. 改動逐折配對，看「每折都變好」還是「平均被一兩折拉動」。逐折差的 mean±std 比兩個獨立 mean±std 更靈敏（消掉受試者難度的共同變異）。可加 Wilcoxon signed-rank（9 折）或至少列每折 delta。
3. **回報區間**：給 mean±std 與每折範圍，不要只給單一數字。一個改動若 9 折有 6 折正、3 折負且平均 +0.01，要誠實標為「未定論」。
4. **疊加要逐項上**：R1→R2→R3 逐個加並各自跑 LOSO，避免把多個改動綁一起無法歸因。
5. **P5 單獨看**：它對所有特徵近隨機，是資料天花板。**把 P5 折納入平均會稀釋訊號**——建議同時報「含 P5」與「排除 P5」兩個數字，否則真實的單目改進會被一個救不了的受試者蓋掉。這也是值得單獨檢視標註/動作是否非典型的線索（雖不在本簡報聚焦範圍，但會直接影響你能不能「看見」改進）。

---

## 5. 建議執行順序（roadmap）

```
階段 0（先確認）：查 manifest 確認是否多視角；確認 MediaPipe visibility 分數是否已存進中介檔。
階段 1（1 週，低成本疊加）：R1 時間平滑 → R3 信賴度加權 → R4 翻轉 TTA。
              每步單獨 LOSO 配對驗證，留下「乾淨 2D 幾何」的強單目基線。
階段 2（1 週，基礎已在）：R2 換 RTMPose/ViTPose backbone，重抽特徵，LOSO 比。
              先驗證 COCO→BlazePose landmark 對齊。
階段 3（2 週，高潛力高風險）：R5 在 R2 的強 2D 上做時間 lifting，
              對照「丟深度的純 2D」消融。
階段 4（條件）：R6 多視角僅作為上限探針，量「單目還能再拿回多少」。
```

**最可能的高 CP 值組合**：R1 + R2 + R3。三者都打中 summary 點名的雜訊來源（抖動、估計精度、遮擋），成本低到中，且基礎建設（rtmpose 路線）已存在。R5 是上限賭注，值得做但要有「可能打平甚至輸」的心理準備，並務必帶消融。

---

## 6. 限制與誠實聲明

- **可回收空間有限**：天花板 ~0.702 且含一個救不了的 P5，單目可回收區間實測只有 ~0.07。不要承諾「逼近或超越 Vicon」——文獻支持單目「角度」可逼近 mocap，但「3D 位置/深度」仍有顯著殘差。
- **效益為方向性估計**：上述 +0.0x 數字是依文獻與雜訊來源的推估，非保證；真實值需 LOSO 配對量測，且部分改動很可能落在雜訊內無法定論。
- **特徵空間非同一**：MediaPipe(2970) 與 Vicon(2340) 維度不同，跨 backbone 比較以「同流程、同 split 下的判別力」為準，已是 summary 既有口徑。
- **與 VideoMAE 分流**：本簡報不碰 VideoMAE 過擬合與融合（summary 建議 2、3），那是另一條線；本線結論是「幾何路線優先、VideoMAE 不單獨採信」。

---

## 來源

- [Concurrent validity of deep-learning markerless mocap during overhead squat — Scientific Reports 2024](https://www.nature.com/articles/s41598-024-79707-2)
- [Assessment of monocular human pose estimation models for clinical movement analysis — Scientific Reports 2025](https://www.nature.com/articles/s41598-025-22626-7) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12589393/))
- [Paving the Way Towards Kinematic Assessment Using Monocular Video — arXiv 2510.02264](https://arxiv.org/pdf/2510.02264)
- [SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos — ECCV 2022 / arXiv 2112.13715](https://arxiv.org/abs/2112.13715) ([code](https://github.com/cure-lab/SmoothNet))
- [Pavllo et al., 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training (VideoPose3D) — CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf) ([code](https://github.com/facebookresearch/VideoPose3D))
- [RTMPose / ViTPose vs MediaPipe keypoint accuracy — Datature 2026 benchmark overview](https://datature.io/blog/what-is-pose-estimation-keypoint-detection-explained-2026)
- [Accuracy of Monocular 2D Pose Estimation vs Reference Standard for Kinematic Multiview Analysis — PMC7781802](https://pmc.ncbi.nlm.nih.gov/articles/PMC7781802/)
- [Exercise quantification from single camera view markerless 3D pose estimation — PMC10951609](https://pmc.ncbi.nlm.nih.gov/articles/PMC10951609/)
- [Learning Loss for Test-Time Augmentation — NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/2ba596643cbbbc20318224181fa46b28-Paper.pdf)
