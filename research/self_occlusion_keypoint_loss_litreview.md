# 深蹲自我遮蔽下骨架關鍵點缺失 — 方法文獻綜整

> 文獻回顧簡報（deep-research / lit-review 模式）— 聚焦「深蹲類動作的 self-occlusion 導致 MediaPipe 關鍵點暫時消失，而消失段恰好是錯誤高發段」這一問題的可用方法族。
> 產製日期：2026-06-12。來源分級與 AI 揭露見文末。本簡報由 AI 輔助研究工具協助整理。
> 關聯：[[rehab24_monocular_pose_improvement_brief]]（同屬「降低單目估計雜訊」主題；本篇專攻遮蔽缺失，該篇專攻深度模糊/抖動的整體判別力差距）。

---

## 1. 問題定性：為什麼這個問題特別棘手

非一般遮蔽，而是三個困難疊加的最壞情況：

1. **自我遮蔽 + 深度模糊**：深蹲到底部時大腿/軀幹互相遮擋，單目相機下髖、膝的 z（深度）本就最不可靠。MediaPipe 的 z 是相對估計，此相位幾乎不可信。
2. **連續多幀缺失**：遮蔽是「持續一段」而非單幀雜訊。文獻明確指出，單幀抖動易補，**連續多幀的關節大幅偏移才是真正困難的情況**（SmoothNet, ECCV 2022）。
3. **缺失段 = 錯誤段（核心矛盾）**：要偵測的錯誤（膝內扣、深度不足、butt wink / 骨盆後傾）正好發生在資訊最差的相位。

第 3 點帶出對「教練回饋」場景至關重要、卻被多數泛用方法忽略的張力：多數補全方法用「正常動作先驗」填洞，而我們恰恰想抓「不正常」。見 §3-C 與 §3 張力段。

---

## 2. 方法族系譜（七類）

### A. 信心/可見度感知 + 時序卷積（不信任被遮點，讓時序撐住）
不把被遮關節當有效觀測；用 2D heatmap 信心或 visibility 過濾，再交時序網路補出完整 3D。
- **Cheng et al., Occlusion-Aware Networks for 3D HPE in Video (ICCV 2019)**：2D 信心 heatmap + 光流一致性過濾不可靠估計 → 2D/3D temporal convolutional networks (TCN) 強制時序平滑。
- **Cheng et al., Spatio-Temporal Networks with Explicit Occlusion Training (AAAI 2020，已驗證)**：多尺度空間特徵 + 多步幅 TCN + 時空判別器，訓練時主動遮點。
- **visibility-aware 評分**：顯式建模可見度，給被遮關節較公允分數（Information Sciences）。

→ **對 x-coach**：最低成本、最契合 local-first。MediaPipe 已輸出每點 visibility，可立即升為一等訊號（§4-1）。

### B. 時序後處理外掛（掛在 MediaPipe 之後，即插即用）
不改估計器，只在輸出端做時序修正。
- **SmoothNet (ECCV 2022，已驗證)**：plug-and-play、純時序、逐關節學長程時序關係，專治「罕見/被遮動作下連續多幀大幅偏移」，跨估計器/資料集可遷移。
- **DeciWatch (ECCV 2022)**：時序高效基線。
- **傳統濾波**：Kalman / Savitzky-Golay 平滑、線性/三次插值。實務界線：插值僅適合短洞（常設上限約 3 連續幀 ≈ 0.12s@25fps），**更長的洞用插值會引入明顯假影**（arXiv 2011.00250）。

→ **對 x-coach**：SmoothNet 本地、離線、即插即用，是骨架側性價比最高的升級。但須注意 C 節警告：平滑與補全都可能抹掉錯誤。

### C. 學習式人體運動先驗 + 補全（生成式，最強也最危險）
學「人體會怎麼動」的先驗，從部分/被遮觀測重建完整運動。
- **HuMoR (ICCV 2021)**：CVAE 運動先驗，可擬合部分關鍵點/含噪關節。
- **RoHM (CVPR 2024 Oral，已驗證)**：diffusion，拆 TrajNet（全域軌跡）+ PoseNet（局部姿態），支援去噪、**空間補全（補被遮關節）、時間補全（補缺失幀）**，比 optimization-based 快約 30×。
- **Di2Pose (NeurIPS 2024，已驗證收錄)**：discrete diffusion 用於被遮 3D 姿態，以姿態量化 (codebook) 約束、降低生成不合理姿態。
- **Masked Modeling for Human Motion Recovery Under Occlusions (2026 預印本)**。

→ **⚠️ 教練場景關鍵警告（本綜整最重要一點）**：這些先驗多在**正常/健康動作**上訓練。把被遮的底部相位交給它補，它傾向填出「一個漂亮的深蹲」——**正好把要偵測的 butt wink / 膝內扣抹平成正常**，是致命的假陰性來源。對動捕是優點，對「錯誤偵測」是陷阱。**生成式補全在本 pipeline 只能當低信心證據，絕不能拿補出來的點直接觸發或否決一條 fault rule。**

### D. 合成遮蔽資料增強（讓估計器先天抗遮）
訓練時主動製造遮蔽，逼模型用人體動力學恢復被遮點。三種遮罩：point-wise（隨機點遮零）、frame-wise（整幀遮零，抗模糊/曝光異常）、continuous（隨機長度連續遮某些點，模擬持續自我遮蔽）。
- 代表：Explicit Occlusion Training (AAAI 2020)、StridedPoseGraphFormer + augmentation (arXiv 2304.12069)、LASOR (合成遮蔽資料)。

→ **對 x-coach**：除非自訓估計器，否則屬高成本「重訓」路線；但 continuous occlusion 思路可借來做**壓力測試**，量測 rule detector 在連續遮蔽下的退化曲線。

### E. 多視角 / 多相機融合（幾何上真正「看到」被遮部位）
自我遮蔽是單視角問題；換角度該關節就不被遮。多視角 2D 三角化/融合出 3D。
- **Multi-view Pose Fusion (arXiv 2408.15810, 2024 預印本)**：多視角單視 3D 預測再融合，顯式 occlusion-aware。
- **Stereo + MediaPipe 用於體能動作 (Sensors 2024, 24:7772，同儕審查)**：兩組 2D + 相機內外參三角化即可重建 3D，隨 2D 精度提升，雙視角立體重建已能匹敵多視角版本。

→ **對 x-coach**：**若拍攝端可加第二台相機，這是對「底部深度模糊」最徹底的真解**——觀測到資訊而非猜測。代價是部署複雜度與標定。

### F. 多模態感測融合（IMU + 視覺）
IMU 不受遮蔽影響、提供直接 3D 量測，相機失效時頂上。
- **Fusing Wearable IMUs with Multi-View Images (CVPR 2020, arXiv 2003.11163)**：IMU 方向作結構先驗融合影像特徵。
- **FusePose (arXiv 2208.11960)**：在參數化人體空間對齊 IMU 與視覺，IMU 校正遮蔽偏移。

→ **對 x-coach**：抗遮、環境穩定、隱私友善；但需穿戴，改變「純影片教練」情境。膝/髖貼少量 IMU 對底部相位最有幫助，屬可選硬體路線。

### G.（綜整推論）不依賴顯式關鍵點的外觀/影片模型 —— x-coach 已有
**VideoMAE 這類時空 RGB 模型不需顯式骨架**，骨架消失的相位影片分支仍能輸出「動作層級錯誤分類」。當前遮蔽基準（VOccl3D 2025、Benchmarking under Occlusions arXiv 2504.10350）顯示所有方法都隨遮蔽加重退化，**沒有單一模態是萬靈丹——冗餘模態是現實解法**。

---

## 3. 跨來源綜整

**收斂的共識**
- 沒有單一銀彈。可運作配方是三段疊加：(i) 用信心/可見度**不去信任**被遮點；(ii) 用時序/運動先驗**撐住或補出**短缺失；(iii) 理想上引入**額外視角或模態**去真正觀測底部相位。
- 連續多幀缺失是分水嶺：短洞（≤~3 幀）插值/濾波夠用；長洞須靠學習式運動先驗或多模態，單純插值會產生假影。

**最關鍵的張力（針對本場景）**
- **「補全」與「錯誤偵測」目標相反**。視覺合理性 ≠ 對錯誤忠實。正常動作上訓練的先驗會系統性把被遮的錯誤相位「修正」成正常，製造假陰性。文獻**尚無**專為「保留錯誤的補全」優化的成熟方法——這正是可發表的研究缺口。

---

## 4. 給 x-coach 的具體建議（依現有 MediaPipe 33-landmark + VideoMAE + rules + RAG/KG 架構排序）

1. **把 MediaPipe visibility 升為一等訊號，門控規則引擎**（零成本、最契合 local-first）。某關節 visibility 低於閾值時，`pose_rule_detector` 對該規則輸出 `occluded/uncertain`，而非在垃圾座標上硬算。**最該先做的一步。**
2. **短洞 vs 長洞分流**：≤~3 連續幀用 Kalman/Savitzky-Golay 補；超過就**標記為不可判定**，交給模態 G，而非插值後當真值。
3. **掛 SmoothNet 式時序後處理**於 MediaPipe 輸出（本地、即插即用），改善連續被遮幀——但結果走「低信心」通道，不直接觸發 fault。
4. **讓 VideoMAE 分支接管被遮相位**：骨架規則（高信心時）與影片層級錯誤分類（全程可用）做晚期融合，用 visibility 驅動加權（被遮窗口提高影片分支權重）。對現有架構最自然、額外成本最低。
5. **若拍攝端可行，加第二台相機**（雙視角三角化）——對「底部深度模糊」最徹底的真解，已在 MediaPipe + 體能動作上同儕審查驗證。唯一「觀測到」而非「猜測」的路線。
6. **把『保留錯誤的遮蔽補全』當研究貢獻點**：現有 RoHM/Di2Pose/HuMoR 偏向生成正常運動。一個 uncertainty-aware、明確保留偏差（或乾脆只標記不補）的方法是文獻空白，契合「explainable coaching」定位——寧可說「此相位資訊不足、置信度低」，也不要自信地補出被洗白的錯誤。

---

## 5. 方法論與限制（透明度聲明）

- 本綜整為 **lit-review 模式**，非 PRISMA 系統性回顧；檢索為英文、web 來源，未涵蓋全部資料庫，可能漏掉中文或非索引文獻。
- **來源分級**：AAAI/ICCV/CVPR/ECCV/NeurIPS/MDPI Sensors 屬同儕審查（Tier 1）；標「預印本」者（Multi-view Pose Fusion、VOccl3D、StridedPoseGraphFormer、Masked Modeling 等）為 arXiv，**未經同儕審查，引用須註明**。
- Di2Pose 內部方法細節以 NeurIPS 2024 官方 proceedings 收錄標題與檢索摘要為據；PDF 為二進位無法逐字解析，細節保守處理，未逾越已驗證範圍。
- **AI 揭露**：本研究使用 AI 輔助工具（Claude / deep-research）進行文獻檢索與綜整。

---

## 來源

- Cheng et al., 3D HPE using Spatio-Temporal Networks with Explicit Occlusion Training (AAAI 2020) — https://arxiv.org/abs/2004.11822
- Cheng et al., Occlusion-Aware Networks for 3D HPE in Video (ICCV 2019) — https://ieeexplore.ieee.org/document/9010921/
- Zeng et al., SmoothNet (ECCV 2022) — https://arxiv.org/abs/2112.13715
- Rempe et al., HuMoR (ICCV 2021) — https://geometry.stanford.edu/projects/humor/
- Zhang et al., RoHM: Robust Human Motion Reconstruction via Diffusion (CVPR 2024 Oral) — https://sanweiliti.github.io/ROHM/ROHM.html
- Di2Pose: Discrete Diffusion Model for Occluded 3D HPE (NeurIPS 2024) — https://proceedings.neurips.cc/paper_files/paper/2024/file/b2e20d7402c9985eae4ba924c65370a8-Paper-Conference.pdf
- Masked Modeling for Human Motion Recovery Under Occlusions (2026 preprint) — https://arxiv.org/html/2601.16079
- Multi-view Pose Fusion for Occlusion-Aware 3D HPE (arXiv 2408.15810, 2024 preprint) — https://arxiv.org/abs/2408.15810
- Stereo Camera Fusion for Physical Exercises with MediaPipe Pose (Sensors 2024, 24:7772) — https://www.mdpi.com/1424-8220/24/23/7772
- Fusing Wearable IMUs with Multi-View Images (arXiv 2003.11163, CVPR 2020) — https://arxiv.org/pdf/2003.11163
- FusePose: IMU-Vision Sensor Fusion (arXiv 2208.11960) — https://arxiv.org/pdf/2208.11960
- StridedPoseGraphFormer + Data Augmentation (arXiv 2304.12069) — https://arxiv.org/abs/2304.12069
- Temporal Smoothing for 3D HPE for Occluded People (arXiv 2011.00250) — https://arxiv.org/pdf/2011.00250
- VOccl3D: Video Benchmark under Real Occlusions (2025 preprint) — https://arxiv.org/html/2508.06757v1
- Benchmarking 3D HPE Models under Occlusions (arXiv 2504.10350) — https://arxiv.org/html/2504.10350v2
- Perceiving heavily occluded human poses by assigning unbiased score (Information Sciences) — https://www.sciencedirect.com/science/article/abs/pii/S0020025520305119
