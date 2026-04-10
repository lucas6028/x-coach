# TODO

## 目標定義

- [ ] 明確定義目前 `Squat` 任務主軸為：
  - `error detection`
  - `error classification`
  - `temporal localization`
  - `contrastive representation learning`
- [ ] 避免將目前資料描述成完整 `AQA score regression`
- [ ] 將計畫書中的敘述微調為：
  - 使用預訓練 `VideoMAE V2` 作為時空特徵提取器
  - 搭配對比學習強化正常動作與錯誤動作的表徵可分性
  - 融合 `MediaPipe` 幾何特徵提升可解釋性

## 資料集現況整理

- [ ] 確認已可直接使用的標註與切分：
  - `data/Squat/Labeled_Dataset/Splits/train_keys.json`
  - `data/Squat/Labeled_Dataset/Splits/val_keys.json`
  - `data/Squat/Labeled_Dataset/Splits/test_keys.json`
- [ ] 記錄目前資料規模：
  - `Labeled_Dataset/videos`: 1739 支影片
  - split union: 1623 支有正式 train/val/test key
  - `Unlabeled_Dataset/videos`: 4970 支影片
- [ ] 記錄標註類型：
  - `error_knees_forward.json`: 影片 key 對應錯誤時間區段
  - `error_knees_inward.json`: 影片 key 對應錯誤時間區段
  - `labels_shallow_depth.json`: 片段/局部 label
- [ ] 記錄標註格式：
  - `[video_id]: [error_start_time, error_end_time]`
  - 無錯誤時為空陣列
- [ ] 記錄影片基本特性：
  - 多數影片約 `30 FPS`
  - 長度約 `3 秒`
  - 解析度約 `480x600`

## 資料可行性結論

- [ ] 在文件中明確寫出：
  - 目前資料集可以進行訓練、驗證、測試
  - 可做 `VideoMAE V2` 特徵提取
  - 可做自監督 / 弱監督對比學習
  - 可做監督式錯誤辨識與片段定位
- [ ] 在文件中補充限制：
  - 沒有完整連續品質分數
  - 不適合直接做傳統 `AQA regression`
  - 類別不平衡，特別是 `knees_inward`
  - 標註粒度不完全一致

## 模型設計

### Stage 1: VideoMAE V2 特徵提取

- [ ] 決定使用預訓練 `VideoMAE V2` 作為 backbone
- [ ] 決定先採用：
  - frozen backbone
  - 或 partial fine-tuning
- [ ] 決定 clip 長度：
  - `16 frames`
  - 或 `32 frames`
- [ ] 決定特徵輸出方式：
  - CLS token
  - temporal average pooling
  - spatiotemporal pooled feature

### Stage 2: Contrastive Learning

- [ ] 使用 `Unlabeled_Dataset/videos` 做 domain adaptation
- [ ] 設計正樣本對：
  - 同一影片不同 temporal crop
  - 同一影片不同 augmentation
- [ ] 設計負樣本對：
  - 不同影片 clip
  - 不同錯誤型態 clip
- [ ] 評估 loss 選項：
  - `InfoNCE`
  - `Supervised Contrastive Loss`
  - `classification loss + contrastive loss`

### Stage 3: Supervised Downstream Task

- [ ] 建立影片級 / clip 級任務：
  - `knees_forward`
  - `knees_inward`
  - `shallow_depth`
- [ ] 決定任務形式：
  - binary classification
  - multi-label classification
  - temporal localization
- [ ] 建立 classification head
- [ ] 建立 threshold tuning 流程

### Stage 4: Pose / Geometry Fusion

- [ ] 保留 `MediaPipe` 或 pose estimation 模組
- [ ] 計算幾何特徵：
  - 膝角
  - 髖角
  - 軀幹傾角
  - 膝蓋與腳尖相對偏移
- [ ] 設計融合方式：
  - late fusion
  - feature concatenation + MLP
- [ ] 將幾何特徵用於可解釋輸出與結果驗證

## 即時性規劃

- [ ] 明確區分兩種系統模式：
  - 離線分析模式
  - 準即時回饋模式
- [ ] 在文件中說明 `VideoMAE` 不適合逐幀超低延遲即時回饋
- [ ] 若要做準即時，考慮：
  - 小型 backbone
  - 較短 clip
  - 較低解析度
  - sliding window inference
- [ ] 系統角色分工：
  - `MediaPipe` 負責即時幾何警示
  - `VideoMAE` 負責高品質時空判斷

## 如何辨識 VideoMAE 的結果

- [ ] 明確區分 `VideoMAE` 原始輸出是特徵向量，不是人類可讀文字
- [ ] 決定下游辨識方式：
  - 接 classification head
  - 用 embedding distance / prototype classifier
  - 用滑動視窗做時間片段辨識
- [ ] 建立輸出格式：
  - 錯誤類型
  - 發生時間點
  - 信心分數
  - 對應建議
- [ ] 加入特徵空間視覺化：
  - `t-SNE`
  - `UMAP`

## 驗證指標

### 影片級二元分類

- [ ] 計算：
  - `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
  - `Precision = TP / (TP + FP)`
  - `Recall = TP / (TP + FN)`
  - `F1 = 2 * Precision * Recall / (Precision + Recall)`

### 多類別 / 多標籤

- [ ] 回報：
  - `Accuracy`
  - `Macro F1`
  - `Weighted F1`
- [ ] 若為多標籤任務，優先觀察：
  - per-class F1
  - `micro F1`
  - `macro F1`

### 時間片段定位

- [ ] 若做 temporal localization，計算：
  - `segment IoU`
  - frame-level `Precision / Recall / F1`
  - 視情況加入 `mAP@IoU`

### 類別不平衡處理

- [ ] 不只看 `Accuracy`
- [ ] 對 `knees_inward` 額外回報：
  - `Precision`
  - `Recall`
  - `F1`
  - `PR-AUC` 或 `ROC-AUC`
- [ ] 在 validation set 上調整 threshold，而不是固定 `0.5`

## 實驗設計

- [ ] 設計 baseline：
  - only `VideoMAE`
  - only pose / geometry
  - `VideoMAE + contrastive`
  - `VideoMAE + pose`
  - `VideoMAE + pose + contrastive`
- [ ] 設計 ablation study：
  - 是否使用 unlabeled pretraining
  - 是否使用 supervised contrastive loss
  - 不同 clip 長度
  - 不同 fusion 方法
- [ ] 規劃最終 test set 僅用於最後一次評估

## 最終成果形式

- [ ] 完成一個可輸入深蹲影片的模型
- [ ] 輸出內容至少包含：
  - 錯誤類型
  - 錯誤發生時間區段
  - 信心分數
  - 結構化回饋
- [ ] 規劃 demo 介面：
  - 左側影片 / 骨架顯示
  - 時間軸錯誤標記
  - 右側顯示觀察 / 原因 / 建議
- [ ] 若結合 RAG，產出：
  - 觀察
  - 生物力學原因
  - 可執行糾正建議

## 建議先做的最小可行版本

- [ ] 第一步先完成 `Squat` 的 `knees_inward` 二元分類
- [ ] 第二步加入 `knees_forward`
- [ ] 第三步加入 `shallow_depth`
- [ ] 第四步加入 contrastive learning
- [ ] 第五步加入 pose fusion
- [ ] 第六步再考慮 RAG 與教練式文字回饋
