## MMPose 技術細節與實作方式

1. 我們使用的是「全身姿態估計」後端。

	意思是模型不只偵測人體主要關節，例如肩膀、手肘、髖部、膝蓋、腳踝，也會偵測更細的部位，例如腳跟、腳趾、手部、臉部等。
	
	在本研究中，選擇 whole-body 很重要，因為深蹲分析不只需要膝蓋、髖部、腳踝，也需要腳跟與腳趾位置來判斷：
	- 膝蓋是否過度前移
	- 腳跟是否抬起
	- 膝蓋與腳尖的相對位置

  2. specifically an RTMPose/RTMW-family whole-body model
	  更精確地說，我們使用的是 RTMPose / RTMW 系列的全身姿態模型。
	
	  RTMPose 和 RTMW 是 MMPose 生態系中的現代姿態估計模型系列。它們比早期的 HRNet、SimpleBaseline 等模型更適合實際應用，通常在速度與準確率之間有較好的平衡。
	
	  這裡的重點是：
	
	  whole-body = 任務類型
	  RTMPose / RTMW = 使用的模型系列
	
	  也就是說，我們不是只說「用了 wholebody」，而是使用 RTMPose/RTMW 系列來執行 whole-body pose estimation。

  3. via rtmlib
	  我們是透過 rtmlib 來執行這個模型。
	
	  rtmlib 可以理解成一個比較輕量、方便部署的推論工具。它使用 ONNX Runtime 來執行姿態估計模型，通常比直接安裝完整 MMPose 環境更容易，尤其是在 Colab 或 Python 3.12 環境中。
	
	  在本專案中，程式預設使用：
	
	  --runtime rtmlib --model balanced
	
	  代表使用 rtmlib 的 whole-body 模型，並採用平衡速度與準確率的設定。

  4. producing COCO-WholeBody keypoints
	  模型輸出的關鍵點格式是 COCO-WholeBody。
	
	  COCO-WholeBody 是一種人體關鍵點標註格式，包含比一般 COCO body pose 更多的點。一般 COCO 人體姿態通常只有 17 個 body keypoints，例如肩膀、髖部、膝蓋、腳踝等。
	
	  COCO-WholeBody 則包含：
	
	  - 身體關鍵點
	  - 足部關鍵點
	  - 臉部關鍵點
	  - 手部關鍵點
	
	  因此它比較適合用在需要腳部資訊的動作分析任務。

  5. that are mapped into a MediaPipe-compatible 33-landmark format
	  最後，我們會把 COCO-WholeBody 的關鍵點轉換成與 MediaPipe 相容的 33 個 landmark 格式。
	
	  原因是本專案原本的下游分析流程，例如：
	
	  - pose feature extraction
	  - rule-based detection
	  - view estimation
	  - classifier training
	
	  都是基於 MediaPipe Pose 的 33-landmark 格式設計的。

  所以我們沒有重寫整個分析流程，而是把 MMPose / RTMW 輸出的 COCO-WholeBody 關鍵點轉成類似 MediaPipe 的格式，讓後續程式可以沿用。

  簡化來說：

  影片
  → rtmlib / RTMPose-RTMW whole-body model
  → COCO-WholeBody keypoints
  → 轉成 MediaPipe 33 landmarks
  → 進入原本的深蹲分析 pipeline

  整句話可以翻成比較自然的中文：

  > 本研究使用全身姿態估計後端，具體而言，是透過 rtmlib 執行 RTMPose/RTMW 系列的全身姿態模型。模型會輸出 COCO-WholeBody 格式的人體關鍵點，接著再將這些關鍵點轉換為與 MediaPipe 相容的 33-
  > landmark 格式，以便銜接既有的姿態特徵擷取與動作品質分析流程。

### 為什麼使用 MMPose whole-body

這次選擇 MMPose `wholebody` inferencer，而不是只用 body-only pose model，主要原因是既有 squat 分析邏輯不只需要 hips、knees、ankles，也需要 heel 與 toe/foot landmarks。

目前 feature extractor 與 rule detector 會使用以下 foot-related 訊號：

- `left_knee_to_toe_x` / `right_knee_to_toe_x`
- `left_knee_to_toe_abs_x` / `right_knee_to_toe_abs_x`
- `knee_forward_ratio`
- `heel_height_delta`
- heel rise rule
- knees-forward rule 的 knee-to-toe projection

如果改用 COCO body-only 17 keypoints，通常只會有 ankle，沒有 heel 與 toe，因此 knees-forward 和 heel-rise 相關特徵會失真或被迫移除。使用 whole-body model 可以保留比較公平的 biomechanical feature set。

### MMPose 輸入與輸出

實作檔案是：

```text
src/mmpose_pose_extraction.py
```

主要執行方式：

```bash
python scripts/run_mmpose_pose_extraction.py \
  --video-dir data/Squat/Labeled_Dataset/videos \
  --split-dir data/Squat/Labeled_Dataset/Splits \
  --output-dir data/Squat/Labeled_Dataset/mmpose_pose_json \
  --model wholebody \
  --device cuda:0
```

程式會對每支影片逐幀讀取 OpenCV frame，送進：

```python
MMPoseInferencer(pose2d="wholebody", device="cuda:0", show_progress=False)
```

每一幀會呼叫 inferencer，取得 MMPose prediction，再取出最高信心的人體 instance。輸出會寫成與既有 MediaPipe pipeline 相容的 JSON：

```json
{
  "metadata": {
    "fps": 30.0,
    "width": 480,
    "height": 600,
    "total_frames": 90,
    "backend": "mmpose",
    "mmpose_model": "wholebody",
    "keypoint_schema": "coco_wholebody_133_to_mediapipe_33",
    "world_landmarks": false
  },
  "frames": [
    {
      "frame_index": 0,
      "landmarks": [],
      "world_landmarks": null,
      "backend_pose_score": 0.91
    }
  ]
}
```

這個格式刻意保留既有欄位名稱：

- `frames[].landmarks`
- `frames[].world_landmarks`
- `frame_index`
- `metadata.fps`
- `metadata.width`
- `metadata.height`
- `metadata.total_frames`

因此 `src/pose_feature_extraction.py`、`src/pose_rule_detector.py`、`src/view_estimation.py` 都可以直接讀取 MMPose JSON，不需要另外寫一套下游 feature/rule/classifier。

### COCO-WholeBody 到 MediaPipe 33 landmarks 的轉換

MMPose whole-body 使用 COCO-WholeBody keypoint schema。既有程式則假設 MediaPipe Pose 33 landmark schema。為了沿用既有 feature/rule pipeline，本次新增 adapter：

```python
coco_wholebody_to_mediapipe_landmarks(...)
```

目前主要 mapping 如下：

| MediaPipe index | MediaPipe landmark | COCO-WholeBody index | 說明 |
| ---: | --- | ---: | --- |
| 0 | nose | 0 | nose |
| 2 | left_eye | 1 | left eye |
| 5 | right_eye | 2 | right eye |
| 7 | left_ear | 3 | left ear |
| 8 | right_ear | 4 | right ear |
| 11 | left_shoulder | 5 | left shoulder |
| 12 | right_shoulder | 6 | right shoulder |
| 13 | left_elbow | 7 | left elbow |
| 14 | right_elbow | 8 | right elbow |
| 15 | left_wrist | 9 | left wrist |
| 16 | right_wrist | 10 | right wrist |
| 23 | left_hip | 11 | left hip |
| 24 | right_hip | 12 | right hip |
| 25 | left_knee | 13 | left knee |
| 26 | right_knee | 14 | right knee |
| 27 | left_ankle | 15 | left ankle |
| 28 | right_ankle | 16 | right ankle |
| 29 | left_heel | 19 | left heel |
| 30 | right_heel | 22 | right heel |
| 31 | left_foot_index | 17、18 加權平均 | left big toe / small toe proxy |
| 32 | right_foot_index | 20、21 加權平均 | right big toe / small toe proxy |

MediaPipe 有 33 個 landmarks，但 COCO-WholeBody 沒有完全對應 MediaPipe 的所有 face/hand/body auxiliary points。沒有可靠對應的 MediaPipe index 會填入：

```json
{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
```

這樣 downstream 的 visibility threshold 會自然忽略這些不存在的點，不會被誤當成有效 landmark。

### 座標與 confidence 處理

MMPose 輸出的 keypoints 是 pixel coordinates。既有 MediaPipe pipeline 使用 normalized coordinates，因此 adapter 會做：

```python
x_normalized = x_pixel / frame_width
y_normalized = y_pixel / frame_height
z = 0.0
visibility = keypoint_score
```

注意：MMPose whole-body 這裡沒有產生 MediaPipe-style 3D world landmarks，所以：

```python
world_landmarks = None
```

後續 feature extractor 原本就有 fallback 機制：

- 如果 `world_landmarks` 可用，優先用 world landmarks 算角度。
- 如果 `world_landmarks` 不可用或有效點不足，改用 image landmarks。

因此 MMPose backend 會主要依賴 2D normalized image geometry。這是與 MediaPipe 比較時需要記錄的限制。

### 多人或背景人體的處理

MMPose inferencer 可能在一幀中回傳多個 person instances。這次採用簡單、可重複的策略：

```python
select_primary_instance(instances)
```

目前選擇方式：

- 優先使用 `bbox_score` 最高的 instance。
- 如果沒有 `bbox_score`，就用 keypoint scores 的平均值當作 instance score。

這個做法符合目前資料集假設：每支 squat video 主要只有一位受測者。若未來影片常出現多個人，應改成 tracking-based selection，例如使用上一幀 pose center 或 bbox IoU 維持同一個 subject。

### 與既有 pipeline 的銜接方式

整體銜接流程是：

```text
video.mp4
  -> MMPose wholebody inferencer
  -> COCO-WholeBody keypoints
  -> MediaPipe-33-compatible JSON
  -> pose_feature_extraction.py
  -> mmpose_pose_features/*.npz
  -> run_videomae_experiment_grid.py
  -> classifier metrics
```

Rule-based 路線則是：

```text
MMPose MediaPipe-compatible JSON
  -> run_view_estimation.py
  -> run_pose_rule_detection.py
  -> evaluate_pose_rule_detection.py
  -> mmpose_pose_rule_validation_metrics.csv
```

這樣做的好處是比較時只替換 pose backend，其餘 downstream feature aggregation、rules、classifier、threshold selection 都維持一致。也就是說，MediaPipe vs MMPose 的差異主要來自 pose estimator 本身，而不是下游訓練或評估邏輯不同。

### Comparison report 的技術做法

比較腳本：

```text
scripts/compare_pose_backends.py
```

會彙整三類資料：

1. Pose extraction quality
   - processed videos
   - mean total frames
   - mean pose detected ratio
   - mean valid lower-body ratio

2. Rule-based metrics
   - 讀取 `pose_rule_validation_metrics.csv`
   - 只取 `view_type = ALL`
   - 比較 `precision`、`recall`、`f1`、`specificity`、`balanced_accuracy`、`mean_segment_iou`

3. Classifier metrics
   - 讀取 `experiment_summary.csv`
   - 只取 `split = test`
   - 只取 `threshold_kind = selected_threshold`
   - 對多 seed 結果計算 mean 與 std

輸出包含：

```text
backend_comparison.csv
backend_comparison.md
```

Markdown table 會以 MediaPipe 作為左側 baseline，MMPose 作為右側 backend，並計算：

```text
mmpose_minus_mediapipe
```

### 為什麼不直接改 feature extractor

這次沒有把 `src/pose_feature_extraction.py` 改成 backend-specific 設計，原因是目前比較目標是「替換 pose estimator，其他條件固定」。如果在 feature extractor 中為 MMPose 加入太多特殊邏輯，會讓比較結果混入 feature engineering 差異。

目前採用 adapter 的方式有三個優點：

- 對 MediaPipe baseline 零影響。
- 所有 downstream code 可以重用。
- 測試範圍集中在 keypoint schema conversion，比較容易驗證。

未來如果要做更完整的 MMPose-native feature，可以再新增獨立的 feature extractor，而不是改動這條公平比較 baseline。


## 更新摘要

本次新增一條以 MMPose whole-body pose estimation 為基礎的特徵抽取與比較流程，用來和既有 MediaPipe pose pipeline 比較 rule-based detection 與 pose-only classifier 的效果。

核心設計是讓 MMPose 輸出轉成既有 MediaPipe 33 landmark JSON 格式，因此後續的 pose feature extraction、rule detector、view estimation、classifier training 都可以沿用現有程式，不需要另外建立一套下游模型流程。

## 新增檔案

- `src/mmpose_pose_extraction.py`
  - 使用 MMPose `wholebody` inferencer 逐幀抽取 pose。
  - 將 COCO-WholeBody keypoints 轉成 MediaPipe 33 landmark schema。
  - 保留 shoulders、hips、knees、ankles、heels、toe/foot points，讓現有深蹲幾何特徵可以繼續使用。
  - 因 RTMW whole-body 輸出為 2D keypoints，`world_landmarks` 會寫成 `None`。

- `scripts/run_mmpose_pose_extraction.py`
  - MMPose pose extraction 的批次執行入口。
  - 支援 train/val/test split、`--limit`、`--overwrite`、`--device` 與 `--model`。

- `scripts/compare_pose_backends.py`
  - 彙整 MediaPipe 與 MMPose 的比較結果。
  - 比較項目包含 pose extraction quality、rule-based metrics、classifier metrics。
  - 輸出 long-format CSV 與 Markdown comparison table。

- `notebooks/run_mmpose_pose_comparison.ipynb`
  - Google Colab T4 GPU 用的完整執行 notebook。
  - 流程包含 MMPose 安裝、pose extraction、feature extraction、view metadata、rule evaluation、classifier grid、最終比較報表。

- `tests/test_mmpose_pose_extraction.py`
  - 測試 COCO-WholeBody 到 MediaPipe 33 landmarks 的轉換。
  - 測試 MMPose inferencer prediction shape 的解析與 primary instance 選擇。

- `tests/test_compare_pose_backends.py`
  - 測試 classifier summary 的多 seed 平均。
  - 測試 rule metrics 僅讀取 `view_type=ALL` 的整體結果。
  - 測試 Markdown comparison table 的 backend delta。

## 執行流程

建議在 Google Colab T4 GPU 上直接執行：

```text
notebooks/run_mmpose_pose_comparison.ipynb
```

Notebook 會依序產生以下主要輸出：

```text
data/Squat/Labeled_Dataset/mmpose_pose_json/
data/Squat/Labeled_Dataset/mmpose_pose_features/
data/Squat/Labeled_Dataset/mmpose_view_metadata.csv
data/Squat/Labeled_Dataset/mmpose_pose_rule_detections/
data/Squat/Labeled_Dataset/mmpose_pose_rule_validation_metrics.csv
data/Squat/mmpose_pose_classifier_experiments/
data/Squat/mmpose_mediapipe_comparison/backend_comparison.csv
data/Squat/mmpose_mediapipe_comparison/backend_comparison.md
```

若只想先 smoke test 少量影片，可先在 Colab 或本機已安裝 MMPose 的環境執行：

```bash
python scripts/run_mmpose_pose_extraction.py \
  --video-dir data/Squat/Labeled_Dataset/videos \
  --output-dir data/Squat/Labeled_Dataset/mmpose_pose_json \
  --splits test \
  --limit 1 \
  --device cuda:0
```

## 比較方式

MMPose pose JSON 會先轉成與 MediaPipe 相同的 feature bundle：

```bash
python scripts/run_pose_feature_extraction.py \
  --pose-json-dir data/Squat/Labeled_Dataset/mmpose_pose_json \
  --output-dir data/Squat/Labeled_Dataset/mmpose_pose_features \
  --overwrite
```

接著可以用同一組 rule detector 和 classifier 設定比較兩個 backend：

- Rule-based：比較 `pose_rule_validation_metrics.csv` 與 `mmpose_pose_rule_validation_metrics.csv`。
- Classifier：比較 MediaPipe pose features 與 MMPose pose features 在相同 label modes、seeds、normalization 與 threshold objective 下的 test selected-threshold metrics。
- Extraction quality：比較 processed videos、pose detected ratio、valid lower-body ratio。

最終彙整命令：

```bash
python scripts/compare_pose_backends.py
```

## 已完成驗證

已在本機完成不依賴 MMPose GPU runtime 的測試：

```bash
python3 -m py_compile src/mmpose_pose_extraction.py scripts/run_mmpose_pose_extraction.py scripts/compare_pose_backends.py
.venv/bin/python -m unittest tests.test_mmpose_pose_extraction tests.test_compare_pose_backends tests.test_pose_rule_detector tests.test_videomae_video_classifier
python3 -m json.tool notebooks/run_mmpose_pose_comparison.ipynb
```

結果：

- Python syntax check 通過。
- 12 個 `unittest` 測試通過。
- Colab notebook JSON 格式有效。

## 注意事項

- 本機尚未實際跑 MMPose extraction，因為 MMPose、MMCV、MMDetection 與 GPU runtime 主要預期在 Colab T4 環境安裝與執行。
- 目前 MMPose 輸出使用 2D keypoints，沒有 MediaPipe `world_landmarks`，因此 feature extractor 會以 image landmarks 做幾何計算。
- MMPose whole-body 是目前預設 backend，原因是 body-only 模型缺少 heel/toe keypoints，會削弱 knees-forward 與 heel-rise 相關特徵與規則。
- `compare_pose_backends.py` 不會自動重新產生 MediaPipe baseline；若 baseline 檔案不存在，會輸出 warning 並只比較已存在的 artifact。
