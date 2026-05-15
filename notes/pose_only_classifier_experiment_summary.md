# Pose-only 分類器實驗摘要

## 背景

在 VideoMAE feature classifier 的 test balanced accuracy 多數只略高於隨機基準後，新增 pose-only baseline。目標是確認以 MediaPipe landmarks 轉出的幾何特徵，是否比單一 VideoMAE video embedding 更適合判斷深蹲錯誤。

目前 pose-only pipeline 先把 labeled videos 轉成 pose JSON，再把每支影片轉成固定長度的 `.npz` feature bundle。輸出的 `.npz` 保留 `video_feature` 欄位，因此可以直接重用既有的 `src/videomae_video_classifier.py` 和 `scripts/run_videomae_experiment_grid.py`。

主要新增或使用的檔案：

- `scripts/run_pose_extraction.py`
- `src/pose_feature_extraction.py`
- `scripts/run_pose_feature_extraction.py`
- `scripts/run_videomae_experiment_grid.py`

## Pose Feature 設計

pose-only 特徵使用 MediaPipe lower-body landmarks，計算每個 frame 的幾何訊號，再聚合成 video-level feature。

目前包含的主要 frame-level 特徵：

- 左右膝角度：hip-knee-ankle
- 左右髖角度：shoulder-hip-knee
- 左右踝角度：knee-ankle-foot
- 左右角度不對稱
- 膝蓋相對 hip-ankle 線的偏移
- 膝蓋寬度相對髖寬與踝寬
- 膝蓋相對 ankle/toe 的水平偏移
- hip、knee、ankle 的垂直位置與相對深度 proxy
- lower-body visibility 與有效 frame ratio

每個 frame-level 特徵分別在 full video 與 bottom phase 上聚合。bottom phase 目前用平均 hip y 較低的區段近似，也就是 normalized image coordinate 中 hip y 較大的 frame。

每個區段使用以下統計：

- mean
- std
- min
- max
- p10
- p25
- p50
- p75
- p90

目前每支影片的 pose-only `video_feature` 維度為 654。

## 實驗設定

實驗重用 VideoMAE classifier 的訓練與評估流程，只更換 feature directory 為 pose features。

主要設定：

- feature: `data/Squat/Labeled_Dataset/pose_features`
- label modes: `combined`, `knees_forward`, `knees_inward`
- seeds: 1, 2, 3, 4, 5
- epochs: 20
- learning rate: 3e-4
- hidden dimension: 128
- dropout: 0.4
- weight decay: 0.01
- early stopping patience: 5
- threshold objective: balanced accuracy
- 主要觀察 test selected-threshold metrics

各 label mode 的 class balance：

| Label mode | Train positives / negatives | Val positives / negatives | Test positives / negatives |
| --- | ---: | ---: | ---: |
| combined | 812 / 324 | 165 / 78 | 172 / 72 |
| knees_forward | 782 / 354 | 158 / 85 | 169 / 75 |
| knees_inward | 160 / 976 | 36 / 207 | 36 / 208 |

## 多 Seed 結果

以下皆為 test selected-threshold metrics。

| Label mode | Seed | Balanced accuracy | Macro F1 | Recall | Specificity | F1 | TP | FP | TN | FN | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined | 1 | 0.618 | 0.623 | 0.820 | 0.417 | 0.794 | 141 | 42 | 30 | 31 | 0.550 |
| combined | 2 | 0.550 | 0.544 | 0.669 | 0.431 | 0.701 | 115 | 41 | 31 | 57 | 0.653 |
| combined | 3 | 0.554 | 0.555 | 0.802 | 0.306 | 0.767 | 138 | 50 | 22 | 34 | 0.474 |
| combined | 4 | 0.573 | 0.575 | 0.855 | 0.292 | 0.795 | 147 | 51 | 21 | 25 | 0.454 |
| combined | 5 | 0.609 | 0.613 | 0.802 | 0.417 | 0.784 | 138 | 42 | 30 | 34 | 0.532 |
| knees_forward | 1 | 0.568 | 0.569 | 0.828 | 0.307 | 0.776 | 140 | 52 | 23 | 29 | 0.538 |
| knees_forward | 2 | 0.576 | 0.578 | 0.846 | 0.307 | 0.786 | 143 | 52 | 23 | 26 | 0.522 |
| knees_forward | 3 | 0.556 | 0.558 | 0.793 | 0.320 | 0.757 | 134 | 51 | 24 | 35 | 0.516 |
| knees_forward | 4 | 0.582 | 0.585 | 0.858 | 0.307 | 0.792 | 145 | 52 | 23 | 24 | 0.500 |
| knees_forward | 5 | 0.583 | 0.586 | 0.793 | 0.373 | 0.766 | 134 | 47 | 28 | 35 | 0.531 |
| knees_inward | 1 | 0.594 | 0.548 | 0.472 | 0.716 | 0.304 | 17 | 59 | 149 | 19 | 0.622 |
| knees_inward | 2 | 0.562 | 0.533 | 0.389 | 0.736 | 0.267 | 14 | 55 | 153 | 22 | 0.504 |
| knees_inward | 3 | 0.537 | 0.454 | 0.556 | 0.519 | 0.256 | 20 | 100 | 108 | 16 | 0.913 |
| knees_inward | 4 | 0.589 | 0.542 | 0.472 | 0.707 | 0.298 | 17 | 61 | 147 | 19 | 0.843 |
| knees_inward | 5 | 0.565 | 0.504 | 0.500 | 0.630 | 0.275 | 18 | 77 | 131 | 18 | 0.534 |

整體平均：

| Label mode | Test balanced accuracy | Macro F1 | Recall | Specificity | F1 | 解讀 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| combined | 0.581 | 0.582 | 0.790 | 0.373 | 0.768 | 比 VideoMAE-only 略好，但仍偏向預測錯誤類。 |
| knees_forward | 0.573 | 0.575 | 0.824 | 0.323 | 0.775 | recall 高，但正常影片辨識仍弱。 |
| knees_inward | 0.569 | 0.516 | 0.478 | 0.662 | 0.280 | specificity 較好，但會漏掉不少 inward 錯誤。 |

固定 threshold 0.5 的 test balanced accuracy 平均：

| Label mode | Fixed 0.5 test balanced accuracy |
| --- | ---: |
| combined | 0.572 |
| knees_forward | 0.579 |
| knees_inward | 0.571 |

## 與 VideoMAE-only 的比較

VideoMAE-only 多 seed 實驗中，combined classifier 的 test balanced accuracy 大致落在 0.53 到 0.58，knees_forward 大多約 0.51 到 0.56，knees_inward 多數約 0.48 到 0.55。

pose-only 的結果顯示：

- combined 平均 test balanced accuracy 約 0.581，最高 0.618，整體略優於 VideoMAE-only。
- knees_forward 平均約 0.573，較 VideoMAE-only 更穩定，但 specificity 仍偏低。
- knees_inward 平均約 0.569，雖然 specificity 較好，但 positive-class F1 低，代表漏判問題仍明顯。

整體來看，pose-only features 確實提供比 VideoMAE-only 更直接的 biomechanical signal，但目前 video-level aggregation 和 MLP classifier 還不足以成為可靠的最終模型。

## 目前判讀

這次結果支持「pose features 比單一 VideoMAE embedding 更適合 squat form error」這個方向，但改善幅度有限。

主要觀察：

- pose-only 不再只是研究想法，已經是一個可跑、可比較的 baseline。
- combined 和 knees_forward 的 recall 偏高，但 specificity 偏低，表示模型仍傾向把正常影片判成有錯。
- knees_inward 的正類很少，test 只有 36 個 positives，因此結果容易受少數樣本影響。
- knees_inward specificity 較高但 recall 偏低，代表模型較保守，會漏掉錯誤影片。
- fixed threshold 0.5 和 selected threshold 的差距不大，表示 threshold tuning 不是主要瓶頸。

可用於報告的結論：

目前 pose-only classifier 的 test balanced accuracy 約在 0.57 到 0.58，略優於 VideoMAE-only baseline，但仍不足以作為可靠的深蹲錯誤判斷器。pose geometry 提供了更貼近錯誤定義的訊號，下一步應優先改善特徵標準化、錯誤案例分析、視角處理與 temporal modeling，而不是只繼續調整 MLP 超參數。

## 建議下一步

短期建議：

- 在 classifier pipeline 加入 train-set feature normalization，避免角度、比例、frame quality 等不同尺度混在一起。
- 匯出並檢查 high-confidence false positives 和 false negatives。
- 分別檢查 combined、knees_forward、knees_inward 的錯誤案例，尤其是視角是否足以觀察該錯誤。
- 檢查 pose quality 指標與錯誤預測的關係，例如 valid lower-body ratio 低的影片是否更容易誤判。

中期建議：

- 做 pose-only temporal model，不只用 video-level summary statistics。
- 嘗試只在 bottom phase 或 descent/ascent phase 訓練針對性特徵。
- 做 VideoMAE-plus-pose fusion，比較 RGB context 是否能補足 pose-only 的視角與遮擋問題。
- 若要改善 knees_inward，應優先處理類別不平衡與視角問題，而不是單純加大 MLP。

## View-aware 分析補充

後續觀察到資料集多數影片為斜後方視角，這會影響不同錯誤類型的可觀察性。特別是 knees_forward 較需要側面視角；若主要資料是 rear-oblique，模型的 false positive 或 specificity 偏低不一定完全代表 pose features 不足，也可能是視角與標籤定義不匹配。

已新增 rule-based view estimation pipeline：

- `src/view_estimation.py`
- `scripts/run_view_estimation.py`

使用方式：

```bash
python scripts/run_view_estimation.py
```

預設輸出：

```text
data/Squat/Labeled_Dataset/view_metadata.csv
```

目前 estimator 預設不輸出 `front` 與 `front_oblique`。人工檢查後發現原本被判為 front/front_oblique 的影片幾乎都是 rear-oblique，主要原因是影片中背景臉部或 MediaPipe pose face landmarks 的 visibility 會造成誤判。因此目前 face visibility 只保留為診斷欄位，不再參與 front/rear score；若未來資料中確實有正面視角，可用 `--allow-front` 重新開啟 front/front_oblique 分類。

更新後的 view metadata 分布：

| View type | Count |
| --- | ---: |
| rear_oblique | 1075 |
| rear | 410 |
| side | 138 |
| front_oblique | 0 |
| front | 0 |

輸出欄位包含：

- `video_id`
- `split`
- `view_type`
- `view_confidence`
- `front_score`
- `rear_score`
- `side_score`
- `oblique_score`
- `face_visibility_mean`
- `torso_width_ratio_mean`
- `orientation_score_mean`
- `z_asymmetry_mean`
- `valid_frame_ratio`

這個 view metadata 目前建議只作為 analysis metadata，不先加入 classifier。下一步可將 prediction CSV 與 `view_metadata.csv` 依 `video_id` join，分別計算不同 `view_type` 下的 balanced accuracy、recall、specificity、false positive rate 和 false negative rate。

### Combined pose-only by-view analysis

已使用 `data/Squat/pose_only/combined_pose_only_predictions.csv` 與 `data/Squat/Labeled_Dataset/view_metadata.csv` 做 view-aware prediction analysis。分析腳本為：

- `scripts/analyze_predictions_by_view.py`

輸出檔案：

```text
data/Squat/pose_only/combined_pose_only_by_view.csv
```

以下為 test split 的 selected-threshold metrics：

| View type | N | Positives | Negatives | Balanced accuracy | Recall | Specificity | Macro F1 | FP | FN | 解讀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ALL | 244 | 172 | 72 | 0.618 | 0.820 | 0.417 | 0.623 | 42 | 31 | 整體仍偏向抓錯誤，正常影片辨識有限。 |
| rear | 98 | 60 | 38 | 0.649 | 0.850 | 0.447 | 0.652 | 21 | 9 | 三種 view 中表現最好，recall 與 specificity 較平衡。 |
| rear_oblique | 132 | 104 | 28 | 0.595 | 0.798 | 0.393 | 0.590 | 17 | 21 | 斜後方視角下 balanced accuracy 下降，漏判增加且 specificity 偏低。 |
| side | 14 | 8 | 6 | 0.604 | 0.875 | 0.333 | 0.591 | 4 | 1 | 樣本太少，只能作為初步觀察。 |

固定 threshold 0.5 的 test balanced accuracy：

| View type | Balanced accuracy | Recall | Specificity |
| --- | ---: | ---: | ---: |
| ALL | 0.612 | 0.919 | 0.306 |
| rear | 0.621 | 0.900 | 0.342 |
| rear_oblique | 0.587 | 0.923 | 0.250 |
| side | 0.667 | 1.000 | 0.333 |

這個分析支持目前的視角假設：rear_oblique 佔 test split 多數，且其 selected-threshold balanced accuracy 比 rear 低約 0.054。rear_oblique 的 false negative 數量也較高，表示斜後方不只造成 false positive，也可能讓錯誤動作的幾何訊號變弱或更不穩定。

需要注意的是，side view 在 test split 只有 14 支影片，不能單獨得出穩定結論。後續若要判斷 knees_forward 是否真的需要側面視角，應增加人工視角標註或擴大 side-view 樣本。

目前建議保留的 baseline：

- VideoMAE-only：研究 baseline，代表泛用 RGB video embedding。
- Pose-only：目前較有 biomechanical 意義的 baseline。
- 後續 fusion：最值得作為下一階段主線。

### Multi-seed knees_forward / knees_inward by-view analysis

已將多 seed pose-only predictions 依 `view_metadata.csv` 分組分析。分析命令：

```bash
python scripts/analyze_predictions_by_view.py \
  --predictions-dir data/Squat/pose_only/predictions-20260510T034739Z-3-001/predictions \
  --view-metadata data/Squat/Labeled_Dataset/view_metadata.csv \
  --output data/Squat/pose_only/multiseed_pose_only_by_view.csv \
  --summary-output data/Squat/pose_only/multiseed_pose_only_by_view_summary.csv
```

輸出檔案：

```text
data/Squat/pose_only/multiseed_pose_only_by_view.csv
data/Squat/pose_only/multiseed_pose_only_by_view_summary.csv
```

以下為 test split、selected-threshold、5 seeds mean +/- std。

| Label mode | View type | N | Pos / Neg | Balanced accuracy | Recall | Specificity | Macro F1 | 解讀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| knees_forward | ALL | 244 | 169 / 75 | 0.573 +/- 0.010 | 0.824 +/- 0.027 | 0.323 +/- 0.026 | 0.575 +/- 0.011 | 整體 recall 高，但正常影片辨識弱。 |
| knees_forward | rear | 98 | 58 / 40 | 0.571 +/- 0.022 | 0.783 +/- 0.037 | 0.360 +/- 0.046 | 0.566 +/- 0.024 | specificity 比 rear_oblique 稍好。 |
| knees_forward | rear_oblique | 132 | 103 / 29 | 0.571 +/- 0.027 | 0.839 +/- 0.035 | 0.303 +/- 0.026 | 0.575 +/- 0.031 | recall 較高，但 false positives 較多。 |
| knees_forward | side | 14 | 8 / 6 | 0.546 +/- 0.097 | 0.925 +/- 0.100 | 0.167 +/- 0.105 | 0.495 +/- 0.113 | 樣本太少且 specificity 很差，暫不下結論。 |
| knees_inward | ALL | 244 | 36 / 208 | 0.570 +/- 0.021 | 0.478 +/- 0.054 | 0.662 +/- 0.080 | 0.516 +/- 0.035 | 整體受 positive 少影響，漏判仍明顯。 |
| knees_inward | rear | 98 | 15 / 83 | 0.618 +/- 0.059 | 0.573 +/- 0.053 | 0.663 +/- 0.078 | 0.549 +/- 0.059 | rear 明顯優於 rear_oblique。 |
| knees_inward | rear_oblique | 132 | 20 / 112 | 0.539 +/- 0.029 | 0.420 +/- 0.087 | 0.657 +/- 0.091 | 0.497 +/- 0.037 | 斜後方下 inward recall 下降，較容易漏判。 |
| knees_inward | side | 14 | 1 / 13 | 0.446 +/- 0.183 | 0.200 +/- 0.400 | 0.692 +/- 0.069 | 0.422 +/- 0.054 | positive 只有 1 支，不能作為視角結論。 |

結論：

- `knees_inward` 有較明顯的 view effect。rear 的 test balanced accuracy 約 `0.618`，rear_oblique 約 `0.539`，差距約 `0.079`。主要差異來自 recall：rear 約 `0.573`，rear_oblique 約 `0.420`，表示斜後方視角較容易漏掉 inward 錯誤。
- `knees_forward` 沒有看到穩定的 view-type 幫助。rear 與 rear_oblique 的 balanced accuracy 幾乎相同，都是約 `0.571`；rear_oblique recall 較高，但 specificity 較低，代表它更偏向預測為錯誤，不一定是真正看得更準。
- side view 目前不能用來回答 knees_forward 是否會更好，因為 test split 只有 14 支 side 影片，且 knees_forward side 的 negatives 只有 6 支，knees_inward side 的 positives 甚至只有 1 支。
- 以目前資料來看，view metadata 最有價值的用途是做分層分析與錯誤診斷；還不建議直接把 `view_type` 加進 classifier 當 feature，因為 side/front 樣本不足，容易讓模型學到資料分布偏差。

### Pose-only train-set normalization experiment

已完成 pose-only feature normalization 實驗。這次在 classifier pipeline 使用 `--normalize-features`，每個 seed 都只用 train split 的 pose-only `video_feature` 計算 mean/std，並將同一組統計套用到 train/val/test。訓練紀錄在：

```text
notebooks/run_videomae_video_classifier.ipynb
data/Squat/pose_only_normalization_metrics/metrics/experiment_summary.csv
```

以下比較未 normalization baseline 與 train-set normalization。數字為 test split、selected-threshold、5 seeds mean +/- std。

| Label mode | Setting | Balanced accuracy | Recall | Specificity | Macro F1 | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| combined | baseline | 0.581 +/- 0.032 | 0.790 +/- 0.071 | 0.372 +/- 0.068 | 0.582 +/- 0.035 | 0.768 +/- 0.039 |
| combined | normalized | 0.635 +/- 0.010 | 0.717 +/- 0.101 | 0.553 +/- 0.111 | 0.622 +/- 0.020 | 0.750 +/- 0.053 |
| knees_forward | baseline | 0.573 +/- 0.011 | 0.824 +/- 0.030 | 0.323 +/- 0.029 | 0.575 +/- 0.012 | 0.775 +/- 0.014 |
| knees_forward | normalized | 0.615 +/- 0.030 | 0.714 +/- 0.136 | 0.517 +/- 0.190 | 0.599 +/- 0.026 | 0.735 +/- 0.055 |
| knees_inward | baseline | 0.570 +/- 0.023 | 0.478 +/- 0.060 | 0.662 +/- 0.089 | 0.516 +/- 0.039 | 0.280 +/- 0.020 |
| knees_inward | normalized | 0.608 +/- 0.054 | 0.578 +/- 0.188 | 0.637 +/- 0.148 | 0.526 +/- 0.064 | 0.315 +/- 0.064 |

整體結論：

- normalization 對三個 label mode 的 selected-threshold test balanced accuracy 都有提升：combined `+0.054`、knees_forward `+0.042`、knees_inward `+0.038`。
- combined 與 knees_forward 的主要改善來自 specificity 提升，表示原本 pose-only classifier 偏向過度預測 positive；normalization 讓正常影片辨識變好，但 recall 與 F1 有所下降。
- knees_inward 則同時提升 balanced accuracy、recall 與 F1，specificity 小幅下降。這表示 normalization 對少數 positive 類別的 inward 錯誤較有幫助，但 seed 間變異仍偏大。
- normalization 應保留為 pose-only baseline 的預設實驗設定；下一步比較 VideoMAE-plus-pose fusion 時，也應明確記錄 pose branch 是否使用 train-set normalization，避免和未 normalization baseline 混在一起比較。
