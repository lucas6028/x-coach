# VideoMAE 分類器實驗摘要

## 背景

目前專案使用已快取的 VideoMAE 影片層級特徵，訓練一個輕量 MLP 分類器來判斷深蹲錯誤。

一開始的標籤是把兩種錯誤合併成同一個二元分類任務：

- 膝蓋前移
- 膝蓋內夾

只要影片有其中任一錯誤，就會被標成正類。

最初重複訓練多次時，測試結果差異很大，因此後續主要檢查訓練穩定性、checkpoint 選擇、threshold 選擇，以及類別不平衡問題。

最初三次測試的 F1 約落在 0.69 到 0.81，表面上看起來有些 run 不錯，但 accuracy 約 0.59 到 0.70，差異偏大，代表結果不穩定。

## 發現的主要問題

原本的分類器評估流程有幾個問題：

- 沒有固定 seed，所以訓練結果不穩定。
- 程式雖然儲存最佳 validation checkpoint，但最後測試時用的是最後一個 epoch 的模型。
- threshold 固定為 0.5，無法清楚觀察 precision 和 recall 的取捨。
- 資料集偏向正類，單看 positive-class F1 容易誤判模型表現。
- 模型可以透過幾乎全部預測為「有錯誤」來得到看似不錯的 F1。

最重要的發現是：高 F1 不代表模型真的能分辨正常深蹲。有些結果雖然能抓到大部分錯誤影片，但幾乎無法正確辨識正常影片。

例如某次測試中，模型的 recall 接近 0.99、F1 約 0.83，但 specificity 只有約 0.04。這代表模型幾乎抓到所有錯誤影片，卻也把大多數正常影片誤判成錯誤。這種結果不適合作為實際教練回饋。

## 已完成的修改

第一階段先修正可重現性與 checkpoint 評估：

- 加入 seed 參數。
- 固定 Python、NumPy、PyTorch、CUDA 和 training DataLoader 的隨機性。
- 最終測試前重新載入 validation 表現最佳的 checkpoint。
- 在 checkpoint 中儲存選定的 threshold 與 validation metrics。

第二階段改善評估方式：

- 加入 specificity 與 false positive rate。
- 加入 balanced accuracy 與 macro F1。
- 加入 always-positive 與 always-negative baseline。
- 同時回報固定 threshold、F1 選出的 threshold，以及指定目標選出的 threshold。
- 將預設 threshold objective 改成 balanced accuracy。

第三階段加入較完整的實驗流程：

- 可調整 dropout。
- 可調整 weight decay。
- 加入 early stopping。
- 匯出 prediction CSV，方便檢查 false positive 和 false negative。

第四階段加入多 seed 與多標籤實驗：

- 支援 combined、knees_forward、knees_inward 三種 label mode。
- 每次實驗輸出 metrics JSON。
- 加入 grid runner，跑三種 label mode 和五個 seed。
- 產生整體 experiment summary CSV。

## 實驗觀察

原本看起來不錯的 F1，主要是因為模型太常預測正類。

always-positive baseline 的 test F1 約 0.83，已經接近某些模型結果。因此如果只看 F1，模型看起來會被高估。

改用 balanced accuracy 並加入 regularization 後，模型對正常深蹲的辨識有改善，但泛化能力仍然偏弱。

加入 regularization 與 early stopping 後，combined classifier 的某次 selected threshold 測試結果約為 balanced accuracy 0.59、specificity 0.65、recall 0.54。這比原本幾乎全判正類的情況好，但仍不算強。

多 seed 實驗後的觀察：

- combined label classifier 是目前較合理的 baseline。
- 分開訓練 knees_forward 和 knees_inward 並沒有穩定優於 combined。
- knees_inward 的正負樣本非常不平衡，因此結果不穩定。
- 單次最佳結果看起來較好，但跨 seed 平均後改善有限。

整體來看，目前 VideoMAE feature classifier 用 balanced accuracy 評估時，只比隨機基準稍好一些。

多 seed grid 中，combined classifier 的 test balanced accuracy 大約在 0.53 到 0.58 左右。knees_forward 多數也落在約 0.51 到 0.56。knees_inward 有單次 selected threshold 約 0.61，但其他 seed 常落在約 0.48 到 0.55，顯示不穩定。

目前 repo 中的 `videomae_feature_classifier/metrics/experiment_summary.csv` 已保留這次 VideoMAE feature classifier 的完整 grid result。以下數字使用 test split 的 `selected_threshold`，也就是 validation balanced accuracy 選出的 threshold：

| Label mode | Test balanced accuracy 平均 | 範圍 | Specificity 平均 | Recall 平均 | 解讀 |
| --- | ---: | ---: | ---: | ---: | --- |
| combined | 0.555 | 0.532-0.584 | 0.647 | 0.463 | 目前最合理的 VideoMAE-only baseline，但仍只略高於隨機基準。 |
| knees_forward | 0.524 | 0.509-0.541 | 0.573 | 0.475 | 分開訓練後沒有優於 combined，整體訊號偏弱。 |
| knees_inward | 0.539 | 0.483-0.608 | 0.728 | 0.350 | 單一 seed 可到 0.608，但 recall 偏低且跨 seed 不穩。 |

若使用固定 threshold 0.5，F1 常會變高，但 specificity 與 recall 的取捨不穩定。例如 combined seed 2 到 seed 4 的 recall 約 0.78-0.90，但 specificity 只有約 0.15-0.39。這再次確認本實驗不應以 positive-class F1 作為主要結論。

## 目前建議使用的 Baseline

目前 baseline 建議設定為：

- label mode: combined
- threshold objective: balanced accuracy
- learning rate: 3e-4
- hidden dimension: 128
- dropout: 0.4
- weight decay: 0.01
- early stopping patience: 5
- 主要評估指標: balanced accuracy、macro F1、specificity、recall

固定 threshold 0.5 仍然應該保留作為 sanity check。

目前 baseline 的目標不是追求最高 F1，而是避免模型只會預測「有錯誤」。因此報告時應優先看 balanced accuracy、macro F1、specificity 和 recall 的平衡。

## 主要修改檔案

主要實作檔案：

- `src/video/videomae_video_classifier.py`
- `scripts/video/run_videomae_experiment_grid.py`
- `notebooks/run_videomae_video_classifier.ipynb`

Colab 實驗會產生的主要輸出：

- 每次實驗的 checkpoint
- 每次實驗的 prediction CSV
- 每次實驗的 metrics JSON
- `experiment_summary.csv`

## 建議下一步

下一步不建議只繼續微調同一個 MLP。從目前結果來看，單一 VideoMAE 影片 embedding 的訊號可能不足。

建議後續工作：

- 檢查 prediction CSV 中高信心的 false positive 和 false negative。
- 對 VideoMAE 特徵做 normalization。
- 嘗試 clip-level 或 temporal aggregation，而不是只用單一影片 embedding。
- 加入 pose 或幾何特徵，特別是針對膝蓋錯誤。
- 比較 VideoMAE-only、pose-only、VideoMAE-plus-pose 三種設定。
- 在特徵與標籤問題釐清後，再考慮 fine-tune VideoMAE。

可用於報告的結論：

目前 VideoMAE feature classifier 的評估流程已經比較可靠，但模型泛化能力仍然偏弱。分開訓練 knees_forward 和 knees_inward 尚未穩定優於 combined classifier。下一步應優先做錯誤分析與特徵改進，而不是只繼續調整分類器本身。

用數字描述時，可以說目前多數設定的 test balanced accuracy 約在 0.53 到 0.59，僅略高於隨機基準 0.50；因此目前模型較適合作為研究 baseline，還不適合作為可靠的最終深蹲錯誤判斷器。

後續已完成 pose-only baseline，結果整理在 `notes/pose_only_classifier_experiment_summary.md`。pose-only 的 combined test balanced accuracy 平均約 0.581、最高約 0.618，略優於 VideoMAE-only，但仍不足以作為可靠的最終模型。下一階段建議比較 VideoMAE-only、pose-only、VideoMAE-plus-pose，並優先做 feature normalization、錯誤案例分析與 temporal modeling。

## 重要指標變化總結

以下整理實驗過程中幾個重要階段的指標變化，作為後續報告與比較依據。

| 實驗階段 | 主要設定 | 關鍵指標 | 觀察 |
| --- | --- | --- | --- |
| 初始三次訓練 | 未固定完整 seed，且測試可能使用最後 epoch 模型 | test F1 約 0.69 到 0.81；accuracy 約 0.59 到 0.70 | 表面上有些 run 不錯，但不同 run 差異大，代表流程不穩定。 |
| 修正 seed 與 checkpoint 評估後 | 載入 validation 最佳 checkpoint，並調整 threshold | 某次 test F1 約 0.83；recall 約 0.99；specificity 約 0.04 | F1 很高，但幾乎無法辨識正常影片，模型偏向全部判成有錯誤。 |
| 加入 baseline 與 specificity | 回報 always-positive、always-negative、specificity、balanced accuracy | always-positive baseline test F1 約 0.83；balanced accuracy 為 0.50 | 證明高 F1 可能只是因為資料偏正類，不代表模型真的有效。 |
| 改用 balanced accuracy 選 threshold | threshold objective 改為 balanced accuracy | 某次 selected threshold test specificity 約 0.63；recall 約 0.42；balanced accuracy 約 0.53 | threshold 可降低 false positive，但 recall 下降，整體泛化仍弱。 |
| 加入 regularization 與 early stopping | lr 3e-4、hidden dim 128、dropout 0.4、weight decay 0.01、patience 5 | combined 單次 test balanced accuracy 約 0.59；specificity 約 0.65；recall 約 0.54 | 模型不再只是預測正類，正常影片辨識改善，但仍不是強模型。 |
| 多 seed / 多 label mode grid | combined、knees_forward、knees_inward，各跑 5 個 seed | selected threshold test balanced accuracy：combined 平均 0.555、範圍 0.532-0.584；knees_forward 平均 0.524、範圍 0.509-0.541；knees_inward 平均 0.539、範圍 0.483-0.608 | 分開訓練錯誤類型尚未穩定優於 combined classifier。 |

各 label mode 的整體比較如下：

| Label mode | Test balanced accuracy 大致範圍 | 穩定性 | 解讀 |
| --- | --- | --- | --- |
| combined | 0.532 到 0.584，平均 0.555 | 相對穩定 | 目前較合理的 baseline，但只略高於隨機基準 0.50。 |
| knees_forward | 0.509 到 0.541，平均 0.524 | 普通 | 沒有穩定優於 combined。 |
| knees_inward | 0.483 到 0.608，平均 0.539 | 不穩定 | 類別非常不平衡，單次好結果不足以代表泛化能力。 |

各指標在本實驗中的用途如下：

| 指標 | 用途 | 本實驗中的判讀 |
| --- | --- | --- |
| F1 | 衡量正類預測表現 | 容易被 positive-heavy dataset 高估，不能單獨作為主要指標。 |
| Recall | 錯誤影片抓到多少 | 高 recall 可能伴隨大量 false positive。 |
| Specificity | 正常影片辨識能力 | 是判斷模型是否只會預測有錯誤的關鍵指標。 |
| Balanced accuracy | 同時考慮 recall 與 specificity | 比 accuracy 和 positive-class F1 更適合目前不平衡資料。 |
| Macro F1 | 同時考慮正類與負類 F1 | 可輔助觀察模型是否只偏向某一類。 |

整體實驗依據可以總結如下：

- F1 高不一定代表模型好，因為 always-positive baseline 的 F1 已經約 0.83。
- specificity 是判斷模型是否能辨識正常影片的關鍵指標。
- balanced accuracy 比 accuracy 和 positive-class F1 更適合目前這個不平衡資料集。
- regularization 與 early stopping 有幫助，但提升有限。
- 多 seed 平均後，模型大多只比隨機基準 0.50 高一些。
- 目前瓶頸較可能在特徵與標籤設計，而不是 MLP 分類器本身。
