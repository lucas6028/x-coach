# Stage A 結果:REHAB24-6 corrected-pooling VideoMAE

對應計畫:[`videomae_dataset_validation_plan.md`](videomae_dataset_validation_plan.md) Stage A。

---

## 0. 事前登錄(pre-registration)

**本節在任何結果產生前寫定。** 四臂設計會同時給出三個對 legacy 的 delta,
若等看到數字再挑一個當「corrected」,等於用 best-of-three 冒充單一假設檢定。

- **Primary arm:`videomae_mean_pool_fc_norm_mean`**
- Secondary / diagnostic:`videomae_mean_pool_fc_norm_max`、`videomae_legacy_first_token_mean`
- Baseline(paired 比較的分母):`videomae_legacy_first_token_max`,即歷史抽取方式的重現

選 `mean` 作為 primary 的理由與結果無關,先寫在這裡:多 clip 的影片分類慣例是對
clip 取平均;且 `fc_norm` 之後每個 clip 向量已是零均值、單位變異,對 clip 取 max 會
讓每個維度都被推向正值,正好抵銷剛套用的 normalization(論證見
`src/video/videomae_pooling.py` docstring)。

計畫的四個通過條件依此判定,primary arm 對 baseline 的 paired delta 為準。

### 兩個必須連同結果一起讀的前提

1. **Legacy 是重新抽取,不是封存數字。** 舊的 `videomae_features/` npz 已不存在於本機,
   且 transformers 版本已不同。歷史基線為 LOSO balanced accuracy `0.536 ± 0.044`。
   若本次重現落在此區間外,delta 的分母就換了,必須明講,不能默默吸收進 delta。
2. **cam18 的預處理不對稱。** cam17 為 1920x1080 橫向、cam18 為 1080x1920 直向,
   受試者在兩者中都是直立的(`-transposed` 指檔案已校正,不是旋轉 bug)。但
   `VideoMAEImageProcessor` 的 shortest_edge=224 + 224 center crop 會把 cam18 縮放後
   398px 高的畫面裁到中間 224 列,**切掉受試者的腳**;cam17 則保留全身。這是視角之間
   真實存在的預處理落差。**本次刻意不修**:改動 crop 會讓 legacy-vs-corrected 的
   paired 比較同時變動兩個因素。它是通過條件 4(視角分層)的解讀警告,也是 corrected
   若表現不佳時的第一個候選 lever。

---

## 1. 方法與實作

### 1.1 修正的是兩個 pooling,不是一個

計畫指出的 token pooling 錯誤已在安裝的 transformers 原始碼確認:
`transformers/models/videomae/modeling_videomae.py` 全檔**沒有任何 `cls_token`**,
encoder 輸出為 `(batch, 1568, 768)` = 8 tubelets x 196 patches,沒有前置的 summary
token。因此 `last_hidden_state[:, 0, :]` 是第一個 tubelet 的左上角 patch,不是 clip
representation。分類路徑實際做的是 `sequence_output.mean(1)` 再套 `fc_norm`(第 745-751 行)。

`fc_norm` 只存在於 `VideoMAEForVideoClassification`:當 checkpoint 的
`use_mean_pooling=True` 時,`VideoMAEModel.layernorm` 是 `None`(第 415-418 行),
所以單獨載入 `VideoMAEModel` 無法重現分類表徵。抽取端因此改載入分類模型,取其
`.videomae` backbone 與 `.fc_norm` 參數。

**計畫未提到的第二個 pooling:對 clip 取 max**(舊 `videomae_features.py:98`)。
理由見上方 pre-registration。處理方式:抽取階段**存下完整的 per-clip stack**,
把 clip aggregation 留成離線可選的軸,確保 token pooling 與 clip aggregation 這兩個
改動不會落在同一個測得的 delta 裡。

### 1.2 一次 forward,兩種 pooling

`legacy_first_token` 與 `mean_pool_fc_norm` 由**同一個** `last_hidden_state` 計算,
因此兩臂在 frames、weights、clip sampling、transformers 版本上是**結構上相同**的,
paired delta 只反映 pooling 修正。這也讓本機與 Kaggle 之間任何版本差異對兩臂等量作用。

### 1.3 Null control 在 repetition 層級置換

`permute_labels_within_subject` 在**每位受試者內、以 repetition 為單位**打亂標籤,
並把置換後的標籤廣播到該 rep 的兩個相機列。

- 保持每位受試者的正例率不變:每個 LOSO fold 只測一位受試者,固定其 class balance,
  null 的 chance level 就與真實 run 相同。
- 以 repetition 而非 sample 為單位:已驗證 REHAB24-6 共 1072 個 rep,**每個都恰好有
  2 個相機列,且沒有任何一個 rep 的兩列標籤不一致**。若逐 sample 打亂,同一個 rep 的
  cam17 與 cam18 會拿到相反標籤——這是真實訓練集永遠不會出現的矛盾,會讓 null 變得
  不公平地困難,進而**灌水 real-minus-null 的差距**,而那正是通過條件 3 的依據。

計畫列了兩種 null(shuffled labels / permuted VideoMAE features)。此處只實作前者:
在本設定下兩者破壞的是同一個對應關係、保留的是同一組邊際分布,故視為等價,不另跑。

### 1.4 產物與檢核

| 步驟 | 指令 |
| --- | --- |
| 抽取(GPU) | `scripts/rehab24/extract_videomae_features.py` → `videomae_raw/` |
| 物化四臂 | `scripts/rehab24/materialize_videomae_features.py` |
| 稽核 | `scripts/rehab24/audit_videomae_features.py <dir>...` |
| Stage A 證據 | `scripts/rehab24/videomae_stage_a.py` |

每個 `.npz` 都蓋上 provenance(model name、pooling、clip length、stride、num clips、
transformers version),稽核會在覆蓋率不全、stem 重複、維度/dtype 混雜、NaN/Inf、
split 對不上、或 provenance 不一致時以非零碼結束,用來 gate 後續的 LOSO。

### 1.5 抽取執行紀錄

| 項目 | 值 |
| --- | --- |
| 執行環境 | Kaggle kernel `haoping6028/rehab24-videomae-extract` v2 |
| 實際裝置 | **CPU**(見下)|
| 耗時 | 6.71 h |
| 產出 | 2144 / 2144 bundles(train 1434 / val 212 / test 498),57.2 MB |
| 模型 | `MCG-NJU/videomae-base-finetuned-kinetics`,clip 16 / stride 2 / 4 clips |
| transformers | Kaggle 5.0.0(本機 smoke 為 5.5.0)|
| `fc_norm` | weight mean 0.6832、bias mean 0.0083 —— 與本機載入**完全一致** |

**為何是 CPU:** Kaggle 兩次都配給 Tesla P100(sm_60),而該映像的
torch 2.10.0+cu128 只內建 sm_70 以上的 kernel,`--accelerator "GPU T4 x2"` 未生效。
第一次執行因此在第一個 `conv3d` 就以
`no kernel image is available for execution on the device` 崩潰。v2 改為在載入模型前先用
一個真實的 `conv3d` 探測 GPU,失敗即退回 CPU,才讓這次跑完。

**這不影響 paired 比較**:兩種 pooling 來自**同一次 forward**,裝置與 transformers 版本
對兩臂等量作用。同理,Kaggle 5.0.0 與本機 5.5.0 的差異也不會進入 delta。

### 1.6 兩個資料面的小發現

1. **2 / 2144 個樣本的 4 個 clip 完全相同(0.1%)。** `Ex5_PM_042_rep21` 的兩個相機列,
   該 repetition 只有 15 frames,短於 clip window(1 + 2 x 15 = 31 frames),所以四個
   clip start 全部塌到同一格。屬合理邊界情況,已確認其餘 2142 個樣本都是 4 個相異
   clip start。對這 2 個樣本而言 max 與 mean aggregation 恆等。
2. **repetition 長度**:min 15 / p5 66 / median 110 / max 585 frames;短於 clip window 的
   只有上述 2 列。

---

## 2. 結果

來源:`data/REHAB24-6/processed/videomae_stage_a.json`,seed 42,9 folds(排除 n=16 的 P10)。

### 2.1 四臂 LOSO balanced accuracy

| Arm | token pooling | clip agg | bal_acc (9 folds) | vs baseline |
| --- | --- | --- | --- | --- |
| `legacy_first_token_max` | first token | max | **0.533 ± 0.048** | baseline |
| `legacy_first_token_mean` | first token | mean | 0.584 ± 0.088 | +0.051 (7/9, p=0.074) |
| `mean_pool_fc_norm_max` | mean+fc_norm | max | 0.638 ± 0.056 | +0.104 (9/9, p=0.004) |
| **`mean_pool_fc_norm_mean`**(primary) | mean+fc_norm | mean | **0.657 ± 0.049** | **+0.124 (9/9, p=0.004)** |

**Legacy 重現成功:** 0.533 ± 0.048 對上歷史的 0.536 ± 0.044,幾乎重疊。計畫沒寫那個
數字用的是 9 folds 還是 10 folds,所以兩種算法都列:9 folds 0.533 ± 0.048、
10 folds 0.548 ± 0.064,歷史值 0.536 落在兩者之間。**兩種慣例下重現都成立**,
delta 的分母是可信的,§0 提的第一個前提解除——+0.124 不是因為基線換了。

**兩個 pooling 軸的分解:**

- 只修 token pooling(仍用 max):+0.104
- 只改 clip aggregation(仍用 first token):+0.051
- 兩者都修:+0.124

Token pooling 是主要修正。誠實地說:**即使只修 token pooling、保留 max,結果仍是
0.638,一樣會通過 Stage A**。所以把 clip aggregation 拆成獨立軸並沒有「拯救」這個研究,
它的價值在於讓 +0.124 能被歸因到正確的軸,而不是把兩個改動混成一個數字。

### 2.2 每位受試者的 paired delta(primary vs baseline)

| Subject | legacy | corrected | delta |
| --- | --- | --- | --- |
| P1 | 0.520 | 0.643 | +0.124 |
| P2 | 0.529 | 0.730 | +0.200 |
| P3 | 0.503 | 0.549 | +0.046 |
| P4 | 0.503 | 0.674 | +0.171 |
| P5 | 0.521 | 0.636 | +0.115 |
| P6 | 0.658 | 0.687 | +0.029 |
| P7 | 0.512 | 0.642 | +0.130 |
| P8 | 0.496 | 0.652 | +0.156 |
| P9 | 0.559 | 0.704 | +0.144 |

**9/9 為正**,非由單一受試者拉高。

### 2.3 Null control(repetition 層級置換,3 個 seed)

| | balanced accuracy |
| --- | --- |
| 真實標籤 | 0.657 |
| 置換標籤 | 0.497(seeds 101/202/303:0.510 / 0.500 / 0.481)|
| gap | **+0.160** |

Null 精準落在 0.50,符合設計:每位受試者的正例率被保留,chance level 未被動過。

### 2.3b 跨 seed 重複(baseline 與 primary)

事前登錄的 seed 是 42。事後補跑兩個 seed 檢查穩定性:

| seed | legacy_max | corrected_mean | delta | 為正 folds | p |
| --- | --- | --- | --- | --- | --- |
| 42(pre-registered)| 0.533 | 0.657 | **+0.124** | 9/9 | 0.004 |
| 7 | 0.556 | 0.643 | +0.087 | 8/9 | 0.012 |
| 1234 | 0.571 | 0.652 | +0.081 | 8/9 | 0.008 |

方向在三個 seed 全為正、都顯著,§3.2 原先列的「只有一個 seed」保留意見解除。

但要照實說:**seed 42 的 +0.124 是三者中最有利的一個**,跨 seed 平均約 +0.097。
差異幾乎全來自 legacy 基線(0.533 / 0.556 / 0.571)——corrected 反而很穩
(0.657 / 0.643 / 0.652)。也就是說,**錯誤 pooling 的那一臂對 seed 較敏感**,
corrected 較穩定。引用時應以 +0.08~0.12 這個區間、而非單一 +0.124 為準。

### 2.4 分層(delta vs baseline)

| Camera | mean delta | 為正的 folds |
| --- | --- | --- |
| cam17(正面、全身)| +0.104 | 9/9 |
| cam18(矢狀、腳被裁掉)| +0.144 | 9/9 |

| Exercise | mean delta | 為正的 folds |
| --- | --- | --- |
| arm abduction | +0.131 | 9/9 |
| leg lunge | +0.141 | 7/7 |
| table push-ups | +0.115 | 5/7 |
| leg abduction | +0.106 | 7/9 |
| squats | +0.098 | 5/9 |
| arm VW | +0.094 | 8/9 |

六個動作的平均 delta 全為正,兩個視角也都 9/9。**值得注意:被裁掉腳的 cam18 反而
獲益更多(+0.144 vs +0.104)**,所以 §0 記的 crop 不對稱並沒有壓制修正效果——它仍是
未來的 lever,但不是本次結論的威脅。一致性最弱的是 squats(5/9),平均仍為 +0.098。

---

## 3. 對照計畫的四個通過條件

| # | 條件 | 結果 | 判定 |
| --- | --- | --- | --- |
| 1 | corrected 相對 legacy 的 paired delta 為正 | +0.124 ± 0.052,p=0.004(跨 3 seeds:+0.124/+0.087/+0.081,全部顯著)| **通過** |
| 2 | 改善出現在多數 held-out subjects | 9/9 folds 為正(其他 seed 8/9)| **通過** |
| 3 | 明顯優於 shuffled/permuted null | 0.657 vs 0.497,gap +0.160 | **通過** |
| 4 | 效果不限於單一背景/camera/exercise | camera 2/2 皆 9/9 為正;exercise 6/6 平均 delta 為正。**背景未檢驗**——REHAB24-6 只有單一實驗室場景,無法分層,計畫把背景對照放在 Stage B | **部分通過**(camera 與 exercise 通過,背景無法在本資料集檢驗)|

**Stage A 通過,建議進入 Stage B(Fitness-AQA)。** 唯一的保留是條件 4 的「背景」那一半
在本資料集結構上無法回答,而非測了沒過。

### 3.1 這改變了什麼結論

舊的「VideoMAE LOSO 0.536,幾乎等於隨機」**不能再用來說 VideoMAE 沒有動作品質訊號**。
那個數字量到的是「第一個 patch token 的分類能力」,不是 clip representation。修正後
0.657,相對 Vicon skeleton 的 ~0.702 已在同一量級,而不是先前那種「一個有訊號、一個沒有」
的對比。

### 3.2 仍未回答的問題(留給 Stage B 及之後)

1. **這 0.657 有多少是動作品質、有多少是場景/視角捷徑?** REHAB24-6 是單一實驗室場景,
   本階段沒有 background-only 或 person-crop 對照——計畫把那個對照放在 Stage B,
   在此之前不能宣稱模型學到的是動作而非場景。
2. **與 skeleton 是否互補?** 本階段只做 VideoMAE-only 與 null,未做
   skeleton-only / skeleton+VideoMAE 的 fusion 對照(計畫實驗矩陣的後兩列)。
   0.657 < 0.702 只說明單獨用比不上,不代表沒有增量資訊。
3. **cam18 的 center crop 切掉雙腳**仍未修。它現在是個「已知會損失資訊、但效果依然出現」
   的因素,修掉後 corrected 應該只會更好——這是 Stage B 前可低成本一試的 lever。
4. ~~只有一個 seed~~ —— 已於 §2.3b 補跑 seed 7 與 1234,方向一致且都顯著。改為:
   **引用時應用 +0.08~0.12 的區間,而非 +0.124 單值**。

### 3.3 對既有基線程式碼的修改(要重跑舊結果的人必讀)

為讓 70 個 fold 的評估在合理時間內跑完,改了 `src/video/videomae_video_classifier.py`
兩處**共用**程式碼。**數值不變**:已用同一組合成特徵驗證改動前後**每個 fold 完全相同**
(`stage_a_dryrun.json` vs `stage_a_dryrun_v2.json`,逐 fold 比對 bit-identical),
全套 1932 個測試通過。

1. `load_video_feature` 加上以路徑為 key 的記憶體快取,`FeatureDataset` 改走這條路徑。
   原本每個 epoch 都重讀每個 `.npz`(每 fold 約 34k 次,每次約 2.6 ms,幾乎全是 zip
   容器開檔成本而非解壓)。快取**回傳副本**,因為 `torch.from_numpy` 與傳入陣列共用記憶體。
2. `build_samples` 改用一次建好的 stem→path 索引,取代「每個 sample id 各跑一次 `rglob`」
   (2144 樣本的目錄約 25 s/次,每個 fold 呼叫三次)。實測 25 s → 0.11 s。

**行為上唯一的差別**:遇到重複 stem 時,原本取目錄走訪順序的第一個(跨平台不保證一致),
現在取排序後的第一個。重複 stem 本來就是稽核會擋下的缺陷。

repo 內既有的 `correctness_loso_*.json` 都是舊 loader 產生的;數值應可重現,但知道這個
改動存在比較安全。
