# REHAB24-6 VideoMAE framing 驗證計畫

> 狀態：**事前登錄草案，尚未執行**  
> 建立日期：2026-08-15  
> 對象：REHAB24-6 correctness 二元分類  
> Primary representation：`mean_pool_fc_norm_mean`（corrected VideoMAE pooling）

## 一句話決策

REHAB24-6 至少應跑 `full_frame_letterbox`，因為目前 `full_frame` 會讓
`VideoMAEImageProcessor` 對直向 cam18 做 center crop，並可能裁掉腳部。若要在論文中把
目前的 `0.657 ± 0.049` 解讀為人體動作品質訊號，而不只是資料集捷徑，則再跑
`background_only`；`person_crop` 是完整拆解人體與背景所需的第三優先臂。

本計畫不是新的模型搜尋，也不重開 pooling、classifier hyperparameter 或 threshold objective。
唯一要改變的是輸入 framing，其餘條件保持成對一致。

## 1. 已知證據與尚未回答的問題

### 1.1 REHAB24-6 已知結果

corrected VideoMAE primary arm `mean_pool_fc_norm_mean` 在排除只有 16 個 sample 的 P10 後，
9-fold LOSO balanced accuracy 為：

| 設定 | balanced accuracy | macro-F1 |
| --- | ---: | ---: |
| `full_frame`，corrected pooling | **0.657 ± 0.049** | 0.646 ± 0.052 |
| 舊 `legacy_first_token_max` | 0.533 ± 0.048 | 0.494 ± 0.077 |

因此舊的 `0.536 ± 0.044` 已不能代表 VideoMAE 的能力；本計畫只以 corrected pooling
作為主分析，不再重新檢驗已通過的 pooling 修正。

### 1.2 目前的 construct-validity 缺口

現有 `full_frame` 並不等於模型看見完整畫面。processor 先把 shortest edge resize 到 224，
再做 224×224 center crop：

- cam17 是橫向畫面，通常保留完整身體；
- cam18 是直向畫面，縮放後高度大於 224，center crop 可能裁掉腳部；
- squat correctness 的主要訊號之一正是腳、踝、膝與垂直位移，因此這不是純外觀差異。

此外，REHAB24-6 只有一個實驗室、兩台固定相機。LOSO 可以阻止同一受試者跨 train/test，
卻不能單靠 split 排除模型使用相機、場景、動作位置、片段長度或人物框幾何。現有
`box_geometry` LOSO 約 `0.528 ± 0.028`，已降低幾何捷徑疑慮，但仍不能替代 pixel-level
`background_only` control。

### 1.3 Fitness-AQA 只能作先驗，不能代替本實驗

Fitness-AQA 的 5×5 repeated splits 已得到：

| arm | balanced accuracy |
| --- | ---: |
| `full_frame` | 0.6706 ± 0.0086 |
| `full_frame_letterbox` | 0.6691 ± 0.0149 |
| `person_crop` | 0.6555 ± 0.0036 |
| `background_only` | 0.6238 ± 0.0088 |

該資料集顯示 letterbox 沒有改善、person crop 略降，而且 background-only 的大部分訊號後來
可由片段長度解釋。但 Fitness-AQA 缺少 participant mapping，場景也比 REHAB24-6 多樣；
它只能設定預期方向，不能回答 REHAB24-6 在 subject-wise LOSO 下的 framing 問題。

## 2. 研究問題與假設

### Primary：F2，完整身體是否修正 cam18 的 preprocessing 缺陷？

比較：`full_frame_letterbox − full_frame`。

- `full_frame`：原始畫面直接交給 processor，保留既有 baseline。
- `full_frame_letterbox`：整張畫面以中性灰補成正方形，不移除背景、不扭曲長寬比，
  使 processor 的 center crop 成為 no-op。

事前預期：整體效果可能小，但若裁腳確實傷害 correctness，增益應主要出現在 cam18，並可能
集中於 Ex4–Ex6 等下肢動作。若 cam17 與 cam18 都同向改善，則不能只歸因於修正直向裁切。

### Secondary：F1，移除背景後是否保留訊號？

比較：`person_crop − full_frame_letterbox`，而不是只比較 `person_crop − full_frame`。

兩者都保留完整身體；前者大幅移除場景，後者保留場景。這是本設計中最接近「背景有無」的
對比，但仍有不可完全消除的 body-scale 與框長寬比差異，因此只能支持「framing arm 的效果」，
不能單憑正 delta 宣稱背景具有因果干擾。

### Negative control：移除人體後還剩多少可分類訊號？

比較：

1. `background_only` 相對 chance level 0.5；
2. `background_only − box_geometry`；
3. `background_only − n_frames`（新增一維 duration-only control，或從既有 box control 拆出）。

`background_only` 若仍顯著高於 duration/geometry control，表示場景像素、相機或動作位置可能
提供捷徑。若它只與 `n_frames` 或 `box_geometry` 打平，則不能把其 above-chance 分數解讀成
「背景懂 correctness」。不顯著也不等於證實無捷徑；必須同時回報 effect size 與區間寬度。

## 3. 實驗 arms

| arm | 人體 | 背景 | 完整身體 | 優先序 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `full_frame` | 有 | 有 | cam18 可能否 | 已有 | paired baseline |
| `full_frame_letterbox` | 有 | 有 | 是 | **P0，必跑** | 修正 center-crop truncation |
| `background_only` | 移除 | 有 | — | **P1，投稿建議** | pixel-level shortcut control |
| `person_crop` | 有 | 大幅移除 | 是 | **P2，完整拆解** | body-only framing |
| `n_frames` | 無像素 | 無 | — | **P0，低成本** | duration shortcut floor |
| `box_geometry` | 無像素 | 無 | — | 已有 | duration + framing geometry floor |

若算力只夠一個新 VideoMAE arm，跑 `full_frame_letterbox`。若要讓研究主張可防禦，至少跑
`full_frame_letterbox + background_only`。三個新 arm 都完成後，才可完整描述 body/background
framing 的相對排序。

## 4. 人物框與影像變換定義

### 4.1 固定使用整支來源影片的一個框

REHAB24-6 的一支來源影片包含多個 repetitions。**不得每個 repetition 各算一個框**，因為
框的位置、大小與動作範圍可能直接編碼該 rep 的 correctness；`person_crop` 或
`background_only` 就會在 pixel transform 階段先看到標籤相關幾何。

每個 `(video_path, camera)` 應從資料集自帶的 mocap-derived `skeleton_2d_path`，對整支影片所有
有限的 `(x, y)` joint coordinates 取 union box，再以既有 `DEFAULT_MARGIN = 0.15` 外擴並限制在
畫面內。該固定框套用到同一來源影片的所有 repetitions 和所有 sampled frames。

這個 mocap box 是研究控制工具，不是 production dependency。結果只能回答「控制相同人體區域
後的表徵差異」，不能宣稱部署時已具備同品質的人物偵測框。

### 4.2 各 arm 的 pixel transform

- `full_frame`：不變換。
- `full_frame_letterbox`：整張 frame 以 `LETTERBOX_FILL = 114` 補成正方形。
- `person_crop`：裁到固定 expanded box，再以相同灰值 letterbox 成正方形。
- `background_only`：以固定 expanded box 遮掉人體；沿用既有 horizontal interpolation fill，
  不使用 temporal median，因為固定機位下 median 仍可能留下可辨識人體殘影。

所有變換在 `read_clip_frames` 後、`encode_clip` 前於記憶體內進行。不要先輸出新 mp4 再重讀，
否則只有變體臂承受額外 lossy encoding，任何掉分都會與 codec generation confounded。

### 4.3 嚴格保持不變的條件

- model：`MCG-NJU/videomae-base-finetuned-kinetics`；
- frames：相同 `clip_starts`、clip length 16、frame stride 2、每 rep 4 clips；
- token pooling：`mean_pool_fc_norm`；
- clip aggregation：`mean`；
- classifier、normalization、early stopping、threshold objective 與 Stage A 相同；
- fold：相同 LOSO test subject 與 deterministic validation-subject selection；
- labels：同一份 `processed/labels/correctness.json`；
- 不依結果改 margin、fill、pooling、seed 或排除規則。

## 5. 實作計畫

### 5.1 最小程式修改

1. 擴充 `src/rehab24/videomae_features.py`：新增 `--variant`，允許
   `full_frame`、`full_frame_letterbox`、`person_crop`、`background_only`。
2. 新增 REHAB24-specific box resolver：由 full-video mocap 2D skeleton 建立固定框並 cache；
   pixel transform 重用 `src/video/squat_video_variants.py` 的 `apply_variant`。
3. provenance 加入 `variant`、box source、margin、fill strategy；不同 variant 不得寫入同一 raw dir。
4. 新增 framing report runner，接受明確的 `arm=feature_dir` mapping，重用
   `src/rehab24/videomae_stage_a.py` 的 LOSO、paired delta 與 camera/exercise stratification。
5. 新增單元測試，至少釘住：
   - full-video box 不因 repetition 範圍改變；
   - cam18 portrait 座標與 frame size 一致；
   - letterbox 不扭曲、不裁切；
   - person/background arms 使用完全相同的固定框；
   - variant 不改 sample ids、clip starts 或 frame count；
   - box 缺失時 fail closed，不可沉默退回 full frame。

### 5.2 產物命名

建議 raw features：

```text
data/REHAB24-6/processed/videomae_raw_full_frame_letterbox/
data/REHAB24-6/processed/videomae_raw_person_crop/
data/REHAB24-6/processed/videomae_raw_background_only/
```

materialized features：

```text
data/REHAB24-6/processed/videomae_framing/<variant>/videomae_mean_pool_fc_norm_mean/
```

報告：

```text
data/REHAB24-6/processed/videomae_framing_seed42.json
data/REHAB24-6/processed/videomae_framing_seed7.json
data/REHAB24-6/processed/videomae_framing_seed1234.json
data/REHAB24-6/processed/videomae_framing_summary.json
```

### 5.3 預計執行順序

1. 實作 fixed-video box resolver、variant CLI 與測試。
2. 先對 cam17/cam18 各一支影片做 smoke extraction。
3. 產生 framing geometry report；在看到 accuracy 前確認各 arm 的身體保留率與截斷率。
4. 抽 `full_frame_letterbox` 全集並 audit；先回答 primary question。
5. 若用途包含投稿或方法論主張，抽 `background_only`；同時完成 `n_frames` control。
6. 若需要完整 body/background 分解，再抽 `person_crop`。
7. 所有通過 audit 的 arms 以 seeds 42、7、1234 跑相同 LOSO。
8. 先寫 paired primary result，再看 camera/exercise strata；不得以事後最佳 subgroup 取代 primary。

## 6. Extraction 與 framing gates

任何 classifier training 前必須全部通過：

### 6.1 Geometry gate

- `full_frame_letterbox` 的 processor-view truncation rate 必須為 0%；
- `person_crop` 的 expanded person box survival 必須為 100%；
- `background_only` 遮罩必須完整涵蓋 expanded box；
- 分 camera 回報 body area、box survival、top loss、bottom loss；
- cam17/cam18 的 skeleton frame、video frame size 與方向必須一致。

### 6.2 Feature gate

- 每 arm 恰有 2144 個 sample bundles，sample-id set 完全一致；
- split、label、person、camera、exercise metadata 完全一致；
- 每個 sample 的 `clip_starts` 與 `full_frame` 相同；
- feature dim/dtype 一致、全為 finite、provenance 單一且正確；
- 對非正方形 REHAB24-6 畫面，`full_frame_letterbox` 不得與 `full_frame` 位元相同；
- `person_crop` / `background_only` 不得因 box lookup 失敗而產生 full-frame features。

若 gate 失敗，停止；不得先看 accuracy 再修 arm。

## 7. 評估與統計計畫

### 7.1 Primary endpoint

每折 validation subject 選 threshold 後的 test **balanced accuracy**。主報告為排除 P10 的
9-fold mean ± sample std；P10-inclusive 10-fold 僅作 sensitivity analysis。

同時回報 macro-F1、recall、specificity，但它們不取代 primary endpoint。accuracy 受類別比例
影響，只列為補充。

### 7.2 獨立單位與 seed 處理

統計獨立單位是 held-out subject，不是 sample，也不是 seed：

1. 每個 arm 跑 seeds `42, 7, 1234`；
2. 對每個 test subject，先在三個 seed 對 balanced accuracy 取平均；
3. 再計算 arm 間的 9 個 paired subject deltas；
4. 報 mean、std、9 折範圍、正向折數與 exact paired Wilcoxon p-value。

不得把 `9 subjects × 3 seeds = 27` 當成 27 個獨立樣本；那會製造 pseudo-replication。
由於 n=9 檢定力低，p≥0.05 必須寫成 **undetermined**，不能寫成「沒有差異」。

### 7.3 Primary 與 secondary comparisons

1. **Primary**：`full_frame_letterbox − full_frame`。
2. Secondary：`person_crop − full_frame_letterbox`。
3. Secondary：`person_crop − full_frame`。
4. Negative control：`background_only − n_frames`、`background_only − box_geometry`。
5. Diagnostic：各 arm 相對 chance 0.5 的 above-chance signal。

Primary 只做一個預先指定的 hypothesis test。其他 comparisons 標為 secondary/exploratory；
若同時對多個 secondary p-values 下顯著結論，使用 Holm correction。所有 raw p-values 與 corrected
p-values 都保留。

### 7.4 Stratified analysis

- **Camera**：cam17、cam18；primary effect 是否集中於 cam18 是事前指定的機制檢查。
- **Exercise**：六個動作全部報告；Ex6 squat 為產品相關重點，但不是新的 primary endpoint。
- **Repetition sensitivity**：secondary analysis 可把同一 rep 的 cam17/cam18 probabilities 先平均，
  再算 repetition-level metrics，避免兩個同步視角被誤當成兩個獨立生理事件。

分層分析只用來解釋機制，不得以最好看的 camera/exercise 取代 overall LOSO 結論。

## 8. 事前判讀規則

以下 `0.02 balanced accuracy` 是 practical-effect band，不是假裝已做完 power analysis 的顯著門檻：

| 結果 | 判讀 |
| --- | --- |
| letterbox paired mean Δ ≥ +0.02，且多數 subjects 同向 | 有實務價值的正向訊號；再看 cam18 是否符合機制 |
| `abs(Δ) < 0.02`，但區間寬 | undetermined / practically small point estimate；不可宣稱 equivalence |
| letterbox Δ ≤ −0.02 | 完整身體的好處被縮小人體等代價蓋過，或現有 crop 反而較適配；需用 framing report 解釋 |
| background-only ≤ n_frames/box_geometry | 未量到額外場景像素訊號，但受 n=9 precision 限制 |
| background-only 明顯高於兩個非像素 controls | 0.657 含可泛化 shortcut；不得全數歸因於人體動作品質 |
| person-crop ≈ letterbox 且 background-only 近 control floor | 較支持訊號位於人體，但仍受 mocap-box 與單一場景外部效度限制 |
| person-crop 顯著低於 letterbox | 移除背景、改變 body scale 或框長寬比有害；本設計不能把三者完全分開 |

無論結果方向如何，都完整報告；不因 `full_frame_letterbox` 無增益而停止並隱藏後續已完成的
negative control，也不因某 seed 較好而更換 primary seed。

## 9. 可支持與不可支持的結論

完成 primary arm 後可回答：

- REHAB24-6 的 portrait-camera center crop 是否實質影響 correctness classification；
- corrected VideoMAE 的 0.657 是否對一種不裁腳的 framing 穩健。

完成 background/person controls 後可進一步回答：

- 在這個單一實驗室、兩固定相機的資料集內，移除人體後還剩多少可泛化訊號；
- body-only framing 是否保留 full-frame 的表現。

仍不可宣稱：

- 模型在新場景、新醫院或消費者手機影片上不使用背景捷徑；
- person crop 的效果可直接部署，因為本實驗人物框來自 mocap 2D ground truth；
- background-only 不顯著即證明完全沒有捷徑；
- 單一 exercise 或 camera 的正結果可泛化到全部動作與視角。

## 10. 與既有文件的關係

- `notes/videomae_stage_a_results.md`：REHAB24-6 corrected-pooling baseline 與 cam18 caveat。
- `notes/rehab24_box_geometry_control.md`：REHAB24-6 非像素 framing/duration control。
- `notes/videomae_b1_repeated_splits_results.md`：Fitness-AQA 四臂 repeated-split 結果與 duration 更正。
- `notes/videomae_person_crop_validation_plan.md`：Fitness-AQA framing 2×2 的原始設計與限制。

本 note 若開始實作，先提交本版本，使 hypotheses、primary comparison、排除規則與判讀口徑在
任何新 accuracy 產生前固定。結果完成後另建 results note；不要回頭把觀察到的結果改寫成
「原先就預期」。
