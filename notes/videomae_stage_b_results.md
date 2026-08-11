# Stage B 結果:Fitness-AQA corrected-pooling squat fault detection

對應計畫:[`videomae_dataset_validation_plan.md`](videomae_dataset_validation_plan.md) Stage B。
前一階段:[`videomae_stage_a_results.md`](videomae_stage_a_results.md)(通過,建議進入本階段)。

---

## 0. 事前登錄(pre-registration)

**本節在任何 Stage B 結果產生前寫定並提交。** Stage A 學到的第一課是:四臂設計會同時
給出三個 delta,等看到數字再挑一個當主結果,等於用 best-of-three 冒充單一假設檢定。
Stage B 更嚴重——計畫列了兩種 fusion(early / late)乘上三個 label mode,共六個候選
delta,再加上兩個控制組。以下在跑任何實驗前先鎖定。

### 0.1 主結果

| 項目 | 值 |
| --- | --- |
| **Primary arm** | **Calibrated late fusion**:normalized pose-only 與 corrected VideoMAE-only 各自訓練,於 validation 上校準後合併機率 |
| **分母(paired 比較的基準)** | **Normalized pose-only**,本機重抽特徵重跑,歷史值 `0.635 ± 0.010` |
| **Primary label mode** | `combined`(與 0.635 這個數字對應的同一個 label mode)|
| **Primary metric** | test split、selected-threshold balanced accuracy,5 個 seed(1–5)的平均 |
| Seeds | `1, 2, 3, 4, 5`——沿用歷史 pose-only 與 VideoMAE-only 的同一組 seed |
| 超參數 | `run_videomae_experiment_grid.py` 全部預設值(epochs 20 / batch 32 / lr 3e-4 / hidden 128 / dropout 0.4 / weight decay 0.01 / patience 5 / threshold objective `balanced_accuracy`),與產生 0.635 的那次執行逐項相同 |

**為什麼 primary 選 late fusion 而不是 early concat**(此判斷與 Stage B 任何結果無關,
純粹來自已完成的 Stage A 與維度事實):

1. Stage A §2.5 已經量到 early concat 在**三個** skeleton backbone 上都只收斂到兩個
   branch 中較強的那一個,沒有一次超越它。那是「這個 fusion 方法測不到互補性」的直接
   證據,不是猜測。
2. 計畫本身把 calibrated late fusion / gating 排進 Stage B,理由寫得很明白:降低弱 RGB
   branch 稀釋 pose signal 的風險。把它當 secondary,等於預先放棄計畫想測的東西。
3. 維度不對等會重演:VideoMAE 是 768 維,pose-only 特徵維度由 `pose_feature_extraction`
   決定,兩者 concat 後正規化能處理尺度、處理不了容量。

Early fusion 仍然會跑,但列為 **secondary/diagnostic**,其 delta 不作為保留決策的依據。

**Late fusion 的三個自由度也在此鎖定**(否則等於關掉六個 delta 又偷偷打開三個):

| 自由度 | 選定 | 理由(與結果無關)|
| --- | --- | --- |
| 校準方法 | **Platt scaling**(在 validation 上以 logistic regression 校準每個 branch 的機率)| val 只有 243 筆且正例率 ~0.68,isotonic 是無參數階梯函數,在這個樣本數上會過擬合 |
| 合併規則 | **兩個校準後機率的等權平均**(zero-parameter)| val 已同時承擔 checkpoint 選擇、threshold 選擇與校準;再加一個由 val 調出來的權重是第四次使用同一批 243 筆。val-tuned 權重列為 diagnostic |
| Seed 配對 | pose-seed-s 對 videomae-seed-s(s = 1…5)| 固定配對讓 fusion 臂剛好 5 次執行,不會變成 25 種組合裡挑一個 |

融合後機率的 threshold 一樣**只在 val 上**以 `find_best_threshold`、同一個
`balanced_accuracy` 目標選出。

**val 預算要照實說:** 243 筆 validation 同時負擔 checkpoint 選擇、threshold 選擇與
校準擬合。這不致命,但它是這個設計最擁擠的地方,結果一律連同這句話一起報。

### 0.2 完整實驗矩陣與各臂角色

| 臂 | 角色 | 用來回答 |
| --- | --- | --- |
| `legacy_first_token_max` VideoMAE-only | 重現歷史抽取 | 分母是否可信(對照歷史 `0.555`,範圍 `0.532–0.584`)|
| `mean_pool_fc_norm_mean` VideoMAE-only | corrected standalone | task-specific 是否有獨立訊號 |
| Normalized pose-only | **主要可部署 baseline** | Stage B 的分母 |
| **Calibrated late fusion** | **primary** | 保留門檻的唯一判準 |
| Regularized early fusion | secondary | 與 Stage A 的 early concat 結論是否一致 |
| Person-crop 控制 | 捷徑檢查 | 訊號是否在身體上 |
| Background-only 控制 | 捷徑檢查 | 訊號是否只是場景/視角 |

Corrected VideoMAE 的 token pooling 與 clip aggregation 組合**沿用 Stage A 的 primary**
(`mean_pool_fc_norm` + `mean`),不在 Stage B 重新挑選;四種組合的特徵都會物化出來,
但只有 primary 進入決策,其餘僅供診斷。

### 0.3 分母門檻(必須先過,才准看 fusion 的數字)

`data/Squat/Labeled_Dataset/pose_features` 已不存在於本機,`pose_json/` 只剩 5 個檔案
(且那 5 個是 view-estimation regression corpus 的一部分,已先移到暫存區,由本次重抽
還原)。**0.635 因此不是封存數字,而是必須重新導出的分母。**

- 重抽方式與歷史完全相同:`src/pose/process_videos.py`(MediaPipe Pose,
  `model_complexity=2`)→ `pose_feature_extraction` → grid runner 加 `--normalize-features`。
- **判準:重抽後的 normalized pose-only combined balanced accuracy 必須落在
  `0.635 ± 0.010` 內。** 落在區間內,分母成立,所有 delta 以 0.635 這條線陳述。
- 附帶:歷史表其實同時公布了三個 label mode 的 normalized pose-only 數字
  (combined `0.635 ± 0.010`、knees_forward `0.615 ± 0.030`、knees_inward
  `0.608 ± 0.054`),grid runner 預設就會跑完三個,所以重現檢核有**三個**獨立的靶,
  不是一個。只有 combined 是門檻;另外兩個作為佐證一併報告。
- 若落在區間外:**不得默默吸收進 delta**。改以重抽值為分母,並同時列出兩個數字與差距。
- 附帶檢核:`tests/test_view_regression_corpus.py` 會用重抽出的那 5 個 test pose JSON
  對照凍結的 view baseline。verdict 若不動,就是重抽 recipe 與歷史一致的獨立證據。

**已先做掉的環境檢查(2026-08-10):MediaPipe 沒有漂移。** `.venv` 在產生 0.635 之後
重建過,若 mediapipe 版本動過,landmark 會動、pose 特徵會動,分母門檻就會因為環境
原因而失敗,事後極難與真正的差異區分。把留存的原始 `33048_1.json` 重抽一次逐值比對:
125 個影格、`landmarks` 與 `world_landmarks` 兩組,**max |delta| = 0.0(完全相同)**,
偵測到/未偵測的影格也完全一致(mediapipe 0.10.14)。view verdict 是字串,能在座標
真的漂移的情況下仍然相等,所以這個數值比對比那個測試更強。

**但有一處已知不同,必須列進門檻的已知變因:** 產生 0.635 的
`videomae_video_classifier.py` 不是現在這一份。Stage A 為了讓 70 個 fold 跑得完,改了
`load_video_feature` 的記憶體快取、`build_samples` 的 stem→path 索引,以及重複 stem
的取用順序(目錄走訪順序 → 排序後第一個)。當時已用同一組合成特徵驗證改動前後**每個
fold bit-identical**,所以預期無影響;但若重抽值落在 `0.635 ± 0.010` 之外,這是第二個
要查的地方,而不是一個意外。

### 0.4 保留門檻(直接引用計畫,不加碼也不放寬)

將 VideoMAE 保留為 production 預設 input 的條件:

1. Fusion 相對 normalized pose-only 的平均 `Δ balanced accuracy ≥ +0.02`。
2. Paired 95% CI 下限大於 0。
3. Recall 或 specificity 任一不得惡化超過 `0.03`。
4. 改善在多數 seeds 中方向一致。
5. Person-crop / background 控制後仍保留效果。

未達門檻但 VideoMAE-only 優於 null ⇒ 保留為研究或選配 branch,不作 production 預設。

### 0.5 統計方法與其已知限制

**Paired 95% CI 的重抽單位是「影片」。** 計畫寫的是 participant-level bootstrap,但計畫
自己的資料表也寫了 Fitness-AQA「沒有可靠 participant mapping」。已驗證的資料事實:
1623 個 labeled id 對應 **1623 個相異前綴**(`32903_8` 的前綴是 `32903`),也就是每個
樣本各自來自不同來源影片,沒有 clip-within-video 的群集結構可用;train/val/test 的前綴
交集皆為 0。因此:

- 重抽單位 = test split 的 244 支影片,paired(兩臂使用同一組重抽索引),2000 次。
- **誠實的限制:同一位健身者可能出現在多支影片中而無從得知,這個 CI 因此可能偏樂觀。**
  沒有任何方法能在這個資料集上修正它,只能陳述。

**Recall / specificity 的 ±0.03 護欄以「5 個 seed 的平均」判定,並同時報告跨 seed 全距。**
歷史 normalized pose-only 的 recall 是 `0.717 ± 0.101`、specificity `0.553 ± 0.111`——
單一 seed 上的 0.03 差距完全在雜訊內,拿單 seed 判護欄等於擲骰子。

**Threshold 與 checkpoint 都只由 validation 決定。** 已核對
`src/video/videomae_video_classifier.py`:`find_best_threshold` 吃的是 val 機率,
checkpoint 也以 val 目標挑選,test 只在最後評估。歷史的 0.635 與 0.555 同屬這個慣例,
所以兩邊可比。

### 0.6 三個必須連同結果一起讀的前提

1. **Legacy 是重新抽取,不是封存數字。** 舊的 Fitness-AQA VideoMAE 特徵不在本機,
   transformers 版本也已不同。§0.2 的 legacy 臂就是為了確認重現落在 `0.532–0.584` 內。
2. **兩個 fusion 的輸入特徵完全相同**,差別只在合併方式;pose branch 一律使用
   train-fold 統計的 normalization,不與未 normalization 的舊 baseline `0.581` 比較。
3. **控制組與主臂讀的是同一個檔案。** 原本寫的是「控制組的變體影片是新編碼的,
   frame count 補齊到來源 header 以對齊 clip start」;§1.3(d) 之後改為由抽取器在記憶體
   中套用 box,完全不產生中間影片,所以三個臂的 clip start 是**同一次解碼**的結果,
   而不是兩條路徑碰巧對齊。這一條因此從「已控制的風險」變成「結構上不存在」。

### 0.7 停止條件

若 corrected VideoMAE-only 在 Fitness-AQA 上仍接近隨機(≤ 0.55),則不再往 fusion 與
EgoExo-Fitness 推進,並依計畫的結果解讀表記為「REHAB24-6 成功、Fitness-AQA 失敗:
VideoMAE 含一般動作品質訊號,但不適合目前 squat fault endpoint」。

---

## 1. 方法與實作

### 1.1 修正的是同一個 pooling,這次在第二個抽取器上

計畫第 1 步要求修正**兩條** VideoMAE extractor。Stage A 修了 REHAB24-6 那條;
`src/video/videomae_feature_extraction.py` 直到本階段仍原封不動地帶著同一個缺陷:
`last_hidden_state[:, 0, :]`(VideoMAE 沒有 CLS token,這是第一個 tubelet 的左上角
patch)加上對 clip 取 `max`。它的預設路徑還指向 `data/Squat/...`,而資料集早已搬到
`data/Fitness-AQA/Squat/`。

修正後與 Stage A 對齊:兩種 token pooling 由**同一次 forward** 算出、存下 per-clip
stack、每個 `.npz` 蓋 provenance、輸出到新目錄。

### 1.2 新增的共用元件

| 模組 | 用途 |
| --- | --- |
| `src/video/videomae_backbone.py` | fc_norm 載入與 default-init 防呆、單次 forward 的雙 pooling、`resolve_device` |
| `src/video/videomae_materialize.py` | 與資料集無關的物化步驟(REHAB24-6 與 Fitness-AQA 共用)|
| `src/video/videomae_audit.py` | 與資料集無關的稽核;Fitness-AQA 由 split-key JSON 驅動 |
| `src/video/squat_dataset.py` | 資料集路徑與 split/label 讀取,純 stdlib |
| `src/video/squat_video_variants.py` | person-crop 與 background-only 變體影片 |

`resolve_device` 把 Stage A 用一次失敗執行換來的教訓寫進程式:Kaggle 會配給 sm_60 的
P100,而該映像的 torch 只內建 sm_70 以上 kernel,錯誤發生在**模型載入之後**的第一個
conv3d。改為載入前先用真正的 conv3d 探測,失敗即退回 CPU。

### 1.3 控制組變體:兩個第一版設計都被實測推翻

兩個變體的第一版都寫完、跑過、看過真實輸出之後才發現不能用。兩次都不是測試抓到的
(測試都綠),是**看圖**與**量分布**抓到的。

**(a) background-only 的「每像素時間中位數」補洞是錯的。** 深蹲時受試者在**每一個**
影格都佔住框的中央,所以那些像素的時間中位數**就是受試者**——輸出是一個認得出來的
糊人影(在 `33048_1` 上實際確認)。這個版本量到的會是「VideoMAE 能不能讀糊掉的人」,
而一旦控制組**存活**,還會被誤讀成「果然是場景捷徑」。

當初的測試通過,是因為它只斷言「框內沒有影格間變化」——中位數當然沒有變化。已改成
斷言**填進去的值等於場景值**,這才是能區分兩者的斷言。

改用**框外左右兩欄的水平內插**:輸出在框內是框外兩欄的 rank-1 函數,身體結構在數學上
無法存活。固定機位拍靜止的人,人背後的真實背景**從未被觀測過**,任何補洞都不可能還原
它;誠實的選擇是一個可證明不帶身體結構的填法。殘留的洩漏是那個矩形的位置與大小,
照實寫進結果,不用工程手段掩蓋。

**(b) person-crop 擴成正方形等於沒有裁。** 站姿的人框是高瘦的,擴成正方形要用長邊,
於是保留下來的面積在 300 支 train 影片上中位數 **77%**(p90 = 100%,16% 的影片幾乎
整張都留著)——控制組幾乎沒有移除背景。landmark 框本身只佔 **25%**。

改成**貼身裁切 + 灰色 letterbox 補成正方形**:保留面積中位數降到 **31%**,而
`VideoMAEImageProcessor` 的短邊 224 + center crop 224 因為畫面已是正方形而成為 no-op,
不會像 Stage A 的 cam18 那樣裁掉腳。

**(c) 每支影片只用一個 box**(跨影格取聯集),避免逐影格框帶入原本不存在的鏡頭運動。

**(d) 再編碼的混淆:不是「計價」,是直接消除。** 第一版把兩個變體寫成實體影片,於是
控制組比 `full_frame` 多付一次 `cv2.VideoWriter` 的有損世代。原本的對策是再建一個
`reencoded` 恆等臂去**量**那個世代的代價。

後來改成更好的做法:**變體是(來源影片 × 一個 box)的確定性函數,所以只讓 box 旅行。**
抽取器新增 `--variant-manifest`,在記憶體中對解碼後的影格套用 box 再進模型,完全不產生
中間影片。四個臂於是解碼**同一個檔案**、走**同一條 decode 路徑**,再編碼的不對稱不復存在,
`reencoded` 對照臂也就不需要了。

**這個差異是實測過的,不是理論顧慮**:同樣 2 支影片、同樣的 person_crop,
「預先編碼的變體影片」與「記憶體內套用」兩條路徑的特徵 cosine 相似度為
**0.9965–0.9975**(`mean_pool_fc_norm`)。方向一致、量不大,但真實存在——而它正好落在
「控制組下降時,到底是人被移除還是編碼損失」這個會左右結論的問題上。

附帶效果:Kaggle 需要上傳的資料從 **4.8 GB 降到 0.97 GB**(只有原始 `videos.zip`,
加上兩個各數十 KB 的 box manifest)。實體變體影片仍留在本機,作為可目視檢查的產物。

### 1.4 完整重跑步驟

所有指令自 repo 根目錄執行,直譯器一律 `.venv\Scripts\python.exe`。

```bash
# 1. pose 重抽(分母的來源;6 個平行 worker,約 4–5 小時)
python scripts/pose/run_pose_extraction.py --dataset labeled --no-video --jobs 6

# 2. pose 特徵
python -m src.pose.pose_feature_extraction \
  --pose-json-dir data/Fitness-AQA/Squat/Labeled_Dataset/pose_json \
  --split-dir     data/Fitness-AQA/Squat/Labeled_Dataset/Splits \
  --output-dir    data/Fitness-AQA/Squat/Labeled_Dataset/pose_features

# 3. 控制組的 box(需要步驟 1 的 pose)。變體影片本身只是可目視檢查的產物,
#    真正被消費的是 manifest.json 裡的 box——抽取器在記憶體中套用。
python scripts/video/build_video_variants.py --variant person_crop     --jobs 4
python scripts/video/build_video_variants.py --variant background_only --jobs 4

# 4. VideoMAE 抽取(Kaggle,每個臂一個 kernel;本機 CPU 實測 5.3 s/clip ⇒ 每個臂 7–10 h)
#    只需上傳原始 videos.zip(0.97 GB);box 隨 src 資料集的 meta.zip 一起走。
#    上傳必須從資料集目錄「裡面」執行:kaggle CLI 會把 -p 的相對路徑
#    直接拼進暫存檔名,路徑含斜線就會 [Errno 2]。
#    (PowerShell)  Push-Location .kaggle_tmp/fitaqa_videomae_input
#                  uv run --with kaggle kaggle datasets create -p .
#                  Pop-Location
#    kernels:      uv run --with kaggle kaggle kernels push -p .kaggle_tmp/fitaqa_videomae_extract
#                  uv run --with kaggle kaggle kernels push -p .kaggle_tmp/fitaqa_videomae_extract_crop
#                  uv run --with kaggle kaggle kernels push -p .kaggle_tmp/fitaqa_videomae_extract_bg
#    本機等價指令(不經 Kaggle):
#      python scripts/video/run_videomae_feature_extraction.py --variant person_crop \
#        --variant-manifest data/Fitness-AQA/Squat/Labeled_Dataset/videos_person_crop/manifest.json \
#        --output-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_raw_person_crop

# 5. 物化四臂 + 稽核(稽核不過就不准往下跑)
python scripts/video/materialize_videomae_features.py --raw-dir <解壓後的 videomae_raw>
python scripts/video/audit_videomae_features.py \
  data/Fitness-AQA/Squat/Labeled_Dataset/videomae_mean_pool_fc_norm_mean \
  data/Fitness-AQA/Squat/Labeled_Dataset/videomae_legacy_first_token_max

# 6. 單分支各臂(5 個 seed;pose 臂必須加 --normalize-features)
python scripts/video/run_videomae_experiment_grid.py \
  --feature-dir data/Fitness-AQA/Squat/Labeled_Dataset/pose_features \
  --train-keys  data/Fitness-AQA/Squat/Labeled_Dataset/Splits/train_keys.json \
  --val-keys    data/Fitness-AQA/Squat/Labeled_Dataset/Splits/val_keys.json \
  --test-keys   data/Fitness-AQA/Squat/Labeled_Dataset/Splits/test_keys.json \
  --forward-labels data/Fitness-AQA/Squat/Labeled_Dataset/Labels/error_knees_forward.json \
  --inward-labels  data/Fitness-AQA/Squat/Labeled_Dataset/Labels/error_knees_inward.json \
  --output-root data/Fitness-AQA/Squat/experiments/pose_only \
  --label-modes combined --normalize-features
# (VideoMAE corrected / legacy / 兩個控制組各一次,--feature-dir 換掉、--output-root 換掉)

# 7. early fusion 的串接特徵(secondary)
python scripts/video/fuse_feature_dirs.py \
  --first-feature-dir  data/Fitness-AQA/Squat/Labeled_Dataset/pose_features \
  --second-feature-dir data/Fitness-AQA/Squat/Labeled_Dataset/videomae_mean_pool_fc_norm_mean \
  --output-dir         data/Fitness-AQA/Squat/Labeled_Dataset/fused_pose_videomae

# 8. 證據表與四個保留條件(late fusion 在此離線算出,不需再訓練)
python scripts/video/run_stage_b_report.py \
  --pose-predictions     data/Fitness-AQA/Squat/experiments/pose_only/predictions \
  --videomae-predictions data/Fitness-AQA/Squat/experiments/videomae_corrected/predictions \
  --arm early_fusion=data/Fitness-AQA/Squat/experiments/early_fusion/predictions \
  --arm videomae_legacy=data/Fitness-AQA/Squat/experiments/videomae_legacy/predictions \
  --output data/Fitness-AQA/Squat/experiments/stage_b_report.json
```

### 1.5 執行紀錄

| 項目 | 狀態 |
| --- | --- |
| pose 重抽 | **完成**:1623/1623,0 失敗,全數通過完整性檢查 |
| pose 特徵 | **完成**:1623/1623,654 維 |
| 分母門檻 | **完成,未通過**(見 §2.1);分母改為 0.650 |
| 控制組 box | **完成**:person_crop 與 background_only 各 1623,0 個無 pose 的退回、0 個影格數不符 |
| Kaggle 抽取 × 3 | *(待執行,等 videos.zip 上傳)* |
| 各臂訓練與證據表 | *(待執行)* |

**Kaggle 執行的兩個小坑(都被 kernel 自帶的檢查擋下,沒有污染結果):**

1. **staleness guard 用錯 token。** kernel 在跑之前會確認掛載到的 extractor 是 Stage B
   那一版。Stage A 的檢查抓字串 `mean_pool_fc_norm`,但 Stage B 的抽取器引用的是**常數**
   `MEAN_POOL_FC_NORM`,從未把值寫成小寫字面量——於是正確的樹被判成舊版而中止。
   改抓 `MEAN_POOL_FC_NORM` 與 `--variant-manifest`。
2. **`src.zip` 是舊的。** 記憶體內套用變體那次改動之後,只重建了 `meta.zip`,忘了重建
   `src.zip`,所以 Kaggle 上掛的抽取器沒有 `--variant-manifest`。**這正是 guard 存在的
   理由**:它在第二次執行就把這件事指名道姓地講出來,而不是讓 kernel 用舊程式跑完 5 小時
   再產出一批無法辨識的特徵。

已完成的前置檢核:

- 抽取器 CPU smoke test 通過(3 支影片),`fc_norm` weight mean 0.6832 / bias mean 0.0083,
  與 Stage A 在本機與 Kaggle 上量到的值**完全一致**。
- 變體影片與來源影片的影格數/fps 逐支相同(125 / 30.00),所以四個臂的 clip start
  完全一致。建構過程中有 6 支被中斷寫壞(0 影格),由新增的驗證步驟抓出並刪除重建——
  這也是把寫檔改成原子性的原因。
- 物化四臂 + 稽核鏈路通過:稽核在只有 3/1623 覆蓋率時正確地以 FAIL 結束。
- MediaPipe 與封存的 pose JSON 逐值相同(見 §0.3)。

---

## 2. 結果

### 2.1 分母門檻:**未通過**,分母改為重抽值 0.650

`pose_features/` 全部重建(1623/1623),依 §0.3 的判準對照歷史三個數字:

| label mode | 重抽(5 seeds) | 歷史公布值 | delta | 判定 |
| --- | --- | --- | --- | --- |
| **combined**(門檻) | **0.6504 ± 0.0118** | 0.635 ± 0.010 | **+0.0154** | **超出 ±0.010** |
| knees_forward | 0.6337 ± 0.0185 | 0.615 ± 0.030 | +0.0187 | 落在該列自身的離散度內 |
| knees_inward | 0.6301 ± 0.0192 | 0.608 ± 0.054 | +0.0221 | 落在該列自身的離散度內 |

combined 的 per-seed:0.639 / 0.650 / 0.644 / 0.649 / 0.670。

**依 §0.3 的預先規定:分母改用重抽值 `0.650`,兩個數字與差距一併列出,不默默吸收
進任何 delta。** 這讓保留門檻實際上變成「fusion 必須 ≥ 0.670」,比原本的 0.655 更難——
這是誠實分母該付的代價。

**差在哪裡:已經逐項排除到只剩訓練期的隨機性。**

| 候選原因 | 檢核結果 |
| --- | --- |
| MediaPipe 漂移 | **排除**:封存的 5 個 pose JSON 與重抽逐值相同(max \|delta\| = 0.0)|
| 特徵建構程式改變 | **排除**:`src/pose/pose_feature_extraction.py` 自基線之前未再修改 |
| 超參數改變 | **排除**:grid runner 的 epochs / batch / lr / hidden / dropout / weight decay / patience / threshold objective / seeds 全部與基線同一版 |
| Stage A 的 loader 改動 | **排除**:改的是 `load_video_feature` 的快取(回傳副本)與 `build_samples` 的索引(重複 stem 改取排序後第一個);本資料集 1623 個 id 無重複,`compute_feature_normalization` 未被更動 |
| 訓練環境 | **唯一剩下的**:歷史那次在 Colab GPU、較舊的 torch;本次在本機 CPU、torch 2.13。同一個 seed 在不同裝置/版本上不會給出同一串亂數,初始化與 shuffle 都會不同 |

**關鍵對照:未正規化的那一臂重現得幾乎完全一致。** 同一批特徵、同一個環境、
只是不加 `--normalize-features`:

| | 重抽 | 歷史 | delta |
| --- | --- | --- | --- |
| pose-only **未** normalization | 0.5854 ± 0.0215 | 0.581 ± 0.032 | **+0.0044** |

所以環境並沒有系統性地把所有結果往上推;只有 normalized 那一臂高了 +0.015。
把它放進統計脈絡:公布值的 ±0.010 是 5 次執行的標準差,平均數的標準誤約 0.0045,
本次約 0.0053,兩者差距 +0.0154 對應 t ≈ 2.2(p ≈ 0.06)。也就是說,**這是一個
邊緣、未達顯著的差距,而不是一個確立的偏移**——但既然預先規定寫的是 ±0.010,
就照規定判為未通過並換分母,不事後放寬。

> 副產品:這一節同時證明了本階段的 pose 分支確實是歷史那條 pipeline 的重建,而不是
> 一個名字相同的新東西——三個 label mode 的排序、未正規化臂的重現、以及 normalization
> 帶來的 +0.065 提升(0.585 → 0.650,歷史為 +0.054)都對得上。

### 2.2 Legacy 臂重現,所以 pooling 修正的 delta 站得住

| 臂 | 本次(5 seeds) | 歷史公布值 | 判定 |
| --- | --- | --- | --- |
| `legacy_first_token_max` VideoMAE-only | **0.5720 ± 0.0192** | 0.555(範圍 0.532–0.584)| **落在公布範圍內,重現成立** |

per-seed:0.558 / 0.551 / 0.578 / 0.600 / 0.573。

這一條是整個 pooling 修正的分母。它重現了,所以下一節的 +0.068 不是「基線換掉了」。

### 2.3 Pooling 修正在**第二個資料集、第二個任務**上獨立複製

| 臂 | balanced accuracy |
| --- | --- |
| `legacy_first_token_max`(舊抽取方式) | 0.5720 ± 0.0192 |
| `mean_pool_fc_norm_mean`(修正後,pre-registered primary)| **0.6401 ± 0.0211** |
| **delta** | **+0.068** |

Stage A 在 REHAB24-6 上量到的是 +0.08~0.12。這裡是不同資料集(in-the-wild YouTube
vs 實驗室)、不同任務(squat fault detection vs binary correctness)、不同評估協定
(固定 split vs subject-wise LOSO),方向一致、量級相同。**Stage A 的核心主張因此有了
一次獨立複製,而不只是同一批資料的再分析。**

### 2.4 主結果:fusion **未達**保留門檻

來源:`data/Fitness-AQA/Squat/experiments/stage_b_report_preliminary.json`,
label mode `combined`,seeds 1–5,threshold 與 checkpoint 皆由 val 決定。

| 臂 | balanced acc | recall | specificity | macro F1 |
| --- | --- | --- | --- | --- |
| **Normalized pose-only(分母)** | **0.650 ± 0.012** | 0.631 ± 0.059 | 0.669 ± 0.065 | 0.618 |
| **Calibrated late fusion(primary)** | **0.648 ± 0.008** | 0.631 ± 0.057 | 0.664 ± 0.052 | 0.617 |
| Corrected VideoMAE-only | 0.640 ± 0.021 | 0.630 ± 0.027 | 0.650 ± 0.018 | 0.611 |
| Regularized early fusion(secondary)| 0.641 ± 0.019 | 0.565 ± 0.050 | 0.717 ± 0.021 | 0.595 |
| Legacy VideoMAE-only | 0.572 ± 0.019 | 0.602 ± 0.038 | 0.542 ± 0.071 | 0.552 |
| Person-crop 控制 | 0.666 ± 0.012 | 0.655 | 0.678 | 0.636 |
| Background-only 控制 | 0.627 ± 0.019 | 0.671 | 0.583 | 0.602 |

> 兩列控制組為**用完整 box 重抽後**的值。第一版因 manifest 缺陷而作廢,見 §2.5;
> 它們的解讀見 §2.6。

**Primary 判定(對照 §0.4 的五個條件):**

| # | 條件 | 值 | 判定 |
| --- | --- | --- | --- |
| 1 | Δ balanced accuracy ≥ +0.02 | **−0.0028** | **不通過** |
| 2 | Paired 95% CI 下限 > 0 | **[−0.059, +0.056]** | **不通過** |
| 3 | recall/specificity 惡化 ≤ 0.03 | −0.006 | 通過 |
| 4 | 多數 seed 方向一致 | 3/5 | 通過 |
| 5 | 控制組後仍保留效果 | 見 §2.6 | **不通過(見下)** |

**條件 1 與 2 不通過,因此依計畫:VideoMAE 不作為 production 預設 input。**

值得注意的是 late fusion 的行為:0.648 對 pose-only 的 0.650、VideoMAE-only 的 0.640——
**它收斂到兩個 branch 中較強的那一個,沒有超越它**。這正是 Stage A 在 REHAB24-6 上對
early concat 量到的同一個現象,現在在不同資料集、不同 fusion 方法上重演。計畫把
calibrated late fusion 排進 Stage B 的理由是「降低弱 branch 稀釋強 branch 的風險」——
它確實做到了(late fusion 0.648 遠優於 early fusion 在 recall 上的崩壞),
**但「不稀釋」不等於「有互補性」。**

Early fusion 另外踩到護欄:recall 掉 0.066(0.631 → 0.565),換來 specificity +0.048。
它把決策往「預測正常」推,而不是變得更準。

**與計畫結果解讀表的對應:**

> 「REHAB24-6 與 Fitness-AQA 都成功,但 fusion 無增益 → VideoMAE 有獨立訊號,
> 但 pose 已捕捉同一資訊 → 保留選配或診斷 branch,不作預設輸入」

corrected VideoMAE-only 是 0.640,遠高於 legacy 的 0.572、也遠高於隨機的 0.500,
所以「有獨立訊號」成立;但它與 pose 融合後拿不到任何增量,所以「pose 已捕捉同一資訊」
也成立。**這一列就是 Stage B 的結論。**


### 2.5 兩個控制組的第一次結果**作廢**:一半的樣本根本沒被變換

第一次跑出來的控制組數字是 person-crop `0.667`、background-only `0.616`。**兩個都不能用。**

原因在 manifest,不在模型。`build_video_variants.py` 的續跑路徑遇到「輸出影片已存在」
時,回傳的是一列 `{"skipped": true}` 的殘缺紀錄——**沒有 box**。而抽取器的
`load_variant_boxes` 把「沒有 box 欄位」對應成 `None`,`apply_variant` 又把 `None`
定義為「這支影片偵測不到人,原樣輸出」。兩個不同的意思被摺成同一個值,於是:

| 控制組 | 沒有 box 的列 | 實際被抽取的內容 |
| --- | --- | --- |
| person_crop | **831 / 1623(51%)** | 一半是**未經變換的原始畫面** |
| background_only | **426 / 1623(26%)** | 四分之一是**未經變換的原始畫面** |

兩個控制組都被原始畫面污染,而且污染方向正好把它們**拉向 full-frame 臂**——也就是
會讓「控制組存活」看起來更像真的。這正是控制組最不能出錯的方向。

**修正:**

1. `describe_variant()` 把「算出 box」與「編碼影片」拆開,所以**每一支影片都會記錄
   box**,不論它的影片檔是否已經存在;新增 `--boxes-only` 模式,重建 manifest 只需要
   幾秒(抽取器本來就只吃 box)。
2. `write_manifest()` 在任何一列缺 box 時**拒絕寫出**。
3. `load_variant_boxes()` 在任何一列缺 box 時**拒絕載入**,並明確區分
   「沒有記錄」(缺陷)與 `null`(真的偵測不到人,本資料集為 0 支)。

三個防線都有測試。兩個控制組已用完整的 box 重新抽取中。

**這個缺陷沒有影響主結果。** full_frame 臂根本不讀 manifest(`--variant full_frame`
不需要 box),legacy 臂與 pose 臂也一樣。§2.1–§2.4 的所有數字不受影響。


### 2.6 捷徑控制:full-frame 的訊號有 **91%** 在把人塗掉之後仍然存在

用完整 box 重抽之後的五個數字(全部 5 seeds、同一組 split 與超參數):

| 臂 | 看得到什麼 | balanced accuracy | 高於隨機 |
| --- | --- | --- | --- |
| `videomae_corrected`(full frame)| 人 + 場景 + 取景 | 0.6401 ± 0.0211 | +0.140 |
| `videomae_person_crop` | **只有人**(背景移除,灰邊補正方)| **0.6662 ± 0.0117** | +0.166 |
| `videomae_background_only` | **只有場景**(人被塗掉,留下矩形)| 0.6271 ± 0.0191 | +0.127 |
| `box_geometry`(zero-parameter)| 沒有像素:人框位置/大小/長寬比 + 片長 | 0.5777 ± 0.0230 | +0.078 |
| `frame_format`(zero-parameter)| 沒有像素、**也沒有任何關於人的資訊**:只有解析度與長寬比 | **0.4979 ± 0.0157** | **−0.002** |

**把 full-frame 高於隨機的 +0.140 拆開:**

| 只給模型 | 高於隨機 | 佔 full-frame 的 |
| --- | --- | --- |
| 純錄影格式(解析度/長寬比)| **+0.000** | **0%** |
| ＋人框的粗略幾何(位置、大小、片長),仍無像素 | +0.078 | 56% |
| ＋場景像素(人被塗掉)| +0.127 | **91%** |
| ＋人本身的像素 | +0.140 | 100% |

**兩個必須一起讀的結論:**

1. **把受試者整個塗掉之後還剩 91%。** corrected VideoMAE 在 full frame 上的 0.640
   **不能**被解讀成「模型看懂了動作品質」。
2. **但這不是「誰在哪裡拍的」那種資料集捷徑。** `frame_format` 這個 zero-parameter
   對照(只有解析度與長寬比、完全沒有關於受試者的資訊)是 **0.498——精準的隨機**。
   錄影格式本身不帶任何標籤資訊。

   **先前版本的本節曾主張「同一個健身房/上傳者/機位重複出現,所以標籤可由拍攝脈絡
   預測」。這個機制被 `frame_format` 對照否證,已刪除。** 剩下能支持「場景像素有貢獻」
   的證據只有一項:background_only 相對 box_geometry 的 **+0.049**。場景確實加了東西,
   但「加的是什麼」本階段沒有測到。

3. **真正扛住那 56% 的,是關於受試者的粗略幾何。** `box_geometry` 的 12 個數字裡,
   有意義的是人框的位置與大小(跨影格取聯集,所以框高其實編碼了**下蹲的垂直行程**)
   與片長(編碼**節奏**)。換句話說,它不是純粹的脈絡洩漏,而是一個**極粗糙的動作代理**。
   這也解釋了為什麼 background_only 有 0.627:人雖然被塗掉,那個矩形仍然把這些幾何
   留在畫面上。

**身體訊號是真的,而且是最強的一臂。** `person_crop`(只有人)0.666 高於
background_only(+0.039)、高於 full frame(+0.026)、也高於 pose-only(+0.016)。
**給模型看整個畫面比只給它看身體更差**——場景不只沒幫助,它是干擾。

**條件 5 判定:不通過。** 沒有可保留的 fusion 增益(條件 1、2 已不通過),而
VideoMAE-only 自身在 full frame 上的表現,有 91% 在受試者被移除後仍然存在。

三個必須連同這節一起讀的限制:

1. **`background_only` 不是純場景。** 補洞後留下的矩形洩漏受試者的位置與大小——
   這正是 `box_geometry` 對照要量的東西(+0.078),場景像素只再加 +0.049。
2. **`person_crop` 也不是純身體。** 灰邊 letterbox 的比例仍編碼人框的長寬比,而
   `box_geometry` 顯示這類幾何單獨就值 +0.078。因此 person-crop 相對 pose-only 的
   +0.016 **無法**在本階段的臂裡與取景幾何分離;它是「最強的一臂」,不是「已證實的增益」。
3. **兩個控制與主臂共用同一次解碼**(§1.3d),三者之間沒有編碼世代差。

---

## 3. Stage B 結論

### 3.1 對照計畫的五個保留條件

| # | 條件 | 結果 | 判定 |
| --- | --- | --- | --- |
| 1 | Δ balanced accuracy ≥ +0.02 | −0.003 | **不通過** |
| 2 | Paired 95% CI 下限 > 0 | [−0.059, +0.056] | **不通過** |
| 3 | recall/specificity 惡化 ≤ 0.03 | −0.006 | 通過 |
| 4 | 多數 seed 方向一致 | 3/5 | 通過 |
| 5 | 控制組後仍保留效果 | 91% 的訊號在人被塗掉後仍在 | **不通過** |

**結論:VideoMAE 不作為 x-coach 的 production 預設 input。**

### 3.2 這一階段確立了什麼

1. **Pooling 修正在第二個資料集上獨立複製:+0.068**(legacy 0.572 → corrected 0.640),
   而 legacy 臂本身重現了歷史公布值。Stage A 的核心主張不再只有一組資料支持。
2. **Fusion 沒有互補性。** late fusion 收斂到較強的 branch(0.648 對 pose 的 0.650),
   early fusion 更差且以 recall 換 specificity。兩種 fusion、兩個資料集,同一個結論。
3. **這個資料集上的 VideoMAE 表現,大部分不需要看見受試者。** 91% 的高於隨機訊號在
   受試者被塗掉後仍然存在,56% 完全不需要像素——但那 56% 是**關於受試者的粗略幾何**
   (框的位置大小、片長),不是錄影脈絡:純錄影格式的對照精準落在隨機(0.498)。
   任何引用 Fitness-AQA VideoMAE 數字(包括歷史的 0.555 與本次的 0.640)的說法
   都必須連帶這一句。
4. **身體訊號是真的,而且被場景稀釋。** person-crop 0.666 是全部臂中最高,
   超過 full frame(+0.026)與 pose-only(+0.016)。

### 3.3 據此建議

- **不進 Stage C。** 計畫規定 Stage A 與 Stage B 都通過才投入 EgoExo-Fitness。Stage B
  未通過,依計畫應停在這裡。
- **VideoMAE 保留為研究/診斷 branch**,不接進 production 推論路徑。
- **若未來要再碰這條線,person-crop 是唯一往上的方向,但它的增益尚未被證實。**
  它是本次最強的單一臂,也是唯一主動移除場景的設定;但它相對 pose-only 的 +0.016
  不只 95% CI 含 0,還**無法與取景幾何分離**——`box_geometry` 顯示這類幾何單獨值
  +0.078,而 letterbox 的比例仍把人框長寬比帶進畫面。要把它當結論,需要一個能拿掉
  取景幾何的新對照(例如把每支影片的 crop 重新縮放到固定尺寸),那是下一次的事。
- **這個 shortcut 發現本身值得寫進論文**:它同時解釋了為什麼歷史的 VideoMAE 數字
  「看起來還行但沒用」,也給了 in-the-wild 健身資料集一個具體的方法學警告。


---

## 4. 事後探索(**不屬於 Stage B 的判定**)

### 4.1 pose + person-crop 的 late fusion 超過了兩個 branch

§3 的判定已經定案,以下不更動它。但「person-crop 值不值得繼續」是一個會影響下一步
投入的問題,而它可以直接量——late fusion 完全離線,兩個 branch 的 prediction CSV
都已存在。**先前所有 fusion 臂用的都是 full-frame VideoMAE;pose + person-crop
從未被測過。**

| 臂 | balanced accuracy |
| --- | --- |
| pose-only | 0.650 ± 0.012 |
| person-crop VideoMAE-only | 0.666 ± 0.012 |
| **late fusion(pose + person-crop)** | **0.682 ± 0.016** |

| 條件 | 值 | 判定 |
| --- | --- | --- |
| 1. Δ ≥ +0.02 | **+0.032** | 通過 |
| 2. CI 下限 > 0 | [−0.018, +0.085] | **不通過** |
| 3. 護欄 | recall +0.050、specificity +0.014,**兩者皆升** | 通過 |
| 4. 多數 seed 同向 | 4/5 | 通過 |

**它超過了兩個 branch 各自的值**(pose 0.650、person-crop 0.666)。這是 Stage A 與
Stage B 全部 fusion 實驗中**第一次**出現的現象:先前每一個 fusion 都只收斂到較強的
branch。這就是互補性的操作型定義。

### 4.2 為什麼這個數字**不能**拿來翻案

1. **它是事後選的。** 事前登錄的 primary 是 full-frame VideoMAE。看到它失敗之後再換
   一個變體重跑,正是 §0 開頭要防的 best-of-N。把 0.682 寫成 Stage B 的結果,等於
   把整份事前登錄作廢。**§3 的判定維持不變。**
2. **條件 2 仍然不通過。** 244 支 test 影片撐不出下限大於 0 的區間。
3. **取景幾何的混淆沒有解決。** person-crop 的 letterbox 比例仍編碼人框長寬比,
   而 `box_geometry` 單獨值 +0.078。
4. 一個合理但**未經檢驗**的機制假說:full-frame 的 VideoMAE 訊號主要是場景與粗略
   幾何,那些東西與 pose 不相關但也沒用;person-crop 逼模型只看身體,於是它帶進的是
   **關鍵點丟掉的身體外觀**(軀幹形變、槓位、實際肢體樣貌),那才是 pose 真正沒有的資訊。

### 4.3 person-crop **不是**乾淨的背景消融——它同時改了四件事

這是本階段最大的未解釋因素,而且是實測出來的,不是推理:

| 它改變了什麼 | 實測(全 1623 支)|
| --- | --- |
| 1. 移除背景(**唯一想要的操作**)| 保留面積 100% → 中位數 31% |
| 2. **救回被 centre crop 截掉的身體** | **53% 的影片是非正方形**(480×600);full-frame 路徑下 **38% 的影片有部分身體被裁掉**,p90 裁掉 14.6% |
| 3. 提高身體的有效解析度 | 受試者佔 224×224 輸入的面積 34.0% → 43.8%(線性約 **1.14×**)|
| 4. 幾何正規化 | 人被置中、尺度 canonical 化,另加灰色 letterbox 邊條 |

第 2 項是 Stage A 在 cam18 上記下的同一個教訓:`VideoMAEImageProcessor` 的短邊 224 +
centre crop 224,在 480×600 上會切掉上下各約 1/6。**full-frame 那一臂有 38% 的影片,
模型根本沒看到完整的人。** 因此「person-crop 比 full-frame 高 +0.026」可以被解釋成
「背景是雜訊」,也可以被解釋成「模型終於看到完整而且更大的身體」——本階段無法分離。

**分離它的對照:`full_frame_letterbox`** —— 把整個畫面 letterbox 成正方形但**不裁掉
背景**,消掉第 2/3/4 項而保留背景:

| 臂 | 背景 | 完整身體 | 值 |
| --- | --- | --- | --- |
| full_frame | 有 | 38% 被截 | 0.640(已知)|
| **full_frame_letterbox** | **有** | **完整** | **缺** |
| person_crop | 無 | 完整 | 0.666(已知)|

若它 ≈ 0.666,person-crop 的增益全部來自取景與截斷,§4.1 的 fusion 機制假說要重寫;
若仍 ≈ 0.640,增益確實來自移除背景。成本是一個 kernel、約 3.4 小時。

### 4.4 若要把它變成結論,需要什麼

> 完整的後續設計已獨立成計畫:
> [`videomae_person_crop_validation_plan.md`](videomae_person_crop_validation_plan.md)。
> 其 Stage B1 就是 §4.3 的 F1×F2 分離,且**必須先於任何以 person-crop 為主的設計**。

- 以 **person-crop 為 primary** 重寫一份事前登錄,連同 seeds、fusion 規則、門檻。
- **先跑 §4.3 的 `full_frame_letterbox`**——在知道 +0.026 到底是背景還是取景之前,
  任何以 person-crop 為主的設計都建立在一個未分離的混淆上。這是下一步的第一件事。
- 一個**消掉取景幾何**的新對照:把每支影片的 crop 重新縮放到固定尺寸,讓框的大小與
  長寬比不再進入畫面;再跑一次 `box_geometry` 式的 zero-parameter 對照確認它掉到隨機。
- 條件 2 需要更高的檢定力:目前 244 支 test 影片是硬限制,可考慮改用 repeated
  splits 或把 Fitness-AQA 其他動作(OHP、BarbellRow)一起納入。
