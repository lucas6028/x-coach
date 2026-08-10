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
3. **控制組的變體影片是新編碼的**。person-crop 與 background-only 由同一組 pose box
   產生,frame count 補齊到來源 header,確保三個變體的 clip start 完全一致——否則「控制
   組掉了幾分」會混進「取樣到不同畫格」這個無關因素。

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

**(d) 再編碼的混淆與 `reencoded` 對照臂。** 兩個控制組都經過一次 `cv2.VideoWriter`
再編碼,`full_frame` 沒有。這個不對稱只可能**製造**退化、不可能掩蓋退化,所以控制組
**存活**可以直接解讀;控制組**下降**則與編碼損失混在一起——而那正是會左右保留決策的
方向。因此另建一個 `reencoded` 恆等變體(同一條 decode/encode 路徑、不動任何像素)。
它只在某個控制組真的下降時才需要抽特徵。

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

# 3. 控制組變體影片(需要步驟 1 的 pose)
python scripts/video/build_video_variants.py --variant person_crop     --jobs 6
python scripts/video/build_video_variants.py --variant background_only --jobs 6
python scripts/video/build_video_variants.py --variant reencoded       --jobs 6   # 只在控制組下降時才需要

# 4. VideoMAE 抽取(Kaggle,每個變體一個 kernel;本機 CPU 實測 5.3 s/clip ⇒ 每個變體 7–10 h)
#    上傳必須從資料集目錄「裡面」執行:kaggle CLI 會把 -p 的相對路徑
#    直接拼進暫存檔名,路徑含斜線就會 [Errno 2]。
#    (PowerShell)  Push-Location .kaggle_tmp/fitaqa_videomae_input
#                  uv run --with kaggle kaggle datasets create -p .
#                  Pop-Location
#    kernel:       uv run --with kaggle kaggle kernels push -p .kaggle_tmp/fitaqa_videomae_extract

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
| pose 重抽 | *(進行中)* |
| pose 特徵 / 分母門檻 | *(待執行)* |
| 變體影片 | *(待執行)* |
| Kaggle 抽取 × 3 | *(待執行)* |
| 各臂訓練與證據表 | *(待執行)* |

已完成的前置檢核:

- 抽取器 CPU smoke test 通過(3 支影片),`fc_norm` weight mean 0.6832 / bias mean 0.0083,
  與 Stage A 在本機與 Kaggle 上量到的值**完全一致**。
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
