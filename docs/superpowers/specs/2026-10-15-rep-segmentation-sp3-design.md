# 其餘 10 個動作的次數判斷（RS-SP3）— 設計

Status: **草案，待核准** · Created 2026-10-15

接在 RS-SP1（`2026-07-26-rep-segmentation-sp1-design.md`）、RS-SP2
（`2026-07-27-rep-segmentation-sp2-design.md`）與 PR #97 之後。PR #97 讓瀏覽器替 Push-up、Lunge、
Row、Bicep Curl、Sit-up 切次數；這份處理剩下的 10 個：Overhead Press、Deadlift、Band Pull Apart、
Arm Abduction、Arm VW、Shoulder Bridge、Leg Abduction、Torso Twist、Jumping Jacks、High Knee。

---

## 0. 結論先講

**分兩層，第一層一次解決全部 10 個動作的「第幾下」，不用移植任何訊號。**

1. **第一層（後端切）**：瀏覽器沒有某動作的 TS 訊號時，照今天一樣密集抽全片，但**不送
   `reps`**。後端拿到的是完整、全速率的資料，自己跑 `segment_reps`——也就是 CLI、研究資料集、
   每個動作設計文件量測時走的同一條 Python 路徑。零移植、零分歧風險，運算量與今天相同（還省掉
   一趟白做的粗掃，見 §1.3）。
2. **第二層（瀏覽器兩趟抽取）**：純粹是**省運算**的最佳化。前提是先把 rep knobs 接進
   `planReps`/`refineSegments`（§3），然後逐動作移植、逐動作用真實資料量過粗掃常數才上線（§4）。
   有些動作應該永遠留在第一層。

**PR #97 的勘誤，兩條：**

- PR #97 的「次數」效果其實第一層就拿得到；它真正額外買到的是那 5 個動作的省運算。當時沒看出
  `segmentation_disabled` 正在壓掉後端的切割（§1.2），所以把「移植訊號」說成了唯一的路。
- 符合預設 knobs 的動作不只那 5 個：**Arm VW** 也是 `polarity="min"`、不 rectify、`extended`、
  0.4 s，當時漏列。它可以用現在的框架直接接（§5 P1）。

**有一個決定必須由使用者做，不能由設計代做**（§2.5）：Jumping Jacks 與 High Knee 註冊時，
使用者是看著「瀏覽器路徑＝整段評分」的誤報率做決定的。改成逐 rep 評分，正確動作被出卡的比例會
從 18.7% 升到 51.7%（JJ）、23.3% 升到 40.0%（HK）。

---

## 1. 現況

### 1.1 16 個 detector 全部都有 `rep_signal`

| 動作 | `rep_signal` | polarity | rectify | rep_start | min_rep_s | 瀏覽器今天 |
|---|---|---|---|---|---|---|
| Squat | `avg_knee_angle` | min | – | extended | 0.4 | 兩趟（SP2） |
| Push-up / Lunge / Row / Bicep Curl / Sit-up | 見 PR #97 | min | – | extended | 0.4 | 兩趟（PR #97） |
| Arm VW | `avg_arm_elevation_deg` | min | – | extended | 0.4 | 整段 |
| Overhead Press | `avg_elbow_angle` | **max** | – | extended | 0.4 | 整段 |
| Arm Abduction | `avg_arm_elevation_deg` | **max** | – | extended | 0.4 | 整段 |
| Band Pull Apart | `wrist_spread_shoulder_norm` | **max** | – | extended | 0.4 | 整段 |
| Shoulder Bridge | `hip_angle_deg` | **max** | – | extended | 0.4 | 整段 |
| Leg Abduction | `max_thigh_trunk_deg` | **max** | – | extended | 0.4 | 整段 |
| Jumping Jacks | `stance_width_ratio` | **max** | – | extended | 0.4 | 整段 |
| Deadlift | `hip_angle_deg` | min | – | **flexed** | 0.4 | 整段 |
| Torso Twist | `twist_offset_ratio` | **max** | **✓** | extended | 0.4 | 整段 |
| High Knee | `thigh_elevation_difference` | **max** | **✓** | extended | **0.15** | 整段 |

（knobs 以 `registry.get_detector(name)` 實際值為準，2026-10-15 確認。）

### 1.2 `segmentation_disabled` 現在是在「關掉」後端的切割

SP2 §4.1 讓沒有 TS 訊號的動作送 `fallback: "segmentation_disabled"`，理由是「行為與今天逐位元
相同」——那時只有 Squat 有 `rep_signal`，送不送都一樣。八月之後 15 個 detector 都補上了
`rep_signal`，這個 fallback 就從「什麼都不改」變成「**強制後端放棄它本來會做的切割**」：
`run_detector` 收到 `rep_plan` 就跳過 `segment_reps`（`src/pose/movements/base.py:157-171`）。

已實測（合成的 4 下 Lunge、密集全片、經真實的 `analyze_pose_payload`）：

| client 送的 `reps` | 後端回傳 |
|---|---|
| 不送 | `detected: 4, analyzed: [1, 3, 4], fallback: null` |
| `segmentation_disabled` | `detected: 0, analyzed: [], fallback: "segmentation_disabled"` |

### 1.3 沒有訊號的動作還白跑一趟粗掃

`extractPoseWithReps`（`frontend/src/lib/poseExtract.ts:271-291`）先載入 Lite、對全片每 3 幀跑
一次（`:285`），**之後**才查 `TS_REP_SIGNALS[movement]`（`:288`）。查不到時整趟粗掃的結果全是
NaN、直接丟掉。今天這 10 個動作每次分析都多付一次 Lite 模型載入加約 1/3 片長的推論。

---

## 2. 第一層：讓後端切（全部動作）

### 2.1 規則

> **client 只有在送出稀疏資料時才送 `reps`。** 密集抽了全片，就不送 `reps`，由後端決定。

`reps` 存在的唯一理由是 SP2 §4.3：區間外的幀沒有 landmarks，後端拿有洞的訊號重切會「默默產出
看似正常但錯誤的區間」。全片都抽了就沒有洞，這個理由不成立，後端自己的切割反而是權威實作。

### 2.2 為什麼安全

- 後端走的是 SP1 路徑（`rep_plan=None`），與 CLI、`/api/analyze`、各動作設計文件量測時完全
  相同。沒有任何新程式碼參與切割。
- `/api/analyze/pose` 本來就支援不送 `reps`（`frontend/src/api.ts` 的 `analyzePose`：
  `if (reps) form.append(...)`；註解寫明省略即走 pre-SP2 路徑）。
- 前端沒有任何 UI 讀 `segmentation_disabled` 這個字串（全 repo 搜尋僅 `poseExtract.ts` 產生它）；
  時間軸讀的是後端回傳的 `reps` 輸出，兩條路徑形狀相同。
- 後端 `_FALLBACKS` 保留 `segmentation_disabled`，舊 client 不受影響。

### 2.3 改動

1. `planReps`（`poseExtract.ts:90-130`）：movement 不在 `TS_REP_SIGNALS` 時回傳 `plan: null`、
   `spans: [全片]`。`RepsPlan | null` 一路傳到 `extractPoseWithReps` 的回傳值，
   `analysisJob.tsx:251-268` 原樣交給 `api.analyzePose`（`reps` 為 `undefined` 即不送）。
2. `extractPoseWithReps`：**先**查 `TS_REP_SIGNALS`，查不到就跳過粗掃與 Lite 的載入，直接以
   使用者 tier 密集抽全片（§1.3）。
3. `refineSegments` 對 `null` plan 原樣回傳。
4. 測試：`lib.poseExtract.test.ts` 的 `"falls back for a movement with no browser-side signal"`
   改為斷言 `plan === null` 且 spans 為全片；新增一個 analyzePose 不帶 `reps` 欄位的測試；後端加
   一個端點測試：多下的密集 pose、不帶 `reps`、非 Squat 動作 → `reps.detected > 0`。

### 2.4 粗掃失敗時要不要也交給後端（建議：要）

`no_reps_detected` / `only_partial_reps` 兩種 fallback 一樣是「密集抽全片」，照 §2.1 的規則也
不該送 `reps`。差別在於：粗掃用的是 Lite＋10 Hz＋3 點中位數，密集全片是使用者 tier＋30 Hz——
粗掃切不出來時，後端在較好的資料上仍可能切得出來；切不出來它自己會回報同一個 fallback。

**這會改到 Squat 與 PR #97 那 5 個動作的行為**（今天粗掃失敗＝一定整段；改後＝後端再試一次），
所以與 §2.3 分開成獨立 commit、獨立測試，並在 PR 說明裡列出。

### 2.5 會改變什麼——要誠實列出

1. **修正多下影片的 phase 錯標**：這正是 SP1 存在的理由——每個 `assign_phases` 都假設一段只有
   一下。今天這 10 個動作在瀏覽器上全部仍帶著這個 bug。
2. **只評前/中/後 3 下**（`DEFAULT_MAX_REPS`）。沒被選中的 rep 裡的錯誤不會出卡；Squat 從 SP1
   起就是如此。
3. **誤報率會變，而且有兩個動作已量過、而使用者是看著舊數字決定上線的**
   （`notes/egoexo-silent-rules-full-archive.md:47-56`；`jumping_jacks.py:699-711`、
   `high_knee.py:705-714`）：

   | 動作 | 路徑 | 正確動作被出卡 | 有問題的動作被出卡 |
   |---|---|---|---|
   | Jumping Jacks | 逐 rep（第一層會變成這個） | 166/321（51.7%） | 28/35（80.0%） |
   | Jumping Jacks | 整段（今天） | 60/321（18.7%） | 20/35（57.1%） |
   | High Knee（側面相機） | 逐 rep | 36/90（40.0%） | 19/22（86.4%） |
   | High Knee（側面相機） | 整段 | 21/90（23.3%） | 11/22（50.0%） |

   逐 rep 是規則門檻**被驗證時**的模式，敏感度也高得多；但誤報加倍以上。這是產品取捨，
   **需要使用者決定**：(a) 兩者一起切到逐 rep；(b) 這兩個動作暫時維持送 `segmentation_disabled`
   （一個明列的 per-movement 例外，附上這張表），其餘 8 個走第一層。
4. **各動作已知的切割弱點會直接被使用者看到**（今天它們只在伺服器路徑出現）：
   - Leg Abduction：REHAB24-6 210 下中 35 下（17%）退回整段，恰好就是有效幀為 0 的那 35 下
     （`leg_abduction.py:84-89`）——這是正確的退回，不是錯切。
   - Shoulder Bridge：靜止時的抖動會被切成 rep（百分位閾值無尺度，`shoulder_bridge.py:379-388`）；
     `action_11_exo_r` 兩下皆 partial → 整段 16 s 當一下評分（`:439-445`）。
   - 全部動作：`TODO.md:42` 的「靜止片段被切成 3 下」仍然存在。

### 2.6 成本

與今天相同，再減去 §1.3 那趟白做的粗掃。

---

## 3. 第二層的前置：把 knobs 接進瀏覽器（不新增任何動作）

今天的瀏覽器程式碼處處假設 Squat 的形狀。PR #97 用 golden test 擋住了「把非預設 knobs 的動作
加進 `TS_REP_SIGNALS`」，這一節是把擋板換成真正的支援。

| # | 改動 | 位置 | 為什麼 |
|---|---|---|---|
| F1 | `TS_REP_SIGNALS` 的值改成 `{ signal, polarity, rectify, repStart, minRepSeconds }`，鏡像 `MovementDetector` | `repSignal.ts` | 讓 knobs 有地方住 |
| F2 | golden test 從「knobs 必須是預設值」改為「TS 條目的 knobs 必須等於 Python detector 的」 | `lib.repSignal.golden.test.ts:43-53` | fixture 已經記錄 knobs（PR #97），只換斷言 |
| F3 | `planReps` 把 knobs 傳給 `segmentReps` | `poseExtract.ts:105` | 今天一律預設值 |
| F4 | **span 錨點改用 oriented 訊號**：`valleyPosition(oriented(smoothed, polarity, rectify), rep)` | `repSpans.ts:61`、`spanForRep:94` | 今天是原始訊號的 argmin。`max` 動作的原始 argmin 是**休息端**（window 邊緣），span 會偏到半下之外；rectify 動作則該錨在 \|x\| 最大處。對 Squat 與 PR #97 那 5 個，oriented 是恆等，量測結果不變 |
| F5 | **`flexed` 的 span 不錨在谷底**：改用 `[start·3 − P, end·3 + P]` | `spanForRep` | `flexed` window 是谷底到谷底，谷底就在 window 兩端；以谷底為中心 ±(半寬 + 24) 只蓋得到半下＋24 幀，rep 超過約 1.6 s 就被截。反過來說，`flexed` 的邊界**本身就是谷底**，而 46 支深蹲量到谷底位置誤差 ≤ 5 幀，所以 P 可能遠小於 24——要量 |
| F6 | `refineSegments` 的 `coarseBand(…, polarity, rectify)`、`refineWindow` 的 `segmentReps(…, knobs)` | `poseExtract.ts:153`、`repSpans.ts:164` | band 必須在同一個 oriented 空間（`coarseBand` 的 docstring 已警告） |
| F7 | 訊號函式回傳**有號值**，rectify 只在 `segmentReps` 內做 | `repSignal.ts` | Python 先對有號值做 5 幀中位數（`base.py:151-154`），再在 `segment_reps` 內取絕對值；「中位數再取絕對值」≠「取絕對值再中位數」。現在的 `planReps` 已經是這個順序，寫成不變式並加測試 |
| F8 | golden test 容差改為逐動作、帶單位 | golden test | 現在是 1e-4 **度**；Band Pull Apart、Jumping Jacks、Torso Twist 的訊號是比值。Leg Abduction 的 Python 在 float32 下先正規化向量再取 arccos，夾角接近 0°/180° 時誤差被放大：兩次獨立的 5000 隨機幀量測，對 float64 重算最大差 0.0065°／0.0097°，約 2.5% 的幀超過 1e-4——要嘛 TS 用 `Math.fround` 模擬，要嘛該動作容差放寬到 0.02°（仍比真正的移植錯誤小兩個數量級）。建議後者，量到的最大值寫進註解 |
| F9 | `tests/test_coarse_segmentation_corpus.py` 參數化為 `(movement, corpus_dirs)`，五個閘門逐動作計算、逐動作設門檻 | 該檔 | 粗掃常數（`COARSE_STRIDE`、`REP_PADDING_FRAMES`、`MAX_VALLEY_ERROR`）只在 46 支深蹲上量過。每個動作上第二層前都要過自己的量測；資料在 `data/`（gitignored），所以與現在一樣在 CI 跳過、在本機咬人 |

F1–F9 是一個 PR、不加任何新動作；完成後**回頭替 PR #97 那 5 個動作補量** F9（它們目前沿用深蹲的
常數、未量測）。

---

## 4. 逐動作

三份程式碼稽核的共同結論：**10 個動作的 `rep_signal` 都只讀當幀**（沒有整段 baseline、沒有整段
中位數、沒有導數）。跨幀的東西只有框架共用的 5 幀中位數，以及各動作 rule 裡的 baseline——都不在
訊號裡。所以粗掃/密集兩趟不會讓訊號本身失真；風險全在「粗掃的時間解析度」與「span 常數」。

### 4.1 只差 knobs——訊號函式現成可用（已驗證）

在 golden fixture 的 29 幀真實 landmarks＋400 幀隨機壓低可見度的變體上，以下三對 Python 輸出
**0 筆不一致**（含 gate 行為）：

| 動作 | 等同於 | TS 函式 | 差在 |
|---|---|---|---|
| Overhead Press | Bicep Curl 的 `avg_elbow_angle` | `bicepCurlAvgElbowAngle` | polarity `max` |
| Shoulder Bridge | Sit-up 的 `hip_angle_deg` | `sitUpHipAngle` | polarity `max` |
| Arm VW ↔ Arm Abduction | 彼此相同 | 新增一個 `gatedAngles(…, [[23,11,13],[24,12,14]], meanFinite)`，gate＝肩、肘、髖 | Arm VW `min`、Arm Abduction `max` |

### 4.2 逐動作判定

| 動作 | 移植難度 | 節奏（量測來源） | 視角 | 量測資料 | 判定 |
|---|---|---|---|---|---|
| **Arm VW** | 新 joint list，helper 現成 | 單下 2.20–19.73 s，中位 4.70 s（REHAB24-6，`arm_vw.py:219-224`） | 正面 | REHAB24-6 Ex2；Fit3D `overhead_trap_raises` | **P1**：預設 knobs，現在就能接；但要先過 F9（rep 長達 20 s，遠超深蹲） |
| **Overhead Press** | 零（重用函式） | 未記錄 | 側面為主 | Fit3D 是否有標準肩推**未確認** | P3，資料先確認 |
| **Arm Abduction** | 零（同 Arm VW） | Fit3D 1.40–4.96 s；REHAB24-6 2.77–10.53 s，中位 4.83 s | 正/背面 | REHAB24-6 Ex1（178 下，全為單臂——休息臂約 26°，振幅約減半）；Fit3D `side_lateral_raise` | P3 |
| **Jumping Jacks** | 新 2-D 距離比（`dist(27,28)/dist(11,12)`，肩寬 > 1e-8） | 片段平均最快 1.14 Hz（≈ 8.8 粗掃點/下，夠；單下時長分佈未記錄） | 三個 EgoExo 相機都能切 | EgoExo 356 組 | P3；**先等 §2.5 的決定** |
| **Leg Abduction** | 新 2-D 向量夾角＋`maxFinite`；gate 含**訊號沒讀的腳踝** | 未記錄（僅 Madsen 的 2-1-2 s 節拍） | 正面 | REHAB24-6 Ex4（210 下） | P3；容差見 F8 |
| **Band Pull Apart** | 新 2-D 距離比；gate 含**雙耳** | Fit3D 1.78–2.92 s（25 下） | **強烈正面**：側面時雙手重疊、振幅消失 | Fit3D（需先抽 pose JSON） | P3 |
| **Deadlift** | 新 2-D 中點夾角（各向量 < 1e-6 → NaN，與 `angle_degrees` 的乘積 1e-8 不同） | 無真實資料（2.5–2.8 s 來自合成片） | 側面 | Fit3D 有 `deadlift`（`src/fit3d/threshold_audit.py:17`）；量切割不需要對錯標註 | P4：需 F5，而 F5 本身要靠這份資料量 |
| **Shoulder Bridge** | 零（重用函式） | 全 repo 只有 2 段片（19.5 s、16.0 s） | 必須近側面；正對身體軸時中位數 90° vs 側面 128–134° | 不足 | **留在第一層**，直到有資料 |
| **Torso Twist** | 新 2-D 投影（有號；零點＝雙手投影在髖中點，免校正） | 未記錄；repo 內沒有坐姿轉體影片 | 未記錄 | 無（Fit3D 的 `standing_ab_twists` 是別的變體） | **留在第一層**，直到有資料 |
| **High Knee** | 新 2-D 餘弦差（有號） | 18 組片段上 0.70–2.20 Hz，中位 1.31 Hz（`high_knee.py:687-690`）；`min_rep_seconds=0.15` | **強烈側面**；正面相機被 rule 的 view gate 靜音 | EgoExo 198 組 | **留在第一層**（見下） |

**High Knee 為什麼不上第二層。** 2.2 Hz 時每下只有約 4.5 個粗掃點，3 點中位數覆蓋 0.3 s，
兩下之間 rectify 後的零點尖谷大約只有一個取樣寬——粗掃會**算錯次數**，而次數錯是 SP2 §8 說過
padding 與精修都救不回來的唯一錯誤。把它的粗掃改成每 1–2 幀一次能解析度夠，但那等於對全片跑
Lite 再加密集，只有使用者 tier 比 Lite 重時才可能省到；這筆帳要量過才知道值不值，不是設計階段
能決定的。第一層對它已經完整正確。

（稽核報告曾推論「High Knee 的 span 會互相合併、省不到」；這不成立——`selectReps` 只取首/中/尾
三下，在長片裡彼此相距甚遠。真正的問題是上面的計數解析度。）

---

## 5. 實作順序

| 階段 | 內容 | 依賴 | 誰受影響 |
|---|---|---|---|
| **P0** | 第一層（§2.1–2.3）＋跳過白做的粗掃 | §2.5 的決定 | 10 個動作開始有次數 |
| P0b | §2.4：粗掃失敗也交給後端 | P0 | Squat＋PR #97 的 5 個（行為微變） |
| **P1** | Arm VW 接上兩趟（預設 knobs） | 本機跑 F9 的 Arm VW 量測 | Arm VW 省運算 |
| **P2** | 框架 F1–F9，不加動作；回補 PR #97 那 5 個的 F9 量測 | – | 無行為變化（oriented 對 min 動作是恆等） |
| P3 | Overhead Press、Arm Abduction、Jumping Jacks、Leg Abduction、Band Pull Apart，逐個一 PR，每個都附自己的 F9 量測結果 | P2；各自的本機資料 | 逐動作省運算 |
| P4 | Deadlift（F5 用 Fit3D 量） | P2 | – |
| — | Shoulder Bridge、Torso Twist、High Knee 留在第一層 | 有資料或量過成本再議 | – |

P0 是唯一一個使用者可見的大變化，也是唯一一個需要先做決定的階段。P1 以後全是省運算，次數判斷
在 P0 之後就已經全部正確（以後端 Python 為準）。

---

## 6. 這份設計**不**主張的事

- 不主張粗掃常數（`COARSE_STRIDE=3`、`REP_PADDING_FRAMES=24`）對任何非深蹲動作成立——包括
  PR #97 已上線的 5 個。它們沒被量過；F9 是讓它們被量的機制。
- 不主張逐 rep 評分對 Jumping Jacks / High Knee「比較好」——它敏感度高、誤報也高，見 §2.5。
- 不主張下列 rep 時長存在：Overhead Press、Leg Abduction、Torso Twist 都未記錄；Jumping Jacks
  與 High Knee 的 Hz 是「次數÷片段長」的平均，不是單下時長的分佈。
- 不主張第一層會讓後端的切割變好——它只是讓瀏覽器使用者拿到伺服器路徑本來就有的結果，包括其
  已知弱點（§2.5 第 4 點）。

---

## 7. 待決事項

1. **Jumping Jacks / High Knee 在第一層是否也切到逐 rep**（§2.5 的 (a) 或 (b)）。
2. 是否接受 P0b 對 Squat 與 PR #97 5 個動作的行為微變（§2.4）。
3. P3 各動作的本機資料是否齊備——特別是 Overhead Press 在 Fit3D 是否有對應動作、Band Pull Apart
   是否已有抽好的 pose JSON。
