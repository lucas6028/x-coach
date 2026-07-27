# 只抽取選中 rep 區間（RS-SP2）— 設計

Status: **設計已核准，待實作** · Created 2026-07-27

讓瀏覽器不再對整段影片逐幀跑 MediaPipe，只對「會被評分的那幾下」密集抽取。
這是 RS-SP1（`docs/superpowers/specs/2026-07-26-rep-segmentation-sp1-design.md`）明文預留的後續，
也是使用者原始需求裡**省運算**那一半真正兌現的地方——SP1 誠實記錄過，它自己那一層幾乎省不到。

既有實作見 `src/pose/rep_segmentation.py`、`src/pose/movements/base.py`、
`frontend/src/lib/poseExtract.ts`；擷取層的原始設計見
`docs/superpowers/specs/2026-07-21-client-side-pose-capture-sp1-design.md`。

---

## 0. 為什麼會有兩趟抽取

要省下「對整片跑 MediaPipe」，得先知道 rep 在哪；要知道 rep 在哪，得先有姿態資料。這是個
迴圈。出路是：**切割需要的資料，比分析需要的少得多。**

| | 切割（找「哪裡」） | 分析（找「哪裡錯」） |
|---|---|---|
| 需要什麼 | 每個取樣點**一個數字**（平均膝角） | 每幀**33 點**完整骨架 |
| 時間解析度 | 一下深蹲 2–3 秒，每秒 5–6 點就看得出那條凹陷 | 逐幀。規則要抓最低點是哪一幀、腳跟何時離地、膝內扣峰值 |
| 模型品質 | Lite 足夠——只需趨勢對，不需角度精準 | 使用者選的 tier（Lite 與 Heavy 對 squat 的判定僅 50% 一致） |

所以粗掃是**便宜且稀疏**的一趟，只回答「第幾秒到第幾秒是一下」；密集是**貴且完整**的一趟，
只跑在選中的區間上。

**單趟做不到，這不是實作偷懶。** `select_reps` 取的是首/中/尾，這要求先知道總共幾下才能選；
串流走到第 3 下時還不知道總共有 5 下。而「就取前三下」正是 SP1 明文否決的做法——最後一下帶的
是疲勞走樣，系統性漏掉它等於把最該提醒的錯誤藏起來。

成本（30 秒、5 下、30fps＝900 幀）：今天是 900 幀 × Heavy；SP2 是 ~300 幀 × Lite（粗掃 10fps）
＋ ~270 幀 × Heavy（3 下 + padding）。貴的那一項從 900 降到 270。

考慮過但否決：**不用姿態切割**（光流、幀差）。它能省掉粗掃，但那是一條全新的、未驗證的訊號，
且會與 Python 的 `avg_knee_angle` 徹底脫鉤——SP1 留下的共用 fixture 就管不到它。用同一個量、
只是取樣稀疏一點，兩邊才對得起來。

---

## 1. 範圍

**做**

- 瀏覽器擷取管線改成兩趟：粗掃切 rep → 密集抽選中區間 → 在密集訊號上精修邊界
- `rep_segmentation.py` 的 TS 移植（`segmentReps` / `selectReps`），共用 SP1 的 fixture
- squat rep 訊號的 TS 移植（`avgKneeAngle` + `centeredMedian`）
- `poseExtract.ts` 重構成「一個取樣函式，兩個呼叫端」
- `/api/analyze/pose` 收選填的 `reps`；`run_detector` 接受外部區間並跳過自己的切割
- `quality` 附加 `extracted_frames` / `extracted_frame_ratio`
- 未分析區段的 UI（Timeline 標示、結果頁一行說明、MetricsCards 分母）
- 三個實測任務（§3），數字決定 padding 與後續決策

**不做**

- 伺服器端 `src/pose/process_videos.py` 的逐幀 RTMPose（研究用批次腳本，離線跑、沒人在等，
  省它的價值低很多）。**範圍只有瀏覽器這一條**——真實使用者走的路徑。
- 重用錄影時的即時 landmarks 當粗掃訊號（見 §2.4，延後且不吃虧）
- squat 以外動作的 TS rep 訊號（seam 留好，走 fallback）
- 改 `quality` 既有欄位的計算、或減少 `frame_metrics` 的列數

---

## 2. 架構

### 2.1 管線

```
blob
 │
 ├─ 1. 粗掃  frame_index = 0, 3, 6, …（Lite）
 │      每點 → avgKneeAngle(landmarks)          ← 移植自 pose_rule_detector.py:143-167
 │      → centeredMedian(window=3)              ← 移植自 geometry.py:108
 │      → segmentReps(signal, fps=10, "min")    ← 移植自 rep_segmentation.py
 │      → selectReps(reps, 3)
 │      → 位置 × COARSE_STRIDE 換回 frame_index
 │
 ├─ 2. padding + 合併重疊區間 → 要密集抽取的 frame_index 集合
 │
 ├─ 3. 密集抽取  對集合內每個 frame_index seek（使用者選的 tier）
 │
 ├─ 4. 邊界精修  對每個 padded span 的 30fps 訊號再跑一次 segmentReps
 │      → 得到「整片密集抽取本來會給的邊界」
 │
 └─ 5. 組 payload：全長 frames（未抽取為 null）+ 精修後的 reps 區塊
```

### 2.1.1 邊界精修（步驟 4）

密集抽取完 `[start - PAD, end + PAD]` 之後，那一段**已經有 30fps 的完整訊號**。對這一小段再跑
一次 `segmentReps`（同樣的參數、`fps=30`、平滑窗回到 SP1 的 5），取其中與粗掃區間重疊最多的那個
window 當作精修結果。

**這讓粗掃的邊界誤差完全消失，而不是被 padding 蓋住。** 差別很實際：`assign_phases` 把區間的前
15% 當 setup，所以起點晚 3 幀就會讓「setup」落在下降段中間——正是 SP1 整份 spec 在修的那個 bug，
只是換了個原因重新發生。padding 加寬區間只能用猜的蓋住它，而且會把站立幀吸進區間（SP1 的
`_climb_backward` 特地不吸收 idle，那等於部分回退）。精修則是直接算對。

**退化情況**：精修在該 span 內切不出任何 window（訊號太爛、或粗掃切錯了）→ **退回粗掃給的邊界**，
並在該 segment 上記 `refined: false`。不中斷、不丟棄，也不假裝精修過。

`reps.segments[]` 送出的是**精修後**的 `start_frame` / `end_frame`。

### 2.2 稀疏 payload 的表達：全長陣列 + null

client 仍送 900 筆 frame，只有選中區間那 ~270 筆有 landmarks，其餘 `landmarks: null`。

- `RepWindow.start/end` 是**序列位置**而非 `frame_index`。全長陣列讓兩者恆等，後端所有索引
  運算一行都不用改。
- `raw_frame_metrics`（`pose_rule_detector.py:130-141`）已把 `landmarks: null` 判成
  `valid: False`；`build_pose_block_from_payload`（`analysis.py:43-45`）已把它變成 `lm: None`。
  兩邊都不需要改。
- `frame_metrics` 仍每幀一列、`metadata.total_frames` 仍等於 `frames.length` 且仍是真實片長，
  所以 `Timeline.tsx:15` 拿它算時長不會壞。SP1 §5 的相容性承諾原封不動。
- 代價：JSON 多帶 ~630 個 null 幀。壓縮後可忽略。

否決的兩個替代：

- **只送抽取到的 frame，靠 `frame_index` 定位**：頻寬小，但後端每處都要把 `frame_index` 映射回
  位置、`RepWindow` 語意要重新定義、`frame_metrics` 列數變少（SP1 明文說不安全）、
  `total_frames` 與 `frames.length` 脫鉤。省下的頻寬不值這些。
- **只送稀疏 frame，後端自己重新切割**：不可行。不是脆弱，是**欠定**——區間外的資料根本不
  存在，`segment_reps` 無從推導。

### 2.3 rep 邊界的權威是 client

由 2.2 直接推出：後端在這條路徑上不再切割，也不再選取（`select_reps` 必須由 client 做，
否則它不知道要密集抽哪幾段）。這是 SP2 最重要的權責轉移，後端的驗證責任（§4.3）因此變重。

### 2.4 粗掃的訊號來源：統一走獨立粗掃

SP1 §21 的原始構想是「錄影時留下 live landmarks 當粗掃訊號」。SP2 **不這麼做**，理由不是
複雜度，是結構：

**上傳模式沒有即時 landmarks，粗掃程式碼無論如何都必須存在。** 重用不是粗掃的替代方案，
而是疊在它之上、只服務錄影模式的額外優化，且只換掉「訊號從哪來」這一個介面——切割器、選取、
密集抽取、payload、後端全都不動。所以**延後做，之後的成本跟現在做一樣**。

現在做則要先付一個測不掉的風險：`performance.now()` 與 blob 時間軸的錨點（MediaRecorder 從
`rec.start()` 到實際編碼出第一幀的延遲）只能在真機上量，而錨錯半秒就會讓區間偏移 → 密集抽到
錯的區間 → 規則在錯的 phase 上跑 → 產生**看起來正常但實際錯誤的判定**，且後端信任它、沒有任何
東西會讓這個偏移浮出來。SP1 對高抬腿下過同一個結論：切錯比不切更糟。

**T3（§3）量出粗掃佔錄完後總等待的比例後，再拿數字決定要不要接。** 這與 SP1 §7 對 Lite 預設
的處理是同一個模式：先驗證，再定預設。

### 2.5 `frame_index` 釘在 30fps 正規格點

`frame_index = round(t * 30)`，粗掃與密集共用同一條規則。

目前 `poseExtract.ts:140` 用的是自增計數器 `i`，只在步長剛好 `1/30` 時與上式等價。粗掃步長不同，
沿用計數器會讓它產出的區間落在**錯的索引空間**，而且不會報錯——這是必須先修的既有隱患。

粗掃取樣是 30 格點的整數跨步：`COARSE_STRIDE = 3`（即 10fps）。粗掃第 k 點 ↔ `frame_index = 3k`，
映射是乘法，不是浮點對齊。

### 2.6 新檔與重構

`frontend/src/lib/`：

- **`repSegmentation.ts`** — `segmentReps` / `selectReps`，`rep_segmentation.py` 的逐行移植。
  門檻用**同名常數**（`ENTER_FRACTION`、`EXIT_FRACTION`、`PERCENTILE_LOW/HIGH`、
  `DEFAULT_MIN_REP_SECONDS`），且讀 SP1 已經留下的 `tests/fixtures/rep_segmentation_cases.json`。
  任一邊改門檻，兩邊測試同時紅——這正是 SP1 §7 蓋那個 fixture 的理由。
- **`repSignal.ts`** — `avgKneeAngle` + `centeredMedian`。膝角**照抄** Python 的公式
  （含 `visible_point` 的可見度 gating 與左右 `mean_finite`），不是自己重寫一個三點夾角。
  另含 `TS_REP_SIGNALS: Record<string, (lm) => number>`，SP2 只填 `Squat`。

**平滑窗是 3 而不是 SP1 的 5，這是刻意的。** SP1 在 30fps 上用 `window=5`，涵蓋 0.17 秒；同一個
數字放在 10fps 的粗掃上會涵蓋 0.5 秒，足以把一下深蹲的底部抹平。窗要照**時間**而非幀數對齊，
10fps 下最接近 0.17 秒的奇數窗是 3（0.3 秒）。這不影響 fixture 的約束——fixture 釘的是
`segmentReps` 本身，平滑發生在它之前。兩邊訊號因取樣率而不同是**由建構決定的**，T2 量的正是
這個差異造成的邊界誤差。

`poseExtract.ts` 重構成：

```ts
sampleFrames(video, landmarker, frameIndices: number[], onProgress) → PoseJsonFrame[]
```

粗掃與密集是同一個函式的兩個呼叫端，差別只在 frame 清單與 tier。`resolveDuration` 那段
（live-record 的 WebM 沒有 Duration，見該檔案 §35-47 的註解）維持不動——粗掃同樣依賴它。

**模型載入**：粗掃恆 Lite、密集用使用者 tier，最壞情況載入兩次；tier 相同時共用同一個 instance。
短片上第二次 init 可能吃掉一部分節省，列入 T3 觀察。

### 2.7 fixture 保護不了什麼

`rep_segmentation_cases.json` 只釘住「**給定一條 1-D 訊號 → 得到哪些區間**」。它**完全沒有**釘住
兩邊算出來的訊號是不是同一條。而 §2.3 之後後端信任 client 的區間，所以訊號分歧永遠不會有東西
讓它浮出來。這是選擇移植 `avg_knee_angle`（而非改用瀏覽器好算的 proxy）的理由：兩邊算的是同一個
量，可以實測對拍。§3 的 T1 就是那個對拍。

---

### 2.8 粗掃取樣對切割結果的影響（已實測）

寫這份 spec 時已經量過一次：把 `tests/fixtures/rep_segmentation_cases.json` 的每個 case 抽三取一
（`signal[::3]`、`fps/3`），模擬粗掃，再與密集結果比對。

**11 個 case 中 10 個 rep 數量完全相同，邊界誤差 ≤ 3 幀（換算回 30fps 格點）。**

這是 `REP_PADDING_FRAMES` 的第一個有證據的下限：合成訊號上量化誤差不超過 3 幀。真實影片會更差
（MediaPipe 抖動），所以 T2 仍要跑，但數量級確認了。

**唯一的分歧必須寫下來**：`flexed_static_glitch` 這個 case，密集路徑正確判定「0 下」（靜止片段
裡的一個抖動不是 rep），**粗掃路徑卻切出 2 下**。原因是抽樣改變了百分位的分布，連帶改變
`enter`/`exit` 帶的位置，讓那個抖動相對於帶變「寬」，於是通過了 `_windows_from_valleys` 的
duration 測試。

兩件事因此成立：

1. **這只發生在 `rep_start="flexed"`**（硬舉）。Squat 走 `"extended"`，SP2 唯一實作的動作不受
   影響。但**任何未來要在 TS 側啟用切割的 flexed 動作，必須先重跑這個比對**，不能假設 squat
   驗過就通用。
2. 這個抽樣比對要成為**常設測試**（§6），把 `flexed_static_glitch` 釘成已知分歧，讓這個不對稱
   永遠看得見，而不是留在一份 spec 的段落裡。

一般性的結論：**粗掃在異常拒絕上比密集寬鬆**。它可能切出密集路徑會拒絕的區間，而後端信任它。
Squat 目前量到的是 0 個這種案例，但這是「已量到 0」，不是「不可能」。

---

## 3. 實測任務（數字決定常數）

| 任務 | 怎麼做 | 決定什麼 |
|---|---|---|
| **T1 訊號 parity** | 拿一支真實深蹲影片密集抽取一次，同一份 landmarks 分別餵 TS 的 `avgKneeAngle` 與 Python 的 `compute_raw`，逐幀比對 | TS 移植對不對。差異應為浮點級 |
| **T2 span 要多寬才包得住真邊界** | 同一支片，量 (a) 全片密集訊號切出的真邊界 與 (b) 粗掃邊界 的距離分布，取上尾 | **`REP_PADDING_FRAMES` 的值** |
| **T3 粗掃佔比** | 量粗掃與密集各自的牆鐘時間（含模型載入） | 之後要不要接即時 landmarks（§2.4）；並驗證整體真的變快 |

**T1 與 T2 必須在 TS 移植（§7 步驟 2）之後跑**——它們比對的正是移植的產物。§7 的順序已照此排。

**padding 的定義（§2.1.1 的精修讓它變乾淨了）**：padding 不吸收邊界誤差——精修才做那件事。
padding 只需要**大到讓真正的邊界確定落在 padded span 之內**，加上平滑要的鄰居：

```
REP_PADDING_FRAMES = 真邊界與粗掃邊界的距離（T2 量，取上尾）+ 平滑半徑 2 幀
```

平滑半徑那一項是硬的：`centered_median`（`geometry.py:117`）在窗內跳過 NaN，所以洞不會污染鄰居、
只會縮小取樣數；≥ 2 幀即保證 rep 內每幀都有完整的 5 點窗。

**這個定義是可以保守取值的，而舊定義不是。** 舊定義下 padding 直接決定最終答案的誤差，取太大
會吃掉節省、取太小會切錯；新定義下 padding 只決定「真邊界有沒有被包進來」，往大取只損失一點
抽取量，不影響正確性。

先寫 **8 幀（0.27 秒）**當佔位，T2 出數字後改掉。它的正當性完全來自 T2，在 T2 跑完前不要當成
已驗證的數字。

**padding 幀不進 rep 區間**：它們的 phase 是 `"rest"`、不被評分。`assign_phases` 拿到的是
**精修後**的 rep 區間。

---

## 4. Fallback、資料契約、後端

### 4.1 Fallback 一律「密集抽取全片」

永遠不送「稀疏又沒有區間」的東西。省不到就是省不到，不拿正確性換。

| 情況 | `reps.fallback` | client 行為 |
|---|---|---|
| 粗掃切不出 rep | `"no_reps_detected"` | 密集抽全片 |
| 只有 partial rep | `"only_partial_reps"` | 密集抽全片 |
| 該動作 TS 側沒有 rep 訊號 | `"segmentation_disabled"` | 密集抽全片 |

三個字串與 SP1 §4.2 完全相同，後端不需要新的分支。

第三格是新的 seam（`TS_REP_SIGNALS` 沒有該 movement 的 key）。SP2 只填 Squat，其他動作自動走
fallback，行為與今天逐位元相同——**SP2 因此不會擋住任何動作的上線**。

### 4.2 client 送出的 `reps`

`/api/analyze/pose` 新增選填欄位：

```jsonc
"reps": {
  "max_reps": 3,
  "fallback": null,
  "segments": [
    {"index": 1, "start_frame": 12, "end_frame": 58, "partial": false, "analyzed": true},
    {"index": 2, "start_frame": 59, "end_frame": 104, "partial": false, "analyzed": false,
     "refined": false}
  ]
}
```

`refined` 只對 `analyzed` 為真的 segment 有意義（沒被密集抽取的 segment 無從精修，一律 `false`）。
它是**診斷欄位**，後端不因它改變行為——見 §8 為什麼不丟棄未精修的 rep。

**這個 `reps` 與 SP1 §5 回傳 payload 裡的 `reps` 不是同一個形狀，不要混用。** 這裡是**輸入**，
用 `segments[].analyzed` 表達選取；SP1 的是**輸出**，另有 `detected` 計數與 `analyzed` 索引
清單。後端讀完輸入的 `reps` 後，仍照 SP1 的規則產出輸出的 `reps`——包括「fallback 時
`reps.analyzed` 是 `[]` 但 `segments[].analyzed` 一律 `true`」那條語意。

`start_frame` / `end_frame` 是 `frame_index`；因為 §2.2 送的是全長陣列，它們同時就是
`RepWindow` 的 `start` / `end` 位置。後端把 `segments` 直接轉成 `RepWindow`，`analyzed` 為真的
那些構成 `rep_plan.analyzed`。

### 4.3 後端

- `run_detector` 增加選填的 `rep_plan`（`reps` / `analyzed` / `fallback` 三件一起）。有傳就
  **跳過 `segment_reps` 與 `select_reps`**；沒傳就走今天的 SP1 路徑——CLI、研究資料集、
  舊 `/api/analyze` 完全不受影響。
- `max_reps` 與 `rep_plan` 同時出現時**`rep_plan` 勝**（client 已套用過，後端再選一次只會少
  分析）。寫進 docstring。
- **`reps` 是不可信輸入，必須驗證**：index 遞增、`start <= end`、落在 frames 範圍內、
  區間不重疊、數量有上限。不合法 → **400，不是默默忽略**。默默忽略會讓後端拿有洞的訊號重新
  切割，產出看起來正常但錯誤的區間——正是本設計最該避免的失效方式。
- **上面那串檢查漏掉最重要的一種違規**，必須另外加：`analyzed` 為真的區間，**至少要有一幀
  landmarks 非 null**。一個指向未抽取區段的區間會通過所有排序/範圍/重疊檢查，然後在全部
  `valid=False` 的幀上被指派 phase 並評分，產出**空的 detection 清單**——也就是「拿沒量到的
  資料生出乾淨判定」，`quality.ts` 整個檔案就是為了防這件事而存在的。違反 → 400。
- `quality` **附加** `extracted_frames` / `extracted_frame_ratio`（landmarks 非 null 的幀數）。
  既有欄位的計算一行不改。

### 4.4 要誠實記錄的副作用

`valid_frame_ratio` 的**數值**會合理地掉到 ~30%，因為分母仍是全片：

- `wasMeasured`（`frontend/src/lib/quality.ts:29`）是類別式的 `> 0`，不受影響
- `chat.py:182` 的 CLEAN REP / NOT MEASURED 分支同樣是類別式的，不受影響
- `perception_to_graph.py:150` 寫進 KG 的仍是「全片有多少幀可測」，那句話依然為真
- 唯一被誤導的是 `MetricsCards` 的顯示——§5 處理

---

## 5. 未分析區段的 UI

這是第一次會有大段影片真的沒有姿態資料。骨架在片中消失、timeline 空白，在使用者眼裡就是 bug
或「這幾下沒問題」。這個 codebase 明文拒絕讓管線的限制被呈現成教練判定
（`poseExtract.ts:59-66`、`quality.ts:1-27` 都是為此而寫）。SP1 蓋了
`reps.segments[].analyzed` 就是為了這一刻。

三處，都很小：

1. **`Timeline.tsx`** — 未分析區段畫**中性**斜紋底（不是紅色、不是警告色；它不是問題，是
   「沒看」）。資料來自 `reps.segments[].analyzed`。
2. **結果頁一行說明** — 「共偵測到 5 下，分析了第 1、3、5 下」。`fallback` 非 null 時改成
   「整段分析」，不列 rep 清單。
3. **`MetricsCards.tsx:57,96`** — 分母改成 `extracted_frames`（有該欄位時），顯示
   「有效 / 已抽取」而非「有效 / 全片」。沒有該欄位時（舊分析、CLI 產出）維持今天的行為。

疊圖不動：沒有點就不畫，這是既有行為；加上 1 與 2 之後使用者知道為什麼。

**注意 `segments[].analyzed` 在 fallback 下一律是 `true`**（SP1 §5 已釘住這個語意）：該片段的
frame 確實被評分了，只是評分方式是整段而非逐 rep。UI 不能因為 `reps.analyzed` 是空的就把片段
標成「沒檢查過」。

---

## 6. 測試

**vitest**（cwd = `frontend/`，見 CLAUDE.md）

- `repSegmentation.ts` 跑完 `tests/fixtures/rep_segmentation_cases.json` 的**每一個 case**
  ——這是與 Python 的唯一硬約束
- **抽樣比對（常設，不只是 T1 的一次性檢查）**：每個 case 抽三取一、`fps/3` 再切割，斷言 rep
  數量與密集相同且邊界誤差 ≤ `REP_PADDING_FRAMES`。`flexed_static_glitch` 釘成**已知分歧**
  （密集 0 下 / 粗掃 2 下，見 §2.8），測試明確斷言它就是這樣——讓這個不對稱永遠看得見，且新增
  的 flexed 動作一啟用就會撞到它
- `avgKneeAngle` / `centeredMedian` 對已知 landmarks 的數值（T1 的自動化版本）
- padding 展開 + 重疊區間合併
- **邊界精修**：粗掃邊界刻意偏移 ±3 幀，精修後回到密集訊號的正確邊界；span 內切不出 window 時
  退回粗掃邊界並記 `refined: false`；精修結果的挑選（與粗掃區間重疊最多的 window）在 span 內有
  多個 window 時仍選對
- `frame_index` 落在 30fps 格點（粗掃與密集都是）
- 三種 fallback 各自產生「全片 frame 清單」
- payload 組裝：全長 frames、未抽取為 null、`total_frames == frames.length`

**pytest**（`.venv\Scripts\python.exe -m pytest tests/`）

- `run_detector` 給 `rep_plan` 時不呼叫 `segment_reps`，且逐 rep 指派 phase
- 非法 `reps` → 400（每一種違規各一個 case）
- 稀疏 payload 的 `quality` 欄位正確、`frame_metrics` 仍每幀一列
- 沒給 `rep_plan` 時行為與 SP1 逐位元相同
- 覆蓋率 `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

**與 CI 對齊**（`.github/workflows/ci.yml`）後才宣稱通過。

---

## 7. 交付順序（供 writing-plans 展開）

1. `frame_index` 改成 `round(t * 30)`（先修既有隱患，與其餘無相依、獨立可驗證）
2. TS 移植：`repSegmentation.ts` + `repSignal.ts`，測試先行，唯一的門檻是 fixture 全綠
   ＋ §6 的抽樣比對
3. T1 / T2 實測（**必須在步驟 2 之後**，它們比對的就是移植的產物）→ 定 `REP_PADDING_FRAMES`
4. `poseExtract.ts` 重構成 `sampleFrames` + 粗掃/密集兩個呼叫端 + 邊界精修（§2.1.1）
   → T3（量粗掃佔比與整體是否真的變快）
5. 後端：`run_detector` 的 `rep_plan` + `/api/analyze/pose` 驗證 + `quality` 附加欄位
6. UI 三處
7. 全套測試 + 覆蓋率

---

## 8. 已知風險

| 風險 | 處理 |
|---|---|
| 粗掃訊號太稀疏，切出的邊界與密集訊號差太多 | §2.1.1 的精修直接消除這一項——最終邊界來自密集訊號。粗掃誤差只需小到讓真邊界落在 padded span 內（合成訊號上 ≤ 3 幀，§2.8；T2 量真實影片）。若誤差大到 padding 吃掉大部分節省，就**降低 `COARSE_STRIDE`**（取樣更密）重量一次 |
| 粗掃在異常拒絕上比密集寬鬆，可能切出密集會拒絕的區間，而後端信任它 | §2.8 已量到 squat（extended 路徑）0 個案例，flexed 路徑 1 個。§6 的抽樣比對是常設測試；新增 flexed 動作時必須重跑。**精修不解決這一項**：一個假 rep 在精修時會切不出 window，但那與「訊號太雜」無法區分，所以只記 `refined: false`、不丟棄——丟棄會讓一個合法但雜訊大的 rep 靜靜消失。這是已知限制，不是疏漏 |
| TS 與 Python 的訊號實作漂移，且後端信任 client 而不會察覺 | T1 對拍 + `avgKneeAngle` 的數值測試。這是選擇移植同一個量而非改用 proxy 的全部理由 |
| `reps` 是使用者可竄改的輸入 | §4.3 的驗證，違規回 400 而非默默忽略 |
| N=3 一定會漏掉沒被選中的 rep 上的錯誤，且 SP2 之後那些 rep **連骨架都沒有** | §5 的三處 UI。這比 SP1 更需要處理，因為洞現在是看得見的 |
| 兩次模型載入吃掉短片上的節省 | T3 含模型載入一起量；tier 相同時共用 instance |
| 錄影模式重複勞動（即時偵測跑過一次又粗掃一次） | 已知並接受，見 §2.4。T3 的數字決定要不要接即時 landmarks |
