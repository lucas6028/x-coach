# 逐 rep 規則偵測（RS-SP1）— 設計

Status: **已實作** · Created 2026-07-26

讓規則偵測器不再把「整段影片」當成一次動作，而是先切出 reps、逐 rep 判定、再彙整。
這同時修掉一個既有的正確性 bug，並為「只抽取特定 rep 區間」（RS-SP2）鋪好前置。

既有實作見 `src/pose/pose_rule_detector.py`、`src/pose/movements/`；
擷取層見 `docs/superpowers/specs/2026-07-21-client-side-pose-capture-sp1-design.md`。

---

## 0. 大局與拆解（本 spec 只做 RS-SP1）

使用者的原始需求是：**不要對整段 skeleton 跑規則，只跑其中幾個 reps**，理由是省運算、
並避免在雜訊段落誤判。往下追之後，這件事其實橫跨兩個獨立的子系統：

| | 子專案 | 內容 | 狀態 |
|---|---|---|---|
| **RS-SP1** | 逐 rep 規則偵測（ML/backend） | 切 rep、每 rep 各自 assign phases、規則逐 rep 執行、同 fault 合併、payload 加 `reps`、`--max-reps` / API 參數 | **本 spec** |
| **RS-SP2** | 只抽取選中 rep 區間（capture） | 錄影時留下 live landmarks 當粗掃訊號 → 切 rep → 前端/後端只對選中區間跑姿態抽取 | 後續，另立 spec |

**RS-SP1 是 RS-SP2 的前置**：SP2 需要切割器已經存在，也需要 payload 能容忍「中間有洞」的
frame 序列。而 RS-SP1 單獨出貨就已經修好多 rep 判錯的 bug，且對**所有現有路徑**
（上傳、library、CLI、研究資料集）立即生效，不需要前端配合。

### 成本的真相（本 spec 誠實記錄）

使用者提出的省運算動機，在 RS-SP1 這一層**幾乎不成立**：`compute_raw` + `centered_median`
是很便宜的 numpy，真正貴的是
`frontend/src/lib/poseExtract.ts:136-143`（逐幀 seek + `detectForVideo`）與
`src/pose/process_videos.py:58-59`（逐幀 RTMPose 推論）——**兩者都在 RS-SP2 的範圍**。
RS-SP1 唯一實際的省時來自合併重複 fault 後減少的 `retrieve_contexts_for_detections`
（KG + vector DB）次數。

所以 RS-SP1 的價值定位是**正確性 + 去雜訊**，不是省時。省時在 RS-SP2 兌現。

---

## 1. 要修的 bug

`assign_phases` 是照「整段影片 = 一個 rep」寫的。三份都是同一個形狀
（`src/pose/pose_rule_detector.py:175`、`src/pose/movements/pushup.py:648`、
`src/pose/movements/overhead_press.py:226`）：

- `bottom_index` 取**全域** argmin → 5 個 rep 的影片裡，第 2 rep 之後的下降段全被標成 `ascent`
- bottom 門檻用**全域**百分位 → 較淺的 rep 永遠拿不到 `bottom` phase
- setup / lockout 直接切前 15% / 後 15% → 中間所有 rep 的起始站姿都被當成動作中

因此多 rep 影片目前的 phase 標記是錯的，而所有規則都 gate 在 phase 上。切 rep 之後這些
**不需要改任何 movement 檔就自動修好**（同一個函式套在單一 rep 上，語意就正確了）。

---

## 2. 範圍

**做**

- 共用 rep 切割器（`src/pose/rep_segmentation.py`），純函式、無 I/O、對 1-D 訊號操作
- `MovementDetector` 增加 5 個 rep 相關欄位（`rep_signal` / `rep_polarity` / `rectify` /
  `rep_start` / `min_rep_seconds`）——欄位一次開齊，即使目前只有 3 個 detector 用得到，
  理由見 §3.4
- `run_detector` 改為：全域算 raw → 切 rep → **逐 rep** assign phases → **逐 rep 切片**跑規則
- rep 選取（預設 N=3，首/中/尾）
- 同 `fault_id` 合併成一筆，帶發生的 rep
- payload **只增不改**地加上 `reps` 區塊
- `--max-reps` CLI 旗標、`config.DEFAULT_MAX_REPS`、`/api/analyze` 與 `/api/analyze/pose`
  的選填 `max_reps`
- 給 RS-SP2 的 TS 移植用的共用 fixture（見 §7）

**不做**

- 任何前端 UI 改動（不加「分析幾下」的選項；N 走 API 參數與後端預設）
- 任何姿態抽取層的改動（那是 RS-SP2）
- 改 `quality` 既有欄位的分母、或減少 `frame_metrics` 的列數

---

## 3. Rep 切割器

`src/pose/rep_segmentation.py`，不依賴 scipy（維持 repo 的 stdlib + numpy 風格）。

### 3.1 介面

```python
@dataclass(frozen=True)
class RepWindow:
    index: int          # 1-based，使用者看到的「第幾下」
    start: int          # frame 序列的 index（非 frame_index），含
    end: int            # 含
    partial: bool       # 影片開頭已在底部，或結尾還沒回到伸展

def segment_reps(
    signal: Sequence[float],
    *,
    fps: float,
    polarity: str = "min",      # "min" = 用力點是最小值（蹲、伏地挺身）
                                # "max" = 用力點是最大值（推舉鎖定）
    rectify: bool = False,      # True = 先取絕對值，把雙極訊號變成單極（軀幹旋轉）
    rep_start: str = "extended",# rep 邊界放在哪一端："extended"（站姿起算，多數動作）
                                # 或 "flexed"（從地板起算，硬舉）
    min_rep_seconds: float = 0.4,
) -> list[RepWindow]:
```

`rectify` / `rep_start` / `min_rep_seconds` 都是為了 §3.4 才存在的；RS-SP1 的三個動作全部
走預設值。它們現在就進介面，是為了讓 SP2 逐動作實作時**填參數而不是改切割器**。

### 3.2 演算法（遲滯門檻）

1. 取有限值樣本；不足 `2 * min_rep_frames` → 回傳 `[]`
2. `rectify` 為真時先取絕對值；`polarity == "max"` 時把訊號取負。
   之後一律以「用力點 = 低值」處理
3. `lo = percentile(5)`、`hi = percentile(95)`；`span = hi - lo`；`span <= 0`（訊號完全平坦，
   例如靜止片段）→ 回傳 `[]`。**這裡刻意沒有「動態範圍下限」這種門檻**——曾經有過
   （`MIN_RANGE_TO_NOISE`，要求 `span` 至少是雜訊估計值的幾倍），但四次不同做法的雜訊估計
   （步幅中位數、步幅低百分位、全片段步幅比例、活動區段內步幅比例）在實測中都會誤傷合法但
   幅度小的真實訓練訊號——暫停的一下、rep 間的休息、慢節奏、鏡頭抖動——因為這四種做法量的
   都是「一段本來就含有靜止 frame 的區域」的**步幅分布**。細節與拒絕理由見
   `src/pose/rep_segmentation.py:34-56` 的模組註解。
4. `enter = lo + 0.35 * (hi - lo)`（進入底部）、`exit = lo + 0.65 * (hi - lo)`（回到伸展）
5. 一個 rep = 訊號跌破 `enter` 之後、再回到 `exit` 之上的整個區間；
   起點是跌破 `enter` 前最後一次在 `exit` 之上的位置，終點是之後第一次回到 `exit` 之上的位置。
   `rep_start == "flexed"` 時邊界改放在**谷底**：一個 rep = 從一次跌破 `enter` 的最低點，
   到下一次跌破 `enter` 的最低點（硬舉：地板 → 鎖定 → 地板）
6. 長度 `< max(min_frames, min_rep_seconds * fps)` 的區間視為雜訊丟棄。**這是切割器唯一的
   雜訊/異常排除機制**：拒絕完全由**激烈期間的長度（duration）**決定，不比較訊號的動態範圍。
   `_climb_backward` / `_climb_forward` 遇到平台（連續相等值）就停止攀爬，讓一個 window 的長度
   確實等於該次激烈期間本身的長度，這個 duration 測試才有意義（見同一段模組註解）。
7. 開頭第一個樣本就在 `enter` 以下 → 該 rep `partial=True`；
   結尾跌破 `enter` 後直到片尾都沒回到 `exit` → 該 rep `partial=True`

門檻 0.35 / 0.65 是動態範圍的比例，不是絕對角度，所以對動作、體型、鏡頭距離都不敏感。

### 3.3 RS-SP1 實作的三個動作

| 動作 | `rep_signal` | `rep_polarity` |
|---|---|---|
| Squat | `avg_knee_angle` | `min` |
| Push-up | `min_elbow_angle` | `min` |
| Overhead Press | `avg_elbow_angle` | `max` |

三個 key 都已經在各自的 `METRIC_KEYS` 裡，不需要新增指標，其餘參數全走預設值。

### 3.4 對 16 個動作的適用性盤點

依 `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.zh-TW.md` 逐一檢視。
**演算法的形狀（1-D excursion + 遲滯 + 相對動態範圍的門檻）對 16 個動作都成立**，
差別在需要哪些旋鈕。

**乾淨的單向 excursion（11 個，全走預設值）**

| 動作 | 訊號 | polarity |
|---|---|---|
| 深蹲 Squat | 平均膝角 | min |
| 弓步 Lunge | 前腳膝角（= 兩腳膝角取 min） | min |
| 伏地挺身 Push-up | 較屈曲側肘角 | min |
| 過頭推舉 Overhead Press | 平均肘角 | max |
| 划船 Row | 肘角 | min |
| 二頭彎舉 Bicep Curl | 肘角 | min |
| 仰臥起坐 Sit-up | 軀幹相對水平角 | max |
| 臀橋 Shoulder Bridge | 髖高 / 髖角 | max |
| 手臂外展 Arm Abduction | 上臂仰角 | max |
| 彈力帶擴胸 Band Pull Apart | 雙手間距 / 肩寬 | max |
| 腿部外展 Leg Abduction | 踝離中線距離 | max |

**手臂 VW** 也屬於這類：V（手臂高舉）→ 下拉成 W → 等長維持 → 回 V，就是標準 excursion
（`arm_elevation_angle`，polarity `min`）。W 位置的等長停頓不影響遲滯判定。

**需要旋鈕的（4 個）**

| 動作 | 問題 | 旋鈕 |
|---|---|---|
| 軀幹旋轉 Torso Twist | spec 明訂「每次單側擺動 = 一下」，訊號**雙極**（中心 → A 側 → 中心 → B 側）。單一 `enter` 門檻只抓得到一個方向 | `rectify=True` + `polarity="max"`，對有號旋轉角取絕對值，A/B 兩側各成一次 excursion |
| 硬舉 Deadlift | rep 從**地板**起算（setup → lift-off → knee-passing → lockout），不是從站姿。邊界放在伸展端會切成「鎖定→地板→鎖定」，`assign_phases` 的前 15% setup 於是落在上一下的鎖定上 | `rep_start="flexed"` |
| 開合跳 Jumping Jacks | 節奏 ~1–2 Hz；且規則釘在「落地那一幀」的單幀事件而非 phase 區間 | `min_rep_seconds` 調小；SP2 的 padding 對它特別關鍵（落地就在區間邊界上） |
| 高抬腿 High Knee | 交替單腳、每步一下，節奏可到 ~3 Hz → 30fps 下一下僅約 10 幀，預設 `min_rep_seconds=0.4`（12 幀）會把**真的 rep 當雜訊丟掉** | 見下方：暫不開啟 |

**雙側 / 交替動作**（高抬腿、交替弓步、交替彎舉）的訊號必須取**兩側的 min/max**，否則只會
抓到一隻腳/手。這是訊號定義問題，不是演算法問題。

**高抬腿：暫不開啟切割，走整段 fallback。**
~10 幀一下的訊號在 MediaPipe 的抖動下能否穩定切割，沒有證據；而**切錯比不切更糟**——切錯
會讓規則在錯的 phase 上跑，產生看起來正常但實際錯誤的判定。它同時是本專案最弱的動作
（general-tier RAG 唯一找不到乾淨來源的一個）。作法：`rep_signal=None` 表示「此動作未啟用
切割」，直接走 §4.2 的 fallback。等有標註資料驗證過再開。

**這一節的性質必須講清楚**：除了深蹲 / 伏地挺身 / 推舉，其餘 13 個動作的 detector
**目前並不存在**。上表是從規則 spec 的動作描述推導出來的**介面設計推論，不是驗證過的事實**。
每個動作實作時仍必須拿真實影片確認切割結果，並且可以推翻上表。RS-SP1 的責任只到
「介面容納得下這些情況」為止。

---

## 4. `run_detector` 的新流程

`src/pose/movements/base.py`：

```
raw = compute_raw(frames, fps)                    # 全部 frame，不變
smoothed = centered_median(..., window=5)         # 全域平滑，不變
reps = segment_reps(smoothed[rep_signal], ...)    # 新
selected = select_reps(reps, max_reps)            # 新
phases = ["rest"] * n                             # 新：預設不屬於任何 rep
for rep in reps:                                  # 每個 rep 各自 assign
    phases[rep.start : rep.end + 1] = detector.assign_phases(raw[rep.start : rep.end + 1])
core = [CoreFrame(...) for ...]                   # 不變
for rep in selected:                              # 規則對「切片」執行
    for rule in detector.rules:
        detections += tag_rep(rule(core[rep.start : rep.end + 1], ctx), rep.index)
detections = merge_by_fault(detections)           # 新
```

四個關鍵決定：

1. **平滑仍在全域做**。切片後才平滑會弄壞每個 rep 的邊界。
   **這是 RS-SP1 專屬的選擇，不是要 RS-SP2 保留的約束**：SP1 全域平滑是因為所有 frame 都
   存在；SP2 只抽取選中區間，區間外的資料根本不存在，因此必然改成「對 padding 過的區間
   逐段平滑」——padding 正是讓那件事安全的原因。SP2 的實作者不該把全域平滑當成前提。
2. **phases 逐 rep 指派，不是餵較少的 frame 給全域函式**。三個 movement 檔一行都不用改。
3. **規則對切片執行，不是把 rep 條件 AND 進現有 mask**。後者會讓 `contiguous_true_segments`
   把 rep 2 尾端與 rep 3 開頭的錯誤黏成一段跨 rep 的 detection。
   `CoreFrame` 帶的是**絕對** `frame_index` / `time`，所以切片不影響輸出的時間戳。
4. **rep 之外的 frame phase 是 `"rest"`**。既有規則都 gate 在各自的 active phase 集合上，
   所以走位、架槓、休息、鏡頭前調整的 frame 自動不被評分——這是「避免雜訊誤判」的主要來源。
   未被選中的 rep，其 frame 仍然拿到正確的 per-rep phase（寫進 `frame_metrics` 供前端用），
   只是不跑規則。

### 4.1 選取

```python
def select_reps(reps: list[RepWindow], max_reps: int | None) -> list[RepWindow]
```

- `max_reps` 為 `None` 或 `0` → 全部
- 有完整 rep 時，`partial` 的不入選
- `len(candidates) <= max_reps` → 全取
- 否則 `np.linspace(0, n - 1, max_reps)` 取整去重 → N=3、n=5 時得 `[0, 2, 4]`，即首 / 中 / 尾

首/中/尾而非「中間三個」：首 rep 抓熱身不足、中間 rep 代表穩定狀態、尾 rep 抓疲勞走樣。
只看中間會系統性地藏起最該提醒的錯誤。

### 4.2 Fallback（先接這條，再接正常路徑）

`segment_reps` 回傳 `[]`（靜止、姿態太爛），或只有 partial rep（library 那種頭尾被切掉的
單 rep 短片）→ **退回今天的整段行為**：對全部 frame 呼叫一次 `assign_phases`、規則跑全段，
並在 payload 記 `reps.fallback`。

絕對不可以回傳空的 detections：把切割失敗呈現成「沒發現任何問題」是這個 codebase 明確
拒絕的失敗模式（見 `poseExtract.ts:59-66` 的同型註解）。

| 情況 | `fallback` 值 |
|---|---|
| 切不出任何 rep | `"no_reps_detected"` |
| 只切出 partial rep | `"only_partial_reps"` |
| 該動作未啟用切割（`rep_signal=None`，見 §3.4 高抬腿） | `"segmentation_disabled"` |
| 正常 | `null` |

### 4.3 合併

同 `fault_id` 只留一筆，取 severity 最高那次當代表——severity、時間段、`evidence`、
`peak_frame` 全部來自那一次，保持內部一致。

`PoseRuleDetection` 新增三個有預設值的欄位（附加式，`asdict` 自動帶進 JSON）：

```python
rep_index: int = 0                      # 代表那次發生在第幾個 rep（fallback 時為 0）
occurred_reps: tuple[int, ...] = ()     # 所有觸發的 rep，1-based
rep_count: int = 0                      # len(occurred_reps)
```

這樣前端的 FaultCard 清單不會被同名錯誤灌滿，chat 也能講「你這三下裡有兩下膝蓋內扣」。

---

## 5. Payload（只增不改）

新增 top-level `reps`：

```jsonc
"reps": {
  "detected": 5,
  "analyzed": [1, 3, 5],
  "max_reps": 3,
  "fallback": null,
  "segments": [
    {"index": 1, "start_frame": 12, "end_frame": 58,
     "start_time": 0.4, "end_time": 1.93,
     "analyzed": true, "partial": false}
  ]
}
```

`reps.analyzed` 是**逐 rep**被評分的 rep index 清單；`fallback` 非 `null` 時一律是 `[]`——不是
因為那些片段沒被檢查，而是因為 fallback 時整段影片被當一個單位評分，不是逐 rep。

`segments[].analyzed` 回答的是不同的問題：「這個片段本身有沒有被檢查過」。正常路徑下兩者一致
（只有被選中的 rep 才 `analyzed=true`）；但在任何 fallback 下，`segments[].analyzed` 一律是
`true`——該片段的 frame 確實包含在整段評分裡，只是評分方式是整段而非逐 rep。`only_partial_reps`
是最容易看反的例子：clip 只有一個（partial）rep，`reps.analyzed=[]`，但這個 rep 的內容真的被
評分了（整段當一個單位），所以它的 `segments[0].analyzed` 必須是 `true`——UI 不能因為
`reps.analyzed` 是空的就把這個片段當成「沒檢查過」，那正是 `segments` 這個欄位存在的理由：
不能讓介面暗示一段實際被評過分的影片是「沒人看過」。

`quality` 增加 `analyzed_frames` 與 `analyzed_frame_ratio`；**既有欄位的分母不動**。
`frame_metrics` 仍然每個 frame 一列（rep 外的 `phase` 是 `"rest"`）。

`"rest"` 是新的 phase 值，已確認不會打破既有消費者：`frontend/src/api.ts:33` 把 phase 型別
定成開放的 `string`（不是 union）；`i18n.tsx:1370-1375` 的 `dataLabel` 對沒有對應 key 的值
退回 `titleCase(raw)`，所以最壞情況是顯示英文 "Rest" 而非崩潰（`lockout` 今天就已經是這個
狀態）；`src/knowledge/perception_to_graph.py` 完全不讀 phase。
另外 detection 的 `phase` 是該區段的 dominant phase，而規則只在 active phase 上觸發，
所以 detection 永遠不會帶 `"rest"`——UI 上看得到的那個欄位不受影響。

理由：`quality` 與 `frame_metrics` 是相容性介面，消費者有
`backend/app/services/analysis.py`、frontend、`src/knowledge/perception_to_graph.py`。
附加安全，改分母或減列數不安全。

`segments` 讓前端**之後**能標示「哪幾段實際被分析過」。本 spec 不動前端，但這是必要的資料
——有整段影片沒被檢查時，UI 不該讓人以為那些段落是乾淨的。

---

## 6. 設定介面

一路往下傳，預設值只有一個地方定義：

```
scripts/pose/run_pose_rule_detection.py  --max-reps N   (0 或 all = 全部)
  → detect_pose_rules_from_json(max_reps=...)
  → detect_pose_rules_from_payload(max_reps=...)
  → run_detector(..., max_reps=3)

backend/app/config.py  DEFAULT_MAX_REPS = 3
  → /api/analyze        選填 max_reps（驗證 0–20，0 = 全部）
  → /api/analyze/pose   同上
```

前端不改，也不送 `max_reps`，所以 Web 使用者拿到後端預設的 3。

---

## 7. 對 RS-SP2 的介面承諾

RS-SP2 要在前端用 TypeScript 重寫一份切割器（使用者已決定 TS/Python 各一份，而非透過
API 往返）。為了讓兩份不會漂移，RS-SP1 必須：

1. `segment_reps` 是**純函式**：只吃 `Sequence[float]` + 純量參數，不碰檔案、不碰時間、
   不依賴 numpy 之外的東西
2. 所有門檻（0.35 / 0.65 / 0.4 秒 / 百分位 5 與 95）是**具名模組常數**，不是行內字面值
3. 產出 `tests/fixtures/rep_segmentation_cases.json`：一組合成訊號 + 期望區間。
   Python 測試讀它、RS-SP2 的 vitest 也讀同一個檔案。任一邊改門檻，兩邊測試同時紅。

### 7.1 未解決：誰算那條訊號（留給 RS-SP2 的 spec）

fixture 只釘住「**給定一條 1-D 訊號 → 得到哪些區間**」，這是切割器本身。但 RS-SP2 的 TS
那一側手上是 **live landmarks**，不是 `avg_knee_angle`——後者是 `compute_raw` 算出來的，
而 `compute_raw` 在 Python。所以 SP2 還得決定：

- (a) 在 TS 重寫該動作的 rep 訊號計算（深蹲只需要膝角，很小；但每加一個動作就多一份），或
- (b) 把 `segment_reps` 的輸入定義成瀏覽器能直接算的東西（例如髖-膝 y 差，不需要三點夾角）

**這是 RS-SP2 要回答的開放問題，RS-SP1 不預先決定**。RS-SP1 只保證切割器本身是可移植的
純函式，且 `rep_signal` 是 `MovementDetector` 上一個可替換的欄位——換訊號不需要改切割器。

---

## 8. 測試

新增 `tests/test_rep_segmentation.py`（純函式，好覆蓋，直接衝 95% gate）：

- 合成 N-rep 正弦波 → 切出 N 個
- 完全靜止的平台（`span == 0`，無動態範圍）→ 0 個。注意這只釘住 `span <= 0` 這個退化情況，
  不是「動態範圍下限」——見 §3.2 步驟 3，切割器刻意沒有那種門檻
- 單一 rep → 1 個
- 尾端截斷 → 最後一個 `partial=True`
- 開頭截斷 → 第一個 `partial=True`
- `polarity="max"`（OHP 形狀）→ 與 `"min"` 鏡像後結果相同
- `rectify=True`（軀幹旋轉形狀：中心→A→中心→B→中心）→ 切出 2 個 rep，不是 1 個
- `rep_start="flexed"`（硬舉形狀）→ 邊界落在谷底，切出的區間數與 `"extended"` 相同但相位差半下
- `rep_signal=None` → 回傳 `[]` 且 fallback 記 `"segmentation_disabled"`
- 過短的抖動不被算成 rep
- `min_rep_seconds` 調小後，快節奏訊號（開合跳形狀）能切出全部 rep
- `select_reps`：n=5/max=3 → `[1, 3, 5]`；n=2/max=3 → 全取；max=0 → 全取；partial 被排除
- 合併語意：同 fault 兩個 rep → 一筆、`rep_count=2`、`occurred_reps=(1,3)`、
  代表值來自 severity 較高那次
- fixture 檔案的每個 case（同時被 RS-SP2 的 TS 測試消費）

現有測試：**預期全部通過，不需修改**。查證後的實情（此處更正本 spec 早期草稿的相反預測）：

- `tests/test_pose_rule_detector.py` 走的是 `compute_frame_metrics` + `detect_rule_segments`
  這條**legacy squat 參考路徑**，本次不動它。
- 所有既有 fixture 都是**靜止**的（同一個 frame 重複 12–14 次），動態範圍為 0 →
  `segment_reps` 回傳 `[]` → 走 §4.2 fallback → 與今天逐位元相同。
  `tests/test_movement_registry.py::test_squat_via_registry_matches_legacy`
  因此仍然成立。

真正的風險因此**反過來**：不是既有測試會爆，而是**既有測試完全不會走到新路徑**。
所以必須新增一個真正動態的多 rep fixture（見下），否則這次改動等於沒有測試覆蓋。

**多 rep 迴歸測試（證明 bug 真的修好）**：以 `tests/test_pose_rule_detector.py` 的 `frame()`
產生 3 個 rep（`hip_y` 依餘弦在 0.45 ↔ 0.92 之間擺動），然後斷言：

- `segment_reps` 切出 3 個 rep
- legacy 路徑把第 2、3 個 rep 的下降段標成 `ascent`（釘住 bug 的存在）
- 新路徑三個 rep 各自都有 `descent` / `bottom` / `ascent`（釘住 bug 已修）

若這個測試在改動前就通過，表示 fixture 不夠動態，要先修 fixture 而不是接受它。

跑法（見 CLAUDE.md）：
`.venv\Scripts\python.exe -m pytest tests/`、
`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

---

## 9. 已知風險

| 風險 | 處理 |
|---|---|
| 遲滯門檻在動作很淺、或 pose 抖動大的影片上切出過多/過少 rep | fixture 涵蓋這些形狀；切不出來時走 §4.2 fallback 而非產生空結果 |
| 既有 Labeled 資料集的短片被切太緊，頭尾都是 partial | `"only_partial_reps"` fallback → 整段當一個 rep，行為與今天一致 |
| `rule_heel_rise` 之類依賴 setup 基線的規則，per-rep 的 setup 是站姿而非真正的起始 | 對第 2 個 rep 之後其實更合理（腳跟本來就該在地上）；在該規則的測試裡明確釘住 |
| N=3 一定會漏掉沒被選中的 rep 上的錯誤 | 使用者已知並接受；`reps.segments` 記錄哪些被分析過，供 UI 之後標示 |
| §3.4 的 13 個動作是**推論**，實作時可能發現訊號選錯或旋鈕不夠 | 旋鈕是 `MovementDetector` 上的資料而非切割器的分支；推翻某一列只需改該動作的欄位。若出現連旋鈕都表達不了的形狀，該動作先走 `rep_signal=None` fallback，不要為它在切割器裡開特例 |
