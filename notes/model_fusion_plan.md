# 模型融合：能不能合併已測模型的優點？——論文規劃與前置判斷

**問題（使用者提出）**：把已試驗過的姿態估計模型（NLF / HMR2.0 / Multi-HMR / MeTRAbs /
RTMPose / MediaPipe / HRNet）各自的優點結合，降低關節誤差，進而提高動作錯誤偵測準確率。

本文件不是實驗結果，是**動手前的判斷**：哪些融合路線已經被我們自己的資料否決、哪些還活著、
先跑哪一個、以及成本在哪裡。所有數字都來自 `notes/` 既有實驗，逐條可回溯。

> **⚠ 已被後續實驗超越——先讀 `notes/model_fusion_gate_results.md`。**
> §3 規劃的閘門實驗 0.5 與 G **已經跑完**，結果推翻了本文件 §2 的路線排序：
> 平均法與逐幀路由**雙雙死亡**（後者連 shuffled 對照都通不過），
> §2 列為存活的 H（2D×3D 混合骨架）被更上游的結果取消，
> §6 的甲/乙選擇判定為**乙（第四次否證）**。
> 本文件保留作為推理紀錄；§0 的張力分析與 §4 的 parity 債務仍然有效。

---

## 0. 必須先講清楚的張力：這條因果鏈中間是斷的

「降低關節誤差 → 提高錯誤偵測準確率」看似自明，但我們有**三組自己量到的數字反對它**。
任何融合論文如果不先處理這三點，第一個審稿人就會用我們自己的 notes 打我們。

**(a) MPJPE 排序 ≠ 線索誤差排序。** HMR2.0 的 MPJPE 比 NLF 差 24 mm（101.9 vs 77.7），
但旋轉不變的膝角只差 0.8°（7.88 vs 7.09），髖角**反而更好**（9.26 vs 10.42）。
`fit3d_model_comparison_summary.md` 已有一整節說明為何 raw MPJPE 不是這裡的度量
（差距來自體型 ~11% + crop frame，不是深度失敗）。**以降低 MPJPE 為目標的融合，優化的是
一個不預測判決的量。**

**(b) 線索誤差贏了，判決卻輸了——三個動作一致。** 這是現成表裡最尖銳的一個反例：

| 動作 | cue 誤差 MeTRAbs / NLF | verdict-flip MeTRAbs / NLF |
|---|---|---|
| squat knee | **6.37** / 7.09 | 13% / **11%** |
| deadlift knee | **7.43** / 7.51 | 17% / **10%** |
| thruster knee | **4.59** / 5.91 | 13% / **9%** |

MeTRAbs 在旋轉不變膝角上**三戰全勝** NLF，在判決翻轉上**三戰全敗**。
若成立，這直接否定「先降低關節/線索誤差、判決自然會好」的整條規劃。
**但這也可能是取樣假象**：verdict 是把一個 rep 視窗縮成極值（nanmin/nanmax），
MeTRAbs 是每 15 幀、NLF 是每幀，樣本少的極值系統性偏保守——
`fit3d_sparse_depth_summary.md` 自己已標註這個 caveat。**這是實驗 0.5，見 §3。**

**(c) 下游訓練式分類器會吸收 2D 殘差，讓深度增益歸零。**
Fitness-AQA 同偵測器開關深度：`nlf_3d − nlf_2d = +0.003`（p=0.813），
偵測器品質才是驅動力（`nlf_2d − mediapipe_2d = +0.122`, p<0.001）；
error-overlap 證明是真冗餘不是天花板（3D 修好 18 個、弄壞 18 個，r=0.943）。
REHAB24 的 early concat fuse 更直接：0.683 **低於** skeleton-only 的 0.723。

**結論**：融合若要有意義，端點必須是**規則式判決**（verdict-flip / false-alarm），
不能是 MPJPE，也不能是訓練式分類器的 balanced accuracy。見 §5。

---

## 1. 已被我們自己的資料否決的三條路線（不要做）

**F1 — 座標層平均集成（把多個 3D 模型的關節取平均/中位數以降 MPJPE）。**
三個獨立理由：(i) 骨架慣例偏移最大到 **176 mm**（thorax），跨慣例平均是加 bias 不是消 bias；
(ii) HMR2.0 的 pa_mpjpe 幾乎不比 mpjpe 好（103.7 vs 101.9），代表殘差是 non-rigid/scale，
不是單一剛性錯位，平均消不掉；(iii) 目標量本身不預測判決（§0a）。

**F2 — 用 NLF 的 `unc` 做不確定度加權融合。**
已測且已否決（`fit3d_uncertainty_summary.md`）：通道本身是真的
（within-joint Spearman 平均 +0.403，16/16 關節為正），但**姿態自己預測自己的誤差更準**
（LOSO MAE 14.16 vs `unc` 的 19.26 mm），把 `unc` 加到 pose 上只買到 **0.01–0.17 mm**；
拿來選最佳視角是 **22.9%，低於 25% 的隨機**。

**F3 — Early feature concat。**
REHAB24 已測：skeleton+VideoMAE fuse test bal_acc 0.683 < skeleton-only 0.723，
train 卻衝到 0.903（落差 0.22）——典型強特徵被弱特徵稀釋。該 note 自己的建議是
late fusion / gating 取代 early concat。

---

## 2. 還活著的路線（依「該先做」排序）

### G — 互補性閘門（必做，且免費）

**在建任何融合之前，先測模型誤差到底相不相關。** 若各模型在同一批幀上一起失敗，
沒有任何融合能幫上忙，整個規劃當場結束——而那本身是可發表的結果。

表面證據顯示 action×cue 層級**贏家會換手**：MeTRAbs 三動作膝角最佳、
HMR2.0 deadlift 髖角最佳（7.78）、Multi-HMR thruster 髖角最佳（8.34）。
但那是 parity 未對齊下的排名，可能是校正差異而非模型差異（§4）。
需要的是**逐幀**的互補性，因為路由器需要的是逐幀決策。

要算的兩件事，皆用現成 npz：
1. 模型間 cue 誤差的相關矩陣（在旋轉不變的 knee\*/hip\* 上算，**不是座標**）。
2. oracle per-frame best 與 oracle per-sequence best，相對最佳單一模型的 headroom。

**判讀**：oracle ≈ best single → 所有融合死亡，改寫成第四個 refutation（§6）。
有 headroom → 知道該建哪一種融合。**兩種結果都是結果。**

### R1 — 可部署的路由器（最安全的正面貢獻）

「按錯誤類型路由——深度/屈曲用 3D、額狀面 valgus 用校正過的 2D」這條結論**已經寫過了**
（`fit3d_decision_fidelity_summary.md` §跨線索 needs-3D map），單獨拿去投稿會被說是重複。

真正的缺口在別處：**所有路由數字都建立在 oracle per-view calibration 上**，
而我們自己的 caveat 寫得很清楚——「oracle debiasing is an upper bound, **not a deployable
calibration** — it uses the GT to compute each camera's offset」。
真實系統沒有 per-view GT。**貢獻是那個沒有 GT 也能跑的路由器，不是那條規則。**

判準來自 Fitness-AQA 的 refinement：關鍵不是矢狀/額狀，而是
**該線索的軸在主導視角下有多少比例投影進相機深度軸**——這個量可以從估計出的姿態自己算，
不需要 GT。（該 note 也已經自我否證過一個 just-so story：squat 與 OHP 的
mediolateral 深度比例其實相當（~0.65），所以視角無法解釋跨動作的不複製。這個限制要繼承。）

**零參數對照（我們自己的系列規定的）**：現有手寫的 `view_type` gate。
學出來的路由器打不贏它，就照實報告。

### B — 跨模型分歧作為不確定度 → 選擇性棄權

`fit3d_uncertainty_summary.md` 的「What this does NOT establish」**自己點名**了唯一未測的替代：
「A different estimator, or an **ensemble / test-time-augmentation variance**, could be less
redundant with its own point estimate.」兩個架構迥異的 3D 模型在同一線索上的分歧，
是一個免費的信心估計，且**不是**單一模型自我報告的那種。

- 必須在 **convention-invariant** 的量（knee\*/hip\* 角）上算，否則量到的是骨架慣例不是難度。
- 必須打敗殺死 blind spot C 的**同一個 51-feature pose predictor**，同樣 LOSO。打不贏就同型死亡。
- 產品故事是全部路線裡最強的：「這支影片我判不了深度」——
  給教練 App 一條 risk–coverage 曲線。**壞建議比不給建議更糟**，棄權在這個領域是真的有價值。

### H — 混合骨架（最窄，但預測最銳利）

RTMPose 的 xy + 3D 模型的 z。這是我們的分解唯一支持的座標層融合：
valgus 是 detector-limited（2D 較好，3D 反而更差：verdict-flip 31% vs 15%），
深度是 projection-limited（3D 較好），兩者贏在**不同的軸**。

但要誠實：因為 in-plane 精度**不 gate** 深度線索（完美偵測器 ≈ 真實偵測器，
detector term −0.70/−0.28/+0.44°），而 2D 自己已經贏 valgus，
所以 H 在大多數線索上會塌回 R1（路由，只是低一層）。
**H 只在需要「兩個軸同時來自不同來源」的線索上才是新的，而那樣的候選只有一個：
`knee_angle`（混合垂直軸與深度軸）。**
銳利預測：hybrid 在 knee_angle 上**同時**打敗兩個分支。打不贏就丟掉 H。

---

## 3. 實驗順序與成本

| # | 實驗 | 需要新推論？ | 成本 | 決定什麼 |
|---|---|---|---|---|
| **0.5** | 把 NLF/HMR2 遮罩到每 15 幀，重算 verdict-flip | 否 | ~1 小時 | §0b 的三連反轉是真發現還是取樣假象 |
| **G** | 逐幀互補性 + oracle headroom | 否 | 一個下午 | **正面論文 vs 第四次否證** |
| R1 | 可部署路由器 + `view_type` 零參數對照 | 否 | 中 | 主要貢獻是否成立 |
| B | ensemble 分歧 vs pose predictor（LOSO）+ 棄權曲線 | 否 | 中 | 第二貢獻 |
| H | hybrid 骨架，只測 knee_angle | 否 | 小 | 留或丟 |
| P | **parity re-extraction**（見 §4） | **是，Kaggle GPU** | **大** | 能否做跨模型深度排名 |

**已查證的前置條件（今天做的）**：6 個模型各 96 檔全覆蓋；陣列都存成完整幀長並以 NaN 補；
全部從 frame 0 起算，stride 為 nlf 1 / hmr2 1 / multihmr 6 / metrabs 15 / mediapipe 15 /
rtmpose 15 → **共同幀為每 30 幀，約 52 幀/影片 × 96 = 約 5,000 幀現成可用**。
（mediapipe/rtmpose 陣列長 1565、其餘 1563，92/96 支影片有此差異；因為都從頭對齊，
只影響尾端 2 幀，逐幀融合安全。）
**所以 0.5 與 G 完全不需要新推論。**

---

## 4. 融合會把各 note 各自隔離的 parity caveat 全部堆進同一個主張

這是本規劃最大的技術債。單獨看每份 note 都誠實地標註了自己的 caveat，但一旦融合，
它們全部落進同一句話裡：

- **MeTRAbs 拿到 Fit3D 真實 per-camera intrinsics；NLF 假設 FOV≈55；
  Multi-HMR 假設 60° FOV 且用非原生的白色 square-pad preprocessing。**
- HMR2.0 在 **crop frame** 回歸朝向。
- 稀疏模型每 15 幀、稠密模型每幀。
- 骨架慣例偏移最大 176 mm。

所以「MeTRAbs 膝角比 NLF 好 0.7°」很可能是**校正 parity 差異，不是模型品質**，
而建立在這個排名上的融合增益，會被歸因到 parity 而不是方法。

**好消息（今天查證的結果）**：旋轉不變的 cue 角度**與 intrinsics 無關**
（`fit3d_model_comparison_summary.md` 已言明 knee/hip 旋轉不變角不依賴 FOV 假設）。
所以 G / B / H 的主體**不需要重抽**。
只有要宣稱「模型 X 的深度比 Y 好」或使用 ez/exy、絕對 mm 時，才需要實驗 P
（NLF 用真 intrinsics 重跑、Multi-HMR 用原生 preprocessing 重跑）。

**這把時程切成兩半**：只做 rotation-invariant 主張 → 兩週；要做跨模型深度排名 → 兩個月。
**建議把 P 推到第二半，或整篇避開跨模型深度排名。**

---

## 5. 端點選擇（不要用 REHAB24 LOSO 當主要端點）

我們自己的 note 寫著：`±0.08 的折間 std ＞ 想衝的進步幅度`。
融合能產生的效果量解析不出來，只會得到一個**無法判讀**的 null。

而三個下游測試的 pattern 本身就值得寫成一個 scope statement：

| 下游形態 | 深度的效果 |
|---|---|
| 規則式固定閾值（Fit3D verdict-flip） | 巨大：76% → 7% |
| 訓練式分類器、in-distribution 標籤（Fitness-AQA） | 歸零：+0.003 (p=0.813) |
| 訓練式分類器、n=9（REHAB24） | 未定：+0.035 |

**深度對規則式判決有用，對能吸收 2D 殘差的訓練式分類器沒用。**
x-coach 部署的正是規則式判決，所以主要端點是 **verdict-flip / false-alarm rate**，
而那兩個分類器 null 從「矛盾」變成「範圍界定」。

---

## 6. 兩種論文骨架——由 G 決定，不是由偏好決定

**(甲) 正面：《合併估計器何時能、何時不能改善教練判決》**
判準（誤差去相關 × 線索軸 × 視角）+ 可部署路由器（R1）+ 棄權機制（B）。
負面的一半（F1/F2/F3 為何失敗）**佔同等篇幅**——這與既有的 blind-spot 系列一致，
也是這個系列的識別度所在。

**(乙) 負面：第四次否證。《模型融合幫不上忙，因為模型誤差是相關的》**
若 G 顯示 oracle ≈ best single，這與 A（軸向旋轉）/ B（器械幾何）/ C（不確定度）**完全同型**：
naive 表徵/集成主張再次失敗，因為既有表徵已隱含編碼了那個資訊。
四連否證加上一條共通機制，是比一個小幅正面增益更強的論文。

---

## 7. 待決事項（需要使用者決定）

1. **跑不跑 G 與 0.5**（都免費、都不需新推論）。G 決定走甲還是乙。
2. **要不要編列實驗 P 的 Kaggle 預算**，或整篇避開跨模型深度排名（推薦後者，先出兩週版本）。
3. 目標場域（期刊/會議）——這決定負面結果能佔多少篇幅。

## 相關 notes

`fit3d_view_dependence_summary.md`（exp2）、`fit3d_depth_recovery_summary.md`（exp1）、
`fit3d_decision_fidelity_summary.md`（exp3）、`fit3d_2d_vs_3d_summary.md`（誤差分解）、
`fit3d_model_comparison_summary.md`（三模型）、`fit3d_sparse_depth_summary.md`（MeTRAbs/MediaPipe）、
`fit3d_uncertainty_summary.md`（blind spot C）、`fit3d_axial_rotation_summary.md`（A）、
`fit3d_bar_geometry_summary.md`（B）、`rehab24_correctness_experiment_summary.md`（下游 LOSO）、
記憶 `fitness-aqa-depth-finding`（in-the-wild 下游）。
