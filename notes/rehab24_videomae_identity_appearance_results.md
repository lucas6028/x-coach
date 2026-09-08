# REHAB24-6：身份與外觀不能解釋 VideoMAE 的分數，但「第幾下」可以

**這份文件在回答什麼**

先前的[取景實驗](rehab24_videomae_framing_results.md)證明 VideoMAE 的分數來自「畫面裡的人」，
而不是背景。但那個結果留下一個更難的問題：模型看的是**動作**，還是**這個人長什麼樣子**？

REHAB24-6 的每一段錄影都只有一位受試者、一套衣服、一間實驗室，而且每段錄影有自己的
「做對比例」。一個只會認人的模型，可以完全不看任何一下動作就拿到 0.66。

本實驗把比較**限制在同一段錄影內部**來排除這個解釋：同一段錄影裡，身份、體型、衣著、
背景、相機、動作項目全部固定不變，所以任何在整段錄影中不變的特徵，都不可能製造出
「把做對的 rep 排在做錯的 rep 前面」的能力。

**事前登錄**：所有假設、主要統計量、排除規則與判讀表都寫在
[`rehab24_videomae_identity_appearance_validation_plan.md`](rehab24_videomae_identity_appearance_validation_plan.md)，
和分析程式、置換檢定亂數種子一起在 commit `f550c156` 定稿，**在讀取任何模型預測之前**。
執行日期：2026-08-23。事前登錄的兩個部分（主檢驗 + appearance-only 控制）**全部完成**，
另外加做兩組事後控制（§7），這兩組**不在事前登錄範圍內**，全文都會標明。

---

## 0. 三句話結論

**第一句（事前登錄的主檢驗，通過）。**
把比較限制在同一段錄影內部之後，模型仍然能把做對的 rep 排在做錯的 rep 前面：
九位受試者的 within-session ROC-AUC 平均 **0.8741 ± 0.0408**（範圍 0.8166–0.9446），
**9/9 位受試者都高於 0.5**，置換檢定 p = 1/10001（見 §4.3 對這個數字的說明）。
**穩定的身份、體型、衣著、背景與錄影條件，不足以解釋 VideoMAE 的 correctness 訊號。**

**第二句（事前登錄的負控制，方向一致）。**
把每支影片的正中間那一幀重複 16 次餵給模型、並讓同一支影片的所有 rep 共用同一份特徵
（`canonical_frame_repeat`，「只有外觀、沒有動作」的條件），balanced accuracy 只有
**0.5241 ± 0.0185**，比完整條件低 **0.1371 ± 0.0520**，**9/9 位受試者全部下降**
（exact Wilcoxon p = 0.0039）。兩個控制方向一致。

**第三句（事後才發現的問題，會改變後續決策）。**
**REHAB24-6 的標籤在錄影裡是分段的：做對的 rep 集中在前面，做錯的集中在後面。**
只用「這是第幾下」這一個數字、零個擬合參數，within-session AUC 就是 **0.1761** ——
反過來讀就是 **0.8239**，和模型的 0.8741 是同一個量級。61 段錄影裡有 **27 段**
是「前面全對、後面全錯」這種單一分界。因此本實驗**能**排除身份／外觀，
但**不能**分辨「模型看的是動作」還是「模型看的是這一下發生在錄影的什麼位置」。

> 這個混淆不在事前登錄計畫第 10 節的免責清單裡——那一節談的是「同一個 rep 之內的
> 時間順序」，而這裡是「rep 在整段錄影中的位置」，是另一件事。

---

## 1. 名詞說明

沒跟過這個專案的人請先讀這一節。

| 名詞 | 意思 |
| --- | --- |
| **rep（repetition）** | 一次動作反覆。REHAB24-6 把每支影片切成一次一次的 rep，每個 rep 有「做對／做錯」標籤。 |
| **session（錄影）** | 同一位受試者、同一個動作項目、同一次錄影。同一段錄影由 cam17、cam18 兩台相機同時拍，所以一個 rep 有兩列資料。 |
| **within-session AUC** | 只在同一段錄影內部，看模型有沒有把做對的 rep 排在做錯的前面。0.5 是亂猜，1.0 是完美，**0.0 是完美但方向相反**。 |
| **LOSO** | Leave-One-Subject-Out：每次留一位受試者當測試，其餘拿去訓練，輪完所有人。模型從沒看過測試者。 |
| **out-of-fold (OOF) probability** | 每個樣本由「沒看過它的那個模型」給出的分數。整份分析都建立在這上面。 |
| **balanced accuracy** | 對兩類各算一次正確率再平均，不會被類別比例灌水。0.5 是亂猜。 |
| **`full_frame_letterbox`** | 主要視覺條件：整張畫面補成正方形再餵給模型，不裁掉任何身體部位。取景實驗中分數最高的條件（0.6612）。 |
| **`canonical_frame_repeat`** | 本實驗新建的「只有外觀」條件，見 §5。 |
| **`background_only`** | 取景實驗建好的條件：把人所在的方框用左右兩側的背景填掉，場景與光線保留、人完全消失。 |
| **置換檢定（permutation test）** | 把標籤隨機打亂很多次，看真實分數贏過多少比例的亂打亂結果，藉此得到 p 值。 |

### 1.1 統計單位：n = 9，不是 1072

九位受試者、61 段可用錄影、1072 個 rep、3 個亂數種子，**推論單位只有「受試者」一個**。
順序固定為：兩台相機的分數先平均成一個 rep 分數 → 每段錄影算一個 AUC → 三個種子平均
→ 每位受試者把自己的錄影平均 → 九位受試者再平均。
種子與錄影都**不是**額外的樣本量。P10 只有 16 個樣本，依計畫只做敏感度分析。

---

## 2. 先過的門檻（看到任何 AUC 之前）

事前登錄計畫第 6 節列了四道門檻，全部通過：

| 門檻 | 要求 | 實測 |
| --- | --- | --- |
| **6.2 分組** | 主要範圍應為 2,128 列、64 段錄影、其中 61 段標籤混合、2,056 列 | 完全吻合；各動作項目的混合錄影數 12/12/8/12/8/9 也吻合 |
| **6.2 相機配對** | 每個 rep 恰有 cam17、cam18 各一列且標籤一致 | 1,072 個 rep 全數通過，零例外 |
| **6.1 OOF 完整性** | 每個種子涵蓋全部 2,128 列，不重複、不缺漏、機率值有限、不得有測試受試者洩漏 | 三個種子皆通過 |
| **6.1 重現既有結果** | 新存的 OOF 必須重現已 commit 的取景實驗每一折 | **每一折的 threshold 與 balanced accuracy 誤差皆為 0.000e+00**（不是「接近」，是完全相同） |
| **6.4 凍結** | 計畫、分析程式、置換種子 20260823、輸出格式在讀取結果前先 commit | commit `f550c156` |

最後一項值得多說一句。「大約等於 0.6612」這種檢查，即使 validation subject 選法漂移、
threshold 目標改掉、或種子套用在錯的位置，都還是會過。**逐折完全相同**才會抓到這些。
本實驗刻意不重寫訓練迴圈，而是讓既有的 `run_arm` 多回傳它本來就算出來的機率值。

另外，rank 型 AUC 的實作與「AUC 的定義式」（逐對比較 P(正 > 負) + 0.5·P(相等)）
逐段錄影核對，最大誤差 **3.3e-16**。觀測值與虛無分布用的是同一個估計量。

---

## 3. 主檢驗：同一段錄影內部，模型仍然排得出對錯

| 受試者 | within-session AUC |
| --- | ---: |
| P1 | 0.8368 |
| P2 | 0.8945 |
| P3 | 0.8745 |
| P4 | 0.8166 |
| P5 | 0.8350 |
| P6 | 0.9446 |
| P7 | 0.8847 |
| P8 | 0.9132 |
| P9 | 0.8673 |
| **平均** | **0.8741 ± 0.0408** |

- 範圍 0.8166–0.9446，**9/9 位受試者高於 0.5**。
- 以受試者為重抽單位的 95% bootstrap 區間：**[0.8502, 0.8988]**。這是不確定性描述，不是檢定。
- 排除的錄影：3 段只有單一類別（`Ex1_PM_000`、`Ex3_PM_044`、`Ex5_PM_028`），
  單一類別算不出 AUC，計畫明訂不得改用別的指標頂替。61 段納入分析。
- P10 一併納入的敏感度分析：0.8867（10 位受試者）。

### 3.1 虛無分布

置換方式：在**每一段錄影內部**打亂 rep 的標籤，保留該段錄影的正負樣本數，
同一個打亂結果同時套用到兩台相機與三個種子。這樣打亂之後，唯一被破壞的東西就是
「同一段錄影內的排序能力」——錄影之間的差異、每段錄影的做對比例都原封不動。

- 虛無分布平均 **0.4999**，標準差 0.0212，第 95 百分位 **0.5351**。
- 觀測值 0.8741 遠在分布之外。

### 3.2 事前登錄的判讀規則

計畫第 7 節第一列要求三個條件同時成立，三個都成立：

| 條件 | 門檻 | 實測 | |
| --- | --- | --- | --- |
| 平均 AUC | ≥ 0.55 | 0.8741 | PASS |
| 高於 0.5 的受試者數 | ≥ 6/9 | 9/9 | PASS |
| 置換檢定 p | < 0.05 | 1/10001 | PASS |

事前登錄的結論文字：**穩定的身份、衣著、體型、背景與 session 條件，不足以解釋
VideoMAE 的全部 correctness 訊號**。

---

## 4. 三個容易誤讀的地方

### 4.1 0.8741 和 0.6612 不能直接比

0.6612 是取景實驗的 balanced accuracy：**逐樣本**、要過一個門檻值、跨錄影一起算。
0.8741 是 within-session AUC：**先把兩台相機和三個種子平均**、只比排序、只在錄影內部比。
兩者的降噪程度與比較範圍都不同，所以 0.8741 **不代表訊號比取景實驗測到的更強**，
其中有一部分只是雜訊被平均掉了。

事前登錄計畫第 7 節第三列預期的是相反的模式（「主檢驗接近 0.5、但整體仍有 0.6612」），
實際結果兩種模式都不是：主檢驗遠高於整體成績。合理的解讀是模型在**同一段錄影內**排序
乾淨，但每段錄影的分數水平有偏移，而一個全域門檻要同時服務所有錄影。這個解讀本身
**沒有被獨立驗證**。

### 4.2 p = 1/10001 是下限，不是估計值

10,000 次置換能回報的最小 p 值就是 1/10001 ≈ 9.999e-5。應該讀成
「小於這個置換次數能分辨的最小值」，不是「p 剛好等於 0.0001」。

### 4.3 次要分析沒有做多重比較校正，因為沒有可校正的家族

計畫第 4.4 節說次要 p 值成組出現時要報 raw 與 Holm 校正值。實際上只有一個次要檢定
產生 p 值（九位受試者 AUC 對 0.5 的 exact Wilcoxon，p = 0.00195），
**Holm 校正在單一 p 值上是恆等變換**，校正後仍是 0.00195。
分相機與分動作項目的次要分析**完全沒有附帶統計檢定**，只有平均值與高於 0.5 的人數。

---

## 5. 事前登錄的負控制：只有外觀能拿到多少分

`canonical_frame_repeat` 的建法：對每一支來源影片，取**整段影片的時間中點那一幀**
（規則只用影片總幀數，完全不看 rep 邊界、不看標籤），把這一幀重複成 16 幀的片段，
再讓**同一支影片的所有 rep 共用同一份特徵**。身份、體型、衣著、背景、相機、光線、
動作項目全部保留，rep 之間的動作與姿勢差異全部移除。

門檻檢查：P1–P9 應有 128 支來源影片、含 P10 應有 130 支——實測 130 支，
且逐支影片對特徵陣列做 SHA-256，**130 支影片 = 130 個相異雜湊、每支影片內部完全相同**，
確認「共用」是真的共用而不是近似。

| 指標 | `canonical_frame_repeat` | `full_frame_letterbox` |
| --- | ---: | ---: |
| balanced accuracy | **0.5241 ± 0.0185** | 0.6612 |
| macro-F1 | 0.4880 | — |
| recall | 0.4967 | — |
| specificity | 0.5516 | — |

- 九位受試者的配對差值：**−0.1371 ± 0.0520**，範圍 −0.2336 至 −0.0688，
  **9/9 位受試者全部下降**，exact Wilcoxon p = 0.0039。
- 各受試者的 appearance-only 分數：0.487–0.551。
- P10 一併納入：0.5217。
- 計畫第 7 節的「捷徑閘門」（appearance-only ≥ 0.55 就要先去捷徑再談 fusion）
  **沒有觸發**。

**這不是等價性宣稱。** 0.5241 只說明「這個特定的 appearance-only 建法沒有顯示足夠訊號」，
不能寫成「外觀完全沒有貢獻」——那需要另做一個有足夠檢定力的等價性設計。

---

## 6. 次要分析（不能取代主結果）

分相機：cam17 平均 0.8206（9/9 高於 0.5）、cam18 平均 0.8670（9/9）。
沒有哪一台相機單獨撐起結果。

分動作項目：

| 動作 | 錄影數 | 平均 within-session AUC | 高於 0.5 的受試者 |
| --- | ---: | ---: | --- |
| Ex1 | 12 | 0.9705 | 9/9 |
| Ex2 | 12 | 0.7193 | 9/9 |
| Ex3 | 8 | 0.9403 | 8/8 |
| Ex4 | 12 | 0.9088 | 9/9 |
| Ex5 | 8 | 0.8846 | 7/7 |
| Ex6 | 9 | 0.8331 | 8/9 |

六個動作全部高於 0.5，最低的 Ex2 是 0.7193。這些分層**沒有附統計檢定**，
也不得挑選有利的一格取代整體結果。

---

## 7. 事後控制：「第幾下」這個捷徑（**不在事前登錄範圍**）

以下兩組控制是**看到主結果之後才加的**，理由寫在下面。它們不改變第 3 節的
事前登錄判讀，但**會改變後續該做什麼**。

加做的理由：`sample_clip_starts` 取樣的範圍剛好就是一個 rep 的起訖幀，
所以「這個 rep 有多長」「這個 rep 在錄影的哪個位置」都直接決定模型看到哪些像素。
本專案在 Fitness-AQA 上已經被片段長度這個捷徑騙過一次。

### 7.1 兩個零參數控制

用**零個擬合參數**的數字當分數，跑完全相同的 within-session 統計：

| 控制 | within-session AUC | 高於 0.5 的受試者 | 置換 p |
| --- | ---: | --- | --- |
| rep 長度（last_frame − first_frame） | 0.5802 | 6/9 | p(較大) = 0.0002 |
| **rep 位置（這是第幾下）** | **0.1761** | 0/9 | **p(較小) = 1/10001** |

**位置這一列要反過來讀。** AUC 0.1761 代表「做對的 rep 幾乎都排在做錯的前面」——
方向相反，但強度等同於 **0.8239**。一個能把標籤反向預測到這種程度的特徵，
和正向預測一樣是捷徑。

直接看標籤結構就明白了：把每段錄影的 rep 按順序排開，數「對／錯」交替幾次，
**中位數只有 3 段**（最少 2、最多 7），而 **61 段裡有 27 段只有 2 段**，
也就是「前面全對、後面全錯」。REHAB24-6 的錄影流程本身就是分段的。

模型確實吃到了這個結構：模型排序與 rep 序號的 Spearman 相關**中位數 −0.570**。

另外一組數字要小心讀。把 61 段錄影依「位置 AUC 是否落在 0.35–0.65」切成兩堆：
中性那堆 3 段，模型平均 AUC 0.7975；另一堆 58 段，模型平均 AUC 0.8774。
**但那 58 段幾乎全部是位置 AUC 低於 0.35 的錄影**，也就是「做對的排在前面」這個
單一方向；這不是「位置資訊強弱」的對照，而是「幾乎所有錄影 vs 剩下三段」。
再加上中性那堆只有 3 段、來自 2 位受試者，**這個對比沒有任何推論價值**，
列出來只是為了完整交代做過什麼。真正承擔論證的是下一節的上界控制。

### 7.2 上界控制：把人塗掉，框外的場景漂移留著

`background_only`（人所在的方框被左右兩側背景填掉，框外的場景與光線、以及它們
隨時間的漂移都保留）跑完全相同的 predict → analyze 流程：

| | within-session AUC |
| --- | ---: |
| `full_frame_letterbox`（完整條件） | **0.8741 ± 0.0408** |
| `background_only`（人被塗掉） | **0.5357 ± 0.0472** |

- `background_only` 各受試者 0.466–0.606，7/9 高於 0.5，
  置換 p = 0.0317，bootstrap 95% 區間 [0.5069, 0.5647]。
  作為對照它的整體 balanced accuracy 是 0.5075（三個種子 0.5113 / 0.5002 / 0.5109）。
- **這是關鍵的上界**：如果 0.8741 主要來自「錄影中的時間位置經由框外的場景／光線漂移
  洩漏」，那麼一個保留了那些漂移、只把人刪掉的條件應該也要接近 0.87。它只有 0.5357。

所以「位置經由**框外的非人物像素**洩漏」這條路徑被壓到約 0.54。
**但這不排除位置經由框內像素洩漏。** 填色用的方框是整支影片共用的固定框（由 mocap
取聯集），所以框**內**隨時間的變化——受試者疲勞、位移、改變站位——在建構上就被移除了，
而那正是位置最可能藏身的地方。這一層本資料集無法再分辨。

---

## 8. 這個實驗**不**支持什麼

- **不支持「模型看的是動作本身」。** 標籤在錄影中是分段的（27/61 段完全分段），
  「這是第幾下」單獨就有 0.8239 的強度。本設計無法分辨「讀動作」與「讀在錄影中的位置」。
  位置中性的錄影只有 3 段、2 位受試者，遠不足以撐起推論。
  **2026-09-06 更新：** 後續的位置控制實驗把線性可讀的位置子空間（16 維）投影掉後
  AUC 仍有 0.8556，份額 4.9%（見 §9 末段）；線性路徑關閉，非線性路徑仍未測。
  **2026-09-08 update:** the SSv2-finetuned checkpoint scores within-session AUC
  0.7805 ± 0.0917 on the same arm (ΔAUC −0.094, 2/9, p = 0.0195), probe median 0.384,
  share 1.2%; see `rehab24_videomae_ssv2_checkpoint_results.md`. The 0.8741 here stays
  the Kinetics number and the quoted one.
- **不支持「模型使用時間順序」。** 同一段錄影內的單幀姿勢差異也可能產生 AUC。
  這需要 repetition 層級的靜態幀控制與時間打亂控制，本實驗都沒做。
- **不支持「外觀完全沒有貢獻」。** 0.5241 是「這個 appearance-only 建法沒顯示足夠訊號」，
  不是等價性證據。
- **不支持跨衣著、跨場地、跨日期、跨相機的穩健性。** REHAB24-6 沒有同一人換衣、
  換房間、換 session 的反事實設計。
- **不支持任何因果機制**，也不能從 REHAB24-6 推廣到其他族群或復健情境。
- **§4.1 對 0.8741 與 0.6612 差距的解釋（錄影層級分數偏移）本身沒有被獨立驗證**，
  它是一個未檢驗的說法。

---

## 9. Fusion 的 go / no-go

**事前登錄的規則說可以開。** 計畫第 7 節第四列：主檢驗通過、且 appearance-only
明顯弱於完整模型 → 允許進入一個新的 fusion 事前登錄。這兩個條件都成立。

**但事後控制（§7）給了一個新的、事前登錄沒有涵蓋的條件。** 建議下一份 fusion
事前登錄在開跑前先鎖定一個位置控制——例如只在標籤非分段的錄影上評估，
或把 rep 序號當作必須被超越的零參數基線。這個判斷是本 note 的建議，
**不是**事前登錄規則，也不覆寫上面那一段。

**優先於 fusion 的，是先確認 0.8741 有多少不是「第幾下」。**
在同一個資料集裡這件事已經做到極限：`background_only` 把非人物路徑壓到 0.5357，
剩下的路徑要另外設計實驗（rep 層級的靜態幀控制、時間打亂控制），
或換一個標籤不分段的資料集。

**2026-09-06 後續：位置控制實驗已完成**
（[`rehab24_videomae_position_control_results.md`](rehab24_videomae_position_control_results.md)）。
特徵確實線性編碼類內位置（probe Spearman 中位數 +0.43，null [−0.086, +0.085]）。
在每個 fold 內投影掉 16 個能預測類內位置的方向再重訓，within-session AUC 從 0.8741 變成
**0.8556 ± 0.0435**（9/9，p = 1/10001），沿位置子空間的份額 **4.9%**；probe 中位數落回
null（+0.05，p = 0.24），但 probe 平均值 +0.12 仍顯著，P4、P8 沒拿乾淨。
依該計畫第 7 節第一列（在「k = 1 primary 正控制失敗、改讀 k = 16 secondary」這個
plan deviation 之下），線性位置路徑關閉，fusion pre-registration **允許開**，條件不變：
只比 NLF 與 `full_frame_letterbox` 的 calibrated late fusion、validation-only calibration、
單一 fusion 規則、以 subject 為配對單位、相對 NLF 沒有改善就停。
非線性的位置路徑仍未測，本節上一段建議的位置基線仍然適用。

---

## 10. 與計畫的差異（plan deviations）

| 差異 | 內容 | 影響 |
| --- | --- | --- |
| 折數 | 計畫寫「9-fold LOSO」，實際跑 10 折（P10 也輪一次測試位置） | 無。P10 樣本數不足以當 validation subject，在其他每一折都在訓練集裡，所以 P1–P9 的機率值與 9 折限制完全相同；額外那一折免費提供 P10 敏感度分析 |
| appearance arm 的取景 | 計畫第 5 節只說「唯一改變是 clip construction」，未指定取景。實作**寫死**為 `full_frame_letterbox`，與主要條件一致 | 讓兩者只差在 clip 建法。刻意不做成參數，避免「跑兩次選好看的」 |
| AUC 參照實作 | 計畫未指定。本機沒有 scikit-learn，改用 AUC 定義式（逐對比較）自行實作為對照 | 無。最大誤差 3.3e-16 |
| Holm 校正 | 計畫要求次要 p 值成組時報 Holm | 實際只有一個次要 p 值，Holm 是恆等變換（見 §4.3） |
| **事後新增控制** | §7 的兩個零參數控制與 `background_only` within-session 分析**都不在事前登錄範圍** | 不改變第 3 節判讀；改變第 9 節建議 |

---

## 11. 重現

產出的檔案（全部在 repo 根目錄相對路徑下）：

```text
data/REHAB24-6/processed/videomae_identity_control/
    oof_seed{42,7,1234}.csv          # 每個樣本的 out-of-fold 機率；欄位由計畫鎖定為
                                     # sample_id, person_id, exercise_id, video_id,
                                     # repetition_number, camera, label, seed,
                                     # test_subject, probability
    folds_seed{42,7,1234}.json       # 逐折 threshold 與指標（重現檢查的依據）
    audit_labels_only.json           # 凍結前跑的、不含任何結果的門檻
    audit.json                       # 完整門檻，含逐折重現檢查
    within_session_summary.json      # 主檢驗 + 次要分析
    permutation_null.npz             # 10,000 次置換的虛無分布
    appearance_only_summary.json     # §5
    shortcut_controls_summary.json   # §7.1（事後）
data/REHAB24-6/processed/videomae_identity_control_background_only/
    within_session_summary.json      # §7.2（事後）
data/REHAB24-6/processed/videomae_raw_canonical_frame_repeat/
data/REHAB24-6/processed/videomae_framing/canonical_frame_repeat/videomae_mean_pool_fc_norm_mean/
```

指令（Windows，一律從 repo 根目錄執行）：

```powershell
# 事前登錄的主線，順序不可調換
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py audit --labels-only
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py audit
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze `
    --permutations 10000 --permutation-seed 20260823

# 事前登錄的 appearance-only 控制
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py extract-appearance `
    --variant canonical_frame_repeat
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py evaluate-appearance --seeds 42 7 1234

# 事後控制（§7），不在事前登錄範圍
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py shortcut-controls
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py predict `
    --arm-dir data\REHAB24-6\processed\videomae_framing\background_only\videomae_mean_pool_fc_norm_mean `
    --output-dir data\REHAB24-6\processed\videomae_identity_control_background_only --seeds 42 7 1234
.venv\Scripts\python.exe scripts\rehab24\videomae_identity_control.py analyze `
    --output-dir data\REHAB24-6\processed\videomae_identity_control_background_only `
    --permutations 10000 --permutation-seed 20260823
```

程式：`src/rehab24/videomae_identity_control.py`、`src/rehab24/videomae_appearance_only.py`。
測試：`.venv\Scripts\python.exe -m pytest tests/test_rehab24_videomae_identity_control.py`（26 項）。
`background_only` 的特徵由取景實驗建立，見
[`rehab24_videomae_framing_results.md`](rehab24_videomae_framing_results.md)。
