# REHAB24-6：VideoMAE 的特徵確實編碼了「錄影位置」，但拿掉線性的位置子空間後，同一段錄影內的排序能力只掉 0.02

**這份文件在回答什麼**

**我們擔心什麼。** 上一份實驗說 VideoMAE 在同一段錄影裡仍能把做對的 rep 排在做錯的前面
（within-session AUC 0.8741），但 REHAB24-6 的標籤在錄影裡是分段的：做對的集中在前面、
做錯的集中在後面。「這是第幾下」一個整數，反過來讀就有 0.8239。所以那個 0.8741 到底是
「模型看懂了動作」，還是「模型讀到這一下發生在錄影的什麼位置」，分不開。

**我們做了什麼。** 把特徵裡「能預測位置」的方向直接拿掉，再重新訓練，看排序能力剩多少。
拿掉的不是原始位置（那和標籤幾乎是同一件事，拿掉一定掉分、掉了也無法判讀），而是
**within-class position**：在同一段錄影、同一個標籤的 rep 裡排第幾——能預測它的方向，
就是「沿錄影時間軸 drift、又出現在畫面裡」的東西（站位、距離、燈光、疲勞）。

**看到什麼。** 位置訊息在特徵裡是真的、而且很寬：要拿掉 16 個方向，probe 才讀不出位置
（原本 Spearman ρ 中位數 +0.43，拿掉 16 個後 +0.05）。但拿乾淨之後，模型在同一段錄影裡
排對錯的能力幾乎沒變：**0.8741 → 0.8556 ± 0.0435，9/9 位受試者仍 > 0.5**。

**所以呢。** 「訊號來自畫面裡的人，不是錄影位置」這句話**保住了**，代價是兩個註記：
線性可讀的位置子空間最多只佔訊號的 4.9%；非線性的位置路徑沒測。原本登錄的「只拿一個
方向就夠」是錯的（拿一個方向後 probe 仍有 +0.26，positive control 沒過），所以登錄的
primary 不能判讀，讀的是 k = 16 這個 secondary——這是 plan deviation，全文都在這個標記下。
Fusion 的 pre-registration 可以開，但要帶著這兩個註記。

**事前登錄**：[`rehab24_videomae_position_control_validation_plan.md`](rehab24_videomae_position_control_validation_plan.md)
（commit `ada39ddd`），程式在產生任何 residualized OOF 之前 commit（`917281a2`）。
執行日期：2026-09-06。

---

## 0. 結論的三個層次，和一段錄影走一遍

這一節回答：主結果是什麼、為什麼要分三層講、在一段真實錄影上它長什麼樣。

**第一層（機制，確定）。VideoMAE 的特徵確實線性編碼了 within-class position（這一下在同類 rep 裡排第幾）。**
在測試受試者的錄影裡，用訓練受試者擬合的 ridge probe 預測 within-class position，subject-macro
Spearman ρ 中位數 **+0.43**（9 位受試者全部為正，範圍 +0.17 到 +0.69），
null 95% 區間只有 [−0.086, +0.085]，雙尾 p = 1/10001。這條門是真的開著的——
如果模型想靠位置得分，特徵裡有材料。

**第二層（事前登錄的 primary，依 gate 規則不得判讀）。拿掉一個方向不夠。**
移除 k = 1 個方向後（residualization），AUC 0.8720 ± 0.0439，看起來幾乎沒動；但事前登錄的 positive control
（§6.3：residualization 之後 probe 中位數必須落回 null 區間）**沒過**——probe 仍有 +0.26。
介入沒有拿乾淨，所以 0.8720 依計畫**不判讀**，不能寫成「位置不是來源」。這是本實驗
最大的意外：計畫假設位置在特徵裡是一條線，實際上是一個至少 16 維的子空間。

**第三層（事前登錄的 secondary，是唯一通過 positive control 的 arm）。拿掉 16 個方向後，AUC 0.8556 ± 0.0435，
9/9 位受試者 > 0.5，p = 1/10001；probe 中位數 +0.05 落回 null 區間（p = 0.24）。**
從 0.8741 到 0.8556 差 0.0185，share（佔訊號 0.8741 − 0.5 的比例）**4.9%**。線性可讀的位置
子空間（16 維以內）最多只解釋 within-session 訊號的二十分之一。
**這一句是 secondary 升格，不是登錄的 primary，見 §5 的 plan deviation。**
而且 probe 的**平均值**（+0.12）仍在其 null 區間外——兩位受試者（P4、P8）的位置訊息
沒有拿乾淨，所以 4.9% 是下界。

**補一句（bound）。** 看得見的 drift（沿錄影時間軸單調變化、又出現在畫面裡的東西）是真的：同一段錄影內，人物框的左緣（AUC 0.6465，8/9）
和面積（0.3743，7/9 反向）都顯著（雙尾 p = 1/10001），但都沒跨過事前訂的 0.15 判準；
亮度不是載體（0.5069，undetermined）。

### 0.1 一段錄影走一遍：Ex4_PM_027（P3，27 個 rep）

大多數錄影長這樣：`11111000001111100000`——做對五下、做錯五下、再做對五下、再做錯五下。
在這種錄影裡「位置」就是一條 shortcut：只靠「越前面越可能做對」這個規則，把做對的排在做錯
前面的機率是 0.75。要看模型有沒有靠它，得找一段位置幫不上忙的錄影。

P3 的 Ex4_PM_027 就是：27 個 rep 的標籤是 `000001101111000001111100000`，對錯交錯了 7 段，
**位置規則在這段錄影的 AUC 是 0.494**——等於亂猜。同一段錄影：

| 分數來源 | 這段錄影的 within-session AUC |
| --- | ---: |
| 位置規則（越前面越可能做對） | 0.494 |
| 模型，k = 0（原始特徵） | 0.854 |
| 模型，k = 16（拿掉 16 個位置方向後重訓） | 0.803 |

位置在這裡完全沒有訊息，模型仍把做對的排在做錯前面 85%；拿掉位置子空間後掉到 80%，
掉的 0.05 就是這段錄影裡「與位置同向的那部分」。全部 61 段錄影平均起來就是 §3 的
0.8741 → 0.8556。反例也有：P1 的 Ex2_PM_003（`1111011111000000`）位置規則 0.921、模型
只有 0.302——在單一分界的錄影裡，模型可以比位置規則差很多，這正是 §4.1 說「27 段單一
分界只報不推論」的原因。

---

## 1. 名詞說明

| 名詞 | 意思 |
| --- | --- |
| **rep** | 一次動作。每支影片有多次 rep，各有做對／做錯標籤。 |
| **session（錄影）** | 同一人、同一動作、同一段錄影（`exercise_id + video_id`），兩台相機各一支影片。P1–P9 共 61 段標籤混合的錄影。 |
| **位置** | `Segmentation.csv` 的 `repetition_number`：這一下是該段錄影的第幾次。 |
| **within-class position（類內位置）** | 在同一段錄影、同一標籤的 rep 裡排第幾，歸一到 [0, 1]。拿掉的是這個而不是原始位置：原始位置和標籤幾乎是同一件事，拿掉它 AUC 一定掉，掉了也無法判讀。within-class position 把標籤固定住，剩下的才是「沿錄影時間軸的 drift」。 |
| **within-session AUC** | 只在同一段錄影內比：隨機抽一個做對、一個做錯的 rep，模型給做對那個較高分的機率。整段不變的東西（人、衣服、背景）對它零貢獻。 |
| **subject-macro** | 每段錄影一個數 → 同一受試者平均 → 九位受試者平均。**推論單位 n = 9**。 |
| **LOSO** | leave-one-subject-out：輪流拿一位受試者當測試集。 |
| **k** | residualization（把能預測 within-class position 的方向從特徵裡投影掉）拿掉的線性方向數。k = 0 不拿；k = 1 是登錄的 primary；k = 4、16 是登錄的 secondary，逐次「擬合、移除、再擬合」。 |
| **probe** | 從凍結特徵預測 within-class position 的 ridge 回歸，用來問「特徵裡還有沒有這個資訊」。 |
| **`full_frame_letterbox`** | framing 實驗得分最高的畫面處理：整幀補成正方形，人和背景都在。本實驗的特徵全部來自它。 |
| **naive arm** | 拿掉的是「原始位置」而非 within-class position 的方向；登錄時預期會過度移除。 |

## 2. 先過的 gate（看到任何新 AUC 之前）

| gate（計畫 §6） | 要求 | 結果 |
| --- | --- | --- |
| 6.1 OOF 完整性 | 每 seed 2,128 筆 primary 列，sample_id 唯一、機率有限、`person_id == test_subject`，每個 rep 恰有兩台相機一列且標籤一致，mixed-label sessions = 61 | **五個 arm × 三個 seed 全部通過** |
| 6.2 Fold 純度 | 每 fold 的 β 只用訓練受試者擬合；十個 fold 的 β̂ hash 互不相同 | 通過：k1 10/10、k4 40/40、k16 160/160、naive 10/10 個方向 hash 全部相異；`train_subjects` 明確排除測試與驗證受試者 |
| 6.3 positive control（前半） | residualization 之前的 probe 顯著 | **通過**：中位數 +0.4324，null [−0.086, +0.085]，p = 1/10001 |
| 6.3 positive control（後半，k = 1） | residualization 之後 probe 中位數落在 null 區間內 | **失敗**：+0.2609，null [−0.085, +0.089]，p = 1/10001 → primary 不判讀 |
| 6.3 positive control（後半，k = 16） | 同上 | 通過（中位數）：+0.0525，null [−0.091, +0.088]，p = 0.243。**平均值不通過**：+0.1183，null [−0.072, +0.071]，p = 0.0016 |
| 6.4 重現 | k = 0 必須 = 0.8741；探索腳本的 0.8497 / 0.8223 / 0.8487 由正式模組重現 | **通過**：k = 0 的 OOF 與身份控制的 OOF **逐筆完全相同**（三個 seed 最大差 0.0），AUC 0.8741；`replicate-exploratory` 重現 0.8741、0.8497、0.8223、0.8487、位置規則 BA 0.7309 到小數第四位 |
| 6.5 Frozen analysis | 計畫、程式、置換種子在讀取新 OOF 前 commit | 通過：`ada39ddd`（計畫）、`917281a2`（程式） |

ridge 的 λ 是計畫沒寫死的一個自由度，在看到任何結果前固定為：在該 fold 的**訓練**受試者內做
leave-one-subject-out，網格 {0.1, 1, 10, 100, 1000, 10000}，取 held-out within-class Spearman
最大者（平手取較大 λ）。十個 fold 選到 100–10000，每 fold 一個 λ，k > 1 的後續方向沿用。

## 3. 主結果：拿掉線性位置子空間之後，排序能力剩多少

這一節回答：每拿掉幾個位置方向，模型在同一段錄影裡排對錯的能力掉多少，以及哪一個 k
才算「拿乾淨」。答案在表後的三點。

同一統計、同一 null（每段錄影內置換標籤、10,000 次、種子 20260906）、同一九位受試者：

| arm | within-session AUC（mean ± SD，n = 9） | 受試者 > 0.5 | bootstrap 95% CI | p | probe 中位數（residualization 之後） | probe 過 positive control？ |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| k = 0（不拿） | **0.8741 ± 0.0408** | 9/9 | [0.850, 0.899] | 1/10001 | +0.4324 | —（residualization 之前） |
| k = 1（登錄 primary） | 0.8720 ± 0.0439 | 9/9 | [0.845, 0.899] | 1/10001 | +0.2609 | **否** |
| k = 4 | 0.8681 ± 0.0436 | 9/9 | [0.843, 0.896] | 1/10001 | +0.2885 | **否** |
| **k = 16** | **0.8556 ± 0.0435** | 9/9 | [0.830, 0.883] | 1/10001 | **+0.0525** | **是（中位數）；平均值 +0.1183 否** |
| naive k = 1（原始位置） | 0.8696 ± 0.0407 | 9/9 | [0.845, 0.895] | 1/10001 | 未跑 | — |

每位受試者（k = 0 → k = 16）：P1 0.837→0.840、P2 0.895→0.862、P3 0.875→0.802、
P4 0.817→0.801、P5 0.835→0.840、P6 0.945→0.944、P7 0.885→0.865、P8 0.913→0.886、
P9 0.867→0.861。**沒有一位掉到 0.80 以下。**

**怎麼讀這張表。** 三件事同時成立：

1. 特徵裡的位置訊息是真的、而且是**寬**的：拿掉 1 個方向後 probe 從 0.43 只掉到 0.26，
   拿掉 4 個後仍 0.29，要到 16 個才落回 null（中位數）。單一線性方向遠遠不夠，
   計畫 §3.1 「k = 1 為 primary」的假設錯了。
2. 分類器的排序能力對這個子空間**不敏感**：拿掉 16 個能預測位置的方向，AUC 只從 0.8741
   掉到 0.8556。依計畫 §7 第二列的公式，沿位置子空間的 share = (0.8741 − 0.8556) / (0.8741 − 0.5)
   = **0.049**。
3. 這個 share 是**下界不是上界**：k = 16 的 probe 平均值仍顯著（P4 +0.32、P8 +0.38、P2 +0.18、
   P6 +0.18），非線性的位置路徑完全沒測（§7）。

### 3.1 事前登錄的 decision table 命中哪一列

計畫 §7 第一列的條件是「primary（k = 1）≥ 0.80、≥ 6/9、p < 0.05，**且** residualization 之前 probe 顯著、
之後 probe 在 null 內」。前四項通過，最後一項**沒過**。依計畫 §6.3 的原文：
「否則介入失敗，primary 不得判讀」。**登錄的 primary 沒有判讀結果。**

k = 16 是計畫 §3.4 登錄的 secondary，它是唯一同時滿足 §6.3 兩個 probe 條件的 arm，
數字也落在第一列的 threshold 內（0.8556 ≥ 0.80、9/9、p < 0.05）。把它讀成第一列，
**是 plan deviation**（§5）。在這個標記下，第一列的判讀是：

> 線性 drift 方向不是 within-session 訊號的來源；位置這條路（線性、16 維以內）關閉。
> framing note 的「訊號來自畫面裡的人」與 identity note 的 0.8741 保留，附上界：
> 「線性 drift 已排除，殘餘 ≤ 0.8741 − 0.8556 = 0.0185」。

計畫說 k 不得事後挑選。這裡的選擇不是挑分數最高的 k——k = 16 是四個 arm 裡**最低**的——
而是挑唯一通過事前登錄 positive control 的 k。這仍然是事後選擇，所以標成 deviation。

## 4. 次要分析（不能取代主結果）

這一節回答三個附帶問題：在位置幫不上忙的 34 段錄影裡結果一樣嗎（4.1）、跨受試者的
balanced accuracy 有沒有被拿掉的東西影響（4.2）、加回 P10 會不會改變方向（4.3）。
答案：一樣；undetermined；不會。

### 4.1 position-balanced 統計（34 段標籤交錯的錄影）

每段錄影內，把 (做對, 做錯) 配對拆成「做對排前面」與「做對排後面」兩半各給 0.5 權重；
位置規則在此恆為 0.5000。這是探索分析 0.8497 的複製，但用**新訓練**的分類器與 residualized 特徵。

| arm | position-balanced AUC | 同 34 段未平衡 | 只取位置指向相反的配對（位置規則 = 0） | 27 段單一分界（8 位受試者，只報） |
| --- | ---: | ---: | ---: | ---: |
| k = 0 | 0.8497 | 0.8487 | 0.8223 | 0.9267 |
| k = 1 | 0.8447 | 0.8475 | 0.8147 | 0.9225 |
| k = 4 | 0.8464 | 0.8478 | 0.8186 | 0.9166 |
| k = 16 | 0.8320 | 0.8369 | 0.7952 | 0.9063 |
| naive k = 1 | 0.8498 | 0.8462 | 0.8248 | 0.9193 |

位置規則在同一批 34 段：未平衡 0.7309、平衡後 0.5000、反向配對 0.0000。
27 段單一分界的錄影（做對全在前、做錯全在後）位置規則是 0.9821——在那 27 段，
位置與標籤幾乎共線，任何分析都分不開（§7）。模型在那 27 段的 0.9063–0.9267
**只報不推論**：與 34 段不是同一堆錄影，不能相減。

### 4.2 LOSO balanced accuracy：framing note 那句話保不保得住

framing 實驗的頭條數字是 subject-macro balanced accuracy 0.6612 ± 0.0567。residualization 之後：

| arm | BA（mean ± SD，n = 9） | paired Δ vs 0.6612 | 正向受試者 | exact Wilcoxon 雙尾 p |
| --- | ---: | ---: | ---: | ---: |
| k = 0 | 0.6612 ± 0.0601 | +0.0000 | 0/9（全部相等） | — |
| k = 1 | 0.6643 ± 0.0571 | +0.0031 | 3/9 | 1.000 |
| k = 4 | 0.6570 ± 0.0514 | −0.0042 | 3/9 | 0.652 |
| k = 16 | 0.6717 ± 0.0511 | +0.0105 | 6/9 | 0.359 |
| naive k = 1 | 0.6660 ± 0.0638 | +0.0048 | 4/9 | 0.570 |

全部 p ≥ 0.05，全部是 **undetermined**——不是「沒有差異」。拿掉位置子空間沒有讓 LOSO
BA 顯著下降，也沒有顯著上升；這個指標 n = 9、SD 0.05，能偵測的最小差異遠大於這些 Δ。
（k = 0 的 SD 0.0601 與 framing note 的 0.0567 差在 ddof：本模組用樣本 SD。）

### 4.3 P10 敏感度（十位受試者）

k = 1：0.8848 ± 0.0579，10/10，p = 1/10001。k = 16：0.8700 ± 0.0614，10/10，p = 1/10001。
P10 只有 16 個樣本、一段錄影，AUC 1.0 把平均拉高；方向與結論不變。

## 5. 與計畫的差異（plan deviations）

| 項目 | 計畫 | 實際 | 影響 |
| --- | --- | --- | --- |
| **Primary 的判讀** | k = 1 為 primary，通過 §6.3 才判讀 | k = 1 的 positive control 失敗；改讀 k = 16（登錄的 secondary，唯一通過 positive control 的 arm） | **本 note 所有「位置路徑關閉」的句子都在這個 deviation 之下**。k = 16 是四個 arm 中分數最低者，不是挑有利結果，但仍是事後選擇 |
| §6.3 positive control 的統計量 | 「subject-macro within-class Spearman 中位數」 | 中位數通過（k = 16），但同一 probe 的平均值不通過（p = 0.0016） | 平均值不是登錄的判準，但表示 P4、P8 的位置訊息沒拿乾淨；§3 的 share 是下界 |
| λ 的選法 | 計畫未指定 | 執行前固定：訓練受試者內 LOSO、六點網格、每 fold 一個 λ（§2） | 無事後調整 |
| §6.4 的 null 平均 | 「null 平均 0.5002 ± 0.0211」 | 正式模組（sessions 排序後迭代）得 0.4999 ± 0.0211；探索腳本的迭代順序得 0.5002 | 同一種子、同一統計量，差別只在 session 迭代順序改變了 RNG 的消耗順序；observed 0.8741 與 p 不變 |
| 測試檔路徑 | `tests/rehab24/test_videomae_position_control.py` | `tests/test_rehab24_videomae_position_control.py`（跟既有測試同一層） | 無 |
| 亮度的解碼 | 「一次 CPU 解碼」 | 每 4 幀取一幀的灰階平均（stride 記在 sidecar，改 stride 會拒絕沿用快取）。解碼器從 OpenCV 改為 ffmpeg（每幀 area 平均到 16×9 再取均值）：OpenCV 路徑在這台機器只有約 11 fps，第一次跑到 1 小時 45 分被系統以記憶體不足砍掉；兩種後端在真實影片前 40 個取樣幀的差最大 0.27 灰階（rank 統計不受影響），合成資料上有測試對照 | 無 |
| §6.4 null 檢查的觸發 | — | `replicate-exploratory` 只在 10,000 次 / 種子 20260906 時斷言 null 數字；五個統計量永遠斷言 | 無 |

## 6. Bound：看得見的 drift 有多強（計畫 §3.3）

這一節回答：§3 拿掉的那個子空間，在畫面裡對應到什麼看得見的東西？答案：人在錄影中
緩慢左右移動、離相機遠近改變，兩者都顯著，但都沒強到跨過事前訂的判準；燈光不是。

零參數：直接拿人物方框的幾何量或亮度當分數，套與 §3 相同的 within-session 統計
（兩台相機先平均、61 段、n = 9），但用**雙尾** p（反向預測和正向預測一樣算 shortcut）。
登錄的「informative」判準：|AUC − 0.5| > 0.15 **且** p < 0.05。

| 代理變數 | 抓的是 | within-session AUC | \|AUC − 0.5\| | 受試者 > 0.5 | 雙尾 p（raw / Holm） | 達登錄判準？ |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 方框 `x0`（左緣，正規化） | 站位左右 drift | 0.6465 | 0.1465 | 8/9 | 1/10001 / 0.0004 | **否**（差 0.0035） |
| 方框面積 | 前後距離 drift | 0.3743 | 0.1257 | 2/9 | 1/10001 / 0.0004 | 否 |
| 方框 `y0`（上緣） | 高度／相機沉降 drift | 0.5177 | 0.0177 | 6/9 | 0.407 / 0.814 | 否，**undetermined** |
| 每 rep 平均亮度（ffmpeg，每 4 幀） | 燈光／曝光 drift | 0.5069 | 0.0069 | 6/9 | 0.744 / 0.814 | 否，**undetermined** |

**怎麼讀。** 兩個幾何代理都顯著：在同一段錄影裡，做對的 rep 人物框偏右（`x0` 較大）、
框較小（面積 AUC 0.37 = 做對的較小），而且 8/9 與 7/9 位受試者同向。這是**看得見的**
drift——人在錄影過程中緩慢移動、離相機遠近改變——而且與標籤同向。它是 §3 拿掉的那個
子空間最直接的候選載體。但兩者都沒有跨過事前訂的 0.15，依計畫 §7 第五列**不指名**任何
一個代理為載體，也不改 §3 的判讀。亮度不是載體（點估計 0.507，p = 0.74，undetermined）。

這和 framing 實驗的「`box_geometry` 12 維在 LOSO 上只有 0.5075」**不矛盾**：那是跨受試者的
balanced accuracy，這裡是同一段錄影內的排序。框幾何跨人不泛化，但在一段錄影內會漂。

## 7. 這個實驗**不**支持什麼

- **只排除了線性路徑，而且只到 16 維。** residualization 是線性投影；非線性地編碼位置的路徑
  （MLP 完全可以用）沒有測。k = 16 的 probe 平均值仍顯著就是提醒。
- **不能把 k = 16 的 probe 落回 null 讀成「特徵已不含位置」**——那是等價主張，需要 power。
- **within-class position ≠ 位置。** 在做對／做錯切換那一瞬間跳變的 drift（例如教練喊停、換燈）
  與標籤完全共線，本設計看不到；27 段單一分界的錄影裡沒有任何分析分得開。只能重錄。
- **不檢驗 rep 內部（16 幀 clip 內）的時間順序。** temporal-shuffle 仍未做。
- **不檢驗跨日、換衣、換場地的泛化。**
- **0.7309 vs 0.6612 不是對等比較**（一參數規則 vs 完整 pipeline，未配對、未檢定）。
- **LOSO BA 的五個 Δ 全部 undetermined**，不是「residualization 不影響 BA」。
- **k = 16 讀成 primary 是 deviation**（§5）。在嚴格的事前登錄下，本實驗的登錄 primary
  沒有結論。

## 8. 對其他 note 的影響

- framing note（`rehab24_videomae_framing_results.md`）「訊號來自畫面裡的人」**保留**，
  加註：「線性可讀的錄影位置子空間（16 維）最多解釋 within-session 訊號的 4.9%；
  非線性路徑未測」。
- identity note（`rehab24_videomae_identity_appearance_results.md`）§9 fusion go/no-go：計畫 §7 第一列在 deviation 標記下允許開
  fusion pre-registration（只比 NLF 與 `full_frame_letterbox` 的 calibrated late fusion、
  單一規則、以 subject 為配對單位、相對 NLF 沒有改善就停）。是否接受這個 deviation
  是讀者的決定，不是本 note 的。

## 9. 重現

程式：`src/rehab24/videomae_position_control.py`（邏輯）、
`scripts/rehab24/videomae_position_control.py`（CLI）、
`tests/test_rehab24_videomae_position_control.py`（42 個合成資料測試）。
`videomae_stage_a.run_arm` 新增可選的 `feature_transform_factory` / `materialize_root`，
預設路徑不變（k = 0 的 OOF 與身份控制逐筆相同即是證明）。

```powershell
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py replicate-exploratory --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 0 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 0 --permutations 10000 --permutation-seed 20260906   # 0.8741
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 0 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 1 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 1 --permutations 10000                                  # positive control 失敗
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 1 --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 4 16 --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py probe --k 16 --permutations 10000
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 16 --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py predict --k 1 --naive --seeds 42 7 1234 --device cpu
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py analyze --k 1 --naive --permutations 10000 --permutation-seed 20260906
.venv\Scripts\python.exe scripts\rehab24\videomae_position_control.py drift-proxies --permutations 10000 --luminance-backend ffmpeg
```

全部 CPU（`.venv`，torch 2.13 CPU）；每個 arm 三個 seed 十折約 20–40 分鐘，k ≥ 1 先把
residualized 特徵實體化到 `features_{arm}/fold_P{n}/`（每個 arm 約 80 MB）。

Artifacts（`data/REHAB24-6/processed/videomae_position_control/`）：
`oof_{k0,k1,k4,k16,naive_k1}_seed{42,7,1234}.csv`、`folds_*_seed*.json`、
`fold_betas.json`（每 fold λ、β̂ hash、訓練受試者）、`probe_summary_{k0,k1,k4,k16}.json`、
`within_session_summary_{arm}.json`、`permutation_null_{arm}.npz`、
`within_session_summary_k{1,16}_with_p10.json`、`drift_proxies.json`、
`luminance_per_sample.csv`、`replicate_exploratory.json`。
探索腳本存檔：`notes/exploratory/rehab24_position_control_20260906/`。
