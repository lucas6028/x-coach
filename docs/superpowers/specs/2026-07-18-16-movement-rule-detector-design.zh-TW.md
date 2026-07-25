# 16 種動作的文獻引用式規則偵測器規格

**狀態:** 設計規格(偵測器實作的基礎) · **日期:** 2026-07-18
**作者:** Claude Fable 5 (x-coach) · **規則數:** 16 種標準動作共 70 條

> 本檔為 [`2026-07-18-16-movement-rule-detector-design.md`](2026-07-18-16-movement-rule-detector-design.md)
> 的繁體中文版。英文原稿為權威版本;兩份文件的規則數、`fault_id`、公式、門檻值與引用完全一致。
> `citation_support` 內的引文一律保留英文原文,不做翻譯——那些是實際讀過的文獻原句,
> 翻成中文就會從「可查證的引文」退化為「無法查證的轉述」。
>
> **術語對照(全文一致):** fault 動作錯誤 · rule 規則 · landmark 關鍵點 · view 視角 ·
> phase 階段 · rep 反覆 · severity ramp 嚴重度梯度 · threshold 門檻 · monocular 單鏡頭 ·
> sagittal plane 矢狀面 · frontal plane 額狀面 · valgus 外翻 · varus 內翻 · lockout 鎖定 ·
> ROM 活動範圍 · proxy 替代指標 · moment arm 力臂 · shear 剪力 · concentric 向心 ·
> eccentric 離心。結構欄位名(`fault_id`、`detection_heuristic`、`observability`、
> `biomechanical_rationale`、`citation`、`citation_support`)與 `view_type` 的值
> (side / front / rear / front_oblique / rear_oblique)、observability 的等級
> (high / medium / low / none)一律沿用英文,因為程式碼直接以這些字串為鍵。

---

## 1. 目的

本文件羅列 x-coach 全部 16 種標準動作的**核心動作錯誤規則**,是姿態規則偵測器
(`src/pose/pose_rule_detector.py`)逐一動作實作時所依據、已審查且有文獻支撐的基礎。
偵測器目前只出貨深蹲;本文的深蹲規則**重述已寫進程式碼的五條規則**(`knees_inward`、
`knees_forward`、`shallow_depth`、`excessive_forward_lean`、`heel_rise`),補上它們確切的
幾何定義與門檻,並首次附上明確的文獻引用(程式碼裡原本只有 KG 查詢字串)。其餘動作的規則
全部是新增的。

**每一條規則都至少有一筆文獻引用,而且是該文獻裡的具體發現支撐這條規則**——沒有任何一條
規則靠常識或無憑據的斷言。每筆引用都附一行 `citation_support`,引述或改寫該文獻的確切發現,
來源是實際讀過的資料(`data/rag/docs/` 底下的 RAG 文件,或研究過程中實際抓取的網頁)。
這就是這項任務所要求的防幻覺保證。

## 2. 方法

- **深度:** 每個動作 3–6 條核心、生物力學上重要、且單鏡頭可偵測的動作錯誤(重質不重量),
  比照既有的深蹲規則集。
- **引用以 RAG 優先:** 既有的 RAG 語料庫(`data/rag/docs/`,81 份文件透過
  `data/paper_metadata.json` 對應到各動作)支撐了 16 種動作中的 15 種。子研究者逐動作讀過
  相關論文,再為每條規則取出具體發現。
- **缺口才上網補:** 有三個動作的 RAG 同儕審查覆蓋率很薄或掛零——**High Knee**(沒有 RAG
  文件)、**Torso Twist** 與 **Jumping Jacks**(只有 Wikipedia)。這幾個動作改以網路搜尋找
  同儕審查來源,並實際抓取原頁面查證。Wikipedia 只用來補充*描述性*的說明,絕不單獨拿來支撐
  傷害風險的主張。
- **權威書目:** RAG 來源的論文,其作者/標題/期刊以 `data/paper_metadata.json` 與參考文獻索引
  (§6)的條目為準;內文引用以 PMCID/PMID/DOI 為鍵,那才是可靠的定位錨點。

## 3. 偵測模型(適用於所有規則)

偵測跑在 **MediaPipe Pose 的 33 個關鍵點**上,座標為正規化影像座標
(x, y ∈ [0,1],y 往下遞增;另有 z 深度與 visibility 分數),**單鏡頭**。全文引用的關鍵點
索引如下(名稱沿用 MediaPipe 官方命名):

| idx | landmark | idx | landmark | idx | landmark |
|----|----------|----|----------|----|----------|
| 0 | nose | 13 | L elbow | 25 | L knee |
| 7 | L ear | 14 | R elbow | 26 | R knee |
| 8 | R ear | 15 | L wrist | 27 | L ankle |
| 11 | L shoulder | 16 | R wrist | 28 | R ankle |
| 12 | R shoulder | 23 | L hip | 29/30 | L/R heel |
| | | 24 | R hip | 31/32 | L/R foot index |

每個 rep 會估出一個 `view_type` ∈ {side, front, rear, front_oblique, rear_oblique}
(`src/pose/view_estimation.py`)。許多動作錯誤只在特定視角看得到;每條規則都註明所需視角
與 **observability** 等級(high / medium / low / none)。所需視角不存在時信心值會下調
(已寫進程式碼的深蹲偵測器是乘上約 0.65),本文沿用這個慣例。

## 4. 單條規則的結構

每條動作錯誤包含:**fault_id**(snake_case)、**fault_name**、**description**(描述)、
**detection_heuristic**(具體的姿態幾何訊號 + 門檻 + 方向)、**observability**(等級 +
所需視角)、**biomechanical_rationale**(傷害或表現層面的理由)、**citation**(書目 +
PMCID/PMID/DOI/URL),以及 **citation_support**(支撐該規則的具體發現)。若某個廣為人知的
動作錯誤**無法**從單鏡頭姿態可靠還原(例如翼狀肩胛、真正的腰椎屈曲),就標成 observability
`low`/`none` 並寫明所用的替代指標,而不是捏造一個看似精確的量測。

## 5. 規則總表

| 分組 | 動作 | 規則數 |
|---|---|---|
| A | 深蹲 Squat、弓步 Lunge、硬舉 Deadlift | 13 |
| B | 伏地挺身 Push-up、過頭推舉 Overhead Press | 10 |
| C | 划船 Row、彈力帶擴胸 Band Pull Apart | 9 |
| D | 二頭彎舉 Bicep Curl、手臂外展 Arm Abduction、手臂 VW Arm VW | 12 |
| E | 仰臥起坐 Sit-up、臀橋 Shoulder Bridge、腿部外展 Leg Abduction | 12 |
| F | 軀幹旋轉 Torso Twist、開合跳 Jumping Jacks、高抬腿 High Knee | 14 |
| | **總計** | **70** |



---

## A 組 — 下肢複合動作 — 深蹲、弓步、硬舉

動作:**Squat(深蹲)**、**Lunge(弓步)**、**Deadlift(硬舉)**。偵測跑在 MediaPipe Pose
(33 個關鍵點,正規化影像座標 x,y ∈ [0,1] + z 深度 + visibility)上,單鏡頭。以下用到的
關鍵點索引:11/12 肩、13/14 肘、15/16 腕、23/24 髖、25/26 膝、27/28 踝、29/30 腳跟、
31/32 腳尖(foot index)。每個 rep 會估出一個 `view_type`
(side / front / rear / front_oblique / rear_oblique)。

嚴重度梯度以驅動指標寫成 `mild → severe`;所需 `view_type` 不存在時信心值下調(比照既有的
深蹲偵測器,該動作錯誤所需視角缺席時信心乘上約 0.65)。

---

### 深蹲 Squat

反覆階段:setup(準備)→ descent(下降)→ bottom(最低點)→ ascent(上升)→ lockout(鎖定)。
(以下 5 條動作錯誤**重述**已寫進 `src/pose/pose_rule_detector.py` 的規則,那些規則先前
只帶了 KG 查詢字串;現在每條都有具體的文獻引用。)

#### 膝蓋內夾 / 膝外翻

- **fault_id**: knees_inward
- **fault_name**: 膝蓋內夾 / 膝外翻(Knees Inward / Knee Valgus)
- **description**: 膝蓋往內側塌陷,使負重階段的兩膝間距比兩踝間距還窄。
- **detection_heuristic**: 在額狀面投影上,`knee_width = ||L_knee(25) − R_knee(26)||`、
  `ankle_width = ||L_ankle(27) − R_ankle(28)||`(2-D,影像平面)。descent/bottom/ascent 期間
  `knee_width / ankle_width < 0.82` 就觸發。嚴重度梯度 0.82 → 0.70(比值愈低愈嚴重)。
- **observability**: **front / rear / front_oblique / rear_oblique** 為 high;side 為 medium
  (內側位移在側面被壓縮,信心 ×0.65)。
- **biomechanical_rationale**: 動態膝外展(外翻)是非接觸性前十字韌帶(ACL)傷害與髕股疼痛的
  主要機轉之一,它讓 ACL 與髕股關節外側承受這個動作本來就無力抵抗的負荷。
- **citation**: Ford KR, et al. "An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus." Open Access J Sports Med (2015). PMC4556293.
- **citation_support**: "knee abduction moment, which directly contributes to dynamic lower extremity valgus, was a significant predictor for future ACL injury risk with 73% sensitivity and 78% specificity in a prospective study of young female athletes";該文將 dynamic lower extremity valgus 定義為 "hip adduction and internal rotation, knee abduction, tibial external rotation and anterior translation, and ankle eversion",並指出 "high knee abduction moment was predictive of both PFP and ACL injury risk"。已於 RAG 文件中查證。

#### 膝蓋前移 / 膝關節前向位移

- **fault_id**: knees_forward
- **fault_name**: 膝蓋前移 / 膝關節前向位移(Knees Forward / Anterior Knee Translation)
- **description**: 矢狀面上膝蓋過度超出腳尖,使小腿在軀幹下降時往前推。
- **detection_heuristic**: 逐腿計算,把膝關節投影到足部向量 `toe − ankle` 上並以足長正規化:
  `knee_forward_ratio = (proj(knee−ankle onto foot) − foot_len)/foot_len`(2-D)。活動階段
  `knee_forward_ratio > 0.10` 觸發;≥ 0.30 為 severe。只在小腿是從側面看得到時才計算。
- **observability**: **side** 為 high(view_confidence ≥ 0.20);其餘為 low(正對鏡頭時
  膝到腳尖的投影不可靠——偵測器在這種情況本來就只回傳一個低可觀測性的占位結果)。
- **biomechanical_rationale**: 讓膝關節跑到腳尖前面會大幅拉高膝伸肌與髕腱負荷,那正是
  膝前側/髕腱過度負荷的機轉。
- **citation**: Zellmer M, et al. "Patellar tendon stress between two variations of the forward step lunge." J Sport Health Sci (2019). PMC6523035. [弓步研究;其機轉可直接轉移到深蹲的膝超腳尖問題。]
- **citation_support**: 在跨步距離標準化的條件下,膝在腳尖前(FSL-FT)相對於膝在腳尖後(FSL-BT),"peak patellar tendon stress … 11.1% greater"、應力衝量 "18.8% greater"、股四頭肌峰值力量高 12.6%、膝伸展力矩峰值高 25.8%(皆 p < 0.001;Table 1)。已於 RAG 文件中查證。

#### 蹲得不夠深

- **fault_id**: shallow_depth
- **fault_name**: 蹲得不夠深(Shallow Depth)
- **description**: 沒有蹲到平行——髖摺線沒有降到膝關節高度,轉換點的膝關節仍過度伸直。
- **detection_heuristic**: 在 `bottom` 階段,取髖中點(23,24)與膝中點(25,26):
  `hip_y − knee_y < −0.02`(影像 y 往下遞增,故髖仍高於膝)**或** `avg_knee_angle > 105°`
  就觸發。嚴重度梯度:髖軸 −0.02 → −0.10;膝角軸 105° → 125°;取兩者較大值。
- **observability**: **side / front / front_oblique** 為 high;rear/rear_oblique 為 medium
  (髖摺線被遮蔽)。
- **biomechanical_rationale**: 長期只在大重量下練部分行程(高於平行)的深蹲,與膝關節及脊椎
  關節的長期退化有關,而且截掉了讓深蹲有效的肌肉做功與 ROM 刺激。至少蹲到平行是訓練與比賽
  公認的標準。
- **citation**: Hartmann H, Wirth K, Klusemann M. "Analysis of the load on the knee joint and vertebral column with changes in squatting depth and weight load." Sports Medicine 43(10):993–1008 (2013). DOI 10.1007/s40279-013-0073-6, PMID 23821469. 另以 Wikipedia "Squat (exercise)" 條目作描述性補充(平行深度標準)。
- **citation_support**: PubMed 摘要(實際抓取):"With the same load configuration as in the deep squat, half and quarter squat training with comparatively supra-maximal loads will favour degenerative changes in the knee joints and spinal joints in the long term",以及 "the deep squat presents an effective training exercise for protection against injuries and strengthening of the lower extremity"。Wikipedia 補充:比賽標準是髖摺線低於膝蓋上緣,且 "incomplete squats … are both less effective and more likely to cause injury"。已透過 WebFetch 與 RAG 文件查證。

#### 軀幹過度前傾

- **fault_id**: excessive_forward_lean
- **fault_name**: 軀幹過度前傾(Excessive Forward Lean)
- **description**: 軀幹往水平方向摺下去(所謂 good-morning 深蹲),肩膀跑到髖關節前方,背角
  變得過於平躺。
- **detection_heuristic**: 影像平面上
  `torso_lean_deg = angle_from_vertical(shoulder_mid(11,12) → hip_mid(23,24))`。
  `torso_lean_deg > 35°` 觸發;嚴重度梯度 35° → 55°。
- **observability**: **side / front_oblique / rear_oblique** 為 high;正對鏡頭為 medium
  (信心 ×0.65——純正面/背面視角下軀幹俯仰會被壓縮)。
- **biomechanical_rationale**: 軀幹愈水平,負荷對腰椎的力臂就愈長,脊椎屈曲力矩與剪力隨之
  升高,下背受傷風險上升,同時把功從股四頭肌轉嫁到下背。
- **citation**: Moreira VM, et al. "Analysis of Muscle Strength and Electromyographic Activity during Different Deadlift Positions." Muscles (2023). PMC12225233. 另以 Ross S(Starting Strength)"The Good Morning Squat"(教練式描述)與 Wikipedia "Squat (exercise)" 補充。
- **citation_support**: PMC12225233:"leaning the trunk forward results in higher spinal flexion torque generated by the barbell. Therefore, ERE [erector spinae] requires higher activation and higher strength to avoid trunk flexion, reducing shear",且前傾愈多的姿勢伴隨愈大的腰椎剪力。Wikipedia:"Over-flexing the torso greatly increases the forces exerted on the lower back, risking a spinal disc herniation"。已於 RAG 文件中查證。

#### 腳跟離地

- **fault_id**: heel_rise
- **fault_name**: 腳跟離地(Heel Rise)
- **description**: 到最低點時腳跟離開地面,重心滾到前足,通常是踝背屈受限的代償。
- **detection_heuristic**: 逐腳計算 `heel_height_delta = heel_y(29/30) − toe_y(31/32)`(影像 y)。
  以 `setup` 階段建立基線(setup 各幀的平均);在 `bottom` 時
  `heel_height_delta − baseline > 0.015` 觸發。嚴重度梯度 0.015 → 0.055。
- **observability**: **side / oblique** 為 medium(腳跟相對腳尖的高度需要側面或斜側視角;
  正對鏡頭幾乎看不出來)。
- **biomechanical_rationale**: 腳跟離地代表踝背屈已耗盡或本來就受限,這會迫使動力鏈上游
  (膝、髖、脊椎)以代償性關節力矩補位,與踝、膝傷害風險有關;同時把負荷從後側鏈(臀肌)
  轉到前足與股四頭肌。
- **citation**: Mata AJ, Hayashi H, Moreno PA, Dudley RI, Sorenson EA. "Hip Flexion Angles During Supine Range of Motion and Bodyweight Squats." Int J Exerc Sci 14(1):912–918 (2021). 另以 Tumminello N, Human Kinetics, "Heel-raised squats aren't bad"(背屈受限的脈絡)與 Wikipedia "Squat (exercise)" 補充。
- **citation_support**: Mata 2021:墊高腳跟使踝關節活動幅度增加(25.9°→34.7° / 24.6°→33.2°,p<0.001)、深蹲深度增加(30.9%→55.0% 腿長,p<0.001),且 "reduced dorsiflexion mobility can lead to compensatory joint moments up the kinetic chain, potentially leading to injury"。Human Kinetics:踝背屈受限 "has been associated with … ankle injuries and knee injuries … abnormal lower extremity biomechanics"。Wikipedia:墊高腳跟會 "reduces the contribution of the gluteus muscles"。已於 RAG 文件中查證。

---

### 弓步 Lunge

反覆階段(以前腳為準):stance/setup(站姿/準備)→ descent(下降)→ bottom(最低點,深弓步)
→ ascent(上升)→ recovery(收腿)。

#### 前腳膝蓋超過腳尖

- **fault_id**: lunge_knee_past_toes
- **fault_name**: 前腳膝蓋超過腳尖 / 膝關節前向位移(Lead Knee Past Toes / Anterior Knee Translation)
- **description**: 下降時前腳膝蓋明顯跑到腳尖前方,前側膝關節被過度負荷。
- **detection_heuristic**: 針對前腳,
  `knee_forward_ratio = (proj(knee−ankle onto (toe−ankle)) − foot_len)/foot_len`(2-D,同深蹲)。
  descent/bottom/ascent 期間 `> 0.10` 觸發;≥ 0.30 為 severe。前腳 = 屈曲較多、位置較前的那隻腳。
- **observability**: **side** 為 high(view_confidence ≥ 0.20);正對鏡頭為 low(矢狀面的膝位移
  無法解析)。
- **biomechanical_rationale**: 讓前腳膝蓋位移到腳尖前方會實質增加髕腱應力與膝伸肌需求——在
  髕腱病變復健裡這是刻意用的進階槓桿,但在一般訓練裡是需要控制的負荷尖峰。
- **citation**: Zellmer M, et al. "Patellar tendon stress between two variations of the forward step lunge." J Sport Health Sci (2019). PMC6523035.
- **citation_support**: 膝在腳尖前的弓步(FSL-FT)相對膝在腳尖後(FSL-BT):"peak patellar tendon stress … 11.1% greater"、應力衝量 "18.8% greater"、股四頭肌峰值力量高 12.6%、膝伸展力矩峰值高 25.8%,膝屈曲峰值 110.2°→124.7°(皆 p<0.001;Table 1)。已於 RAG 文件中查證。

#### 前腳膝外翻

- **fault_id**: lunge_knee_valgus
- **fault_name**: 前腳膝外翻 / 內側塌陷(Lead Knee Valgus / Medial Collapse)
- **description**: 負重階段前腳膝蓋相對髖–踝連線往內側塌陷(膝蓋跑到腳掌內側)。
- **detection_heuristic**: 在額狀面投影上取前腳的髖(23/24)→踝(27/28)連線;量測膝
  (25/26)x 座標相對該線的帶號內側偏移量,並以髖寬正規化。往身體中線偏移 > 約 0.10·hip_width
  觸發;梯度 0.10 → 0.25。(這是額狀面膝外展的替代指標——單鏡頭姿態拿不到真正的 3-D 外展角。)
- **observability**: **front / front_oblique** 為 high;side 為 low(矢狀面看不到額狀面的塌陷)。
- **biomechanical_rationale**: 膝關節處的動態下肢外翻(髖內收/內轉 + 膝外展)是已被記錄的
  ACL 斷裂與髕股疼痛預測因子;單腳負重的弓步正是髖外展肌/旋轉肌控制失能會浮現的地方。
- **citation**: Ford KR, et al. "An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus." PMC4556293 (2015).
- **citation_support**: "knee abduction moment … was a significant predictor for future ACL injury risk with 73% sensitivity and 78% specificity";"the inability to eccentrically control hip adduction and internal rotation may lead to greater dynamic lower extremity valgus commonly seen during landing, squatting, and running"。已於 RAG 文件中查證。

#### 弓步深度不足

- **fault_id**: lunge_insufficient_depth
- **fault_name**: 深度不足(Insufficient Depth)
- **description**: 最低點時前腳膝蓋始終沒有接近直角,工作行程與股四頭肌需求都被截短。
- **detection_heuristic**: 前腳膝角 = `angle(hip(23/24), knee(25/26), ankle(27/28))`。整個 rep
  的前腳膝角最小值 `> 100°`(即屈曲不到約 80°)就觸發。嚴重度梯度 100° → 130°(愈伸直愈嚴重)。
  標準目標是膝屈曲約 90°。
- **observability**: **side / front_oblique** 為 high;正對鏡頭為 medium。
- **biomechanical_rationale**: 在中段行程內,髕股與股四頭肌負荷隨前腳膝屈曲單調上升;長期蹲太淺
  的弓步等於放棄那份強化刺激(臨床與實驗室的標準目標是屈曲 90°)。
- **citation**: Alkjær T, et al. "Forward lunge before and after anterior cruciate ligament reconstruction." PLoS One (2020), PMC6980669. 另以 Escamilla R, et al. "Patellofemoral Joint Loading During the Performance of the Forward and Side Lunge with Step Height Variations." IJSPT (2022), PMC8805090 補充。
- **citation_support**: PMC6980669 的實驗流程把目標深度定義為 "flexing the knee to 90°",而膝屈曲/伸肌力矩下降則標示功能受損(non-coper)。PMC8805090:"patellofemoral joint force and stress generally increased progressively as knee flexion increased during the descent phase"——也就是說,深度正是產生負荷與強化刺激的來源。已於 RAG 文件中查證。

#### 骨盆下掉 / 軀幹側傾

- **fault_id**: lunge_pelvic_drop
- **fault_name**: 骨盆下掉 / 對側軀幹側傾(Pelvic Drop / Contralateral Trunk Lean, Trendelenburg)
- **description**: 非支撐側骨盆下掉,以及/或軀幹倒向前腳那側,代表前腳髖外展肌控制不足。
- **detection_heuristic**: 額狀面投影:`pelvis_tilt_deg = angle_from_horizontal(L_hip(23) → R_hip(24))`;
  另計 x–y 平面上的 `trunk_lateral_lean = angle_from_vertical(shoulder_mid → hip_mid)`。
  bottom/ascent 全程持續 `pelvis_tilt_deg > 8°`(對側髖較低)就觸發;梯度 8° → 20°。
- **observability**: **front / rear** 為 medium;正對鏡頭的歧義靠髖關鍵點的 visibility 解決;
  純側面視角看不到。
- **biomechanical_rationale**: 對側骨盆下掉/同側軀幹側傾是髖外展肌(臀中肌)不足的外顯特徵,
  屬於動態外翻鏈的一環,會拉高 ACL 與髕股疼痛風險,也讓單腳負重失去穩定。
- **citation**: Ford KR, et al. PMC4556293 (2015). 交叉支撐:Alkjær T, et al. PMC6980669 (2020).
- **citation_support**: PMC4556293:"Failure to produce the abduction force is observed as a Trendelenburg posture, with the contralateral pelvis dropping",且以髖為核心的訓練減少了 "ipsilateral trunk inclination, and contralateral pelvis depression during a single leg squat"。PMC6980669 發現 ACL 受傷者的臀中肌肌電 "significantly higher for the ACL injured participants … possibly a compensatory mechanism to control the trunk and pelvis in the frontal plane"。已於 RAG 文件中查證。

---

### 硬舉 Deadlift

反覆階段:setup(準備)→ lift-off(離地)→ knee-passing(過膝,mid-pull)→ lockout(鎖定)。
**單鏡頭的先天限制:** 硬舉從側面拍,而其中最重要的動作錯誤——負重下的腰椎屈曲——用 MediaPipe
關鍵點只能微弱地觀察到(肩與髖之間沒有任何脊椎標記點),以下如實標註。

#### 負重下腰椎屈曲

- **fault_id**: deadlift_lumbar_flexion
- **fault_name**: 下背拱起 / 腰椎屈曲(Rounded Lower Back / Lumbar Flexion)
- **description**: 下背在離地(全程剪力最高的一刻)失去中立/伸展姿勢,拱成屈曲。
- **detection_heuristic**: **僅為替代指標。** MediaPipe 沒有腰椎關鍵點;肩(11/12)→髖(23/24)
  這段連線無法把真正的腰椎屈曲與髖鉸鏈或胸椎曲度分開。目前最好的替代作法:追蹤肩–髖線段角度,
  在 setup 與 lift-off 之間偵測與剛性髖鉸鏈不相容的背角*變化*(例如髖幾乎不動的情況下肩–髖
  線段在影像上縮短)。回報時給低信心值,並以視角品質為準。
- **observability**: **low**,需要 **side / side_oblique**;真正拱背與中立的差別,用 33 關鍵點
  的單鏡頭姿態無法可靠解析。此處不得宣稱精確度。
- **biomechanical_rationale**: 大重量下的腰椎屈曲會把剪力集中在椎間盤與後側結構上,是典型的
  硬舉下背傷害機轉;豎脊肌之所以要用力發力,正是為了阻止軀幹/脊椎屈曲。
- **citation**: Moreira VM, et al. "Analysis of Muscle Strength and Electromyographic Activity during Different Deadlift Positions." Muscles (2023). PMC12225233.
- **citation_support**: "The lift-off position in DL, using the powerlift posture, generates greater lumbar spine shear force",而豎脊肌活化在 lift-off/mid-pull 最高,因為 "leaning the trunk forward results in higher spinal flexion torque … ERE requires higher activation and higher strength to avoid trunk flexion, reducing shear"。已於 RAG 文件中查證。(這條規則因為教練價值而保留,但依如實回報原則標為低可觀測性。)

#### 槓鈴離開身體

- **fault_id**: deadlift_bar_drift
- **fault_name**: 槓鈴前飄 / 槓鈴離開小腿(Bar Drift / Bar Away From Shins)
- **description**: 槓鈴/雙手跑到足中線前方,而不是貼著腿走,把負荷對背部的力臂拉長。
- **detection_heuristic**: 沒有槓鈴的關鍵點,改用**手腕**當槓鈴的替代指標。側面視角:
  `bar_offset = wrist_x(15/16) − midfoot_x`,其中
  `midfoot_x = mean(ankle(27/28)_x, foot_index(31/32)_x)`,以足長正規化。lift-off/mid-pull 期間
  `bar_offset`(足中線前方)超過約 0.5·foot_len 觸發;梯度 0.5 → 1.2。
- **observability**: **side** 為 medium(需要側面視角;正對鏡頭時前後軸整個塌掉)。
- **biomechanical_rationale**: 槓鈴貼身可以把對腰椎的水平力臂壓到最小;讓它前飄會加大下背的
  力臂應力與背伸肌需求,安全性與效率同時變差。
- **citation**: Hanen NC, et al. "Biomechanical analysis of conventional and sumo deadlift." Front Bioeng Biotechnol (2025). PMC12148905, DOI 10.3389/fbioe.2025.1597209.
- **citation_support**: "keeping the barbell closer to the body during the SDL reduces the lever arm stress, thereby decreasing mechanical stress on the lower back";而寬站距的拉法之所以能減輕下背負荷,正是因為軀幹更直立、傾角更小。已於 RAG 文件中查證。

#### 髖比肩先起(直腿硬舉化 / 節段脫節)

- **fault_id**: deadlift_hips_shoot_up
- **fault_name**: 髖先於肩上升 / 軀幹過度前傾(Hips Rise Before Shoulders / Trunk Over-Inclination)
- **description**: 離地瞬間髖部往上竄而肩膀落後,軀幹倒向水平,整趟拉變成靠背硬撐。
- **detection_heuristic**: 側面視角。追蹤
  `torso_pitch = angle_from_vertical(shoulder_mid(11,12) → hip_mid(23,24))`,以及 lift-off 期間
  髖高與肩高的變化率。當 `torso_pitch` 在拉起初期*增加*(軀幹變平),亦即
  `Δ(hip_y) rises faster than Δ(shoulder_y)` 且 `torso_pitch > 55°`(相對垂直)就觸發。
  以俯仰角峰值 55° → 75° 作梯度。
- **observability**: **side / side_oblique** 為 high–medium;正對鏡頭看不到。
- **biomechanical_rationale**: 髖跑贏肩時軀幹更接近水平,脊椎屈曲力矩與背伸肌負荷都上升,對應
  文獻所指的更高腰椎需求;同時剝奪股四頭肌與髖伸肌的槓桿,拉的表現也變差。
- **citation**: Moreira VM, et al. PMC12225233 (2023). 交叉支撐:Hanen NC, et al. PMC12148905 (2025).
- **citation_support**: PMC12225233:"In these two positions [lift-off, mid-pull], leaning the trunk forward results in higher spinal flexion torque generated by the barbell",因此需要更大的豎脊肌力量來抵抗屈曲。PMC12148905:寬站距拉法 "maintain[s] a more upright posture … with a significantly reduced trunk inclination angle",藉此減少下背的力臂應力——反過來說,軀幹過度前傾正是要避免的受力狀態。已於 RAG 文件中查證。

#### 鎖定不完全

- **fault_id**: deadlift_incomplete_lockout
- **fault_name**: 鎖定不完全(Incomplete Lockout)
- **description**: 這一下結束時髖與膝沒有完全直立伸展(髖仍屈曲 / 軀幹沒站直)。
- **detection_heuristic**: 在最高點階段,`hip_angle = angle(shoulder(11/12), hip(23/24), knee(25/26))`、
  `knee_angle = angle(hip, knee, ankle(27/28))`。rep 結束時 `hip_angle < 165°` 或 `knee_angle < 165°`
  觸發(目標約 180° 的三關節伸展)。嚴重度梯度 165° → 140°。
- **observability**: **side** 為 high;斜側/正面為 medium(正對鏡頭時髖伸展部分被壓縮)。
- **biomechanical_rationale**: 一趟完成的硬舉,定義就是三關節伸展——髖與膝完全伸直、軀幹直立、
  肩胛後收;停在半路等於這一下的末端 ROM(以及臀肌/豎脊肌在鎖定處的需求)從未達成。
- **citation**: Hanen NC, et al. PMC12148905 (2025). 交叉支撐:Moreira VM, et al. PMC12225233 (2023).
- **citation_support**: PMC12148905:"the third event, lift completion, is achieved when the athlete assumes a fully upright position with extended hips and knees, with scapular retraction",以及 "A trial was considered successful if, at the end of the concentric phase, the participant stood upright with fully extended knees and hips, a straight torso, and retracted shoulders"。PMC12225233 把鎖定定義為 "the lifter's trunk reaches the vertical position … with the bar positioned at its highest point"。已於 RAG 文件中查證。

---

### 附註 / 誠實揭露的缺口

- **硬舉腰椎屈曲(deadlift_lumbar_flexion)** 在臨床上是硬舉最重要的動作錯誤,但用單鏡頭
  33 關鍵點的 MediaPipe 姿態只能達到 **low** 可觀測性(沒有脊椎關鍵點);上面的啟發式是明確
  標示的替代指標,不是精確量測。這一點如實標出,不遮掩。
- 深蹲 **knees_forward** 的引用(PMC6523035)其實是*前跨弓步*的研究;它「膝在腳尖前 vs 腳尖後」
  的對照是目前對膝關節前向位移負荷機轉最乾淨的量化,可直接轉移,但它終究不是純背蹲論文。
  在此據實註明。
- 以上所有 citation_support 字串都取自本次工作中實際讀過的來源(`data/rag/docs/` 底下的
  RAG 文件,加上以 WebFetch 抓取的 Hartmann 2013 PubMed 摘要)。沒有任何 UNVERIFIED 條目。


---

## B 組 — 上肢推 — 伏地挺身、過頭推舉

姿態模型:MediaPipe Pose,33 個關鍵點,單鏡頭,正規化影像座標(x,y ∈ [0,1] + z 深度 +
visibility)。關鍵點索引依 shared_context.md。

涵蓋:**Push-up(伏地挺身)**、**Overhead Press(過頭推舉)**。

---

### 伏地挺身 Push-up

反覆階段:**setup/top plank(準備/上位棒式)** → **descent (eccentric)(下降,離心)** →
**bottom(最低點)** → **ascent (concentric)(上升,向心)** → **lockout/top(鎖定/最高點)**。
偵測假設受試者是水平的;`side` 視角(相機在矢狀面內,垂直於身體長軸)是主要可用視角。
手掌與手肘寬度類的動作錯誤需要沿身體長軸的 `front`/`rear` 視角(相機放在頭側或腳側,略微抬高)。

#### 髖部下沉 / 棒式排列崩掉(腰椎過度伸展)

- **fault_id**: `pushup_hip_sag`
- **fault_name**: 髖部下沉 / 棒式線條斷掉(Hip sag / broken plank line)
- **description**: 髖部掉到肩–髖–踝直線以下(或拱到線以上),軀幹與腿不再構成一塊剛性棒式。
- **detection_heuristic**: 側面視角。取通過肩中點(11/12 中點)與踝中點(27/28 中點)的直線;
  量測髖中點(23/24 中點)到該線的帶號垂直偏移,以肩→踝長度正規化。髖往地面方向(y 較大)
  偏移超過身長的約 0.06 時判為 `sag`;髖高過該線同樣幅度時判為 `pike`。等價寫法:髖角
  (肩–髖–踝)偏離 180° 超過約 12°。
- **observability**: high —— `side`(矢狀面)。從 `front`/`rear` 幾乎為 `none`(偏移在該平面內
  被壓縮掉)。
- **biomechanical_rationale**: 中立棒式一旦失去,負荷就從腹壁轉移到腰椎的被動結構上,拉高
  L4–L5 的脊椎負荷;過大或反覆的腰椎負荷是伏地挺身變化式的傷害疑慮。
- **citation**: Freeman S, Karpowicz A, Gray J, McGill S. Med Sci Sports Exerc (2006).
  DOI 10.1249/01.mss.0000189317.08635.1b.
- **citation_support**: 該研究 "quantify[ied] the normalized amplitudes of the
  abdominal wall and back extensor musculature" 以及 "their impact on spinal loading by
  calculating spinal compression and torque generation in the L4-5 area",發現伏地挺身的
  姿勢會造成 L4–L5 壓縮力的大幅差異(單手伏地挺身產生 "the highest spine compression")。
  這確立了伏地挺身的軀幹姿勢主導腰椎負荷;而下沉(過度伸展)的軀幹正是拉高腰椎被動負荷的
  那種姿勢。註:該論文量的是各變化式的脊椎負荷,不是下沉角度本身,所以「下沉→負荷」這一步
  是從它所量化的負荷機轉推導出來的。

#### 深度不足 / 手肘 ROM 不完整

- **fault_id**: `pushup_shallow_depth`
- **fault_name**: 深度不足(部分行程)(Shallow depth, partial rep)
- **description**: 最低點手肘彎得不夠,胸口從未接近地面,整下只用了部分行程。
- **detection_heuristic**: 側面視角。肘角 = 肘關節處的夾角(肩 11 → 肘 13 → 腕 15),取左右
  較清楚的一側。在最低點那一幀(腕–肩垂直距離最小處),最小肘屈曲角 > 約 100–110°
  判為 `shallow`(完整一下大約要到 ≤90°)。可再用最高點到最低點之間肩部離地距離的變化很小
  來佐證。
- **observability**: high —— `side` / `front_oblique`;真正正對鏡頭的 `front`/`rear` 為 medium,
  因為肘角會被壓縮。
- **biomechanical_rationale**: 行程做短會減少機械功與目標肌群刺激,因為外部負荷與肩胛穩定肌
  的需求都隨著手肘進入更深的屈曲而上升。
- **citation**: San Juan JG, Suprak DN, Roach SM, Lyda M. BMC Musculoskelet Disord (2015)
  PMC4327800.
- **citation_support**: 該研究以 5° 為間隔量測整個伏地挺身行程的肘關節運動學,垂直地面反作用力
  "displayed a significant linear decrease across the ROM",且 "highest during the traditional
  PUP at 90° … of elbow flexion and lowest at 20°",同時前鋸肌等肌群的肌電隨手肘伸展而上升。
  肘屈曲愈深 = 力量與需求愈高,所以做太淺、從未進入深屈曲位置的一下,等於放棄了刺激中最大的
  那一塊。

#### 手肘外開 / 雙手過寬

- **fault_id**: `pushup_elbow_flare`
- **fault_name**: 手肘外開 / 手掌間距過寬(Flared elbows / excessive hand width)
- **description**: 雙手放得比肩膀寬很多,上臂大幅外展遠離軀幹,手肘朝外而不是往後走。
- **detection_heuristic**: 最好的替代指標要用沿身體長軸的 `front`/`rear` 視角:手寬比 =
  腕–腕距離(15↔16)/ 肩寬(11↔12);比值 > 約 1.6 觸發。若上臂看得到,可再用軀幹與上臂的
  夾角(軀幹向量 肩→髖 對上臂向量 肩→肘)超過約 65° 來佐證。純 `side` 視角下這條幾乎無法觀察
  (兩隻手腕重疊)。
- **observability**: medium —— `front` / `rear`(沿長軸);從 `side` 為 low/`none`。
- **biomechanical_rationale**: 手掌位置會實質改變手肘的節段間負荷,包括外翻(內側韌帶)力矩,
  所以異常寬或位移的手掌位置會把關節負荷帶離訓練預期的模式,並可能提高內側手肘的應力。
- **citation**: Donkers MJ, An KN, Chao EY, Morrey BF. J Biomech (1993).
  DOI 10.1016/0021-9290(93)90026-b.
- **citation_support**: 該研究記錄六種手掌位置下的肘關節受力:"peak forces exerted
  on the elbow joint along the forearm axis averaged 45% of the body weight for the
  'normal' hand position and were significantly decreased if hands were positioned either
  'apart' or 'superior'",而 "the maximum valgus torque at the elbow opposed by the
  medial ligamentous structure … was significantly increased if the hand was positioned
  superiorly"(單手時更上升 42%)。手掌位置因此強力調控肘關節負荷,足以支撐一條「偏離肩寬
  基準就標記」的規則。

#### 翼狀肩胛 / 缺乏肩胛控制

- **fault_id**: `pushup_scapular_winging`
- **fault_name**: 翼狀肩胛 / 肩胛控制不完整(Scapular winging / incomplete scapular control)
- **description**: 肩胛骨的內側緣/下角離開胸廓翹起(翼狀),而不是由前鋸肌把肩胛壓平/前引。
- **detection_heuristic**: 沒有可靠的單鏡頭訊號—— MediaPipe 的 33 個關鍵點不含肩胛骨邊緣的
  標記點,肩胛的傾斜/旋轉/翼狀都無法直接量測。只有一個很弱的間接替代指標:從 `rear` 視角看
  上背整體圓背程度所推得的肩胛區域外形,並不可信。建議不要輸出有信心的判定,只當作資訊呈現。
- **observability**: low/`none` —— 33 關鍵點模型從任何視角都解析不出來。
- **biomechanical_rationale**: 前鋸肌無力會讓肩胛翹起並過度內轉/前傾,縮小肩峰下空間,使人
  易發生肩夾擠——這正是伏地挺身/push-up-plus 被拿來訓練前鋸肌的原因。
- **citation**: Lee S, Lee D, Park J. J Phys Ther Sci (2013) PMC3820220;
  由 Abdollahi S et al. J Orthop Surg Res (2025) PMC12366113 佐證。
- **citation_support**: PMC3820220 指出 "Weakening of the serratus anterior muscle leads
  to excessive activation of the upper trapezius … reducing the dynamic stability of the
  scapula",進而導致 "a clash between the subacromion and the head of the humerus";
  PMC12366113 同樣提到前鋸肌疲勞會造成 "increased internal rotation and decreased posterior
  tilt of the scapula"。這個動作錯誤在生物力學上真實且重要,但老實說單鏡頭觀察不到,因此
  observability 標為 `none`。

#### 頭部前引 / 頸部下沉

- **fault_id**: `pushup_head_drop`
- **fault_name**: 頭部前引 / 頸部下沉(Forward head / neck drop)
- **description**: 頭往前伸或往下掉,頸部脫離脊椎的直線,常見於下巴搶在胸口前面去搆地板。
- **detection_heuristic**: 側面視角。頸線角 = 肩關節處,耳→肩向量(耳 7/8 → 肩 11/12)與
  肩→髖軀幹向量之間的夾角;當頭偏到軀幹線以下(鼻/耳的 `y` 明顯低於肩–髖連線)超過約 15°,
  或鼻 0 沿身體長軸明顯落在肩膀前方時,判為 `head_drop`。
- **observability**: medium —— `side` / `front_oblique`;從 `front`/`rear` 為 low。
- **biomechanical_rationale**: 正確的伏地挺身姿勢要讓 "the head, spine and pelvis …
  in a straight line, in a neutral state";頭部下沉/前引會讓頸椎持續處在非中立的負荷下,
  也是與肩夾擠相關的那種前引姿勢模式的指標。
- **citation**: Lee S et al. J Phys Ther Sci (2013) PMC3820220(姿勢/中立排列的標準);
  機轉由 Al Hammadi MI et al. Cureus (2025) PMC12514857 佐證。
- **citation_support**: PMC3820220 的實驗流程要求 "the head, spine, and pelvis
  were positioned in a straight line, in a neutral state" 且 "the cervical vertebrae in
  a neutral position",把中立的頸椎排列定義為正確姿勢;PMC12514857 則把 "forward head posture"
  列為會 "interfere with scapular movement … leading to a reduction in subacromial space"
  的姿勢因子之一,提供傷害面的理由。針對伏地挺身本身的頸椎傷害證據很薄,所以這條規則靠的是
  排列標準加上一般性的「前引姿勢→夾擠」機轉。

---

### 過頭推舉 Overhead Press

反覆階段:**rack/start(槓在肩上)** → **press (concentric)(向上推)** →
**lockout/top(鎖定/最高點,槓在頭頂)** → **lowering (eccentric)(下放)**。可站可坐。
`side` 視角(矢狀面)是脊椎、鎖定與槓路類動作錯誤的主要視角;左右不對稱則需要 `front` 視角。

#### 腰椎過度伸展 / 後仰

- **fault_id**: `ohp_lumbar_hyperextension`
- **fault_name**: 過度後仰 / 腰椎過度伸展(肋廓外翻)(Excessive back-lean / lumbar hyperextension, rib flare)
- **description**: 為了把槓推上去,身體往後仰、下背拱起,肋廓上抬,肩膀跑到髖關節後方。
- **detection_heuristic**: 側面視角。軀幹傾角 = 髖→肩向量(23/24 中點 → 11/12 中點)相對真垂直
  的夾角;推到中段以後肩膀落在髖關節後方超過約 10–15° 就觸發(軀幹略微前傾是正常的,往後仰
  才是錯誤)。可再用髖部前移增加(髖 `x` 跑到踝 `x` 前面)佐證。
- **observability**: high —— `side`(矢狀面);從 `front`/`rear` 為 low。
- **biomechanical_rationale**: 用腰椎伸展的後仰去換肩關節活動度,會把負荷集中到下背;歷史上
  這個代償正是造成推舉項目一連串下背傷害的原因。
- **citation**: Soriano MA, Suchomel TJ, Comfort P. "Weightlifting Overhead Pressing
  Derivatives: A Review of the Literature." Sports Med (2019) PMC6548056.
- **citation_support**: 該回顧記述比賽用的推舉如何退化成 "continental press",其
  "characterised by a considerable quick backbend before the lift",而且 "a long list of lower
  back injuries due to the accentuated backbend drove the IWF to eliminate the press from all
  future competitions"。過度後仰(腰椎過度伸展)在文中被直接點名為過頭推舉造成下背傷害的機轉。

#### 手肘鎖定不完全

- **fault_id**: `ohp_incomplete_lockout`
- **fault_name**: 頂點鎖定不完全(Incomplete lockout at the top)
- **description**: 最高點時手肘沒有完全打直,這一下停在穩定的過頭鎖定之前。
- **detection_heuristic**: 側面或正面視角。取槓位最高時的肘角(肩 11/12 → 肘 13/14 → 腕 15/16);
  肘伸展峰值 < 約 160° 判為 `incomplete_lockout`(完全鎖定約 175–180°)。取兩手中較差的一側。
- **observability**: high —— `side` / `front`。
- **biomechanical_rationale**: 收尾靠的是肘伸肌,它在接近完全伸展時才成為主導;一趟從未達到
  肘伸展的推舉就漏掉了定義「完成」的鎖定,也讓負荷停在關節上方而無支撐。
- **citation**: Evangelista P, Rum L, Picerno P, Biscarini A. "Decoding the Contribution of Shoulder and Elbow
  Mechanics … Sticking Region in Bench and Overhead Press." J Funct Morphol Kinesiol (2025) PMC12372072, DOI 10.3390/jfmk10030322.
- **citation_support**: 該連桿鏈模型發現 "elbow extensors contributed minimally
  during early lift phases but became dominant near full extension",而這個動作只有
  "when the elbow is fully extended … and the barbell reaches its
  final position" 才算完成,因此建議訓練策略要 "target … elbow strength near
  lockout"。完全的肘伸展就是「一下做完」的力學定義,停在半路是實質的 ROM 缺失。

#### 鎖定時頭部前引

- **fault_id**: `ohp_forward_head`
- **fault_name**: 鎖定時頭部前引(Forward head posture at lockout)
- **description**: 鎖定前後,頭往前伸出肩線之外。
- **detection_heuristic**: 側面視角。耳(7/8)沿前向軸相對肩(11/12)的水平偏移量,以肩寬
  正規化;耳往前超過約 0.3 個肩寬就觸發。
- **observability**: medium–high —— `side`(此線索為矢狀面偏移);從 `front` 為 low。

> **已撤回的子準則——槓路落在中線前方。** 本規則原本還帶第二個線索:*「(b) 槓在前方:鎖定時
> 腕(15/16)相對肩的水平前向偏移;腕沒有大致垂直疊在肩膀上方(偏移 > 約 0.3 個肩寬)就觸發。」*
> 該子準則已於 2026-07-25 **撤回**,實作中的 `wrist_forward_offset` 指標亦一併刪除,理由有三:
>
> 1. **它其實是在重述後仰。** 把槓路的參考點放在 *肩膀* 會和軀幹後仰混為一談:後仰時肩膀往後
>    移動,而槓仍停在支撐基底上方,因此「槓在肩膀前方」**就是** 後仰的力學特徵。實測上,一個
>    純粹後仰的動作會以 severity 1.0 / confidence 1.0 觸發本錯誤,把真正發生的
>    `ohp_lumbar_hyperextension`(0.41)壓在下面。
> 2. **正確的參考座標量不到。** 描述自己的用詞(疊在肩膀 **與足中線** 正上方)意味著槓路該以
>    足中線為參考,而那需要一個自行發明的足中線代理量——本專案「每個門檻都要有文獻支撐」的
>    前提不允許。
> 3. **引用文獻並不支持它。** Abdelraouf 等人(PMC13116542)是以 **頭顱脊椎角** 定義頭部前引,
>    那是一個「耳相對於肩」的量測,因此耳對肩的參考方式正是文獻在量的東西。該文獻,以及
>    PMC13086636 / PMC12514857,都完全沒有談到槓的位置。
>
> **待決的規格問題:** 過頭推舉規則集到底要不要一條真正的槓路錯誤?若要,它需要 (a) 一個
> MediaPipe 真的解得出來的支撐基底參考點,以及 (b) 它自己的引用文獻。在兩者到位之前,本規則
> 只是一條頭部前引線索。這是「待決定前先撤回」,不是默默改寫語意。
- **biomechanical_rationale**: 頭部前引/胸椎後凸的過頭姿勢會減少可用的肩胛上旋與肩屈曲,並壓縮
  肩峰下空間,一邊砍掉可達成的過頭 ROM,一邊拉高夾擠風險;而推舉本身的疲勞已被證實會把頭推向
  這種前引姿勢。
- **citation**: Abdelraouf OR et al. J Clin Med (2026) PMC13116542;機轉引自 Gregori
  P et al. J Exp Orthop (2026) PMC13086636 與 Al Hammadi MI et al. Cureus (2025)
  PMC12514857。
- **citation_support**: PMC13116542 發現大重量過頭推舉練到力竭會顯著降低頭顱脊椎角
  (該研究將 "a craniovertebral angle … less than 48 degrees … as forward head posture"),
  亦即推舉確實會造成可量測的頭部前引。PMC13086636 指出 "greater thoracic kyphosis is associated
  with … reduced shoulder abduction … and flexion",PMC12514857 則指出頭部前引/胸椎後凸會
  "reduce[s] the subacromial space",分別提供表現(ROM)與傷害(夾擠)兩面的理由。

#### 推舉左右不對稱

- **fault_id**: `ohp_asymmetric_press`
- **fault_name**: 推舉不對稱(單側領先)(Asymmetric press, one side leading)
- **description**: 一隻手推得比另一隻高或快,槓/雙手收在不同高度,肩帶因而傾斜。
- **detection_heuristic**: 正面視角。腕高垂直差 |y(15) − y(16)| 以肩寬正規化;鎖定前後 > 約 0.15
  判為 `asymmetric`,以及/或左右肘伸展角差 > 約 15°。可再用肩線傾斜(11↔12 不水平)佐證。
- **observability**: medium —— `front` / `rear`(需要額狀面);從 `side` 為 low(手臂重疊)。
  底層的肩胛貢獻無法直接追蹤。
- **biomechanical_rationale**: 持續的左右高度/時序差反映肩帶不對稱(肩胛運動障礙),這與肩胛
  穩定性受損及肩部傷害風險升高有關。
- **citation**: Abdelraouf OR et al. J Clin Med (2026) PMC13116542.
- **citation_support**: 該研究以肩胛平衡角與肩胛外移距離把肩帶不對稱操作化,將肩胛運動障礙定義為
  "a difference between the two sides of the body of more than 7 degrees in the scapular angle
  or more than 1.5 cm in the lateral shift distance",並發現大重量推舉練到力竭
  "resulted in a more protracted scapular position and shoulder girdle asymmetry
  (scapular dyskinesis)"。可量測的左右不對稱因此是一個經驗證、且臨床上有意義的動作錯誤標記
  (腕高只是替代指標,真正的肩胛量測 MediaPipe 抓不到)。

#### 過頭高度不足(槓沒有真正到頭頂上方)

- **fault_id**: `ohp_insufficient_elevation`
- **fault_name**: 過頭高度不足 / 推程過短(Insufficient overhead elevation / short press)
- **description**: 槓/雙手從未真正走到頭部上方——推舉停在額頭或眼睛高度(常伴隨聳肩卡住),
  而不是手臂過頭。
- **detection_heuristic**: 側面或正面視角。在最高點那一幀,若腕(15/16)沒有越過頭部——亦即腕 `y`
  沒有高過鼻 `y`(0)至少約 0.5 個頭高——卻仍被計為一下,就判為 `insufficient_elevation`。
  與 `ohp_incomplete_lockout` 的區別:這裡是腕的高度本身太低,即使肘角看起來有部分伸展。
- **observability**: medium–high —— `side` / `front`。
- **biomechanical_rationale**: 一趟完整的過頭推舉需要肩胛上旋、肱骨外展/屈曲與肘伸展同時發生,
  才能把負荷真正帶到頭頂上方;停在過頭之前,代表目標動作(以及對應的肩部肌群)從未被完整負荷。
- **citation**: Coratella G et al. Front Physiol (2022) PMC9354811;終點姿勢由 Bini et al. (2025) PMC12372072 佐證。
- **citation_support**: PMC9354811 指出 "the simultaneous scapular upward rotation
  …, together with the humerus abduction and elbow extension … makes the overhead press
  suitable to stimulate upper trapezius, deltoids and triceps",把完整的過頭終點姿勢定義為
  肩胛上旋 + 肱骨上舉 + 肘伸展的組合;PMC12372072 則規定唯有 "the barbell reaches its
  final position" 且完全伸展才算完成。停在頭頂高度以下的推舉並未達到這個終點姿勢。

---

### 附註 / 缺口

- **伏地挺身翼狀肩胛(`pushup_scapular_winging`)** 的 observability 標為 `none`:它是真實且
  引用充分的動作錯誤,但 MediaPipe 的 33 個關鍵點不含肩胛骨邊緣的點,單鏡頭量不到。這裡如實列出,
  不造假。
- **髖部下沉的引用**(McGill 2006)量的是各伏地挺身變化式的 L4–L5 脊椎負荷,不是下沉角度本身;
  「下沉→腰椎負荷」是從該論文量化的負荷機轉推導出來的(已寫在 citation_support 裡)。其餘
  citation_support 條目都是完整讀過的來源的直接發現。
- PMC6548056(WOPD 回顧)與 PMC12372072(sticking-region 模型)的作者名是從 RAG 文字推測的
  第一作者,屬近似值(那兩份文件沒有清楚的作者列);PMCID 才是權威依據——正式使用前請先核對
  作者字串。
- 不需要動用網路搜尋:這兩個動作的 RAG 覆蓋足以支撐所有輸出的規則。


---

## C 組 — 上肢拉 — 划船、彈力帶擴胸

偵測模型:MediaPipe Pose,33 個關鍵點,正規化影像座標(x,y ∈ [0,1]、z 深度、visibility)。
單鏡頭。每個 rep 估一次 view_type(side / front / rear / front_oblique / rear_oblique)。
關鍵點索引依 shared_context.md。

---

### 划船 Row(俯身划船 / 槓鈴划船)

反覆階段:**setup(髖鉸鏈,軀幹固定)→ concentric pull(槓/手拉向軀幹,肩胛後收)→
peak hold(槓到腹部,肩胛後收)→ eccentric lower(下放)→ return(回位)。**

#### 軀幹抬起 / 髖鉸鏈失守

| 欄位 | 內容 |
|---|---|
| **fault_id** | `torso_rising_hip_hinge_loss` |
| **fault_name** | 軀幹抬起(髖鉸鏈失守)(Torso rising, loss of hip-hinge) |
| **description** | 向心拉的過程中軀幹從 setup 的鉸鏈角度(接近水平)慢慢立起來,用髖伸展去幫忙搬動負荷。 |
| **detection_heuristic** | 軀幹向量 = midpoint(shoulders 11,12) → midpoint(hips 23,24)。分別在 setup 基線與拉的峰值計算 `trunk_angle_from_horizontal`。若 `trunk_angle_peak - trunk_angle_setup > 15deg`(軀幹變得更直立)就觸發。方向:角度增加 = 錯誤。 |
| **observability** | high —— side / front_oblique / rear_oblique(要有側向分量才讀得到軀幹俯仰)。純 front/rear 為 low。 |
| **biomechanical_rationale** | 俯身划船是一項軀幹穩定任務,豎脊肌必須等長維持鉸鏈姿勢;讓軀幹抬起等於用髖/腰椎的動量取代背肌做功,目標負荷下降,並形成動力鏈上近端穩定失守後的「補位」型態,與傷害有關。 |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; Owens LP et al. IJSPT / Int J Sports Phys Ther (2026) PMC13232157. |
| **citation_support** | Saeterbakken:自由重量俯身划船的豎脊肌肌電高於機械划船,雙側與單側皆然——亦即鉸鏈姿勢的自由重量划船施加了高強度、持續的軀幹伸肌穩定需求,而軀幹抬起正是放棄這個需求。Owens:描述動力鏈(KC)序列,指出 "breaks in efficient KC sequencing require distal segments to increase functional capacity... described as the 'catch-up' phenomenon",並特別採用軀幹平行地面的俯臥姿勢來控制划船時的軀幹姿勢。 |

#### 胸腰椎拱起(屈曲)

| 欄位 | 內容 |
|---|---|
| **fault_id** | `rounded_thoracolumbar_spine` |
| **fault_name** | 拉/持位時脊椎拱起(屈曲)(Rounded, flexed spine during the pull/hold) |
| **description** | 背部在負重下失去中立的微幅反弓,往前屈曲(拱背),而不是在鉸鏈姿勢中保持平直。 |
| **detection_heuristic** | 脊椎線曲度的替代指標:以肩中點(11,12)、合成的軀幹中點 = 0.5·(shoulder_mid + hip_mid)、髖中點(23,24)三點取中段脊椎夾角;或者追蹤肩→髖連線相對 setup 的直線參考。若肩中點掉到肩–髖直線以下、正規化下沉量 > 0.04(上背朝地面拱出)就判為屈曲。僅為單鏡頭替代指標——真正的脊椎屈曲無法直接量測。 |
| **observability** | medium —— side / front_oblique / rear_oblique;下沉替代指標很粗。純 front/rear 為 low。 |
| **biomechanical_rationale** | 鉸鏈姿勢的划船讓腰部豎脊肌承受高負荷,拱背划船會把負荷從主動收縮的伸肌轉到被動的脊椎結構(椎間盤/韌帶)上,那正是負重鉸鏈下腰部拉傷的典型機轉。 |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664;俯身划船 Wikipedia(描述性補充)—— `data/rag/docs/row_wiki.txt`。 |
| **citation_support** | Saeterbakken:自由重量俯身划船引發最高的豎脊肌肌電活性,確立鉸鏈姿勢下腰椎承受高伸肌負荷(這是同儕審查的負荷主張)。Wiki(僅描述性):建議 "maintaining an arch (a slight concavity) in the spine for a healthy lower back",並指出腹直肌收縮 "would cause the back to round and de-activate the lower back"。註:沒有任何 RAG 來源以實驗檢驗划船負荷下脊椎屈曲的傷害;傷害推論靠的是已記錄的高伸肌負荷,加上描述性的中立脊椎提示。 |

#### ROM 不完整 / 後收不足

| 欄位 | 內容 |
|---|---|
| **fault_id** | `incomplete_retraction_rom` |
| **fault_name** | ROM 不完整(槓/手沒有拉到軀幹;後收不足)(Incomplete ROM) |
| **description** | 拉到一半就停——槓/雙手從未碰到軀幹,肩胛也從未完全後收,行程頂端(峰值收縮)那一段被跳過。 |
| **detection_heuristic** | (a) 拉的深度:整個 rep 中腕(15/16)到髖(23/24)或到軀幹線的最小正規化距離;若 `min_wrist_to_torso_dist > 0.12`(手從未接近腹部)就觸發。(b) 峰值肘屈曲:頂點時 `elbow_angle (11-13-15 / 12-14-16) > 100deg` = 沒拉完。方向:殘餘距離愈大 / 肘角愈大 = 錯誤。 |
| **observability** | high —— 拉的深度用 side / oblique;肘的行程用 front/rear 也可以。 |
| **biomechanical_rationale** | 划船 ROM 的上半段(槓接近軀幹)才是闊背肌興奮度最高的地方,而完整協調的肩胛後收則是負荷中背、最佳化肩肱力量傳遞的關鍵;拉不到底就等於放棄闊背肌與肩胛後收肌的峰值負荷。 |
| **citation** | Fischer J et al. J Electromyogr Kinesiol (2025) PMID 40513198; Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611. |
| **citation_support** | Fischer(俯臥槓鈴划船,3 種 ROM):"The LD showed significantly higher mean muscle excitation in the upper-half ROM compared to both the lower-half ROM (p < 0.001) and full ROM (p < 0.001)"——拉的頂端(槓接近軀幹)驅動闊背肌的興奮峰值。Padovan:把划船描述為由 "scapular retraction, external rotation, and posterior tilt [which] contributes to optimizing glenohumeral alignment and force transmission" 所驅動,而向心終點 "defined when the handle reached the abdominal target"。 |

#### 借力 / 甩動(body English)

| 欄位 | 內容 |
|---|---|
| **fault_id** | `momentum_jerk_body_english` |
| **fault_name** | 借力甩動(Momentum / jerk, using body English) |
| **description** | 槓是被爆發式甩上來、全身一起晃,而不是受控地拉,造成速度尖峰與底部的鬆脫。 |
| **detection_heuristic** | 沿拉的軸向計算腕(15/16)的逐幀速度與加速度;若向心期腕加速度峰值超過該 rep 向心加速度中位數的約 3 倍,或軀幹角速度尖峰與腕的尖峰同時出現(整個人甩),就觸發。方向:出現急遽的加速度暫態 = 錯誤。 |
| **observability** | medium —— 只要看得到拉的那隻手腕,任何視角都行;需要穩定的影格率。 |
| **biomechanical_rationale** | 加速負荷等於卸掉肌肉的負荷:彈道式的向心會短暫拉高向心峰值力,卻在別處流失機械張力(離心需求尤其下降),而甩動造成的瞬間鬆脫/失重會把受控張力從目標肌群拿走,還可能在底部對脊椎造成衝擊負荷。 |
| **citation** | Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611;俯身划船 Wikipedia(描述性補充)—— `data/rag/docs/row_wiki.txt`。 |
| **citation_support** | Padovan:"Accelerating a given load during dynamic contractions increases force requirements during the concentric phase, whereas the same load imposes lower mechanical demands during the eccentric phase"——動量把負荷重新分配、並降低了這個動作原本要的受控張力;他們的實驗流程統一為向心 2 秒 / 離心 2 秒的受控節奏。Wiki(描述性):建議 "a slow tempo and avoiding jerking... prevents momentum from creating momentary weightlessness or slack in the muscles during the ascent, or... a jerking catch on the bottom of the lift"。 |

#### 左右拉不對稱

| 欄位 | 內容 |
|---|---|
| **fault_id** | `asymmetric_pull` |
| **fault_name** | 拉的左右不對稱(單側較高/領先)(Asymmetric pull) |
| **description** | 一隻手拉得比另一隻高或遠,肩線因而傾斜,並帶進軀幹旋轉。 |
| **detection_heuristic** | 在峰值時比較左右肘高 `|y13 - y14|` 與腕到髖的行程 `| dist(15,23) - dist(16,24) |`;另計肩線相對 setup 的傾斜 `|y11 - y12|`。肘高不對稱 > 0.05(正規化)或肩線傾斜相對 setup 增加 > 0.04 就觸發。方向:左右差持續擴大 = 錯誤。 |
| **observability** | high —— front / rear(兩側肩與肘都看得到);純側面視角為 low。 |
| **biomechanical_rationale** | 不對稱的拉會讓軀幹旋轉、把划船推向單側模式,顯著拉高抗旋轉核心(腹外斜肌)需求與不均的脊椎負荷,也讓雙側肩胛無法平衡後收。 |
| **citation** | Saeterbakken A et al. Int J Sports Med (2015) PMID 26134664; Padovan R et al. J Funct Morphol Kinesiol (2025) PMC12821611. |
| **citation_support** | Saeterbakken:"unilateral performance of exercises activated the external oblique more than bilateral performance, regardless of exercise"——非刻意的單側(不對稱)拉會帶來單側划船特有的較高抗旋轉/腹斜肌負荷。Padovan:把正確的划船界定為 "coordinated scapulothoracic motion" 與雙側肩胛內收到腹部目標——不對稱破壞的正是這個協調的雙側後收。 |

---

### 彈力帶擴胸 Band Pull Apart

反覆階段:**start(手臂前伸,彈力帶在胸口高度,雙手併攏)→ concentric horizontal abduction
(往兩側拉開 + 肩胛後收)→ peak(彈力帶碰胸,雙手張到最開)→ eccentric return(回位)。**

#### 聳肩(上斜方肌主導)

| 欄位 | 內容 |
|---|---|
| **fault_id** | `shrugging_upper_trap_dominance` |
| **fault_name** | 聳肩(肩膀往耳朵抬)(Shrugging, shoulders rise toward ears) |
| **description** | 拉開的過程中肩膀往耳朵方向上抬,代表上斜方肌過度活化,而不是由中/下斜方肌後收。 |
| **detection_heuristic** | 肩到耳的垂直間距:`gap = y_shoulder(11/12) - y_ear(7/8)`(影像 y 往下遞增,所以間距變小 = 肩膀抬起)。分別在 setup 基線與峰值計算;任一側 `gap_peak < gap_setup - 0.03`(肩膀上抬)就判為聳肩。方向:肩耳間距縮小 = 錯誤。 |
| **observability** | high —— front / rear(兩側肩膀與耳朵都看得到)。 |
| **biomechanical_rationale** | 彈力帶擴胸的用意是優先負荷中/下斜方肌與後側旋轉肌群,而上斜方肌貢獻要低;上斜方肌主導對肩痛/夾擠族群反而有害,因為它會加大肩胛前傾並可能壓縮肩峰下空間。 |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026; Camargo PR & Neumann DA, Braz J Phys Ther (2019) 23(6):467–475, PMC6849087, DOI 10.1016/j.bjpt.2019.01.011. |
| **citation_support** | Fukunaga:"it has been suggested that exercises should aim to preferentially target the middle trapezius, lower trapezius, and posterior RTC, with lower contributions from the upper trapezius and deltoid muscles"——聳肩把原本要的低上斜方肌型態整個反轉。Camargo & Neumann:"Exercises that increase the strength or relative activation of the upper trapezius may be counterproductive in many patients with shoulder pain, especially those with symptoms of impingement",因為 "the upper trapezius naturally causes an increased anterior tilt of the scapula, which may compromise the volume within the subacromial space"。 |

#### 水平外展 ROM 不完整

| 欄位 | 內容 |
|---|---|
| **fault_id** | `incomplete_horizontal_abduction_rom` |
| **fault_name** | ROM 不完整(雙手沒張到底 / 彈力帶沒碰到胸)(Incomplete ROM) |
| **description** | 雙手停在完整水平外展之前——彈力帶從未碰到胸口,手臂從未完全張開。 |
| **detection_heuristic** | 腕距峰值:`wrist_spread = dist(wrist15, wrist16)`,以肩寬 `dist(11,12)` 正規化。若 `wrist_spread_peak / shoulder_width < 1.6`(手臂沒有帶過軀幹線)就觸發,以及/或檢查肘伸展是否維持 `elbow_angle > ~150deg`(彎肘變成彎舉式偷吃步 = 錯誤)。方向:張開比值愈小 = 錯誤。 |
| **observability** | high —— front / rear。 |
| **biomechanical_rationale** | 擴胸的肌肉活性隨著對抗阻力所涵蓋的行程上升;停在完整水平外展之前,就放棄了大行程才有的較高肩胛肌活化,也到不了末端的肩胛後收。 |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga:肌肉活性峰值橫跨 "15.3% to 72.6% of MVC across muscles and exercise conditions",而斜上方向(行程最大、對抗重力)產生最高的斜方肌活性——"the diagonal up movement showing the highest shoulder-girdle muscle activity is understandable as the arm is moving against gravity, resulting in higher overall load"——亦即對抗彈力帶涵蓋愈多行程,目標肌群活化愈高,而截短的拉會失去這一塊。 |

#### 肩胛後收缺失

| 欄位 | 內容 |
|---|---|
| **fault_id** | `loss_of_scapular_retraction` |
| **fault_name** | 肩胛後收缺失(純手臂在拉)(Loss of scapular retraction, arms-only pull) |
| **description** | 彈力帶只被手臂拉開,肩胛始終前引(肩膀沒有往後、往中間收),整個動作變成純肩肱關節動作。 |
| **detection_heuristic** | 單鏡頭替代指標——肩胛後收從正面看不到。替代作法:從 REAR/rear_oblique 視角追蹤兩肩間距 `dist(11,12)`;真正的後收會隨肩胛內收而讓後側肩點略微變窄,而沒有後收的純水平外展則維持不變。若腕距增加超過門檻但 `dist(11,12)` 的變化 < 0.01(手動了、肩胛沒動),判為 `no_retraction`。這只是粗略的替代指標。 |
| **observability** | low–medium —— 以 rear / rear_oblique 為佳;純正面視角幾乎為 none(肩胛被遮蔽)。 |
| **biomechanical_rationale** | 擴胸的治療目標是中/下斜方肌的肩胛後收;若肩胛從未後收,肩胛周邊的後收肌群就被繞過,這個動作也失去訓練肩胛穩定肌的效果。 |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga:中斜方肌的活性明顯由偏後收的方向所驅動(斜上/水平高於斜下),而整個動作的設計是為了徵召 "periscapular muscles" 以達成 "scapular stabilization"——後收就是機轉,純手臂拉則把它拿掉。誠實的限制:肩胛位置本身無法從單鏡頭正面姿態可靠還原,因此可觀測性偏低。 |

#### 軀幹伸展代償(往後仰)

| 欄位 | 內容 |
|---|---|
| **fault_id** | `trunk_extension_compensation` |
| **fault_name** | 軀幹伸展代償(往後仰)(Trunk-extension compensation, leaning back) |
| **description** | 用軀幹往後仰/甩進腰椎伸展,把彈力帶甩開,而不是用肩帶肌群拉開。 |
| **detection_heuristic** | 軀幹向量 midpoint(shoulders 11,12) → midpoint(hips 23,24);計算相對垂直的傾角。若 `trunk_lean_backward > 10deg` 超過 setup 基線,或軀幹角速度尖峰與向心拉同時出現(甩),就觸發。方向:後仰增加且與拉同步 = 錯誤。 |
| **observability** | high —— side / oblique(要有側向分量才讀得到軀幹俯仰);純 front/rear 為 low。 |
| **biomechanical_rationale** | 站姿擴胸應由水平外展/肩胛後收驅動;改用軀幹伸展等於徵召腰椎動量來搬動彈力帶,既卸掉原本要的肩胛肌群負荷,又加上不受控的腰椎伸展負荷。 |
| **citation** | Fukunaga T et al. Int J Sports Phys Ther (2022) PMC8975561, DOI 10.26603/001c.33026. |
| **citation_support** | Fukunaga 確立擴胸是站姿的水平外展/對角肩胛運動,負荷應來自肩帶;該文提到軀幹/髖伸展可以被*刻意*使用,但目標肌群是肩胛周邊/旋轉肌群——因此,用後仰甩動去*取代*(而非穩定支撐)水平外展,會把動作帶離它預設的肌群。註:沒有任何 RAG/EMG 來源直接量化「後仰偷吃步」造成的傷害;這條是以受控執行/表現損失為依據、立基於該動作預期的水平外展力學,所以代償這個框架帶有部分推論成分(動作錯誤本身可觀測性高,但傷害主張是間接支撐的)。 |

---

### 查證 / 缺口小結

- 所有划船規則都錨定在同儕審查的 RAG 文件上(PMID 26134664、PMC12821611、PMID 40513198、
  PMC13232157);`row_wiki.txt` 只作描述性補充,絕不單獨支撐傷害主張。
- `rounded_thoracolumbar_spine`:*負荷量級*有同儕審查支撐(Saeterbakken 的豎脊肌),但沒有任何
  RAG 來源以實驗檢驗划船負荷下脊椎屈曲的傷害——已於條目內註明。
- 彈力帶擴胸由 Fukunaga PMC8975561 支撐,上斜方肌主導的傷害主張另加一筆已查證的同儕審查補充來源
  Camargo & Neumann PMC6849087(實際抓取並引述)。
- `trunk_extension_compensation`:傷害主張帶部分推論(Fukunaga 甚至指出軀幹伸展可能有益)——
  已於條目內註明;動作錯誤本身則高度可觀測。


---

## D 組 — 手臂 / 肩胛孤立訓練 — 二頭彎舉、手臂外展、手臂 VW

動作:**Bicep Curl(二頭彎舉)**、**Arm Abduction(站姿側平舉 / 肩外展)**、
**Arm VW(肩胛 V 到 W 的前引/後收)**。

偵測模型:MediaPipe Pose,33 個關鍵點,正規化影像座標(x,y ∈ [0,1]、z 深度、visibility),
單鏡頭。以下用到的關鍵點索引:0 鼻;7/8 左/右耳;11/12 左/右肩;13/14 左/右肘;15/16 左/右腕;
19/20 左/右食指;23/24 左/右髖。`view_type` ∈ {side, front, rear, front_oblique, rear_oblique},
每個 rep 估一次。

角度慣例(比照深蹲偵測器的寫法):
- `elbow_angle` = 以肘為頂點的夾角(肩–肘–腕),180° = 手臂打直。
- `arm_elevation_angle` = 軀幹向量(肩→髖)與上臂向量(肩→肘)之間的夾角;
  約 0° = 手臂垂在身側,約 90° = 水平,約 180° = 完全過頭。
- `torso_lean_deg` = 肩中點→髖中點向量相對影像垂直的角度
  (帶號:視角不同分別代表矢狀面的前後傾或額狀面的側傾)。
- `neck_gap` = (ear_y − shoulder_y),正規化單位;肩膀上抬時變小(y 往下遞增)。
  與該 rep 的 setup 基線比較。

---

### 二頭彎舉 Bicep Curl

反覆階段:**setup/bottom**(手臂垂於身側伸直,`elbow_angle`≈170–180°)→
**concentric**(肘屈曲,舉起)→ **top**(屈曲峰值,`elbow_angle`≈40–55°)→
**eccentric**(受控下放)→ 回到 bottom。

#### 手肘前移
- **fault_id**: `elbow_drift_forward`
- **fault_name**: 手肘前移(失去手肘固定)(Elbow drifts forward, loss of elbow fixation)
- **description**: 舉起時手肘往前/往上跑、離開軀幹,等於加進了肩屈曲,而不是把上臂固定在身側。
- **detection_heuristic**: `upper_arm_lean = angle(shoulder→elbow vector, image-vertical-down)`。setup 時上臂大致垂直(約 0–10°)。向心期任一幀 `upper_arm_lean > 25°` 偏向前側(手腕側),或肘部相對肩–髖垂直線的前向 x 位移超過 `0.5 × upper_arm_length`,就觸發。方向:肘相對肩往前/往上移動。
- **observability**: medium —— 需要 **side** 或 **front_oblique**(前移主要發生在矢狀面;純 **front** 視角下它會塌進深度 z,可靠性低)。
- **biomechanical_rationale**: 手肘前移會把彎舉變成部分的肩屈曲,負荷從肱二頭肌轉到前三角肌,目標肌群的刺激下降(表現損失)。
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: 該論文經驗證的正確執行流程規定手臂 "fully extended at the sides, with the elbows kept close to the torso throughout the whole movement",並由兩位研究者目視監控執行——亦即手肘固定貼著軀幹就是定義出來的正確姿勢,往前移即為偏離。(已查證——讀自 RAG 文件。)

#### 軀幹擺盪 / 借力
- **fault_id**: `trunk_swing_momentum`
- **fault_name**: 軀幹擺盪 / 後仰借力(Trunk swing / back-lean momentum)
- **description**: 身體往後仰,用髖/軀幹的動量把重量甩上去,而不是孤立肘屈曲。
- **detection_heuristic**: 追蹤整個 rep 的 `torso_lean_deg`(肩中點 11/12 → 髖中點 23/24 相對垂直)。若 rep 內振盪 `max(torso_lean_deg) − min(torso_lean_deg) > 12°`,或向心期的後仰超過 setup 基線 `> 10°`,就觸發。方向:向心開始時肩相對髖往後/往上移動。
- **observability**: high —— 以 **side**/**front_oblique** 最佳(矢狀面的後仰)。從 **front** 視角為 medium(矢狀面的傾斜投影到深度,只看得出粗略的振盪)。
- **biomechanical_rationale**: 軀幹動量會卸掉肱二頭肌的負荷(削弱預期刺激),並施加反覆的腰椎伸展/剪力負荷,構成下背傷害風險。
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: 受試者執行彎舉時 "avoiding trunk movements and jerky motions",且 "two experienced investigators visually monitored trunk movements and knee flexion to ensure the proper execution"——軀幹動作被明確當成需要排除的偷吃步/代償偏差。(已查證——讀自 RAG 文件。)

#### 活動範圍不完整
- **fault_id**: `incomplete_rom`
- **fault_name**: 活動範圍不完整(部分行程彎舉)(Incomplete range of motion, partial curl)
- **description**: 這一下在底部沒有把手肘完全伸直,以及/或在頂部沒有完全屈曲。
- **detection_heuristic**: `elbow_angle` = angle(肩 11/12 – 肘 13/14 – 腕 15/16)。整個 rep 的 `max(elbow_angle)` `< 150°` 判為**伸展不完整**;`min(elbow_angle) > 60°` 判為**屈曲不完整**。
- **observability**: medium–high —— 以 **side**/**front_oblique** 最佳(前臂落在影像平面內;**front** 視角下前臂被壓縮,`elbow_angle` 精度下降 → medium)。
- **biomechanical_rationale**: 訓練涵蓋完整關節行程,能在各個手臂角度上帶來更好的力量適應;長期只做部分行程,會放棄行程兩端(伸直端與屈曲端)本來可得的力量增益(表現損失)。
- **citation**: Havers et al., *European Journal of Sport Science* (2025), DOI 10.1002/ejsc.70087 (PubMed 41247250);由 Parpa K et al., *Muscles* (2025), PMC12550948 佐證。
- **citation_support**: Havers 等人發現完整 ROM(0–140°)的力量增益優於起始段部分 ROM——1RM 更大(SMD≈0.17),在 100° 肘角的 MVC 也更高(SMD≈0.24)。RAG 文件(Parpa)則規定 "a slow, controlled lowering of the dumbbells back to the starting position through the full range of motion"。(已查證——抓取 Havers 的 PubMed 摘要 + 讀 RAG 文件。)

#### 手腕屈曲
- **fault_id**: `wrist_flexion_curl`
- **fault_name**: 手腕屈曲(用手腕在彎)(Wrist flexion, curling with the wrist)
- **description**: 手腕彎進屈曲來幫忙把負荷撐上去,而不是維持中立與剛性。
- **detection_heuristic**: `wrist_angle` = angle(肘 13/14 → 腕 15/16 向量,腕 → 食指 19/20 向量);約 180° = 打直。偏向屈曲 `> 30°` 就觸發。**僅為目前最好的替代指標**——見 observability。
- **observability**: low(任何視角)—— 手部關鍵點(食指/拇指)很小,常被啞鈴遮住,而手腕屈曲屬於小關節動作,單鏡頭姿態解析得很差;偵測結果一律視為低信心。
- **biomechanical_rationale**: 手腕屈曲會徵召腕屈肌並可能拉傷腕關節,同時把出力從肘屈肌轉走,降低肱二頭肌負荷。
- **citation**: Parpa K et al., *Muscles* (2025), PMC12550948, DOI 10.3390/muscles4040045.
- **citation_support**: 該論文提到彎舉 "involves elbow flexion accompanied by … wrist supination or pronation",且握法/腕位會影響屈肌徵召,而規定的姿勢是旋後、受控的握法。把手腕*屈曲*當成動作錯誤的支撐是間接的(只說明握法/腕位重要),而且它並非單鏡頭可觀測——就其具體的傷害量級標為 low/UNVERIFIED。(已查證來源確實討論腕/握法的影響;手腕屈曲的傷害風險量級在此來源中為 UNVERIFIED。)

---

### 手臂外展 Arm Abduction(站姿側平舉 / 肩外展)

反覆階段:**setup/bottom**(手臂內收於身側,`arm_elevation_angle`≈0°)→
**concentric**(外展/上舉)→ **top**(目標約 90°)→ **eccentric**(受控下放)。

#### 聳肩(上斜方肌主導)
- **fault_id**: `shoulder_shrug_elevation`
- **fault_name**: 聳肩 / 肩胛上抬(Shrugging / scapular elevation)
- **description**: 上舉過程中肩膀往耳朵方向聳起(上斜方肌主導),尤其在手臂通過約 90° 之後。
- **detection_heuristic**: 逐側計算 `neck_gap = ear_y (7/8) − shoulder_y (11/12)`,與 setup 基線的 `neck_gap` 比較。上舉期間 `neck_gap` 縮到低於基線 `> 18%`(肩膀往耳朵抬)就觸發;若同時 `arm_elevation_angle > 90°` 則升高嚴重度。**混淆因子**:在高舉位置,肩峰/肩膀本來就有一部分正常的肩肱節律性上抬——真正有鑑別力的錯誤訊號是*過早或不成比例*的聳肩(`arm_elevation_angle < 90°` 時 `neck_gap` 就塌下去),因此要對早期階段的聳肩加重權重,避免在乾淨的高舉動作上誤觸發。
- **observability**: high —— **front** 或 **rear** 視角(肩膀的垂直上抬落在平面內,解析清楚)。
- **biomechanical_rationale**: 上斜方肌持續過度活化、而下段肩胛穩定肌活性不足,會造成肩胛運動障礙,並提高肩峰下夾擠與肩肱關節不穩的風險。
- **citation**: Mun WL, Jung EY, Lei S, Roh SY, *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645.
- **citation_support**: "Persistent overactivity of the UT can lead to scapular dysfunction (or dyskinesia), such as subacromial impingement or glenohumeral instability",而上斜方肌活化 "consistently increases as the shoulder abduction angle surpasses 120°",因此 "care should be taken to avoid the excessive activation of the UT"。(已查證——讀自 RAG 文件。)

#### 舉過夾擠弧
- **fault_id**: `excessive_elevation_impingement_arc`
- **fault_name**: 舉超過安全/目標 ROM(夾擠弧)(Raising past safe/target ROM, impingement arc)
- **description**: 手臂被推進(並穿過)中段外展的疼痛弧,或明顯超過規定的目標高度,而肩胛控制不佳。
- **detection_heuristic**: `arm_elevation_angle` = 額狀面上 angle(肩→髖 vs 肩→肘)。若在約 70–120° 持續停留且同時出現聳肩(`shoulder_shrug_elevation` 為真),或舉到 `> target + 15°`(例如目標為 90° 時舉到 `>105°`),就觸發。
- **observability**: high —— **front** 視角(額狀面的上舉量測良好)。從 **side** 視角手臂與軀幹重疊 → low。
- **biomechanical_rationale**: 外展約 70–120° 之間肩峰下空間變窄,棘上肌/肱二頭肌長頭肌腱與肩峰下滑液囊受到擠壓(所謂疼痛弧);在肩胛上旋不足的情況下反覆負荷通過這段弧,有夾擠風險。
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026);由 Mun WL et al., *Medicina* (2025), PMC12029123 佐證。
- **citation_support**: StatPearls:疼痛弧出現在 "between approximately 70° and 120° of active shoulder abduction",該區間內肩峰下空間(正常 1–1.5 cm)"narrows physiologically with abduction",擠壓棘上肌腱、肱二頭肌長頭與肩峰下–三角肌下滑液囊。Mun 等人則佐證 120° 以上的上斜方肌/夾擠風險上升。(已查證——抓取 StatPearls + 讀 RAG 文件。)

#### 對側軀幹側傾
- **fault_id**: `contralateral_trunk_lean`
- **fault_name**: 軀幹倒向另一側(Trunk lean to the opposite side)
- **description**: 軀幹往離開工作手臂的方向側彎,藉此把手臂撐上去(額狀面代償)。
- **detection_heuristic**: `lateral_trunk_lean` = 額狀面上肩中點→髖中點向量相對影像垂直的角度(用肩中點與髖中點之間的 x 偏移計算)。若向心期往離開上舉手臂的方向側傾 `> 12°`,或整組隨負荷增加而擴大,就觸發。(單手上舉時,側傾的正負號相對工作側定義。)
- **observability**: high —— **front**/**rear** 視角(側傾落在平面內)。
- **biomechanical_rationale**: 對側側傾用軀幹側屈肌取代三角肌/肩胛的工作,降低目標負荷,也顯示肩部力量/控制不足;伴隨而來的肩胛力學不良是夾擠風險型態的一環。
- **citation**: Creech JA, Busse A, Li D, et al. *Shoulder Impingement Syndrome*, StatPearls (NCBI Bookshelf NBK554518, updated 2026).
- **citation_support**: StatPearls 把夾擠部分歸因於 "inadequate scapular upward rotation and posterior tilt"——亦即上舉過程中無法控制肩胛的代償,而對側軀幹側傾正是其中最粗放的一種。傷害機轉(上舉時肩胛控制不良造成夾擠)已由 StatPearls 查證。但外展過程中額狀面軀幹側屈這個特定的替代動作,在同儕審查來源中為 **UNVERIFIED**(讀過的來源沒有任何一篇單獨處理外展時的軀幹側屈;網路搜尋只找到健身教學類來源,不足以作為傷害風險的支撐)。(部分查證——傷害機轉已查證;軀幹側傾本身的肌電/運動學發現為 UNVERIFIED。)

#### 左右不對稱
- **fault_id**: `lr_abduction_asymmetry`
- **fault_name**: 左右不對稱(Left vs right asymmetry)
- **description**: 雙側上舉時一隻手落後、舉得比較少,或時序與另一隻不同。
- **detection_heuristic**: 左右比較:`asym = |arm_elevation_angle_L − arm_elevation_angle_R|`。若頂點持位時 `asym > 12°`,或腕高峰值相差 `> 0.05`(正規化單位)並跨多下持續出現,就觸發。
- **observability**: high —— **front**/**rear** 視角(雙臂都看得到,上舉落在平面內)。
- **biomechanical_rationale**: 肢段間不對稱反映力量/肩胛控制失衡;約 10–15% 這個區間的不對稱與傷害風險升高、表現下降有關。
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153.
- **citation_support**: 該論文指出 "asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance",並採用肢段對稱量表(不對稱 0–79%、臨界 80–89%、正常/對稱 90–100%)。(已查證——抓取 PMC 全文。)

---

### 手臂 VW Arm VW(肩胛 V 到 W 的前引/後收)

開放鏈的肩胛訓練:手臂過頭張成 **V(Y)**、肩胛上抬/上旋 → 把手肘往下往後拉成 **W**,同時肩胛
後收 + 下壓 → 短暫等長維持 → 回到 V。反覆階段:**V/前引-上抬** → **下拉/後收** →
**W 持位(等長)** → **回到 V**。

#### 肩胛 / 手臂行程不完整
- **fault_id**: `incomplete_scapular_rom`
- **fault_name**: 前引/後收行程不完整(Incomplete protraction/retraction excursion)
- **description**: 動作做得太淺——手臂/肩胛在 W 沒有到達完整的後收+下壓(或在 V 沒有到達完整上抬)。
- **detection_heuristic**: 用看得見的手臂行程當作(看不見的)肩胛移動的替代指標:各階段之間腕/肘的垂直移動 `excursion = wrist_y(V) − elbow_y(W)`,以及 W 時手肘下降到肩線的程度。若 V 與 W 之間的 `arm_elevation_angle` 擺幅 `< 40°`,或 W 時手肘沒有降到肩線的 `0.05`(正規化 y)以內,就觸發。真正的前後向肩胛後收沒有被直接量測(見 observability)。
- **observability**: **手臂上舉行程**為 medium(正面視角);真正的肩胛前引/後收為 **low**,那是前後向的深度動作,單鏡頭正面視角解析不出來。
- **biomechanical_rationale**: 肩胛行程愈大,斜方肌徵召愈多;行程被截短,這項訓練原本要練的中/下斜方肌就負荷不足(表現損失)。
- **citation**: Jung EY, Roh SY, Mun WL, *Life* (2025), PMC12734928, DOI 10.3390/life15121840.
- **citation_support**: 該研究發現行程較大的變化式(胸骨下沉,STD)"elicited higher trapezius activation, especially during large scapular excursions",且 "greater scapular excursion is known to increase muscle activation"(末端位置以標記點驗證)。(已查證——讀自 RAG 文件。)

#### 聳肩替代
- **fault_id**: `shrug_substitution`
- **fault_name**: 上斜方肌聳肩替代(Upper-trap shrug substitution)
- **description**: 上斜方肌接管(肩膀往耳朵聳起),而不是由下斜方肌/前鋸肌執行肩胛下壓與後收。
- **detection_heuristic**: `neck_gap = ear_y − shoulder_y` 與 setup 基線比較。在下拉/後收與 W 持位這兩個肩膀本來應該保持下壓的階段,若 `neck_gap` 縮到低於基線 `> 18%`(肩膀在抬)就觸發。**混淆因子**:V 階段本來就會合理地抬起肩膀(手臂過頭),所以這個判定只限用在預期要下壓的下拉/W 持位階段;有鑑別力的訊號是「該下壓時肩膀卻在上抬」,不是絕對的上抬量。
- **observability**: high —— **front**/**rear** 視角(肩膀的垂直上抬落在平面內)。
- **biomechanical_rationale**: 上斜方肌主導(UT/LT 與 UT/SA 活化比偏高)是肩胛運動障礙的失能型態,也讓下斜方肌/前鋸肌的訓練目的落空。
- **citation**: Abiara S et al., *PeerJ* (2025), PMC12335237, DOI 10.7717/peerj.19861;由 Jung EY et al., *Life* (2025), PMC12734928 佐證。
- **citation_support**: Abiara 等人:"ratios lower than 1.0 for the UT/LT ratio are preferred … although lower than 0.6 are ideal",而肩痛的特徵是 "characterized by increased activation of the upper trapezius and decreased activation of the lower trapezius and serratus anterior"。Jung 等人:"excessive UT dominance is linked to scapular dyskinesis",較低的 UT/SA 比則反映 "a more favorable stabilization pattern"。(已查證——兩份 RAG 文件皆已讀。)

#### 手臂上舉角度流失
- **fault_id**: `loss_of_elevation_angle`
- **fault_name**: V/W 目標上舉角度流失(Loss of target V/W elevation angle)
- **description**: 手臂在 V 掉到規定的上舉高度以下(或在 W 手肘掉得太低),離開下斜方肌的最佳位置。
- **detection_heuristic**: 逐側計算 `arm_elevation_angle`。若 V 階段峰值 `< 120°`(手臂舉得不夠高)或 W 階段外展 `< 75°`(手肘往身體塌陷),就觸發。拇指朝上/前臂朝向在單鏡頭下無法可靠量測,也不列為判定條件。
- **observability**: high —— **front** 視角(額狀面的上舉量測良好)。
- **biomechanical_rationale**: 下斜方肌活化在肩外展約 135° 附近最大(與其肌纖維方向一致);上舉角度流失會讓肩胛離開下斜方肌的最佳位置,削弱這項訓練的針對性效果。
- **citation**: Mun WL et al., *Medicina* (2025), PMC12029123, DOI 10.3390/medicina61040645;由 Abiara S et al., *PeerJ* (2025), PMC12335237 佐證。
- **citation_support**: Mun 等人:"the LT activation was the highest at a 135° shoulder abduction angle, with excessively high angles leading to a decrease",研究者並 "recommend shoulder abduction near 145°, aligning with the muscle fiber direction, for maximum LT activation"。Abiara 等人描述針對下斜方肌的訓練是以 "arms abducted above 90°, thumbs up" 執行。(已查證——兩份 RAG 文件皆已讀。)

#### 左右不對稱
- **fault_id**: `lr_vw_asymmetry`
- **fault_name**: 左右肩胛不對稱(Left vs right scapular asymmetry)
- **description**: V→W 循環中,一側的手臂/肩胛落後、位置偏低,或後收較少。
- **detection_heuristic**: 在 V 峰值與 W 持位時計算 `asym = |arm_elevation_angle_L − arm_elevation_angle_R|`。若 `asym > 12°`,或 `|wrist_y_L − wrist_y_R| > 0.05`(正規化)並跨多下持續出現,就觸發。
- **observability**: high —— **front**/**rear** 視角(雙臂都看得到,上舉落在平面內;但前後向的後收不對稱本身仍屬低可觀測性)。
- **biomechanical_rationale**: 肩胛控制不對稱反映左右穩定肌失衡;約 10–15% 的肢段間不對稱與傷害風險升高、表現下降有關。
- **citation**: Terré M, Solana-Tramunt M, *Healthcare (Basel)* (2025), 13(10):1153, PMC12110944, DOI 10.3390/healthcare13101153;肩胛運動障礙的脈絡引自 Jung EY et al., *Life* (2025), PMC12734928。
- **citation_support**: Terré & Solana-Tramunt:"asymmetries between 10% and 15% are often associated with a higher risk of injury and reduced performance"(肢段對稱量表:正常 90–100%)。Jung 等人把肩胛肌群活化失衡連結到肩胛運動障礙。(已查證——抓取 PMC 全文 + 讀 RAG 文件。)

---

### 查證 / 誠實揭露附註
- 以上所有 citation_support 字串都取自實際讀過的來源(四份 RAG 文件完整讀過;Havers 2025、
  StatPearls NBK554518 與 Terré 2025 為抓取後引述)。
- **UNVERIFIED / 部分查證**:(1) `wrist_flexion_curl`—— RAG 來源討論握法/腕位的影響,但沒有
  處理手腕屈曲的傷害量級;而且它也不是單鏡頭可觀測(observability 低)。(2)
  `contralateral_trunk_lean`—— 夾擠/肩胛控制的傷害機轉已查證(StatPearls),但讀過的同儕審查
  來源沒有任何一篇單獨處理外展時額狀面的軀幹側屈(網路搜尋只找到健身教學類來源,不足以作為
  傷害風險支撐,因此未列為引用)。
- 依賴深度的肩胛前引/後收(前後向動作)在單鏡頭正面姿態下本質上就是低可觀測性;VW 的啟發式
  退回用看得見的手臂上舉當替代指標,並已於文中說明。


---

## E 組 — 核心 / 復健 — 仰臥起坐、臀橋、腿部外展

動作:**Sit-up(仰臥起坐 / curl-up)**、**Shoulder Bridge(仰臥臀橋)**、**Leg Abduction(側躺 / 站姿髖外展)**。

偵測模型:MediaPipe Pose,33 個關鍵點,正規化影像座標(x, y, z, visibility),單鏡頭。關鍵點索引依共用脈絡(0 鼻;7/8 耳;11/12 肩;13/14 肘;15/16 腕;23/24 髖;25/26 膝;27/28 踝;29/30 腳跟;31/32 腳尖)。

以下使用的慣例:
- **軀幹屈曲角(Trunk-flexion angle)** = 肩中點→髖中點向量相對地面/水平的角度(0deg = 平躺,90deg = 完全坐直)。
- **髖角(Hip angle)** = 髖關鍵點處由 肩→髖→膝 構成的夾角(≈180deg = 軀幹與大腿成一直線;<180deg = 髖屈曲;>180deg = 髖過度伸展/拱起)。
- **骨盆傾斜(額狀面)(Pelvic-tilt, frontal)** = 左髖(23)→右髖(24)連線相對水平的帶號角度。
- **軀幹側傾(Trunk lateral-lean)** = 肩中點相對髖中點的水平偏移,以軀幹長度正規化。

---

### 仰臥起坐 Sit-up(curl-up)

反覆階段:**setup(仰臥)** → **concentric trunk flexion(捲起)** → **top(頂點)** → **eccentric return(下放)** → **rest(休息)**。

#### excessive_speed_trunk_control_loss

- **fault_id**: `situp_excessive_speed`
- **fault_name**: 速度過快 / 軀幹控制失守(Excessive speed / loss of trunk control)
- **description**: 捲起被快速、急促、彈道式地甩上來,而不是緩慢受控地抬起,軀幹還會偏離矢狀線晃動。
- **detection_heuristic**: 向心階段時長(從離地到頂點的影格數)與軀幹屈曲角的角速度峰值。若向心階段 < 約 1.0 秒(大致是實驗中測過的最快節奏),或 |d(trunk_angle)/dt| 峰值大幅超過該使用者的基線,就觸發;次要訊號 = 逐幀加速度(jerk)出現高尖峰,以及肩中點的內外側晃動增加(相對矢狀路徑的 x 變異數)。
- **observability**: medium —— 速度/ROM 需要 **side** 視角;內外側晃動需要 **front**/**front_oblique**。絕對速度可以量測;「控制失守」則是替代指標。
- **biomechanical_rationale**: 快速捲腹會增加軀幹角動量與脊椎負荷/椎間盤內壓,並壓縮神經肌肉修正的可用時間,因此對於動作控制缺損或下背問題的人應謹慎使用。
- **citation**: Barbado D, Moreno-Navarro P, Vera-Garcia FJ, et al. "Effect of Performance Speed on Trunk Movement Control During the Curl-Up Exercise." J Hum Kinet (2015). PMC4519219, DOI 10.1515/hukin-2015-0031.
- **citation_support**: "the linear variability of COP_ML significantly increased as curl-up exercise speed increased",而受試者 "performed a greater neuromuscular effort to control trunk motion during the fastest curl-up exercises";該論文並指出 "due to the effect of performance speed on the spinal loads and intradiscal pressure, fast curl-up exercises should be used with caution in people with motor control deficits or low-back disorders, as well as in novice, untrained or unfit individuals",速度愈快角動量愈大,且 "greater difficulty to slow down the trunk flexion motion"。VERIFIED(已讀 RAG 文件)。

#### hip_flexor_dominance_anchored

- **fault_id**: `situp_hip_flexor_dominance`
- **fault_name**: 髖屈肌主導(固定雙腳 / 全身打直硬拉)(Hip-flexor dominance, anchored feet / rigid straight-body pull)
- **description**: 軀幹以剛性直桿的方式繞著髖關節轉上來(脊椎沒有逐節捲曲),雙腳被固定住,整個動作由髖屈肌驅動,而不是靠腹肌的分節軀幹屈曲。
- **detection_heuristic**: 以向心階段中 肩→髖→膝 共線性的變化來量測軀幹「捲曲」程度。若脊椎維持接近直線(肩–髖–膝 仍近乎共線,trunk_curl 變化 < 約 10-15deg)而軀幹–大腿(髖)夾角卻快速收合——亦即骨盆固定下的剛體旋轉——就觸發。輔助替代指標:雙腳/腳跟(29/30、31/32)維持不動(位移很小)且膝蓋沒有抬起,代表雙腳被固定。
- **observability**: medium —— 需要 **side** 視角。真正的肌肉徵召看不到;剛性 vs 分節的軀幹動作是可觀測的替代指標。
- **biomechanical_rationale**: 固定雙腳、直腿拉起會徵召髂腰肌/股直肌,增加腰椎前凸與不均勻的椎間盤負荷,同時讓腹肌活化不足,既違背這個動作的目的,也拉高下背壓力。
- **citation**: Mandroukas A, Michailidis Y, Metaxas T. "Surface Electromyographic Activity of the Rectus Abdominis and External Oblique during Isometric and Dynamic Exercises." J Funct Morphol Kinesiol (2022). PMC9505236, DOI 10.3390/jfmk7030067.
- **citation_support**: "support on the feet activates the hip flexors and reduces the activity of the abdominal muscles";建議 curl-up 應 "with flexed unsupported knees, without holding the knees or feet ... to isolate the activity of the hip flexors",而由 "the hip flexors, particularly by the iliopsoas, rectus femoris, and sartorius ... increases lordosis in the lumbar spine"。快速、剛性的起始也 "did not [give] enough time for the abdominal muscles to contract"。VERIFIED(已讀 RAG 文件)。

#### excessive_rom_full_situp

- **fault_id**: `situp_excessive_rom`
- **fault_name**: ROM 過大(超出 curl-up 範圍的完整仰臥起坐)(Excessive ROM, full sit-up past the curl-up range)
- **description**: 軀幹越過部分捲腹繼續往上,做成完整坐起,把整段腰椎都抬離地面,而不是在肩胛離地後就停。
- **detection_heuristic**: 頂點的軀幹屈曲角峰值。正確的 curl-up 只讓肩胛剛好離地,軀幹相對地面只到約 35-40deg;若軀幹屈曲角峰值 > 約 50-60deg(接近完整坐姿),或髖角(肩–髖–膝)收到 < 約 110deg,代表做成了軀幹貼大腿的完整仰臥起坐,就判為 `excessive_rom`。
- **observability**: high —— **side** 視角;軀幹屈曲角在矢狀面可直接量測。
- **biomechanical_rationale**: 完整仰臥起坐會明顯拉高腰椎椎間盤壓力(文獻記錄在 L3),而把軀幹屈曲限制在約 35-40deg 可讓腰椎留在地面、降低椎間盤負荷,同時仍能訓練腹肌——所以做過頭是拿安全換取極有限的額外好處。
- **citation**: Mandroukas A, Michailidis Y, Metaxas T. J Funct Morphol Kinesiol (2022). PMC9505236, DOI 10.3390/jfmk7030067.
- **citation_support**: "The stress placed on the lumbar spine decreases by limiting the amount of trunk flexion to 35-40deg ... curl-ups performed through a partial range may be an effective method of gaining abdominal muscle strength, while protecting the lumbar spine",以及 "Nachemson reported increased pressure on the intervertebral disc at the level of L3 during the execution of full sit-ups"。VERIFIED(已讀 RAG 文件)。

#### incomplete_rom_scapula

- **fault_id**: `situp_incomplete_rom`
- **fault_name**: ROM 不完整(肩胛沒有離地)(Incomplete ROM, scapulae not lifted)
- **description**: 頭/頸抬起來了,但肩膀/肩胛幾乎沒離地,實際上幾乎沒有軀幹屈曲。
- **detection_heuristic**: 頂點的軀幹屈曲角峰值。若軀幹屈曲角峰值 < 約 20deg(肩中點相對髖幾乎沒上升;肩胛未離地)就判為 `incomplete_rom`。要與純頭部動作區分,檢查肩中點的垂直位移——而不只是鼻(0)的位移——是否維持很小。
- **observability**: high —— **side** 視角;肩相對髖在矢狀面的位移可直接量測。
- **biomechanical_rationale**: curl-up 的定義就是抬到 "to the point where the scapula was lifted";肩胛若始終沒離地,目標腹肌行程沒達到,訓練刺激也就消失(這屬於表現/有效性的錯誤,而非傷害)。
- **citation**: Barbado D et al. J Hum Kinet (2015). PMC4519219, DOI 10.1515/hukin-2015-0031;由 Mandroukas A et al. PMC9505236 佐證。
- **citation_support**: Barbado 把 curl-up 定義為 "a head, arms and upper trunk lift to the point where the scapula was lifted from the force plate, then returning to the starting position";Mandroukas:curl-up 是 "with a rounded back to approximately 35-40deg from the floor" 的抬起。肩胛離地這個終點提供了可觀測的 ROM 目標。VERIFIED(已讀 RAG 文件)。

---

### 臀橋 Shoulder Bridge(仰臥橋式)

反覆階段:**setup(仰臥,髖/膝屈曲,雙腳踩平)** → **concentric hip extension(抬起骨盆)** → **top(等長維持)** → **eccentric lower(下放)** → **rest**。

#### incomplete_hip_extension_top

- **fault_id**: `bridge_incomplete_hip_extension`
- **fault_name**: 頂點髖伸展不完全(Incomplete hip extension at top)
- **description**: 頂點時骨盆抬得不夠高,肩、髖、膝沒有連成一直線,髖仍屈曲下沉。
- **detection_heuristic**: 抬到頂點時的髖角(肩→髖→膝)。目標是一直線(約 170-180deg,髖屈曲 0deg);若髖角峰值 < 約 160deg(髖明顯仍屈曲/骨盆偏低)就判為 `incomplete_extension`。取左右平均;峰值取自頂點維持的那幾幀。
- **observability**: high —— **side** 視角;肩–髖–膝夾角在矢狀面可直接量測。
- **biomechanical_rationale**: 臀大肌徵召與髖伸展力矩在接近完全髖伸展時最大,所以停在半路會讓臀肌負荷不足,失去這個動作的意義。
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. "Supine Bridge Exercise: A Narrative Review of the Literature (Part I)." Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349;終點由 Escamilla RF et al. Bioengineering (2024). PMC11048684, DOI 10.3390/bioengineering11040356 佐證。
- **citation_support**: Colonna:"the pelvis is lifted from the floor until it reaches the neutral angular position of the hip",以及 "The greatest hip extension torque during the SBE occurs when the hip is nearly fully extended. In this position, the GM is recruited more than at any other angle within the range of motion"。Escamilla 把終點定義為抬到 "until the hips were in a neutral position with 0deg hip flexion, with the knees, hips, and shoulders approximately in a straight line"。VERIFIED(已讀 RAG 文件)。

#### lumbar_hyperextension_overarch

- **fault_id**: `bridge_lumbar_hyperextension`
- **fault_name**: 腰椎過度伸展 / 拱過頭(Lumbar hyperextension / overarching)
- **description**: 骨盆被推得太高,下背拱起(骨盆前傾/腰椎前凸),而不是讓肩–髖–大腿保持一直線——髖伸展被背伸展取代。
- **detection_heuristic**: 頂點時的髖角(肩→髖→膝);若髖角峰值越過直線繼續進入伸展(> 約 190deg,亦即髖高過肩–膝連線而拱起)就判為 `lumbar_hyperextension`。輔助替代指標:頂點時髖中點的 y 高過肩中點與膝中點之間內插的直線。
- **observability**: medium —— **side** 視角。整體的髖/背伸展看得到,但要從體表關鍵點分離腰椎與髖的貢獻只能近似,所以這是拱背的替代指標。
- **biomechanical_rationale**: 豎脊肌主導所造成的過度、不受控的腰椎前凸與骨盆前傾,會增加腰部與骨盆區域的壓縮應力,反覆下來可能導致次發性功能失調。
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "In patients performing bridging exercises, excessive and uncontrolled lumbar lordosis and anterior pelvic tilt (APT) are frequently observed due to the dominant hyperactivity of the ES. The repetitive motion associated with this activity could increase compression stress on the lumbar and pelvic regions."有些作者 "recommend maintaining a straight alignment of the shoulders, hips, and thighs during bridging to prevent excessive APT caused by dominant ES activity"。VERIFIED(已讀 RAG 文件)。

#### asymmetric_pelvic_drop

- **fault_id**: `bridge_asymmetric_pelvic_drop`
- **fault_name**: 骨盆不對稱下掉(尤其單腳橋式)(Asymmetric pelvic drop, esp. single-leg bridge)
- **description**: 維持期間骨盆一側相對另一側下沉——類似 Trendelenburg 的額狀面傾斜,在單腳變化式最常見。
- **detection_heuristic**: 額狀面骨盆傾角 = 左髖(23)→右髖(24)連線相對水平的角度,取頂點維持的那幾幀。若 |骨盆傾角| 超過約 8-10deg(或相對 setup 基線出現大幅不對稱)就判為 `pelvic_drop`。單腳時骨盆通常往無支撐(擺動腿)那側掉。
- **observability**: medium —— 需要 **front** 或 **rear** 視角(額狀面傾斜);純側面視角幾乎看不到。front/rear_oblique 部分可用。
- **biomechanical_rationale**: 單腳支撐時,臀中肌必須產生髖外展力量把骨盆維持水平;失敗時骨盆會往對側掉(Trendelenburg),穩定性下降,長期還會拉高下背與下肢的負荷。
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "In a Trendelenburg gait, the Gmed is unable to maintain the pelvis on the opposite side during single-leg support, causing the pelvis to drop when the swing leg is in the air. This pelvic drop occurs when the Gmed fails to generate enough of an internal hip abduction force to counteract the external hip adduction force that happens during single-leg stance."VERIFIED(已讀 RAG 文件)。

#### knee_valgus_bridge

- **fault_id**: `bridge_knee_valgus`
- **fault_name**: 膝外翻(膝蓋往內塌)(Knee valgus, knees collapse inward)
- **description**: 抬起過程中膝蓋相對雙腳往內側(彼此靠攏)漂移,而不是對齊腳掌前進。
- **detection_heuristic**: 在額狀面比較膝距與踝/足距:`knee_width/ankle_width`,其中 knee_width = |x(25)-x(26)|、ankle_width = |x(27)-x(28)|。抬起過程中此比值掉到約 0.85 以下(膝比腳還靠近彼此)就判為 `knee_valgus`——比照深蹲的膝內夾啟發式。
- **observability**: medium —— 需要 **front**/**front_oblique** 視角;側面看不到。
- **biomechanical_rationale**: 髖外展肌/外旋肌無力會允許過度的髖內收與內轉,產生膝外翻,增加膝關節韌帶結構的應力。
- **citation**: Colonna S, D'Alessandro A, Tarozzi R, Casacci F. Cureus (2025). PMC11981018, DOI 10.7759/cureus.80349.
- **citation_support**: "Powers theorized that hip abductor and external rotator weakness may lead to excessive hip adduction and internal rotation, resulting in increased knee valgus. This position can place excessive stress on the knee's ligamentous structures."VERIFIED(已讀 RAG 文件)。

---

### 腿部外展 Leg Abduction(側躺 / 站姿髖外展)

反覆階段:**setup(髖中立)** → **concentric abduction(抬腿/側跨)** → **peak abduction(外展峰值)** → **eccentric return(內收回位)** → **rest**。

#### pelvic_drop_trunk_lean_compensation

- **fault_id**: `abd_pelvic_drop_trunk_lean`
- **fault_name**: 骨盆下掉 / 軀幹側傾代償(類 Trendelenburg)(Pelvic drop / trunk lateral-lean compensation)
- **description**: 不是乾淨的孤立外展,而是骨盆下掉或軀幹側傾,用抬/傾骨盆的方式把腿偷渡出去。
- **detection_heuristic**: 兩個耦合訊號:(1) 額狀面骨盆傾角 = 左髖(23)→右髖(24)連線相對水平的角度;(2) 軀幹側傾 = 肩中點(11,12)相對髖中點(23,24)的水平偏移,以軀幹長度正規化。外展階段中骨盆傾角相對 setup 變化 > 約 8-10deg,或側傾超過軀幹長度的約 0.10-0.15,就觸發。
- **observability**: **站姿**髖外展從 **front**/**rear** 視角為 medium/high(額狀面正對相機);**側躺**為 medium,軀幹側傾從 **side** 視角看得到,但骨盆下掉大致落在平面外。
- **biomechanical_rationale**: 髖外展肌無力時,身體會用對側骨盆下掉以及/或同側軀幹側傾來代償,這既卸掉外展肌的負荷、讓訓練失效,也反映出造成 Trendelenburg 步態的同一機轉。
- **citation**: González-de-la-Flor Á. "Optimizing Hip Abductor Strengthening ... Monster Walk and Lateral Band Walk." J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294;由 Rodrigues R et al. PLoS One (2025). PMC12416692, DOI 10.1371/journal.pone.0331553 佐證。
- **citation_support**: González-de-la-Flor:"weakness leads to a characteristic Trendelenburg gait or compensatory trunk lean",以及 "excessive sway or lateral trunk lean may reduce abductor demand by mechanically offloading the stance limb ... The optimal technique involves maintaining frontal plane neutrality"。Rodrigues:髖外展肌無力會被 "by increasing ipsilateral trunk lean" 代償,而單腳任務中較大的骨盆下掉與較大的髖內收有關。VERIFIED(已讀 RAG 文件)。

#### hip_flexion_external_rotation_substitution

- **fault_id**: `abd_hip_flexion_er_substitution`
- **fault_name**: 髖屈曲 + 外轉替代(Hip flexion + external rotation substitution)
- **description**: 腿往前飄(髖屈曲)以及/或腳尖朝上外轉——徵召闊筋膜張肌/髖屈肌——而不是在髖中立下做純額狀面的外展。
- **detection_heuristic**: 前飄替代指標(側面視角):外展過程中腳尖(31/32)/踝(27/28)的 x 位置越過髖的矢狀線往前超過一個小容差(腳跑到髖前面),而不是純粹往側向/垂直移動。外轉替代指標:足向量(踝→腳尖)朝向的變化,顯示腳尖往上/往外轉。當前飄量相對純外展行程偏大時觸發。
- **observability**: low/medium —— 前飄(髖屈曲)成分需要 **side** 視角;由腳尖關鍵點推的外轉在單鏡頭姿態下是很弱、很吵的替代指標,不宜過度信任。單靠 **front** 視角無法可靠地把它與真正的外展分開。
- **biomechanical_rationale**: 用髖屈曲與旋轉來替代,會把負荷從臀中肌轉到闊筋膜張肌與髖屈肌,削弱目標臀肌徵召並強化錯誤的動作模式;要選擇性地負荷臀肌,就必須維持額狀面中立。
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: 該回顧強調 "maintaining frontal plane neutrality" 與促成 "gluteal over TFL recruitment" 的提示,指出闊筋膜張肌(髖屈肌/外展肌)是代償肌,而股骨扭轉/旋轉與姿勢會 "influence movement quality";彈力帶放在遠端則 "introduces a slight external rotation torque"。前飄/腳尖朝上這個特定替代動作,是額狀面中立喪失的臨床表現。中立/闊筋膜張肌原則為 VERIFIED;腳尖朝上這個確切提示屬推得的臨床描述(支撐度標為 MODERATE)。

#### insufficient_abduction_rom

- **fault_id**: `abd_insufficient_rom`
- **fault_name**: 外展 ROM 不足(Insufficient abduction ROM)
- **description**: 腿只抬起/跨出一點點,離完整的外展行程還很遠。
- **detection_heuristic**: 外展角峰值 = 額狀面上大腿向量(髖 23/24 → 膝 25/26)相對骨盆中線/垂直的角度。站姿/側躺外展的外展角峰值 < 約 25-30deg(常見目標範圍約 30-45deg)就判為 `insufficient_rom`。彈力帶橫走則改用步寬(踝間距)是否低於該使用者的 setup 門檻。
- **observability**: medium/high —— 站姿外展用 **front**/**rear** 視角;側躺用 **side** 視角(抬腿所在平面正好側對相機)。
- **biomechanical_rationale**: 側躺髖外展在其行程中產生高度的臀中肌活化(約 80% MVIC);把行程做短會降低訓練刺激,也削弱支撐骨盆穩定與傷害預防的強化效果。
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: "side-lying hip abduction ... generates high levels of gluteus medius activation, reaching approximately 80% of maximal voluntary isometric contraction (MVIC) ... greater muscle activation than other closed-chain or multi-joint exercises such as clamshells, lunges, and hops";該回顧並強調 "optimal squat depth" 與足夠的行程才能有效負荷。VERIFIED(已讀 RAG 文件)。註:那個具體的角度門檻是實務目標,不是來源明載的數值。

#### momentum_uncontrolled

- **fault_id**: `abd_momentum`
- **fault_name**: 借力 / 不受控的擺盪(Using momentum / uncontrolled swing)
- **description**: 腿是被彈道式甩出去再彈回來,而不是受控地抬起與放下。
- **detection_heuristic**: 大腿(髖→膝)外展角的角速度峰值,以及向心與離心兩階段之間的對稱性;另看頂點處的 jerk(逐幀加速度)尖峰。若角速度峰值遠超該使用者的基線,或離心(回位)階段比向心快很多(腿是被甩/被放掉的),就判為 `momentum`。
- **observability**: medium —— 站姿用 **front**/**rear**,側躺用 **side**;速度可量測,但「借力」這個判斷是從運動學推得的替代指標。
- **biomechanical_rationale**: 靠動量甩動會讓非目標組織與重力代替臀肌做功,削弱強化刺激,也拿掉這個動作原本要訓練的控制成分。
- **citation**: González-de-la-Flor Á. J Funct Morphol Kinesiol (2025). PMC12372021, DOI 10.3390/jfmk10030294.
- **citation_support**: "Proper execution requires control of the trunk and pelvis, optimal squat depth, and consistent band tension",技術指引並強調 "controlled steps" 以確保 "the targeted muscles are effectively engaged"。控制的要求有明確支撐;具體的速度門檻則是實務替代值(支撐度 MODERATE)。VERIFIED(已讀 RAG 文件)。

---

### 查證附註

以上所有 citation_support 的引述/改寫都取自本次工作中實際讀過的六份 RAG 文件(PMC4519219、PMC9505236、PMC11981018、PMC11048684、PMC12372021、PMC12416692)。有兩項在臨床提示超出來源字面說法之處,如實降級為 MODERATE 支撐:`abd_hip_flexion_er_substitution` 的「腳尖朝上外轉」成分,以及 `abd_momentum` 的速度門檻(來源分別支撐額狀面中立與動作控制的原則,但沒有明訂具體的姿態門檻)。沒有任何一條動作錯誤是建立在無支撐的傷害風險主張上。


---

## F 組 — 動態 / 旋轉 — 軀幹旋轉、開合跳、高抬腿

動作:**Torso Twist(軀幹旋轉,坐姿俄羅斯轉體 / 站姿轉體)**、**Jumping Jacks(開合跳,side-straddle hop)**、**High Knee(高抬腿,跑步訓練 / 原地踏步)**。

偵測模型:MediaPipe Pose,33 個關鍵點,正規化影像座標(x,y ∈ [0,1],y 往下遞增),單鏡頭。關鍵點索引依共用脈絡。當某個動作錯誤無法乾淨地由單鏡頭觀測時,會給出最好的幾何替代指標,並如實下修可觀測性。

來源附註:這三個動作的同儕審查 RAG 覆蓋很薄或掛零,因此以下每筆引用都是透過網路搜尋找到,並實際抓取來源頁面(PubMed / PMC / 期刊)查證具體發現。Wikipedia 只作*描述性*的補充引用,絕不單獨支撐傷害風險主張。

---

### 軀幹旋轉 Torso Twist

反覆階段:**center(繃緊的準備位)→ rotate to side A(轉向 A 側峰值)→ return through center(回中)→ rotate to side B(轉向 B 側峰值)→ center**。每次單側擺動 = 一下。坐姿俄羅斯轉體的幾何:臀部固定在地、膝蓋彎曲、軀幹維持離地約 45°;旋轉應該來自胸椎,腰椎則保持繃緊。站姿變化式則維持軀幹垂直。

#### tt_lumbar_rotation_dominant
- **fault_id**: `tt_lumbar_rotation_dominant`
- **fault_name**: 用下背而非胸椎旋轉(Rotating from the lower back instead of the thoracic spine)
- **description**: 骨盆/髖線跟著肩膀一起轉(整個軀幹一起扭),或者背部在旋轉負荷下拱起,使扭轉是靠腰椎而非上軀幹驅動。
- **detection_heuristic**: 比較整段擺動中肩線(11→12 向量)與髖線(23→24 向量)的旋轉量。在正面/斜側視角,扭轉是以成對關鍵點投影水平間距的變化(|x11−x12| 與 |x23−x24|)加上左右 x 排序翻轉來讀取。當髖線旋轉量 ≥ 約 0.6 × 肩線旋轉量(骨盆跟著軀幹轉而非固定不動),或整下之中肩中點相對髖中點以漸增的幅度下沉/前移(脊椎拱起),就觸發。僅為替代指標——33 個稀疏關鍵點無法解析真正的胸椎與腰椎分段。
- **observability**: low–medium;需要正面或 front_oblique / rear_oblique 視角。純側面視角不可靠(旋轉是進出影像平面的動作)。
- **biomechanical_rationale**: 軸向旋轉下,軀幹肌群共同活化主要是為了*穩定*腰椎節段,而不是產生力矩;讓旋轉塌成拱背扭腰,等於拿掉那層保護性支撐,把扭轉負荷導向被動的椎間盤與小面關節結構(這是公認的扭轉傷害途徑)。
- **citation**: McGill, S.M. (1991). "Electromyographic activity of the abdominal and low back musculature during the generation of isometric and dynamic axial trunk torque: implications for lumbar mechanics." *Journal of Orthopaedic Research* 9(1):91–103. PMID 1824571. https://pubmed.ncbi.nlm.nih.gov/1824571/
- **citation_support**: VERIFIED(已抓取)。最大軸向力矩出力時,腹斜肌是主導的腹部作用肌(腹外斜肌 52%、腹內斜肌 55% MVC,腹直肌僅 22%),該論文並下結論 "stabilization of the joints during twisting is far more important to the lumbar spine than production of large levels of axial torque",而分析模型嚴重低估力矩(預測 14 Nm、實測 91 Nm)——亦即扭轉是穩定性問題,而扭轉負荷對纖維環/小面關節是傷害機轉。

#### tt_trunk_not_braced
- **fault_id**: `tt_trunk_not_braced`
- **fault_name**: 失去繃緊直立的軀幹(塌陷/拱背)(Losing the braced upright torso)
- **description**: 軀幹離開它該維持的約 45°(坐姿)或垂直(站姿)位置——通常是脊椎拱起,或在兩次擺動之間往地面癱下去。
- **detection_heuristic**: 追蹤軀幹向量 髖中點→肩中點 相對垂直的角度。在第一個繃緊的影格建立 setup 基線。當軀幹角偏離基線 > 約 15°,或(坐姿、側面視角)肩中點的 y 掉出 setup 帶往地面靠,代表支撐鬆掉,就觸發。可再結合脊椎拱起的替代指標(側面視角下肩中點在 x 上跑到髖中點前方)。
- **observability**: medium;坐姿 45° 維持以側面視角最佳,站姿變化式則正面/側面皆可。
- **biomechanical_rationale**: 維持軀幹繃緊可以保留旋轉時穩定腰椎節段的共同活化(見上述 McGill);在旋轉/慣性負荷下拱背或塌陷,會把扭轉應力轉到被動的椎間盤/韌帶組織上。
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571(同上)。描述性補充:Wikipedia, "Russian twist" (CC BY-SA) —— `data/rag/docs/torso_twist_russian_wiki.txt`。
- **citation_support**: VERIFIED。McGill(已抓取)支撐穩定性的理由。RAG 的 Wikipedia 文件(已讀)只提供技術目標:"the torso is kept straight with the back kept off the ground at a 45-degree angle"——這是對直立支撐幾何的描述性支撐,不是傷害主張。

#### tt_insufficient_rotation_rom
- **fault_id**: `tt_insufficient_rotation_rom`
- **fault_name**: 旋轉活動範圍不足(Insufficient rotation range of motion)
- **description**: 扭轉太淺——雙手/肩膀幾乎沒有越過身體中線,腹斜肌並未通過有意義的旋轉行程。
- **detection_heuristic**: 量測每次單側峰值時腕中點(15,16)相對髖中線 x(23,24 的平均)的水平行程峰值,以及/或肩線旋轉峰值。若某一下腕中點在該側越過髖中線 x 的量不超過一個小帶寬(例如 |x_wrist_mid − x_hip_mid| < 肩寬的約 0.08),代表旋轉極小,就標記該下。正面 / front_oblique 視角。
- **observability**: medium;正面或斜側視角(旋轉從側面投影很差)。
- **biomechanical_rationale**: 俄羅斯轉體的處方目的就是透過軀幹旋轉負荷腹內/外斜肌;ROM 被截短會讓這些主導旋轉與軀幹穩定的肌群徵召不足,削弱訓練刺激(屬表現損失而非傷害)。
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571. https://pubmed.ncbi.nlm.nih.gov/1824571/
- **citation_support**: VERIFIED(已抓取)。McGill 確立腹斜肌是軸向力矩產生時主導的腹部旋轉肌(腹外斜肌 52%、腹內斜肌 55% MVC)——扭轉 ROM 被截短,負荷不足的正是這些肌群。(Escamilla et al. 2006, *Phys Ther* 86(5):656–671, PMID 16649890 已抓取並檢查過,但它處理的是腹部*屈曲*運動而非旋轉,因此刻意不列為引用。)

#### tt_momentum_over_control
- **fault_id**: `tt_momentum_over_control`
- **fault_name**: 用動量甩而非受控旋轉(Swinging with momentum instead of controlled rotation)
- **description**: 手臂/重物被彈道式地左右甩、中間不停頓,驅動扭轉的是動量而不是肌肉控制。
- **detection_heuristic**: 時序訊號。逐幀計算腕中點繞髖中點的角速度;若某一下的角速度峰值超過節奏門檻,以及/或在單側峰值處沒有出現接近零速度的停留(沒有控制性的停頓),就標記。可再結合每秒下數超過設定上限。
- **observability**: medium;任何解析得出擺動的視角(正面/斜側)。這是節奏啟發式,不是單幀幾何。
- **biomechanical_rationale**: 彈道式動量會拉高扭轉負荷峰值,並把它在末端行程轉嫁到被動組織上,同時縮短腹斜肌的張力時間;受控節奏才能維持穩定性的共同活化。這條同時放在控制與傷害兩個面向。
- **citation**: McGill, S.M. (1991), *J Orthop Res* 9(1):91–103, PMID 1824571;描述性補充:Wikipedia "Russian twist"(`data/rag/docs/torso_twist_russian_wiki.txt`)。
- **citation_support**: VERIFIED(描述性)+ VERIFIED(機轉)。RAG 文件(已讀)明確指出 "The slower one moves the arms from side to side, the harder the exercise becomes",並警告不要依賴每下之間的動量——這是對受控節奏的直接描述性支撐。McGill 則提供穩定性/扭轉的理由。

---

### 開合跳 Jumping Jacks

反覆階段:**closed(雙腳併攏、手臂在身側)→ open(雙腳大幅張開、手臂過頭)→ landing back to closed(落回併攏)**。每次觸地都是衝擊/落地事件(張開的觸地與回到併攏的觸地都算)。關鍵點:肩 11/12、腕 15/16、髖 23/24、膝 25/26、踝 27/28、鼻 0。

#### jj_knee_valgus_landing
- **fault_id**: `jj_knee_valgus_landing`
- **fault_name**: 落地時膝外翻(內側塌陷)(Knee valgus on landing)
- **description**: 觸地時膝蓋相對踝/腳往內塌(額狀面的膝塌陷)。
- **detection_heuristic**: 在落地那一幀計算 knee-width/ankle-width = |x25−x26| / |x27−x28|。比值 < 約 0.82(比照已寫進程式碼的深蹲膝內夾規則)就判為外翻,亦即膝蓋被拉進踝底面之內。可另確認每側 knee_x 落在同側 ankle_x 的內側。正面視角。
- **observability**: high;需要正面(或 front_oblique)視角。純側面視角看不到。
- **biomechanical_rationale**: 以動態膝外翻落地,會把衝擊吸收從髖轉嫁到膝,拉高膝關節負荷與隨之而來的(ACL/髕股)傷害風險。
- **citation**: Tamura, A. et al. (2017). "Dynamic knee valgus alignment influences impact attenuation in the lower extremity during the deceleration phase of a single-leg landing." *PLoS ONE* 12(6):e0179810. https://pmc.ncbi.nlm.nih.gov/articles/PMC5478135/
- **citation_support**: VERIFIED(已抓取)。外翻落地者的膝角衝量顯著較大(0.093 vs 0.045 Nms/kg·m,p<0.01),髖角衝量較小(0.019 vs 0.067,p<0.01),相對於內翻/中立落地者;該論文下結論指出外翻 "may increase the impact the knee joint needs to attenuate",把負荷從髖移向膝——這是已被記錄的傷害風險型態。

#### jj_stiff_landing
- **fault_id**: `jj_stiff_landing`
- **fault_name**: 硬式落地、膝屈曲不足(Stiff, hard landing with insufficient knee flexion)
- **description**: 以近乎打直的腿落地、膝幾乎不彎,造成又硬又高衝擊的觸地,而不是靠屈曲吸收。
- **detection_heuristic**: 在每次觸地當下與稍後計算膝角(23→25→27 與 24→26→28)。若整個衝擊窗口內膝屈曲峰值都很淺——膝角維持 > 約 160°(即最大屈曲 < 約 20°)——或從觸地到最低點的膝屈曲行程低於設定帶寬,就判為硬式落地。側面或 front_oblique 視角能取得最乾淨的膝角;正面視角只是近似。
- **observability**: medium–high;以 side / oblique 最佳。純額狀面視角會低估矢狀面的膝屈曲。
- **biomechanical_rationale**: 硬式落地產生的地面反作用力遠大於軟式(屈曲較深)落地,把衝擊集中到關節與被動組織上;較大的膝/髖屈曲則讓髖與膝的大肌群得以耗散落地能量。
- **citation**: DeVita, P. & Skelly, W.A. (1992). "Effect of landing stiffness on joint kinetics and energetics in the lower extremity." *Medicine & Science in Sports & Exercise* 24(1):108–115. PMID 1548984. https://pubmed.ncbi.nlm.nih.gov/1548984/
- **citation_support**: VERIFIED(已抓取)。軟式與硬式落地的膝屈曲平均為 117° vs 77°,而 "the stiff landing had larger GRFs";軟式落地時髖與膝的肌群是主要的能量吸收者,硬式落地則由踝部肌群主導。以膝屈曲 ≥/< 90° 區分軟硬落地的慣例即出自此。

#### jj_incomplete_arm_rom
- **fault_id**: `jj_incomplete_arm_rom`
- **fault_name**: 手臂活動範圍不完整(Incomplete arm range of motion)
- **description**: 張開階段手臂沒有完全舉到頭頂上方(雙手停在肩/頭高度附近)。
- **detection_heuristic**: 在張開階段的峰值,比較腕高與頭部:若兩隻手腕都沒有升到鼻子上方(y15 與 y16 > y0,注意 y 往下遞增)或某條以肩為基準的線之上,就觸發。正面視角。
- **observability**: high;正面或 front_oblique 視角。
- **biomechanical_rationale**: 手臂完整過頭是這個動作(side-straddle hop)的定義性 ROM;做不完整會減少肩部/心肺的工作量——大致屬於表現/完整度層面的錯誤。(註:RAG 文件也記載反覆做完整過頭的開合跳曾被連結到旋轉肌袖刺激,這正是縮減 ROM 的「半開合跳」存在的原因——所以硬拚極端 ROM 未必更安全;這條規則針對的是明顯不完整的那些下。)
- **citation**: Wikipedia, "Jumping jack" (CC BY-SA) —— `data/rag/docs/jumping_jacks_wiki.txt`。
- **citation_support**: VERIFIED(描述性,已讀)。"The hands go overhead, sometimes in a clap, and then return to a position with the feet together and the arms at the sides" 定義了完整 ROM 目標;半開合跳的段落則指出它們 "were created to prevent rotator cuff injuries, which have been linked to the repetitive movements of the exercise"。這個特定的 ROM 錯誤找不到同儕審查來源——只有描述性支撐。

#### jj_incomplete_leg_rom
- **fault_id**: `jj_incomplete_leg_rom`
- **fault_name**: 張開階段腿部外展不完整(站距太窄)(Incomplete leg abduction, narrow stance)
- **description**: 張開階段雙腳沒有跨到完整的寬站距——變成拖著腳的窄幅開合跳。
- **detection_heuristic**: 在張開階段峰值計算 ankle-width/shoulder-width = |x27−x28| / |x11−x12|。若比值維持在約 1.3 以下(雙腳只比肩膀寬一點點),而不是寬幅的側跨,就觸發。正面視角。
- **observability**: high;front / front_oblique 視角。
- **biomechanical_rationale**: 雙腿大幅張開是這個動作定義性的下肢 ROM;站距太窄會減少預期的外展/內收肌與心肺負荷。屬表現/完整度錯誤。
- **citation**: Wikipedia, "Jumping jack" (CC BY-SA) —— `data/rag/docs/jumping_jacks_wiki.txt`。
- **citation_support**: VERIFIED(描述性,已讀)。定義為 "jumping to a position with the legs spread wide" 再回到 "with the feet together"。窄站距這個錯誤本身找不到同儕審查來源——只有描述性支撐。

#### jj_landing_asymmetry
- **fault_id**: `jj_landing_asymmetry`
- **fault_name**: 左右不對稱(Left/right asymmetry)
- **description**: 一側持續在伸展、落地或吸收上與另一側不同(手臂舉不齊、跨步不等寬,或單側膝塌陷)。
- **detection_heuristic**: 逐下比較左右成對量:腕高峰值(y15 vs y16)、踝相對髖中線的側向行程,以及逐側的膝外翻比值。若正規化的左右差跨多下持續超過約 15–20%,就觸發。正面視角。
- **observability**: medium;正面視角;需要多下的一致性才能把真正的不對稱與雜訊分開。
- **biomechanical_rationale**: 落地力學上持續的左右不對稱會把負荷集中到單一肢段;外翻那側的肢段承受 Tamura 等人記錄的較高膝負荷,因此持續單側塌陷就是風險較高的型態。
- **citation**: Tamura, A. et al. (2017), *PLoS ONE* 12(6):e0179810, PMC5478135(同上)。
- **citation_support**: VERIFIED。同一筆已抓取的 Tamura 發現——以外翻落地的那隻腿承受較大的膝衝量;不對稱意味著同一隻腿反覆扮演這個角色。把不對稱當成錯誤是這項發現的應用,而非另有一項專門量測不對稱的研究(如實註明)。

---

### 高抬腿 High Knee

反覆階段(跑步訓練/原地踏步):**drive(快速髖屈曲到膝抬最高)→ foot strike(觸地支撐)→ 換另一腳抬膝**。每一步交替單腳支撐。關鍵點:鼻 0、肩 11/12、髖 23/24、膝 25/26、踝 27/28。

#### hk_insufficient_knee_lift
- **fault_id**: `hk_insufficient_knee_lift`
- **fault_name**: 抬膝高度不足(髖屈曲 ROM 受限)(Insufficient knee-lift height)
- **description**: 驅動腿的大腿抬得太低——沒有到達目標高度(大腿至少高於水平約 45°,高抬版本要接近平行),因此定義這項訓練的髖屈曲 ROM 沒有達成。
- **detection_heuristic**: 在驅動峰值量測驅動腿膝與同側髖的垂直關係:在影像座標(y 往下)中,若膝從未升到接近髖高,例如 (y_knee − y_hip) 維持 > +0.05(膝明顯低於髖),就觸發。等價作法是由髖→膝向量估出大腿相對水平的角度,當大腿抬起峰值低於約 45° 目標時觸發。側面視角最準,正面視角可作為堪用的替代。
- **observability**: high;以側面視角為佳,正面可接受。
- **biomechanical_rationale**: 這項訓練的教學目標是高抬膝、大腿至少高於水平約 45°(進階版本更高);達不到就拿掉了這項訓練存在的目的——擺動期的髖屈曲刺激(表現損失)。
- **citation**: Matijašević, P. et al. (2025). "Development and validation of a running drill test battery to predict 5 m and 20 m sprint performance." *International Journal of Exercise Science* 18(8):1269–1285. https://pmc.ncbi.nlm.nih.gov/articles/PMC12591607/ (DOI 10.70252/LYKE8231)
- **citation_support**: VERIFIED(已抓取)。A-skip 的評分準則要求擺動期 "the thigh to reach approximately 45° relative to the ground"(B-skip 則抬更高),而這項訓練的目的被描述為 "promoting knee elevation and optimizing kinematics"——直接支撐抬膝高度目標。(該論文同時發現 A-skip 與衝刺成績只有微不足道的相關,所以這要定位成技術/ROM 目標,而非表現保證。)

#### hk_trunk_lean_back
- **fault_id**: `hk_trunk_lean_back`
- **fault_name**: 用軀幹後仰把膝蓋撐上來(Leaning the trunk backward to hoist the knee)
- **description**: 把上半身往後甩(腰椎過度伸展)去把膝蓋盪上來,而不是用髖屈肌把它驅動起來。
- **detection_heuristic**: 側面視角追蹤軀幹向量 髖中點→肩中點 相對垂直的角度;當肩中點的 x 跑到髖中點的 x *後面*,且抬膝峰值時的後仰角超過約 10–15°,就觸發。支撐期建立直立基線。side / side_oblique 視角。
- **observability**: medium;側面或斜側視角(後仰是矢狀面線索)。純正面視角解析不出來。
- **biomechanical_rationale**: 後仰是用軀幹/腰椎伸展去替代真正的髖屈曲,在偷高度的同時讓腰椎反覆進入過度伸展的負荷;這項訓練需要直立、受控的身體位置才能練到預期的衝刺擺動力學。
- **citation**: Matijašević, P. et al. (2025), *Int J Exerc Sci* 18(8):1269–1285, PMC12591607(同上)。
- **citation_support**: VERIFIED(姿勢目標)/ UNVERIFIED(傷害量級)。已抓取的論文把這類訓練定位為訓練 "proper body position" 與擺動期運動學,支撐直立軀幹這項準則。找不到任何專門量化高抬腿訓練中腰椎過度伸展傷害風險的來源——傷害這一層是機轉推論,在此標為缺口。

#### hk_forward_trunk_collapse
- **fault_id**: `hk_forward_trunk_collapse`
- **fault_name**: 軀幹前塌(Forward trunk collapse)
- **description**: 支撐期/支撐中期軀幹往前傾倒而不是維持挺立,整個人摺在支撐腿上。
- **detection_heuristic**: 側面視角計算軀幹向量(髖中點→肩中點)相對垂直的角度;當肩中點跑到髖中點前方,且支撐中期的前傾角超過直立基線約 15–20°,就觸發。side / side_oblique 視角。
- **observability**: medium;側面或斜側視角。
- **biomechanical_rationale**: 支撐中期較大的軀幹前傾,是區分受傷跑者與健康跑者的運動學型態之一,與對側骨盆下掉並列——屬於與跑步相關軟組織傷害有關的控制錯誤。
- **citation**: Bramah, C., Preece, S.J., Gill, N., Herrington, L. (2018). "Is There a Pathological Gait Associated With Common Soft Tissue Running Injuries?" *American Journal of Sports Medicine* 46(12):3023–3031. PMID 30193080. https://pubmed.ncbi.nlm.nih.gov/30193080/
- **citation_support**: VERIFIED(已抓取 PubMed 摘要)。受傷跑者相較健康跑者 "demonstrated greater contralateral pelvic drop (CPD) and forward trunk lean at midstance",且此結果在四個傷害次分組(PFP、ITBS、MTSS、阿基里斯腱病變)中一致——直接支撐把軀幹前傾當成與傷害相關的步態錯誤。

#### hk_contralateral_pelvic_drop
- **fault_id**: `hk_contralateral_pelvic_drop`
- **fault_name**: 對側骨盆下掉(Contralateral pelvic drop)
- **description**: 單腳支撐期骨盆傾斜,使擺動腿側的髖低於支撐腿側的髖(額狀面提髖失敗 / Trendelenburg 型態)。
- **detection_heuristic**: 單腳支撐期量測髖線傾斜 = 帶號的 (y23 − y24),以髖寬 |x23−x24| 正規化;由哪隻腳著地(踝 y 較低)判定支撐側與擺動側。當擺動側髖相對支撐側髖下掉超過門檻(例如骨盆傾斜角 > 約 5–8°,依關鍵點雜訊調校)就觸發。front / front_oblique 或 rear 視角。
- **observability**: high;正面或背面視角。純側面視角看不到額狀面的骨盆傾斜。
- **biomechanical_rationale**: 對側骨盆下掉是與常見跑步相關軟組織傷害關聯最強的單一運動學變數;每多一度,受傷勝算就明顯上升,反映額狀面骨盆(髖外展肌)控制的喪失。
- **citation**: Bramah, C. et al. (2018), *Am J Sports Med* 46(12):3023–3031, PMID 30193080(同上)。
- **citation_support**: VERIFIED(已抓取 PubMed 摘要)。"CPD was found to be the most important variable predicting the classification of participants as healthy or injured",且 "for every 1° increase in pelvic drop, there was an 80% increase in the odds of being classified as injured"。(附帶說明:McCarney et al. 2020, *Chiropr Man Therap* 28:53, PMC7570029——已抓取——在健康成人身上只發現平均約 4.6° 的下掉,且觀察到的 Trendelenburg 骨盆下掉與實測的髖外展肌力量*沒有*相關,所以這裡把*下掉*本身當成可觀測、與傷害相關的訊號,而不是外展肌無力的直接讀數。)

#### hk_stride_asymmetry
- **fault_id**: `hk_stride_asymmetry`
- **fault_name**: 左右步態不對稱(Left/right stride asymmetry)
- **description**: 某一腳持續抬得比較低,或某一側的骨盆下掉/軀幹位移比另一側大。
- **detection_heuristic**: 跨多下比較左右步的抬膝高度峰值(膝與髖的 y 關係)與逐側骨盆下掉角;若正規化的左右差持續超過約 15–20% 就觸發。正面視角(抬膝高度 + 骨盆下掉),並跨多下彙總。
- **observability**: medium;正面 / 斜側視角,需要數個步幅才可靠。
- **biomechanical_rationale**: 持續的左右不對稱會把與傷害相關的型態(骨盆下掉、髖屈曲不足)集中在單一肢段;對側骨盆下掉是跑步中與傷害關聯最強的變數,所以習慣性下掉/驅動不足的那一側就是風險升高的那隻腿。
- **citation**: Bramah, C. et al. (2018), *Am J Sports Med* 46(12):3023–3031, PMID 30193080(同上)。
- **citation_support**: VERIFIED(應用)。Bramah 的 CPD 發現支撐把較差的一側單獨挑出來;不對稱這個框架是該結果的應用,而不是一項專門的不對稱研究(如實說明)。

---

### 所用引用彙整

| 動作 | 主要同儕審查來源 | 狀態 |
|---|---|---|
| 軀幹旋轉 Torso Twist | McGill 1991, *J Orthop Res* (PMID 1824571) | 已抓取並查證(Escamilla 2006 檢查後判定離題,捨棄) |
| 開合跳 Jumping Jacks | Tamura 2017, *PLoS ONE* (PMC5478135); DeVita & Skelly 1992, *MSSE* (PMID 1548984) | 兩筆皆已抓取並查證 |
| 高抬腿 High Knee | Matijašević 2025, *Int J Exerc Sci* (PMC12591607); Bramah 2018, *Am J Sports Med* (PMID 30193080) | 兩筆皆已抓取並查證 |

Wikipedia 的 RAG 文件(`torso_twist_russian_wiki.txt`、`jumping_jacks_wiki.txt`)只作描述性補充。高抬腿完全沒有 RAG 文件(已知缺口)——全部由網路取得的同儕審查文獻覆蓋。

誠實揭露的缺口 / UNVERIFIED 項目:
- `jj_incomplete_arm_rom`、`jj_incomplete_leg_rom`:只有描述性(Wikipedia)支撐;這兩個完整度錯誤找不到同儕審查來源。
- `hk_trunk_lean_back`:直立姿勢目標已查證;腰椎過度伸展的*傷害*量級屬機轉推論,找不到專門來源。
- `hk_contralateral_pelvic_drop`:骨盆下掉→傷害的關聯有強力查證(Bramah);骨盆下掉→外展肌無力的關聯有爭議(McCarney 2020),因此刻意不主張。


---

## 6. 參考文獻索引

RAG 語料庫來源(作者/標題/期刊以 `data/paper_metadata.json` 為權威),以 PMCID 為鍵:

- **PMC11048684** — Escamilla RF, Thompson IS, Carinci J, et al. (2024). Effects of Ankle Position While Performing One- and Two-Leg Floor Bridging Exercises on Core and Lower Extremity Muscle Recruitment. Bioengineering (Basel, Switzerland). PMC11048684.
- **PMC11981018** — Colonna S, D'Alessandro A, Tarozzi R, Casacci F. (2025). Supine Bridge Exercise: A Narrative Review of the Literature (Part I). Cureus. PMC11981018.
- **PMC12029123** — Mun WL, Jung EY, Lei S, Roh SY. (2025). Scapular Muscle Activation at Different Shoulder Abduction Angles During Pilates Reformer Arm Work Exercise. Medicina (Kaunas, Lithuania). PMC12029123
- **PMC12148905** — Hanen NC et al. Frontiers in bioengineering and biotechnology (2025). PMC12148905
- **PMC12225233** — Moreira VM et al. Muscles (Basel, Switzerland) (2023). PMC12225233
- **PMC12335237** — Abiara S, Heinrichs V, Chorneyko A, Lang AE. (2025). Acute effects of lower trapezius activation exercises on shoulder muscle activation during overhead functional tasks in symptomatic and asymptomatic adults. PeerJ. PMC12335237
- **PMC12366113** — Abdollahi S, Sheikhhoseini R, Salsali M, Piri H, Hides JA. (2025). The influence of hand position on scapular kinematics in push-ups: comparing athletes with chronic shoulder pain and healthy controls. Journal of orthopaedic surgery and research. PMC12366113.
- **PMC12372021** — González-de-la-Flor Á. (2025). Optimizing Hip Abductor Strengthening for Lower Extremity Rehabilitation: A Narrative Review on the Role of Monster Walk and Lateral Band Walk. Journal of functional morphology and kinesiology. PMC12372021.
- **PMC12372072** — Evangelista P, Rum L, Picerno P, Biscarini A. (2025). Decoding the Contribution of Shoulder and Elbow Mechanics to Barbell Kinematics and the Sticking Region in Bench and Overhead Press Exercises: A Link-Chain Model with Single- and Two-Joint Muscles. Journal of functional morphology and kinesiology
- **PMC12416692** — Rodrigues R, Sonda FC, Frigotto MF, et al. (2025). Sex as a moderator of the relationship between hip abduction strength and muscle activation during single-leg stance. PloS one. PMC12416692.
- **PMC12514857** — Al Hammadi MI, Shah ZA, Rathod RK, Seddik MA. (2025). Shoulder Impingement Pain Syndrome: Pathophysiology, Diagnosis, and a Review of Current Treatment Strategies. Cureus
- **PMC12550948** — Parpa K, Vasiliou A, Michaelides M, et al. (2025). An Exploratory Study of Biceps Brachii Electromyographic Activity During Traditional Dumbbell Versus Bayesian Cable Curls. Muscles (Basel, Switzerland). PMC12550948
- **PMC12734928** — Jung EY, Roh SY, Mun WL. (2025). Electromyographic Patterns of Scapular Muscles During Four Variations of Protraction-Retraction Exercises. Life (Basel, Switzerland). PMC12734928
- **PMC12821611** — Padovan R, Cè E, Longo S, et al. (2025). High-Density Surface Electromyography Excitation of Prime Movers Across Scapular Positions in the Seated Row. Journal of functional morphology and kinesiology.
- **PMC13086636** — Gregori P, La Bruna M, Papalia GF, Giurazza G, Caria C, Paciotti M, Russo F, Franceschetti E, Longo UG, Papalia R. (2026). Spine alignment influences shoulder range of motion and scapular orientation: A systematic review from the FP-UCBM Shoulder Study Group. Journal of experimental orthopaedics
- **PMC13116542** — Abdelraouf OR, Abdel-Aziem AA, Alkhamees NH, Ibrahim ZM, Aboelela EM, Dawood RS, Ashour AA. (2026). Acute Effects of High-Load Training to Failure vs. Non-Failure on Posture and Core Endurance in Collegiate Weightlifters: A Crossover Study. Journal of clinical medicine
- **PMC13232157** — Owens LP, Coyles G, Khaiyat O (2026). Whole Body Kinetic Chain Muscle Activity during selected Rehabilitation Exercises in Healthy and Injured Overhead Throwing Athletes. International journal of sports physical therapy.
- **PMC3820220** — Lee S, Lee D, Park J. (2013). The Effect of Hand Position Changes on Electromyographic Activity of Shoulder Stabilizers during Push-up Plus Exercise on Stable and Unstable Surfaces. Journal of physical therapy science. PMC3820220.
- **PMC4327800** — San Juan JG, Suprak DN, Roach SM, Lyda M. (2015). The effects of exercise type and elbow angle on vertical ground reaction force and muscle activity during a push-up plus exercise. BMC musculoskeletal disorders. PMC4327800.
- **PMC4519219** — Barbado D, Elvira JL, Moreno FJ, Vera-Garcia FJ. (2015). Effect of Performance Speed on Trunk Movement Control During the Curl-Up Exercise. Journal of human kinetics. PMC4519219.
- **PMC4556293** — Ford KR, Nguyen AD, Dischiavi SL, Hegedus EJ, Zuk EF, Taylor JB. (2015). An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus. Open access journal of sports medicine. PMC4556293.
- **PMC6523035** — Zellmer M, Kernozek TW, Gheidi N, Hove J, Torry M. (2019). Patellar tendon stress between two variations of the forward step lunge. Journal of sport and health science. PMC6523035.
- **PMC6548056** — Soriano MA, Suchomel TJ, Comfort P. (2019). Weightlifting Overhead Pressing Derivatives: A Review of the Literature. Sports medicine (Auckland, N.Z.)
- **PMC6980669** — Alkjær T, Smale KB, Flaxman TE, Marker IF, Simonsen EB, Benoit DL, Krogsgaard MR. (2020). Forward lunge before and after anterior cruciate ligament reconstruction: Faster movement but unchanged knee joint biomechanics. PloS one. PMC6980669.
- **PMC8805090** — Escamilla R, Zheng N, MacLeod TD, Imamura R, Wilk KE, Wang S, Rubenstein I, Yamashiro K, Fleisig GS. (2022). Patellofemoral Joint Loading During the Performance of the Forward and Side Lunge with Step Height Variations. International journal of sports physical therapy. PMC8805090.
- **PMC8975561** — Fukunaga T et al. International journal of sports physical therapy (2022). PMC8975561
- **PMC9354811** — Coratella G, Tornatore G, Longo S, Esposito F, Cè E. (2022). Front vs Back and Barbell vs Machine Overhead Press: An Electromyographic Analysis and Implications For Resistance Training. Frontiers in physiology
- **PMC9505236** — Mandroukas A, Michailidis Y, Kyranoudis AE, Christoulas K, Metaxas T. (2022). Surface Electromyographic Activity of the Rectus Abdominis and External Oblique during Isometric and Dynamic Exercises. Journal of functional morphology and kinesiology. PMC9505236.

網路抓取 / 外部同儕審查來源。**每一筆都獨立重新抓取,確認能解析到所述的標題、作者、年份與發現**(它們沒有本地 metadata 可比對,因此逐筆查證):

- Hartmann H, Wirth K, Klusemann M. (2013). Analysis of the load on the knee joint and vertebral column with changes in squatting depth and weight load. *Sports Medicine* 43(10):993–1008. DOI 10.1007/s40279-013-0073-6, PMID 23821469. (WebFetched —— 深蹲深度)
- McGill SM. (1991). Electromyographic activity of the abdominal and low back musculature during the generation of isometric and dynamic axial trunk torque. *J Orthop Res* 9(1):91–103. PMID 1824571. (WebFetched —— 軀幹旋轉)
- Tamura A, et al. (2017). Dynamic knee valgus alignment influences impact attenuation in the lower extremity during the deceleration phase of a single-leg landing. *PLoS ONE* 12(6):e0179810. PMC5478135. (WebFetched —— 開合跳)
- DeVita P, Skelly WA. (1992). Effect of landing stiffness on joint kinetics and energetics in the lower extremity. *Med Sci Sports Exerc* 24(1):108–115. PMID 1548984. (WebFetched —— 開合跳)
- Matijašević P, et al. (2025). Development and validation of a running drill test battery to predict 5 m and 20 m sprint performance. *Int J Exerc Sci* 18(8):1269–1285. PMC12591607, DOI 10.70252/LYKE8231. (WebFetched —— 高抬腿)
- Bramah C, Preece SJ, Gill N, Herrington L. (2018). Is There a Pathological Gait Associated With Common Soft Tissue Running Injuries? *Am J Sports Med* 46(12):3023–3031. PMID 30193080. (WebFetched —— 高抬腿)
- McCarney L, Andrews A, Henry P, Fazalbhoy A, Selva Raj I, Lythgo N, Kendall JC. (2020). Determining Trendelenburg test validity and reliability using 3-dimensional motion analysis and muscle dynamometry. *Chiropr Man Therap* 28:53. PMC7570029. (WebFetched —— 高抬腿,骨盆下掉與外展肌力量的爭議性連結)
- Camargo PR, Neumann DA. (2019). Kinesiologic considerations for targeting activation of scapulothoracic muscles – part 2: trapezius. *Braz J Phys Ther* 23(6):467–475. PMC6849087, DOI 10.1016/j.bjpt.2019.01.011. (WebFetched —— 彈力帶擴胸 / 上斜方肌)
- Havers T, Wagner N, Held S, Geisler S, Wiewelhove T. (2025). Partial Range, Full Gains? The Effect of 8 Weeks of Partial Range of Motion Training at Long Muscle Lengths on Elbow Flexor Hypertrophy and Strength in Trained Individuals. *European Journal of Sport Science*. DOI 10.1002/ejsc.70087, PMID 41247250. (WebFetched —— 二頭彎舉 ROM)
- Creech JA, Busse A, Li D, et al. Shoulder Impingement Syndrome. *StatPearls* (NCBI Bookshelf NBK554518, updated 2026). (WebFetched —— 手臂外展疼痛弧 / 夾擠)
- Terré M, Solana-Tramunt M. (2025). Muscle Recruitment and Asymmetry in Bilateral Shoulder Injury Prevention Exercises: A Cross-Sectional Comparison Between Tennis Players and Non-Tennis Players. *Healthcare (Basel)* 13(10):1153. PMC12110944, DOI 10.3390/healthcare13101153. (WebFetched —— 手臂 VW 不對稱)

描述性補充(Wikipedia,CC BY-SA——只用於動作定義/技術目標,絕不單獨支撐傷害風險):`squat_wiki.txt`、`lunge_wiki.txt`、`pushup_wiki.txt`、`ohp_wiki.txt`、`row_wiki.txt`、`torso_twist_russian_wiki.txt`、`jumping_jacks_wiki.txt`。

*下列 PMCID 屬於上一區塊列出的網路抓取外部來源,不是 RAG 語料庫條目:PMC12110944、PMC12591607、PMC5478135、PMC6849087、PMC7570029。*

---

## 7. 誠實揭露的限制與缺口

依本規格的誠實要求,以下如實列出而不遮掩:

- **硬舉腰椎屈曲**(`deadlift_lumbar_flexion`)——臨床上最重要的硬舉動作錯誤——屬於
  **低可觀測性**:MediaPipe 在肩與髖之間沒有任何脊椎關鍵點,所以那套啟發式是明確的替代指標,
  不是真正的拱背 vs 中立脊椎量測。
- **伏地挺身翼狀肩胛**(`pushup_scapular_winging`)真實且有引用,但單鏡頭姿態的
  **observability 為 none**(沒有肩胛關鍵點);列出是為了完整,不是為了偵測。
- **彈力帶擴胸/划船的肩胛後收缺失**屬**低可觀測性**——肩胛位置無法從單鏡頭正面視角可靠還原;
  文中給出的是背面視角的寬度替代指標。
- **划船拱背傷害**與**彈力帶擴胸後仰**:*負荷量級*有同儕審查支撐,但負重下屈曲與*傷害*的連結
  屬推論(已於條目內註明)。
- **軀幹旋轉的腰椎 vs 胸椎旋轉**屬**低可觀測性**——稀疏關鍵點無法把胸椎與腰椎的旋轉分段;
  文中改用髖線對肩線的替代指標。
- **開合跳的 ROM 完整度錯誤**(`jj_incomplete_arm_rom`、`jj_incomplete_leg_rom`)僅有
  Wikipedia 的描述性支撐——這兩項本身找不到同儕審查來源。
- **高抬腿後仰**的傷害量級屬機轉推論(直立姿勢目標有來源;腰椎過度伸展的危害沒有另行量化)。
- **仰臥起坐 / 腿部外展**的提示門檻(腳尖朝上外轉、速度上限)超出來源的字面說法;背後的原則
  有支撐,確切數字則是待實證驗證的調校目標。
- **對側骨盆下掉**(高抬腿、弓步)是強力的*傷害關聯*訊號(Bramah 2018);但**不**主張它是
  髖外展肌無力的直接讀數,那一點仍有爭議(McCarney 2020)。

**視角判定(view estimation)的方向性限制(2026-07-25,於 `body_axis_extent` 改為方向感知的
身體延伸量測時新增;以下四點以 `src/pose/view_estimation.py` 模組 docstring 內的版本為準):**

1. `signed_orientation`(`sign(left.x - right.x)`)是影像座標系上的左右順序,其正面/背面意義
   只在**直立**受試者身上經過驗證。對於水平的身體,正面軸不再對應到影像的 x 軸,因此
   `front`/`rear`/`*_oblique` 這幾個標籤在該情境下不具任何經驗證的意義。不要用它們去為水平動作
   (例如伏地挺身)的規則設閘門。
2. `estimate_view_for_pose` 在正式環境路徑(`src/pose/pose_rule_detector.py`)中是以
   `allow_front=False` 呼叫的,因此 `front` 與 `front_oblique` 在那裡永遠不會被觸及;下游只會
   輸出 `side`、`rear`、`rear_oblique`、`unknown` 這四種。
3. `_visible_midpoint` 要求一對關鍵點的**左右兩側**都高於 0.35 可見度才會被納入身體軸線計算。
   只要一側肩膀被遮擋——或是腳踝與髖部這一對關鍵點不完整——就會讓 `body_axis_extent` 悄悄退回
   修正前的垂直方向替代值,而不是真正的身體軸線,而且不會出現 NaN,也不會有任何其他訊號提示。
   實測:在一個水平姿勢的測試素材中,把第 12 號關鍵點(右肩)的可見度強制設為 0.1,量到的軸線
   延伸值是 0.070,而不是預期的 ~0.60(低了 8.6 倍)。這不是回歸——這個退回值就是 2026-07-25
   修正之前的行為,對直立深蹲而言是正確的——但它會在最容易觸發的情境下,悄悄地把方向感知修正的
   效果抵銷掉:矢狀面(側面)視角正是遠側關鍵點最常被遮擋的視角。
4. 當一支影片完全沒有方向性證據(`front_score == rear_score == 0.0`),但仍靠軀幹寬度證據單獨
   跨過證據門檻時,`score_view` 的分支階梯會把它判成 `rear_oblique` 而不是 `unknown`——因為在
   `allow_front=False` 下,`front_score >= rear_score` 這個分支在兩者平手(0.0 對 0.0)時會被
   無條件採用。往下游看,`rear_oblique` 落在 `src/pose/movements/squat.py` 裡
   `rule_knees_inward` 的 `observable_alignment` 閘門集合內(舊的 `side` 判定不在其中),而該
   閘門沒有信心下限——於是一支毫無證據的影片,`knees_inward` 反而可能得到 **confidence 1.000 /
   observability "high"** 而不是被排除在外,相對於這次改動之前的 **confidence 0.650 /
   "medium"**。**此次刻意不修正**:在該閘門加上信心下限會改變深蹲規則的輸出,而
   `tests/test_movement_registry.py` 釘住了「動作註冊表路徑」與 `pose_rule_detector.py` 內
   legacy oracle 逐位元組相同的比對測試,若不同步套用相同修改,該閘門測試就會失敗。這是已知、
   有實測數字佐證的缺陷,等待後續獨立處理,不是被忽略。

## 8. 後續步驟

1. 使用者審閱本規格。
2. 在 `src/pose/pose_rule_detector.py` 實作各動作的偵測器,把逐幀指標與分段偵測機制從深蹲擴展
   出去(需要各動作的階段分段,以及本文定義的新幾何訊號)。
3. 把每條動作錯誤的 `citation`/`citation_support` 接進 KG/RAG 檢索層,讓教練對話能呈現依據來源,
   如同深蹲的動作錯誤目前透過 `kg_query` 所做的那樣。
4. 在為某個動作出貨分析功能之前,先以該動作的標註資料驗證門檻。

**狀態(2026-07-18):** 基礎已出貨(動作註冊表 + 引用 + 行為不變的深蹲遷移 + 過頭推舉),
位於 `feat/movement-rule-detector-spec` 分支。**過頭推舉的門檻是由規格推導、尚未驗證**——
目前沒有已標註的過頭推舉資料(§8.4)。其餘 14 個動作將沿用本框架,以逐動作的計畫接續進行。

**狀態(2026-07-25):** 過頭推舉的 5 條規則已 **全數(5/5)** 實作於
`src/pose/movements/overhead_press.py`(`ohp_incomplete_lockout`、`ohp_lumbar_hyperextension`、
`ohp_asymmetric_press`、`ohp_insufficient_elevation`、`ohp_forward_head`)。實作與上文
偵測啟發式有數處刻意的偏離,程式碼內亦已註明:

- `ohp_insufficient_elevation`——上文「約 0.5 個頭高」的寫法 **無法實作**:MediaPipe 的 33 個
  關鍵點裡根本沒有可量測頭高的東西(鼻、眼、耳、嘴全部落在臉部範圍內,任兩點都跨不過整顆頭)。
  實作改為 **替代準則**:以肩寬正規化的鼻部淨空高度,當
  `(wrist_mean_y − nose_y) / shoulder_width > −0.15` 時觸發。這是 **替代,不是單位換算**——
  沒有假設或引入任何「頭高對肩寬」的人體測量常數。
- `ohp_forward_head`(由 `ohp_forward_head_barpath` 更名)——實作為 **硬性視角閘門**,而非
  上文「側面中高、正面低」所隱含的可觀察度降級。該線索是純水平位移,方向在不知道受試者面向時
  無法判定,因此在 `{side, front_oblique}` 以外的視角 **完全不輸出偵測**,而不是輸出低信心的
  偵測:給錯方向比沉默更糟。另外,它所用的肩寬正規化在被閘門允許的側面視角下條件最差。
- `ohp_forward_head`——閘門另外要求 `view_confidence >= 0.20`
  (`src/pose/pose_rule_detector.py` 中的 `SIDE_VIEW_CONF_THRESHOLD`),沿用深蹲
  `rule_knees_forward` 的先例。**這是對該先例的擴充**:深蹲只對 `side` 設閘,這裡把同一道信心
  下限也套用到 `front_oblique`,理由是「分類信心很低的斜側視角」和「分類信心很低的正側視角」
  一樣沒有資格支撐一個帶方向的宣稱。沒有引入新數字——常數與深蹲共用。
- `ohp_forward_head`——規格中的 **槓路子準則已撤回**;完整理由與它留下的待決規格問題,見上文
  該規則條目中的引用方塊。
- `ohp_incomplete_lockout`——遮罩的觸發條件是 `elbow_flag OR wrist_flag`,因此嚴重度改為同時
  計算 **兩條** 規格內的斜坡(肘 160→140 度、腕 0.0→0.15)後取較差者。原本以「肘的讀數是否
  有限」來選斜坡,會在區段只由腕準則觸發時算錯:一個槓根本沒離開肩膀高度、但手肘打直的動作,
  會以 **severity 0.0 / confidence 0.0** 輸出,而且帶著與自己所指錯誤互相矛盾的證據
  (「峰值肘角 178 度 vs 門檻 160 度」)。沒有引入新門檻。

五條過頭推舉門檻全部尚未以標註資料驗證(§8.4)。

**已知的未竟事項(不是偏離,是缺口):** 有三個過頭推舉的 `kg_query` 字串在
`data/kg/sports_kg_v3.graphml` 上 **完全解析不到任何節點**——`"Incomplete Elbow Lockout"`、
`"Lumbar Hyperextension"`、`"Asymmetric Press"`(以 `graph_retrieval.resolve_nodes` 驗證,
限定動作與不限定動作都一樣)。`"Forward Head Posture"` 與 `"Limited Shoulder Elevation"` 則
可正常解析。圖上也沒有可以改指過去的相近節點(最接近的過頭推舉節點是 `Near Lockout`、
`Thoracolumbar Extension`、`Elbow Extensor Torque`),所以這需要補 KG 內容,不是改字串就能解決。
目前那三條錯誤送到對話層時有引用文獻,但沒有檢索到的依據。
