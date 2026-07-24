# 單目健身動作品質評估的深度瓶頸:視角失真量測、image-to-3D 修正,與可解釋教練系統的感測選型

投稿類別:醫療科技 —— 運動科技
版本:第一版初稿(2026-07-23)

---

## 摘要

家用復健與健身場域能拿到的感測條件通常只有一支手機鏡頭,而動作品質評估的既有做法多半沿用單視角 2D 骨架特徵。本研究在 Fit3D 與 REHAB24-6 兩組具光學動作捕捉真值的資料上量測這個設定的極限,並定出誤差的來源。以四台同步相機讀取同一次深蹲,單視角 2D 的膝關節角度相對 mocap 真值的合併絕對誤差為 42.4°;某次蹲到底的膝角真值為 78°,四個機位分別讀成 108°、118°、119° 與 133°。將單視角管線的線索誤差拆解為偵測器項與投影項後,深蹲膝角的偵測器項僅 −0.70°,受試者間標準差 ±1.5°,兩者統計上不可區分,誤差幾乎全部來自投影幾何。換更強的 2D 骨架模型因此無法補救:REHAB24-6 的正確性分類上,HRNet-w48 與 RTMPose 的留一受試者交叉驗證平衡準確率相差 +0.005 ± 0.071(Wilcoxon p = 0.734)。改由影像直接回歸 3D 則能取回深度軸——NLF 的深度軸誤差 42.4mm 與平面內軸相當(ez/exy = 1.16),膝角誤差自 18.42° 降至 7.09°。這項差距會直接改變教練判定:原始單視角 2D 對深蹲深度的通過與犯規判定,在 76% 的重複動作上與真值相反,其中把合格動作誤判為犯規的比率達 82%。在同一套逐機位偏移校正下比較,2D 的翻轉率降到 16%,卻是以漏判 42% 換來假警報 14%,無法同時壓低兩類錯誤,因為其殘餘誤差逐次重複變動;NLF 則同時做到假警報 7% 與漏判 0%。該校正需動用真值計算,兩者的校正後數字都是效能上限而非可部署值,其意義在於顯示 3D 讀數把誤差降到了校正得以生效的程度。此一恢復能力並非特定模型的偽跡,NLF、HMR2.0、Multi-HMR 與 MeTRAbs 四種架構皆具備;關鍵成分是深度品質而非網格密度,同為稀疏骨架輸出,具真實深度的 MeTRAbs 膝角誤差 6.37°,採啟發式深度的 MediaPipe 則為 14.18°。據此,本研究主張動作品質系統應依錯誤型態分流感測:矢狀面的深度與屈曲判定需要真實 3D,額狀面的膝內扣則由偵測器精度主導,改良 2D 即可。文末說明這套結論如何落在本團隊開發的可解釋教練系統 x-coach 的感測選型上。

**關鍵詞**:動作品質評估、人體姿態估計、單目 3D 重建、復健科技、可解釋人工智慧

## Abstract

Home rehabilitation and fitness feedback are usually limited to a single phone camera, yet most action quality assessment pipelines still read biomechanical cues from single-view 2D skeletons. We measure the limits of that setting on Fit3D and REHAB24-6, both with optical motion capture ground truth, and localise where the error comes from. Reading the same squat from four synchronised cameras, single-view 2D knee angle deviates from mocap truth by 42.4° pooled MAE; at one squat bottom whose true knee angle is 78°, the four views report 108°, 118°, 119° and 133°. Decomposing the single-view error into a detector term and a projection term, the squat knee detector term is only −0.70° against a per-subject spread of ±1.5° — statistically indistinguishable, meaning essentially all of the error is projection geometry. A stronger 2D backbone therefore cannot help: on REHAB24-6 correctness classification, HRNet-w48 and RTMPose differ by +0.005 ± 0.071 balanced accuracy under leave-one-subject-out validation (Wilcoxon p = 0.734). Direct image-to-3D regression does recover the depth axis: NLF reaches 42.4mm depth-axis error, on par with its in-plane error (ez/exy = 1.16), cutting knee-angle error from 18.42° to 7.09°. The gap changes coaching verdicts: raw single-view 2D disagrees with ground truth on the squat-depth pass/fail call for 76% of repetitions, with an 82% false-alarm rate on legal squats. Under an identical per-camera offset calibration, 2D drops to a 16% flip rate but only by trading 14% false alarms against 42% misses, since its residual scatter is rep-dependent, whereas NLF reaches 7% false alarms at 0% misses. That calibration is computed from ground truth, so both calibrated figures are upper bounds rather than deployable numbers; their point is that 3D lowers the cue error enough for calibration to work at all. The recovery generalises across NLF, HMR2.0, Multi-HMR and MeTRAbs, and the decisive ingredient is depth quality rather than mesh density: with equally sparse skeletal output, MeTRAbs (true depth) reaches 6.37° knee error while MediaPipe (heuristic depth) reaches 14.18°. We argue for routing sensing by fault type — true 3D for sagittal-plane depth and flexion verdicts, improved 2D for frontal-plane valgus — and report how this shapes the sensing design of x-coach, our explainable coaching system.

**Keywords**: action quality assessment, human pose estimation, monocular 3D reconstruction, rehabilitation technology, explainable AI

---

## 一、緒論

復健科與運動傷害防護的門診時間有限,病人回到家後的動作品質沒有人看。近年動作品質評估(action quality assessment, AQA)的研究提供了一條技術路線:用一支相機拍下動作,由模型輸出評分或缺陷標籤。這條路線在實務上卡在兩個地方。其一是輸出形式,多數 AQA 模型只給分數,不說明哪個關節在哪個時間點出了什麼問題,病人拿到 6.2 分不知道下一次該改什麼。其二比較隱蔽,也是本研究的重點:這些系統讀出的生物力學線索,本身可能是錯的。

單視角 2D 骨架把三維動作投影到影像平面,矢狀面的資訊在這個過程中被壓掉。深蹲是否蹲到大腿與地面平行、膝關節屈曲是否足夠,這類判定仰賴的正是被壓掉的那個軸。既有工作多半把資源投在更強的 2D 偵測器上,假設關鍵點座標越準,下游的角度與比值就越準。本研究要指出這個假設在深度線索上不成立,並量化錯得多離譜。

本研究的貢獻有四項。第一,在具動作捕捉真值的多視角資料上量測單視角 2D 讀出的生物力學線索有多失真,以及失真如何隨相機位置改變。第二,把單視角管線的誤差拆解成偵測器誤差與投影誤差,證明深度線索的誤差幾乎全部落在後者,因此改善 2D 偵測不是解方。第三,驗證直接由影像回歸 3D 能取回深度軸,並確認這是跨架構的通用機制而非單一模型的性質,同時分離出真正起作用的成分是深度品質而非骨架密度。第四,把上述測量轉換成教練判定層級的錯誤率,說明部署現況會誤判多少合格動作,並據此提出依錯誤型態分流感測的設計原則。

## 二、相關研究

動作品質評估的主流做法是端到端的影片迴歸。以 VideoMAE 為代表的時空特徵抽取器在動作辨識上表現良好,套到品質判定時卻容易學到與品質無關的捷徑。本研究在 REHAB24-6 上重現了這個現象:VideoMAE 單獨使用的留一受試者平衡準確率為 0.536 ± 0.044,訓練集與測試集的落差達 0.34,遠高於骨架特徵的 0.10。在自建深蹲資料上,VideoMAE 單獨使用的多種子平均平衡準確率 0.555,而全部預測為正類的基準線 F1 已達約 0.83——F1 高不代表模型有判別力,這一點在正類偏多的品質資料上特別容易誤導。

姿態估計這一側,單目 3D 有兩條路。一條是先偵測 2D 關鍵點再抬升到 3D(2D-to-3D lifting),另一條是由影像直接回歸 3D 或人體網格,代表工作包括 MeTRAbs、HMR2.0/4D-Humans、Multi-HMR 與 NLF。本團隊先前的實驗顯示前者無法還原真實深度:以 TCN 與預訓練 VideoPose3D 抬升出的座標,在正確性分類上未能超越其 2D 來源。這個結果本身不足以證明深度是瓶頸,因為抬升失敗也可能只是該類方法的限制,本研究因此改採直接量測。

在可解釋性這一側,知識圖譜與檢索增強生成(retrieval-augmented generation, RAG)被用來約束語言模型的輸出,使建議可以追溯到文獻來源。本團隊的 x-coach 系統採取這個路線,細節見第三節。

標註品質是這個領域少被檢視的上限。本研究在 EgoExo-Fitness 上計算標註者間一致性,技術要點查核(technical keypoint verification)的 Krippendorff α 為 0.40,動作品質評分的序數 α 為 0.38,97 條準則中只有 28 條達到 α ≥ 0.4。其中要求受試者保持背部打直的準則 α = −0.07(n = 92),等於標註者之間沒有共識。原始論文報告的基準 F1 約 0.52 至 0.55,這個數字可能已經接近標籤雜訊的天花板而非模型能力的上限。本研究把主要量測建立在具物理真值的資料上,正是為了避開這層雜訊。

## 三、系統設計

x-coach 是本團隊開發的可解釋教練系統,分為感知層、知識層與生成層,前端為 React 搭配 Vite 與 TypeScript,後端為 FastAPI。

感知層以 MediaPipe Pose 逐幀輸出 33 個 2D 與 3D 地標,規則式偵測器取下肢的髖、膝、踝、腳跟與腳趾共 10 個關鍵點,計算膝關節角度、髖關節角度、軀幹前傾角與膝蓋前移比例等逐幀指標,再以閾值判定缺陷,例如膝蓋前移比例 0.10 判為輕度、0.30 判為重度。規則式路線的取捨很明確:它的準確度受限於幾何量測本身,但每一個判定都能指出是哪個關節在哪一幀違反哪一條閾值,這是端到端迴歸給不出來的。本研究第五節量測的,正是這些幾何量測的可信度。

知識層以 networkx 建構本地持久化的標記屬性圖並存為 GraphML,節點型別涵蓋動作、錯誤、解剖構造、風險與矯正方式,邊型別包括潛在錯誤、成因弱點、風險提升與矯正對應。目前的圖檔 sports_kg_v3 含 2,142 個節點與 3,285 條邊,由 Gemini 2.5 Flash 搭配結構化輸出從運動生物力學文獻與教科書抽取而成。檢索層以偵測到的錯誤術語為起點做多跳查詢,把邊分成階段、證據、成因、風險、矯正與品質影響等桶位;向量庫則以遞迴字元切塊建立,支援離線運行。

生成層將偵測結果與檢索到的知識送入語言模型,以 SSE 串流回覆使用者的追問。這一層的約束是硬性的:模型只能就已偵測到的缺陷,以及知識層檢索回來的成因與矯正建議作答,不得杜撰未偵測到的錯誤或風險。系統目前為單機示範等級,分析呼叫同步阻塞,尚無任務佇列與物件儲存。

這套系統的可信度全部壓在感知層的第一步。如果膝角讀數本身有 40° 的偏差,後面的知識檢索與語言生成再嚴謹,得到的也是一份有據可查的錯誤診斷。以下三節量測這一步。

## 四、量測方法

### 4.1 資料

Fit3D 提供 4 台同步相機與逐幀 3D 真值骨架,本研究使用其訓練分割中的 8 位受試者。主要分析涵蓋深蹲、硬舉與過頭推(thruster)三個動作,每動作 32 段影片,即 8 位受試者乘以 4 個機位。深蹲的重複動作層級分析為 40 次重複乘以 4 個機位,共 160 筆配對讀值。REHAB24-6 提供 2 台 RGB 相機與 16 台動作捕捉相機,10 位受試者、6 種復健動作、1,072 筆重複動作標註,其中正確 568 筆、錯誤 504 筆;分類實驗採留一受試者交叉驗證,因 P10 僅 16 個樣本而排除,實際為 9 折。

### 4.2 線索讀取與誤差分解

所有線索都用同一份公式計算,差別只在輸入的座標來源。以 3D 真值計算得到的讀數為基準,單視角 2D 的讀數由真值投影到各機位影像後計算,單目 3D 模型的讀數則由模型輸出的關節座標計算。誤差以絕對值對真值取平均,角度以度為單位,比值為無因次。

單視角管線的誤差可以寫成兩項之和:

    真實 2D 誤差 = 偵測器誤差(真實 2D − mocap 投影 2D) + 投影誤差(mocap 投影 2D − 3D 真值)

其中 mocap 投影 2D 是把 3D 真值投影到影像,代表一個誤差為零的完美偵測器。真實 2D 由 RTMPose(rtmlib Wholebody, RTMW-x-L)在同一批影格上推論而得,兩者的差就是偵測器項。RTMPose 在約 900 像素的影像上,關鍵點絕對誤差為 55 至 59 像素。

為了避免結論被絕對尺度的系統偏差污染,跨模型比較採用對偏移容忍的指標:深度軸誤差與平面內誤差的比值(ez/exy)、旋轉不變的關節角度,以及扣除各機位平均偏移後的判定翻轉率。

### 4.3 判定保真度

線索誤差幾度並不直接說明教練會不會判錯。本研究因此在每次重複動作的谷底極值上,以相同閾值分別用真值、單視角 2D 與單目 3D 讀數判定通過或犯規。翻轉率指判定與真值相反的比率,假警報率指合格動作被判為犯規的比率,漏判率則相反,指真正犯規卻被放過的比率。這三個數字必須一起看,因為把閾值調鬆可以壓低假警報,代價是漏判上升。深蹲深度的真實犯規盛行率為 8%。另計算一組扣除各機位相對真值的平均偏移後的校正版本,這個校正使用真值計算,不是可部署的方案,而是單視角 2D 經過完美個人化校正後的效能上限。

### 4.4 單目 3D 模型

比較的模型涵蓋密集網格與稀疏骨架兩類。NLF 以 detect_smpl_batched 於半解析度推論,輸出 SMPL-24 對映至 Human3.6M-17;HMR2.0/4D-Humans 為參數式 SMPL transformer;Multi-HMR 為單階段全幀 SMPL-X;MeTRAbs 輸出稀疏 SMPL-24 並使用真實相機內參;MediaPipe 輸出稀疏 33 點並以啟發式方式估計 z 軸。左右方向依真值解析,深蹲上這項校正使 MPJPE 從 317.2mm 降至 77.7mm,相差 4.1 倍。

## 五、結果

### 5.1 單視角 2D 的矢狀面線索被視角破壞

同一次深蹲從四個機位讀出的膝關節角度差異極大。表一列出各線索相對 3D 真值的合併絕對誤差,以及視角間標準差與重複動作間標準差的比值,後者可視為訊噪比的倒數:比值大於 1 表示換個機位造成的讀數變化比不同人蹲得好不好造成的變化還大。

**表一、單視角 2D 生物力學線索相對 mocap 真值的失真(深蹲,40 次重複 × 4 機位 × 8 位受試者)**

| 線索 | 合併 MAE | 偏差 | 相關係數 r | 視角雜訊/訊號 | 判讀 |
|---|---|---|---|---|---|
| 膝關節角度 | 42.4° | +41.2° | 0.60 | 1.21 | 視角破壞 |
| 髖關節角度 | 40.7° | — | — | 0.84 | 視角敏感 |
| 深度比 | 0.25 | — | — | 1.21 | 視角破壞 |
| 軀幹前傾角 | 21.7° | −21.7° | 0.87 | 0.37 | 視角穩健 |
| 膝寬比(膝內扣) | 0.02 | — | 0.90 | 0.44 | 視角穩健 |

膝角與深度比的雜訊訊號比皆為 1.21,單視角讀數在這兩個線索上帶不動有意義的判別。逐機位來看,膝角的絕對誤差在 30.9° 到 58.1° 之間,取決於相機擺哪裡。一個具體的例子說明份量:某次蹲到底的真實膝角為 78°,四個機位分別讀成 108°、118°、119° 與 133°,沒有任何一個機位讀到接近真值,而 78° 與 108° 分屬蹲得夠深與蹲得不夠深兩種判定。額狀面的軀幹前傾與膝寬比則穩定得多,雜訊訊號比分別為 0.37 與 0.44。

### 5.2 誤差來自投影,不是偵測器

表二把單視角誤差拆成兩項。深蹲膝角上,真實偵測器讀出 17.63°,完美偵測器讀出 18.34°,兩者相差 −0.70°;硬舉為 −0.28°,過頭推為 +0.44°。逐受試者計算的偵測器項為 −0.7 ± 1.5°、−0.3 ± 1.4° 與 +0.4 ± 1.3°,全部落在受試者間標準差之內,統計上與零不可區分。換句話說,把 RTMPose 換成一個零誤差的理想偵測器,膝角讀數不會變好。

**表二、單視角 2D 線索誤差分解(誤差對 mocap 3D 真值,每動作 32 序列 / 8 位受試者)**

| 動作 | 線索 | 真實 2D | 完美 2D | 最佳 3D | 偵測器項 | 判讀 |
|---|---|---|---|---|---|---|
| 深蹲 | 膝角 | 17.63° | 18.34° | 7.09° | −0.70° | 需要 3D |
| 深蹲 | 髖角 | 19.42° | 18.18° | 9.26° | +1.24° | 需要 3D |
| 深蹲 | 膝內扣 | 0.07 | 0.04 | — | +0.03 | 改良 2D |
| 硬舉 | 膝角 | 14.45° | 14.72° | 7.51° | −0.28° | 需要 3D |
| 硬舉 | 髖角 | 25.16° | 17.75° | 7.78° | +7.41° | 需要 3D |
| 硬舉 | 膝內扣 | 0.07 | 0.03 | — | +0.04 | 改良 2D |
| 過頭推 | 膝角 | 17.71° | 17.26° | 5.91° | +0.44° | 需要 3D |
| 過頭推 | 膝內扣 | 0.07 | 0.04 | — | +0.03 | 改良 2D |

膝內扣的走向相反:三個動作的偵測器項皆為正值,誤差由偵測器主導,改良 2D 就能改善。這是後續分流設計的依據。

下游任務給出一致的答案。REHAB24-6 正確性分類上,HRNet-w48 的留一受試者平衡準確率 0.575 ± 0.075,RTMPose 為 0.570 ± 0.051,配對差值 +0.005 ± 0.071,Wilcoxon 檢定 p = 0.734。換上更強的 2D 骨幹,判別力沒有改變。

### 5.3 直接 image-to-3D 取回深度軸

NLF 在 Fit3D 深蹲上的 MPJPE 為 77.7mm,對齊後為 65.4mm,深度軸誤差 42.4mm,與平面內軸的比值 ez/exy = 1.16。深度軸與平面內軸誤差相當,這正是單視角 2D 做不到的事。線索層級的改善見表三。

**表三、單視角 2D 與 NLF 的線索誤差(對 mocap 3D 真值)**

| 動作 | 線索 | 單視角 2D | NLF |
|---|---|---|---|
| 深蹲 | 膝角 | 18.42° | 7.09° |
| 深蹲 | 髖角 | 18.3° | 10.4° |
| 深蹲 | 軀幹前傾 | 12.5° | 5.4° |
| 硬舉 | 髖角 | 17.8° | 7.9° |
| 硬舉 | 軀幹前傾 | 13.9° | 5.5° |
| 過頭推 | 膝角 | 17.4° | 5.9° |
| 過頭推 | 軀幹前傾 | 6.9° | 4.0° |

矢狀面角度誤差大致減半,硬舉與過頭推的深度軸比值分別為 1.17 與 0.95,顯示這不是深蹲獨有的現象。

### 5.4 誤差如何變成錯誤的教練判定

表四把線索誤差換算成判定層級。原始單視角 2D 在 76% 的重複動作上給出與真值相反的深度判定,漏判率為 0,假警報率 82%——實際能蹲到標準的動作,有八成被系統告知蹲得不夠深。這是把 x-coach 目前的規則式偵測器直接部署在手機影片上會發生的事。

**表四、深蹲深度判定的保真度(160 筆讀值,真實犯規盛行率 8%)**

| 讀數來源 | 翻轉率 | 假警報率 | 漏判率 |
|---|---|---|---|
| 原始單視角 2D | 76% | 82% | 0% |
| 逐機位校正 2D | 16% | 14% | 42% |
| 原始 NLF | 7% | 0% | 92% |
| 逐機位校正 NLF | 7% | 7% | 0% |

逐機位校正把 2D 的翻轉率壓到 16%,代價是漏判率升到 42%,而且這個校正需要真值才算得出來,不是能部署的方案。即使給了這個上限,校正後的 2D 仍有 16% 的判定與真值相反,原因是殘餘誤差逐次重複變動,不是可以一次扣掉的常數。

NLF 的兩列需要分開讀。原始 NLF 的翻轉率雖然只有 7%,假警報為零,但漏判率高達 92%,亦即它幾乎放過了所有真正蹲不夠深的重複動作;翻轉率之所以仍低,是因為真實犯規只佔 8%,漏判在總量上不顯眼。真正兼顧兩端的是校正後的 NLF,假警報 7%、漏判 0%。這個對比說明單看翻轉率會被盛行率誤導。要強調的是,這一列與校正後的 2D 用的是同一套逐機位偏移校正,同樣需要真值才算得出來,因此它不是可部署的成績,而是效能上限;兩者對照的意義在於同一個上限之下 2D 仍卡在漏判 42%,NLF 卻能把兩類錯誤一起壓下去。差別的根源在殘餘誤差的形態:2D 的殘差逐次重複變動,扣不掉;NLF 的殘差接近一個固定偏移,扣得掉。逐受試者來看,校正後 NLF 的翻轉率在 8 位受試者中有 7 位不高於校正後的 2D(5 位嚴格較低、2 位持平、1 位高出 5 個百分點)。深蹲的逐受試者平均為 2D 76 ± 17% 對 NLF 7 ± 13%,過頭推為 69 ± 31% 對 22 ± 24%。

### 5.5 恢復能力是通用機制,關鍵成分是深度品質

若只測一個模型,無法排除深度恢復是該模型的偽跡。表五比較四種架構迥異的模型。

**表五、跨模型的深度恢復能力(深蹲)**

| 模型 | 輸出型態 | 深度來源 | ez/exy | 膝角誤差 |
|---|---|---|---|---|
| 單視角 2D | 稀疏 2D | 無 | — | 18.42° |
| MediaPipe | 稀疏骨架 | 啟發式 z | 1.61 | 14.18° |
| Multi-HMR | 密集網格 | 回歸 | 1.50 | 9.67° |
| HMR2.0 | 密集網格 | 回歸 | 1.57 | 7.88° |
| NLF | 密集網格 | 回歸 | 1.16 | 7.09° |
| MeTRAbs | 稀疏骨架 | 回歸(真實內參) | 1.29 | 6.37° |

NLF、HMR2.0 與 Multi-HMR 三者都恢復了各動作定義性的旋轉不變線索,深度恢復因此是通用機制。更關鍵的是最後兩列的對照:MediaPipe 與 MeTRAbs 同樣輸出稀疏骨架,前者膝角誤差 14.18°、判定翻轉率 24%(不優於校正後 2D 的 21%),後者膝角誤差 6.37°、翻轉率 13%,達到甚至超越密集網格模型的水準。起作用的成分是深度品質,不是網格密度或關節數量。這對部署有直接意義:輕量的稀疏模型只要深度是真的回歸出來的,就足以支撐深度判定。

### 5.6 下游正確性分類

表六為 REHAB24-6 的留一受試者結果,可視為上述量測在一個獨立任務上的檢核。

**表六、REHAB24-6 正確性分類的留一受試者平衡準確率(9 折)**

| 特徵來源 | 平衡準確率 |
|---|---|
| Vicon 骨架(真實 3D) | 0.702 ± 0.078 |
| NLF(直接 image-to-3D) | 0.668 ± 0.044 |
| MediaPipe 骨架 | 0.633 ± 0.055 |
| HRNet-w48(2D) | 0.575 ± 0.075 |
| RTMPose(2D) | 0.570 ± 0.051 |
| VideoMAE(影片特徵) | 0.536 ± 0.044 |

排序與前述量測一致:具真實 3D 的動作捕捉最高,直接 image-to-3D 次之,單目 2D 骨架再次之,純影片特徵最低。NLF 相對 MediaPipe 的配對差值為 +0.034 ± 0.063,p = 0.203,方向一致但 n = 9 不足以達到統計顯著,這一點本研究不做過度解讀。時間平滑的結果同樣呈現動作特異性:Savitzky–Golay 平滑對全動作聚合沒有效益(視窗 7 與 11 的配對差值分別為 −0.0148 與 −0.0130,p 皆大於 0.30),但在深蹲上穩定帶來 +0.024,在手臂外展上反而下降,兩者互相抵銷。

## 六、討論

### 6.1 依錯誤型態分流感測

本研究的量測指向一個不對稱的設計原則。矢狀面的深度與屈曲判定,誤差來自投影幾何,再強的 2D 偵測器也補不回來,必須換成能直接回歸深度的模型。額狀面的膝內扣則相反,偵測器項為正,單視角 2D 的表現甚至優於單目 3D:掃描閾值後的判定翻轉率,深蹲上 2D 為 15% 而 NLF 為 31%,硬舉 9% 對 17%,過頭推 9% 對 20%。把所有線索一律送進 3D 模型並不划算,正確的做法是依錯誤型態決定走哪一條管線。

這條原則落在 x-coach 上的意義很具體。系統目前的規則式偵測器完全建立在 MediaPipe 的單目讀數上,對膝內扣類的判定可以維持現狀,對蹲深不足類的判定則不應直接輸出給使用者,因為 82% 的假警報率會摧毀使用者對系統的信任,而且錯誤的診斷仍會被知識層完整地補上成因與矯正建議,看起來反而更有說服力。可解釋性放大了感知層錯誤的傷害,而不是稀釋它。

### 6.2 部署上的取捨

MeTRAbs 的結果讓輕量部署成為可能。深度恢復不需要密集網格,一個稀疏骨架模型只要深度是真的回歸出來的就夠用,這對手機端或邊緣裝置的推論成本是好消息。校正路線則不可行:逐機位校正確實把 2D 的翻轉率從 76% 壓到 16%,但它需要動作捕捉真值才算得出來,而且殘餘誤差逐次重複變動,校正無法根除。

### 6.3 限制

Fit3D 的四台相機均為約 ±45° 的斜角,沒有真正的正側面視角。正側面是教練實際拍攝深蹲的慣用機位,膝關節屈曲在該視角下的可讀性應優於本研究測得的結果,因此表一與表二的 2D 誤差可能高估了實務上的失真程度。這是本研究最需要後續補強的一點,現實中拍攝角度是否落在斜角區間,決定了結論的適用範圍。

判定保真度的 160 筆讀值來自 40 次重複的 4 個機位,彼此不獨立,本研究未對其做顯著性檢定,表四的數字應視為描述性統計。REHAB24-6 的 NLF 對 MediaPipe 比較 n = 9,p = 0.203,方向一致但未達顯著。跨模型比較中,MeTRAbs 使用真實相機內參而 NLF 使用假設視場角,絕對深度的毫米比較有對等性疑慮,故排序改用旋轉不變與內參無關的指標;Multi-HMR 使用非原生前處理與假設視場角,不構成乾淨的頭對頭比較,其排名不應用來否定全幀模型的深度能力。NLF 對相機距離低估約 18%,絕對尺度有偏,本研究的角度與比值指標不受此影響,但涉及絕對長度的應用需另行校正。

## 七、結論

單目動作品質評估的瓶頸在深度,不在 2D 精度。本研究以誤差分解在動作捕捉真值上直接證明了這一點:深蹲膝角的偵測器項為 −0.70°,落在受試者間標準差之內,14° 至 18° 的線索誤差幾乎全部來自投影幾何。這個誤差不是學術上的小數點問題,它讓部署現況把 82% 的合格深蹲判為蹲深不足。直接由影像回歸 3D 能取回深度軸,把矢狀面角度誤差減半,並使殘餘誤差收斂成一個扣得掉的固定偏移,讓閾值校正得以同時壓低假警報與漏判——這是單視角 2D 在同一套校正下做不到的。此能力見於 NLF、HMR2.0、Multi-HMR 與 MeTRAbs 四種架構,關鍵成分是深度品質而非骨架密度。

綜合上述實驗,本研究建議動作品質系統依錯誤型態分流:矢狀面判定走直接 image-to-3D,額狀面判定留在改良後的 2D。後續工作將補上正側面視角的量測,把稀疏且具真實深度的模型接入 x-coach 的感知層,並在真實使用者的手機影片上驗證判定保真度是否維持。

---

## 參考文獻

> 初稿註記:以下為本研究實際使用的資料集與模型,出處需在投稿前逐筆核對卷期與頁碼。

[1] Fieraru, M., Zanfir, M., Pirlea, S. C., Olaru, V., Sminchisescu, C. AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training. CVPR, 2021.(Fit3D 資料集)

[2] REHAB24-6: A Multimodal Dataset for Physical Rehabilitation Exercise Assessment, 2024.

[3] Sárándi, I., Pons-Moll, G. Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation. NeurIPS, 2024.(NLF)

[4] Sárándi, I., Linder, T., Arras, K. O., Leibe, B. MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation. IEEE T-BIOM, 2021.

[5] Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., Malik, J. Humans in 4D: Reconstructing and Tracking Humans with Transformers. ICCV, 2023.(HMR2.0 / 4D-Humans)

[6] Baradel, F., et al. Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot. ECCV, 2024.

[7] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., Grundmann, M. BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204, 2020.(MediaPipe Pose)

[8] Jiang, T., et al. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. arXiv:2303.07399, 2023.

[9] Tong, Z., Song, Y., Wang, J., Wang, L. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS, 2022.

[10] EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding. ECCV, 2024.

[11] Vakanski, A., Jun, H.-p., Paul, D., Baker, R. A Data Set of Human Body Movements for Physical Rehabilitation Exercises. Data, 3(1):2, 2018.(UI-PRMD)

[12] Krippendorff, K. Content Analysis: An Introduction to Its Methodology. 4th ed., SAGE, 2018.
