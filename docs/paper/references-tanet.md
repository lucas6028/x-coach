# TANET 投稿論文：引用稽核與參考文獻

**唯一依據是 `docs/paper/TANET_論文-陳皓平.pdf`。** 章節位置、引用順序與編號都以該 PDF 正文為準。
`TANET_論文-陳皓平.docx` 與 `TANET投稿論文.md` 皆為已作廢舊稿，不同步、不作依據。

PDF 第 5 頁現存的「參考文獻」是 TANET 範例樣板（Eason、Maxwell、Jacobs…），與內容無關，整段要換掉。

格式規則：作者六位以上用 et al.，其餘全列；題名除專有名詞與縮寫外僅首字大寫。

---

## 〇、逐處定位（依 PDF 頁序，21 處）

「原句」為 PDF 現況，「改成」為插入／改號後的樣子。頁碼以 PDF 實際頁面計。

| # | 頁 | 節 | 原句（定位用） | 改成 |
|---|---|---|---|---|
| 1 | 1 | 1 前言 | 但最終輸出仍常以整段影片的分數為中心 **[1]**。 | 不變 |
| 2 | 1 | 1 前言 | 可改善知識密集任務的事實性與可追溯性 **[4]**； | …可追溯性 **[2]**； |
| 3 | 1 | 1 前言 | 修正之間的關係 **[5]**。 | …的關係 **[3]**。 |
| 4 | 1 | 2.1 | 說明語意描述可協助品質表徵 **[1]**。 | 不變 |
| 5 | 1 | 2.1 | 以姿態對比學習與動作解耦處理細微錯誤 **[2]**。 | …細微錯誤 **[4]**。 |
| 6 | 1 | 2.1 | 做得如何因而能一起評估 **[6]**。 | …一起評估 **[5]**。 |
| 7 | 1 | 2.1 | 指出錯誤復健動作中需要關注的關節 **[7]**。 | …的關節 **[6]**。 |
| 8 | 1 | 2.2 | 並針對行動裝置的即時推論設計 **[8]**。 | …即時推論設計 **[7]**。 |
| 9 | 2 | 2.2 圖 1 圖說 | 圖 1 MediaPipe Pose 的 33 個節點。 | 圖 1　MediaPipe Pose 的 33 個關鍵點與五個關節角。來源：Chen et al. **[8]**，依 CC BY 4.0 使用。 |
| 10 | 2 | 2.3 | 依查詢取得較新且可追溯的內容 **[4]**。 | …的內容 **[2]**。 |
| 11 | 2 | 2.3 | 處理跨文件、跨概念的關聯問題 **[5]**。 | …的關聯問題 **[3]**。 |
| 12 | 2 | 2.3 末段 | 本研究將工具呼叫的名稱、查詢與回傳的概念或文件來源隨對話儲存， | 段首補回 RAGAS 整句 **[9]**（全文見下方 B-2） |
| 13 | 2 | 3.1 | 主要深蹲資料使用資料集為 Fitness-AQA，包含 1,739 支有標註影片， | …資料集為 Fitness-AQA **[4]**，包含… |
| 14 | 2 | 3.1 | 逐次動作邊界及正確／錯誤標籤 **[11]**。 | …正確／錯誤標籤 **[10]**。 |
| 15 | 2 | 3.1 | 技術要點與關節層級回饋的外部驗證 **[3], [6], [7]**。 | …外部驗證 **[11], [5], [6]**。 |
| 16 | 2 | 3.2 | 每一個動作偵測器之規則皆從相關論文或期刊中擷取，確保每項規則皆具備生物力學依據、精確無幻覺。 | 句尾接：…精確無幻覺；以深蹲為例，膝外翻 **[12]**、膝前移 **[13]**、蹲深 **[14]**、軀幹過度前傾 **[15]** 與腳跟離地 **[16]** 五條規則各有對應的生物力學依據。 |
| 17 | 3 | 4.2 | 舉例來說，肩胛翼狀突出要看肩胛骨相對於胸廓的位置， | …肩胛翼狀突出 **[17]** 要看… |
| 18 | 3 | 4.2 | 聳肩代償則是量到了錯的骨頭， | 聳肩代償 **[18]** 則是… |
| 19 | 3 | 4.2 | MediaPipe 的肩點是盂肱關節而非肩峰。 | …而非肩峰 **[7]**。 |
| 20 | 3 | 4.3 | 深蹲的五條規則中有三條對得上 Fitness-AQA 的錯誤標註。 | …對得上 Fitness-AQA 的錯誤標註 **[4]**。 |
| 21 | 3 | 4.3 | 並把同一組規則換到 RTMPose 骨架上重播作為對照。 | …換到 RTMPose 骨架 **[19]** 上重播作為對照。 |
| 22 | 4 | 4.4 | 直接由影像估計的 NLF 三維骨架為 0.668， | …NLF 三維骨架 **[20]** 為 0.668， |

選配三處（C 類）：
第 1 頁 2.2 節「投影、遮擋與深度模糊都躲不掉」可掛 **[20]**；
第 4 頁 4.4 節「以 MediaPipe 的 Lite 模型畫即時骨架，分析則預設走 Heavy」可掛 **[7]**；
第 5 頁 5 節「或改用能取回離面深度的姿態估計」可掛 **[20]**。

## 一、引用稽核：文章哪裡需要引用

### A. 已有標記，只需重新編號（11 處）

| 節 | 位置 | 舊 | 新 |
|---|---|---|---|
| 1 前言 | 早期 AQA 工作…分數為中心 | [1] | [1] |
| 1 前言 | 檢索增強生成…事實性與可追溯性 | [4] | **[2]** |
| 1 前言 | 圖結構檢索則保存動作、錯誤…的關係 | [5] | **[3]** |
| 2.1 | Parmar 與 Morris…多任務架構 | [1] | [1] |
| 2.1 | Fitness-AQA…姿態對比學習與動作解耦 | [2] | **[4]** |
| 2.1 | EgoExo-Fitness 的標註更細 | [6] | **[5]** |
| 2.1 | ExeChecker 走的是另一條路 | [7] | **[6]** |
| 2.2 | MediaPipe BlazePose…33 個人體關鍵點 | [8] | **[7]** |
| 2.3 | RAG 把生成模型接上外部文件索引 | [4] | **[2]** |
| 2.3 | 圖式 RAG 則把文件中的實體及關係組織成圖 | [5] | **[3]** |
| 3.1 | 外部驗證使用 REHAB24-6 | [11] | **[10]** |
| 3.1 | FLEX、EgoExo-Fitness 與 ExeChecker 留待後續 | [3],[6],[7] | **[11],[5],[6]** |

### B. 必補：目前完全沒有引用，但一定要有（10 處）

**1. 2.2 節 圖 1 — 圖片來源｜新 [8]**

圖 1 取自他人論文（含 elbow／shoulder／hip／knee／ankle 五個角度弧線），未標來源。
該圖以 CC BY 4.0 授權可以使用，但必須標明出處與授權。圖說改成：

> 圖 1　MediaPipe Pose 的 33 個關鍵點與五個關節角。來源：Chen et al. [8]，依 CC BY 4.0 使用。

現行圖說「33 個節點」與圖不符（圖上另標五個關節角，而 2.2 節正文用的是髖—膝—踝夾角、
膝踝相對位移與軀幹角度）。另一條路是依 [7] 的定義自行重繪只有 33 點的圖，圖說寫
「依 [7] 之定義繪製」，此時 [8] 不需要。

**2. 2.3 節 — RAGAS 整句被刪｜新 [9]**

現行段落開頭突兀（直接由「本研究將工具呼叫的名稱…」起始）。整段換回：

> 生成文字的流暢度不等於忠實度。RAGAS 把檢索相關性、回答忠實度與回答品質分開評估，提供了不依賴完整人工標準答案的 RAG 評估方向 [9]。本研究據此把來源保留寫進系統的資料結構：工具呼叫的名稱、查詢與回傳的概念或文件來源都隨對話儲存，檢索工具若沒有來源，介面就不顯示引文。這降低了生成內容在沒有證據時被當成專業結論的風險，但 RAG 並未因此消除幻覺，仍須以專家評閱與自動化忠實度指標驗證。

第 5 節「需要檢索品質指標與領域專家評閱」即由此得到出處。

**3. 3.1 節 — Fitness-AQA 資料集｜沿用 [4]**

「主要深蹲資料使用資料集為 Fitness-AQA，包含 1,739 支有標註影片…另有 4,970 支未標註影片」
整段沒有任何標記，1,739／1,623／4,970 這些數字看起來像本研究自產。方法章首次介紹資料集
必須帶引用：

> 主要深蹲資料使用資料集為 Fitness-AQA [4]，包含 1,739 支有標註影片

**4. 3.2 節 — 「規則皆從相關論文或期刊中擷取」｜新 [12]–[16]**

> 每一個動作偵測器之規則皆從相關論文或期刊中擷取，確保每項規則皆具備生物力學依據、精確無幻覺。

**這是全篇最需要引用的一句**：宣稱 48 條規則都有文獻依據，卻一條文獻都沒列，等於要審查人
相信你。規格文件 `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`
其實逐條附了 61 筆 citation，把深蹲五條規則的來源搬上來當代表即可（深蹲也正是 4.3 節唯一
驗證的動作）：

> 每一個動作偵測器之規則皆從相關論文或期刊中擷取，確保每項規則皆具備生物力學依據、精確無幻覺；以深蹲為例，膝外翻 [12]、膝前移 [13]、蹲深 [14]、軀幹過度前傾 [15] 與腳跟離地 [16] 五條規則各有對應的生物力學依據。

**5. 4.2 節 — 肩胛翼狀突出｜新 [17]**

> 舉例來說，肩胛翼狀突出 [17] 要看肩胛骨相對於胸廓的位置

**6. 4.2 節 — 聳肩代償｜新 [18]**

> 聳肩代償 [18] 則是量到了錯的骨頭

**7. 4.2 節 — MediaPipe 肩點為盂肱關節｜沿用 [7]**

> MediaPipe 的肩點是盂肱關節而非肩峰 [7]

**8. 4.3 節 — 錯誤標籤定義｜沿用 [4]**

knees forward／knees inward／shallow depth 是 Fitness-AQA 定義的標籤，4.3 節首次以中文
「膝蓋前移（knee forward）」出現時應回指 [4]。

**9. 4.3 節 — RTMPose｜新 [19]**

表 2 與表 3 都拿 RTMPose 當對照，正文從未給出處。首次出現處：

> 並把同一組規則換到 RTMPose 骨架 [19] 上重播作為對照

**10. 4.4 節 — NLF｜新 [20]**

表 3 的最高分列同樣缺出處。首次出現處：

> 直接由影像估計的 NLF 三維骨架 [20] 為 0.668

### C. 建議補、不補也能過（審查人可能問）

- **2.2 節**「投影、遮擋與深度模糊都躲不掉」目前無出處。可掛 [20]（直接 image-to-3D 正是為此），
  或明講這是本研究 4.3／4.4 節的觀察。
- **5 節**「改用能取回離面深度的姿態估計」→ [20]。
- **4.4 節** MediaPipe Lite／Heavy 兩個模型變體 → [7] 或 MediaPipe 官方文件。
- **1 節**「AQA 因此嘗試由影片預測表現分數或技術品質」只掛 [1]，可再加一篇 AQA 綜述。
- 關鍵詞列了「可解釋人工智慧」，但全文沒有任何 XAI 引用。

---

## 二、參考文獻清單（20 筆，依內文引用順序）

[1] P. Parmar and B. T. Morris, "What and how well you performed? A multitask learning approach to action quality assessment," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 304–313.

[2] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, vol. 33, 2020.

[3] D. Edge et al., "From local to global: A graph RAG approach to query-focused summarization," arXiv:2404.16130, 2024.

[4] P. Parmar, A. Gharat, and H. Rhodin, "Domain knowledge-informed self-supervised representations for workout form assessment," in *Computer Vision – ECCV 2022*, vol. 13698, 2022, pp. 105–123.

[5] Y.-M. Li et al., "EgoExo-Fitness: Towards egocentric and exocentric full-body action understanding," arXiv:2406.08877, 2024.

[6] Y. Gu, M. Patel, and M. Betke, "ExeChecker: Where did I go wrong?" arXiv:2412.10573, 2024.

[7] V. Bazarevsky et al., "BlazePose: On-device real-time body pose tracking," arXiv:2006.10204, 2020.

[8] H. Chen, M. C. Leu, M. Moniruzzaman, Z. Yin, and S. Hajmohammadi, "Advancements in repetitive action counting: Joint-based PoseRAC model with improved performance," arXiv:2308.08632, 2023.

[9] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated evaluation of retrieval augmented generation," arXiv:2309.15217, 2023.

[10] A. Černek, J. Sedmidubský, and P. Budíková, "REHAB24-6: Physical therapy dataset for analyzing pose estimation methods," in *Proc. 17th Int. Conf. Similarity Search and Applications (SISAP)*, 2024, pp. 18–33.

[11] H. Yin et al., "FLEX: A large-scale multi-modal multi-action dataset for fitness action quality assessment," arXiv:2506.03198, 2025.

[12] K. R. Ford et al., "An evidence-based review of hip-focused neuromuscular exercise interventions to address dynamic lower extremity valgus," *Open Access J. Sports Med.*, 2015. PMC4556293.

[13] M. Zellmer et al., "Patellar tendon stress between two variations of the forward step lunge," *J. Sport Health Sci.*, 2019. PMC6523035.

[14] H. Hartmann, K. Wirth, and M. Klusemann, "Analysis of the load on the knee joint and vertebral column with changes in squatting depth and weight load," *Sports Med.*, vol. 43, no. 10, pp. 993–1008, 2013.

[15] V. M. Moreira et al., "Analysis of muscle strength and electromyographic activity during different deadlift positions," *Muscles*, 2023. PMC12225233.

[16] A. J. Mata, H. Hayashi, P. A. Moreno, R. I. Dudley, and E. A. Sorenson, "Hip flexion angles during supine range of motion and bodyweight squats," *Int. J. Exerc. Sci.*, vol. 14, no. 1, pp. 912–918, 2021.

[17] S. Lee, D. Lee, and J. Park, "The effect of hand position changes on electromyographic activity of shoulder stabilizers during push-up plus exercise on stable and unstable surfaces," *J. Phys. Ther. Sci.*, vol. 25, no. 8, pp. 981–984, 2013.

[18] W.-L. Mun, E.-Y. Jung, S. Lei, and S.-Y. Roh, "Scapular muscle activation at different shoulder abduction angles during Pilates reformer arm work exercise," *Medicina*, vol. 61, no. 4, art. 645, 2025.

[19] T. Jiang et al., "RTMPose: Real-time multi-person pose estimation based on MMPose," arXiv:2303.07399, 2023.

[20] I. Sárándi and G. Pons-Moll, "Neural localizer fields for continuous 3D human pose and shape estimation," in *Advances in Neural Information Processing Systems*, 2024. arXiv:2407.07532.

---

## 三、備註與取捨

- PDF 現行編號不連續（用到 [11] 但 [9]、[10] 從未出現），是舊稿的 [9] VideoMAE V2 與 [10] RAGAS
  兩處引用被刪後未重編所致。VideoMAE 在 PDF 已完全沒有提及，該筆不需保留。
- 二十筆皆已正式出版或為 arXiv 預印本，無「未出版(unpublished)」或「即將出版(in press)」情形。
- 一手核對過的：[8]（arXiv:2308.08632，CC BY 4.0）、[17]（PMC3820220）、[18]（PMC12029123）、
  [19]（arXiv:2303.07399）、[20]（arXiv:2407.07532）。
  [12]–[16] 的期刊、年份與 PMC／DOI 取自規格文件 `2026-07-18-16-movement-rule-detector-design.md`
  的逐條 citation 欄位，題名與作者未再回原始頁核對，投稿前建議查一次。
- **精簡版**：若嫌 20 筆太多，3.2 節只留 4.3 節真正有結果的兩條規則來源（膝前移、膝內夾），
  刪掉 [14]–[16]，總數 17 筆，其後編號各減三。
- 圖 1 來源是靠圖說完全相同比對出來的（ResearchGate 圖片頁 fig1_373246080 對應 arXiv:2308.08632）。
  你給的連結是 publication 383542754，ResearchGate 擋掉自動存取（HTTP 403），無法確認掛在哪篇底下。
  請自行開該頁確認是不是 Chen et al.；若是別篇，授權要重新確認，[8] 也要換。
