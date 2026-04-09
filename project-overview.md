
# 專案概覽：結合知識檢索與多模態推理的可解釋性 AI 教練框架

## 題目
**《結合知識檢索與多模態推理的可解釋性 AI 教練框架》**

## 核心問題
傳統 AQA (Action Quality Assessment) 只給分不給建議；通用 LLM 給建議但不準確且容易產生幻覺。

## 研究方法

### 1. 知識庫構建
爬取並結構化運動生物力學文獻、規則手冊，建立向量資料庫 (Vector Database) 與知識圖譜 (Knowledge Graph)。

### 2. 系統架構設計
本研究提出的 AI 教練系統由四個核心模組組成，形成從「細粒度感知」到「有據推理」的閉環：

#### A. 多模態感知模組 (Multimodal Perception Module)
*   **關注「身體怎麼動、關節怎麼對齊」**： 結合 MediaPipe 與 GCN，提取骨架關鍵點並進行姿勢引導對齊 (Pose-Guided Alignment)，將「膝蓋內扣」等幾何特徵直接映射至語義空間。
*   **關注「動作看起來對不對、順不順」**： 採用 VideoMAE 提取時空特徵，並透過細粒度對比學習 (Fine-grained Contrastive Learning)，利用 Hard Negative Mining 技術區分「標準動作」與「細微錯誤動作」。
*   **時間定位**： 識別錯誤發生的具體時間戳 (Start-End Timestamp)，實現「針對特定幀」的精確分析。

#### B. GraphRAG Knowledge 模組
*   **建構領域專用知識圖譜**： 儲存運動生物力學規則、解剖學知識與訓練處方。
*   **FKG (Fitness Knowledge Graph) 擴充**： 使用 FLEX dataset 中的 FKG 為基礎，結構化連結「動作→錯誤→改正方向」，並根據運科期刊文獻動態更新節點。
*   **Multi-hop Reasoning**： 以感知模組輸出的「錯誤術語標籤」為 Query，觸發 GraphRAG 機制，從錯誤現象追溯至深層肌肉成因（如：膝內扣→臀中肌無力）。

#### C. 推理與生成模組 (Reasoning & Generation Module)
*   **核心功能**： 將感知數據與專業知識轉化為「可解釋、可執行」的教練建議。
*   **技術架構**：
    *   **Visual-Language Adapter (視覺-語言適配器)**： 引入一個輕量級映射層 (Linear Projection / MLP)，將 VideoMAE 提取的高維視覺特徵 (Visual Embeddings) 對齊至 LLM 的語義空間。這讓模型不僅知道「有錯誤」，還能感知錯誤的「嚴重程度」與「動態變化」。
    *   **Physics-Informed Prompting (物理感知提示工程)**： 設計包含「角色設定 (Persona)」、「動作上下文 (Context)」、「檢索知識 (Retrieved Knowledge)」與「視覺特徵 (Visual Tokens)」的結構化提示，確保生成內容符合生物力學原理。
    *   **Chain-of-Thought (思維鏈)**： 強制模型按照「觀察現象 (Observation) → 歸因分析 (Reasoning) → 給出處方 (Prescription)」的邏輯生成回應，避免直接跳轉結論。
    *   **LLM Backbone**： 採用 LLaMA-3 (8B/70B) 或 Qwen-72B 作為推理核心，具備強大的邏輯推理與多語言能力。
*   **輸出格式**：
    *   **診斷報告**： 包含動作評分 (Score)、主要錯誤 (Major Errors) 與 風險評估 (Risk Assessment)。
    *   **修正指導**： 針對每個錯誤提供具體的 Hero Action (如：「試著把膝蓋往外推」) 與 輔助訓練建議 (如：「建議做彈力帶深蹲」)。

#### D. 互動式前端 (Interactive Frontend)
*   開發 Web 介面，支援視訊上傳、骨架可視化疊加及對話式指導，並能顯示錯誤發生的時間片段。

### 3. 評估指標與驗證方法
*   **動作評分一致性**： 使用 Spearman 相關係數 (ρ)，驗證 AI 評分與人類裁判 (Ground Truth) 之間的排序一致性。
*   **RAG 生成品質與幻覺檢測**： 採用 RAGAS 框架，檢測生成的建議是否嚴格基於檢索到的文獻，以及 GraphRAG 是否能精準抓取視覺錯誤特徵。
*   **使用者體驗測試**： 邀請 3-5 位不同經驗的使用者進行測試，評估「視覺化骨架」與「文字建議」的有效性。

## 專案時程規劃 (8個月)

本研究預計於 8 個月內完成，分為三個主要階段：

### 第一階段：基礎建設與資料構建 (Mon 1-3)
*   **Month 1**: 文獻回顧、定義動作標準與錯誤類型 (Domain Knowledge Definition)。
*   **Month 2**: 構建 GraphRAG 知識庫 (FKG + 運科文獻) 與 向量檢索系統。
*   **Month 3**: 搜集與標註動作影片數據，訓練基礎感知模型 (MediaPipe + GCN)。

### 第二階段：核心模型開發 (Mon 4-6)
*   **Month 4**: 開發 VideoMAE 視覺特徵提取器，進行細粒度對比學習訓練。
*   **Month 5**: 整合感知模組與 LLM，訓練 Visual-Language Adapter。
*   **Month 6**: 微調推理生成模組 (Prompt Engineering + Chain-of-Thought)，實現端到端推理。

### 第三階段：系統整合與評估 (Mon 7-8)
*   **Month 7**: 開發互動式 Web 前端，串接後端推理 API，進行系統整合測試。
*   **Month 8**: 進行使用者實驗 (User Study) 與 客觀指標評估 (RAGAS/Spearman)，撰寫論文與結案報告。

## 預期貢獻
實現第一個「有據可依」的 AI 教練，解決 AI 幻覺問題，提供真正具備生物力學依據的運動指導。
