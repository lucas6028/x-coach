# AI 教練整體 Pipeline 與 Agent Harness 設計

Status: **design proposal** · Created 2026-07-06
前置文件：`project-overview.md`（研究願景）、`specs/llm-chat-spec.md`（chat v1–v2.1）、
`docs/kg-schema-generalization.md` + `docs/movement-kg-expansion-plan.md`（多動作 KG）。

本文件回答三個問題：

1. **既有功能**：目前 AI 教練實際上做到了什麼（以程式碼為準，非願景）。
2. **未來功能**：依據研究發現與產品缺口，值得加入什麼。
3. **整體 pipeline 與 agent harness**：把上述功能組織成一個可分階段落地的
   系統架構——包含 LLM agent 的角色分工、工具介面、接地（grounding）契約、
   驗證與降級規則。

---

## 1. 現況盤點（既有功能）

### 1.1 目前的端到端流程（單發式，無 agent）

```
上傳影片 ──► /api/analyze（同步，semaphore 限流）
              │  1. MediaPipe Pose 抽 33 關鍵點（2D，model_complexity=2）
              │  2. pose_rule_detector：5 條深蹲規則（閾值式，平滑後分段）
              │  3. 每個 fault 附 KG（1-hop）或 RAG（hash-BoW top-k）檢索
              ▼
         result JSON（view/quality/detections/retrievals/pose overlay）
              │  （登入者 best-effort 存 Supabase：videos/analyses）
              ▼
前端 Studio ──► 骨架疊加 + 生物力學 HUD + fault 時間軸
              │  FaultCard：cause → risk → fix 因果階梯
              │  KnowledgeGraphWidget：radial 子圖
              ▼
/api/chat ──► 伺服器端組 system prompt（只餵 analysis 事實）
              └─ OpenRouter 單發串流補全（SSE），無 tool calling
                 + followup chips（獨立快速模型）
```

事實依據（探索報告已逐條以 path:line 驗證）：

| 模組 | 現況 | 關鍵位置 |
|---|---|---|
| 感知 | MediaPipe 2D、33 點；RTMPose/mmpose 僅研究路徑 | `src/pose/process_videos.py`、`src/pose/rtmpose_pose_extraction.py` |
| 判斷 | **只有 squat**；5 條規則（valgus、膝前移、深度、前傾、heel rise）；有 phase 標註、**無 rep 切分** | `src/pose/pose_rule_detector.py` |
| 知識 | KG 仍釘在 `squat_kg_v2.graphml`（v3 已建好未切換）；RAG 為 hash bag-of-words（非神經 embedding） | `backend/app/config.py:24`、`src/knowledge/rag_vector_db.py` |
| 推理 | 單發 grounded chat；「只能講事實、不得杜撰」規則寫死在 prompt；無工具、無多步推理 | `backend/app/services/chat.py:42-55` |
| 資料 | Supabase：videos / analyses（result JSONB）/ conversations（messages JSONB），RLS | `backend/app/services/store.py` |
| 已訓練模型 | VideoMAE、correctness MLP、2D→3D lifting 全是 research-only，**backend 一個都沒接** | `src/video/`、`src/rehab24/` |

### 1.2 這個架構的優點（要保留的設計原則）

- **接地優先**：LLM 只能就 pipeline 產出的事實說話，幻覺面積小——這是
  本專案的核心賣點（可解釋、有據可依），任何 agent 化都不能犧牲它。
- **檢索在分析期做完**：chat 延遲低、成本可控。
- **離線友善**：除 KG 抽取外全部可離線跑。

### 1.3 主要缺口

| # | 缺口 | 影響 |
|---|---|---|
| G1 | 只支援 squat；KG v3（多動作 schema）已建好但 app 未切換 | 多動作價值被卡住 |
| G2 | 無 rep 切分 → fault 只能以「連續片段」呈現，無法說「第 3 下太淺」 | 教練語言不自然、無法逐 rep 計分 |
| G3 | 2D 視角依賴：深度/屈曲類 cue 在斜視角下系統性失真（Fit3D 實驗已量化：raw 2D 對好深蹲的深度誤判 82%，direct 3D 降到 7%） | 深度判定可信度低 |
| G4 | RAG 是 hash-BoW，檢索品質天花板低 | 知識召回弱 |
| G5 | chat 是單發補全：不能查 KG、不能看數據、不能跨分析比較 | 追問深度受限（spec v3 已標記 deferred） |
| G6 | 分析是同步請求：長影片/多人併發會塞死 | 產品化瓶頸（程式註解已預告 Celery+Redis） |
| G7 | 沒有品質評估迴路（RAGAS、grounding 檢查、golden set） | 無法量化「有據可依」的程度 |

---

## 2. 未來功能（依模組分層）

### 2.1 感知層（Perception）

| 功能 | 內容 | 依據 / 前置 |
|---|---|---|
| **Rep 切分** | 以膝/髖角度序列的極值切 rep（squat 可直接用現有 knee-angle 訊號），輸出 per-rep metrics | G2；REHAB24 分支已有 per-rep 概念 |
| **依 fault 類型路由 2D/3D**（研究亮點 → 產品） | 深度/屈曲類 cue 走 direct image→3D（NLF 類模型，GPU 服務）；valgus 類走校正後 2D。Fit3D 實驗證明這個路由是對的：投影誤差 ~17° 只有 3D 能救，valgus 則是 detector 主導 | G3；需 GPU 推論服務 |
| **多動作規則包** | 每個旗艦動作（Lunge → Push-up → OHP → Row）一個 rule pack：`{metrics, phase model, fault rules, retrieval queries}`，介面與 squat 對齊 | G1；`movement-kg-expansion-plan.md` |
| **動作識別（movement ID）** | 上傳後自動判斷是哪個動作（先用姿勢統計特徵的輕量分類器），決定載入哪個 rule pack | 多動作的入口 |
| VideoMAE 特徵融合 | 規則抓不到的「順不順」類缺陷用學習式分類器補（video-level error classifier 已有研究雛形） | 研究計畫 Phase 2 |

### 2.2 知識層（Knowledge）

| 功能 | 內容 | 依據 |
|---|---|---|
| **KG v3 切換** | backend 改指 `sports_kg_v3.graphml` + movement-aware 檢索（scoped `Movement:Name` + shared pivot 層） | G1；v3 已存在 |
| **跨動作 multi-hop** | shared pivot（Cause/Cue/Risk）讓「臀中肌無力 → squat 與 lunge 都會 valgus」這類推理成立，是 agent 工具的殺手級查詢 | `kg-schema-generalization.md` |
| **神經 embedding RAG** | hash-BoW → sentence-transformer 類本地模型（保持離線原則），vector_db 介面不變 | G4 |
| KG 動態擴充 | 依新文獻增量抽取（現有 extract_kg + canonical mapping 流程），加 provenance 欄位供引用 | 研究願景 §B |

### 2.3 推理／互動層（Reasoning & Interaction）——本文件核心，見 §4

| 功能 | 內容 |
|---|---|
| **Tool-calling coach agent** | chat 從單發補全升級為工具迴圈：查 KG、查 RAG、讀 per-rep 數據、跨分析比較（= spec v3 的 live RAG，擴大範圍） |
| **結構化 CoT 報告** | 觀察（Observation）→ 歸因（Reasoning）→ 處方（Prescription）三段式，對應研究計畫的 Physics-Informed Prompting |
| **訓練處方生成** | fault → KG `CORRECTED_BY` → 矯正動作/劑量建議（drill library），可輸出成課表 |
| **跨分析記憶／進步追蹤** | 「比上週深了 5°、valgus 消失」——用 analyses 歷史做趨勢工具 |
| **Grounding 驗證器（critic）** | 回答送出前檢查每個主張都能對應到 evidence bundle（RAGAS-lite），不過關就重生成或降級 |

### 2.4 產品層（Delivery）

| 功能 | 內容 | 依據 |
|---|---|---|
| **非同步分析佇列** | Celery/Redis 或 Supabase queue；前端進度事件 | G6 |
| Per-rep UI | 時間軸按 rep 分格、逐 rep 評分卡 | 依賴 rep 切分 |
| 週報／進步儀表板 | 由 Progress 工具聚合，含趨勢圖 | 依賴跨分析記憶 |
| 語音輸入/朗讀 | composer 已預留 UI 槽位 | `CoachTray.tsx` |
| 即時 webcam 模式 | 輕量 pose + 即時 cue（僅規則，不走 LLM） | 遠期 |

---

## 3. 目標整體 Pipeline

```
                ┌────────────────────────────────────────────────────────┐
                │                    INGESTION（非同步）                   │
   upload ────► │  格式檢查 → 存儲 → job queue → 進度事件（SSE/poll）      │
                └───────────────┬────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────┐
                │                    PERCEPTION                          │
                │  view/quality gate → movement ID → pose 抽取            │
                │  （預設 MediaPipe 2D；深度類 cue 可路由 direct-3D 服務） │
                │  → rep 切分 → per-rep metrics                           │
                └───────────────┬────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────┐
                │                    ASSESSMENT                          │
                │  movement rule pack（規則）＋ 學習式分類器（選配）        │
                │  → detections[]：fault_id, rep, severity, evidence,     │
                │    frame span, observability                           │
                └───────────────┬────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────┐
                │                    KNOWLEDGE                           │
                │  movement-aware GraphRAG（sports_kg_v3 scoped+shared）  │
                │  ＋ 神經 embedding RAG                                  │
                │  → 每個 fault 一份 evidence bundle（含 node id 供引用）  │
                └───────────────┬────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────┐
                │              REASONING（agent harness，§4）             │
                │  Coach Agent（工具迴圈）→ 結構化報告 + 對話              │
                │  Critic 驗證 grounding → 通過才送出                     │
                └───────────────┬────────────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────────────────────────┐
                │                    DELIVERY                            │
                │  Studio（overlay/per-rep 時間軸/FaultCard/KG widget）    │
                │  Chat（SSE + tool 事件）· History · 進步儀表板           │
                └────────────────────────────────────────────────────────┘

                （旁路）EVALUATION：golden-set 規則回歸 · RAGAS faithfulness
                        · per-turn grounding score 記錄 → 迭代 prompt/規則
```

與現況的差異：INGESTION 改非同步；PERCEPTION 多了 movement ID、rep 切分、
3D 路由；KNOWLEDGE 換 v3 + 神經 RAG；REASONING 從「單發補全」變成
「agent harness」；新增 EVALUATION 旁路。**分析期做重檢索、對話期做輕檢索**
的原則保留：evidence bundle 仍在分析期建好，agent 工具只做增量查詢。

---

## 4. Agent Harness 設計

### 4.1 設計原則

1. **接地契約高於一切**：agent 的每個對使用者的主張，必須可追溯到
   （a）本次 analysis 的 detection/metric、（b）KG node/edge id、或
   （c）RAG chunk id。工具回傳值一律附 `fact_id`，最終回答附引用。
2. **單一編排者、少量專才**：一個 Coach Orchestrator 面對使用者，
   複雜子任務派給專才 agent；避免 agent 網狀互聊。
3. **便宜的路先走**：路由/followup 用快速模型，教練主回合用強模型，
   critic 用中等模型——沿用現行「followups 釘快速模型」的成功模式。
4. **失敗要降級不要硬撐**：critic 不過 → 重生成一次 → 再不過就退回
   「模板式事實陳述」（現有 FaultCard 內容），永遠有可用輸出。

### 4.2 角色與分工

```
使用者 ◄──SSE──► Coach Orchestrator（強模型，工具迴圈，每回合 ≤ N 次工具）
                     │
        ┌────────────┼──────────────┬────────────────┬──────────────┐
        ▼            ▼              ▼                ▼              ▼
   [工具集]     Knowledge      Progress         Program        Critic
                Navigator      Tracker          Designer       （每回合必跑）
                （多跳 KG      （跨 analysis    （處方/課表）    grounding 驗證
                  推理）         趨勢）
```

| 角色 | 模型檔位 | 職責 | 觸發 |
|---|---|---|---|
| **Coach Orchestrator** | 強（使用者可選，同現行 model picker） | 對話主體；決定叫哪些工具；產出 Observation→Reasoning→Prescription 結構的回答 | 每則使用者訊息 |
| **Knowledge Navigator** | 中 | 多跳 KG 查詢與歸納（如「valgus 的深層成因鏈」「哪些動作共享這個 Cause」），回傳結論 + node id 清單 | Orchestrator 判斷需要 >1-hop 知識時 |
| **Progress Tracker** | 中 | 拉使用者歷史 analyses，算趨勢（severity、深度角度、fault 消長），回傳摘要 + 數據點 | 使用者問進步/比較時 |
| **Program Designer** | 中 | 由 faults + `CORRECTED_BY` 子圖 + 器材限制生成矯正課表 | 使用者要訓練建議時 |
| **Critic / Grounding Verifier** | 中（固定，不可被使用者換掉） | 對最終回答逐主張比對 evidence bundle + 工具回傳；輸出 pass/fail + 違規句 | 每次回答送出前 |
| Followup Generator | 快（現行 gpt-oss-120b 模式保留） | 追問 chips | 回答完成後 fire-and-forget |

### 4.3 工具介面（Orchestrator 可呼叫）

| 工具 | 簽名（概念） | 資料來源 |
|---|---|---|
| `get_analysis` | `(analysis_id) → {view, quality, detections[], per_rep_metrics[]}` | analyses JSONB |
| `get_rep_detail` | `(analysis_id, rep_no) → 該 rep 的逐幀 metrics 摘要` | 分析產物（需開始保留 per-rep frame_metrics 摘要，取代現在整包丟棄） |
| `kg_query` | `(seed, movement, hops≤3, edge_filter) → 子圖 + node ids` | sports_kg_v3（現有 `/api/knowledge/graph` 內化為工具） |
| `rag_search` | `(query, top_k) → chunks + ids` | 神經 embedding vector_db |
| `compare_analyses` | `(id_a, id_b) → 指標差異表` | analyses 歷史 |
| `list_user_history` | `(limit) → 近期 analyses 摘要` | analyses 歷史 |
| `make_drill_plan` | `(fault_ids, equipment?) → 課表草案（含 KG 引用）` | KG `CORRECTED_BY` + drill library |
| `cite_frame` | `(t_start, t_end) → 前端可深連結的影格參照` | 前端 seek 整合（FaultCard 點擊跳轉的既有機制泛化） |

實作備註：OpenRouter 走 OpenAI-compatible function calling；SSE 事件流在現有
`delta/done/error` 之上加 `tool_start/tool_result`，前端可顯示「正在查知識圖譜…」。
工具呼叫軌跡（trace）連同 messages 存進 conversations JSONB，可重播、可稽核。

### 4.4 Harness 運行契約（借鏡本機 dev harness 的派工三件套）

每次 Orchestrator 派工給專才 agent，prompt 必含：

1. **目標＋context**：要回答什麼、給齊 analysis_id / fault_id / 先前結論
   （agent 之間無共享記憶）。
2. **驗收標準**：可機檢——「每個主張附 fact_id」「回傳 ≤15 行」「查不到就說
   查不到，不得補腦」。
3. **回報格式**：結論先行 + id 清單；禁止整段知識庫轉貼。

**回合預算**：Orchestrator 每回合工具呼叫 ≤6 次、專才派工 ≤2 個、
子 agent 不得再派工（深度 1）。超過預算 → 直接以現有事實作答並明說限制。

**升降級階梯**：
- 工具錯誤 1 次 → 換參數重試 1 次 → 放棄該工具、據既有事實作答。
- Critic FAIL → 附違規句重生成 1 次 → 再 FAIL → 降級為模板式事實陳述
  （FaultCard 資料直出），並記 log 供離線分析。
- 使用者訊息屬閒聊/致謝 → 路由層（快模型）直接回，不進工具迴圈。

**安全欄杆**：
- 疼痛/受傷/醫療徵狀關鍵詞 → 固定安全回覆（建議就醫），不得生成復健處方。
- Critic 除 grounding 外同時檢查：無誇大療效、無醫療診斷用語。
- 匿名使用者維持現行 gating（chat 需登入）；工具層 RLS 隔離不變。

### 4.5 評估迴路（讓「有據可依」可量化）

| 檢查 | 方法 | 頻率 |
|---|---|---|
| 規則回歸 | golden set（已標註影片）跑 rule pack，比 detection 差異 | CI / 每次規則改動 |
| Chat faithfulness | RAGAS-style：抽樣對話，逐主張比對 evidence bundle | 每週離線批次 |
| Grounding score | Critic 的 pass/fail + 違規句即時入庫 | 每回合（線上） |
| 檢索品質 | KG/RAG 召回標註集（fault → 應命中的 node/chunk） | 換 embedding / KG 版本時 |

---

## 5. 落地順序（每階段獨立可交付）

| 階段 | 內容 | 主要風險 |
|---|---|---|
| **P0（地基）** | KG v3 切換 + movement 參數；rep 切分 + per-rep metrics；非同步分析佇列 | v3 檢索行為差異需回歸測試 |
| **P1（agent 最小可用）** | chat 升級為工具迴圈（先 3 個工具：`get_analysis`/`kg_query`/`rag_search`）＋ Critic-lite（只查 grounding）＋ SSE tool 事件 | 工具迴圈延遲；用「分析期預建 bundle + 增量查詢」壓住 |
| **P2（多動作＋記憶）** | Lunge rule pack（KG/RAG 文件已備）；`compare_analyses`/`list_user_history` 工具與進步追蹤；drill library + `make_drill_plan` | 每個動作的規則閾值需標註樣本 |
| **P3（感知升級）** | 深度類 cue 的 direct-3D 路由（GPU 服務）；神經 embedding RAG；VideoMAE 融合分類器 | GPU 成本；3D 服務可先做成離線批次重分析 |

P0/P1 純工程、無研究風險，直接兌現 spec v3 承諾；P2 兌現多動作計畫；
P3 把已驗證的研究發現（3D 路由、VideoMAE）產品化。

---

## 6. 開放問題

1. **frame_metrics 保留策略**：`get_rep_detail` 需要逐幀資料，但現在分析後即
   丟棄——存全量（JSONB 膨脹）還是存 per-rep 統計摘要？建議先摘要。
2. **3D 推論部署**：NLF 類模型需 GPU；先做「非同步重分析」（使用者按
   「深入分析」才排隊跑）可避免常駐 GPU 成本。
3. **Critic 的誤殺率**：grounding 檢查太嚴會把合理的常識性銜接句判 FAIL，
   需要在 RAGAS 離線評估中校準閾值。
4. **多語**：KG 節點是英文、使用者介面是中文——引用顯示層需要 label 對照
   （canonical mapping 已有雛形可延用）。
