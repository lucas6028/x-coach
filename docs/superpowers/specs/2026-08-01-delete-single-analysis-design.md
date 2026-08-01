# 使用者刪除單筆紀錄 — 設計 spec

**日期**：2026-08-01
**分支**：`feat/delete-single-analysis`（開自 `main`）
**狀態**：設計已核可，待實作

## 1. 目標

讓登入的使用者在「我的紀錄」（`/history`）**逐筆刪除**自己的分析紀錄。

現況：批次「清除全部紀錄」已經存在（Settings 危險區 → `api.deleteAnalyses()` →
`DELETE /api/analyses` → `store.delete_all_analyses`）。缺的只有單筆刪除——目前使用者想丟掉
一次失敗的錄影，唯一的辦法是把整份歷史清光。本 spec 只補這個缺口，**不做**多選批次刪除、
不做垃圾桶／軟刪除、不做 undo。

## 2. 關鍵前提（設計依據）

這幾條是從現有 schema 與程式碼查證出來的，決定了下面的設計：

- **不需要 migration。** `db/migrations/20260620000000_init_videos_analyses.sql` 的 RLS policy
  是 `for all ... using (auth.uid() = user_id)`，DELETE 對擁有者本來就已放行。
- **`conversations` 的唯一鍵是 `(user_id, video_id)`，不是 analysis_id**
  （`20260704000000_conversations.sql`），而 `analyses.video_id` 既無 FK 也無唯一性。
  `store.persist_analysis` 對 `videos` 是 upsert、對 `analyses` 是 insert，所以**同一支影片重跑
  分析會產生 N 筆 `analyses` 共用 1 列 `videos` 與 1 串 `conversations`**。
  → 逐筆刪除若照抄 `delete_all_analyses` 的無條件三表刪除，會把**兄弟紀錄**的聊天與影片列一併
  清掉。這是本功能唯一會靜默毀損資料的地方。
- **RLS 之下，「別人的 id」與「不存在的 id」不可分辨**：查詢單純回空。既有的
  `GET /api/analyses/{id}` 就是以此為據回 404，本端點沿用同一語意。
- **磁碟檔案不在範圍內。** `delete_all_analyses` 不碰 `runtime/uploads/`，單筆刪除保持一致，
  在 docstring 註明這是刻意的取捨而非疏漏。

## 3. 資料層：`store.delete_analysis()`

位置：`backend/app/services/store.py`，緊接在 `delete_all_analyses` 之後。

```python
def delete_analysis(*, token: str, analysis_id: str, user_id: str) -> bool:
    """刪除呼叫者的一筆分析；有刪到回 True，找不到（或不屬於他）回 False。"""
```

步驟：

1. 先 `select("video_id").eq("id", analysis_id)` 讀出這筆的 `video_id`。RLS 已把範圍鎖在
   `auth.uid()`，讀不到 → 直接回 `False`（不存在或不屬於呼叫者）。
2. `client.table("analyses").delete().eq("id", analysis_id).eq("user_id", user_id).execute()`。
   `user_id` 明確帶上當第二道保險，與 `delete_all_analyses` 同風格。
3. 刪完後 `select("id", count="exact").eq("user_id", user_id).eq("video_id", video_id).limit(1)`
   數剩餘的 `analyses`。
4. **只有在剩餘筆數為 0 時**，才刪 `videos` 與 `conversations` 中對應 `(user_id, video_id)` 的列。
   還有兄弟紀錄就完全不動這兩張表。
5. 回傳步驟 2 是否真的刪到列（`bool(resp.data)`）。

## 4. API：`DELETE /api/analyses/{analysis_id}`

位置：`backend/app/routers/analyses.py`，放在 `get_my_analysis` 旁邊。

```python
@router.delete("/analyses/{analysis_id}")
def delete_my_analysis(analysis_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
```

- **先驗 UUID**：`uuid.UUID(analysis_id)` 失敗即 `404`。不驗的話 `.eq("id", "abc")` 會讓 Postgres
  丟 `22P02 invalid input syntax for type uuid`，冒出去變成 500。（`get_analysis` 有同樣的潛在
  問題，但那是既有行為，本次不順手改動——只確保新端點不帶著它上線。）
- `store.delete_analysis(...)` 回 `False` → `HTTPException(404, f"No analysis '{analysis_id}'.")`，
  與 GET 同一套訊息格式。
- 成功 → `{"deleted": 1}`（與 `DELETE /api/analyses` 的 `{"deleted": n}` 形狀一致）。
- 認證由既有的 `Depends(get_current_user)` 負責；未帶 token 為 401。

**路由順序**：`/api/analyses`（無參數）的 DELETE 與 `/api/analyses/{analysis_id}` 的 DELETE 路徑
不同層，FastAPI 不會誤配，維持檔案現有的宣告順序即可。

## 5. 前端

### 5.1 `frontend/src/api.ts`

貼著現有的 `deleteAnalyses()` 新增：

```ts
async deleteAnalysis(id: string): Promise<{ deleted: number }>
```

`fetch(`/api/analyses/${encodeURIComponent(id)}`, { method: "DELETE", headers: await authHeader() })`，
錯誤處理沿用該檔案既有的 helper 寫法。

### 5.2 `frontend/src/pages/History.tsx`

**結構問題必須先解**：目前整列被 `<Link>` 包住（`History.tsx:149`）。`<button>` 放進 `<a>` 裡是
無效 HTML，且點擊會冒泡成導航——`stopPropagation` 修不了巢狀本身。改法：

- `<li className="relative">`，`<Link>` 照舊填滿整列（右側 padding 加大，讓出刪除鈕的位置）。
- 刪除鈕是 `<Link>` 的**絕對定位手足**，不是子孫。桌機 `opacity-0 group-hover:opacity-100`
  + `focus-visible:opacity-100`（鍵盤可達），觸控裝置（`@media (hover: none)`）常駐顯示。

**確認流程**沿用 Settings 危險區的既有慣用法，不開 modal：以 `useState` 存
`pendingId: string | null` 與 `deleting: string | null`；點垃圾桶 → 該列就地切換成
「取消 ／ 確認刪除」兩顆鈕；確認後該列進入 working 態（鈕 disabled）。同一時間只有一列處於
確認態（點另一列的垃圾桶會取代前一個 `pendingId`）。

**成功**：本地 `setItems(prev => prev.filter(it => it.id !== id))`，不重新 fetch。`groups` 是
`useMemo` 衍生值，某一天的紀錄被刪光時該日期標題會自己消失；刪到一筆不剩會落回既有的空狀態。

**失敗**：該列下方顯示一行錯誤訊息（沿用 `history.errorTitle` 區塊的 danger 樣式），列保留、
狀態回 idle，使用者可重試。

### 5.3 i18n

`frontend/src/lib/i18n.tsx` 的 en 與 zh-TW **兩份都要補**（缺 key 會直接把 key 原文渲染出來）：

| key | en | zh-TW |
| --- | --- | --- |
| `history.deleteCta` | Delete | 刪除 |
| `history.deleteDesc` | Delete this record permanently? | 確定要永久刪除這筆紀錄嗎？ |
| `history.deleteConfirm` | Delete | 刪除 |
| `history.deleteCancel` | Cancel | 取消 |
| `history.deleting` | Deleting… | 刪除中… |
| `history.deleteError` | Couldn't delete this record. Please try again. | 無法刪除這筆紀錄，請再試一次。 |
| `history.deleteAria` | Delete this record | 刪除這筆紀錄 |

`history.deleteAria` 用於垃圾桶 icon 鈕的 `aria-label`（純 icon 無文字）。

## 6. 測試

### 後端 `tests/test_backend.py`

沿用該檔既有的 `_fake_client` / `mock.patch.object(store, "_user_client")` 風格，不碰 live Supabase。

`StoreDeleteTests` 新增：

1. `test_delete_one_returns_true_and_filters_by_id_and_user` — 有刪到回 `True`，delete 帶了
   `id` 與 `user_id` 兩個條件。
2. **`test_delete_one_keeps_video_and_conversation_when_siblings_remain`** — 剩餘計數 > 0 時，
   `client.table` 只被呼叫在 `analyses` 上，`videos` / `conversations` 完全沒被 delete。
   **這是本功能的核心迴歸測試**，前提 §2 的靜默毀損就靠它擋住。
3. `test_delete_one_drops_video_and_conversation_when_last` — 剩餘計數為 0 時，三張表都清掉。
4. `test_delete_one_returns_false_when_absent` — 讀不到 `video_id` → 回 `False`，且不發出任何
   delete。

`AnalysesRouterTests` 新增：

5. `test_delete_one_returns_deleted_count` — 200 + `{"deleted": 1}`，並斷言
   `delete_analysis` 以 `token=/analysis_id=/user_id=` 被呼叫一次。
6. `test_delete_one_missing_is_404`。
7. `test_delete_one_bad_uuid_is_404` — 非 UUID 路徑參數回 404 而非 500，且 store 未被呼叫。
8. `test_delete_one_requires_auth` — 清掉 `dependency_overrides` 後為 401。

### 前端 `frontend/src/test/pages.History.test.tsx`

9. 點垃圾桶 → 出現確認鈕；點確認 → `api.deleteAnalysis` 收到該筆 id，該列從畫面消失。
10. 刪除失敗 → 顯示錯誤訊息且該列仍在。
11. 點取消 → 不呼叫 API，回到原本的列樣貌。

### 驗證指令

- `.venv\Scripts\python.exe -m pytest tests/`
- `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
- `yarn test`（cwd = `frontend/`）

## 7. 明確不做（YAGNI）

- 多選／批次刪除（「清除全部」已涵蓋另一端的需求）
- 軟刪除、垃圾桶、undo
- 刪除 `runtime/uploads/` 的磁碟檔（與 `delete_all_analyses` 保持一致；要做就兩邊一起做，
  屬另一個 change）
- 順手修 `get_analysis` 的非 UUID → 500 問題（既有行為，另案處理）
