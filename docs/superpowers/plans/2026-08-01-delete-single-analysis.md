# 使用者刪除單筆紀錄 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓登入使用者在「我的紀錄」頁逐筆刪除自己的分析紀錄。

**Architecture:** 新增 `store.delete_analysis()`（先讀 `video_id` → 刪 `analyses` → 只有在同一
`video_id` 沒有其他紀錄時才清 `videos` / `conversations`）、`DELETE /api/analyses/{analysis_id}`
端點，以及 History 頁每列的刪除鈕 + 就地確認。無 DB migration。

**Tech Stack:** FastAPI + supabase-py（PostgREST，走使用者 JWT）／React 18 + Vite + TypeScript +
Tailwind／pytest（`unittest.TestCase`）+ vitest + Testing Library。

**Spec:** `docs/superpowers/specs/2026-08-01-delete-single-analysis-design.md`

## Global Constraints

- Python 一律用 `.venv\Scripts\python.exe`（本機 PATH 上沒有 `python`）。測試一律 scope 到
  `tests/`：`.venv\Scripts\python.exe -m pytest tests/`。
- 所有 yarn / vitest 指令的 cwd 必須是 `frontend/`（Bash 與 PowerShell 工具共用一個 cwd，跑錯目錄
  會讓 vitest 大量失敗）。
- 後端 coverage gate：`.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`。
- **不新增 DB migration。** RLS policy 是 `for all`，DELETE 對擁有者已放行。
- **不刪 `runtime/uploads/` 的磁碟檔**，與 `delete_all_analyses` 保持一致。
- 前端所有字串走 `t()`，en 與 zh-TW **兩份**都要補（缺 key 會把 key 原文渲染出來）。
- 後端測試不得連 live Supabase：一律 `mock.patch.object(store, "_user_client", ...)`。
- 分支：`feat/delete-single-analysis`（已建立，spec 已在上面 commit）。

---

### Task 1: `store.delete_analysis()` — 資料層與兄弟紀錄保護

**Files:**
- Modify: `backend/app/services/store.py`（在 `delete_all_analyses` 之後，`get_analysis` 之前）
- Test: `tests/test_backend.py`（`_FakeQuery` 附近新增測試替身；`StoreDeleteTests` 內新增測試）

**Interfaces:**
- Consumes: `store._user_client(token)`（既有）
- Produces: `store.delete_analysis(*, token: str, analysis_id: str, user_id: str) -> bool`
  — 有刪到回 `True`；該 id 讀不到或不屬於呼叫者回 `False`。

- [ ] **Step 1: 新增多表測試替身**

現有的 `_fake_client` 只回**單一**預設 response，而 `delete_analysis` 會依序發三個查詢（讀
`video_id` → delete → 數剩餘筆數），每個要不同的回應，所以需要一個能排隊回應、且能記錄
`eq()` 條件的替身。加在 `tests/test_backend.py` 的 `_fake_client`（約 1329 行）之後：

```python
class _FakeTable:
    """One table's chained PostgREST calls; execute() pops the next preset response.

    `_fake_client` returns a single query with a single canned response, which can't express a
    multi-step store function (read -> delete -> count). This variant queues one response per
    execute() and records the eq() filters so a test can assert *what* was deleted.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []          # "select" / "delete", in order
        self.eq_filters: list[tuple] = []   # (column, value) per eq()

    def select(self, *a, **k):
        self.calls.append("select")
        return self

    def delete(self, *a, **k):
        self.calls.append("delete")
        return self

    def eq(self, column, value):
        self.eq_filters.append((column, value))
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return self._responses.pop(0) if self._responses else _Resp(data=[])


def _fake_tables(**by_table: list) -> tuple[mock.Mock, dict[str, _FakeTable]]:
    """A client whose .table(name) returns a per-table fake with its own response queue.

    Tables not preset are created on demand, so an unexpected table access doesn't crash the
    test -- assert on `client.table.call_args_list` to catch it instead.
    """
    tables: dict[str, _FakeTable] = {n: _FakeTable(r) for n, r in by_table.items()}
    client = mock.Mock()
    client.table.side_effect = lambda name: tables.setdefault(name, _FakeTable([]))
    return client, tables


def _tables_touched(client: mock.Mock) -> list[str]:
    """Table names passed to client.table(), in call order."""
    return [c.args[0] for c in client.table.call_args_list]
```

- [ ] **Step 2: 寫失敗測試**

加在 `tests/test_backend.py` 的 `StoreDeleteTests` 類別內（`test_delete_all_handles_empty` 之後）：

```python
    def test_delete_one_returns_true_and_filters_by_id_and_user(self) -> None:
        # read video_id -> delete (1 row) -> sibling count 0
        client, tables = _fake_tables(
            analyses=[
                _Resp(data=[{"video_id": "upload_1"}]),
                _Resp(data=[{"id": "a1"}]),
                _Resp(data=[], count=0),
            ]
        )
        with mock.patch.object(store, "_user_client", return_value=client):
            ok = store.delete_analysis(token="t", analysis_id="a1", user_id="u1")
        self.assertTrue(ok)
        # The delete is scoped by BOTH the row id and the owner (RLS is the backstop, not the filter).
        self.assertIn(("id", "a1"), tables["analyses"].eq_filters)
        self.assertIn(("user_id", "u1"), tables["analyses"].eq_filters)

    def test_delete_one_keeps_video_and_conversation_when_siblings_remain(self) -> None:
        """Re-analysing one clip inserts a second `analyses` row against the same `video_id`, while
        `videos`/`conversations` are unique per (user, video_id). Copying delete_all_analyses'
        unconditional three-table delete would wipe the SIBLING record's chat thread and video row.
        """
        client, tables = _fake_tables(
            analyses=[
                _Resp(data=[{"video_id": "upload_1"}]),
                _Resp(data=[{"id": "a1"}]),
                _Resp(data=[{"id": "a2"}], count=1),  # a sibling still references upload_1
            ]
        )
        with mock.patch.object(store, "_user_client", return_value=client):
            ok = store.delete_analysis(token="t", analysis_id="a1", user_id="u1")
        self.assertTrue(ok)
        touched = _tables_touched(client)
        self.assertNotIn("videos", touched)
        self.assertNotIn("conversations", touched)

    def test_delete_one_drops_video_and_conversation_when_last(self) -> None:
        client, tables = _fake_tables(
            analyses=[
                _Resp(data=[{"video_id": "upload_1"}]),
                _Resp(data=[{"id": "a1"}]),
                _Resp(data=[], count=0),  # nothing left referencing upload_1
            ]
        )
        with mock.patch.object(store, "_user_client", return_value=client):
            store.delete_analysis(token="t", analysis_id="a1", user_id="u1")
        touched = _tables_touched(client)
        self.assertIn("videos", touched)
        self.assertIn("conversations", touched)
        # Both cascades are scoped to the freed video, not to the whole account.
        self.assertIn(("video_id", "upload_1"), tables["videos"].eq_filters)
        self.assertIn(("user_id", "u1"), tables["videos"].eq_filters)
        self.assertIn(("video_id", "upload_1"), tables["conversations"].eq_filters)

    def test_delete_one_returns_false_when_absent(self) -> None:
        # RLS makes someone else's id indistinguishable from a missing one: the read comes back empty.
        client, tables = _fake_tables(analyses=[_Resp(data=[])])
        with mock.patch.object(store, "_user_client", return_value=client):
            ok = store.delete_analysis(token="t", analysis_id="ghost", user_id="u1")
        self.assertFalse(ok)
        self.assertNotIn("delete", tables["analyses"].calls)  # nothing was deleted
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py::StoreDeleteTests -v`
Expected: FAIL — `AttributeError: <module 'backend.app.services.store'> does not have the attribute 'delete_analysis'`（4 個新測試皆紅）

- [ ] **Step 4: 實作**

加在 `backend/app/services/store.py` 的 `delete_all_analyses` 之後：

```python
def delete_analysis(*, token: str, analysis_id: str, user_id: str) -> bool:
    """Delete ONE analysis owned by the caller; return whether a row was actually removed.

    The video row and chat thread are keyed on ``video_id``, not on the analysis: re-analysing one
    clip inserts a *second* ``analyses`` row against the same ``video_id`` (``persist_analysis``
    upserts ``videos`` but inserts ``analyses``). So they are dropped only once this was the LAST
    analysis referencing that video -- never unconditionally the way ``delete_all_analyses`` can,
    or deleting one record would silently wipe a sibling record's chat thread.

    The uploaded file under ``runtime/uploads/`` is deliberately left on disk, matching the
    clear-all path; reaping those is a separate change that should cover both.
    """
    client = _user_client(token)
    found = (
        client.table("analyses")
        .select("video_id")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
    )
    rows = found.data or []
    if not rows:
        # Missing, or someone else's -- RLS scopes the read, so the two are indistinguishable.
        return False
    video_id = rows[0]["video_id"]

    # RLS already scopes writes to auth.uid() = user_id; the explicit predicate is a second guard.
    resp = (
        client.table("analyses")
        .delete()
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not (resp.data or []):
        return False

    siblings = (
        client.table("analyses")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    if (siblings.count or 0) == 0:
        client.table("videos").delete().eq("user_id", user_id).eq("video_id", video_id).execute()
        client.table("conversations").delete().eq("user_id", user_id).eq(
            "video_id", video_id
        ).execute()
    return True
```

- [ ] **Step 5: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py::StoreDeleteTests -v`
Expected: PASS（含既有的 2 個 delete-all 測試，共 6 個）

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/store.py tests/test_backend.py
git commit -m "feat(store): delete one analysis, cascading only when no sibling shares the video"
```

---

### Task 2: `DELETE /api/analyses/{analysis_id}` 端點

**Files:**
- Modify: `backend/app/routers/analyses.py`
- Test: `tests/test_backend.py`（`AnalysesRouterTests` 類別）

**Interfaces:**
- Consumes: `store.delete_analysis(*, token, analysis_id, user_id) -> bool`（Task 1）、
  `get_current_user` → `CurrentUser`（有 `.token` 與 `.id`）
- Produces: `DELETE /api/analyses/{analysis_id}` → `200 {"deleted": 1}`／`404`／未帶 token `401`

- [ ] **Step 1: 寫失敗測試**

加在 `tests/test_backend.py` 的 `AnalysesRouterTests` 內（`test_delete_all_returns_count` 之後）：

```python
    def test_delete_one_returns_deleted_count(self) -> None:
        with mock.patch.object(store, "delete_analysis", return_value=True) as da:
            resp = self.client.delete("/api/analyses/3f2a5c1e-0000-4000-8000-000000000001")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"deleted": 1})
        da.assert_called_once_with(
            token="tok",
            analysis_id="3f2a5c1e-0000-4000-8000-000000000001",
            user_id="u1",
        )

    def test_delete_one_missing_is_404(self) -> None:
        with mock.patch.object(store, "delete_analysis", return_value=False):
            resp = self.client.delete("/api/analyses/3f2a5c1e-0000-4000-8000-000000000002")
        self.assertEqual(resp.status_code, 404)

    def test_delete_one_bad_uuid_is_404(self) -> None:
        """`.eq("id", "not-a-uuid")` makes Postgres raise 22P02, which would surface as a 500.
        The id is validated before the store is reached, so a junk path param is a plain 404."""
        with mock.patch.object(store, "delete_analysis") as da:
            resp = self.client.delete("/api/analyses/not-a-uuid")
        self.assertEqual(resp.status_code, 404)
        da.assert_not_called()

    def test_delete_one_requires_auth(self) -> None:
        app.dependency_overrides.clear()  # drop the override -> real dependency runs
        resp = self.client.delete("/api/analyses/3f2a5c1e-0000-4000-8000-000000000003")
        self.assertEqual(resp.status_code, 401)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py::AnalysesRouterTests -v`
Expected: FAIL — 前三個回 `405 Method Not Allowed`（路由不存在）

- [ ] **Step 3: 實作端點**

`backend/app/routers/analyses.py`：檔頭 import 補上 `import uuid`（放在 `from __future__` 之後、
`from fastapi ...` 之前的 stdlib 區塊）。然後在 `get_my_analysis` 之後新增：

```python
@router.delete("/analyses/{analysis_id}")
def delete_my_analysis(
    analysis_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    """Delete one of the caller's analyses: ``{"deleted": 1}``, or 404.

    A non-UUID path param is rejected here rather than reaching PostgREST, where it would raise
    ``22P02 invalid input syntax for type uuid`` and surface as a 500. Someone else's id is a 404
    for the same reason the GET is: under RLS it is indistinguishable from a missing row.
    """
    try:
        uuid.UUID(analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"No analysis '{analysis_id}'.") from exc
    if not store.delete_analysis(token=user.token, analysis_id=analysis_id, user_id=user.id):
        raise HTTPException(status_code=404, detail=f"No analysis '{analysis_id}'.")
    return {"deleted": 1}
```

同時把模組 docstring 第一行的 `list and fetch` 改為 `list, fetch, and delete`。

- [ ] **Step 4: 跑測試確認通過**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backend.py::AnalysesRouterTests -v`
Expected: PASS（全類別綠燈）

- [ ] **Step 5: 跑整套後端測試 + coverage gate**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Expected: PASS（若出現 `test_concurrent_analyses_are_bounded` 失敗，那是需要 live Supabase 的既有
案例，不是本次改動造成的——確認訊息是連線類錯誤後照舊往下走）

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS（≥ 95%）

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/analyses.py tests/test_backend.py
git commit -m "feat(api): DELETE /api/analyses/{id} for per-record deletion"
```

---

### Task 3: 前端接線 — API client + i18n 字串

**Files:**
- Modify: `frontend/src/api.ts`（`deleteAnalyses()` 之後，約 517-521 行）
- Modify: `frontend/src/lib/i18n.tsx`（en 區塊約 483 行後、zh-TW 區塊約 1152 行後）

**Interfaces:**
- Consumes: `DELETE /api/analyses/{analysis_id}`（Task 2）、`authHeader()`（既有）
- Produces: `api.deleteAnalysis(id: string): Promise<{ deleted: number }>`；i18n keys
  `history.deleteAria` / `history.deleteCta` / `history.deleteDesc` / `history.deleteConfirm` /
  `history.deleteCancel` / `history.deleting` / `history.deleteError`

- [ ] **Step 1: 新增 API client**

`frontend/src/api.ts`，緊接在 `deleteAnalyses()` 之後：

```ts
  // Delete ONE saved analysis (requires a session). 404 if it isn't the caller's row.
  async deleteAnalysis(id: string): Promise<{ deleted: number }> {
    const path = `/api/analyses/${encodeURIComponent(id)}`;
    const res = await fetch(path, { method: "DELETE", headers: await authHeader() });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
    return (await res.json()) as { deleted: number };
  },
```

- [ ] **Step 2: 新增英文字串**

`frontend/src/lib/i18n.tsx`，接在 `"history.faultMany"`（約 483 行）之後：

```ts
  "history.deleteAria": "Delete this record",
  "history.deleteCta": "Delete",
  "history.deleteDesc": "Delete this record permanently?",
  "history.deleteConfirm": "Delete",
  "history.deleteCancel": "Cancel",
  "history.deleting": "Deleting…",
  "history.deleteError": "Couldn't delete this record. Please try again.",
```

- [ ] **Step 3: 新增繁中字串**

同檔案，接在 zh-TW 的 `"history.faultMany"`（約 1152 行）之後：

```ts
  "history.deleteAria": "刪除這筆紀錄",
  "history.deleteCta": "刪除",
  "history.deleteDesc": "確定要永久刪除這筆紀錄嗎？",
  "history.deleteConfirm": "刪除",
  "history.deleteCancel": "取消",
  "history.deleting": "刪除中…",
  "history.deleteError": "無法刪除這筆紀錄，請再試一次。",
```

- [ ] **Step 4: 型別檢查**

Run（cwd = `frontend/`）：`yarn build`
Expected: 成功，無 TypeScript 錯誤

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/lib/i18n.tsx
git commit -m "feat(frontend): deleteAnalysis client + history delete strings (en/zh-TW)"
```

---

### Task 4: History 頁的每列刪除鈕與就地確認

**Files:**
- Modify: `frontend/src/pages/History.tsx`
- Test: `frontend/src/test/pages.History.test.tsx`

**Interfaces:**
- Consumes: `api.deleteAnalysis(id)`（Task 3）、上述七個 i18n keys
- Produces: 無（頁面終點）

- [ ] **Step 1: 寫失敗測試**

加在 `frontend/src/test/pages.History.test.tsx` 的 `describe("History", ...)` 內，
`"signs out from the account menu..."` 之前：

```tsx
  it("deletes a row after confirmation and removes it from the list", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({
      total: 2,
      items: [
        item({ id: "a", created_at: "2026-06-20T12:00:00.000Z" }),
        item({ id: "b", fault_count: 0, created_at: "2026-06-20T13:00:00.000Z" }),
      ],
    });
    const del = vi.spyOn(api, "deleteAnalysis").mockResolvedValue({ deleted: 1 });
    renderHistory();
    await screen.findByText("2 faults");

    // Each row carries its own delete control; the first row is the newest ("a").
    await userEvent.click(screen.getAllByRole("button", { name: "Delete this record" })[0]);
    // The icon button's accessible name comes from its aria-label ("Delete this record"), so a
    // full-string match on "Delete" hits only the confirm button.
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(del).toHaveBeenCalledWith("a"));
    // The deleted row is spliced out locally; the surviving row stays.
    await waitFor(() => expect(screen.queryByText("2 faults")).not.toBeInTheDocument());
    expect(screen.getByText("clean rep")).toBeInTheDocument();
  });

  it("keeps the row and shows an error when the delete fails", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [item()] });
    vi.spyOn(api, "deleteAnalysis").mockRejectedValue(new Error("500 Internal Server Error"));
    renderHistory();
    await screen.findByText("2 faults");

    await userEvent.click(screen.getByRole("button", { name: "Delete this record" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(
      await screen.findByText("Couldn't delete this record. Please try again.")
    ).toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument(); // row survives, retryable
  });

  it("cancelling the confirmation calls no API and restores the row", async () => {
    vi.spyOn(api, "listAnalyses").mockResolvedValue({ total: 1, items: [item()] });
    const del = vi.spyOn(api, "deleteAnalysis").mockResolvedValue({ deleted: 1 });
    renderHistory();
    await screen.findByText("2 faults");

    await userEvent.click(screen.getByRole("button", { name: "Delete this record" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(del).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(screen.getByText("2 faults")).toBeInTheDocument();
  });
```

- [ ] **Step 2: 跑測試確認失敗**

Run（cwd = `frontend/`）：`yarn test src/test/pages.History.test.tsx`
Expected: FAIL — 三個新測試都找不到 `Delete this record` 按鈕

- [ ] **Step 3: 加入刪除狀態與處理函式**

`frontend/src/pages/History.tsx`：

3a. 檔頭 import 補上 `Trash`（`@phosphor-icons/react` 既有的具名匯入清單內，維持字母序放在
`PersonSimpleRun` 之後）：

```tsx
import {
  CaretRight,
  FilmSlate,
  PersonSimpleRun,
  Trash,
  VideoCamera,
  WarningCircle,
} from "@phosphor-icons/react";
```

3b. 在 `const [error, setError] = useState("");`（約 25 行）之後新增：

```tsx
  // Per-row deletion. Only one row can be in the confirm state at a time; `deleteError` is keyed by
  // row id so the message appears under the row it belongs to, not at the top of the page.
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null);
```

3c. 在 `load` 的 `useCallback` 之後（約 38 行，`useEffect` 之前）新增：

```tsx
  // Splice the row out locally rather than refetching: `groups` is derived, so an emptied day
  // header disappears on its own and deleting the last row falls back to the empty state.
  const runDelete = async (id: string) => {
    setDeletingId(id);
    setDeleteError(null);
    try {
      await api.deleteAnalysis(id);
      setItems((prev) => prev.filter((it) => it.id !== id));
      setPendingId(null);
    } catch (e) {
      setDeleteError({ id, message: e instanceof Error ? e.message : String(e) });
      setPendingId(null);
    } finally {
      setDeletingId(null);
    }
  };
```

- [ ] **Step 4: 改寫列的 JSX**

把 `History.tsx` 現有的 `<li key={it.id}>…</li>` 整塊（約 148-193 行，從 `<li key={it.id}>` 到
對應的 `</li>`）換成下面這段。三個關鍵改動：`group` 從 `<Link>` 移到 `<li>`（刪除鈕是 Link 的
**手足**而非子孫——`<button>` 巢狀在 `<a>` 裡是無效 HTML，點擊會冒泡成導航，`stopPropagation`
修不了巢狀本身）；`<Link>` 加 `pr-14` 讓出刪除鈕的位置；確認列排在 Link 下方而非疊在上面。

```tsx
                      <li key={it.id} className="group relative">
                        <Link
                          to={`/app?analysis=${it.id}`}
                          className="flex items-center gap-4 rounded-2xl border border-border-dark bg-surface-dark p-4 pr-14 transition-colors hover:border-primary/40 hover:bg-content/[0.03]"
                        >
                          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                            <PersonSimpleRun size={22} weight="duotone" />
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate font-medium text-content">
                              {t("history.rowTitle", {
                                view: viewLabel(t, it.view_type ?? "unknown"),
                                movement: movementLabel(t, it.movement ?? "Squat"),
                              })}
                            </p>
                            <p className="mt-0.5 flex items-center gap-2 font-mono text-xs text-muted">
                              {fmtTime(it.created_at)}
                              <span className="rounded bg-content/5 px-1.5 py-0.5 text-[11px] font-medium text-muted">
                                {/* The promoted column, then Squat. `HistoryItem` carries no
                                    `result` -- list_analyses selects only the promoted columns, not
                                    the heavy document -- so there is no per-row echo to fall back
                                    to here. Rows predating the column are Squat by construction:
                                    every analysis before this change was pinned to it. */}
                                {movementLabel(t, it.movement ?? "Squat")}
                              </span>
                            </p>
                          </div>
                          <span
                            className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${
                              clean
                                ? "bg-secondary/15 text-secondary"
                                : "bg-[rgb(var(--c-fault))]/15 text-[rgb(var(--c-fault))]"
                            }`}
                          >
                            {clean
                              ? t("history.clean")
                              : it.fault_count === 1
                                ? t("history.faultOne")
                                : t("history.faultMany", { count: it.fault_count })}
                          </span>
                          <CaretRight
                            size={18}
                            className="shrink-0 text-muted transition-transform group-hover:translate-x-0.5"
                          />
                        </Link>

                        {/* Sibling of the Link, not a child: a <button> inside an <a> is invalid
                            HTML and its click bubbles into navigation. Hidden until the row is
                            hovered or the button is focused; always visible on touch. */}
                        <button
                          type="button"
                          aria-label={t("history.deleteAria")}
                          title={t("history.deleteCta")}
                          onClick={() => {
                            setPendingId(pendingId === it.id ? null : it.id);
                            // Only this row's error — `deleteError` is one page-level slot, so an
                            // unconditional clear would silently wipe ANOTHER row's unresolved failure.
                            setDeleteError((prev) => (prev?.id === it.id ? null : prev));
                          }}
                          className="absolute right-3 top-9 -translate-y-1/2 rounded-lg p-2 text-muted opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
                        >
                          <Trash size={16} weight="duotone" />
                        </button>

                        {pendingId === it.id && (
                          <div className="mt-1.5 flex flex-wrap items-center gap-2 rounded-xl border border-danger/30 bg-danger/[0.04] px-3 py-2">
                            <span className="mr-auto text-xs text-muted">
                              {t("history.deleteDesc")}
                            </span>
                            <button
                              type="button"
                              onClick={() => setPendingId(null)}
                              disabled={deletingId === it.id}
                              className="rounded-lg px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-content/5 hover:text-content disabled:opacity-60"
                            >
                              {t("history.deleteCancel")}
                            </button>
                            <button
                              type="button"
                              onClick={() => void runDelete(it.id)}
                              disabled={deletingId === it.id}
                              className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-red-700 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
                            >
                              <Trash size={14} weight="fill" />
                              {deletingId === it.id
                                ? t("history.deleting")
                                : t("history.deleteConfirm")}
                            </button>
                          </div>
                        )}

                        {deleteError?.id === it.id && (
                          <p className="mt-1.5 flex items-center gap-1.5 px-1 text-xs text-danger">
                            <WarningCircle size={14} weight="fill" className="shrink-0" />
                            {t("history.deleteError")}
                          </p>
                        )}
                      </li>
```

- [ ] **Step 5: 跑測試確認通過**

Run（cwd = `frontend/`）：`yarn test src/test/pages.History.test.tsx`
Expected: PASS（既有 9 個 + 新增 3 個 = 12 個）

`getByRole` 的 `name` 傳字串時是**整串比對** accessible name，而圖示鈕的 accessible name 來自
`aria-label`（`Delete this record`）而非 `title`（`Delete`）——aria-label 的優先序高於 title——所以
`{ name: "Delete" }` 只會命中確認鈕。若這裡真的抓到兩個元素，代表 `aria-label` 沒掛上去，回頭檢查
Step 4 的按鈕而不是放寬 query。

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/History.tsx frontend/src/test/pages.History.test.tsx
git commit -m "feat(history): per-row delete with inline confirmation"
```

---

### Task 5: 全套驗證

**Files:** 無（純驗證）

- [ ] **Step 1: 後端全套**

Run: `.venv\Scripts\python.exe -m pytest tests/`
Expected: PASS（唯一可接受的失敗是需要 live Supabase 的既有案例，且錯誤訊息須為連線類）

- [ ] **Step 2: 後端 coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Expected: PASS

- [ ] **Step 3: 前端全套**

Run（cwd = `frontend/`）：`yarn test`
Expected: PASS

- [ ] **Step 4: 前端 build**

Run（cwd = `frontend/`）：`yarn build`
Expected: 成功

- [ ] **Step 5: 更新知識圖譜**

Run: `graphify update .`
Expected: 完成（AST-only，無 API 成本）

- [ ] **Step 6: Commit（若有殘留變更）**

`git add -A` **不可用**——工作區有一個與本功能無關的未追蹤檔案
（`notes/camera-placement-hypothesis.md`），只 add 圖譜輸出：

```bash
git status --short
git add graphify-out/
git commit -m "chore: refresh graphify graph after per-record deletion"
```

若 `git status --short` 顯示 `graphify-out/` 無變更，跳過這步。

---

## 附註：實作時容易踩的三個點

1. **兄弟紀錄**（Task 1）——這是唯一會靜默毀損資料的地方，`test_delete_one_keeps_video_and_
   conversation_when_siblings_remain` 就是為它存在的，不要為了讓程式變短而拿掉剩餘筆數的查詢。
2. **`<button>` 不能巢狀在 `<a>` 裡**（Task 4）——必須是手足，且 `group` 要放在 `<li>` 上，否則
   hover 顯示會失效。
3. **非 UUID 的路徑參數**（Task 2）——不擋會變成 500 而不是 404。
