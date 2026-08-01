# 刪除確認改為對話框 — 設計 spec

**日期**：2026-08-01
**分支**：`feat/delete-single-analysis`（沿用同一分支，接續 per-record 刪除功能）
**狀態**：設計已核可，待實作

## 1. 目標

把「我的紀錄」逐列刪除的**確認態**從就地插入的確認列改成置中彈出對話框。

前一版（spec `2026-08-01-delete-single-analysis-design.md`，已實作並通過審查）在點下垃圾桶後，於該列**下方插入**一條確認列。使用者回報不直觀，具體癥結是：**確認列把後面的列整個往下推，版面跳動，而且它視覺上更貼近「下一列」，看不出到底要刪哪一筆。**

本 spec 只改確認態的呈現方式。刪除的後端行為、API、`runDelete` 的狀態流轉、每列的錯誤顯示都不動。

## 2. 既有慣用法（設計依據）

- 專案**沒有**共用的 modal／dialog primitive。`frontend/src/components/ui/` 不存在。
- `KnowledgeGraphWidget.tsx:334-344` 是最接近的先例：`role="dialog"` + `aria-modal="true"` +
  `fixed inset-0 z-50`，Escape 由 `useEffect` 掛 `document` keydown 關閉，**不使用 portal**。
- `ComplexitySelector.tsx:32-44` 是 popover，同樣的 Escape + 點外面關閉的 effect 寫法。
- 破壞性按鈕的既有樣式在 `Settings.tsx:244`：`bg-red-600 ... text-white hover:bg-red-700`。

新元件照這些慣用法寫，不引入新的相依套件、不引入 portal。

## 3. 新元件 `frontend/src/components/ConfirmDialog.tsx`

自足的小元件，只負責「問一個是非題」。不知道 analysis、不知道 History。

```tsx
type ConfirmDialogProps = {
  open: boolean;
  title: string;
  description: string;
  detail?: string;          // 被操作對象的識別文字，例如「側面 深蹲 · 10:24」
  confirmLabel: string;
  cancelLabel: string;
  busy?: boolean;           // 動作進行中：鈕 disabled，且關不掉
  busyLabel?: string;       // busy 時確認鈕顯示的字
  onConfirm: () => void;
  onCancel: () => void;
};
```

行為：

- `open` 為 false 時 return `null`（不留任何 DOM）。
- 版面：半透明 backdrop（`fixed inset-0 z-50` + `bg-black/60`）+ 置中卡片，卡片沿用專案的
  `rounded-2xl border border-border-dark bg-surface-dark` 語彙。
- 無障礙：`role="dialog"`、`aria-modal="true"`、`aria-labelledby` 指向標題的 id，
  `aria-describedby` 指向說明的 id。id 用 `useId()` 產生，避免同頁多個實例撞號。
- **Escape 關閉**（`document` 上的 keydown，effect 內註冊、卸載時移除，與既有兩處寫法一致）。
- **點 backdrop 關閉**；點卡片本身不關閉。
- **`busy` 為 true 時，Escape 與 backdrop 都不觸發 `onCancel`**，兩顆鈕都 disabled。刪除進行中
  被關掉會讓使用者不知道結果，且 `deletingId` 仍指向該列。
- 開啟時焦點移到**取消鈕**（破壞性操作的安全預設，避免連續 Enter 誤刪）。
- 關閉時把焦點還給開啟前的 `document.activeElement`（也就是該列的垃圾桶鈕）。
- 確認鈕樣式沿用 `Settings.tsx` 危險區：`bg-red-600 / hover:bg-red-700 / text-white`。
- 不做 focus trap、不鎖 body scroll——既有的 `KnowledgeGraphWidget` 也沒做，保持一致；
  這是刻意的取捨，不是疏漏。

## 4. `History.tsx` 的改動

1. **移除**現有那塊 `{pendingId === it.id && (<div className="mt-1.5 ...">…</div>)}` 就地確認列
   （`History.tsx:239-264`）。
2. 垃圾桶鈕的 `onClick` 簡化為只設 `pendingId`（本來還會切換開闔；現在對話框自己有關閉途徑）：

   ```tsx
   onClick={() => {
     setPendingId(it.id);
     setDeleteError((prev) => (prev?.id === it.id ? null : prev));
   }}
   ```

   `setDeleteError` 的 per-row 條件清除**保持不變**——這是前一輪人工裁決的結果。
3. **對話框在整頁 render 一次**，放在清單之後、`</main>` 之前，不是每列一個。要刪的那筆從
   `items.find((it) => it.id === pendingId)` 取得；找不到就 `open={false}`。
4. `detail` 組成沿用列上已有的資料，不新增 i18n key：
   `` `${t("history.rowTitle", { view, movement })} · ${fmtTime(it.created_at)}` ``。
5. `runDelete` 完全不動（成功時 splice、失敗時寫 per-row error、finally 清 `deletingId`）。
   `busy` 傳 `deletingId === pendingId`。
6. 每列的錯誤訊息區塊（`History.tsx:266-271`）保持原樣。

## 5. i18n（`frontend/src/lib/i18n.tsx`，en 與 zh-TW 兩份都要）

| key | en | zh-TW | 狀態 |
| --- | --- | --- | --- |
| `history.deleteTitle` | Delete this record? | 刪除這筆紀錄？ | **新增** |
| `history.deleteDesc` | This can't be undone. | 此操作無法復原。 | **改寫**（原為完整問句，問句移到標題） |
| `history.deleteAria` | Delete this record | 刪除這筆紀錄 | 不變 |
| `history.deleteCta` | Delete | 刪除 | 不變 |
| `history.deleteConfirm` | Delete | 刪除 | 不變 |
| `history.deleteCancel` | Cancel | 取消 | 不變 |
| `history.deleting` | Deleting… | 刪除中… | 不變 |
| `history.deleteError` | Couldn't delete this record. Please try again. | 無法刪除這筆紀錄，請再試一次。 | 不變 |

`frontend/src/test/lib.i18n.test.ts` 已在測試層強制 en→zh-Hant 的 key 對等，新 key 會被它涵蓋。

## 6. 測試

### 新檔 `frontend/src/test/components.ConfirmDialog.test.tsx`

1. `open={false}` 時完全不 render（`queryByRole("dialog")` 為 null）。
2. `open` 時 render 標題、說明、detail，且 `role="dialog"` 帶 `aria-modal="true"`。
3. Escape 觸發 `onCancel`。
4. 點 backdrop 觸發 `onCancel`；點卡片內部**不**觸發。
5. `busy` 時：Escape 不觸發 `onCancel`、backdrop 不觸發 `onCancel`、兩顆鈕皆 disabled、
   確認鈕顯示 `busyLabel`。
6. 點確認鈕觸發 `onConfirm`。
7. 開啟時焦點在取消鈕上。

### 改寫 `frontend/src/test/pages.History.test.tsx`

現有四個刪除相關測試改成走對話框（可訪問名稱不變，多數 query 仍可用），另補：

8. 對話框顯示的 `detail` 是**被點那一列**的識別文字（兩列的情境下點第二列，斷言出現的是第二列
   的動作名稱與時間，而不是第一列的）。

前一輪保留的三個測試意圖不變：刪成功後該列消失、失敗後該列還在且顯示錯誤、取消不呼叫 API。

### 驗證指令

- `yarn test`（cwd = `frontend/`）
- `yarn build`（cwd = `frontend/`）
- 後端不受影響，但合併前仍跑一次 `.venv\Scripts\python.exe -m pytest tests/`

## 7. 明確不做（YAGNI）

- 不把 `Settings.tsx` 危險區的就地確認也改成對話框（不同情境、不在本次範圍）。
- 不做 focus trap、不鎖 body scroll（與既有 dialog 慣用法一致）。
- 不做 undo／軟刪除。
- 不把 `ConfirmDialog` 提前一般化成完整的 design-system primitive；等第二個使用者出現再說。
