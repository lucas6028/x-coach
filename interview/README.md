# 每日資工所推甄面試模擬

每天 **台北時間 09:00**（UTC 01:00）由排程自動產生一份模擬面試題組，
共 **21 題**：資工六大科各 3 題 + x-coach 專題 3 題，題目不與過去重複。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `config.md` | 六大科清單、專題出題方向、出題與評分格式。**要調整內容改這裡就好。** |
| `asked-questions.md` | 題庫帳本，出題前讀、出題後追加，用來去重。 |
| `sessions/YYYY-MM-DD.md` | 每天產生的題組（題目 + 評分重點 + 追問）。 |

## 排程怎麼跑

排程每天以**全新 session** 觸發，流程固定為：

1. `git fetch origin claude/daily-taipei-interview-simulation-movzej` 並切到該分支。
2. 讀 `interview/config.md`（出題規則）與 `interview/asked-questions.md`（已出過的題）。
3. 讀 `project-overview.md` / `研究計畫.md` / `notes/` 產生專題題目。
4. 寫出 `interview/sessions/<今天日期>.md`。
5. 把 21 題摘要追加到 `interview/asked-questions.md`。
6. commit 並 push 回同一個分支。

## 怎麼用

- 早上打開當天的 `sessions/YYYY-MM-DD.md`，**先蓋住評分重點**自己講一遍答案，
  再對照評分重點檢查漏了什麼。
- 覺得某題答得爛想再練，把 `asked-questions.md` 裡對應那行刪掉，之後就可能重出。
- 要換科目或調整難度，改 `config.md`，隔天的排程就會照新規則出題。

## 調整排程

排程是一個 Routine（cron `0 1 * * *`，UTC）。要改時間、暫停或刪除，
直接跟 Claude 說即可，或在 claude.ai 的 Routines 設定裡調整。
