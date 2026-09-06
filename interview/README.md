# 每日資工所推甄面試模擬

每天 **台北時間 09:00**（UTC 01:00）由排程自動產生一份模擬面試題組，
**`config.md` 裡列的每個科目各 3 題 + x-coach 專題 3 題**，題目不與過去重複。

題目會**直接貼在當天那個 session 的對話裡**（先「作答區」只有題目、分隔線後
才是「對答區」的評分重點與追問），不必開 GitHub 才看得到。這裡的檔案是存檔，
用途是跨日去重與事後回顧。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `config.md` | 科目清單、專題出題方向、出題與評分格式。**要加減科目或調難度改這裡就好，排程不用動。** |
| `asked-questions.md` | 題庫帳本，出題前讀、出題後追加，用來去重。 |
| `sessions/YYYY-MM-DD.md` | 每天產生的題組（題目 + 評分重點 + 追問）。 |

## 存放位置（搬遷中）

正式的家是 **`lucas6028/master-application-materials` 的 `interview/`（分支 `main`）**。
排程每天第一步會 `add_repo` 取得該 repo 權限；若該 repo 尚未授權給 Claude 的
GitHub App，會退回這裡（x-coach 的 `claude/daily-taipei-interview-simulation-movzej`
分支）當暫時存放處。等授權完成，排程會在第一次成功的執行時自動把整個
`interview/` 目錄搬過去。

## 排程怎麼跑

排程每天以**全新 session** 觸發，流程固定為：

1. 呼叫 `add_repo` 取得存放 repo 的權限並 clone（排程的 session 預設沒有掛任何
   repo，少了這步 git push 會被 proxy 擋 403）。
2. 讀 `interview/config.md`（科目清單與出題規則）與 `interview/asked-questions.md`（已出過的題）。
3. 讀 x-coach 的 `project-overview.md` / `研究計畫.md` / `notes/` 產生專題題目。
4. 寫出 `interview/sessions/<今天日期>.md`。
5. 把當天所有題目的摘要追加到 `interview/asked-questions.md`。
6. commit 並 push 回同一個分支。
7. 把當天題目完整貼回對話（作答區 / 對答區兩段）。**存檔失敗也照貼**，並附上
   當天要追加到帳本的那幾行，讓去重紀錄不會斷。

## 怎麼用

- 早上直接看排程 session 的訊息：先在「作答區」把每題自己講一遍，再往下捲到
  「對答區」對答案。想事後複習再開 `sessions/YYYY-MM-DD.md`。
- 覺得某題答得爛想再練，把 `asked-questions.md` 裡對應那行刪掉，之後就可能重出。
- 要加減科目或調整難度，改 `config.md`，隔天的排程就會照新清單出題——題數是
  「科目數 × 3 + 專題 3」，不寫死在排程裡。

## 調整排程

排程是一個 Routine（cron `0 1 * * *`，UTC）。要改時間、暫停或刪除，
直接跟 Claude 說即可，或在 claude.ai 的 Routines 設定裡調整。
