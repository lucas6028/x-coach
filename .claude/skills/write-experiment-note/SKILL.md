---
name: write-experiment-note
description: The standing convention for every note under notes/ — 寫 note、寫實驗筆記、寫實驗結果、整理實驗結果、記錄實驗、把結果寫成 note、實驗跑完要寫下來、寫事前登錄計畫、改寫 note、重寫 note、把 note 寫給大家看/寫得讓人看得懂、潤飾實驗筆記; also write, draft, or clean up an experiment note, results note, findings note, ablation/control writeup, or pre-registration plan in notes/. Use whenever an experiment, ablation, control, measurement, or validation run has finished and the outcome needs writing down — and whenever an existing notes/*.md is unclear, jargon-heavy, or written only for its author.
---

# Notes convention (`notes/*.md`)

**This is how every note in `notes/` gets written from now on** — new ones first,
rewrites second. Paths are relative to the repo root.

Worked reference: **`notes/rehab24_videomae_framing_results.md`**. Read it before
writing your first note in this genre.

A PostToolUse hook (`.claude/settings.json`) lints every `notes/*.md` written with
the **Write or Edit tool** and reports back automatically. It is advisory — never
blocks, structure only. **A `sed`/heredoc write through Bash bypasses it entirely**
(the payload carries a command string, not a file path), so when you write a note
that way, run the linter by hand.

## Writing a new note

1. **Before the experiment runs**, if it has a pre-registered plan, write
   `notes/<stem>_validation_plan.md` first: hypotheses, the primary comparison,
   exclusion rules, and the interpretation table. Fix them before any number
   exists. (See `rehab24_videomae_framing_validation_plan.md` §7–8 for the shape.)
2. **Title states the finding or the question** — never "實驗 3 結果".
3. **Front the answer**: an emphasised block before the first `##`
   (`**Question.**` / `**Verdict:**` / `**這份文件在回答什麼**`) or a first section
   that is the lead (`## 一句話` / `## 兩句話` / `## 0.`). Method comes after.
4. **Report the gates** that had to pass before you looked at any accuracy, and
   the deviations from the plan — including ones that changed nothing.
5. **Results with dispersion**: point estimate, ± std, how many subjects/folds
   went the same way, and the interval. Never a bare mean.
6. **Write the ❌ list** — see rule 4 below. Write it even when the result is good.
7. **End with reproduction**: artifact paths under `data/`, the script to re-run.
8. **Run the linter** (below) before you call it done.

## The six rules

Each is the corpus convention, not taste — cited notes are where to see it.

1. **The title states the finding or the question.**
   `videomae_b1_repeated_splits_results.md`: 「person-crop 的 +0.026 是分割雜訊」.
   `rehab24_box_geometry_control.md`: 「不泛化,但查出了更大的東西」.

2. **Front the answer.** `fit3d_sparse_depth_summary.md` (`**Question.**`),
   `camera-placement-hypothesis.md` (`**Verdict:**`), `videomae_b1_...` (`## 一句話`).

3. **`p ≥ 0.05` is "undetermined", never "no difference" and never "equivalent."**
   Pre-registered in `rehab24_videomae_framing_validation_plan.md` §7.2 and the
   single most load-bearing rule in the corpus. Claiming equivalence needs an
   equivalence test and power this project does not have.

4. **Every note carries a "what this does NOT support" section.** Named variously:
   `可以說 / 不能說`, `確立 / 未測`, `沒做的事`, `Caveats (honest)`,
   `One confound named`.

5. **Every claim traces to a measured number, a pre-registered rule, or a cited
   note** — never to reasoning you invented while writing. If a rule lives in a
   sibling `*_validation_plan.md`, **read the plan and spell the rule out inline**;
   a bare `§8` is unreadable to anyone without that file open.

6. **End with reproduction.** See the `## 重現` / `Reproduce:` tail in
   `model_fusion_gate_results.md`, `fit3d_sparse_depth_summary.md`.

## Register — who it is for

Default audience: **a lab-mate who knows basic ML but nothing about this project.**
Keep the statistics (balanced accuracy, LOSO, Wilcoxon, Holm); kill the private
codes. Concretely:

- Every internal identifier (`full_frame_letterbox`, `person_crop`, arm/臂, gate,
  floor) gets a one-line plain gloss on first use. Keep the identifier — it matches
  filenames — but never use one naked.
- `P0`/`P1`/`P2`-style labels: say what the thing is *for* instead.
- Add a 名詞說明 section when the note leans on more than ~3 pieces of vocabulary.
- `[[some-name]]` is a **memory-file link** — it dangles for every reader outside
  this machine. Cite the note path or the number in prose.

If the user asks for a wider audience than that, ask what they want dropped — do
not unilaterally cut the statistics.

## Harness

```bash
# structural lint (one file, or the whole corpus)
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/foo.md
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/*.md

# REWRITES ONLY: did any measured figure disappear vs git HEAD?
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py --numbers notes/foo.md
```

The linter checks structure only. Register, whether jargon was actually expanded,
and whether a claim traces to a measured number are **not** machine-checkable.
A `pass` does not mean the note is good.

Baseline for calibration: **24 of 39 notes pass, including all 15 current-standard
results notes**; the 15 failures are older `## 背景`-first summaries, walkthroughs,
and dataset references. If a note you consider good fails, fix the rule in
`check_note.py`, not the note.

## Rewriting an existing note

Everything above still applies. Additionally:

1. `--numbers` before and after. **Every figure survives verbatim.**
2. **Move, don't delete.** Audit-trail material (plan deviations, extraction logs,
   gate output, reproduction checks) goes to appendices — but a gate result that is
   itself a finding stays in the body.
3. **Collapse revision-history narration.** A note saying 「本 note 的前一版寫…」 is
   talking to itself; state the resolved position.
4. **A stale claim contradicting a later section is a real bug — fix it by grounding
   in a measured number, never by authoring new reasoning.** If you cannot ground it,
   flag it to the user instead of inventing a justification.
5. **A derived number is not a verified number.** Re-deriving `87.6% ≈ (75.3+100)/2`
   is a guess until you open the JSON.

## Gotchas

- **No bare `python` on this machine** — a guard hook blocks it. Always
  `.venv/Scripts/python.exe`.
- **The Bash and PowerShell tools share one cwd.** A `cd notes` in one call persists
  and silently breaks every later relative path. Use paths from the repo root.
- **`jq` is not installed here.** The hook parses its stdin payload in Python for
  exactly this reason — do not "simplify" it back to a jq pipeline.
- **Multiset number-diffing is useless** for a restructuring rewrite — section
  numbers (`§7.2`, `## 3.1`) dominate the noise. `check_note.py` strips those and
  set-diffs; do not change it back to counting occurrences.
- **The equivalence warning over-fires by design.** It flags any line pairing a
  p-value with 等價/沒有差異 — including lines that correctly say "p = 0.570 是
  未定,不是等價". Read it, don't silence it.
- **Results tables can disagree across sections legitimately** (a pooled rate vs a
  per-camera vs a per-frame rate). When two of your own numbers look contradictory,
  label the measurement definition — a reader will otherwise read it as an error.
