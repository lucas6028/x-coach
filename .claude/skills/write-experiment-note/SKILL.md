---
name: write-experiment-note
description: The standing convention for every note under notes/ — 寫 note、寫實驗筆記、寫實驗結果、整理實驗結果、記錄實驗、把結果寫成 note、實驗跑完要寫下來、寫事前登錄計畫、改寫 note、重寫 note、把 note 寫給大家看/寫得讓人看得懂、潤飾實驗筆記; also write, draft, or clean up an experiment note, results note, findings note, ablation/control writeup, or pre-registration plan in notes/. Use whenever an experiment, ablation, control, measurement, or validation run has finished and the outcome needs writing down — and whenever an existing notes/*.md is unclear, jargon-heavy, or written only for its author. Notes are written in English regardless of the language of the request.
---

# Experiment notes (`notes/*.md`)

**Notes are written in English.** This holds regardless of the language the request came
in; ask for a note in Chinese and the note is still English. Only reply to the user in
their language, and only outside the file.

The genre is an **internal analysis memo**: a curated retrospective report, organised by
question and result, written after the run for a teammate who was not there. It is *not*
a lab notebook. Chronology, dead ends, timing benchmarks and failed attempts do not
belong in the note; if they need keeping, put them in a run log next to the artifacts.

Reference note, matching this convention exactly:
**`notes/rehab24_videomae_position_control_results.en.md`**. Read it before writing your
first note in this genre. Older notes in `notes/` are Chinese and predate this rule; they
are still valid, but do not copy their register.

## The skeleton

```markdown
# <Declarative title carrying the finding, with the number if there is one>

*<dataset> · <what kind of experiment> · run <date> · <link to the plan or sibling note>*

**Result.** The headline in two or three sentences: what was done, the one number that
answers the question, with dispersion and n.

**Caveat on reading it.** Only if something qualifies the headline: a failed gate, a plan
deviation, a scope limit that a reader must carry with the number. Omit if there is none.

## Background
Why this experiment had to happen. Two to four sentences and a pointer to the prior note.
Not a motivational essay.

## Method
What was measured or intervened on, and the one design decision a reader would otherwise
get wrong. Then the procedure, compactly. State what was fixed before results existed.

## Gates
Table of the checks that had to pass before any outcome was looked at, with pass/fail.
A failed gate goes here in full, not in a footnote.

## Results
The main table: arms as rows, the statistic with dispersion, n above chance, interval,
p. Per-unit values below it. Then at most three sentences saying what pattern to see.

## Secondary
Analyses that qualify but never replace the main result. Compact.

## <Bounds / controls>
Zero-parameter or upper-bound controls, named for what they measure.

## Deviations from the plan
Table: item, what the plan said, what happened, what it changes. Include deviations that
changed nothing.

## Not supported
Bulleted. What the result does not license, including the limits a reader would otherwise
assume away.

## Reproduce
Code paths, the commands in order, runtime and hardware, artifact paths.
```

Sections are dropped when they have no content, not padded. A note without a plan has no
Gates or Deviations section; a note with no controls has no Bounds section. The spine that
always survives: title, Result, Method, Results, Not supported, Reproduce.

## Register

Flat reporting. The reader already wants the answer, so nothing needs to be sold.

- **Declarative title stating the finding**, with the number when there is one.
  "Removing the linear position subspace costs 0.019 of within-session AUC", not
  "Position control results" and not a question.
- **Tables carry the numbers.** Prose says what pattern to see in them, in a few
  sentences. A measurement staged as a prose set-piece is a rhetorical device.
- **No second person, no rhetorical framing.** Cut "a nastier story", "this door was wide
  open", "sail straight through untouched", and open questions ranked by how much they
  should worry the reader. Vivid prose gets audited less carefully than flat prose, which
  is the opposite of what a control experiment is for.
- **One idea per sentence.** Short paragraphs. No narrative walk-through; if one session
  or fold is worth showing, give its numbers in three lines and move on.
- **Standard terminology, used as-is.** Expand an uncommon acronym on first use. Do not
  invent shorthand, and do not translate terms of art.
- **Identifiers get a one-line gloss on first use.** `full_frame_letterbox`,
  `person_crop`, arm, gate, floor. Keep the identifier, it matches filenames, but never
  use one naked. Say what a `P0`/`P1`-style label is *for* instead of using the label.
- **Add a terminology table** when the note leans on more than about three pieces of
  project-specific vocabulary.
- **`[[some-name]]` is a memory-file link** and dangles for every reader outside this
  machine. Cite the note path or the number in prose.

Default audience: a teammate who knows basic ML but nothing about this project. Keep the
statistics. If a wider audience is wanted, ask what to drop rather than cutting them.

## The rules

1. **Write the plan first.** If the experiment is pre-registered, write
   `notes/<stem>_validation_plan.md` before the run: hypotheses, the primary comparison,
   exclusion rules, the interpretation table. Fix them before any number exists, and
   commit it before producing results.
2. **The title states the finding**, per Register above.
3. **Front the answer** in a `**Result.**` block before the first `##`. Method comes after.
4. **Report the gates**, including the ones that failed, and every plan deviation,
   including the ones that changed nothing.
5. **Every number carries dispersion**: point estimate, ± SD, how many subjects or folds
   went the same way, and the interval. Never a bare mean.
6. **`p ≥ 0.05` is "undetermined"**, never "no difference" and never "equivalent". An
   equivalence claim needs an equivalence test and power this project does not have. This
   is the single most load-bearing rule in the corpus.
7. **A number that could be a floor or a ceiling is labelled as one.** If a control was
   only partly effective, say what that does to the estimate.
8. **Every claim traces to a measured number, a pre-registered rule, or a cited note**,
   never to reasoning invented while writing. When a rule lives in a sibling
   `*_validation_plan.md`, read the plan and spell the rule out inline; a bare `§8` is
   unreadable to anyone without that file open.
9. **Write the "Not supported" section**, even when the result is good.
10. **End with reproduction**: code paths, commands, runtime, artifact paths under `data/`.
11. **Run the linter** before calling it done.

## Rewriting an existing note

Everything above still applies. Additionally:

1. **`--numbers` before and after.** Every measured figure survives. When the diff reports
   a dropped token, check it individually: a figure now written at higher precision, or a
   narrative restatement of a number that is still present, is fine; a lost measurement is
   not.
2. **Move, don't delete.** Audit-trail material goes to an appendix. A gate result that is
   itself a finding stays in the body.
3. **Collapse revision-history narration.** A note saying "the previous version said…" is
   talking to itself. State the resolved position.
4. **A stale claim contradicting a later section is a real bug.** Fix it by grounding in a
   measured number, never by authoring new reasoning. If it cannot be grounded, flag it to
   the user instead of inventing a justification.
5. **A derived number is not a verified number.** Re-deriving `87.6% ≈ (75.3+100)/2` is a
   guess until the JSON is open.
6. **Translating an existing Chinese note is a rewrite**, so `--numbers` applies to it.

## Harness

```bash
# structural lint (one file, or the whole corpus)
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/foo.md
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py notes/*.md

# REWRITES ONLY: did any measured figure disappear vs git HEAD?
.venv/Scripts/python.exe .claude/skills/write-experiment-note/check_note.py --numbers notes/foo.md
```

A PostToolUse hook (`.claude/settings.json`) lints every `notes/*.md` written with the
Write or Edit tool and reports back. It is advisory and never blocks. **A `sed`/heredoc
write through Bash bypasses it entirely**, so after one of those, run `check_note.py` by
hand.

The linter checks structure only. Register, whether terminology was actually glossed, and
whether a claim traces to a measured number are **not** machine-checkable. A pass does not
mean the note is good.

Calibration: 29 of 44 notes pass. The 15 failures are older `## 背景`-first summaries,
walkthroughs and dataset references that predate the convention. If a note you consider
good fails, fix the rule in `check_note.py`, not the note.

## Gotchas

- **No bare `python` on this machine.** A guard hook blocks it. Always
  `.venv/Scripts/python.exe`.
- **The Bash and PowerShell tools share one cwd.** A `cd notes` in one call persists and
  silently breaks every later relative path. Use paths from the repo root.
- **`jq` is not installed here.** The hook parses its stdin payload in Python for exactly
  this reason; do not "simplify" it back to a jq pipeline.
- **Multiset number-diffing is useless** for a restructuring rewrite, because section
  numbers dominate the noise. `check_note.py` strips those and set-diffs; do not change it
  back to counting occurrences.
- **The equivalence warning over-fires by design.** It flags any line pairing a p-value
  with an equivalence word, including lines that correctly say "p = 0.570 is undetermined,
  not equivalent". Read it, do not silence it.
- **Results tables can disagree across sections legitimately**: a pooled rate, a
  per-camera rate and a per-frame rate are different measurements. When two of your own
  numbers look contradictory, label the measurement definition, or a reader will read it
  as an error.
