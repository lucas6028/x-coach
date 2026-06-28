# EgoExo-Fitness scripts

Thin CLI wrappers over `src/egoexo/`. Run everything from the repository root.

## What this pipeline is

EgoExo-Fitness (ECCV 2024) — synchronized egocentric + exocentric fitness videos with
*interpretable action judgement*: for each single action, a technical-keypoint checklist
(TKV), a 1–5 quality score, and a natural-language comment + canonical guidance. This
pipeline targets the explainable-coaching experiments:

- **E1** guidance-based execution verification — predict the per-action TKV checklist (multi-label).
- **E3** inter-annotator agreement — the human-performance ceiling for E1/E2.
- **E7** extend the knowledge graph / RAG corpus from squat-only to all 12 actions.

The **sample unit is one judged action** = one entry in
`interpretable_action_judgement.json` (key `<record_id>_action_<N>`). Each entry is
self-contained (its own `st_ed_frame` + `frame_root`); the `action_N` index does **not**
align with `action_level_annotations.json` IDs, so we do not join against that file.

## Stage 1 — manifest, splits, labels

```bash
python scripts/egoexo/build_manifest.py
```

Reads `data/EgoExo-Fitness/raw_annotations/{meta_records,interpretable_action_judgement}.json`
and writes under `data/EgoExo-Fitness/processed/`:

| Artifact | Contents |
| --- | --- |
| `manifest.csv` | one row per judged action (913 rows): ids, participant, action, segment, views, label summaries |
| `splits/{train,val,test}_keys.json` | **participant-disjoint** sample ids (70 / 15 / 15 by action count) |
| `labels/tkv.json` | per criterion `{n_true, n_false, fault}` (default `fault` = strict-majority False) |
| `labels/score.json` | per-annotator 1–5 scores + mean/median/mode |
| `labels/comments.json` / `labels/guidance.json` | NL feedback per action / canonical form per action class |
| `criteria_catalog.json` | E1 label space: criteria per action + a global criterion index (97 criteria) |

Options: `--val-frac` / `--test-frac` (default `0.15` / `0.15`), `--annotation-root`,
`--processed-root`, `--manifest-output`.

### Splits: why participant-disjoint *by name*

`original_actor` is formatted `DATE-Name`, and the same person recorded on several dates
appears under several strings (e.g. `Jianan` spans six dates → 76 records collapse to ~41
participants). Splitting on the raw string would leak a person across train/test, so we
group by the **name** part (`participant_id`). Grouping by name is the safe direction: at
worst it merges two distinct people who share a name, which only makes the split more
conservative, never leaky.

The dataset's official ActivityNet-style substep split is **not** used for form assessment:
it places every record in both train and test (fine for temporal localization, leaky for
per-subject form tasks). Numbers obtained on that split are not directly comparable here.

## Next stages (planned)

- **CLIP-B features** — download the HuggingFace `Lymann/EgoExo-Fitness` frame-wise CLIP-B
  features and pool over `st_ed_frame` per view. (Only the first split-archive part of
  `frames_open` is on disk, so pose-based features wait on the full frame download.)
- **E1 / E3 / E7** modules under `src/egoexo/` — see `notes/` for experiment write-ups.

## Experiment Menu
Theme 1 — Interpretable form assessment (the core mission; this is the unlock)

- ★ E1 — Guidance-based execution verification (FLAGSHIP). Per action segment, predict pass/fail for each technical keypoint (multi-label, per-action heads). The output is the explanation ("back not straight," "depth insufficient") instead of a black-box "incorrect." This is the dataset's own headline task and the most direct realization of your project's thesis. Reuse the rehab24 feature→classifier infra; swap the binary head for pos-weighted multi-label. Report macro per-criterion AP/F1.
- E2 — Interpretable AQA. Predict the 1–5 score (Spearman ρ), then build a two-level model where the per-keypoint predictions aggregate into the score — "3/5 because 2 keypoints failed." Quick prior check: does #failed-keypoints correlate with the human score?
- E3 — Human ceiling / label-noise analysis (cheap, annotation-only, do in parallel). Fleiss' κ on TKV, ICC on scores. Tells you the achievable accuracy so you don't chase noise — directly addresses your past underpowered/ambiguous results.

Theme 2 — View & depth (continuation of your depth-bottleneck thread)

- ★ E4 — Ego vs Exo vs fused, paired. Train E1/E2 on ego-only / exo-only / fused features over the same actions. Reframes the depth problem as "do more views recover the faults one view misses?" — and it's deployment-real: ego = head-mounted coach (you can't see your own back), exo = gym camera. Strong testable hypothesis: exo ≫ ego on back/torso keypoints. Paired over ~900 actions → far better powered than your n=9 NLF test.
- E5 — Multi-view 3D vs monocular 3D (contingent). If calibration is obtainable, triangulate exo views → 3D and compare to your NLF monocular-3D on TKV/AQA — the clean test REHAB24-6 couldn't run. Verify the public release ships extrinsics before committing; it may not.

Theme 3 — Knowledge grounding (leverages your existing rules + KG + RAG)

- ★ E6 — Validate your squat rule detector against human labels (sleeper win). EgoExo has Sumo Squat + two lunges. Run your pose_rule_detector and check agreement between your heuristic faults and the human TKV labels (your depth rule vs "squat low enough," etc.). Your rules are currently tuned heuristics never validated against ground-truth human judgement — this is the first external validation. (Needs frames+pose.)
- E7 — Extend KG/RAG from squat-only to all 12 actions (annotation-only, immediate). Mine the 97 criteria + action_guidance + comment to grow squat_kg_v2.graphml and the RAG corpus (reuse extract_kg.py). Feeds retrieval for E1's predicted faults.
- E8 — Grounded feedback generation (downstream). Predicted faults + retrieved context → generated coaching comment, evaluated against the human comment. The full perception→retrieval→generation vision; do after E1+E7.

Recommended sequence

1. Today (no download): scaffold src/egoexo/dataset.py (manifest + subject-disjoint splits + parse TKV/score/guidance/comment to labels) + E3 + start E7.
2. As features land: E1 via CLIP-B features (skips frame and pose extraction — fastest baseline), then E4 by partitioning those features by view.
3. Once full frames extract: pose backends → E6, plus pose-feature variants of E1/E4, and E2.

Honest caveats: subject IDs aren't obvious (record_id is opaque) — recovering subject grouping for leakage-free splits is a real design task; TKV has class imbalance + per-criterion sparsity (use per-action heads, report AP not accuracy); E5 hinges on calibration that may not be public.
