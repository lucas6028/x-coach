# Connecting the three movement detectors to the web app

Date: 2026-07-25
Branch: `feat/movement-rule-detector-spec`
Status: design, approved for planning

## 1. Context

Three movement rule detectors are implemented and registered:
`Squat` (`src/pose/movements/squat.py`), `Overhead Press`
(`src/pose/movements/overhead_press.py`), `Push-up` (`src/pose/movements/pushup.py`).

They are already reachable from the ML pipeline. `src/pose/process_videos.py` (MediaPipe
extraction) is movement-agnostic — it emits 33 landmarks per frame and knows nothing about
exercises. `detect_pose_rules_from_payload` already takes a `movement` argument, resolves it
through `src/pose/movements/registry.py`, and runs the matching detector
(`pose_rule_detector.py:595-599`). The CLI exposes this as `--movement`.

**The gap is entirely in the web app**, which pins every analysis to Squat:

| Location | Current state |
| --- | --- |
| `backend/app/config.py:29` | `DEFAULT_ANALYSIS_MOVEMENT = "Squat"` |
| `backend/app/services/analysis.py:91` | passes that constant; no per-request movement |
| `backend/app/routers/analyze.py:26` | `POST /api/analyze` accepts a file and nothing else |
| analysis result payload | carries no `movement` field at all |
| `backend/app/services/store.py:131` | `persist_analysis` never records the movement |
| `backend/app/services/chat.py:51,70` | prompt hardcodes "the x-coach **squat** coach" |
| `frontend/src/lib/movements.ts:35` | `ANALYZABLE_MOVEMENTS = ["Squat"]`, hand-maintained |
| `frontend/src/App.tsx:64` | `api.analyzeUpload(file)` — no movement, no picker |

## 2. Goal

Let a user choose Squat, Push-up, or Overhead Press in the UI and have that choice drive which
detector runs, then survive into history, the chat coach, and the rendered verdict.

## 3. Non-goals

- **Automatic movement classification from video.** The movement is user-asserted input.
- **The demo library.** `backend/app/services/library.py:154` stays pinned to
  `DEFAULT_ANALYSIS_MOVEMENT`; its clips are pre-processed squats. No new demo clips are sourced.
- **A movement/video plausibility check.** Explicitly considered and declined; see §9.
- **The remaining 13 movements.** They stay listed-but-inert on `/movements`.
- **Client-side MediaPipe capture** (PR #44 `/api/analyze/pose`) is not on this branch and is out
  of scope. The `movement` parameter designed here is the same seam that path will use.

## 4. Decisions

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | Movement is chosen on `/movements`, carried as `/app?movement=X`, and confirmed by a selector in the studio. | `/movements` already exists as a menu of all 16; making three cards live is the natural entry point. The studio selector means the choice is visible and changeable at upload time, not just at navigation time. |
| D2 | The **Python registry is the source of truth** for which movements are analyzable and whether each is validated. A new `GET /api/movements` derives the list from `_REGISTRY`. | `ANALYZABLE_MOVEMENTS` is currently a hand-maintained frontend constant whose own comment warns it must be updated by hand. With 3 entries now and 13 coming, two lists will drift. Registering a 4th detector should make it appear in the UI with zero frontend edits. |
| D3 | `Push-up` and `Overhead Press` are marked **Beta** in the UI; `Squat` is not. | Only Squat is validated against labeled data. The other two are literature-derived and never checked against ground truth. |
| D4 | `MovementDetector.validated` defaults to **`False`**; Squat opts in. | A future detector then fails toward Beta rather than silently presenting as validated. |
| D5 | No plausibility check on movement-vs-video. Instead, **every verdict names the movement whose rules ran**. | §9. |
| D6 | Scope covers chat grounding and a history `movement` column + badge. Demo library excluded. | Chat is a correctness issue (a push-up analysis must not get squat advice); history is needed to tell past analyses apart. |

## 5. Architecture

### 5.1 ML layer (`src/`)

**`src/pose/movements/base.py`** — `MovementDetector` gains a field:

```python
@dataclass(frozen=True)
class MovementDetector:
    name: str
    metric_keys: tuple[str, ...]
    compute_raw: Callable[[Sequence[object], float], list[dict]]
    assign_phases: Callable[[list[dict]], list[str]]
    rules: tuple[RuleFn, ...]
    validated: bool = False   # NEW — last field; the dataclass is frozen and positionally constructed
```

`validated` is last so the three existing positional constructor calls
(`squat.py:302`, `pushup.py:1592`, `overhead_press.py:695`) keep working unchanged.

**`src/pose/movements/squat.py`** — passes `validated=True`. Push-up and OHP take the default.

**`src/pose/movements/registry.py`** — add:

```python
def list_detectors() -> list[MovementDetector]:
    """Every registered detector, in a deterministic order (registration order)."""
```

Order is registration order, which is import order in `registry.py` (Squat, Overhead Press,
Push-up) — deterministic, and puts the validated detector first without a sort key that encodes
a UI preference in the ML layer.

**`src/pose/pose_rule_detector.py`** — in `detect_pose_rules_from_payload`, after resolving the
detector, add the canonical echo to the result dict:

```python
"movement": detector.name,
```

Placed at the top level alongside `view` / `quality` / `detections`. Two consequences worth
stating: the CLI's written JSON gains the field for free, and a stored analysis becomes
self-describing — it records which rules produced it, permanently, without depending on a
database column.

The echo uses `detector.name`, not the caller's string, so `--movement push-up` normalises to
`"Push-up"`. This matters: that exact string is what `retrieve_graph_context(movement=)` scopes
the KG by and what the frontend's `movement.Push-up` i18n key is looked up with. Detector names
were verified to be exactly `"Squat"`, `"Push-up"`, `"Overhead Press"`.

### 5.2 Backend (`backend/app/`)

**`backend/app/routers/movements.py`** (new):

```
GET /api/movements  ->  {"movements": [{"name": "Squat", "validated": true}, ...]}
```

Public, matching the existing `/api/knowledge/*` endpoints. `/movements` itself is behind
`RequireAuth` (`main.tsx:73-79`), but **`/app` is not** — it is the anonymous public demo, and the
studio needs the list to render its selector and validate `?movement=` before enabling the
dropzone. Derived from `registry.list_detectors()`. Registered in `backend/app/main.py` alongside
the existing routers.

**`backend/app/routers/analyze.py`** — `POST /api/analyze` gains:

```python
movement: str = Form(config.DEFAULT_ANALYSIS_MOVEMENT),
```

A multipart form field, so the existing single-file request shape still works and an omitted
field defaults to Squat (backward compatible for any caller not yet updated).

Validation happens **before** `save_upload` and before pose extraction: an unregistered movement
returns `400` immediately rather than burning a full MediaPipe pass. Validation is a registry
membership test, matching `registry.get_detector`'s case-insensitive key lookup.

**`backend/app/services/analysis.py`** — `analyze_video_file(source_path, *, video_id=None,
movement=None)`; `movement or config.DEFAULT_ANALYSIS_MOVEMENT` is forwarded to
`detect_pose_rules_from_json`.

**`backend/app/services/store.py`** — extend `_summarize` (used at line 159) to also return the
movement, read from `result["movement"]`, and include it in the `analyses` insert. Extending
`_summarize` rather than reading `result["movement"]` inline at the insert keeps every derived
column on one seam.

**`db/migrations/20260725000000_analysis_movement.sql`** (new):

```sql
ALTER TABLE analyses ADD COLUMN movement text;
```

Nullable and additive: existing rows keep working, RLS policies are unaffected, and the
`admin_user_overview` view (`db/migrations/20260713000200_admin_user_overview.sql`) needs no
change. This is a manual migration step for the operator, consistent with how prior migrations
in this repo are applied.

**`backend/app/services/chat.py`** — `_SYSTEM_PREAMBLE` and `_FOLLOWUP_INSTRUCTION` become
functions of the movement name:

- `"You are the x-coach squat coach ... one squat repetition"` →
  `"You are the x-coach {movement} coach ... one {movement} repetition"`.
- `"about THIS squat"` → `"about THIS {movement}"`.
- The CLEAN REP branch (`chat.py:169`) names the movement:
  `"This is a CLEAN {movement} REP: no {movement} faults were detected."`

The movement comes from `ChatContext.movement` (new optional field, defaulting to
`DEFAULT_ANALYSIS_MOVEMENT` so an older client's payload still renders a coherent prompt).

**`backend/app/config.py`** — `DEFAULT_ANALYSIS_MOVEMENT` stays. It is now the *fallback*
(library path, omitted form field) rather than the pin. Its comment is updated to say so.

**`backend/app/main.py`** — the app description's "video analysis is squat-only today" is no
longer true; update it.

### 5.3 Frontend (`frontend/src/`)

**`frontend/src/lib/movements.ts`** — `ANALYZABLE_MOVEMENTS` and `isAnalyzable` are removed.
`MOVEMENT_GROUPS` / `ALL_MOVEMENTS` (the browsable catalog of all 16) stay — that list is a
content decision, not a pipeline fact. Their only consumers are `Movements.tsx:42` and
`frontend/src/test/pages.Movements.test.tsx:66,73`, which asserts card counts against the
constant and must be reworked to mock the fetch instead. A new type describes the fetched entry:

```ts
export interface AnalyzableMovement { name: string; validated: boolean }
```

**`frontend/src/api.ts`** — `getMovements()`; `analyzeUpload(file, movement)` appends the
`movement` form field; `ChatContext` and `Analysis` gain `movement?: string`.

**`frontend/src/lib/grounding.ts`** — `buildChatContext` copies `analysis.movement` through.

**`frontend/src/pages/Movements.tsx`** — fetches `/api/movements` and renders each card as live
(with a Beta tag when `validated` is false) or "Soon", instead of consulting a local constant.
Live cards navigate to `/app?movement=<name>`. If the fetch fails, fall back to Squat-only so the
page still functions.

**`frontend/src/App.tsx`** — reads `?movement=` from the URL, validates it against the fetched
list *before* enabling the dropzone (a hand-typed `?movement=Lunge` must not cost the user an
upload to discover a 400), renders a movement selector plus the Beta note, and passes the choice
to `analyzeUpload`.

**`frontend/src/components/CoachTray.tsx`** — the clean-rep banner names the movement.

**`frontend/src/pages/History.tsx`** — a movement badge per row, sourced from the column and
falling back to `result.movement`, then to `"Squat"` for rows predating both.

**`frontend/src/lib/i18n.tsx`** — `feedback.noFaults` becomes interpolated
(`"No {movement} faults detected. Clean rep."`); the `t()` implementation already supports
`{var}` substitution (`i18n.tsx:1311-1314`). New keys for the Beta tag, the Beta explanation, and
the movement-selector label, **in both locales** — the parity guard added in `2a5d3e64` asserts
every `en` key has a `zh` translation.

All 16 movement display names already exist in both locales via `movementLabel` /
`dataLabel` (`i18n.tsx:1339`), so no per-movement label work is needed.

## 6. Data flow

```
/movements  ──fetch──►  GET /api/movements   ◄── registry.list_detectors()
    │                   [{Squat,true},{Overhead Press,false},{Push-up,false}]
    │ click card
    ▼
/app?movement=Push-up
    │ validate against fetched list ──► invalid: dropzone stays disabled
    │ upload
    ▼
POST /api/analyze  (file + movement)
    │ registry membership check ──► 400 before any extraction
    ▼
analysis.analyze_video_file(path, movement="Push-up")
    │
    ├─► process_video()                          # MediaPipe, movement-agnostic
    └─► detect_pose_rules_from_json(movement=)
            │ registry.get_detector("push-up")
            │ run_detector(...)
            │ result["movement"] = detector.name  # "Push-up", canonical
            └─► retrieve_contexts_for_detections(movement=)   # KG scoped to Push-up
    ▼
result
  ├─► store.persist_analysis  ──► analyses.movement column   (History badge)
  ├─► buildChatContext        ──► ChatContext.movement       (coach prompt)
  └─► CoachTray / MetricsCards                               (verdict names the movement)
```

The library path (`services/library.py`) is untouched and keeps passing
`DEFAULT_ANALYSIS_MOVEMENT`. It nonetheless gets `result["movement"] = "Squat"` for free from the
shared `detect_pose_rules_from_payload`, so history badges and chat grounding work for library
clips with no special case.

## 7. Error handling

| Case | Behaviour |
| --- | --- |
| Unregistered movement in the form field | `400` before `save_upload` and before pose extraction — a bad request costs no compute. |
| Hand-typed `?movement=Lunge` | Frontend validates against the fetched list; the dropzone stays disabled with an explanatory note. No round trip. |
| `GET /api/movements` unreachable | Frontend falls back to Squat-only. The studio still works; the other two cards read "Soon". |
| Analysis rows predating the column | `movement` is null → badge falls back to `result.movement` (present for anything analyzed after this change), then to `"Squat"`. |
| `ChatContext.movement` absent (older client) | Defaults to `DEFAULT_ANALYSIS_MOVEMENT`; the prompt stays coherent. |

## 8. Testing

**ML (`tests/`)** — `detect_pose_rules_from_payload` emits `movement` for each of the three
detectors and normalises case (`"push-up"` → `"Push-up"`); `list_detectors()` returns all three
with the expected `validated` values; `MovementDetector` construction still works positionally.

**Backend (`tests/`)** — `GET /api/movements` shape and contents; `POST /api/analyze` rejects an
unknown movement with `400` *without* invoking pose extraction; a valid movement reaches
`analyze_video_file`; the omitted field defaults to Squat; `_summarize` carries the movement into
the insert; the chat system prompt names the movement in the preamble, the follow-up instruction,
and the clean-rep branch.

`tests/test_chat_endpoint.py` gained 57 lines in `2a5d3e64` that assert on prompt text, including
the CLEAN REP branch string. Parameterising `_SYSTEM_PREAMBLE` and that branch changes text those
assertions match on, so **existing tests get updated, not just new ones added** — a red chat suite
during implementation is expected here, not a regression signal.

**Frontend (`frontend/src/test/`)** — `Movements.tsx` renders live/Beta/Soon from the fetched
list and falls back on fetch failure; `App.tsx` rejects an unknown `?movement=`;
`analyzeUpload` sends the form field; `buildChatContext` carries the movement; the CoachTray
clean-rep banner names it; the History badge falls back correctly; the i18n parity guard passes
with the new keys.

**Verification gates — both must pass:**

```
.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95
```

```
# cwd MUST be frontend/ — the Bash and PowerShell tools share one cwd, and a stray
# cd to the repo root mass-fails vitest
yarn test:coverage
```

Note `test_concurrent_analyses_are_bounded` is a known load-dependent flake (recorded in
`2a5d3e64`), not a regression signal.

## 9. Residual risk: a measurable clip measured by the wrong rules

**This change creates a failure state that was previously unreachable, and does not close it.**

While every analysis was pinned to Squat, "the wrong detector ran" was impossible by
construction. Letting the user choose makes it possible, and the validity gates do not catch it:

- `ohp_compute_raw` (`overhead_press.py:99-105`) requires only both shoulders, elbows, wrists and
  hips to mark a frame `valid`. All four pairs are plainly visible in squat footage.
- `pushup_compute_raw` (`pushup.py:473-480`) adds both ankles — also visible in a squat clip.

So a squat video analyzed as "Overhead Press" yields `valid_frame_ratio ≈ 1.0`, hence
`measured = True` in `chat.py`'s branch (`valid_frame_ratio > 0`, mirroring
`frontend/src/lib/quality.ts`). If the OHP rules then emit nothing — plausible, since no press
phase exists in the clip — `detections` is empty and the prompt takes the
`elif not faults:` CLEAN REP branch.

The `wasMeasured` guard landed in `2a5d3e64` closes *unmeasurable* clips. It cannot close this
one: the clip **was** measured, just by rules that do not describe it. The `MODULE-WIDE SILENCE
RISK` notes at the top of both modules document the `valid=False` route and not this one.

**Accepted mitigation (D5): every verdict names the movement whose rules ran.** The studio header
shows the active movement throughout, the clean-rep banner reads "No Overhead Press faults
detected", and the coach says "clean Overhead Press rep". The claim is then true *given the
user's own assertion about what they filmed*, and the assertion is on screen at the moment of the
verdict rather than buried in a URL parameter.

**What this does not do:** it does not detect the mismatch. A user who sets the dropdown wrong and
does not read the label still receives a congratulatory verdict on a movement they did not
perform. Two closures were considered and deferred:

1. **A low-observability stub per movement**, following the precedent of squat's `knees_forward`
   block (`pose_rule_detector.py:383-407`), which emits a `severity=0.0, observability="low"`
   detection when its view gate fails so `detections` is never empty-because-unevaluable. Applying
   the same pattern to OHP and Push-up would make the CLEAN REP branch unreachable in this state.
2. **An orientation plausibility check** (push-up expects a horizontal body axis; squat and OHP
   vertical), warning on disagreement.

Either would close it at the cost of new detection logic and thresholds. Both are recorded here
as follow-up work rather than silently omitted.

## 10. Prerequisite: three Overhead Press `kg_query` strings do not resolve

The data-flow diagram shows `retrieve_contexts_for_detections(movement=)` as a working step. It is
working for two of the three movements. Measured on this branch against
`data/kg/sports_kg_v3.graphml` (present locally; the artifact is gitignored, so this must be
re-checked on any deploy target):

| Movement | `kg_query` strings resolving to a node |
| --- | --- |
| Squat | 4 / 4 |
| Push-up | 4 / 4 |
| **Overhead Press** | **2 / 5** |

The three that return `matched_nodes=[]`, `results=[]`:

| `kg_query` | Source |
| --- | --- |
| `"Incomplete Elbow Lockout"` | `overhead_press.py:362` |
| `"Lumbar Hyperextension"` | `overhead_press.py:417` |
| `"Asymmetric Press"` | `overhead_press.py:475` |

This is **not** a missing-knowledge problem. `list_movement_faults(movement="Overhead Press")`
returns 89 fault nodes — more than Squat's 74 or Push-up's 31. The strings simply do not match the
node names the graph actually carries: it has `Excessive Lower Back Arching` (degree 3),
`Thoracolumbar Extension`, `Elbows Locked`, and no `Asymmetric Press` node at all.

**Consequence if shipped as-is.** Those three faults reach `_build_system_prompt` with an empty
retrieval context, and `_fmt_list` (`chat.py:102-107`) renders each as `—`:

```
  2. Excessive back-lean / lumbar hyperextension (rib flare) — phase: ..., severity: 0.72
     likely causes: —
     injury risks: —
     corrective cues: —
```

The coach names the fault and can say nothing about it — in a project whose stated purpose is
*explainable* coaching feedback. Squat never exposed this because all four of its strings resolve.

**Decision: re-pointing these three strings is knowledge work, not wiring, and is a prerequisite
to marking Overhead Press live — not part of this change.** `Lumbar Hyperextension` →
`Excessive Lower Back Arching` looks mechanically obvious, but `Incomplete Elbow Lockout` →
`Elbows Locked` does not: the graph node plausibly describes locking out *too hard*, the opposite
fault. Guessing a mapping to make retrieval non-empty would produce confidently-worded advice
about the wrong mechanism, which is worse than the `—` it replaces. Each re-point needs the same
verification the push-up strings received (`pushup.py:1502` carries a comment recording exactly
that check).

Two acceptable orderings, to be settled in the implementation plan:

1. Resolve the three strings first, then ship all three movements live.
2. Ship this change with Overhead Press still gated off the `/api/movements` list, and open it once
   the strings resolve. Push-up goes live either way.

**Related, lower severity:** `"Forward Head Posture"` is shared by `pushup.py:1274` and
`overhead_press.py:652` and matches a movement-generic `Cause` node whose only 1-hop edge runs to
`Overhead Press:Subacromial Impingement Syndrome`. A push-up head-drop fault therefore retrieves an
OHP-specific risk. It resolves, so it is not silent, but the `movement=` scope does not actually
isolate it. Worth a look during implementation; not a blocker.

**Not an issue:** RAG-mode retrieval (`pose_rule_detector.py:674-680`) passes no `movement` filter
to `query_vector_db`, so a `rag`-mode rule could retrieve a doc for a different movement from the
general tier. Verified this cannot fire here — `retrieval_mode="rag"` appears only at
`squat.py:277` (heel rise). Push-up and Overhead Press are entirely `kg`-mode. No change needed;
recorded so the next detector's author knows the filter is absent.

## 11. Out of this change but worth noting

- `backend/app/routers/analyze.py:31`'s docstring says "Accept a squat video"; update it.
- `frontend/src/lib/i18n.tsx:234` describes x-coach as reading "a squat video"; and
  `movements.subtitle` says "Squat analysis is live today; the rest are on the way." Both become
  inaccurate and need rewording in both locales.
