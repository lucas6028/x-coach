# Adding a movement fault rule: checklist

What to check before a new fault rule in `src/pose/movements/` goes live. The stages run in
order, and each one can stop the rule; where it stops decides its status. Distilled from the
16-movement build (parent spec `docs/superpowers/specs/2026-07-18-16-movement-rule-detector-design.md`)
and the per-movement validation notes in `notes/*-rule-validation.md`.

## Rule statuses

| status | in code | meaning |
|---|---|---|
| **live** | function calls `build_detection(...)` | fires; users see a fault card |
| **silent** | registered function whose body is only `return []` | the fault is real and the citation holds, but it cannot be measured reliably yet; data can un-silence it |
| **withdrawn** | absent; a dated note in the parent spec | the claim itself fails (citation, or a measurement refuted the metric); a stub would assert something false |

Registering a detector ships every live rule in it (`GET /api/movements` derives from
`registry._REGISTRY`). "Live" is not "validated": `validated=True` is a detector-level flag, and
only Squat has it.

## 1. Citation — failure means **withdrawn**

Read the source itself, not a paraphrase. Failure modes already hit in this repo:

- The source studies a **different exercise** (the most common failure).
- It describes a **diagnostic sign**, not a cause (the impingement arc).
- The quantity or threshold **never appears** in it, or sits behind the paper's own reference
  marker (secondary sourcing).
- The source **measured a null** on the proposed proxy (Barbado's `SG_ML` did not change with
  speed).
- Source and measurement disagree on the fault's **sign** (pelvic drop vs hiking).
- The paraphrase **inverts** the source's instruction, or a counter-indication sits in the same
  source.
- The source gives a **graded family**; the spec cites one grade and implements another.
- Garbled or fabricated author string — diff against `data/paper_metadata.json`.

Verifying a quotation is not verifying the claim built on it.

## 2. Measurability — failure means **silent**

- **Landmarks exist?** MediaPipe's 33 points have nothing between shoulders (11/12) and hips
  (23/24) and nothing on the scapula: spinal curvature, winging and retraction are out.
- **Metric degenerate?** Write the construction out. Three collinear points always give 180°; a
  point's distance to a line through itself is always 0 (Row's spine rule, `row.py` header).
- **Measures what it names?** MediaPipe's shoulder tracks the humerus, so a "shrug" neck-gap
  metric reads arm elevation.
- **View:** which view does it need, and can the estimator supply it? It is inverted on supine
  subjects, and MediaPipe is not roll-equivariant.

## 3. Implementation traps

- `angle_degrees` (`src/pose/geometry.py`) is **unsigned and symmetric about 180°**: a bridge
  20° short of straight and one 20° past it read the same angle (`shoulder_bridge.py` header), so
  a "> 180°" rule never fires and a "< 160°" rule mislabels over-arching.
- Signed body-relative quantities: **dot products onto a body axis, never cross products** (those
  flip under mirroring).
- Unsigned deviation from a **pre-brace setup baseline** inverts the fault — make it directional.
- Phase-scoped rules need `min_frames = max(3, ceil(0.2·fps))` frames in the phase
  (`base.py` `run_detector`), and `segment_reps` **trims** the window. Check against the trimmed
  window; a short phase can structurally silence the rule.
- Percentile phase boundaries and absolute per-frame thresholds don't compose — score the rep's
  peak or aggregate.
- Whole-rep "not enough travel" rules fire at severity 1.0 on a **motionless clip** (jitter
  segments into reps).
- `run_detector` applies a 5-frame median; test helpers that build `CoreFrame`s directly don't.
- OR'd two-term rules pick the evidence axis by comparing **severities**; keep `score_values`
  **unclipped** — users see those numbers.
- `fault_id` carries the movement prefix and is globally unique; `kg_query` resolves in `kg`
  mode.

## 4. Controls and measurement — failure means **silent** or **withdrawn**

1. **Zero-parameter control.** Build the input with the fault removed by construction; does the
   rule still fire? Take joint counts (fires on both / only on the real input), not a difference
   of two rates. Worked example: `notes/jumping-jacks-rule-validation.md` §5.
2. **Camera control.** Simultaneous views disagreeing is pure projection error; compare it with
   the threshold (`notes/high-knee-rule-validation.md`).
3. **False-positive rate** on reps humans judged correct.
4. **Separability** against labels: AUC with participants held apart, plus per subject.
5. Fire rates through the **real `run_detector` on segmented reps**, not annotation windows.
6. **Pre-register** the pass conditions in `docs/superpowers/specs/` before running; write the
   result under `notes/` with the `write-experiment-note` skill.

No labeled data? State that in the rule's docstring and the module header. There is no numeric
go-live bar; precedents are Leg Abduction's lean rule (AUC 0.840, above chance in 9/9 subjects)
and pre-registered criteria. Going live is a proposal to the project owner, not an automatic step.

## 5. Tests

- Every test asserting silence or a guard needs a **companion on the same path that fires**.
- Mutate the rule and confirm the tests catch it.

## 6. Wiring a live rule in (each enforced by a test)

1. `build_detection(...)` with `fault_id` and `kg_query` as string literals or module-level
   constants.
2. `kg_query` resolves — `tests/test_kg_query_resolution.py`.
3. `frontend/src/lib/movementMistakes.ts`: a `mistake(...)` entry at the rule's position in
   detector source order, both languages, `why` from `citation_support` —
   `tests/test_movement_mistakes_roster.py`.
4. Regenerate `frontend/src/lib/faultCitations.json` with
   `.venv\Scripts\python.exe scripts/pose/export_fault_citations.py` — `tests/test_fault_citations.py`.
5. Replace any pinned "always silent" tests; update hard-coded rule lists in
   `tests/test_movement_registry.py`.
6. Update the module header, the rule docstring, and the parent spec.

**Registering a new detector** additionally needs: the import in `registry.py`; the movement in
`REGISTERED` (`tests/test_movement_mistakes_roster.py`) and in
`test_names_are_the_canonical_spellings`; the `movement.<Name>` i18n key; removal from
`ALL_SILENT_MODULES` (`tests/test_kg_query_resolution.py`); and, if it is Jumping Jacks,
repointing `test_unknown_movement_returns_coming_soon_without_detector` at another unregistered
movement. `validated` stays `False` until a separate decision.

## 7. Ship

Full backend suite plus the 95% coverage gate, and frontend `yarn test:coverage` (the roster and
citation JSON are in the frontend bundle). Redeploy **both** images.
