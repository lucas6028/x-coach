# Movement Expansion + KG/RAG Collection Plan

Status: **decisions locked** · Created 2026-07-05

**Locked decisions (2026-07-05):**
- Flagship set = **5 movements**: Squat (anchor), Lunge, Push-up, Overhead Press, Row.
- Canonical merge policy = **loose, by movement pattern**: Sumo Squat→Squat;
  all lunge variants→Lunge; Kneeling Push-up→Push-up; one-arm row ≡ barbell row→Row.
  Maximizes flagship subgraph size and cross-dataset sample count.

Goal: expand x-coach beyond squat by adding new movements, tiered by cross-dataset
overlap ("flagship" = appears in ≥2 of our datasets, "general" = single dataset),
and organize their RAG + KG knowledge. This doc records the verified overlap matrix,
the flagship/general split, the KG schema generalization, and the collection plan.

## 1. Verified dataset movement inventory

Authoritative per-dataset lists (verified from paper/repo, not recall — see note below):

- **Fitness-AQA** (ECCV'22, Parmar et al.) — 3 exercises:
  `Back Squat`, `Barbell Row`, `Overhead Press`.
  (NOTE: **not** deadlift — a common misremembering. Binary + per-error labels.)
- **REHAB24-6** — 6 exercises (from `src/rehab24/dataset.py`):
  `arm abduction`, `arm VW`, `push-up`, `leg abduction`, `leg lunge`, `squat`.
- **EgoExo-Fitness** (ECCV'24) — 12 action categories:
  `Kneeling Push-ups`, `Push-ups`, `Kneeling Torso Twist`,
  `Knee Raise & Abdominal Contract`, `Shoulder Bridge`, `Sit-ups`,
  `Leg Reverse Lunge`, `Leg Lunge with Knee Lift`, `Sumo Squat`,
  `Jumping Jacks`, `High Knee`, `Clap Jacks`.
- **Fit3D** (CVPR'21, AIFit) — 47 exercises (17 unnamed warmups + ~30 named).
  Confirmed named exercises relevant here: `squat`, `deadlift`,
  `dumbbell reverse lunge`, `diamond push-up`, `one-arm row`,
  `overhead shoulder press`, `dumbbell bicep curl`, `band pull apart`,
  side/lateral raises. (Full 30-name list still to be pulled from the Fit3D site.)

## 2. Cross-dataset overlap matrix (canonical movement families)

| Canonical movement | Fit3D | Fitness-AQA | REHAB24-6 | EgoExo-Fitness | #datasets | Tier |
|---|---|---|---|---|---|---|
| **Squat**          | squat | back squat | squat | sumo squat* | **4** | ★ Flagship (anchor — built) |
| **Lunge**          | db reverse lunge | — | leg lunge | reverse lunge + knee-lift lunge | **3** | ★ Flagship |
| **Push-up**        | diamond push-up | — | push-up | push-ups + kneeling push-ups | **3** | ★ Flagship |
| **Overhead Press** | oh shoulder press | overhead press | — | — | **2** | ★ Flagship |
| **Row**            | one-arm row* | barbell row | — | — | **2** | ★ Flagship (implement differs) |
| Torso/Ab twist     | standing ab twist* | — | — | kneeling torso twist* | 2? | borderline (weak match) |
| Deadlift           | deadlift | — | — | — | 1 | General |
| Bicep Curl         | db bicep curl | — | — | — | 1 | General |
| Band Pull Apart    | band pull apart | — | — | — | 1 | General |
| Arm abduction      | — | — | ✓ | — | 1 | General |
| Arm VW / raise     | (w/lateral raise?) | — | ✓ | — | 1–2? | General |
| Leg abduction      | — | — | ✓ | — | 1 | General |
| Sit-up             | — | — | — | ✓ | 1 | General |
| Shoulder Bridge    | — | — | — | ✓ | 1 | General |
| Jumping/Clap Jacks | — | — | — | ✓✓ | 1 | General |
| High Knee          | — | — | — | ✓ | 1 | General |
| Knee Raise + Ab    | — | — | — | ✓ | 1 | General |

`*` = canonical-merge judgment call (see §3). Even without the starred merges,
Squat=3, Lunge=3, Push-up=3 remain solidly flagship.

**Proposed flagship set: Squat (anchor), Lunge, Push-up, Overhead Press, Row.**
Everything else → general.

## 3. Canonical exercise vocabulary — decisions needed

Overlap is a *canonical-vocabulary* problem, not string matching. Mirror the existing
fault-level `squat_canonical_mapping_v1.json` at the **exercise level** in a new
`data/kg/exercise_canonical_mapping_v1.json`. These merges are genuine judgment calls
and change the flagship count — decide explicitly rather than silently:

1. **Sumo Squat → Squat?** Wider stance, more hip-abduction/adduction emphasis; fault
   profile overlaps but not identical. (If no: Squat drops 4→3, still flagship.)
2. **Kneeling Push-up → Push-up?** Same upper-body pattern, reduced load. (Likely yes.)
3. **Merge all lunge variants** (db reverse lunge / leg lunge / reverse lunge /
   knee-lift lunge) → one `Lunge`, or split forward vs reverse? (Reverse vs forward
   shift knee/hip load meaningfully.)
4. **One-arm row (Fit3D) ≡ Barbell row (Fitness-AQA)?** Same hip-hinge + horizontal
   pull pattern, different implement/symmetry. Merge as `Row` or keep separate?
5. **Torso/ab twist** — merge Fit3D standing ab twist + EgoExo kneeling torso twist?
   Weak match; recommend keep general unless we want an anti-rotation movement.

## 4. Multi-movement KG architecture

Do **not** build siloed per-movement graphs. Use a layered schema so GraphRAG multi-hop
pays off across movements (weak hip abductor → valgus in *both* squat and lunge):

- **Movement layer (per-movement, namespaced):**
  - `Exercise` — Squat, Lunge, Push-up, Overhead Press, Row.
  - `Phase` — instances are per-movement (Squat:Descent/Bottom/Ascent ≠ Push-up phases).
    Tag with a `movement` attribute or `Movement:Phase` id.
  - `Fault` occurrence — the fault as it manifests in that movement's context.
- **Shared mechanism layer (cross-movement, single instance):**
  - `Anatomy/Muscle`, `Cause`, `Correction/Cue`, `Injury/Consequence`, `QualityDimension`.
- **Edges:** `Exercise -has_phase-> Phase`; `Exercise|Phase -exhibits-> Fault`;
  `Fault -caused_by-> Cause`; `Cause -involves-> Anatomy`; `Fault -corrected_by->
  Correction`; `Fault -risks-> Injury`.

The current `squat_kg_v2.graphml` already follows Exercise→Phase→Fault→Cause→Correction→
Injury; generalization = (a) add a `movement` tag, (b) split shared vs movement-scoped
nodes, (c) merge per-movement graphs into one `sports_kg_v3.graphml`.

## 5. Tiered RAG + KG collection plan

- **Flagship (deep coverage, ~8–15 sources each):** biomechanics, phase definitions,
  common faults, corrective cues, injury links. Build full KG subgraph via
  `src/knowledge/extract_kg.py` → canonical cleanup → merge into unified graph.
  Squat is already done and serves as the template.
- **General (light, ~2–4 sources each):** RAG docs only, plus a minimal KG stub
  (`Exercise` + top faults, no deep causal chains). Cheap breadth.

Source types: PMC/PubMed biomechanics papers, NSCA/ACSM guideline text, reputable
physio/coaching sites (already have some in `data/rag/docs`), optional coaching-video
transcripts. Reuse the offline `HashEmbeddingBackend` RAG store — no external API.

## 6. Execution sequence

1. **Lock canonical vocab + flagship set** — write `exercise_canonical_mapping_v1.json`
   after the §3 decisions. (blocks everything downstream)
2. **Generalize KG schema** — design note + migrate `squat_kg_v2` into the layered schema
   with `movement` tags; establish shared vs scoped node sets. **DONE (2026-07-06)** — see
   `docs/kg-schema-generalization.md` and §7 below.
3. **Per flagship movement** (Lunge → Push-up → Overhead Press → Row): collect RAG docs
   → `extract_kg.py` → canonical cleanup → merge into `sports_kg_v3.graphml`.
4. **General movements** — RAG docs + minimal KG stubs, batched.
5. **Make retrieval movement-aware** — scope `graph_retrieval` / `perception_to_graph`
   queries by the detected movement so flagship subgraphs are queried in isolation.

## 7. Collection log

### KG schema generalization (step 2) — DONE (2026-07-06)

Design note: `docs/kg-schema-generalization.md`. Reframed the "clean the noisy 1027-node squat
graph" problem by the one test that matters — *does this duplicate block cross-movement multi-hop?*
Scoped-layer noise (Phase/Fault/EvidenceSignal, per-movement namespace) is deferred; shared-layer
noise (Cause/Cue/Risk/QualityDimension — the multi-hop pivot) is the actual work and was cleaned.

Artifacts:
| File | What |
|---|---|
| `docs/kg-schema-generalization.md` | Design note: scoped vs shared cut, `Movement:Name` namespacing, pivot mechanism, movement-aware retrieval, deferred backlog. |
| `data/kg/shared_vocab_v1.json` | Controlled shared vocabulary + label-scoped alias→canonical map. Steers extraction and drives migration collapse. |
| `src/knowledge/kg_schema.py` | Single source of truth: `SCOPED/SHARED_LABELS`, `resolve_node_id`, vocab loaders. Shared by extract/retrieve/migrate so they never drift. |
| `scripts/knowledge/migrate_to_v3.py` | One-shot migration. |
| `data/kg/sports_kg_v3.graphml` | Migrated graph: **1027→895 nodes**, namespaced + movement-tagged. Cause 174→116, Cue 84→63, Risk 78→55, QualityDimension 184→166. |

Code wired to the schema: `extract_kg.py` (`--movement`, namespacing on write, vocab-steered
generalized prompt, targets v3), `graph_retrieval.py` (`movement` filter + `name`-attr indexing,
targets v3), `perception_to_graph.py` (passes movement through). Tests: 37 KG tests green.

Validated (synthetic Lunge injection through the real `resolve_node_id`): with a matching name,
`Lunge:Knee Valgus` and `Squat:Knee Valgus` collapse their cause onto the single shared
`Weak Hip Abductors` node; a `hops=2` query from that pivot reaches **both** movements' valgus
faults; the `movement` filter isolates each movement's scoped nodes. So the *mechanism*
(collapse + namespacing + movement filter) is proven. The remaining uncertain link — that **real**
Lunge extraction emits cause/risk names that actually land on the shared pivots — is **pending real
extraction** and is the true test of step 3: after the first Lunge extraction, spot-check its new
`Cause`/`Risk` nodes (and any `[vocab-review]` log lines) against `shared_vocab_v1.json` before
trusting the cross-movement multi-hop.

**Key findings (recorded):** `squat_canonical_mapping_v1.json` is stale — only 5/28 merge keys
exist in the current graph. The "weak hip abductor" pivot was split ~12 ways and the ACL family
~8 ways before the collapse.

**Not migrated (deliberate):** the live web app (`backend/app/config.py` `KG_GRAPH_FILE`) still
points at `squat_kg_v2.graphml` — pointing it at v3 needs the `movement` param threaded through
`backend/app/services/{analysis,knowledge,library}.py`, a response-shape/`test_backend` review, and
a check of `scripts/knowledge/{audit_kg,clean_kg}.py` (they may assume un-namespaced ids). This is
a separate follow-up, not folded into step 2 — **but it gates user-visible value**: step 3's Lunge
knowledge writes into v3 and stays invisible to the app until this cutover, so sequence it before
relying on multi-movement KG in the app. Also deferred: scoped-layer canonicalization
(55 Phase variants, 74 Faults), the shared-layer long tail, and LLM/embedding-assisted merge
(needs `OPENROUTER_API_KEY`). Full backlog in `kg-schema-generalization.md §8`.

### Lunge (flagship #2) — RAG collection DONE (2026-07-05)

8 cleanly-licensed sources added to `data/rag/docs/`, registered in
`data/paper_metadata.json` with a new `movement: "Lunge"` field (flows into chunk
metadata via `base_meta.update(doc_meta)` → enables movement-scoped retrieval), and
indexed (vector DB rebuilt → 38 sources / 2932 chunks; 306 chunks tagged Lunge).

| File | License/source | KG coverage |
|---|---|---|
| `paah_lunge_squat_loading.pdf` | OA (paahjournal) | loading positions, training-experience, ACL tension |
| `PMC6523035_patellar_tendon_lunge.txt` | OA CC BY-NC-ND | patellar tendon stress, forward step lunge |
| `PMC6980669_forward_lunge_acl.txt` | OA CC BY | ACL reconstruction / injury mechanism |
| `PMC4556293_hip_neuromuscular_valgus.txt` | OA CC BY-NC | valgus → glute-med (SHARED w/ Squat) |
| `PMC8805090_pfj_forward_side_lunge.txt` | OA | anterior knee drift → PFJ stress, step height |
| `PMC6063068_emg_lunge_variations_ABSTRACT.txt` | non-OA → abstract only | EMG hip/thigh, lunge variations |
| `PMC4641539_joint_kinetics_rehab_ABSTRACT.txt` | non-OA → abstract only | joint kinetics, rehab exercises |
| `lunge_wiki.txt` | CC BY-SA (Wikipedia) | phase/general reference |

**Method note:** NCBI/MDPI block automated PDF download here (bot stubs / Cloudflare
403). Reliable route used = EuropePMC REST `fullTextXML` for OA articles → JATS→txt
(`scratchpad/jats2txt.py`); PMC OA status checked via EuropePMC `search` API before
ingest so only redistributable (CC) full texts are stored; non-OA → abstract-only stub.

**Dropped / deferred:**
- MDPI *Reverse Lunge EMG* (`app142411480`, CC-BY) — Cloudflare-blocked, not PMC-indexed.
- ptkorea *Backward vs Forward Lunge trunk* — not PMC-indexed. Both are optional manual adds.
- Proprietary coaching blogs (Seedman / Eric Roberts / Lance Goyke / Prehab Guys) — NOT
  auto-scraped (copyright). Recommended as **manual** adds for coaching-cue coverage if
  the user has rights; academic set + wiki already cover phases/faults/corrections/injuries.

**Not yet done for Lunge:** KG extraction (`extract_kg.py`, needs Gemini key) — blocked on
step 2 (multi-movement schema generalization) so extracted nodes merge into a unified
graph rather than a squat-only one. Backfill `movement:"Squat"` on existing 28 entries
is a low-risk follow-up.
