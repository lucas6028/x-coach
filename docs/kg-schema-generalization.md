# KG Schema Generalization — Squat-only → Multi-movement

Status: **design locked, migration in progress** · Created 2026-07-06 · Step 2 of `movement-kg-expansion-plan.md`

This note pins the schema that lets a single knowledge graph hold multiple movements
(Squat, Lunge, Push-up, Overhead Press, Row) so that GraphRAG multi-hop pays off *across*
movements — e.g. a weak hip abductor explains knee valgus in **both** squat and lunge — while
each movement's phases and faults stay cleanly separable for movement-scoped retrieval.

It supersedes the sketch in `movement-kg-expansion-plan.md §4` with decisions grounded in the
actual contents of `squat_kg_v2.graphml` (1027 nodes / 1360 edges), not the idealized version.

---

## 1. The one test that decides everything: does the noise block cross-movement multi-hop?

`squat_kg_v2.graphml` is a raw LLM extraction and is heavily fragmented (55 "Phase" variants,
174 "Cause", 184 "QualityDimension", casing dupes, stray `Unknown`/`?` labels). Cleaning all
1027 nodes is a scope trap. The discriminating question for **every** node is:

> If this node is duplicated, does it break a query that spans two movements?

That splits the noise into two piles:

- **Scoped-layer noise** (Phase, Fault, EvidenceSignal) — lives inside one movement's namespace.
  `Squat:Ascent` vs `Squat:Ascent Phase` being duplicated is ugly, but Lunge nodes live under
  `Lunge:*`, so it does **not** block Lunge from merging into the graph. → **Defer** (backlog §8).
- **Shared-layer noise** (Cause, Cue, Risk, QualityDimension) — this is the **pivot layer** where
  the architecture pays off. The multi-hop `Squat:Knee Valgus → [Weak Hip Abductor] ← Lunge:Knee
  Valgus` only fires if both faults' `CAUSED_BY` edges land on the **same** Cause node. Today
  "weak hip abductor" is split across ~12 nodes (`Gluteus Medius Weakness`, `Hip Abduction
  Strength`, `Weak Hip Abductor Muscles`, `Weak Hip Abductors`, `Low Hip Abductor Torque`, …).
  If they stay split, the cross-movement query returns nothing. → **This is the work of step 2.**

**So: clean the shared layer, defer the scoped layer, and write the deferral down (don't let it be silent).**

---

## 2. Node scoping — the scoped vs shared cut

| Label | Scope | ID form | `movement` attr | Rationale |
|---|---|---|---|---|
| `Action` (Exercise) | **scoped** | `Squat` (the movement name is the id) | `Squat` | The movement anchor. |
| `Phase` | **scoped** | `Squat:Descent` | `Squat` | Squat phases ≠ Push-up phases. |
| `Fault` | **scoped** | `Squat:Knee Valgus` | `Squat` | The fault as it manifests in that movement's context. |
| `EvidenceSignal` | **scoped** | `Squat:Increased Knee Adduction Angle` | `Squat` | Movement- and view-specific perception cues. Judgment call — see §6. |
| `Cause` | **shared** | `Weak Hip Abductors` (plain id) | `shared` | **The pivot.** Anatomy/muscle/mobility deficits are movement-agnostic. |
| `Cue` (Correction) | **shared** | `Drive Knees Out` | `shared` | Coaching cues transfer across movements. |
| `Risk` (Injury) | **shared** | `ACL Injury` | `shared` | Injuries are consequences at the anatomy level, not the movement level. |
| `QualityDimension` | **shared** | `Depth` | `shared` | Abstract assessment axes (Depth, Stability, Symmetry, Alignment). |

**Anatomy/Muscle note.** Plan §4 listed `Anatomy/Muscle` as its own shared type, but the code
folds anatomy into `Cause` (`extract_kg.py` maps `anatomy`/`anatomycause` → `Cause`). We **keep
the fold**: a shared `Cause` node like `Weak Hip Abductors` already carries the anatomy, and the
Cause pivot is sufficient for the multi-hop payoff. Splitting anatomy into its own node layer is
optional future work (backlog §8).

---

## 3. Namespacing convention

Scoped nodes are keyed `"<Movement>:<Name>"` and carry `movement="<Movement>"` plus
`name="<Name>"` (the unprefixed display string). This is **required, not cosmetic**: GraphML/
NetworkX key nodes by id, and two movements will genuinely share scoped names (both squat and
lunge have a `Descent` phase and a `Knee Valgus` fault). Without the prefix they collide into one
node and the movements bleed together.

Shared nodes keep a **plain id** (`Weak Hip Abductors`) and carry `movement="shared"`. A plain id
is what makes them a single reuse point that every movement's edges converge on.

```
Squat:Knee Valgus  (Fault,  movement=Squat)  --CAUSED_BY-->  Weak Hip Abductors  (Cause, movement=shared)
Lunge:Knee Valgus  (Fault,  movement=Lunge)  --CAUSED_BY-->  Weak Hip Abductors  (Cause, movement=shared)
                                                                     ^ single node → multi-hop fires
```

Movement-aware retrieval then becomes trivial: filter seeds by the id prefix / `movement` attr.

---

## 4. Edges — unchanged

The edge vocabulary is unchanged from the squat schema; only the endpoints' ids change:

`HAS_PHASE`, `OCCURS_IN_PHASE`, `INDICATED_BY`, `CAUSED_BY`, `INCREASES_RISK_OF`,
`CORRECTED_BY`, `AFFECTS_QUALITY`, `HAS_FAULT`.

Chain: `Action --HAS_PHASE--> Phase`; `Action|Phase --HAS_FAULT/OCCURS_IN_PHASE--> Fault`;
`Fault --INDICATED_BY--> EvidenceSignal`; `Fault --CAUSED_BY--> Cause`;
`Fault --INCREASES_RISK_OF--> Risk`; `Fault --CORRECTED_BY--> Cue`; `Fault --AFFECTS_QUALITY--> QualityDimension`.

Cross-movement links are **emergent**, not new edge types: two scoped faults pointing at one
shared Cause is what connects the movements. No `SAME_AS` / `CROSS_MOVEMENT` edge is needed.

---

## 5. The shared controlled vocabulary (why extraction must be steered)

A one-time cleanup of today's shared layer is necessary but **not sufficient**. When we extract
Lunge, `extract_kg.py` will emit *fresh* shared-layer fragments (`Weak Glute Medius`,
`Hip Abductor Weakness`, …) that miss the squat nodes all over again. So the shared layer is
maintained as a **controlled vocabulary that extraction is constrained toward**:

1. `data/kg/shared_vocab_v1.json` — canonical shared node names + an alias→canonical map,
   curated for the high-value pivot clusters.
2. The extraction prompt is seeded with the canonical list ("prefer these existing Cause / Cue /
   Risk / QualityDimension names when applicable").
3. A post-extraction reconcile pass maps any new shared node whose alias is known back to canonical
   (and flags genuinely-new shared nodes for vocab review).

Establishing this **before** Lunge extraction is the whole point — otherwise fragmentation
compounds and the later reconcile is bigger. Coverage is deliberately partial: the vocab curates
the pivot families that carry cross-movement load (hip abductor / quad / hamstring / ankle mobility
/ core weakness; ACL / knee-pain / lumbar injury; knees-out / depth / neutral-spine cues; the
abstract quality dims). The long tail stays as-is — usable, just not deduped.

---

## 6. Movement-aware retrieval

`graph_retrieval.retrieve_graph_context` gains an optional `movement` filter:

- When `movement="Squat"`, seed resolution is restricted to scoped nodes of that movement
  (`Squat:*`) plus shared nodes. This keeps one movement's phases/faults from surfacing under
  another's query.
- Traversal still crosses freely into the shared layer (that's the point), and a `hops=2` query
  from a shared Cause can *intentionally* reach other movements' faults — that's the cross-movement
  payoff, exposed on demand rather than by accident.
- `perception_to_graph.retrieve_from_pose_faults` already threads an `action`/movement label; it
  passes that through as the `movement` filter.

`EvidenceSignal` scope is the one genuine judgment call. We keep it **scoped**: it doesn't affect
the payoff (sharing happens at `Cause`), and evidence signals are perception-side and view-specific,
so movement-scoping them is the honest representation.

---

## 7. Findings that shaped this (recorded so they aren't rediscovered)

- **`squat_canonical_mapping_v1.json` is stale.** Only **5 of its 28 `merge` keys** exist as node
  ids in the *current* `squat_kg_v2.graphml` — it was authored against an earlier graph and never
  applied. The migration applies the 5 valid merges and otherwise relies on `shared_vocab_v1.json`
  built from the current node set, not the stale mapping.
- **Scoped-layer fragmentation is large but low-risk** (55 Phase variants, 74 Fault). Deferred.
- **Shared-layer fragmentation is the blocker** (esp. the ~12-way split of "weak hip abductor",
  the ~8-way ACL injury family). Curated in `shared_vocab_v1.json`; applied in migration.
- **Junk nodes exist**: `Unknown`-labelled nodes and two nodes whose ids are literally edge/label
  strings (`INDICATED_BY`, `Fault`). Migration drops the label-string junk and relabels the
  salvageable `Unknown` nodes (`Ankle Mobility` → Cause, etc.).

---

## 8. Deferred backlog (explicit, not silent)

1. **Scoped-layer canonicalization** — collapse the 55 Phase variants → 5 canonical squat phases
   and dedup the 74 Faults, per an expanded canonical mapping. Degrades retrieval quality within
   squat but does not block multi-movement.
2. **Iterative shared-layer dedup** — the long tail of Cause/Cue/Risk/QualityDimension beyond the
   curated pivot clusters.
3. **Anatomy/Muscle as a first-class node layer** — currently folded into `Cause`.
4. **QualityDimension rationalization** — 184 nodes mixing true dims with metrics/kinematics;
   only the abstract dims are canonicalized in v1.
5. **LLM/embedding-assisted merge** — the current vocab is hand-curated; a semantic merge pass
   (needs `OPENROUTER_API_KEY`) would extend coverage.

---

## 9. Deliverables of this step

- **A.** This design note. ✅
- **B.** `data/kg/shared_vocab_v1.json` — controlled shared vocabulary + alias map.
- **C.** `scripts/knowledge/migrate_to_v3.py` + `data/kg/sports_kg_v3.graphml` — namespaced,
  movement-tagged, shared-layer-collapsed graph.
- **D.** Code: `extract_kg.py` (`--movement`, namespacing, vocab-steered generalized prompt,
  target v3), `graph_retrieval.py` (`movement` filter, target v3), `perception_to_graph.py`
  (movement passthrough).
