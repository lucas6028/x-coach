# Per-Movement Selection in the Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user pick Squat, Push-up, or Overhead Press in the UI and have that choice drive which rule detector runs, surviving into history, the chat coach, and the rendered verdict — with all three movements live from day one.

**Architecture:** The Python detector registry (`src/pose/movements/registry.py`) becomes the single source of truth for which movements are analyzable and whether each is validated against labeled data. A new `GET /api/movements` publishes that list; the frontend renders from it instead of a hand-maintained constant. A request-scoped `movement` string threads from the UI through `POST /api/analyze` into the existing `detect_pose_rules_from_payload(movement=)` seam, and is echoed back into the result as `result["movement"]` so a stored analysis is self-describing. Before any of that, three Overhead Press knowledge-graph queries that currently resolve to nothing are fixed.

**Tech Stack:** Python 3.11/3.12 + FastAPI + networkx + numpy (backend/ML); React 18 + Vite + TypeScript + Tailwind + vitest (frontend); Supabase/Postgres (persistence).

**Spec:** `docs/superpowers/specs/2026-07-25-movement-selection-webapp-design.md`

## Global Constraints

- **Python interpreter is always `.venv\Scripts\python.exe`** from the repo root. There is NO `python` on PATH on this machine. Never `source .venv/bin/activate`.
- **Run all Python from the repository root.** Modules import by absolute package path (`from src.pose... import ...`).
- **Backend/ML tests are always scoped to `tests/`**: `.venv\Scripts\python.exe -m pytest tests/`. Never bare `pytest`.
- **All frontend commands run with cwd = `frontend/`.** The Bash and PowerShell tools share one cwd; a stray `cd` to the repo root mass-fails vitest.
- **Backend coverage gate (CI enforces 95%):** `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
- **Frontend coverage gate:** `yarn test:coverage` from `frontend/`.
- **Canonical movement names are exactly `"Squat"`, `"Push-up"`, `"Overhead Press"`** — verbatim. These strings are simultaneously the `movement` node attribute in `sports_kg_v3`, the `movement=` scope on `/api/knowledge/*`, and the frontend's `movement.<Name>` i18n key. A casing or spelling drift breaks KG scoping and the display label silently.
- **`data/kg/sports_kg_v3.graphml` is gitignored.** Scripts that build it are checked in; the artifact is not. Any graph change must be delivered as a re-runnable script, and re-verified on every deploy target.
- **i18n parity guard:** `frontend/src/test/lib.i18n.test.ts` asserts every `en` key has a `zh` translation. Every new user-facing string needs both.
- **Tests are `unittest.TestCase` classes** under `tests/` (backend + ML) and vitest files under `frontend/src/test/`.
- **`test_concurrent_analyses_are_bounded` is a known load-dependent flake**, not a regression signal.

---

# Phase A — Knowledge-graph prerequisite

Three of the five Overhead Press `kg_query` strings resolve to nothing today, so those faults reach the coach with `likely causes: —, injury risks: —, corrective cues: —`. Fixed before any wiring.

### Task 1: Author the `Incomplete Elbow Lockout` fault node

`rule_incomplete_lockout` queries `"Incomplete Elbow Lockout"`. No node fits. Every in-graph candidate was checked and rejected: `Elbows Locked`'s only correction is `Avoid Elbow Locking` — the *opposite* fault; `Failure To Fix Barbell` and `Failure In Jerk` are Olympic-jerk scoped; `Sticking Region` describes a normal feature of every press.

The node is named `"Incomplete Elbow Lockout"` so the existing `kg_query` string needs no change.

Edges are grounded in the citation the rule **already** carries (`overhead_press.py:378-388`) — Evangelista P, Rum L, Picerno P, Biscarini A (2025), *J Funct Morphol Kinesiol*, PMC12372072, DOI 10.3390/jfmk10030322 — which establishes that elbow extensors become dominant near full extension and that the lift is complete only "when the elbow is fully extended". No new source is needed, and **no `INCREASES_RISK_OF` edge is added**: that citation does not establish an injury risk for stopping short, and inventing one would be exactly the fabrication this spec forbids.

**Files:**
- Create: `scripts/knowledge/author_ohp_lockout_v3.py`
- Create: `tests/test_kg_ohp_lockout.py`

**Interfaces:**
- Consumes: `src.knowledge.kg_schema.resolve_node_id(name, label, movement) -> tuple[str, dict]`; `src.knowledge.graph_retrieval.retrieve_graph_context(query, *, graph_file, hops, max_seeds, movement) -> dict`
- Produces: graph node `"Overhead Press:Incomplete Elbow Lockout"` (label `Fault`, movement `Overhead Press`); shared nodes `"Weak Triceps Brachii"` (`Cause`) and `"Full Elbow Extension At Lockout"` (`Cue`); a re-runnable authoring script with `--dry-run`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kg_ohp_lockout.py`:

```python
"""The authored Overhead Press lockout fault node must resolve for the detector's kg_query.

This is a data test against data/kg/sports_kg_v3.graphml, which is a gitignored build
artifact. It SKIPS when the graph is absent (a fresh clone) rather than failing, but must
pass wherever the graph exists -- including every deploy target, whose graph is built
separately.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"


@unittest.skipUnless(GRAPH_FILE.exists(), "sports_kg_v3.graphml not built in this checkout")
class TestOhpLockoutNode(unittest.TestCase):
    def _context(self, query: str) -> dict:
        from src.knowledge.graph_retrieval import retrieve_graph_context

        return retrieve_graph_context(
            query, graph_file=GRAPH_FILE, hops=1, max_seeds=3, movement="Overhead Press"
        )

    def test_incomplete_elbow_lockout_resolves(self) -> None:
        ctx = self._context("Incomplete Elbow Lockout")
        self.assertIn("Overhead Press:Incomplete Elbow Lockout", ctx["matched_nodes"])

    def test_lockout_seed_carries_coaching_content(self) -> None:
        """A node that resolves but has no causes/corrections is no better than the dashes it
        replaces -- assert the buckets the chat prompt actually renders."""
        ctx = self._context("Incomplete Elbow Lockout")
        seed = next(
            r for r in ctx["results"]
            if r["seed"]["node_id"] == "Overhead Press:Incomplete Elbow Lockout"
        )
        summary = seed["summary"]
        self.assertTrue(summary.get("causes"), "no CAUSED_BY -> Cause edges")
        self.assertTrue(summary.get("corrections"), "no CORRECTED_BY -> Cue edges")
        names = {c["name"] for c in summary["causes"]}
        self.assertIn("Weak Triceps Brachii", names)
        cues = {c["name"] for c in summary["corrections"]}
        self.assertIn("Full Elbow Extension At Lockout", cues)

    def test_no_fabricated_risk_edge(self) -> None:
        """The rule's citation establishes the mechanism, not an injury outcome. An
        INCREASES_RISK_OF edge here would be an uncited claim."""
        ctx = self._context("Incomplete Elbow Lockout")
        seed = next(
            r for r in ctx["results"]
            if r["seed"]["node_id"] == "Overhead Press:Incomplete Elbow Lockout"
        )
        self.assertFalse(seed["summary"].get("risks"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_kg_ohp_lockout.py -v`

Expected: FAIL — `test_incomplete_elbow_lockout_resolves` asserts on an empty `matched_nodes` list.

- [ ] **Step 3: Write the authoring script**

Create `scripts/knowledge/author_ohp_lockout_v3.py`:

```python
"""Author the Overhead Press 'Incomplete Elbow Lockout' fault node in sports_kg_v3.graphml.

WHY THIS EXISTS. src/pose/movements/overhead_press.py's rule_incomplete_lockout queries
"Incomplete Elbow Lockout" and the graph had no node for it, so that fault reached the chat
prompt with `likely causes: -, injury risks: -, corrective cues: -`. Every existing candidate
was checked and rejected on evidence:

  * "Elbows Locked"          -- its only correction is "Avoid Elbow Locking", the OPPOSITE
                                fault. Pointing here would tell a user who failed to finish
                                the press to lock out LESS.
  * "Failure To Fix Barbell" -- related_actions: Jerk From Chest. Olympic-jerk scoped, not
                                the strict press the detector measures.
  * "Failure In Jerk"        -- same scope problem.
  * "Sticking Region"        -- rich (degree 11) but describes a normal feature of EVERY
                                press, not a fault. Adjacent mechanism, not the same one.

CITATION. Every edge below is grounded in the citation rule_incomplete_lockout ALREADY
carries (overhead_press.py:378-388): Evangelista P, Rum L, Picerno P, Biscarini A. (2025).
Decoding the Contribution of Shoulder and Elbow Mechanics to Barbell Kinematics and the
Sticking Region in Bench and Overhead Press Exercises. J Funct Morphol Kinesiol.
PMC12372072, DOI 10.3390/jfmk10030322. It establishes that elbow extensors contribute
minimally early but become DOMINANT near full extension, and that the lift is complete only
"when the elbow is fully extended ... and the barbell reaches its final position".

NO RISK EDGE. That citation establishes a mechanism, not an injury outcome. An
INCREASES_RISK_OF edge would be an uncited claim, so none is added. test_no_fabricated_risk_edge
in tests/test_kg_ohp_lockout.py pins that absence.

IDEMPOTENT. Safe to re-run; adds only what is missing. The graph is a gitignored build
artifact, so this script is the reproducible deliverable and must be re-run on any deploy
target.

Run from repo root:  .venv\\Scripts\\python.exe scripts/knowledge/author_ohp_lockout_v3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from src.knowledge.kg_schema import resolve_node_id

GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
MOVEMENT = "Overhead Press"

FAULT_NAME = "Incomplete Elbow Lockout"

# (name, label) for nodes this script may need to create. Shared labels (Cause/Cue) get plain
# ids; scoped labels get "Movement:Name" -- resolve_node_id encodes that cut, so we never
# hand-build an id here.
NEW_CAUSE = ("Weak Triceps Brachii", "Cause")
NEW_CUE = ("Full Elbow Extension At Lockout", "Cue")

# Nodes that must ALREADY exist. If one is missing the graph is not the one this script was
# written against, and silently creating it would fabricate structure.
REQUIRED_EXISTING = [
    "Overhead Press",                              # Action anchor
    "Overhead Press:Near Lockout",                 # Phase
    "Overhead Press:Elbow Extensor Torque",        # EvidenceSignal
]


def add_edge_dedup(G: nx.MultiDiGraph, u: str, v: str, etype: str) -> bool:
    """Add a typed edge unless an identical one is already present. Mirrors the helper in
    reconcile_ohp_v3.py so repeated runs cannot multiply parallel edges."""
    if u == v:
        return False
    data = G.get_edge_data(u, v)
    if data:
        for _, ed in data.items():
            if isinstance(ed, dict) and ed.get("type") == etype:
                return False
    G.add_edge(u, v, type=etype)
    return True


def add_node_if_absent(G: nx.MultiDiGraph, name: str, label: str, log: list[str]) -> str:
    node_id, attrs = resolve_node_id(name, label, MOVEMENT)
    if node_id in G:
        log.append(f"  node exists: {node_id!r}")
    else:
        G.add_node(node_id, **attrs)
        log.append(f"  node ADDED:  {node_id!r} {attrs}")
    return node_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Author the OHP Incomplete Elbow Lockout fault node.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not GRAPH_FILE.exists():
        raise SystemExit(f"Graph not found: {GRAPH_FILE}. Build sports_kg_v3 first.")

    G = nx.read_graphml(GRAPH_FILE)
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f"Loaded {GRAPH_FILE.name}: {n0} nodes, {e0} edges")

    missing = [n for n in REQUIRED_EXISTING if n not in G]
    if missing:
        raise SystemExit(
            "Expected nodes are absent, so this is not the graph this script targets: "
            + ", ".join(repr(m) for m in missing)
        )

    log: list[str] = []
    print("\n--- nodes ---")
    fault_id = add_node_if_absent(G, FAULT_NAME, "Fault", log)
    cause_id = add_node_if_absent(G, *NEW_CAUSE, log)
    cue_id = add_node_if_absent(G, *NEW_CUE, log)
    print("\n".join(log)); log.clear()

    # Every edge below traces to the Evangelista 2025 citation in the module docstring.
    edges = [
        # The movement owns the fault -- mirrors Overhead Press:Excessive Lower Back Arching.
        ("Overhead Press", fault_id, "HAS_FAULT"),
        # The paper defines completion at full elbow extension, so the fault lives at lockout.
        (fault_id, "Overhead Press:Near Lockout", "OCCURS_IN_PHASE"),
        # Elbow extensor torque is the measured quantity the paper models.
        (fault_id, "Overhead Press:Elbow Extensor Torque", "INDICATED_BY"),
        # Elbow extensors become dominant near full extension -> insufficiency stops the rep short.
        (fault_id, cause_id, "CAUSED_BY"),
        # The corrective cue is the paper's own completion criterion.
        (fault_id, cue_id, "CORRECTED_BY"),
    ]
    print("\n--- edges ---")
    added = 0
    for u, v, etype in edges:
        if add_edge_dedup(G, u, v, etype):
            added += 1
            print(f"  edge ADDED:  {u!r} --{etype}--> {v!r}")
        else:
            print(f"  edge exists: {u!r} --{etype}--> {v!r}")

    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    print(f"\nResult: {n0}->{n1} nodes ({n1 - n0:+d}), {e0}->{e1} edges ({e1 - e0:+d})")
    print(f"  edges added this run: {added}")

    if args.dry_run:
        print("\n[dry-run] graph NOT written.")
    else:
        nx.write_graphml(G, GRAPH_FILE)
        print(f"\nWrote {GRAPH_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run the script and read the output**

Run: `.venv\Scripts\python.exe scripts/knowledge/author_ohp_lockout_v3.py --dry-run`

Expected: 3 nodes ADDED, 5 edges ADDED, `[dry-run] graph NOT written.` If any `REQUIRED_EXISTING` node is reported missing, STOP — the graph is not the expected build.

- [ ] **Step 5: Snapshot the graph before writing to it**

`data/kg/sports_kg_v3.graphml` is gitignored — a bad write cannot be recovered with `git checkout`, and rebuilding it from scratch is expensive. The repo already keeps `sports_kg_v3.post-ohp-raw.graphml` and `sports_kg_v3.post-lunge-raw.graphml` snapshots, so this follows the established habit:

```bash
cp data/kg/sports_kg_v3.graphml data/kg/sports_kg_v3.pre-lockout.graphml
```

- [ ] **Step 6: Apply it**

Run: `.venv\Scripts\python.exe scripts/knowledge/author_ohp_lockout_v3.py`

Expected: `Wrote .../sports_kg_v3.graphml`, node count `+3`, edge count `+5`.

If anything looks wrong, restore with `cp data/kg/sports_kg_v3.pre-lockout.graphml data/kg/sports_kg_v3.graphml`.

- [ ] **Step 7: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_kg_ohp_lockout.py -v`

Expected: 3 passed.

- [ ] **Step 8: Verify idempotency**

Run: `.venv\Scripts\python.exe scripts/knowledge/author_ohp_lockout_v3.py --dry-run`

Expected: every line reads `node exists:` / `edge exists:`, `edges added this run: 0`.

- [ ] **Step 9: Commit**

```bash
git add scripts/knowledge/author_ohp_lockout_v3.py tests/test_kg_ohp_lockout.py
git commit -m "feat(kg): author the Overhead Press Incomplete Elbow Lockout fault node

rule_incomplete_lockout queried a node that did not exist, so that fault
reached the chat prompt with empty causes/risks/cues. Every existing
candidate was rejected on evidence -- decisively 'Elbows Locked', whose
only correction is 'Avoid Elbow Locking', the opposite fault.

Edges are grounded in the citation the rule already carries (Evangelista
2025, PMC12372072). No INCREASES_RISK_OF edge: that source establishes a
mechanism, not an injury outcome, and test_no_fabricated_risk_edge pins
the absence."
```

---

### Task 2: Re-point the other two Overhead Press queries and pin all three detectors' retrieval

`"Lumbar Hyperextension"` and `"Asymmetric Press"` match no node. Both have a verified target, measured with `retrieve_graph_context(..., movement="Overhead Press")`:

| Old `kg_query` | New | 1-hop content that justifies it |
| --- | --- | --- |
| `"Lumbar Hyperextension"` | `"Excessive Lower Back Arching"` | corrections: `Ribcage Down`, `Gluteus Active`, `Core Active` |
| `"Asymmetric Press"` | `"Muscle Imbalance"` | risks: `Shoulder Injury`; causes: `Overuse`, `Weak External Rotators`, `Weak Abductors` |

**Files:**
- Modify: `src/pose/movements/overhead_press.py:417` (`rule_excessive_back_lean`), `src/pose/movements/overhead_press.py:475` (`rule_asymmetric_press`)
- Create: `tests/test_kg_query_resolution.py`

**Interfaces:**
- Consumes: `src.pose.movements.registry.list_detectors` does not exist yet — this test reads `kg_query` strings from the detector modules directly, so it has no dependency on Task 3.
- Produces: a corpus gate asserting every `kg_query` in every registered detector resolves.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kg_query_resolution.py`:

```python
"""Every kg-mode rule's kg_query must resolve to a node, for every movement.

This is a corpus gate, not a spot check. A kg_query that resolves to nothing is silent:
the detection still renders, but the coach gets `likely causes: -, injury risks: -,
corrective cues: -` for it. Squat never exposed this because all four of its strings
resolve; Overhead Press shipped with three of five broken.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
MOVEMENTS_DIR = REPO_ROOT / "src" / "pose" / "movements"

# module filename -> canonical movement name used for KG scoping.
MODULE_MOVEMENTS = {
    "squat.py": "Squat",
    "pushup.py": "Push-up",
    "overhead_press.py": "Overhead Press",
}


def _kg_queries(module_path: Path) -> list[str]:
    """Every literal kg_query= value in a detector module, paired with the retrieval_mode of
    the same build_detection call. Parsed from the AST rather than imported, so this gate does
    not depend on numpy/detector import side effects."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        query = kwargs.get("kg_query")
        mode = kwargs.get("retrieval_mode")
        if not isinstance(query, ast.Constant) or not isinstance(query.value, str):
            continue
        if isinstance(mode, ast.Constant) and mode.value != "kg":
            continue  # rag-mode rules query the vector DB, not the graph
        found.append(query.value)
    return found


class TestKgQueryCorpus(unittest.TestCase):
    def test_every_module_is_covered(self) -> None:
        """Guard against a new detector module being added and silently skipped by this gate."""
        present = {p.name for p in MOVEMENTS_DIR.glob("*.py")} - {
            "__init__.py", "base.py", "registry.py"
        }
        self.assertEqual(present, set(MODULE_MOVEMENTS))

    def test_queries_were_actually_found(self) -> None:
        """A parser that silently returns [] would make the resolution test vacuously pass."""
        for filename in MODULE_MOVEMENTS:
            with self.subTest(module=filename):
                self.assertGreater(len(_kg_queries(MOVEMENTS_DIR / filename)), 0)

    @unittest.skipUnless(GRAPH_FILE.exists(), "sports_kg_v3.graphml not built in this checkout")
    def test_every_kg_query_resolves(self) -> None:
        from src.knowledge.graph_retrieval import retrieve_graph_context

        unresolved: list[str] = []
        for filename, movement in MODULE_MOVEMENTS.items():
            for query in _kg_queries(MOVEMENTS_DIR / filename):
                ctx = retrieve_graph_context(
                    query, graph_file=GRAPH_FILE, hops=1, max_seeds=3, movement=movement
                )
                if not ctx["matched_nodes"]:
                    unresolved.append(f"{movement}: {query!r}")
        self.assertEqual(unresolved, [], f"kg_query strings resolving to nothing: {unresolved}")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_kg_query_resolution.py -v`

Expected: `test_every_kg_query_resolves` FAILS listing exactly two entries — `Overhead Press: 'Lumbar Hyperextension'` and `Overhead Press: 'Asymmetric Press'`. (`Incomplete Elbow Lockout` already passes from Task 1.) The other two tests pass.

- [ ] **Step 3: Re-point the two strings**

In `src/pose/movements/overhead_press.py`, inside `rule_excessive_back_lean`'s `build_detection` call, replace:

```python
                kg_query="Lumbar Hyperextension",
```

with:

```python
                # Verified to resolve: retrieve_graph_context(..., movement="Overhead Press")
                # returns "Overhead Press:Excessive Lower Back Arching", whose CORRECTED_BY
                # edges are exactly the cues this fault needs -- Ribcage Down, Gluteus Active,
                # Core Active. The rule's own fault name says "rib flare"; the node's first
                # correction is "Ribcage Down". "Lumbar Hyperextension" resolved to NOTHING.
                kg_query="Excessive Lower Back Arching",
```

In the same file, inside `rule_asymmetric_press`'s `build_detection` call, replace:

```python
                kg_query="Asymmetric Press",
```

with:

```python
                # Verified to resolve: returns TWO seeds -- a movement-generic "Muscle Imbalance"
                # Cause node (1 edge, no summary buckets) and "Overhead Press:Muscle Imbalance"
                # (Fault, 5 edges), which carries the content: risks Shoulder Injury; causes
                # Overuse, Weak External Rotators, Weak Abductors. Both are returned; the scoped
                # one supplies the coaching material. "Asymmetric Press" is not a node at all.
                kg_query="Muscle Imbalance",
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_kg_query_resolution.py -v`

Expected: 3 passed.

- [ ] **Step 5: Run the full ML suite for regressions**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v -k "ohp or overhead or pose_rule"`

Expected: all pass. Existing OHP tests assert on `fault_id`/severity/evidence, not `kg_query`; if one does assert the old string, update it to the new value and note why in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/pose/movements/overhead_press.py tests/test_kg_query_resolution.py
git commit -m "fix(pose): re-point the two unresolved OHP kg_query strings

'Lumbar Hyperextension' and 'Asymmetric Press' matched no node, so those
faults reached the coach with empty causes/risks/cues. Both targets were
measured against the graph rather than picked by name similarity.

Adds a corpus gate over every kg-mode rule in every detector module, so a
future query that resolves to nothing fails a test instead of silently
degrading the explanation. The gate asserts the parser actually found
queries, so it cannot pass vacuously."
```

---

# Phase B — ML layer

### Task 3: `MovementDetector.validated` and `registry.list_detectors()`

Only Squat is validated against labeled data. `validated` defaults to `False` so a future fourth detector fails toward Beta rather than silently presenting as validated.

**Files:**
- Modify: `src/pose/movements/base.py:37-43`
- Modify: `src/pose/movements/squat.py:302-307`
- Modify: `src/pose/movements/registry.py`
- Create: `tests/test_movement_registry.py`

**Interfaces:**
- Produces: `MovementDetector.validated: bool` (default `False`); `registry.list_detectors() -> list[MovementDetector]` in registration order.

- [ ] **Step 1: Write the failing test**

Create `tests/test_movement_registry.py`:

```python
from __future__ import annotations

import unittest


class TestMovementRegistry(unittest.TestCase):
    def test_lists_all_three_detectors_in_registration_order(self) -> None:
        from src.pose.movements import registry

        names = [d.name for d in registry.list_detectors()]
        self.assertEqual(names, ["Squat", "Overhead Press", "Push-up"])

    def test_only_squat_is_validated(self) -> None:
        """Push-up and Overhead Press rules are literature-derived and never checked against
        ground-truth labels. The UI marks them Beta off this flag."""
        from src.pose.movements import registry

        validated = {d.name: d.validated for d in registry.list_detectors()}
        self.assertEqual(validated, {"Squat": True, "Overhead Press": False, "Push-up": False})

    def test_validated_defaults_to_false(self) -> None:
        """A new detector must fail toward Beta, never silently present as validated."""
        from src.pose.movements.base import MovementDetector

        detector = MovementDetector("Test", (), lambda frames, fps: [], lambda raw: [], ())
        self.assertFalse(detector.validated)

    def test_names_are_the_canonical_spellings(self) -> None:
        """These strings are simultaneously the KG `movement` scope and the frontend's
        movement.<Name> i18n key. A drift breaks both silently."""
        from src.pose.movements import registry

        self.assertEqual(
            {d.name for d in registry.list_detectors()},
            {"Squat", "Push-up", "Overhead Press"},
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -v`

Expected: FAIL with `AttributeError: module 'src.pose.movements.registry' has no attribute 'list_detectors'`.

- [ ] **Step 3: Add the field**

In `src/pose/movements/base.py`, extend the dataclass. `validated` goes **last** — the dataclass is frozen and all three call sites construct it positionally, so any other position breaks them:

```python
@dataclass(frozen=True)
class MovementDetector:
    name: str
    metric_keys: tuple[str, ...]
    compute_raw: Callable[[Sequence[object], float], list[dict]]
    assign_phases: Callable[[list[dict]], list[str]]
    rules: tuple[RuleFn, ...]
    # Whether this detector's rules have been checked against labeled ground truth. Defaults to
    # False so a newly registered detector surfaces as Beta in the UI rather than silently
    # presenting as validated; Squat opts in explicitly.
    validated: bool = False
```

- [ ] **Step 4: Opt Squat in**

In `src/pose/movements/squat.py`, change the constructor call:

```python
SQUAT_DETECTOR = MovementDetector(
    "Squat",
    METRIC_KEYS,
    compute_raw,
    assign_phases,
    (rule_knees_inward, rule_knees_forward, rule_shallow_depth, rule_forward_lean, rule_heel_rise),
    validated=True,
)
```

Leave `pushup.py` and `overhead_press.py` untouched — they take the default.

- [ ] **Step 5: Add the listing function**

In `src/pose/movements/registry.py`, add above the trailing imports:

```python
def list_detectors() -> list[MovementDetector]:
    """Every registered detector, in registration order.

    Registration order is the import order at the bottom of this module (Squat, Overhead Press,
    Push-up) -- deterministic, and it puts the validated detector first without encoding a UI
    preference in the ML layer. Backs GET /api/movements, which is why the frontend needs no
    hand-maintained list of analyzable movements.
    """
    return list(_REGISTRY.values())
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movement_registry.py -v`

Expected: 4 passed.

- [ ] **Step 7: Run the full ML suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`

Expected: all pass except the known `test_concurrent_analyses_are_bounded` flake.

- [ ] **Step 8: Commit**

```bash
git add src/pose/movements/base.py src/pose/movements/squat.py src/pose/movements/registry.py tests/test_movement_registry.py
git commit -m "feat(pose): add MovementDetector.validated and registry.list_detectors()

Makes the Python registry the single source of truth for which movements
are analyzable and which are validated, so GET /api/movements can derive
both and the frontend needs no hand-maintained constant.

validated defaults to False and Squat opts in, so a future detector fails
toward Beta rather than silently presenting as validated."
```

---

### Task 4: Echo the resolved movement into the analysis result

`result["movement"]` is set from `detector.name`, not the caller's string, so `--movement push-up` normalises to `"Push-up"`. Setting it here (rather than in the backend service) means the CLI's written JSON gains it for free and a stored analysis is self-describing.

**Files:**
- Modify: `src/pose/pose_rule_detector.py:600-616`
- Create: `tests/test_pose_rule_movement_echo.py`

**Interfaces:**
- Consumes: `registry.get_detector(movement)` (existing).
- Produces: `result["movement"]: str` — the canonical detector name — in every payload from `detect_pose_rules_from_payload` and therefore `detect_pose_rules_from_json`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pose_rule_movement_echo.py`:

```python
from __future__ import annotations

import unittest


def _empty_payload() -> dict:
    """A structurally valid payload with no frames -- enough to exercise routing without
    depending on any fixture video."""
    return {"metadata": {"fps": 30.0}, "frames": []}


class TestMovementEcho(unittest.TestCase):
    def _detect(self, movement: str | None) -> dict:
        from src.pose.pose_rule_detector import detect_pose_rules_from_payload

        return detect_pose_rules_from_payload(_empty_payload(), movement=movement)

    def test_echoes_each_registered_movement(self) -> None:
        for movement in ("Squat", "Push-up", "Overhead Press"):
            with self.subTest(movement=movement):
                self.assertEqual(self._detect(movement)["movement"], movement)

    def test_normalises_caller_casing_to_the_canonical_name(self) -> None:
        """get_detector lowercases its lookup key but detector.name keeps its case. Echoing
        detector.name (not the caller's string) is what keeps the KG scope and the frontend
        i18n key correct for '--movement push-up'."""
        self.assertEqual(self._detect("push-up")["movement"], "Push-up")
        self.assertEqual(self._detect("OVERHEAD PRESS")["movement"], "Overhead Press")

    def test_defaults_to_squat_when_movement_is_none(self) -> None:
        self.assertEqual(self._detect(None)["movement"], "Squat")

    def test_unknown_movement_still_raises(self) -> None:
        with self.assertRaises(KeyError):
            self._detect("Cartwheel")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_movement_echo.py -v`

Expected: FAIL with `KeyError: 'movement'` on the first three tests.

- [ ] **Step 3: Add the echo**

In `src/pose/pose_rule_detector.py`, in `detect_pose_rules_from_payload`, add `movement` to the result dict immediately after `video_id`:

```python
    valid_frames = [c for c in core if c.valid]
    result = {
        "video_id": video_id or (pose_json_path.stem if pose_json_path else ""),
        # The CANONICAL movement name, taken from the resolved detector rather than the caller's
        # string, so "push-up" normalises to "Push-up". That exact spelling is simultaneously the
        # KG `movement` scope and the frontend's movement.<Name> i18n key. Echoing it here (not in
        # the web layer) means the CLI's written JSON carries it too, and a stored analysis records
        # which rules produced it -- permanently, without depending on a database column.
        "movement": detector.name,
        "pose_json_path": str(pose_json_path) if pose_json_path else "",
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_pose_rule_movement_echo.py -v`

Expected: 4 passed.

- [ ] **Step 5: Run the full backend + ML suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`

Expected: all pass except the known flake. If a golden/oracle comparison test fails on the new key, it compares whole dicts — update its expected payload and say so in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/pose/pose_rule_detector.py tests/test_pose_rule_movement_echo.py
git commit -m "feat(pose): echo the canonical movement name into the analysis result

result['movement'] comes from detector.name, not the caller's string, so
'push-up' normalises to 'Push-up' -- the spelling the KG scope and the
frontend i18n key both depend on.

Set in detect_pose_rules_from_payload rather than the web layer so the CLI
output carries it too and a stored analysis records which rules produced
it without depending on a database column."
```

---

# Phase C — Backend

### Task 5: `GET /api/movements`

**Files:**
- Create: `backend/app/routers/movements.py`
- Modify: `backend/app/main.py`
- Create: `tests/test_movements_endpoint.py`

**Interfaces:**
- Consumes: `registry.list_detectors()` from Task 3.
- Produces: `GET /api/movements` → `{"movements": [{"name": str, "validated": bool}, ...]}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_movements_endpoint.py`:

```python
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import app


class TestMovementsEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_lists_the_registered_movements(self) -> None:
        resp = self.client.get("/api/movements")
        self.assertEqual(resp.status_code, 200)
        names = [m["name"] for m in resp.json()["movements"]]
        self.assertEqual(names, ["Squat", "Overhead Press", "Push-up"])

    def test_reports_validation_status(self) -> None:
        resp = self.client.get("/api/movements")
        flags = {m["name"]: m["validated"] for m in resp.json()["movements"]}
        self.assertEqual(flags, {"Squat": True, "Overhead Press": False, "Push-up": False})

    def test_is_public(self) -> None:
        """/app is the anonymous public demo and needs this list to render its selector and
        validate ?movement= before enabling the dropzone, so no auth header is required."""
        self.assertEqual(self.client.get("/api/movements").status_code, 200)

    def test_derives_from_the_registry_not_a_literal(self) -> None:
        """If someone replaces the body with a hardcoded list, registering a fourth detector
        stops surfacing it -- the exact drift this endpoint exists to prevent."""
        from src.pose.movements import registry

        resp = self.client.get("/api/movements")
        self.assertEqual(
            [m["name"] for m in resp.json()["movements"]],
            [d.name for d in registry.list_detectors()],
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movements_endpoint.py -v`

Expected: FAIL — all four get `404`.

- [ ] **Step 3: Write the router**

Create `backend/app/routers/movements.py`:

```python
"""The analyzable-movement catalog, derived from the pose detector registry.

Public, like /api/knowledge/*. The /movements page itself is behind RequireAuth, but /app is
the anonymous public demo and needs this list to render its movement selector and to validate
a ?movement= URL parameter BEFORE enabling the dropzone -- otherwise a hand-typed movement
costs the user a full upload to discover a 400.

The list is derived from src/pose/movements/registry.py rather than restated here, so
registering a fourth detector surfaces it in the UI with no backend or frontend edit.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["movements"])


@router.get("/movements")
def movements() -> dict:
    """Every registered detector: canonical name plus whether its rules are validated against
    labeled data. The frontend renders unvalidated movements with a Beta tag."""
    # Imported lazily: the registry pulls in the detector modules (numpy), and the API layer is
    # tested without the heavy ML stack installed. Matches services/analysis.py's deferred-import
    # rationale.
    from src.pose.movements import registry

    return {
        "movements": [
            {"name": d.name, "validated": d.validated} for d in registry.list_detectors()
        ]
    }
```

- [ ] **Step 4: Register the router**

In `backend/app/main.py`, add `movements` to the alphabetised router import block at line 15-25:

```python
from backend.app.routers import (
    admin,
    analyses,
    analyze,
    auth_line,
    chat,
    conversations,
    knowledge,
    line_webhook,
    movements,
    videos,
)
```

and register it with the other routers (after line 45, `app.include_router(knowledge.router)`):

```python
app.include_router(movements.router)
```

While in this file, correct the now-false app description (line 30):

```python
    description="Explainable movement coaching: pose perception + biomechanics rules + KG/RAG retrieval over a 16-movement graph (video analysis covers Squat, Push-up and Overhead Press).",
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_movements_endpoint.py -v`

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/movements.py backend/app/main.py tests/test_movements_endpoint.py
git commit -m "feat(api): add GET /api/movements derived from the detector registry

Replaces the frontend's hand-maintained ANALYZABLE_MOVEMENTS constant as
the source of truth for which movements can be analysed and which are
validated. Public, because /app is the anonymous demo and needs the list
to validate ?movement= before enabling the dropzone.

A test asserts the payload tracks registry.list_detectors() rather than a
literal, so the drift this endpoint prevents cannot creep back in."
```

---

### Task 6: Accept and validate `movement` on `POST /api/analyze`

Validation runs **before** `save_upload` and before pose extraction, so a bad request costs no compute.

**Files:**
- Modify: `backend/app/routers/analyze.py`
- Modify: `backend/app/services/analysis.py:66-96`
- Modify: `backend/app/config.py:28-29`
- Create: `tests/test_analyze_movement.py`

**Interfaces:**
- Consumes: `registry.list_detectors()` (Task 3); `result["movement"]` (Task 4).
- Produces: `analysis.analyze_video_file(source_path, *, video_id=None, movement=None) -> dict`; `POST /api/analyze` multipart field `movement`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_movement.py`:

```python
from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app


def _stub_result(movement: str) -> dict:
    """A response-shaped stub. The route returns the analysis verbatim, so keep the real keys --
    a thin dict would pass today but hide a shape regression if the route ever post-processes."""
    return {
        "video_id": "vid1",
        "movement": movement,
        "metadata": {"fps": 30.0},
        "view": {"view_type": "side", "view_confidence": 0.8},
        "quality": {"total_frames": 10, "valid_frames": 9, "valid_frame_ratio": 0.9},
        "detections": [],
        "retrievals": [],
        "pose": {"fps": 30.0, "width": 640, "height": 480, "frames": []},
        "source": "upload",
    }


class TestAnalyzeMovement(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _post(self, movement: str | None):
        data = {"movement": movement} if movement is not None else None
        return self.client.post(
            "/api/analyze",
            files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
            data=data,
        )

    def test_rejects_an_unregistered_movement(self) -> None:
        resp = self._post("Cartwheel")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cartwheel", resp.json()["detail"])

    def test_rejects_before_running_the_pipeline(self) -> None:
        """A bad movement must not cost a full MediaPipe pass -- neither the upload write nor
        the analysis may be reached."""
        with patch("backend.app.services.analysis.save_upload") as save, patch(
            "backend.app.services.analysis.analyze_video_file"
        ) as run:
            self._post("Cartwheel")
        save.assert_not_called()
        run.assert_not_called()

    def test_forwards_a_valid_movement_to_the_pipeline(self) -> None:
        with patch(
            "backend.app.services.analysis.save_upload", return_value=("vid1", "/tmp/vid1.mp4")
        ), patch(
            "backend.app.services.analysis.analyze_video_file",
            return_value=_stub_result("Push-up"),
        ) as run:
            resp = self._post("Push-up")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(run.call_args.kwargs["movement"], "Push-up")

    def test_defaults_to_squat_when_the_field_is_omitted(self) -> None:
        """Backward compatible: a caller that has not been updated still works."""
        with patch(
            "backend.app.services.analysis.save_upload", return_value=("vid1", "/tmp/vid1.mp4")
        ), patch(
            "backend.app.services.analysis.analyze_video_file",
            return_value=_stub_result("Squat"),
        ) as run:
            resp = self._post(None)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(run.call_args.kwargs["movement"], "Squat")

    def test_accepts_any_registered_movement(self) -> None:
        for movement in ("Squat", "Push-up", "Overhead Press"):
            with self.subTest(movement=movement):
                with patch(
                    "backend.app.services.analysis.save_upload",
                    return_value=("vid1", "/tmp/vid1.mp4"),
                ), patch(
                    "backend.app.services.analysis.analyze_video_file",
                    return_value=_stub_result(movement),
                ):
                    self.assertEqual(self._post(movement).status_code, 200)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_movement.py -v`

Expected: FAIL — the unknown movement is accepted (no 400) and `analyze_video_file` is called without a `movement` kwarg.

- [ ] **Step 3: Thread the movement through the service**

In `backend/app/services/analysis.py`, change the signature and the detector call:

```python
def analyze_video_file(
    source_path: Path, *, video_id: str | None = None, movement: str | None = None
) -> dict[str, Any]:
```

and inside it:

```python
    result = detect_pose_rules_from_json(
        pose_json_path,
        video_id=vid,
        include_retrieval=True,
        graph_file=config.KG_GRAPH_FILE,
        rag_db_dir=config.RAG_DB_DIR,
        movement=movement or config.DEFAULT_ANALYSIS_MOVEMENT,
    )
```

- [ ] **Step 4: Add validation and the form field to the router**

In `backend/app/routers/analyze.py`, add `Form` to the fastapi import, then add a module-level helper below `_ANALYSIS_SEMAPHORE`:

```python
def _validated_movement(movement: str) -> str:
    """Resolve a requested movement to its canonical name, or 400.

    Rejecting HERE -- before save_upload and before pose extraction -- means a bad request
    costs no compute. The registry lookup is case-insensitive (get_detector lowercases its
    key), so the canonical spelling is what comes back.
    """
    from src.pose.movements import registry

    try:
        return registry.get_detector(movement).name
    except KeyError:
        known = ", ".join(d.name for d in registry.list_detectors())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported movement '{movement}'. Analyzable movements: {known}.",
        ) from None
```

Add the parameter to the endpoint signature, after `file`:

```python
async def analyze(
    file: UploadFile = File(...),
    movement: str = Form(config.DEFAULT_ANALYSIS_MOVEMENT),
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict:
```

Immediately after the suffix check and before `data = await file.read()`, validate:

```python
    canonical_movement = await run_in_threadpool(_validated_movement, movement)
```

Then pass it to the analysis call:

```python
            result = await run_in_threadpool(
                analysis.analyze_video_file,
                saved_path,
                video_id=video_id,
                movement=canonical_movement,
            )
```

Finally correct the docstring's first line, which currently says "Accept a squat video":

```python
    """Accept a video of a supported movement, extract pose, detect faults, and return the analysis.
```

- [ ] **Step 5: Update the config comment**

In `backend/app/config.py`, replace the stale comment above `DEFAULT_ANALYSIS_MOVEMENT`:

```python
# The FALLBACK movement, not a pin: it is what an omitted `movement` form field and the
# pre-processed demo library (services/library.py) resolve to. Live analysis is chosen per
# request from the detector registry -- see GET /api/movements.
DEFAULT_ANALYSIS_MOVEMENT = "Squat"
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_analyze_movement.py -v`

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/routers/analyze.py backend/app/services/analysis.py backend/app/config.py tests/test_analyze_movement.py
git commit -m "feat(api): accept a per-request movement on POST /api/analyze

Validation runs before save_upload and before pose extraction, so an
unknown movement costs no MediaPipe pass -- pinned by a test that asserts
neither is reached. An omitted field still defaults to Squat, so callers
that have not been updated keep working.

DEFAULT_ANALYSIS_MOVEMENT is now the fallback (library path, omitted
field) rather than a pin; its comment says so."
```

---

### Task 7: Persist the movement and expose it on the history list

**Files:**
- Create: `db/migrations/20260725000000_analysis_movement.sql`
- Modify: `backend/app/services/store.py:31-36`, `:159-171`, `:186`
- Create: `tests/test_store_movement.py`

**Interfaces:**
- Consumes: `result["movement"]` (Task 4).
- Produces: `_summarize(result) -> tuple[str | None, int, str | None, str | None]` (view_type, fault_count, pipeline_version, **movement**); `movement` in the `analyses` insert and in the history list select.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store_movement.py`:

```python
from __future__ import annotations

import unittest


class TestSummarizeMovement(unittest.TestCase):
    def test_promotes_the_movement_out_of_the_result(self) -> None:
        from backend.app.services.store import _summarize

        result = {
            "view": {"view_type": "side"},
            "detections": [{"fault_id": "a"}, {"fault_id": "b"}],
            "pipeline_version": "v1",
            "movement": "Push-up",
        }
        view_type, fault_count, pipeline_version, movement = _summarize(result)
        self.assertEqual(view_type, "side")
        self.assertEqual(fault_count, 2)
        self.assertEqual(pipeline_version, "v1")
        self.assertEqual(movement, "Push-up")

    def test_movement_is_none_when_absent(self) -> None:
        """Analyses produced before the echo landed have no movement; the column is nullable
        and the frontend falls back rather than inventing 'Squat' at the storage layer."""
        from backend.app.services.store import _summarize

        self.assertIsNone(_summarize({"view": {}, "detections": []})[3])

    def test_history_select_includes_movement(self) -> None:
        """The history badge reads the promoted column; if the select drops it, every row
        renders the fallback and the badge silently lies."""
        import inspect

        from backend.app.services import store

        source = inspect.getsource(store.list_analyses)
        self.assertIn("movement", source)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_store_movement.py -v`

Expected: FAIL — `_summarize` returns a 3-tuple, so unpacking into four names raises `ValueError`.

- [ ] **Step 3: Write the migration**

Create `db/migrations/20260725000000_analysis_movement.sql`:

```sql
-- Record which movement's rules produced each analysis.
--
-- Additive and nullable on purpose: existing rows keep working (they predate per-movement
-- analysis, when everything was Squat), RLS policies are unaffected, and the
-- admin_user_overview view needs no change.
--
-- The analysis document in `result` also carries `movement` for anything analysed after this
-- lands, so the frontend can fall back to it for rows where this column is null.

ALTER TABLE analyses ADD COLUMN movement text;
```

- [ ] **Step 4: Extend `_summarize` and its callers**

In `backend/app/services/store.py`:

```python
def _summarize(result: dict[str, Any]) -> tuple[str | None, int, str | None, str | None]:
    """Promote the list-view columns out of the nested analysis document."""
    view_type = (result.get("view") or {}).get("view_type")
    fault_count = len(result.get("detections") or [])
    pipeline_version = result.get("pipeline_version")
    # Which detector produced this. Null for rows predating per-movement analysis; the frontend
    # falls back to result["movement"], then to Squat, rather than guessing here.
    movement = result.get("movement")
    return view_type, fault_count, pipeline_version, movement
```

Update the unpack at the insert (line ~159) and add the column:

```python
    view_type, fault_count, pipeline_version, movement = _summarize(result)
    resp = (
        client.table("analyses")
        .insert(
            {
                "user_id": user_id,
                "video_id": video_id,
                "source": source,
                "view_type": view_type,
                "fault_count": fault_count,
                "pipeline_version": pipeline_version,
                "movement": movement,
                "result": result,
            }
```

Add `movement` to the history list select (line ~186):

```python
        .select(
            "id, video_id, source, view_type, fault_count, movement, created_at",
            count="exact",
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_store_movement.py -v`

Expected: 3 passed.

- [ ] **Step 6: Run the whole backend suite**

Run: `.venv\Scripts\python.exe -m pytest tests/ -v`

Expected: all pass except the known flake. Any other test that unpacks `_summarize` needs the fourth name.

- [ ] **Step 7: Commit**

```bash
git add db/migrations/20260725000000_analysis_movement.sql backend/app/services/store.py tests/test_store_movement.py
git commit -m "feat(store): record which movement produced each analysis

Extends _summarize rather than reading result['movement'] inline at the
insert, so every derived column stays on one seam. Column is nullable and
additive: existing rows keep working and admin_user_overview needs no
change.

MANUAL STEP: apply db/migrations/20260725000000_analysis_movement.sql to
Supabase before deploying."
```

- [ ] **Step 8: Tell the user about the manual migration**

This migration is not applied automatically. Report to the user that `db/migrations/20260725000000_analysis_movement.sql` must be run against Supabase before deploy, and that until then it is not merely a degraded fallback: `store.py` writes the `movement` column unconditionally, so against an unmigrated database `POST /api/analyze` silently loses every persisted row (postgrest raises, `analyze.py`'s broad exception guard swallows it, the response is still 200 with `analysis_id: null` and only a log line survives), and `GET /api/analyses` hard-500s for every signed-in user (no exception handling around that `select`).

---

### Task 8: Make the chat coach movement-aware

`services/chat.py` currently hardcodes "the x-coach **squat** coach" and "about THIS **squat**". Without this, chatting about a push-up analysis produces squat advice.

This task also implements the spec's §9 mitigation: the CLEAN REP branch names the movement, so a clean verdict is scoped to the movement the user asserted rather than stated bare.

**Files:**
- Modify: `backend/app/services/chat.py:50-75`, `:109-170`
- Modify: `backend/app/routers/chat.py:45-54`
- Modify: `tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `ChatContext.movement` from the client (Task 9).
- Produces: `ChatContext.movement: str | None`; `_build_system_prompt(context)` renders the movement name in the preamble, the fault framing, and the clean-rep instruction.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat_endpoint.py` (keep the existing imports and class style used in that file):

```python
class TestChatPromptMovement(unittest.TestCase):
    def _prompt(self, **context) -> str:
        from backend.app.services.chat import _build_system_prompt

        base = {"quality": {"valid_frame_ratio": 0.9}, "faults": [], "fault_count": 0}
        base.update(context)
        return _build_system_prompt(base)

    def test_preamble_names_the_movement(self) -> None:
        prompt = self._prompt(movement="Push-up")
        self.assertIn("Push-up coach", prompt)
        self.assertNotIn("squat coach", prompt.lower())

    def test_clean_rep_branch_names_the_movement(self) -> None:
        """The spec's section 9 mitigation: a measurable clip measured by the WRONG rules is
        now reachable, so the clean verdict must be scoped to the movement the user asserted
        rather than stated bare."""
        prompt = self._prompt(movement="Overhead Press")
        self.assertIn("CLEAN Overhead Press REP", prompt)

    def test_unmeasured_branch_is_unchanged_by_movement(self) -> None:
        """An unmeasured clip must still refuse to congratulate, whatever the movement."""
        prompt = self._prompt(movement="Push-up", quality={"valid_frame_ratio": 0.0})
        self.assertIn("NOT MEASURED", prompt)
        self.assertNotIn("CLEAN", prompt)

    def test_defaults_to_squat_for_an_older_client(self) -> None:
        """A client that predates ChatContext.movement must still get a coherent prompt."""
        self.assertIn("Squat coach", self._prompt())

    def test_followup_instruction_names_the_movement(self) -> None:
        from backend.app.services.chat import _followup_instruction

        self.assertIn("THIS Push-up", _followup_instruction("Push-up"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v -k Movement`

Expected: FAIL — the preamble is a constant naming squat, and `_followup_instruction` does not exist.

- [ ] **Step 3: Parameterise the preamble and follow-up instruction**

In `backend/app/services/chat.py`, replace the `_SYSTEM_PREAMBLE` constant with a function, keeping the grounding rules verbatim:

```python
def _system_preamble(movement: str) -> str:
    """The grounded preamble, scoped to the movement whose rules actually ran.

    The movement is named rather than assumed because it is now USER-ASSERTED input: the studio
    lets the user pick, so a clip can be measured by rules that do not describe it (spec section
    9). Naming it makes every claim true relative to the assertion the user made, and puts that
    assertion in front of the model instead of leaving it implicit.
    """
    return (
        f"You are the x-coach {movement} coach. You explain an ALREADY-COMPUTED analysis of one "
        f"{movement} repetition and answer the user's follow-up questions about it.\n\n"
        "GROUNDING RULES — these are absolute:\n"
        "- Speak ONLY from the analysis facts given below. They are the single source of truth.\n"
        "- Do NOT invent faults, causes, injury risks, corrective cues, measurements, or camera "
        "views that are not listed. If the user asks about something not in the analysis, say it "
        "was not detected or not measured in this rep — never fabricate it.\n"
        "- Base any corrective advice on the retrieved corrections/cues for the detected faults.\n"
        "- Be concise, specific, and encouraging. Reference the timecodes and phases when useful.\n"
        "- Reply in the same language the user writes in.\n"
        "- You may use light Markdown for readability — bold for key cues, short bulleted lists, "
        "and inline code for measurements/timecodes. Formatting never loosens the grounding rules "
        "above.\n"
    )


def _followup_instruction(movement: str) -> str:
    """Appended to the grounded system prompt for the follow-up call. The full analysis grounding
    precedes it, so a suggested question can never reference a fault, cue, or measurement outside
    the analysis — the same honesty bar the answer holds."""
    return (
        "FOLLOW-UP TASK: The user has just read your answer. Propose EXACTLY TWO short follow-up "
        f"questions the user might naturally ask you next about THIS {movement}. Each is from the "
        "user's point of view (addressed to you, the coach), grounded ONLY in the analysis facts "
        "above (never a fault/cue/measurement not listed), at most ~12 words, in the user's "
        'language. Output ONLY a compact JSON array of exactly two strings and nothing else — '
        'e.g. ["...", "..."].'
    )
```

- [ ] **Step 4: Use them in `_build_system_prompt`**

At the top of `_build_system_prompt`, resolve the movement and build the opening lines from it:

```python
    # Falls back to the pipeline default so a client predating ChatContext.movement still gets a
    # coherent prompt rather than an empty movement name.
    movement = str(context.get("movement") or config.DEFAULT_ANALYSIS_MOVEMENT)
    lines: list[str] = [_system_preamble(movement), "ANALYSIS FACTS:"]
```

`chat.py` does **not** currently import config — add it to the module imports, below `from typing import Any`:

```python
from backend.app import config
```

Change the clean-rep branch to name the movement (leave the NOT MEASURED branch's text alone):

```python
    elif not faults:
        lines.append(
            f"- This is a CLEAN {movement} REP: no {movement} faults were detected. "
            "Congratulate the user and reinforce what good form looks like; do not manufacture "
            "problems."
        )
```

Then update `suggest_followups` to call `_followup_instruction(movement)` where it previously used the `_FOLLOWUP_INSTRUCTION` constant, resolving `movement` from its `context` argument the same way.

- [ ] **Step 5: Add the field to the request model**

In `backend/app/routers/chat.py`, add to `ChatContext`:

```python
class ChatContext(BaseModel):
    """Compact grounding blob built by ``buildChatContext(analysis)`` on the client."""

    video_id: str | None = None
    # Which detector produced this analysis. Optional so a client predating per-movement
    # analysis still validates; _build_system_prompt falls back to the pipeline default.
    movement: str | None = None
    view_type: str | None = None
    view_confidence: float | None = None
    fault_count: int = 0
    quality: dict[str, Any] = Field(default_factory=dict)
    faults: list[FaultContext] = Field(default_factory=list)
```

- [ ] **Step 6: Update the existing prompt assertions**

`tests/test_chat_endpoint.py` already asserts on prompt text from commit `2a5d3e64`, including the CLEAN REP branch string. Run the file and fix every assertion that matched the old wording:

Run: `.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v`

Expected initially: the pre-existing clean-rep assertions FAIL because the string is now `CLEAN Squat REP`. Update them to the movement-scoped wording. This is expected churn, not a regression.

- [ ] **Step 7: Run the test to verify everything passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v`

Expected: all pass.

- [ ] **Step 8: Run the backend coverage gate**

Run: `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

Expected: pass at ≥95%.

- [ ] **Step 9: Commit**

```bash
git add backend/app/services/chat.py backend/app/routers/chat.py tests/test_chat_endpoint.py
git commit -m "feat(chat): ground the coach in the movement that was actually analysed

The preamble hardcoded 'the x-coach squat coach', so chatting about a
push-up analysis produced squat advice.

Also implements the spec's section 9 mitigation: the CLEAN REP branch now
names the movement. A measurable clip measured by the WRONG rules is newly
reachable (OHP's validity gate passes on squat footage), and wasMeasured
cannot catch it -- so the clean verdict is scoped to the movement the user
asserted rather than stated bare. The NOT MEASURED branch is unchanged.

Updates the prompt assertions added in 2a5d3e64 to the new wording."
```

---

# Phase D — Frontend

All commands in this phase run with **cwd = `frontend/`**.

### Task 9: API client, types, and chat grounding

**Files:**
- Modify: `frontend/src/lib/movements.ts`
- Modify: `frontend/src/api.ts:87-115`, `:149-156`, `:543-556`
- Modify: `frontend/src/lib/grounding.ts`
- Create: `frontend/src/test/lib.movements.test.ts`
- Modify: `frontend/src/test/api.test.ts`

**Interfaces:**
- Consumes: `GET /api/movements` (Task 5); `POST /api/analyze` `movement` field (Task 6).
- Produces: `AnalyzableMovement { name: string; validated: boolean }`; `api.getMovements(): Promise<AnalyzableMovement[]>`; `api.analyzeUpload(file: File, movement: string): Promise<Analysis>`; `Analysis.movement?: string`; `HistoryItem.movement?: string | null`; `ChatContext.movement?: string`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/lib.movements.test.ts`:

```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../api";

afterEach(() => vi.restoreAllMocks());

describe("api.getMovements", () => {
  it("returns the analyzable movements with their validation flags", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        movements: [
          { name: "Squat", validated: true },
          { name: "Overhead Press", validated: false },
          { name: "Push-up", validated: false },
        ],
      }),
    } as Response);

    const movements = await api.getMovements();
    expect(movements.map((m) => m.name)).toEqual(["Squat", "Overhead Press", "Push-up"]);
    expect(movements.find((m) => m.name === "Squat")?.validated).toBe(true);
    expect(movements.find((m) => m.name === "Push-up")?.validated).toBe(false);
  });
});

describe("api.analyzeUpload", () => {
  it("sends the chosen movement as a form field", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ video_id: "v1", movement: "Push-up" }),
    } as Response);

    await api.analyzeUpload(new File(["x"], "clip.mp4"), "Push-up");

    const body = fetchSpy.mock.calls[0][1]?.body as FormData;
    expect(body.get("movement")).toBe("Push-up");
    expect(body.get("file")).toBeInstanceOf(File);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (cwd `frontend/`): `yarn test src/test/lib.movements.test.ts`

Expected: FAIL — `api.getMovements is not a function`.

- [ ] **Step 3: Replace the hardcoded constant**

In `frontend/src/lib/movements.ts`, delete `ANALYZABLE_MOVEMENTS` and `isAnalyzable` (lines 31-37) and replace with the type. Keep `MOVEMENT_GROUPS` / `ALL_MOVEMENTS` — that catalog of all 16 is a content decision, not a pipeline fact:

```ts
// Which movements the pipeline can analyse is NOT stated here. It comes from GET /api/movements,
// which derives it from the Python detector registry, so registering a fourth detector surfaces
// it in the UI with no frontend edit. The previous hand-maintained ANALYZABLE_MOVEMENTS constant
// was a second list that had to be kept in sync by hand.
export interface AnalyzableMovement {
  name: string;
  /** False when the rules are literature-derived but never checked against labeled ground
   *  truth — rendered with a Beta tag. */
  validated: boolean;
}
```

- [ ] **Step 4: Add the API methods and types**

In `frontend/src/api.ts`, add to the `Analysis` interface:

```ts
  /** Which detector produced this analysis. Absent on analyses predating per-movement
   *  selection; consumers fall back to "Squat". */
  movement?: string;
```

Add to `HistoryItem`:

```ts
  movement?: string | null;
```

Add to `ChatContext`:

```ts
  movement?: string;
```

Add the import of the type at the top of the file (`import type { AnalyzableMovement } from "./lib/movements";`) and add the two methods to the exported `api` object, next to the other endpoints:

```ts
  // The movements the pipeline can actually analyse, derived server-side from the detector
  // registry. Backs the /movements cards and the studio selector.
  getMovements: () =>
    getJSON<{ movements: AnalyzableMovement[] }>("/api/movements").then((r) => r.movements),
```

and change `analyzeUpload` to take and send the movement:

```ts
  async analyzeUpload(file: File, movement: string): Promise<Analysis> {
    const form = new FormData();
    form.append("file", file);
    // Which detector runs. The backend rejects an unregistered value with 400 before it spends
    // a MediaPipe pass, and echoes the canonical spelling back as `movement` on the result.
    form.append("movement", movement);
    const res = await fetch("/api/analyze", {
```

(keep the rest of the existing body verbatim).

- [ ] **Step 5: Carry the movement into the chat context**

In `frontend/src/lib/grounding.ts`, inside `buildChatContext`, add the field to the returned object:

```ts
    // So the coach is grounded in the movement whose rules actually ran, not "squat" by
    // assumption. Absent for analyses predating per-movement selection; the backend falls back.
    movement: analysis.movement,
```

- [ ] **Step 6: Run the test to verify it passes**

Run (cwd `frontend/`): `yarn test src/test/lib.movements.test.ts`

Expected: 2 passed.

- [ ] **Step 7: Fix the existing api test if it calls `analyzeUpload`**

Run (cwd `frontend/`): `yarn test src/test/api.test.ts`

If any call site passes only a file, add the movement argument (`"Squat"`).

- [ ] **Step 8: Commit**

```bash
git add frontend/src/lib/movements.ts frontend/src/api.ts frontend/src/lib/grounding.ts frontend/src/test/lib.movements.test.ts frontend/src/test/api.test.ts
git commit -m "feat(frontend): fetch the analyzable movements instead of hardcoding them

Deletes ANALYZABLE_MOVEMENTS, whose own comment warned it had to be
updated by hand, in favour of GET /api/movements. analyzeUpload now sends
the chosen movement, and buildChatContext carries it so the coach is
grounded in the rules that actually ran."
```

---

### Task 10: Render the Movements menu from the fetched list, with Beta tags

**Files:**
- Modify: `frontend/src/pages/Movements.tsx`
- Modify: `frontend/src/lib/i18n.tsx`
- Modify: `frontend/src/test/pages.Movements.test.tsx`

**Interfaces:**
- Consumes: `api.getMovements()` (Task 9).
- Produces: live cards navigate to `/app?movement=<name>`.

- [ ] **Step 1: Write the failing test**

Rewrite the assertions in `frontend/src/test/pages.Movements.test.tsx`. Keep the existing `motion/react` and `react-router-dom` mocks verbatim (they are load-bearing — the global setup freezes `requestAnimationFrame`). Replace the `ANALYZABLE_MOVEMENTS` import with a fetch mock and these cases:

```ts
import { api } from "../api";

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
  { name: "Push-up", validated: false },
];

describe("Movements page", () => {
  beforeEach(() => {
    vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
    navigate.mockClear();
  });

  it("makes exactly the analyzable movements actionable", async () => {
    renderWithProviders(<Movements />);
    const buttons = await screen.findAllByRole("button");
    expect(buttons).toHaveLength(LIVE.length);
  });

  it("lists the rest as inert, not as disabled buttons", async () => {
    renderWithProviders(<Movements />);
    await screen.findAllByRole("button");
    expect(screen.getAllByText(/Soon|即將開放/).length).toBe(ALL_MOVEMENTS.length - LIVE.length);
  });

  it("tags unvalidated movements Beta and leaves Squat untagged", async () => {
    renderWithProviders(<Movements />);
    const betas = await screen.findAllByText("Beta");
    expect(betas).toHaveLength(2);
    const squatCard = screen.getByText("Squat").closest("button")!;
    expect(within(squatCard).queryByText("Beta")).toBeNull();
  });

  it("navigates to the studio with the chosen movement", async () => {
    renderWithProviders(<Movements />);
    const pushup = await screen.findByText("Push-up");
    await userEvent.click(pushup.closest("button")!);
    expect(navigate).toHaveBeenCalledWith("/app?movement=Push-up");
  });

  it("falls back to Squat-only when the list cannot be fetched", async () => {
    vi.spyOn(api, "getMovements").mockRejectedValue(new Error("offline"));
    renderWithProviders(<Movements />);
    const buttons = await screen.findAllByRole("button");
    expect(buttons).toHaveLength(1);
    expect(within(buttons[0]).getByText("Squat")).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (cwd `frontend/`): `yarn test src/test/pages.Movements.test.tsx`

Expected: FAIL — the page still reads the deleted constant, so it will not compile.

- [ ] **Step 3: Add the i18n strings**

In `frontend/src/lib/i18n.tsx`, add to the **`en`** dict near the other `movements.*` keys:

```ts
  "movements.beta": "Beta",
  "movements.betaNote":
    "Rules for this movement are literature-derived and have not yet been validated against labeled data.",
```

and to the **`zh`** dict (the parity guard requires every `en` key to have a `zh` translation):

```ts
  "movements.beta": "Beta",
  "movements.betaNote": "此動作的規則來自文獻推導，尚未以標註資料驗證。",
```

Also correct the now-false subtitle in both locales:

```ts
  "movements.subtitle":
    "Every movement the coach knows. Squat, Push-up and Overhead Press analysis are live today; the rest are on the way.",
```

```ts
  "movements.subtitle": "教練目前認識的所有動作。深蹲、伏地挺身與肩上推舉分析已經上線，其餘動作陸續開放。",
```

- [ ] **Step 4: Render from the fetched list**

In `frontend/src/pages/Movements.tsx`, replace the `isAnalyzable` import and usage. Add state and the fetch:

```tsx
import { useEffect, useState } from "react";
import { api } from "../api";
import { MOVEMENT_GROUPS, type AnalyzableMovement } from "../lib/movements";
```

Inside the component, above the `groupOffsets` computation:

```tsx
  // Which movements are analyzable comes from the server, derived from the detector registry, so
  // this page needs no edit when a detector is registered. On failure fall back to Squat-only:
  // the studio still works, and the alternative -- offering every movement -- would run the wrong
  // rules over the user's video.
  const [live, setLive] = useState<AnalyzableMovement[]>([{ name: "Squat", validated: true }]);
  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setLive(ms);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);
```

Replace `const live = isAnalyzable(movement);` inside the map with a lookup, and use it for both the action and the Beta tag:

```tsx
                    const entry = live.find((m) => m.name === movement);
                    const name = movementLabel(t, movement);
```

Change the `{live ? (` conditional to `{entry ? (`, change the `onClick` to carry the movement:

```tsx
                            onClick={() => navigate(`/app?movement=${encodeURIComponent(movement)}`)}
```

and add the Beta tag inside the live card's label block, right after the movement name span:

```tsx
                              {!entry.validated && (
                                <span
                                  title={t("movements.betaNote")}
                                  className="ml-2 rounded px-1.5 py-0.5 align-middle text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
                                >
                                  {t("movements.beta")}
                                </span>
                              )}
```

- [ ] **Step 5: Run the test to verify it passes**

Run (cwd `frontend/`): `yarn test src/test/pages.Movements.test.tsx`

Expected: 5 passed.

- [ ] **Step 6: Verify the i18n parity guard still passes**

Run (cwd `frontend/`): `yarn test src/test/lib.i18n.test.ts`

Expected: pass. If it fails, a key was added to `en` without a `zh` translation.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/Movements.tsx frontend/src/lib/i18n.tsx frontend/src/test/pages.Movements.test.tsx
git commit -m "feat(frontend): render the movement menu from GET /api/movements

Squat, Push-up and Overhead Press become actionable and hand off to the
studio with the movement in the URL. Unvalidated detectors carry a Beta
tag so the research state is legible.

A failed fetch falls back to Squat-only rather than offering everything --
the failure mode of the alternative is running the wrong rules over the
user's video."
```

---

### Task 11: Movement selector in the studio, and `?movement=` validation

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/DemoIntro.tsx`
- Modify: `frontend/src/lib/i18n.tsx`
- Create: `frontend/src/test/App.movement.test.tsx`

**Interfaces:**
- Consumes: `api.getMovements()`, `api.analyzeUpload(file, movement)` (Task 9).
- Produces: `DemoIntro` gains props `movements: AnalyzableMovement[]`, `movement: string`, `onMovementChange: (m: string) => void`, `movementError: string`.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/App.movement.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { api } from "../api";
import App from "../App";

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
  { name: "Push-up", validated: false },
];

describe("studio movement selection", () => {
  beforeEach(() => {
    vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
  });
  afterEach(() => vi.restoreAllMocks());

  it("preselects the movement from the URL", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    const select = (await screen.findByLabelText(/movement/i)) as HTMLSelectElement;
    expect(select.value).toBe("Push-up");
  });

  it("defaults to Squat when the URL says nothing", async () => {
    renderWithProviders(<App />, { route: "/app" });
    const select = (await screen.findByLabelText(/movement/i)) as HTMLSelectElement;
    expect(select.value).toBe("Squat");
  });

  it("sends the selected movement with the upload", async () => {
    const upload = vi
      .spyOn(api, "analyzeUpload")
      .mockResolvedValue({ video_id: "v1", movement: "Push-up" } as never);
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    await screen.findByLabelText(/movement/i);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await userEvent.upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
    expect(upload).toHaveBeenCalledWith(expect.any(File), "Push-up");
  });

  it("refuses an unanalyzable movement in the URL without spending an upload", async () => {
    const upload = vi.spyOn(api, "analyzeUpload");
    renderWithProviders(<App />, { route: "/app?movement=Lunge" });
    expect(await screen.findByText(/not.*analys|尚未/i)).toBeTruthy();
    expect(document.querySelector('input[type="file"]')).toBeNull();
    expect(upload).not.toHaveBeenCalled();
  });

  it("shows the Beta note for an unvalidated movement only", async () => {
    renderWithProviders(<App />, { route: "/app?movement=Overhead Press" });
    expect(await screen.findByText("Beta")).toBeTruthy();
  });
});
```

`renderWithProviders` does **not** currently accept a route — its `MemoryRouter` takes no `initialEntries`. Extend it in `frontend/src/test/renderWithProviders.tsx`, keeping the existing provider nesting:

```tsx
import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

// `route` seeds the MemoryRouter so a component can be rendered at a URL that carries query
// params (the studio reads ?movement=). Defaults to "/" so every existing caller is unaffected.
export function renderWithProviders(
  ui: ReactElement,
  options?: RenderOptions & { route?: string }
) {
  const { route = "/", ...renderOptions } = options ?? {};
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>
        <I18nProvider>{ui}</I18nProvider>
      </AuthProvider>
    </MemoryRouter>,
    renderOptions
  );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run (cwd `frontend/`): `yarn test src/test/App.movement.test.tsx`

Expected: FAIL — there is no movement control to find.

- [ ] **Step 3: Add the i18n strings**

Add to **`en`**:

```ts
  "studio.movement": "Movement",
  "studio.movementUnavailable":
    "\"{movement}\" cannot be analysed yet. Pick one of the available movements.",
```

Add to **`zh`**:

```ts
  "studio.movement": "動作",
  "studio.movementUnavailable": "「{movement}」尚未支援分析，請選擇其他已開放的動作。",
```

- [ ] **Step 4: Wire the state in `App.tsx`**

Add the fetch and derived selection above `runUpload`:

```tsx
  const [movements, setMovements] = useState<AnalyzableMovement[]>([
    { name: "Squat", validated: true },
  ]);
  // Tracked separately from the list itself: the seed value is a FALLBACK, not an answer, and
  // treating it as one would flash "Push-up cannot be analysed yet" on every slow load before
  // the real list arrives.
  const [movementsLoaded, setMovementsLoaded] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setMovements(ms);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setMovementsLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The movement is user-asserted input, taken from the URL when the studio is entered from the
  // /movements menu. Validate it against the fetched list BEFORE enabling the dropzone: the
  // backend would 400 anyway, but only after the user picked and uploaded a file.
  const requestedMovement = searchParams.get("movement");
  const [movement, setMovement] = useState<string>(requestedMovement || "Squat");
  useEffect(() => {
    if (requestedMovement) setMovement(requestedMovement);
  }, [requestedMovement]);

  const known = movements.some((m) => m.name === movement);
  // Only an ANSWERED "not analyzable" is an error. While the list is in flight we know nothing,
  // and "we don't know yet" must not render as "no".
  const movementError =
    !movementsLoaded || known ? "" : t("studio.movementUnavailable", { movement });
```

Add the imports at the top: `import { api, type Analysis } from "./api";` already exists — extend it, and add `import type { AnalyzableMovement } from "./lib/movements";`.

Change the upload call:

```tsx
      const data = await api.analyzeUpload(file, movement);
```

and add `movement` to `runUpload`'s dependency array.

Pass the new props to `DemoIntro`:

```tsx
        <DemoIntro
          onFile={runUpload}
          onOpenLibrary={() => setPickerOpen(true)}
          loading={loading}
          statusMsg={statusMsg}
          error={error}
          movements={movements}
          movement={movement}
          onMovementChange={setMovement}
          movementError={movementError}
          movementsLoaded={movementsLoaded}
        />
```

Add a test pinning that the warning does not flash before the list resolves — this is the bug the `movementsLoaded` gate exists to prevent, and without the test the gate can be refactored away:

```tsx
  it("does not claim a movement is unavailable while the list is still loading", async () => {
    let resolve!: (ms: { name: string; validated: boolean }[]) => void;
    vi.spyOn(api, "getMovements").mockReturnValue(
      new Promise((r) => {
        resolve = r;
      })
    );
    renderWithProviders(<App />, { route: "/app?movement=Push-up" });
    // In flight: "we don't know yet" must not render as "no".
    expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
    resolve(LIVE);
    expect(await screen.findByLabelText(/movement/i)).toBeTruthy();
    expect(screen.queryByText(/not.*analys|尚未/i)).toBeNull();
  });
```

- [ ] **Step 5: Render the selector in `DemoIntro.tsx`**

Extend `Props`:

```tsx
  movements: AnalyzableMovement[];
  movement: string;
  onMovementChange: (movement: string) => void;
  /** Non-empty when the requested movement is KNOWN not to be analyzable; the dropzone stays
   *  hidden. Empty while the catalog is still in flight. */
  movementError: string;
  /** False until GET /api/movements settles. The dropzone waits, so a slow network cannot let
   *  someone upload against a movement we have not confirmed. */
  movementsLoaded: boolean;
```

Add to the destructured parameters, and render above the dropzone. The dropzone is replaced (not merely disabled) when the movement is unusable, so there is nothing to upload into:

```tsx
        <div className="mb-4 flex items-center gap-2">
          <label htmlFor="movement-select" className="text-sm font-medium text-muted">
            {t("studio.movement")}
          </label>
          <select
            id="movement-select"
            value={movement}
            onChange={(e) => onMovementChange(e.target.value)}
            className="rounded-lg border border-border-dark bg-surface px-2.5 py-1.5 text-sm text-content"
          >
            {movements.map((m) => (
              <option key={m.name} value={m.name}>
                {movementLabel(t, m.name)}
              </option>
            ))}
            {/* Keep an unanalyzable URL-supplied movement visible rather than silently
                snapping the control to something the user did not choose. */}
            {!movements.some((m) => m.name === movement) && (
              <option value={movement}>{movementLabel(t, movement)}</option>
            )}
          </select>
          {movements.find((m) => m.name === movement)?.validated === false && (
            <span
              title={t("movements.betaNote")}
              className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
            >
              {t("movements.beta")}
            </span>
          )}
        </div>

        {movementError ? (
          <p className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-content">
            {movementError}
          </p>
        ) : loading || !movementsLoaded ? (
          <LumenLoader variant="scan" caption={statusMsg} />
        ) : (
          <UploadDropzone onFile={onFile} />
        )}
```

(replacing the existing `{loading ? ... : <UploadDropzone .../>}` conditional). Import `movementLabel` from `../lib/i18n` and the `AnalyzableMovement` type.

- [ ] **Step 6: Run the test to verify it passes**

Run (cwd `frontend/`): `yarn test src/test/App.movement.test.tsx`

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/DemoIntro.tsx frontend/src/lib/i18n.tsx frontend/src/test/App.movement.test.tsx frontend/src/test/renderWithProviders.tsx
git commit -m "feat(frontend): choose the movement in the studio

Reads ?movement= from the /movements handoff and validates it against the
fetched list before enabling the dropzone, so a hand-typed movement costs
no upload to discover a 400. The selector stays visible with the analysis,
and unvalidated movements carry the Beta tag."
```

---

### Task 12: Name the movement in the verdict, and badge it in history

The last part of the §9 mitigation: the clean-rep banner names the movement, so a mis-set selector is visible at the moment of the verdict rather than buried in a URL parameter.

**Files:**
- Modify: `frontend/src/components/CoachTray.tsx:316-335`
- Modify: `frontend/src/pages/History.tsx`
- Modify: `frontend/src/lib/i18n.tsx`
- Create: `frontend/src/test/components.CoachTray.movement.test.tsx`
- Modify: `frontend/src/test/pages.History.test.tsx` (if it exists; create the assertions inline in the new test file if not)

**Interfaces:**
- Consumes: `Analysis.movement`, `HistoryItem.movement` (Task 9).

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/components.CoachTray.movement.test.tsx`:

Reuse the shared fixtures the existing `components.CoachTray.test.tsx` uses — `mockCleanAnalysis` and `mockUnmeasuredAnalysis` from `./fixtures`. That file needs **no** chat mocks (the coaching-feedback half of the tray renders without them) and `activeFaultId` is optional, so follow its call shape exactly rather than hand-rolling an `Analysis` factory:

```tsx
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import CoachTray from "../components/CoachTray";
import { mockCleanAnalysis, mockUnmeasuredAnalysis } from "./fixtures";

describe("CoachTray — the clean-rep verdict names the movement", () => {
  it("names the movement whose rules ran", () => {
    renderWithProviders(
      <CoachTray
        analysis={{ ...mockCleanAnalysis, movement: "Push-up" }}
        currentTime={0}
        onSeek={vi.fn()}
      />
    );
    expect(screen.getByText(/No Push-up faults detected/i)).toBeInTheDocument();
  });

  it("falls back to Squat for an analysis predating per-movement selection", () => {
    renderWithProviders(
      <CoachTray analysis={mockCleanAnalysis} currentTime={0} onSeek={vi.fn()} />
    );
    expect(screen.getByText(/No Squat faults detected/i)).toBeInTheDocument();
  });

  it("still refuses to claim a clean rep on an unmeasured clip", () => {
    renderWithProviders(
      <CoachTray
        analysis={{ ...mockUnmeasuredAnalysis, movement: "Overhead Press" }}
        currentTime={0}
        onSeek={vi.fn()}
      />
    );
    expect(screen.queryByText(/No Overhead Press faults detected/i)).not.toBeInTheDocument();
    expect(screen.getByText(/could not be measured|無法測量/i)).toBeInTheDocument();
  });
});
```

Note `components.CoachTray.test.tsx:52` already asserts `/No biomechanical faults/i` against `mockCleanAnalysis`. That assertion **will** fail once the string is movement-scoped — update it to `/No Squat faults detected/i`. Expected churn from `2a5d3e64`'s assertions, not a regression.

- [ ] **Step 2: Run the test to verify it fails**

Run (cwd `frontend/`): `yarn test src/test/components.CoachTray.movement.test.tsx`

Expected: FAIL — the banner reads the movement-free "No biomechanical faults detected. Clean rep."

- [ ] **Step 3: Make the clean-rep string movement-scoped**

In `frontend/src/lib/i18n.tsx`, change `feedback.noFaults` in **`en`** (the `t()` implementation already substitutes `{var}` — see `i18n.tsx:1311-1314`):

```ts
  // Names the movement because it is USER-ASSERTED: the studio lets the user pick, so a clip can
  // be measured by rules that do not describe it. Naming it makes the claim true relative to the
  // user's own assertion and puts that assertion in front of them at the verdict.
  "feedback.noFaults": "No {movement} faults detected. Clean rep.",
```

and in **`zh`**:

```ts
  "feedback.noFaults": "未偵測到{movement}的生物力學錯誤，這一下很標準。",
```

- [ ] **Step 4: Pass the movement at the call site**

In `frontend/src/components/CoachTray.tsx`, find the `t("feedback.noFaults")` call in the clean-rep banner and give it the variable, falling back for older analyses:

```tsx
{t("feedback.noFaults", { movement: movementLabel(t, analysis.movement ?? "Squat") })}
```

Import `movementLabel` from `../lib/i18n` alongside the existing `useI18n` import. Leave the `wasMeasured` guard and the `feedback.notMeasured` branch exactly as they are.

- [ ] **Step 5: Add the history badge**

In `frontend/src/lib/i18n.tsx` there is nothing new to add — `movementLabel` already covers all 16 names in both locales.

In `frontend/src/pages/History.tsx`, inside the row rendering (near the existing `viewLabel(t, it.view_type ?? "unknown")` usage around line 159), add the badge:

```tsx
                            <span className="rounded bg-content/5 px-1.5 py-0.5 text-[11px] font-medium text-muted">
                              {/* The promoted column, then Squat. `HistoryItem` carries no
                                  `result` -- list_analyses selects only the promoted columns, not
                                  the heavy document -- so there is no per-row echo to fall back
                                  to here. Rows predating the column are Squat by construction:
                                  every analysis before this change was pinned to it. */}
                              {movementLabel(t, it.movement ?? "Squat")}
                            </span>
```

Import `movementLabel` from `../lib/i18n` — `History.tsx` already imports `viewLabel` from there, so extend that import.

- [ ] **Step 6: Run the test to verify it passes**

Run (cwd `frontend/`): `yarn test src/test/components.CoachTray.movement.test.tsx`

Expected: 3 passed.

- [ ] **Step 7: Run the whole frontend suite**

Run (cwd `frontend/`): `yarn test`

Expected: all pass. Existing CoachTray tests that assert the old clean-rep string need updating to the movement-scoped wording — that is expected churn from `2a5d3e64`'s assertions, not a regression.

- [ ] **Step 8: Run both coverage gates**

Run (cwd `frontend/`): `yarn test:coverage`

Run (cwd repo root): `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`

Expected: both pass.

- [ ] **Step 9: Fix the remaining stale copy**

In `frontend/src/lib/i18n.tsx:234`, the landing description says x-coach "reads a squat video". Update in both locales to reflect three movements, e.g.:

```ts
  "landing.subtitle":
    "x-coach reads a squat, push-up or overhead-press video, locates the fault, traces its cause in a biomechanics knowledge graph spanning 16 movements, and explains the fix.",
```

Run the i18n parity guard once more: `yarn test src/test/lib.i18n.test.ts`

- [ ] **Step 10: Commit**

```bash
git add frontend/src/components/CoachTray.tsx frontend/src/pages/History.tsx frontend/src/lib/i18n.tsx frontend/src/test/components.CoachTray.movement.test.tsx
git commit -m "feat(frontend): name the movement in the verdict and badge it in history

Completes the spec's section 9 mitigation. A measurable clip measured by
the wrong rules is newly reachable and the wasMeasured guard cannot catch
it, so the clean-rep banner names the movement whose rules ran -- putting
the user's own assertion in front of them at the moment of the verdict
instead of leaving it buried in a URL parameter.

The unmeasured branch is untouched and still refuses to claim a clean rep.
History rows carry a movement badge, falling back to Squat for rows that
predate the column."
```

---

## Final verification

- [ ] **Both gates, from a clean tree**

Run (repo root): `.venv\Scripts\python.exe -m pytest tests/ -v`
Run (repo root): `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95`
Run (cwd `frontend/`): `yarn test:coverage`
Run (cwd `frontend/`): `yarn build`

- [ ] **Manual end-to-end check**

Start the backend and `yarn dev`, then: `/movements` shows Squat, Push-up and Overhead Press as live (the latter two tagged Beta); clicking Push-up lands on `/app?movement=Push-up` with the selector preset; uploading a push-up clip returns push-up faults with non-empty causes/risks/cues; the coach opens as the "Push-up coach"; the analysis appears in History with a Push-up badge.

- [ ] **Report the manual migration**

`db/migrations/20260725000000_analysis_movement.sql` must be applied to Supabase. Until it is, `persist_analysis` will fail on the unknown column — this is the one step that cannot be verified locally.

- [ ] **Re-verify the graph on the deploy target**

`data/kg/sports_kg_v3.graphml` is gitignored and built separately per environment. Run `.venv\Scripts\python.exe scripts/knowledge/author_ohp_lockout_v3.py` there, then `.venv\Scripts\python.exe -m pytest tests/test_kg_query_resolution.py -v` to confirm all 13 kg-mode queries resolve.
