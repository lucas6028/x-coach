"""Author minimal KG stubs for the GENERAL-tier movements into sports_kg_v3.graphml.

Design: docs/movement-kg-expansion-plan.md §5 — general movements get "a minimal KG stub
(Exercise + top faults, no deep causal chains). Cheap breadth." This is a DETERMINISTIC
authoring pass, NOT a gemini extraction: no API cost, full control, honest provenance.

Discipline (carried from the flagship reconciles + advisor guidance):
  * Faults are DATASET-GROUNDED where a dataset ships a fault taxonomy (EgoExo-Fitness TKV
    technical-keypoint checklists — the top-failed criteria per action, by real failure rate);
    TEXTBOOK-AUTHORED (2-3 canonical, uncontroversial faults) where the dataset ships only
    binary correctness (REHAB24-6) or no labels (Fit3D). Provenance recorded per movement.
  * Shared-layer links are added ONLY when DEFINITIONAL, and ONLY to shared nodes that
    ALREADY EXIST in the graph. We NEVER mint a new shared node here (that would re-fragment
    the layer the flagship work cleaned). A link whose target is absent is WARNED and SKIPPED.
  * These are AUTHORED STUB LINKS, not extraction-observed bridges. We do not chase a bridge
    count. Many stubs are legitimately sparse or isolated — that is the honest state.

Movement set is locked by data/kg/exercise_canonical_mapping_v1.json (tier == "general").

Run (dry-run first, always):
    .venv\\Scripts\\python.exe scripts/knowledge/stub_general_movements_v3.py --dry-run
    .venv\\Scripts\\python.exe scripts/knowledge/stub_general_movements_v3.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.knowledge.kg_schema import (  # noqa: E402
    SCOPED_LABELS,
    SHARED_LABELS,
    resolve_node_id,
    shared_aliases,
)

GRAPH = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
BACKUP = REPO_ROOT / "data" / "kg" / "sports_kg_v3.pre-general-stubs.bak.graphml"

# Fault -> shared target's edge type is derived from the EXISTING target node's label.
LABEL_TO_EDGE = {
    "Cause": "CAUSED_BY",
    "Risk": "INCREASES_RISK_OF",
    "Cue": "CORRECTED_BY",
    "QualityDimension": "AFFECTS_QUALITY",
}

# --------------------------------------------------------------------------------------
# STUB SPEC. Per movement: grounding provenance + faults, each fault a (name, [shared ids]).
# Shared ids are EXACT existing canonical node names (verified against the live graph's
# shared inventory); the script asserts each pre-exists and derives the edge type from its
# label. Empty link list = an honestly isolated / movement-specific fault (kept, no bridge).
# --------------------------------------------------------------------------------------
STUB_SPEC: dict[str, dict] = {
    # ---- Fit3D (no shipped fault labels -> textbook-authored, canonical hip-hinge/isolation) ----
    "Deadlift": {
        "grounding": "textbook (Fit3D ships 3D GT + rep boundaries only, no fault labels)",
        "faults": {
            "Lumbar Flexion": ["Lumbar Spine Injury", "Maintain Neutral Spine"],
            "Hyperextension At Lockout": ["Lumbar Spine Injury"],
            "Insufficient Hip Hinge": ["Hip Hinge"],
            "Bar Drift From Body": ["Lower Back Load"],
        },
    },
    "Bicep Curl": {
        "grounding": "textbook (Fit3D, no fault labels)",
        "faults": {
            "Elbow Drift Forward": [],
            "Using Momentum": ["Forward Momentum"],
            "Incomplete Range Of Motion": ["Range Of Motion"],
            "Wrist Flexion Under Load": ["Wrists In Line With Forearms"],
        },
    },
    "Band Pull Apart": {
        "grounding": "textbook (Fit3D, no fault labels)",
        "faults": {
            "Insufficient Scapular Retraction": ["Limited Scapular Retraction"],
            "Shoulder Shrugging": ["Shoulder Depression", "Weak Scapular Stabilizers"],
            "Bent Elbows": [],
        },
    },
    # ---- REHAB24-6 (binary correctness only -> textbook-authored) ----
    "Arm Abduction": {
        "grounding": "textbook (REHAB24-6 ships binary correctness only, no fault taxonomy)",
        "faults": {
            "Compensatory Shoulder Shrug": ["Shoulder Depression"],
            "Trunk Lean Compensation": ["No Compensatory Trunk Movement"],
            "Incomplete Elevation": ["Humerus Abduction", "Limited Shoulder ROM"],
        },
    },
    "Arm VW": {
        "grounding": "textbook (REHAB24-6 binary correctness only)",
        "faults": {
            "Insufficient Scapular Retraction": ["Limited Scapular Retraction"],
            "Compensatory Shoulder Shrug": ["Shoulder Depression"],
            "Trunk Lean Compensation": ["No Compensatory Trunk Movement"],
        },
    },
    "Leg Abduction": {
        "grounding": "textbook (REHAB24-6 binary correctness only)",
        "faults": {
            "Insufficient Abduction Range": ["Weak Hip Abductors", "Hip Abduction"],
            "Trunk Lean Compensation": ["No Compensatory Trunk Movement"],
            "Pelvic Hiking": ["Pelvic Control"],
        },
    },
    # ---- EgoExo-Fitness (TKV technical-keypoint checklist -> DATASET-GROUNDED top-failed criteria) ----
    "Sit-up": {
        "grounding": "EgoExo-Fitness TKV (top-failed criteria: feet-together 54%, arms-overhead 44%, "
                     "forward-reach 34%; + core engagement)",
        "faults": {
            "Feet Not Together": [],
            "Arms Not Extended Overhead": [],
            "Incomplete Forward Reach": ["Range Of Motion"],
            "Abdominal Disengagement": ["Core Stability", "Weak Core Stability"],
        },
    },
    "Shoulder Bridge": {
        "grounding": "EgoExo-Fitness TKV (top form failures: incomplete hip/shoulder alignment 9%, "
                     "non-segmental lowering 6%, core)",
        "faults": {
            "Incomplete Hip Extension": ["Poor Hip Extension", "Weak Gluteus Maximus", "Squeeze Glutes"],
            "No Segmental Spinal Articulation": [],
            "Loss Of Core Engagement": ["Core Stability"],
        },
    },
    "Jumping Jacks": {
        "grounding": "EgoExo-Fitness TKV (Jumping/Clap Jacks: arm tension 8-27%, foot split 10%, "
                     "arm-leg coordination)",
        "faults": {
            "Insufficient Arm Tension": [],
            "Poor Arm-Leg Coordination": ["Poor Neuromuscular Control"],
            "Incomplete Foot Split": [],
        },
    },
    "High Knee": {
        "grounding": "EgoExo-Fitness TKV (top-failed: cadence/speed 44%, arm rhythm 26%, "
                     "upper-body stability 15%, knee lift 10%)",
        "faults": {
            "Slow Cadence": ["Maintain Even Tempo"],
            "Unstable Upper Body": ["Trunk Stability", "Core Stability"],
            "Poor Arm-Leg Rhythm": ["Poor Neuromuscular Control"],
            "Insufficient Knee Lift": [],
        },
    },
    "Torso Twist": {
        "grounding": "EgoExo-Fitness TKV (Kneeling Side Torso Twist: pause-at-bottom 23%, "
                     "lateral-flexion depth 21%, base 13%, abs)",
        "faults": {
            "Insufficient Lateral Flexion Depth": ["Range Of Motion"],
            "Poor Abdominal Engagement": ["Core Stability", "Weak Core Stability"],
            "Unstable Base": [],
        },
    },
}


def resolve_shared_target(G: nx.MultiDiGraph, name: str) -> tuple[str, str] | None:
    """Return (existing_node_id, label) for a shared target, or None if it does NOT already
    exist as a shared node. Tries the exact id first, then the shared alias map. Never creates."""
    def ok(nid: str) -> bool:
        return nid in G and str(G.nodes[nid].get("movement")) == "shared"

    if ok(name):
        return name, str(G.nodes[name]["label"])
    aliases = shared_aliases()
    for label in SHARED_LABELS:
        canon = aliases.get(label, {}).get(name)
        if canon and ok(canon):
            return canon, str(G.nodes[canon]["label"])
    return None


def author_stubs(G: nx.MultiDiGraph, log: list[str]) -> dict:
    stats = {"actions": 0, "faults": 0, "has_fault": 0, "shared_links": 0,
             "skipped_links": [], "collisions": []}
    edge_by_type: Counter = Counter()

    for movement, spec in STUB_SPEC.items():
        act_id, act_attrs = resolve_node_id(movement, "Action", movement)
        if act_id in G:
            stats["collisions"].append(f"action {act_id!r} already exists")
            continue
        G.add_node(act_id, **act_attrs)
        stats["actions"] += 1
        log.append(f"\n[{movement}]  ({spec['grounding']})")
        log.append(f"  + Action {act_id!r}")

        for fault_name, targets in spec["faults"].items():
            f_id, f_attrs = resolve_node_id(fault_name, "Fault", movement)
            if f_id in G:
                stats["collisions"].append(f"fault {f_id!r} already exists")
                continue
            G.add_node(f_id, **f_attrs)
            G.add_edge(act_id, f_id, type="HAS_FAULT")
            stats["faults"] += 1
            stats["has_fault"] += 1
            linkstr = []
            for tgt in targets:
                resolved = resolve_shared_target(G, tgt)
                if resolved is None:
                    stats["skipped_links"].append(f"{movement}/{fault_name} -> {tgt!r} (absent)")
                    log.append(f"      ! SKIP link -> {tgt!r} (not an existing shared node)")
                    continue
                tid, tlabel = resolved
                etype = LABEL_TO_EDGE[tlabel]
                G.add_edge(f_id, tid, type=etype)
                stats["shared_links"] += 1
                edge_by_type[etype] += 1
                linkstr.append(f"{tid!r}[{tlabel}] ({etype})")
            log.append(f"    + Fault {fault_name!r}"
                       + (f"  ->  {', '.join(linkstr)}" if linkstr else "  (isolated)"))
    stats["edge_by_type"] = dict(edge_by_type)
    return stats


def verify(G: nx.MultiDiGraph, log: list[str]) -> bool:
    log.append("\n=== VERIFY ===")
    ok = True
    labels = Counter(str(d.get("label")) for _, d in G.nodes(data=True))
    valid_labels = SCOPED_LABELS | SHARED_LABELS
    bad_labels = {l: c for l, c in labels.items() if l not in valid_labels}
    none_mv = [n for n, d in G.nodes(data=True) if str(d.get("movement")) == "None"]
    edge_types = set(LABEL_TO_EDGE.values()) | {"HAS_FAULT", "HAS_PHASE", "OCCURS_IN_PHASE", "INDICATED_BY"}
    bad_edges = [(u, v, d.get("type")) for u, v, d in G.edges(data=True) if d.get("type") not in edge_types]

    log.append(f"  labels: {dict(labels)}")
    log.append(f"  non-standard labels: {bad_labels or 'NONE'}")
    log.append(f"  movement=None nodes: {len(none_mv)}")
    log.append(f"  non-canonical edges: {len(bad_edges)}")
    if bad_labels or none_mv or bad_edges:
        ok = False

    # every general movement present + reachable Action->Fault->(shared)
    for movement in STUB_SPEC:
        if movement not in G:
            log.append(f"  MISSING movement node: {movement}"); ok = False; continue
        faults = [v for _, v, d in G.out_edges(movement, data=True) if d.get("type") == "HAS_FAULT"]
        n_bridge = 0
        for f in faults:
            for _, t, d in G.out_edges(f, data=True):
                if str(G.nodes[t].get("movement")) == "shared":
                    # authored stub link crosses to a shared node also used by >=1 other movement?
                    others = {str(G.nodes[p].get("movement")) for p in G.predecessors(t)} - {movement, "shared"}
                    if others - {"None"}:
                        n_bridge += 1
        log.append(f"  {movement:16s}: {len(faults)} faults, {n_bridge} authored links onto shared nodes shared with other movements")

    log.append(f"  => VERIFY {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Author general-movement KG stubs into sports_kg_v3.")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; do not write")
    args = ap.parse_args()

    G = nx.read_graphml(GRAPH)
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    log: list[str] = [f"loaded {GRAPH.name}: {n0} nodes / {e0} edges"]

    stats = author_stubs(G, log)
    log.append("\n=== SUMMARY ===")
    log.append(f"  + {stats['actions']} Action nodes, {stats['faults']} Fault nodes, "
               f"{stats['has_fault']} HAS_FAULT edges, {stats['shared_links']} shared links")
    log.append(f"  shared link edge types: {stats['edge_by_type']}")
    if stats["skipped_links"]:
        log.append(f"  SKIPPED (absent shared target) x{len(stats['skipped_links'])}:")
        for s in stats["skipped_links"]:
            log.append(f"    - {s}")
    else:
        log.append("  skipped links: NONE (every shared target pre-existed)")
    if stats["collisions"]:
        log.append(f"  COLLISIONS x{len(stats['collisions'])}: {stats['collisions']}")
    log.append(f"  graph now: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges "
               f"(+{G.number_of_nodes()-n0} / +{G.number_of_edges()-e0})")

    ok = verify(G, log)
    print("\n".join(log))

    if stats["collisions"] or not ok:
        raise SystemExit("\nABORT: collisions or verify failure — not writing.")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    if not BACKUP.exists():
        nx.write_graphml(nx.read_graphml(GRAPH), BACKUP)
        print(f"\nbackup -> {BACKUP.name}")
    nx.write_graphml(G, GRAPH)
    print(f"wrote {GRAPH.name}: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()
