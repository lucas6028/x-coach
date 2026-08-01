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
