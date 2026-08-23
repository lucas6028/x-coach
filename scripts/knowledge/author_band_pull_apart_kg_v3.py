"""Close the two recorded Band Pull Apart KG gaps in sports_kg_v3.graphml.

WHY THIS EXISTS. `src/pose/movements/band_pull_apart.py` shipped with two gaps recorded
rather than masked (its STEP 0 comment block, and section 5 of
docs/superpowers/specs/2026-08-09-band-pull-apart-detector-design.md):

  * rule 2 `bpa_incomplete_rom` queries "Bent Elbows". The node EXISTS but has connectivity
    0 -- only the HAS_FAULT backlink -- so its FaultCard renders
    `likely causes: -, injury risks: -, corrective cues: -`.
  * rule 4 `bpa_trunk_extension_compensation` queried "No Compensatory Trunk Movement", a
    SHARED QualityDimension with zero out-edges. Seeding there returns empty buckets
    (`summarize_seed` keeps a `quality_impacts` counterpart only when it is itself a
    QualityDimension, so the four inbound Fault edges are filtered out). There was no
    Band-Pull-Apart-scoped node for the fault at all.

This script adds the minimum that is CITABLE, and nothing more. Run it after
`stub_general_movements_v3.py`; it is idempotent, so re-running is safe and adds only what is
missing.

CITATION. The movement's only source is the one both rules already carry
(band_pull_apart.py): Fukunaga T et al. Band Pull-Apart Exercise: Effects of Movement
Direction and Hand Position on Shoulder Muscle Activity. Int J Sports Phys Ther (2022).
PMC8975561, DOI 10.26603/001c.33026. Local copy:
data/rag/docs/PMC8975561_band_pull_apart_shoulder_emg.txt.

WHAT THE CITATION DOES AND DOES NOT SUPPORT -- read before adding an edge here.

  Bent Elbows -> Range Of Motion (AFFECTS_QUALITY). Fukunaga's headline direction effect is a
  RANGE argument: the diagonal-up movement, the largest excursion against gravity, produced
  the highest shoulder-girdle activity ("resulting in higher overall load"). Bending the
  elbows shortens the arc the hands travel, which is the same quantity. The edge is also
  definitional for this detector: `rule_incomplete_rom` fires on wrist spread OR elbow angle
  under ONE fault_id (`bpa_incomplete_rom`), so "Bent Elbows" IS this movement's
  incomplete-ROM node. Mirrors `Bicep Curl:Incomplete Range Of Motion -> Range Of Motion` in
  stub_general_movements_v3.py -- the same fault class, authored in the same pass.

  NOT the same thing as the rejected option. Section 5 of the design doc rejected pointing
  the QUERY at the shared `Range Of Motion` node, because SEEDING there collects that node's
  own out-edge `CORRECTED_BY -> Wrapping Surface Adjustment`, a correction meaningless for
  this movement. An AFFECTS_QUALITY edge FROM the scoped fault is the opposite direction: at
  `hops=1` it puts `Range Of Motion` in `quality_impacts` and collects nothing of that node's
  own. Every production caller uses hops=1 (`pose_rule_detector.py:775` hard-codes it;
  `backend/app/settings.py:197` `_DEFAULT_KG_HOPS = 1`; `chat.py` clamps to 1-2, default 1).
  STATED LIMITATION: `Range Of Motion` is a degree-62 hub, so an admin who raises `kg_hops`
  to 2 WILL pull squat content and that stray cue into this card. That is a pre-existing
  property of the hub, shared by Squat:Insufficient Depth, Sit-up:Incomplete Forward Reach
  and Bicep Curl:Incomplete Range Of Motion -- not introduced here, and not fixed here.

  Trunk Extension Compensation -> No Compensatory Trunk Movement (AFFECTS_QUALITY).
  Definitional: the fault IS compensatory trunk movement, and the quality dimension is its
  absence. The same single edge Arm Abduction / Arm VW / Leg Abduction's `Trunk Lean
  Compensation` nodes carry.

  NO RISK EDGE, AND NO CAUSE OR CUE, ON EITHER FAULT. Deliberate, and each absence has its
  own reason:
    - Fukunaga never mentions the elbow. Not in the protocol (subjects were instructed only
      to "hold the band without slack or tension at the beginning of each exercise
      movement"), not in the results, not in the discussion. A CAUSED_BY or CORRECTED_BY edge
      on Bent Elbows would attach Fukunaga's citation to a claim Fukunaga does not make.
    - On the trunk fault the source runs the OTHER WAY: it frames the standing variant's
      trunk involvement as a feature -- "performing strengthening exercises with hip and
      trunk extension may be beneficial". `rule_trunk_extension_compensation`'s own
      citation_support already says its harm claim is "partly inferential". An
      INCREASES_RISK_OF edge would convert an acknowledged inference into a graph fact that
      contradicts its source.
  `tests/test_kg_band_pull_apart.py` pins all three empty buckets, so a later pass cannot pad
  these cards without deleting a test that says why not.

TWO PLACES, ONE TRUTH. `STUB_SPEC["Band Pull Apart"]` in
scripts/knowledge/stub_general_movements_v3.py carries the SAME two additions, so a graph
built from scratch and a graph patched by this script agree. `author_stubs` skips a movement
whose Action node already exists, so on an EXISTING graph the STUB_SPEC edit is inert and
this script is what applies the fix. Change one, change the other.

DEPLOY. The graphml is a gitignored build artifact and Azure Files receives the LOCAL file by
`az storage file upload` (docs/azure-deployment.md:221-224), so running this script locally
and re-uploading `data/kg/sports_kg_v3.graphml` IS the deploy step. Rebuild order:
scripts/knowledge/README.md.

Run from repo root:
    .venv\\Scripts\\python.exe scripts/knowledge/author_band_pull_apart_kg_v3.py --dry-run
    .venv\\Scripts\\python.exe scripts/knowledge/author_band_pull_apart_kg_v3.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402

from src.knowledge.kg_schema import resolve_node_id  # noqa: E402

GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
MOVEMENT = "Band Pull Apart"

# The fault this script CREATES. `rule_trunk_extension_compensation` names it
# "Trunk-Extension Compensation (Leaning Back)" for display; the graph name drops the
# parenthetical and the hyphen so `resolve_nodes` matches the detector's kg_query exactly.
NEW_FAULT = "Trunk Extension Compensation"
# The fault that already exists and only needs its first edge.
EXISTING_FAULT = "Bent Elbows"

# Must ALREADY exist. A missing one means this is not the graph the script was written
# against (stub_general_movements_v3.py has not run), and creating them here would fabricate
# structure -- in particular it would mint a shared node, which that script forbids.
REQUIRED_EXISTING = [
    MOVEMENT,                                    # Action anchor
    f"{MOVEMENT}:{EXISTING_FAULT}",              # Fault, currently connectivity 0
    "Range Of Motion",                           # shared QualityDimension
    "No Compensatory Trunk Movement",            # shared QualityDimension
]


def add_edge_dedup(G: nx.MultiDiGraph, u: str, v: str, etype: str) -> bool:
    """Add a typed edge unless an identical one is already present. Mirrors the helper in
    author_ohp_lockout_v3.py so repeated runs cannot multiply parallel edges."""
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
    ap = argparse.ArgumentParser(description="Close the Band Pull Apart KG gaps in sports_kg_v3.")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; do not write")
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
            "Expected nodes are absent, so this is not the graph this script targets "
            "(run stub_general_movements_v3.py first): " + ", ".join(repr(m) for m in missing)
        )

    log: list[str] = []
    print("\n--- nodes ---")
    bent_id = f"{MOVEMENT}:{EXISTING_FAULT}"
    trunk_id = add_node_if_absent(G, NEW_FAULT, "Fault", log)
    print("\n".join(log))

    edges = [
        # The movement owns the new fault -- the same shape as the three stub faults.
        (MOVEMENT, trunk_id, "HAS_FAULT"),
        # Definitional: the fault IS compensatory trunk movement.
        (trunk_id, "No Compensatory Trunk Movement", "AFFECTS_QUALITY"),
        # Fukunaga's direction effect is a range argument; bent elbows shorten the arc.
        (bent_id, "Range Of Motion", "AFFECTS_QUALITY"),
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
