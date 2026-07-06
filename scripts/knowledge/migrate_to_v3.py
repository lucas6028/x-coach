"""Migrate the squat-only KG (squat_kg_v2.graphml) into the multi-movement layered
schema (sports_kg_v3.graphml).

What it does (see docs/kg-schema-generalization.md):
  - Scoped labels (Action, Phase, Fault, EvidenceSignal): namespace id -> "Squat:<name>",
    tag movement="Squat", keep name="<name>". The primary anchor keeps id "Squat".
  - Shared labels (Cause, Cue, Risk, QualityDimension): collapse duplicates via
    shared_vocab_v1.json (label-scoped alias->canonical), keep a plain id, tag movement="shared".
  - Apply the (valid) within-scope merges from squat_canonical_mapping_v1.json (phase/fault/etc).
  - Drop label-string junk nodes; salvage the recoverable Unknown-labelled nodes.
  - Rewire every edge through the id remap, drop edges to dropped nodes, dedup (u, v, type).

Run from repo root:  python scripts/knowledge/migrate_to_v3.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_GRAPH = PROJECT_ROOT / "data" / "kg" / "squat_kg_v2.graphml"
DST_GRAPH = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
SHARED_VOCAB = PROJECT_ROOT / "data" / "kg" / "shared_vocab_v1.json"
SQUAT_MAPPING = PROJECT_ROOT / "data" / "kg" / "squat_canonical_mapping_v1.json"

MOVEMENT = "Squat"
SCOPED_LABELS = {"Action", "Phase", "Fault", "EvidenceSignal"}
SHARED_LABELS = {"Cause", "Cue", "Risk", "QualityDimension"}

# Nodes whose label is missing/junk and whose id is really an edge/label string -> drop.
DROP_IDS = {"INDICATED_BY", "Fault", "HAS_FAULT", "OCCURS_IN_PHASE"}

# Recover the 5 Unknown-labelled nodes: id -> (new_label, new_name) or None to drop.
SALVAGE = {
    "Ankle Mobility": ("Cause", "Limited Ankle Mobility"),
    "Improve Ankle Mobility": ("Cue", "Improve Ankle Mobility"),
    "Movement Stability": ("QualityDimension", "Stability"),
    "Subtalar Joint Rotation": None,
    "Physiological Response": None,
}


def build_scoped_rename(mapping: dict) -> dict:
    """Raw scoped-node name -> canonical name, from squat_canonical_mapping *_rules.merge/.keep."""
    rename = {}
    for key, rules in mapping.items():
        if not (isinstance(rules, dict) and key.endswith("_rules")):
            continue
        for bucket in ("keep", "merge"):
            for raw, canon in rules.get(bucket, {}).items():
                rename[raw] = canon
    return rename


def main() -> int:
    if not SRC_GRAPH.exists():
        print(f"Source graph not found: {SRC_GRAPH}")
        return 1

    vocab = json.loads(SHARED_VOCAB.read_text(encoding="utf-8"))
    shared_aliases = vocab["aliases"]  # {label: {raw: canonical}}
    squat_map = json.loads(SQUAT_MAPPING.read_text(encoding="utf-8"))
    scoped_rename = build_scoped_rename(squat_map)

    G = nx.read_graphml(SRC_GRAPH)
    if not G.is_multigraph():
        G = nx.MultiDiGraph(G)

    id_map: dict[str, str] = {}          # old id -> new id (dropped nodes absent)
    new_nodes: dict[str, dict] = {}      # new id -> attrs
    stats = Counter()

    for old_id, attrs in G.nodes(data=True):
        old_id = str(old_id)
        label = str(attrs.get("label", "")).strip()

        # --- junk / salvage ---------------------------------------------------
        if old_id in DROP_IDS or label in {"", "?"}:
            stats["dropped_junk"] += 1
            continue
        if label == "Unknown":
            if old_id not in SALVAGE or SALVAGE[old_id] is None:
                stats["dropped_unknown"] += 1
                continue
            label, name = SALVAGE[old_id]
        else:
            name = old_id

        # --- shared layer: collapse via vocab, plain id, movement=shared ------
        if label in SHARED_LABELS:
            canon = shared_aliases.get(label, {}).get(name, name)
            new_id = canon
            node_attrs = {"label": label, "name": canon, "movement": "shared"}
            if canon != name:
                stats["shared_collapsed"] += 1

        # --- scoped layer: namespace id, movement=Squat -----------------------
        elif label in SCOPED_LABELS:
            canon = scoped_rename.get(name, name)
            if canon != name:
                stats["scoped_merged"] += 1
            if label == "Action" and canon == MOVEMENT:
                new_id = MOVEMENT  # the anchor keeps a bare id
            else:
                new_id = f"{MOVEMENT}:{canon}"
            node_attrs = {"label": label, "name": canon, "movement": MOVEMENT}
        else:
            # Any other label: keep scoped to be safe.
            new_id = f"{MOVEMENT}:{name}"
            node_attrs = {"label": label, "name": name, "movement": MOVEMENT}
            stats["other_label"] += 1

        id_map[old_id] = new_id
        if new_id in new_nodes:
            # collision -> node collapse; keep first label, prefer non-shared movement info
            stats["node_collapsed"] += 1
        else:
            new_nodes[new_id] = node_attrs

    # --- rebuild graph --------------------------------------------------------
    H = nx.MultiDiGraph()
    for nid, a in new_nodes.items():
        H.add_node(nid, **a)

    seen_edges: set[tuple[str, str, str]] = set()
    for u, v, edata in G.edges(data=True):
        u, v = str(u), str(v)
        if u not in id_map or v not in id_map:
            stats["edges_dropped"] += 1
            continue
        nu, nv = id_map[u], id_map[v]
        etype = str(edata.get("type", "RELATED_TO"))
        if nu == nv:
            stats["edges_selfloop_dropped"] += 1
            continue
        key = (nu, nv, etype)
        if key in seen_edges:
            stats["edges_deduped"] += 1
            continue
        seen_edges.add(key)
        H.add_edge(nu, nv, type=etype)

    nx.write_graphml(H, DST_GRAPH)

    # --- report ---------------------------------------------------------------
    src_by = Counter(str(a.get("label", "?")) for _, a in G.nodes(data=True))
    dst_by = Counter(str(a.get("label", "?")) for _, a in H.nodes(data=True))
    mv = Counter(str(a.get("movement", "?")) for _, a in H.nodes(data=True))
    print(f"Source: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"Output: {H.number_of_nodes()} nodes / {H.number_of_edges()} edges -> {DST_GRAPH.relative_to(PROJECT_ROOT)}")
    print("\nStats:", dict(stats))
    print("\nNodes per label (src -> dst):")
    for lbl in sorted(set(src_by) | set(dst_by)):
        print(f"  {lbl:18} {src_by.get(lbl,0):4} -> {dst_by.get(lbl,0):4}")
    print("\nNodes per movement tag:", dict(mv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
