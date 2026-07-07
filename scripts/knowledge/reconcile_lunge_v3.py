"""Post-extraction reconcile for the multi-movement KG (sports_kg_v3.graphml).

Cleans up the first real Lunge extraction (docs/movement-kg-expansion-plan.md step 3):
  1. Repair field-swapped edges — gemini sometimes emits an Edge with `target` and
     `type` swapped, so the relationship name became a bogus *node* (`CAUSED_BY`,
     `OCCURS_IN_PHASE`, ...) and the real target name became the edge `type`
     (`WEAK_HIP_ABDUCTORS`). These are deterministically recoverable (swap them back);
     several are the `Valgus --CAUSED_BY--> Weak Hip Abductors` cross-movement bridge.
  2. Fix the pre-existing Squat malformed edges (HAS_FAULT_PHASE typo; backwards
     QUALITYDIMENSION-as-type edges).
  3. Collapse fragmented shared nodes into their canonical vocab entry (the hip-weakness
     cluster -> Weak Hip Abductors; the joint-angle/ROM + trunk-position long tail ->
     Range Of Motion / Alignment). Conservative, explicit maps only.
  4. Tag orphaned movement=None nodes; drop the now-orphaned edge-type-string junk nodes.
  5. Persist the new merges into shared_vocab_v1.json aliases so future extractions
     (Push-up, Row) reuse the canonical shared ids instead of re-fragmenting.

Run from repo root:  python scripts/knowledge/reconcile_lunge_v3.py
Add --dry-run to print the plan without writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from src.knowledge.kg_schema import resolve_node_id

GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
VOCAB_FILE = PROJECT_ROOT / "data" / "kg" / "shared_vocab_v1.json"

EDGE_TYPES = {
    "HAS_PHASE", "HAS_FAULT", "OCCURS_IN_PHASE", "INDICATED_BY",
    "CAUSED_BY", "INCREASES_RISK_OF", "CORRECTED_BY", "AFFECTS_QUALITY",
}

# For repairing a swapped edge: the (recovered) edge type implies the target's label.
TYPE_TARGET_LABEL = {
    "HAS_PHASE": "Phase",
    "HAS_FAULT": "Fault",
    "OCCURS_IN_PHASE": "Phase",
    "INDICATED_BY": "EvidenceSignal",
    "CAUSED_BY": "Cause",
    "INCREASES_RISK_OF": "Risk",
    "CORRECTED_BY": "Cue",
    "AFFECTS_QUALITY": "QualityDimension",
}

# Generic "weak hip" synonyms -> canonical pivot. Biomechanically-distinct hip causes
# (external/internal rotators, flexors, adductors, mobility) are deliberately NOT merged.
CAUSE_MERGE = {
    "Weak Hip Strength": "Weak Hip Abductors",
    "Hip Weakness": "Weak Hip Abductors",
    "Hip Strength Deficit": "Weak Hip Abductors",
    "Hip Strength": "Weak Hip Abductors",
    "Underdeveloped Hip Musculature": "Weak Hip Abductors",
    "Gluteal Strength": "Weak Hip Abductors",
}

# Joint-angle / ROM measures -> Range Of Motion; trunk/stance -> Alignment; casing dups.
# Kinetic measures (moments, velocity, impulse) are intentionally left alone — they are
# not range-of-motion and would be wrong to fold into it.
QD_MERGE = {
    "Ankle Range Of Motion": "Range Of Motion",
    "Hip Range Of Motion": "Range Of Motion",
    "Knee Range Of Motion": "Range Of Motion",
    "Pelvic Range Of Motion": "Range Of Motion",
    "Sagittal Plane Range Of Motion": "Range Of Motion",
    "High Range Of Motion": "Range Of Motion",
    "Reduced Range Of Motion": "Range Of Motion",
    "Hip Abduction Range Of Motion": "Range Of Motion",
    "Ankle Dorsiflexion ROM": "Range Of Motion",
    "Ankle Dorsiflexion Range": "Range Of Motion",
    "Joint Angle": "Range Of Motion",
    "Hip Joint Angle": "Range Of Motion",
    "Knee Joint Angle": "Range Of Motion",
    "Ankle Joint Angle": "Range Of Motion",
    "Peak Joint Angles": "Range Of Motion",
    "Straight Trunk Position": "Alignment",
    "Vertical Trunk Position": "Alignment",
    "Neutral Upright Stance": "Alignment",
    "Trunk Position": "Alignment",
    "Lumbar Spine Position": "Alignment",
    "High-bar Position": "High Bar Position",
    "Low-bar Position": "Low Bar Position",
}


def denorm(upper: str) -> str:
    """WEAK_HIP_ABDUCTORS -> 'Weak Hip Abductors'."""
    return upper.replace("_", " ").title()


def load_graph() -> nx.MultiDiGraph:
    G = nx.read_graphml(GRAPH_FILE)
    if not G.is_multigraph():
        G = nx.MultiDiGraph(G)
    return G


def ensure_node(G, node_id, label, name, movement):
    if not G.has_node(node_id):
        G.add_node(node_id, label=label, name=name, movement=movement)


def add_edge_dedup(G, u, v, etype):
    if u == v:
        return False
    data = G.get_edge_data(u, v)
    if data:
        for _, ed in data.items():
            if isinstance(ed, dict) and ed.get("type") == etype:
                return False
    G.add_edge(u, v, type=etype)
    return True


def repair_swapped_edges(G, log):
    """Repair edges whose type is non-canonical. Two cases:
    (A) target node is itself an edge-type string  -> field swap, recover.
    (B) target is a real node, type is a typo/label -> fix in place / drop.
    """
    to_remove = []
    to_add = []  # (u, v, type)
    for u, v, key, data in list(G.edges(keys=True, data=True)):
        etype = data.get("type")
        if etype in EDGE_TYPES:
            continue
        if v in EDGE_TYPES:
            # (A) swap: real type is v, real target name is the current `type`.
            new_type = v
            tgt_label = TYPE_TARGET_LABEL[new_type]
            tgt_name = denorm(etype)
            src_mv = str(G.nodes[u].get("movement", "Squat"))
            mv_for_resolve = "Squat" if src_mv == "shared" else src_mv
            tgt_id, attrs = resolve_node_id(tgt_name, tgt_label, mv_for_resolve)
            ensure_node(G, tgt_id, attrs["label"], attrs["name"], attrs["movement"])
            to_remove.append((u, v, key))
            to_add.append((u, tgt_id, new_type))
            log.append(f"  [swap] ({u}) --{etype}--> {v}   =>   ({u}) --{new_type}--> {tgt_id}")
        else:
            # (B) pre-existing malformed types.
            if etype == "HAS_FAULT_PHASE":
                to_remove.append((u, v, key))
                to_add.append((u, v, "HAS_FAULT"))
                log.append(f"  [retype] ({u}) --HAS_FAULT_PHASE--> {v}   =>   HAS_FAULT")
            elif etype == "QUALITYDIMENSION":
                # backwards: <QualityDimension> --QUALITYDIMENSION--> <Action>
                # intended: <Action> --AFFECTS_QUALITY--> <QualityDimension>
                to_remove.append((u, v, key))
                to_add.append((v, u, "AFFECTS_QUALITY"))
                log.append(f"  [reverse] ({u}) --QUALITYDIMENSION--> {v}   =>   ({v}) --AFFECTS_QUALITY--> {u}")
            else:
                log.append(f"  [SKIP unknown malformed] ({u}) --{etype}--> {v}")
    for u, v, key in to_remove:
        G.remove_edge(u, v, key)
    for u, v, etype in to_add:
        add_edge_dedup(G, u, v, etype)
    return len(to_remove)


def merge_nodes(G, merge_map, log, tag):
    """Redirect all edges from each `old` node onto `new`, then drop `old`."""
    merged = 0
    for old, new in merge_map.items():
        if not G.has_node(old):
            continue
        if not G.has_node(new):
            # Promote: rename old -> new by keeping old's attrs but canonical id.
            attrs = dict(G.nodes[old])
            attrs["name"] = new
            G.add_node(new, **attrs)
        for _, tgt, ed in list(G.out_edges(old, data=True)):
            add_edge_dedup(G, new, tgt, ed.get("type"))
        for src, _, ed in list(G.in_edges(old, data=True)):
            add_edge_dedup(G, src, new, ed.get("type"))
        G.remove_node(old)
        merged += 1
        log.append(f"  [{tag}] {old!r} -> {new!r}")
    return merged


def tag_orphan_movement(G, log):
    """movement=None nodes that are real concepts get tagged shared/Cause;
    edge-type-string junk nodes that are now isolated get dropped."""
    fixed, dropped = 0, 0
    for n in list(G.nodes()):
        d = G.nodes[n]
        mv = str(d.get("movement"))
        if mv not in ("None", ""):
            continue
        if str(n) in EDGE_TYPES:
            if G.degree(n) == 0:
                G.remove_node(n)
                dropped += 1
                log.append(f"  [drop junk] {n!r}")
            else:
                log.append(f"  [WARN junk still linked] {n!r} deg={G.degree(n)}")
        else:
            d["movement"] = "shared"
            if not d.get("label") or str(d.get("label")) == "None":
                d["label"] = "Cause"
            if not d.get("name"):
                d["name"] = str(n)
            fixed += 1
            log.append(f"  [tag shared] {n!r} -> movement=shared label={d['label']}")
    return fixed, dropped


def update_vocab(dry_run, log):
    vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    aliases = vocab.setdefault("aliases", {})
    added = 0
    for label, mm in (("Cause", CAUSE_MERGE), ("QualityDimension", QD_MERGE)):
        bucket = aliases.setdefault(label, {})
        for raw, canon in mm.items():
            if bucket.get(raw) != canon:
                bucket[raw] = canon
                added += 1
    log.append(f"  [vocab] +{added} alias entries into shared_vocab_v1.json")
    if not dry_run:
        VOCAB_FILE.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    return added


def dedup_and_clean(G):
    # remove self loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # dedup parallel edges with identical type
    seen = set()
    for u, v, key, data in list(G.edges(keys=True, data=True)):
        sig = (u, v, data.get("type"))
        if sig in seen:
            G.remove_edge(u, v, key)
        else:
            seen.add(sig)


def verify(G):
    print("\n=== VERIFY: cross-movement hip-abductor pivot ===")
    pivot = "Weak Hip Abductors"
    if not G.has_node(pivot):
        print("  MISSING pivot node!")
        return
    movers = {}
    for src, _, ed in G.in_edges(pivot, data=True):
        mv = str(G.nodes[src].get("movement"))
        movers.setdefault(mv, []).append((src, ed.get("type")))
    for mv, items in sorted(movers.items()):
        print(f"  movement={mv}: {len(items)} incoming")
        for src, t in items:
            print(f"      {src}  --{t}-->  {pivot}")
    squat_reach = any(mv == "Squat" for mv in movers)
    lunge_reach = any(mv == "Lunge" for mv in movers)
    print(f"  >> Squat reaches pivot: {squat_reach} | Lunge reaches pivot: {lunge_reach} "
          f"| CROSS-MOVEMENT BRIDGE: {squat_reach and lunge_reach}")


def main():
    ap = argparse.ArgumentParser(description="Reconcile the multi-movement KG after Lunge extraction.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan without writing the graph/vocab.")
    args = ap.parse_args()

    G = load_graph()
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f"Loaded {GRAPH_FILE.name}: {n0} nodes, {e0} edges")

    log: list[str] = []
    print("\n--- 1. repair malformed/swapped edges ---")
    n_rep = repair_swapped_edges(G, log)
    print("\n".join(l for l in log if l.strip().startswith(("[swap", "[retype", "[reverse", "[SKIP")) or "[" in l))
    log.clear()

    print("\n--- 2. merge fragmented shared Cause nodes ---")
    n_cause = merge_nodes(G, CAUSE_MERGE, log, "cause-merge")
    print("\n".join(log)); log.clear()

    print("\n--- 3. merge fragmented shared QualityDimension nodes ---")
    n_qd = merge_nodes(G, QD_MERGE, log, "qd-merge")
    print("\n".join(log)); log.clear()

    print("\n--- 4. tag/drop movement=None nodes ---")
    fixed, dropped = tag_orphan_movement(G, log)
    print("\n".join(log)); log.clear()

    print("\n--- 5. update shared_vocab_v1.json ---")
    update_vocab(args.dry_run, log)
    print("\n".join(log)); log.clear()

    dedup_and_clean(G)

    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    print(f"\nResult: {n0}->{n1} nodes ({n1-n0:+d}), {e0}->{e1} edges ({e1-e0:+d})")
    print(f"  repaired edges: {n_rep} | cause merges: {n_cause} | qd merges: {n_qd} | "
          f"orphans tagged: {fixed} | junk dropped: {dropped}")

    verify(G)

    if args.dry_run:
        print("\n[dry-run] graph NOT written.")
    else:
        nx.write_graphml(G, GRAPH_FILE)
        print(f"\nWrote {GRAPH_FILE}")


if __name__ == "__main__":
    main()
