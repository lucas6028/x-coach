"""Post-extraction reconcile for the multi-movement KG (sports_kg_v3.graphml) — Push-up.

Companion to reconcile_lunge_v3.py, for the first *upper-body* movement (docs/
movement-kg-expansion-plan.md step 3). The target/type swap-guard added to
extract_kg.py.normalize_kg did its job — this extraction emitted 0 field-swap junk
nodes, 0 movement=None nodes, 0 non-canonical edges — so unlike the Lunge reconcile
there is nothing to REPAIR. The whole job here is conservative shared-layer
canonicalisation of the +135 new shared nodes.

Design decisions (see the "honest bridge" note in the plan):
  * Push-up shares NO natural lower-body pivot with Squat/Lunge. The honest cross-
    movement bridges it already forms (verified pre-reconcile) are the GENERIC ones:
    Cause 'Poor Neuromuscular Control' and QDs 'Range Of Motion' / 'Stability' /
    'Depth'. This reconcile STRENGTHENS those by folding fragments into them.
  * The rich scapular / serratus / shoulder-impingement layer is Push-up's real
    contribution, but it bridges to Overhead Press + Row, which are NOT extracted yet.
    We therefore DEDUP that cluster into clean canonical nodes (and register them in
    shared_vocab canonical so OHP/Row steer onto them) but DO NOT force-merge any of it
    into something Squat references — that would manufacture a fake bridge and corrupt
    the shared layer. "No shoulder bridge yet" is the expected, correct state.

Run from repo root:  python scripts/knowledge/reconcile_pushup_v3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"
VOCAB_FILE = PROJECT_ROOT / "data" / "kg" / "shared_vocab_v1.json"

EDGE_TYPES = {
    "HAS_PHASE", "HAS_FAULT", "OCCURS_IN_PHASE", "INDICATED_BY",
    "CAUSED_BY", "INCREASES_RISK_OF", "CORRECTED_BY", "AFFECTS_QUALITY",
}

# --- Cause dedup -------------------------------------------------------------
# (a) study-apparatus / surface noise from the EMG-on-Swiss-ball papers -> one node.
# (b) clearly-negative neuromuscular synonyms -> the EXISTING shared bridge cause
#     'Poor Neuromuscular Control' (Squat+Lunge already point at it) -> strengthens it.
# Biomechanically-distinct scapular muscle causes (serratus / upper+lower trap /
# scapular stabilizers / loss of scapular control) are deliberately KEPT SEPARATE, same
# way the Lunge reconcile kept the distinct hip muscles apart.
CAUSE_MERGE = {
    "Swiss Ball": "Surface Instability",
    "Swiss Ball Under Feet": "Surface Instability",
    "Swiss Ball Under Hands": "Surface Instability",
    "Unstable Surface": "Surface Instability",
    "Unstable Condition": "Surface Instability",
    "Neuromuscular Dysfunction": "Poor Neuromuscular Control",
    "Poor Muscular Cooperation": "Poor Neuromuscular Control",
}

# --- QualityDimension dedup --------------------------------------------------
# Same coarsening philosophy as the Lunge reconcile: joint-angle/ROM measures ->
# Range Of Motion; posture/neutral-position -> Alignment; GENERIC (non-scapular,
# non-shoulder) stability -> Stability. Hand-placement casing/probe artifacts ->
# Hand Position. Core QD casing dups -> a single Core Stability.
# KEPT DISTINCT: every scapular/shoulder/serratus quality (upper-body signal, future
# OHP/Row pivots) and mis-labelled muscle names (Pectoralis Major, Triceps Brachii, ...).
QD_MERGE = {
    # ROM / joint-angle family
    "Elbow Angle": "Range Of Motion",
    "Elbow Extension": "Range Of Motion",
    "Elbow Flexion": "Range Of Motion",
    "Shoulder Flexion": "Range Of Motion",
    "Arm Elevation": "Range Of Motion",
    "Full Repetition": "Range Of Motion",
    # posture / neutral alignment
    "Body Alignment": "Alignment",
    "Spine Alignment": "Alignment",
    "Neck Alignment": "Alignment",
    "Neutral Spine": "Alignment",
    "Neutral Pelvis": "Alignment",
    "Neutral Position": "Alignment",
    "Back Straight": "Alignment",
    # generic (non-scapular/shoulder) stability
    "Elbow Stability": "Stability",
    "Forearm Stability": "Stability",
    "Wrist Stability": "Stability",
    "Joint Stabilization": "Stability",
    "Full Body Tension": "Stability",
    "Plank Position Stability": "Stability",
    "Support Surface Stability": "Stability",
    "Knee Extensor Stability": "Stability",
    "Hip/spine Flexor Stability": "Stability",
    # core QD casing dups -> single node
    "Core Strength": "Core Stability",
    "Core Integration": "Core Stability",
    # hand-placement casing / experiment-probe artifacts
    "Hand Position P1": "Hand Position",
    "Hand Position P2": "Hand Position",
    "Hand Position P3": "Hand Position",
    "Hand Placement": "Hand Position",
    "Hand Orientation": "Hand Position",
    # scapular / shoulder INTERNAL dedup (kept out of generic Stability on purpose)
    "Serratus Anterior Activity": "Serratus Anterior Activation",
    "SA Activation": "Serratus Anterior Activation",
    "Shoulder Joint Stability": "Shoulder Stability",
    "Shoulder Stabilization": "Shoulder Stability",
    "Scapular Positioning": "Scapular Control",
    "LT Activation": "Scapular Stabilizer Activation",
}

# --- Risk dedup --------------------------------------------------------------
# Subacromial/shoulder-impingement is one clinical entity fragmented 5 ways -> collapse.
# Clear shoulder synonyms collapse; distinct entities (instability, pathology,
# dysfunction, elbow load vs elbow injury, spine/carpal-tunnel) are kept apart.
RISK_MERGE = {
    "Shoulder Impingement": "Subacromial Impingement",
    "Shoulder Impingement Syndrome": "Subacromial Impingement",
    "Subacromial Impingement Syndrome": "Subacromial Impingement",
    "Subacromial Soft Tissue Impingement": "Subacromial Impingement",
    "Shoulder Joint Injury": "Shoulder Injury",
    "Glenohumeral Joint Pain": "Shoulder Pain",
}

# New canonical shared nodes to register so OHP/Row extractions steer onto them
# instead of re-fragmenting the upper-body layer.
NEW_CANONICAL = {
    "Cause": ["Weak Serratus Anterior", "Weak Scapular Stabilizers", "Loss Of Scapular Control",
              "Weak Lower Trapezius", "Weak Upper Trapezius", "Surface Instability"],
    "Risk": ["Subacromial Impingement", "Shoulder Instability", "Shoulder Injury",
             "Shoulder Pain", "Elbow Joint Load"],
    "QualityDimension": ["Scapular Stability", "Scapular Control", "Serratus Anterior Activation",
                         "Shoulder Stability", "Core Stability"],
}


def load_graph() -> nx.MultiDiGraph:
    G = nx.read_graphml(GRAPH_FILE)
    if not G.is_multigraph():
        G = nx.MultiDiGraph(G)
    return G


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


def merge_nodes(G, merge_map, log, tag):
    """Redirect all edges from each `old` node onto `new` (by NAME attr), then drop old.

    Only SHARED nodes are eligible as merge sources/targets. Scoped nodes carry an
    un-namespaced `name` (id ``Push-up:Hand Position`` but name ``Hand Position``), so
    matching on `name` without this guard could pick a scoped node whose concept name
    collides with a canonical shared name, consume it, and redirect its edges across a
    label boundary. The merge maps only ever refer to shared-layer concepts, so
    restricting the index to ``movement=="shared"`` is both correct and the fix.
    """
    # index shared nodes by their display name
    name_to_id = {}
    for n in G.nodes:
        if str(G.nodes[n].get("movement")) != "shared":
            continue
        name_to_id.setdefault(str(G.nodes[n].get("name", n)), n)
    merged = 0
    for old_name, new_name in merge_map.items():
        old = name_to_id.get(old_name)
        if old is None or not G.has_node(old):
            continue
        new = name_to_id.get(new_name)
        if new is None or not G.has_node(new):
            # promote: keep old's attrs but rename to canonical
            attrs = dict(G.nodes[old])
            attrs["name"] = new_name
            new = new_name
            G.add_node(new, **attrs)
            name_to_id[new_name] = new
        if old == new:
            continue
        for _, tgt, ed in list(G.out_edges(old, data=True)):
            add_edge_dedup(G, new, tgt, ed.get("type"))
        for src, _, ed in list(G.in_edges(old, data=True)):
            add_edge_dedup(G, src, new, ed.get("type"))
        G.remove_node(old)
        merged += 1
        log.append(f"  [{tag}] {old_name!r} -> {new_name!r}")
    return merged


def dedup_and_clean(G):
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    seen = set()
    for u, v, key, data in list(G.edges(keys=True, data=True)):
        sig = (u, v, data.get("type"))
        if sig in seen:
            G.remove_edge(u, v, key)
        else:
            seen.add(sig)


def update_vocab(dry_run, log):
    vocab = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    aliases = vocab.setdefault("aliases", {})
    canonical = vocab.setdefault("canonical", {})
    added_a = added_c = 0
    for label, mm in (("Cause", CAUSE_MERGE), ("QualityDimension", QD_MERGE), ("Risk", RISK_MERGE)):
        bucket = aliases.setdefault(label, {})
        for raw, canon in mm.items():
            if bucket.get(raw) != canon:
                bucket[raw] = canon
                added_a += 1
    for label, names in NEW_CANONICAL.items():
        clist = canonical.setdefault(label, [])
        for nm in names:
            if nm not in clist:
                clist.append(nm)
                added_c += 1
    log.append(f"  [vocab] +{added_a} aliases, +{added_c} canonical names into shared_vocab_v1.json")
    if not dry_run:
        VOCAB_FILE.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    return added_a, added_c


def verify(G):
    def nm(n): return str(G.nodes[n].get("name", n))
    def mv(n): return str(G.nodes[n].get("movement"))
    def lab(n): return str(G.nodes[n].get("label"))

    print("\n=== VERIFY: honest cross-movement bridges via Push-up ===")
    hit = {}
    for u, v, d in G.edges(data=True):
        if mv(u) == "Push-up" and mv(v) == "shared":
            hit.setdefault(v, set())
    bridges = []
    for s in hit:
        others = set(mv(p) for p in G.predecessors(s)) - {"Push-up", "shared"}
        if others & {"Squat", "Lunge"}:
            bridges.append((lab(s), nm(s), sorted(others)))
    for l, n, o in sorted(bridges):
        print(f"  BRIDGE [{l}] {n!r}  <- also {o}")
    print(f"  >> Push-up forms {len(bridges)} cross-movement bridges "
          f"(to Squat/Lunge via shared pivots).")

    # Report the upper-body cluster that legitimately awaits OHP + Row.
    scap = [nm(n) for n in G.nodes
            if mv(n) == "shared" and any(w in nm(n).lower()
            for w in ("scapul", "serratus", "impinge", "rotator", "shoulder", "trapezius"))]
    print(f"  >> {len(scap)} scapular/shoulder shared pivots created — these await "
          f"Overhead Press + Row (expected: no bridge yet).")


def main():
    ap = argparse.ArgumentParser(description="Reconcile the multi-movement KG after Push-up extraction.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    G = load_graph()
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f"Loaded {GRAPH_FILE.name}: {n0} nodes, {e0} edges")

    log: list[str] = []
    print("\n--- 1. merge fragmented shared Cause nodes ---")
    n_cause = merge_nodes(G, CAUSE_MERGE, log, "cause-merge")
    print("\n".join(log)); log.clear()

    print("\n--- 2. merge fragmented shared QualityDimension nodes ---")
    n_qd = merge_nodes(G, QD_MERGE, log, "qd-merge")
    print("\n".join(log)); log.clear()

    print("\n--- 3. merge fragmented shared Risk nodes ---")
    n_risk = merge_nodes(G, RISK_MERGE, log, "risk-merge")
    print("\n".join(log)); log.clear()

    print("\n--- 4. update shared_vocab_v1.json ---")
    update_vocab(args.dry_run, log)
    print("\n".join(log)); log.clear()

    dedup_and_clean(G)

    n1, e1 = G.number_of_nodes(), G.number_of_edges()
    print(f"\nResult: {n0}->{n1} nodes ({n1-n0:+d}), {e0}->{e1} edges ({e1-e0:+d})")
    print(f"  cause merges: {n_cause} | qd merges: {n_qd} | risk merges: {n_risk}")

    verify(G)

    if args.dry_run:
        print("\n[dry-run] graph NOT written.")
    else:
        nx.write_graphml(G, GRAPH_FILE)
        print(f"\nWrote {GRAPH_FILE}")


if __name__ == "__main__":
    main()
