"""Post-extraction reconcile for the multi-movement KG (sports_kg_v3.graphml) — Row
(flagship #5, the final flagship; docs/movement-kg-expansion-plan.md step 3).

Companion to reconcile_{lunge,pushup,ohp}_v3.py. Row is horizontal PULLING — the
counterpart to the pressing movements (Push-up, Overhead Press). Its defining scapular
action is RETRACTION (rhomboids / middle + lower trapezius), the opposite of the
protraction / upward-rotation (serratus / upper trapezius) that defines the presses.

Design decisions (advisor discipline — merge only on CONCEPT-IDENTITY; observe the
bridge, never engineer it):
  * RETRACTION DISCIPLINE: Row's retractor nodes (Rhomboids Activation, Middle Trapezius
    Activation, Scapular Retraction, Weak Middle Trapezius, Weak Posterior Deltoid) are
    KEPT DISTINCT and are NEVER folded onto the presses' Weak Serratus Anterior /
    Serratus Anterior Activation. Row's honest bridge to Push-up + Overhead Press runs
    through the COARSE shared scapular nodes both genuinely share — Scapular Control,
    Scapular Stability, Weak Scapular Stabilizers, Scapular Dyskinesis, Subacromial
    Impingement, Shoulder Stability/Injury/Pain — plus the spine/core link to
    Squat/Lunge (Alignment, Core Stability, Erector Spinae Activation, Maintain Neutral
    Spine, Technique Quality). Pre-reconcile this already yields 27 bridges; the merges
    below only fold near-duplicate fragments into them.
  * SCHEMA CLEANUP unique to this extraction: drop 2 movement=None artifacts
    ('INDICATED_BY' — an edge-type that leaked in as a node; 'W-shape' — a vague orphan)
    and relabel the 5 non-standard 'Technique' nodes (execution variants like 'Elbows
    Close To Torso') to the canonical scoped label 'Action'.

Run from repo root:  python scripts/knowledge/reconcile_row_v3.py [--dry-run]
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
MOVEMENT = "Row"
OTHER_MOVEMENTS = {"Squat", "Lunge", "Push-up", "Overhead Press"}

EDGE_TYPES = {
    "HAS_PHASE", "HAS_FAULT", "OCCURS_IN_PHASE", "INDICATED_BY",
    "CAUSED_BY", "INCREASES_RISK_OF", "CORRECTED_BY", "AFFECTS_QUALITY",
}

# movement=None artifacts to drop, and the non-standard scoped label to normalise.
DROP_NONE_NODES = ["INDICATED_BY", "W-shape"]
RELABEL = {"Technique": "Action"}

# --- Cause dedup -------------------------------------------------------------
# Distinct muscle weaknesses (Weak Infraspinatus / Posterior Deltoid / Gluteus Maximus)
# and the retraction-specific faults (Limited Scapular Retraction, Scapular Constraint,
# Unconstrained Scapulothoracic Motion) are KEPT SEPARATE.
CAUSE_MERGE = {
    # scapular-dysfunction synonyms -> OHP canonical Scapular Dyskinesis
    "Scapular Muscle Imbalance": "Scapular Dyskinesis",
    "Altered Scapulohumeral Rhythm": "Scapular Dyskinesis",
    # unstable surface -> Push-up canonical
    "Unstable Support Surface": "Surface Instability",
    # bent-over trunk-angle study variables -> one node
    "30 Degree Trunk Inclination": "Body Inclination",
    "Angle Between Body And Ground": "Body Inclination",
}

# --- QualityDimension dedup --------------------------------------------------
# EMG activation/excitation/strengthening casing families -> the *Activation form.
# Retractor muscles stay as their OWN activation nodes (not folded into serratus/
# generic). Spine-neutral -> Alignment; spine/core stability -> Core Stability;
# scapular adduction == retraction.
QD_MERGE = {
    # middle trapezius (a primary retractor) — casing family kept as its own node
    "Middle Trapezius": "Middle Trapezius Activation",
    "Middle Trapezius Excitation": "Middle Trapezius Activation",
    "Middle Trapezius Strengthening": "Middle Trapezius Activation",
    "Trapezius Transversus Excitation": "Middle Trapezius Activation",
    # posterior deltoid casing family
    "Posterior Deltoid Excitation": "Posterior Deltoid Activation",
    "Posterior Deltoid Strengthening": "Posterior Deltoid Activation",
    "Rear Deltoid Excitation": "Posterior Deltoid Activation",
    # lateral deltoid == middle deltoid (same muscle head)
    "Lateral Deltoid Activation": "Middle Deltoid Activation",
    # latissimus dorsi casing / regional variants
    "Latissimus Dorsi Excitation": "Latissimus Dorsi Activation",
    "Thoracic Latissimus Dorsi Activation": "Latissimus Dorsi Activation",
    "Lumbopelvic Costal Latissimus Dorsi Activation": "Latissimus Dorsi Activation",
    # biceps / trap strengthening casing pairs
    "Biceps Brachii Excitation": "Biceps Brachii Activation",
    "Upper Trapezius Strengthening": "Upper Trapezius Activation",
    "Lower Trapezius Strengthening": "Lower Trapezius Activation",
    # scapular umbrella -> canonical (kinematic directions kept apart)
    "Sustained Scapular Stabilization": "Scapular Stability",
    "Scapular Adduction Range Of Motion": "Scapular Retraction",
    "Scapular Muscle Performance": "Scapular Control",
    "Free Scapular Motion": "Free Scapular Position",
    # spine / core / ROM / generic
    "Spine Neutral Position": "Alignment",
    "Upright Torso": "Alignment",
    "Spine Stability": "Core Stability",
    "Core Muscle Activation": "Core Stability",
    "Arm Extension": "Range Of Motion",
    "Stable Form": "Stability",
    "Wider Grip": "Grip Type",
}

# --- Risk dedup --------------------------------------------------------------
RISK_MERGE = {
    "Injury": "Injury Risk",
    "Musculoskeletal Injuries": "Injury Risk",
    "Shoulder Disorders": "Shoulder Dysfunction",
    # loading == load: bridges Row<->OHP on the shared mechanical-load node
    "Shoulder Joint Loading": "Shoulder Joint Load",
}

# Register Row's retraction / back canonicals so the later GENERAL movements
# (deadlift, bicep curl, band pull-apart, ...) steer onto them.
NEW_CANONICAL = {
    "Cause": ["Weak Middle Trapezius", "Weak Posterior Deltoid", "Weak Infraspinatus",
              "Limited Scapular Retraction", "Weak Gluteus Maximus", "Body Inclination"],
    "Risk": ["Rotator Cuff Strain"],
    "QualityDimension": ["Scapular Retraction", "Rhomboids Activation",
                         "Middle Trapezius Activation", "Latissimus Dorsi Activation",
                         "Posterior Deltoid Activation", "Rotator Cuff Activation"],
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


def cleanup_schema(G, log):
    """Drop movement=None artifacts and relabel non-standard scoped labels."""
    dropped = 0
    for name in DROP_NONE_NODES:
        if G.has_node(name):
            G.remove_node(name)
            dropped += 1
            log.append(f"  [drop-none] {name!r}")
    relabelled = 0
    for n in list(G.nodes):
        old = str(G.nodes[n].get("label"))
        if old in RELABEL:
            G.nodes[n]["label"] = RELABEL[old]
            relabelled += 1
            log.append(f"  [relabel] {str(G.nodes[n].get('name', n))!r}: {old} -> {RELABEL[old]}")
    return dropped, relabelled


def merge_nodes(G, merge_map, log, tag):
    """Redirect all edges from each `old` SHARED node onto `new`, then drop old.

    Only shared nodes (movement=="shared") are eligible as merge sources/targets, so a
    scoped node whose un-namespaced name collides with a canonical shared name can never
    be consumed (the Push-up reconcile name-collision bug). The merge maps only ever
    refer to shared-layer concepts.
    """
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

    print(f"\n=== VERIFY: cross-movement bridges via {MOVEMENT} ===")
    targets = set()
    for u, v, d in G.edges(data=True):
        if mv(u) == MOVEMENT and mv(v) == "shared":
            targets.add(v)
    bridges = []
    for s in targets:
        others = (set(mv(p) for p in G.predecessors(s)) - {MOVEMENT, "shared"}) & OTHER_MOVEMENTS
        if others:
            bridges.append((lab(s), nm(s), sorted(others)))
    for l, n, o in sorted(bridges):
        print(f"  BRIDGE [{l}] {n!r}  <- also {o}")
    print(f"  >> {MOVEMENT} forms {len(bridges)} cross-movement bridges "
          f"(to {sorted(OTHER_MOVEMENTS)} via shared pivots).")

    to_press = sorted({n for _, n, o in bridges if {"Push-up", "Overhead Press"} & set(o)})
    to_lower = sorted({n for _, n, o in bridges if {"Squat", "Lunge"} & set(o)})
    print(f"  >> shoulder/scapular bridge to presses ({len(to_press)}): {to_press}")
    print(f"  >> spine/core/technique bridge to Squat/Lunge ({len(to_lower)}): {to_lower}")

    # Retraction-side identity must survive as its own shared nodes (never folded to serratus).
    retr = sorted({nm(n) for n in G.nodes if mv(n) == "shared" and any(w in nm(n).lower()
                   for w in ("retract", "rhomboid", "middle trapezius"))})
    ser = [nm(n) for n in G.nodes if mv(n) == "shared" and "serratus" in nm(n).lower()]
    print(f"  >> retraction identity preserved (distinct from serratus): {retr}")
    print(f"  >> serratus nodes (unchanged, protraction side): {sorted(ser)}")


def main():
    ap = argparse.ArgumentParser(description="Reconcile the multi-movement KG after Row extraction.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    G = load_graph()
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    scoped0 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == MOVEMENT)
    none0 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == "None")
    print(f"Loaded {GRAPH_FILE.name}: {n0} nodes, {e0} edges ({MOVEMENT} scoped: {scoped0}, movement=None: {none0})")

    log: list[str] = []
    print("\n--- 0. schema cleanup (drop None artifacts, relabel Technique) ---")
    n_drop, n_rel = cleanup_schema(G, log)
    print("\n".join(log)); log.clear()

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
    scoped1 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == MOVEMENT)
    none1 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == "None")
    print(f"\nResult: {n0}->{n1} nodes ({n1-n0:+d}), {e0}->{e1} edges ({e1-e0:+d})")
    print(f"  drops: {n_drop} | relabels: {n_rel} | cause: {n_cause} | qd: {n_qd} | risk: {n_risk}")
    print(f"  {MOVEMENT} scoped: {scoped0}->{scoped1} "
          f"({'UNCHANGED (guard OK)' if scoped0 == scoped1 else 'CHANGED -- CHECK!'})  | movement=None: {none0}->{none1}")

    verify(G)

    if args.dry_run:
        print("\n[dry-run] graph NOT written.")
    else:
        nx.write_graphml(G, GRAPH_FILE)
        print(f"\nWrote {GRAPH_FILE}")


if __name__ == "__main__":
    main()
