"""Post-extraction reconcile for the multi-movement KG (sports_kg_v3.graphml) —
Overhead Press (flagship #4, docs/movement-kg-expansion-plan.md step 3).

Companion to reconcile_lunge_v3.py / reconcile_pushup_v3.py. The extract_kg.py
swap-guard again held (0 field-swap junk, 0 movement=None, 0 non-canonical edges),
so — as with Push-up — there is nothing to REPAIR; the whole job is conservative
shared-layer canonicalisation of the +236 new shared nodes.

Design decisions (advisor discipline — merge only on CONCEPT-IDENTITY, never to
engineer a bridge):
  * The cross-movement bridge OHP forms is OBSERVED, not manufactured. Pre-reconcile,
    vocab-steering already lands OHP on 26 shared pivots reached by other movements —
    the shoulder/scapular cluster Push-up seeded (Weak Serratus Anterior, Serratus
    Anterior Activation, Scapular Control, Scapular Upward/External/Posterior rotation,
    Shoulder Stability/Strength, Subacromial Impingement, Shoulder Injury/Pain) plus the
    generics (Poor Neuromuscular Control, Weak Core Stability, Alignment, Range Of
    Motion, Stability, Lumbar Spine Injury). This reconcile only STRENGTHENS those by
    folding near-duplicate fragments into them.
  * DISTINCT scapular KINEMATIC directions (anterior/posterior tilt, internal/external/
    upward rotation, scapulohumeral rhythm) are KEPT SEPARATE — real biomechanics, not
    casing noise. Only exact duplicates (External Scapular Rotation == Scapular External
    Rotation) and vague umbrella terms (Scapular Movement/Orientation -> Scapular
    Control) are folded.
  * OHP's rich extraction seeded the RETRACTION-side nodes that Row will legitimately
    bridge to (Scapular Retraction, Weak Middle Trapezius, Weak Latissimus Dorsi,
    Erector Spinae Activation). We register the new shoulder/back/scapular canonicals so
    Row steers onto them instead of re-fragmenting the upper-body layer.

Run from repo root:  python scripts/knowledge/reconcile_ohp_v3.py [--dry-run]
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
MOVEMENT = "Overhead Press"
OTHER_MOVEMENTS = {"Squat", "Lunge", "Push-up"}

EDGE_TYPES = {
    "HAS_PHASE", "HAS_FAULT", "OCCURS_IN_PHASE", "INDICATED_BY",
    "CAUSED_BY", "INCREASES_RISK_OF", "CORRECTED_BY", "AFFECTS_QUALITY",
}

# --- Cause dedup -------------------------------------------------------------
# Fold clear concept-identity duplicates only. Distinct anatomical findings
# (Hooked Acromion, Idiopathic Scoliosis, Lumbar Lordosis, Tight Pectoralis Major,
# Tight Posterior Shoulder Capsule, ...) are KEPT SEPARATE.
CAUSE_MERGE = {
    # impingement structural near-duplicates
    "Thickened Coracoacromial Ligaments": "Coracoacromial Ligament Thickening",
    "Acromion Configuration": "Acromion Shape",
    "Acromion Structural Abnormalities": "Acromion Shape",
    "Bony Spurs On Acromion": "Osteophytes",
    "Osteophytes In Acromioclavicular Joint": "Osteophytes",
    # rotator-cuff weakness synonyms
    "Decreased Rotator Cuff Strength": "Weak Rotator Cuff",
    "Weakness In Rotator Cuff Muscles": "Weak Rotator Cuff",
    # fatigue -> existing shared Muscle Fatigue (bridges to Push-up/Squat)
    "General Muscle Fatigue": "Muscle Fatigue",
    "Local Muscle Fatigue": "Muscle Fatigue",
    # scapular dyskinesis synonym
    "Altered Scapular Kinematics": "Scapular Dyskinesis",
    # posture
    "Slouching Position": "Poor Posture",
    "Greater Thoracic Kyphosis": "Thoracic Kyphosis",
    "Excessive Thoracic Kyphosis": "Thoracic Kyphosis",
    "Kyphosis": "Thoracic Kyphosis",
    # overuse phrasing
    "Repetitive Overhead Tasks": "Repetitive Overhead Activities",
    # center-of-mass modelling artifacts
    "Differences In Center Of Mass": "Center Of Gravity Location",
    "Different Center Of Gravity": "Center Of Gravity Location",
}

# --- QualityDimension dedup --------------------------------------------------
# Established coarsening (as Lunge/Push-up): joint-angle/ROM -> Range Of Motion;
# generic (non-scapular/shoulder) stability -> Stability; core* -> Core Stability.
# EMG activation-vs-excitation casing pairs -> the *Activation form. Shoulder/scapular
# umbrella terms -> Push-up canonicals. DISTINCT scapular kinematic directions kept.
QD_MERGE = {
    # ROM / joint-angle / degree-condition family
    "Shoulder Range Of Motion": "Range Of Motion",
    "Ideal Shoulder Range Of Motion": "Range Of Motion",
    "Elbow Angle 90 Degrees": "Range Of Motion",
    "Elbow Angle 135 Degrees": "Range Of Motion",
    "Elbow Angle 180 Degrees": "Range Of Motion",
    "Full Arm Extension": "Range Of Motion",
    # generic (non-scapular/shoulder) stability
    "Greater Stabilization": "Stability",
    "Muscle Stabilization": "Stability",
    "Stabilization Function": "Stability",
    "Load Stability": "Stability",
    "Stable Load": "Stability",
    "Lower Body Stability": "Stability",
    # core
    "Core Endurance": "Core Stability",
    "Core Engagement": "Core Stability",
    "Core Stabilization": "Core Stability",
    "Normal Trunk Stability": "Core Stability",
    # shoulder/scapular umbrella -> Push-up canonicals (kinematic directions kept apart)
    "Scapular Stabilization": "Scapular Stability",
    "Glenohumeral Joint Stability": "Shoulder Stability",
    "External Scapular Rotation": "Scapular External Rotation",
    "Scapular Movement": "Scapular Control",
    "Scapular Orientation": "Scapular Control",
    # deltoid EMG casing pairs
    "Deltoid": "Deltoid Activation",
    "Deltoid Excitation": "Deltoid Activation",
    "Medial Deltoid Excitation": "Deltoid Activation",
    "Anterior Deltoid Excitation": "Anterior Deltoid Activation",
    # trapezius EMG casing pairs
    "Upper Trapezius": "Upper Trapezius Activation",
    "Upper Trapezius Excitation": "Upper Trapezius Activation",
    "Lower Trapezius": "Lower Trapezius Activation",
    # triceps EMG casing pair
    "Triceps Brachii Excitation": "Triceps Brachii Activation",
    # velocity / trajectory
    "Bar Speed": "Barbell Velocity",
    "Straight Barbell Trajectory": "Barbell Trajectory",
    "Straight Line Press": "Barbell Trajectory",
    # power
    "Upper Body Power": "Power Output",
    "High Power Output": "Power Output",
    # military-press variants mis-labelled as a QualityDimension
    "Front Barbell Military Press": "Overhead Press Technique",
    "Back Barbell Military Press": "Overhead Press Technique",
}

# --- Risk dedup --------------------------------------------------------------
# Collapse clear synonyms of one clinical entity; keep genuinely distinct disorders
# (Adhesive Capsulitis, Complex Regional Pain Syndrome, Cuff Tear Arthropathy,
# Anterior Shoulder Instability, Neck Pain, Trunk Injury) apart.
RISK_MERGE = {
    "Subacromial Compression": "Subacromial Impingement",
    "Subacromial Pain Syndrome": "Subacromial Impingement",
    "Reduced Subacromial Space": "Subacromial Impingement",
    "Shoulder Injuries": "Shoulder Injury",
    "Shoulder Discomfort": "Shoulder Pain",
    "Shoulder Disorder": "Shoulder Dysfunction",
    "Shoulder Conditions": "Shoulder Dysfunction",
    "Acute Injuries": "Acute Injury",
    "Injuries": "Injury Risk",
    # NOT merged: 'Shoulder Joint Load' (a mechanical stressor, not the injury outcome)
    # and 'Rotator Cuff Disease' (a vague umbrella, distinct from the specific
    # Degeneration/Tear/RCRSP entities) are kept as their own nodes.
}

# New canonical shared nodes to register so the Row extraction steers onto them
# instead of re-fragmenting the upper-body / back layer. Row's honest bridge to OHP +
# Push-up runs through the scapular-stability / retraction / spine nodes below.
NEW_CANONICAL = {
    "Cause": ["Weak Rotator Cuff", "Scapular Dyskinesis", "Thoracic Kyphosis",
              "Forward Head Posture", "Weak Lower Trapezius", "Weak Upper Trapezius",
              "Weak Middle Trapezius", "Weak External Rotators", "Weak Latissimus Dorsi"],
    "Risk": ["Rotator Cuff Tear", "Rotator Cuff Related Shoulder Pain",
             "Rotator Cuff Degeneration", "Shoulder Dysfunction"],
    "QualityDimension": ["Deltoid Activation", "Upper Trapezius Activation",
                         "Lower Trapezius Activation", "Scapular Retraction",
                         "Erector Spinae Activation", "Thoracic Extension"],
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
    """Redirect all edges from each `old` shared node onto `new`, then drop old.

    Only SHARED nodes are eligible as merge sources/targets (movement=="shared").
    Scoped nodes carry an un-namespaced `name`, so matching on `name` without this
    guard could consume a scoped node whose concept name collides with a canonical
    shared name and redirect its edges across a label boundary (the Push-up
    reconcile bug). The merge maps only ever refer to shared-layer concepts.
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

    # How many of those bridges are the shoulder/scapular cluster Push-up seeded?
    shoulder = [n for _, n, o in bridges
                if "Push-up" in o and any(w in n.lower()
                for w in ("scapul", "serratus", "shoulder", "impinge", "trapezius", "rotator", "deltoid"))]
    print(f"  >> {len(shoulder)} of them are the shoulder/scapular cluster now shared "
          f"OHP<->Push-up: {sorted(shoulder)}")

    # Retraction-side pivots seeded for Row (expect: OHP-only until Row lands).
    retr = [nm(n) for n in G.nodes if mv(n) == "shared" and any(w in nm(n).lower()
            for w in ("retract", "middle trapezius", "latissimus", "erector spinae", "rhomboid"))]
    print(f"  >> {len(retr)} retraction/back pivots seeded for Row: {sorted(retr)}")


def main():
    ap = argparse.ArgumentParser(description="Reconcile the multi-movement KG after Overhead Press extraction.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    G = load_graph()
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    scoped0 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == MOVEMENT)
    print(f"Loaded {GRAPH_FILE.name}: {n0} nodes, {e0} edges ({MOVEMENT} scoped: {scoped0})")

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
    scoped1 = sum(1 for n in G.nodes if str(G.nodes[n].get("movement")) == MOVEMENT)
    print(f"\nResult: {n0}->{n1} nodes ({n1-n0:+d}), {e0}->{e1} edges ({e1-e0:+d})")
    print(f"  cause merges: {n_cause} | qd merges: {n_qd} | risk merges: {n_risk}")
    print(f"  {MOVEMENT} scoped: {scoped0}->{scoped1} "
          f"({'UNCHANGED (guard OK)' if scoped0 == scoped1 else 'CHANGED -- BUG!'})")

    verify(G)

    if args.dry_run:
        print("\n[dry-run] graph NOT written.")
    else:
        nx.write_graphml(G, GRAPH_FILE)
        print(f"\nWrote {GRAPH_FILE}")


if __name__ == "__main__":
    main()
