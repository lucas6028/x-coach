"""Single source of truth for the multi-movement KG schema.

Encodes the scoped-vs-shared node cut and the `Movement:Name` namespacing convention
from docs/kg-schema-generalization.md, plus loading of the shared controlled vocabulary
(data/kg/shared_vocab_v1.json). Imported by extract_kg, graph_retrieval, and the migration
so extraction, retrieval, and migration can never drift on how a node id is formed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_VOCAB_FILE = PROJECT_ROOT / "data" / "kg" / "shared_vocab_v1.json"
DEFAULT_GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "sports_kg_v3.graphml"

# Node scope by label (see design note 2).
SCOPED_LABELS = frozenset({"Action", "Phase", "Fault", "EvidenceSignal"})
SHARED_LABELS = frozenset({"Cause", "Cue", "Risk", "QualityDimension"})
SHARED_TAG = "shared"


@lru_cache(maxsize=1)
def load_shared_vocab(path: str | None = None) -> dict:
    vocab_path = Path(path) if path else SHARED_VOCAB_FILE
    if not vocab_path.exists():
        return {"canonical": {}, "aliases": {}}
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def shared_aliases() -> dict[str, dict[str, str]]:
    """{label: {raw_name: canonical_name}} for the shared layer."""
    return load_shared_vocab().get("aliases", {})


def canonical_shared_names() -> dict[str, list[str]]:
    """{label: [canonical names]} — the controlled vocab extraction is steered toward."""
    return load_shared_vocab().get("canonical", {})


def resolve_node_id(name: str, label: str, movement: str) -> tuple[str, dict]:
    """Map an extracted (name, label) to its canonical graph id + node attributes.

    Scoped labels  -> id "Movement:Name" (anchor Action keeps a bare movement id), movement=<Movement>.
    Shared labels  -> plain canonical id (vocab alias applied),                     movement="shared".
    """
    name = name.strip()
    if label in SHARED_LABELS:
        canon = shared_aliases().get(label, {}).get(name, name)
        return canon, {"label": label, "name": canon, "movement": SHARED_TAG}

    # scoped (and any unrecognized label, kept scoped to be safe)
    if label == "Action" and name.strip().lower() == movement.strip().lower():
        node_id = movement
    else:
        node_id = f"{movement}:{name}"
    return node_id, {"label": label, "name": name, "movement": movement}


def is_scoped_for(node_attrs: dict, movement: str) -> bool:
    """True if a node is either shared or scoped to the given movement."""
    mv = str(node_attrs.get("movement", ""))
    return mv == SHARED_TAG or mv.lower() == movement.strip().lower()
