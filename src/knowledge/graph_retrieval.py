from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

from src.knowledge.kg_schema import DEFAULT_GRAPH_FILE, is_scoped_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EDGE_BUCKETS = {
    "HAS_PHASE": "phases",
    "OCCURS_IN_PHASE": "phases",
    "INDICATED_BY": "evidence",
    "CAUSED_BY": "causes",
    "INCREASES_RISK_OF": "risks",
    "CORRECTED_BY": "corrections",
    "AFFECTS_QUALITY": "quality_impacts",
    "HAS_FAULT": "related_actions",
}

EDGE_BUCKET_LABELS = {
    "phases": {"Phase"},
    "evidence": {"EvidenceSignal"},
    "causes": {"Cause"},
    "risks": {"Risk"},
    "corrections": {"Cue"},
    "quality_impacts": {"QualityDimension"},
    "related_actions": {"Action"},
}

NODE_ALIASES = {
    "knees cave in": "Knee Valgus",
    "knee cave in": "Knee Valgus",
    "knees collapse inward": "Knee Valgus",
    "knees collapse in": "Knee Valgus",
    "valgus knee": "Knee Valgus",
    "knee valgus": "Knee Valgus",
    "foot pronation": "Foot Pronation",
    "pronated foot": "Foot Pronation",
    "heels rise": "Heel Rise",
    "heel rise": "Heel Rise",
    "heels lifting": "Heel Rise",
    "shallow squat": "Shallow Depth",
    "shallow depth": "Shallow Depth",
    "anterior knee pain": "Anterior Knee Pain",
}


def canonicalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonicalize_whitespace(text).lower())


def to_title_case(text: str) -> str:
    words = canonicalize_whitespace(text).split(" ")
    return " ".join(word if word.isupper() else word.capitalize() for word in words if word)


def normalize_query(text: str) -> str:
    normalized = canonicalize_whitespace(text)
    alias = NODE_ALIASES.get(normalized.lower())
    if alias:
        return alias
    return to_title_case(normalized)


@dataclass
class EdgeRecord:
    source: str
    target: str
    relation: str
    direction: str

    def to_json(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "direction": self.direction,
        }


# Labels whose 1-hop neighbours are what GraphScene renders around a fault
# (causes / risks / corrections / evidence). A fault with zero such neighbours has
# no graph to show — used to compute per-fault connectivity for the Explore browser.
FAULT_NEIGHBOR_LABELS = frozenset({"Cause", "Risk", "Cue", "EvidenceSignal"})


@lru_cache(maxsize=8)
def _read_graph_cached(path_str: str, _mtime: float) -> nx.MultiDiGraph:
    graph = nx.read_graphml(path_str)
    if not graph.is_multigraph():
        graph = nx.MultiDiGraph(graph)
    return graph


def load_graph(graph_file: Path = DEFAULT_GRAPH_FILE) -> nx.MultiDiGraph:
    """Load a KG graphml, memoised by (path, mtime) so repeated queries in one process
    don't re-parse the 2000+ node file. Callers MUST treat the result as read-only —
    it is a shared instance. The mtime key invalidates the cache when the file is rebuilt."""
    path = Path(graph_file)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _read_graph_cached(str(path), mtime)


def list_movement_faults(
    graph_file: Path = DEFAULT_GRAPH_FILE, movement: str | None = None
) -> list[dict[str, Any]]:
    """Every ``Fault`` node for a movement, by the ``movement`` node attribute (not traversal,
    which under- or over-counts), each with its 1-hop ``connectivity`` — the count of distinct
    Cause/Risk/Cue/EvidenceSignal neighbours GraphScene would render. connectivity == 0 means the
    fault has no graph to show yet. Sorted by display name."""
    graph = load_graph(graph_file)
    faults: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        if str(attrs.get("label")) != "Fault":
            continue
        if movement is not None and str(attrs.get("movement")) != movement:
            continue
        neighbours: set[str] = set()
        for _, target in graph.out_edges(node_id):
            if str(graph.nodes[target].get("label")) in FAULT_NEIGHBOR_LABELS:
                neighbours.add(target)
        for source, _ in graph.in_edges(node_id):
            if str(graph.nodes[source].get("label")) in FAULT_NEIGHBOR_LABELS:
                neighbours.add(source)
        faults.append({"name": str(attrs.get("name", node_id)), "connectivity": len(neighbours)})
    faults.sort(key=lambda f: str(f["name"]).lower())
    return faults


def build_lookup(graph: nx.MultiDiGraph) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", "Unknown"))
        node_text = str(node_id)
        lookup[compact_key(node_text)].add(node_text)
        lookup[compact_key(normalize_query(node_text))].add(node_text)
        lookup[compact_key(label)].add(node_text)
        lookup[compact_key(f"{node_text} {label}")].add(node_text)
        # Index the unprefixed name too, so "knee valgus" resolves the scoped id
        # "Squat:Knee Valgus" (the movement filter then picks the right movement).
        name = str(attrs.get("name", "")).strip()
        if name:
            lookup[compact_key(name)].add(node_text)
            lookup[compact_key(normalize_query(name))].add(node_text)
    return lookup


def _movement_ok(graph: nx.MultiDiGraph, node_id: str, movement: str | None) -> bool:
    if not movement:
        return True
    return is_scoped_for(graph.nodes[node_id], movement)


def resolve_nodes(
    graph: nx.MultiDiGraph, query: str, *, limit: int = 10, movement: str | None = None
) -> list[str]:
    """Resolve a query to seed node ids. When `movement` is set, seeds are restricted to
    that movement's scoped nodes plus shared nodes (traversal from those seeds is still free,
    so a 2-hop query can intentionally cross into other movements via the shared layer)."""
    normalized = normalize_query(query)
    lookup = build_lookup(graph)
    exact = rank_nodes(graph, {n for n in lookup.get(compact_key(normalized), set()) if _movement_ok(graph, n, movement)})
    if exact:
        return exact[:limit]

    query_key = compact_key(normalized)
    partial_matches: list[str] = []
    for node_id in graph.nodes():
        node_text = str(node_id)
        node_key = compact_key(node_text)
        if query_key and (query_key in node_key or node_key in query_key) and _movement_ok(graph, node_text, movement):
            partial_matches.append(node_text)
    return rank_nodes(graph, set(partial_matches))[:limit]


def rank_nodes(graph: nx.MultiDiGraph, candidates: set[str]) -> list[str]:
    def score(node_id: str) -> tuple[int, int, str]:
        label = str(graph.nodes[node_id].get("label", "Unknown"))
        normalized_penalty = 0 if node_id == normalize_query(node_id) else 1
        label_priority = {
            "Fault": 0,
            "EvidenceSignal": 1,
            "Cause": 2,
            "Risk": 3,
            "Cue": 4,
            "Phase": 5,
            "Action": 6,
            "QualityDimension": 7,
        }.get(label, 99)
        return (normalized_penalty, label_priority, node_id)

    return sorted(candidates, key=score)


def collect_edges_for_seed(
    graph: nx.MultiDiGraph,
    seed: str,
    *,
    hops: int,
) -> tuple[list[EdgeRecord], set[str]]:
    visited = {seed}
    frontier = {seed}
    collected: list[EdgeRecord] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    for _ in range(max(hops, 0)):
        next_frontier: set[str] = set()
        for node in frontier:
            for _, target, attrs in graph.out_edges(node, data=True):
                relation = str(attrs.get("type", "RELATED_TO"))
                edge_key = (str(node), str(target), relation, "out")
                if edge_key not in seen_edges:
                    collected.append(
                        EdgeRecord(source=str(node), target=str(target), relation=relation, direction="out")
                    )
                    seen_edges.add(edge_key)
                if str(target) not in visited:
                    next_frontier.add(str(target))
                    visited.add(str(target))
            for source, _, attrs in graph.in_edges(node, data=True):
                relation = str(attrs.get("type", "RELATED_TO"))
                edge_key = (str(source), str(node), relation, "in")
                if edge_key not in seen_edges:
                    collected.append(
                        EdgeRecord(source=str(source), target=str(node), relation=relation, direction="in")
                    )
                    seen_edges.add(edge_key)
                if str(source) not in visited:
                    next_frontier.add(str(source))
                    visited.add(str(source))
        frontier = next_frontier
        if not frontier:
            break

    return collected, visited


def summarize_seed(graph: nx.MultiDiGraph, seed: str, edges: list[EdgeRecord]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for edge in edges:
        bucket = EDGE_BUCKETS.get(edge.relation)
        if bucket is None:
            continue

        if edge.direction == "out":
            counterpart = edge.target
        else:
            counterpart = edge.source

        counterpart_label = str(graph.nodes[counterpart].get("label", "Unknown"))
        expected_labels = EDGE_BUCKET_LABELS.get(bucket)
        if expected_labels and counterpart_label not in expected_labels:
            continue

        grouped[bucket].append(
            {
                "node_id": counterpart,
                "name": str(graph.nodes[counterpart].get("name", counterpart)),
                "label": counterpart_label,
                "relation": edge.relation,
                "direction": edge.direction,
            }
        )

    deduped_grouped: dict[str, list[dict[str, str]]] = {}
    for bucket, items in grouped.items():
        seen: set[tuple[str, str, str, str]] = set()
        deduped_items: list[dict[str, str]] = []
        for item in items:
            item_key = (
                item["node_id"],
                item["label"],
                item["relation"],
                item["direction"],
            )
            if item_key in seen:
                continue
            seen.add(item_key)
            deduped_items.append(item)
        deduped_grouped[bucket] = deduped_items[:10]

    return {
        "seed": {
            "node_id": seed,
            "name": str(graph.nodes[seed].get("name", seed)),
            "label": str(graph.nodes[seed].get("label", "Unknown")),
        },
        "summary": deduped_grouped,
    }


def retrieve_graph_context(
    query: str,
    *,
    graph_file: Path = DEFAULT_GRAPH_FILE,
    hops: int = 1,
    max_seeds: int = 5,
    movement: str | None = None,
) -> dict[str, Any]:
    graph = load_graph(graph_file)
    matched_nodes = resolve_nodes(graph, query, limit=max_seeds, movement=movement)

    results: list[dict[str, Any]] = []
    subgraph_nodes: set[str] = set()
    subgraph_edges: list[dict[str, str]] = []

    for seed in matched_nodes:
        edges, nodes = collect_edges_for_seed(graph, seed, hops=hops)
        subgraph_nodes.update(nodes)
        subgraph_edges.extend(edge.to_json() for edge in edges)
        seed_summary = summarize_seed(graph, seed, edges)
        seed_summary["edges"] = [edge.to_json() for edge in edges]
        results.append(seed_summary)

    nodes_payload = [
        {
            "node_id": node_id,
            "name": str(graph.nodes[node_id].get("name", node_id)),
            "label": str(graph.nodes[node_id].get("label", "Unknown")),
        }
        for node_id in sorted(subgraph_nodes)
    ]

    return {
        "query": query,
        "normalized_query": normalize_query(query),
        "movement": movement,
        "graph_file": str(graph_file),
        "matched_nodes": matched_nodes,
        "results": results,
        "subgraph": {
            "nodes": nodes_payload,
            "edges": subgraph_edges,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve graph context from the multi-movement sports knowledge graph.")
    parser.add_argument("query", type=str, help="Node or concept to retrieve from the knowledge graph.")
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH_FILE)
    parser.add_argument("--hops", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--movement", type=str, default=None,
                        help="Restrict seeds to this movement's scoped nodes + shared nodes (e.g. Squat, Lunge).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = retrieve_graph_context(
        args.query,
        graph_file=args.graph_file,
        hops=args.hops,
        max_seeds=args.max_seeds,
        movement=args.movement,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
