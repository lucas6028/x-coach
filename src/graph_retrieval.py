from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_FILE = PROJECT_ROOT / "data" / "kg" / "squat_kg_v2.graphml"

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


def load_graph(graph_file: Path = DEFAULT_GRAPH_FILE) -> nx.MultiDiGraph:
    graph = nx.read_graphml(graph_file)
    if not graph.is_multigraph():
        graph = nx.MultiDiGraph(graph)
    return graph


def build_lookup(graph: nx.MultiDiGraph) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for node_id, attrs in graph.nodes(data=True):
        label = str(attrs.get("label", "Unknown"))
        node_text = str(node_id)
        lookup[compact_key(node_text)].add(node_text)
        lookup[compact_key(normalize_query(node_text))].add(node_text)
        lookup[compact_key(label)].add(node_text)
        lookup[compact_key(f"{node_text} {label}")].add(node_text)
    return lookup


def resolve_nodes(graph: nx.MultiDiGraph, query: str, *, limit: int = 10) -> list[str]:
    normalized = normalize_query(query)
    lookup = build_lookup(graph)
    exact = rank_nodes(graph, lookup.get(compact_key(normalized), set()))
    if exact:
        return exact[:limit]

    query_key = compact_key(normalized)
    partial_matches: list[str] = []
    for node_id in graph.nodes():
        node_text = str(node_id)
        node_key = compact_key(node_text)
        if query_key and (query_key in node_key or node_key in query_key):
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
) -> dict[str, Any]:
    graph = load_graph(graph_file)
    matched_nodes = resolve_nodes(graph, query, limit=max_seeds)

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
            "label": str(graph.nodes[node_id].get("label", "Unknown")),
        }
        for node_id in sorted(subgraph_nodes)
    ]

    return {
        "query": query,
        "normalized_query": normalize_query(query),
        "graph_file": str(graph_file),
        "matched_nodes": matched_nodes,
        "results": results,
        "subgraph": {
            "nodes": nodes_payload,
            "edges": subgraph_edges,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve graph context from the squat knowledge graph.")
    parser.add_argument("query", type=str, help="Node or concept to retrieve from the knowledge graph.")
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH_FILE)
    parser.add_argument("--hops", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = retrieve_graph_context(
        args.query,
        graph_file=args.graph_file,
        hops=args.hops,
        max_seeds=args.max_seeds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
