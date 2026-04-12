from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx


DEFAULT_GRAPH = Path("data/kg/squat_kg_v2.graphml")
DEFAULT_MAPPING = Path("data/kg/docs/squat_canonical_mapping_v1.json")
DEFAULT_OUTPUT = Path("data/kg/squat_kg_v2.graphml")


def load_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_node_id(node_id: str, label: str, mapping: dict) -> tuple[str, str]:
    if label == "Phase":
        if node_id in mapping["phase_rules"]["keep"]:
            return mapping["phase_rules"]["keep"][node_id], "keep"
        if node_id in mapping["phase_rules"]["merge"]:
            return mapping["phase_rules"]["merge"][node_id], "merge"
        if node_id in mapping["phase_rules"]["review"]:
            return node_id, "review"
    if label == "Fault":
        if node_id in mapping["fault_rules"]["keep"]:
            return mapping["fault_rules"]["keep"][node_id], "keep"
        if node_id in mapping["fault_rules"]["merge"]:
            return mapping["fault_rules"]["merge"][node_id], "merge"
        if node_id in mapping["fault_rules"]["review"]:
            return node_id, "review"
    if label == "Risk":
        if node_id in mapping["risk_rules"]["keep"]:
            return mapping["risk_rules"]["keep"][node_id], "keep"
        if node_id in mapping["risk_rules"]["merge"]:
            return mapping["risk_rules"]["merge"][node_id], "merge"
        if node_id in mapping["risk_rules"]["review"]:
            return node_id, "review"
    return node_id, "keep"


def canonical_label(node_id: str, label: str, mapping: dict) -> str:
    for canonical_label_name, canonical_ids in mapping["canonical_labels"].items():
        if node_id in canonical_ids:
            return canonical_label_name
    return label


def clean_graph(graph: nx.MultiDiGraph, mapping: dict) -> tuple[nx.MultiDiGraph, dict[str, list[str]]]:
    cleaned = nx.MultiDiGraph()
    merge_report = {"merged": [], "review": []}
    node_id_map: dict[str, str] = {}

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "Unknown")
        canonical_id, action = canonical_node_id(str(node_id), str(label), mapping)
        canonical_node_label = canonical_label(canonical_id, str(label), mapping)
        node_id_map[str(node_id)] = canonical_id
        if not cleaned.has_node(canonical_id):
            cleaned.add_node(canonical_id, label=canonical_node_label)
        if action == "merge" and canonical_id != node_id:
            merge_report["merged"].append(f"{node_id} -> {canonical_id}")
        elif action == "review":
            merge_report["review"].append(str(node_id))

    seen_edges: set[tuple[str, str, str]] = set()
    for source, target, attrs in graph.edges(data=True):
        canonical_source = node_id_map[str(source)]
        canonical_target = node_id_map[str(target)]
        edge_type = str(attrs.get("type", "RELATED_TO"))
        edge_key = (canonical_source, canonical_target, edge_type)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        cleaned.add_edge(canonical_source, canonical_target, type=edge_type)

    return cleaned, merge_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply canonical cleanup rules to a KG GraphML file.")
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mapping = load_mapping(args.mapping_file)
    graph = nx.read_graphml(args.graph_file)
    if not graph.is_multigraph():
        graph = nx.MultiDiGraph(graph)

    cleaned, merge_report = clean_graph(graph, mapping)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(cleaned, args.output_file)

    print(f"Input graph: {args.graph_file}")
    print(f"Output graph: {args.output_file}")
    print(f"Nodes: {graph.number_of_nodes()} -> {cleaned.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()} -> {cleaned.number_of_edges()}")
    print("Merged nodes:")
    if merge_report["merged"]:
        for item in merge_report["merged"]:
            print(f"- {item}")
    else:
        print("- None")
    print("Review nodes:")
    if merge_report["review"]:
        for item in merge_report["review"]:
            print(f"- {item}")
    else:
        print("- None")


if __name__ == "__main__":
    main()
