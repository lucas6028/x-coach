from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


def compact_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def load_nodes(graph_path: Path) -> list[tuple[str, str]]:
    graph = nx.read_graphml(graph_path)
    return [(str(node_id), str(attrs.get("label", "Unknown"))) for node_id, attrs in graph.nodes(data=True)]


def print_grouped_duplicates(nodes: list[tuple[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node_id, label in nodes:
        grouped[(label, compact_key(node_id))].append(node_id)

    print("Potential duplicate node IDs:")
    found = False
    for (label, _), ids in sorted(grouped.items()):
        unique_ids = sorted(set(ids))
        if len(unique_ids) > 1:
            found = True
            print(f"- {label}: {', '.join(unique_ids)}")
    if not found:
        print("- None")


def print_label_summary(nodes: list[tuple[str, str]]) -> None:
    counts = Counter(label for _, label in nodes)
    print("Node label counts:")
    for label, count in sorted(counts.items()):
        print(f"- {label}: {count}")


def print_phase_fault_details(nodes: list[tuple[str, str]]) -> None:
    phases = sorted(node_id for node_id, label in nodes if label == "Phase")
    faults = sorted(node_id for node_id, label in nodes if label == "Fault")
    print("Phase nodes:")
    for phase in phases:
        print(f"- {phase}")
    print("Fault nodes:")
    for fault in faults:
        print(f"- {fault}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a GraphML knowledge graph for duplicate-like nodes.")
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=Path("data/kg/squat_kg_v2.graphml"),
        help="Path to the GraphML file to audit.",
    )
    args = parser.parse_args()

    nodes = load_nodes(args.graph_file)
    print(f"Graph: {args.graph_file}")
    print(f"Total nodes: {len(nodes)}")
    print_label_summary(nodes)
    print()
    print_grouped_duplicates(nodes)
    print()
    print_phase_fault_details(nodes)


if __name__ == "__main__":
    main()
