"""Unit tests for the knowledge-graph retrieval helpers (src/knowledge/graph_retrieval.py).

Covers the string-normalization utilities, node lookup/resolution/ranking over a small
hand-built MultiDiGraph, edge collection with hop expansion, bucketed summarization, and
an end-to-end retrieve_graph_context against a temporary .graphml file.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from src.knowledge.graph_retrieval import (
    canonicalize_whitespace,
    collect_edges_for_seed,
    compact_key,
    build_lookup,
    normalize_query,
    rank_nodes,
    resolve_nodes,
    retrieve_graph_context,
    summarize_seed,
    to_title_case,
)


def _build_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("Knee Valgus", label="Fault")
    graph.add_node("Narrow Stance Width", label="EvidenceSignal")
    graph.add_node("Weak Glutes", label="Cause")
    graph.add_node("ACL Injury", label="Risk")
    graph.add_node("Push Knees Out", label="Cue")
    graph.add_node("Descent", label="Phase")

    graph.add_edge("Knee Valgus", "Narrow Stance Width", type="INDICATED_BY")
    graph.add_edge("Knee Valgus", "Weak Glutes", type="CAUSED_BY")
    graph.add_edge("Knee Valgus", "ACL Injury", type="INCREASES_RISK_OF")
    graph.add_edge("Knee Valgus", "Push Knees Out", type="CORRECTED_BY")
    graph.add_edge("Knee Valgus", "Descent", type="OCCURS_IN_PHASE")
    return graph


class StringNormalizationTests(unittest.TestCase):
    def test_canonicalize_whitespace_collapses_runs(self):
        self.assertEqual(canonicalize_whitespace("  knee   valgus\n cue "), "knee valgus cue")

    def test_canonicalize_whitespace_handles_empty(self):
        self.assertEqual(canonicalize_whitespace(""), "")

    def test_compact_key_strips_non_alphanumeric_and_lowercases(self):
        self.assertEqual(compact_key("Knee-Valgus!"), "kneevalgus")

    def test_to_title_case_capitalizes_words(self):
        self.assertEqual(to_title_case("weak glutes cause"), "Weak Glutes Cause")

    def test_to_title_case_preserves_all_caps_tokens(self):
        self.assertEqual(to_title_case("ACL injury"), "ACL Injury")

    def test_normalize_query_resolves_alias(self):
        self.assertEqual(normalize_query("knees cave in"), "Knee Valgus")

    def test_normalize_query_falls_back_to_title_case(self):
        self.assertEqual(normalize_query("weak glutes"), "Weak Glutes")


class LookupAndResolveTests(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph()

    def test_build_lookup_maps_compact_key_to_node(self):
        lookup = build_lookup(self.graph)
        self.assertIn("Knee Valgus", lookup[compact_key("Knee Valgus")])

    def test_resolve_exact_via_alias(self):
        self.assertEqual(resolve_nodes(self.graph, "knees cave in"), ["Knee Valgus"])

    def test_resolve_partial_substring_match(self):
        self.assertIn("Knee Valgus", resolve_nodes(self.graph, "valgus"))

    def test_resolve_unknown_query_returns_empty(self):
        self.assertEqual(resolve_nodes(self.graph, "completely unrelated concept"), [])

    def test_resolve_respects_limit(self):
        self.assertLessEqual(len(resolve_nodes(self.graph, "knee", limit=1)), 1)

    def test_rank_nodes_prioritizes_fault_over_risk(self):
        ranked = rank_nodes(self.graph, {"ACL Injury", "Knee Valgus"})
        self.assertEqual(ranked, ["Knee Valgus", "ACL Injury"])


class CollectEdgesTests(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph()

    def test_one_hop_collects_all_outgoing_edges(self):
        edges, visited = collect_edges_for_seed(self.graph, "Knee Valgus", hops=1)
        self.assertEqual(len(edges), 5)
        self.assertTrue(all(edge.direction == "out" for edge in edges))
        self.assertIn("ACL Injury", visited)

    def test_incoming_edges_are_labeled_in_direction(self):
        edges, _ = collect_edges_for_seed(self.graph, "ACL Injury", hops=1)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].direction, "in")
        self.assertEqual(edges[0].source, "Knee Valgus")
        self.assertEqual(edges[0].relation, "INCREASES_RISK_OF")

    def test_zero_hops_collects_nothing(self):
        edges, visited = collect_edges_for_seed(self.graph, "Knee Valgus", hops=0)
        self.assertEqual(edges, [])
        self.assertEqual(visited, {"Knee Valgus"})


class SummarizeSeedTests(unittest.TestCase):
    def setUp(self):
        self.graph = _build_graph()

    def test_edges_are_grouped_into_expected_buckets(self):
        edges, _ = collect_edges_for_seed(self.graph, "Knee Valgus", hops=1)
        summary = summarize_seed(self.graph, "Knee Valgus", edges)
        self.assertEqual(summary["seed"], {"node_id": "Knee Valgus", "label": "Fault"})
        self.assertEqual(
            set(summary["summary"].keys()),
            {"phases", "evidence", "causes", "risks", "corrections"},
        )
        self.assertEqual(summary["summary"]["risks"][0]["node_id"], "ACL Injury")

    def test_counterpart_with_wrong_label_is_dropped(self):
        # An evidence edge pointing at a non-EvidenceSignal node must not survive bucketing.
        graph = _build_graph()
        graph.add_node("Bogus Cause", label="Cause")
        graph.add_edge("Knee Valgus", "Bogus Cause", type="INDICATED_BY")
        edges, _ = collect_edges_for_seed(graph, "Knee Valgus", hops=1)
        summary = summarize_seed(graph, "Knee Valgus", edges)
        evidence_nodes = [item["node_id"] for item in summary["summary"].get("evidence", [])]
        self.assertNotIn("Bogus Cause", evidence_nodes)


class RetrieveGraphContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.graph_file = self.tmp / "kg.graphml"
        nx.write_graphml(_build_graph(), self.graph_file)

    def test_end_to_end_retrieval(self):
        result = retrieve_graph_context("knee valgus", graph_file=self.graph_file, hops=1)
        self.assertEqual(result["normalized_query"], "Knee Valgus")
        self.assertIn("Knee Valgus", result["matched_nodes"])
        node_ids = {node["node_id"] for node in result["subgraph"]["nodes"]}
        self.assertIn("ACL Injury", node_ids)
        self.assertTrue(result["subgraph"]["edges"])

    def test_unmatched_query_yields_empty_results(self):
        result = retrieve_graph_context("xyzzy", graph_file=self.graph_file, hops=1)
        self.assertEqual(result["matched_nodes"], [])
        self.assertEqual(result["results"], [])


if __name__ == "__main__":
    unittest.main()
