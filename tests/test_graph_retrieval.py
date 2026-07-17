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
    list_movement_faults,
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
        # Seed carries a bare `name` (falls back to node_id when no `name` attr is set).
        self.assertEqual(
            summary["seed"], {"node_id": "Knee Valgus", "name": "Knee Valgus", "label": "Fault"}
        )
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


def _build_v3_graph() -> nx.MultiDiGraph:
    """A v3-shaped graph: scoped ids namespaced `Movement:Name` carry a `movement` tag and a bare
    `name`; shared nodes carry `movement="shared"`. Two movements share the fault name "Knee Valgus"
    so the movement filter has something to discriminate; a shared node sits 2 hops from the Squat seed."""
    graph = nx.MultiDiGraph()
    graph.add_node("Squat:Knee Valgus", label="Fault", name="Knee Valgus", movement="Squat")
    graph.add_node("Lunge:Knee Valgus", label="Fault", name="Knee Valgus", movement="Lunge")
    graph.add_node("Squat:Narrow Stance", label="EvidenceSignal", name="Narrow Stance", movement="Squat")
    graph.add_node("Pelvic Control", label="Cause", name="Pelvic Control", movement="shared")

    graph.add_edge("Squat:Knee Valgus", "Squat:Narrow Stance", type="INDICATED_BY")
    graph.add_edge("Squat:Narrow Stance", "Pelvic Control", type="CAUSED_BY")
    graph.add_edge("Lunge:Knee Valgus", "Pelvic Control", type="CAUSED_BY")
    return graph


class ScopedRetrievalTests(unittest.TestCase):
    """Cover the v3 movement-scoping path (is_scoped_for / _movement_ok) end to end."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.graph_file = self.tmp / "kg_v3.graphml"
        nx.write_graphml(_build_v3_graph(), self.graph_file)

    def test_movement_scoped_seeds_traversal_and_name_payload(self):
        result = retrieve_graph_context(
            "knee valgus", graph_file=self.graph_file, hops=2, movement="Squat"
        )
        # (a) seeds are restricted to the Squat-scoped node; the Lunge-scoped same-name node is excluded.
        self.assertIn("Squat:Knee Valgus", result["matched_nodes"])
        self.assertNotIn("Lunge:Knee Valgus", result["matched_nodes"])
        # (b) a 2-hop traversal still crosses (Squat:Narrow Stance ->) into the shared node.
        node_ids = {node["node_id"] for node in result["subgraph"]["nodes"]}
        self.assertIn("Pelvic Control", node_ids)
        # (c) node payloads now carry a bare `name` — the scoped seed renders "Knee Valgus", not the id.
        by_id = {node["node_id"]: node for node in result["subgraph"]["nodes"]}
        self.assertEqual(by_id["Squat:Knee Valgus"]["name"], "Knee Valgus")

    def test_movement_none_spans_all_movements(self):
        # movement=None is a deliberate cross-movement mode: a fault name shared by two movements
        # resolves BOTH seeds (this is the intended behaviour of the multi-movement graph, and the
        # analyze/library paths guard against it by always passing movement="Squat").
        result = retrieve_graph_context("knee valgus", graph_file=self.graph_file, hops=1, movement=None)
        self.assertIn("Squat:Knee Valgus", result["matched_nodes"])
        self.assertIn("Lunge:Knee Valgus", result["matched_nodes"])


def _build_movement_faults_graph() -> nx.MultiDiGraph:
    """A movement whose faults connect to the Action node by DIFFERENT paths — one directly
    (HAS_FAULT), one only THROUGH a phase, and one with no cause/cue/risk/evidence at all. This is
    the shape that breaks a hop-limited enumeration: a 1-hop-from-root traversal misses the
    phase-only fault, so `list_movement_faults` must enumerate by the `movement` attribute instead."""
    g = nx.MultiDiGraph()
    g.add_node("Squat", label="Action", name="Squat", movement="Squat")
    g.add_node("Squat:Descent", label="Phase", name="Descent", movement="Squat")
    g.add_node("Squat:Direct Fault", label="Fault", name="Direct Fault", movement="Squat")
    g.add_node("Squat:Phase Only Fault", label="Fault", name="Phase Only Fault", movement="Squat")
    g.add_node("Squat:Isolated Fault", label="Fault", name="Isolated Fault", movement="Squat")
    g.add_node("Lunge:Other Fault", label="Fault", name="Other Fault", movement="Lunge")
    g.add_node("Weak Glutes", label="Cause", name="Weak Glutes", movement="shared")
    g.add_node("Drive Knees Out", label="Cue", name="Drive Knees Out", movement="shared")

    g.add_edge("Squat", "Squat:Descent", type="HAS_PHASE")
    g.add_edge("Squat", "Squat:Direct Fault", type="HAS_FAULT")           # reachable 1 hop from root
    g.add_edge("Squat:Descent", "Squat:Phase Only Fault", type="HAS_FAULT")  # reachable only via phase
    g.add_edge("Squat:Direct Fault", "Weak Glutes", type="CAUSED_BY")     # connectivity > 0
    g.add_edge("Squat:Phase Only Fault", "Drive Knees Out", type="CORRECTED_BY")  # connectivity > 0
    g.add_edge("Lunge:Other Fault", "Weak Glutes", type="CAUSED_BY")
    return g


class ListMovementFaultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.graph_file = self.tmp / "kg.graphml"
        nx.write_graphml(_build_movement_faults_graph(), self.graph_file)

    def test_enumerates_all_movement_faults_including_phase_only(self):
        faults = list_movement_faults(self.graph_file, "Squat")
        names = {f["name"] for f in faults}
        # ALL three Squat faults are returned — crucially the phase-only one that a hop-limited query
        # from the movement root would omit. The Lunge fault is excluded by the movement filter.
        self.assertEqual(names, {"Direct Fault", "Phase Only Fault", "Isolated Fault"})
        self.assertNotIn("Other Fault", names)

    def test_phase_only_fault_is_missed_by_hop_limited_traversal(self):
        # Demonstrates the bug the attribute-filter fixes: a 1-hop query seeded at the movement root
        # does NOT reach the phase-only fault, so deriving the fault list from a subgraph undercounts.
        ctx = retrieve_graph_context("Squat", graph_file=self.graph_file, hops=1, movement="Squat")
        reached = {n["name"] for n in ctx["subgraph"]["nodes"] if n.get("label") == "Fault"}
        self.assertIn("Direct Fault", reached)
        self.assertNotIn("Phase Only Fault", reached)

    def test_connectivity_flags_unlinked_faults(self):
        conn = {f["name"]: f["connectivity"] for f in list_movement_faults(self.graph_file, "Squat")}
        self.assertGreater(conn["Direct Fault"], 0)
        self.assertGreater(conn["Phase Only Fault"], 0)
        self.assertEqual(conn["Isolated Fault"], 0)  # drives the Explore "no linked graph" state

    def test_results_sorted_by_name(self):
        names = [f["name"] for f in list_movement_faults(self.graph_file, "Squat")]
        self.assertEqual(names, sorted(names, key=str.lower))


if __name__ == "__main__":
    unittest.main()
