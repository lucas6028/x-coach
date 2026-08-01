"""The authored Overhead Press lockout fault node must resolve for the detector's kg_query.

This is a data test against data/kg/sports_kg_v3.graphml, which is a gitignored build
artifact. It SKIPS when the graph is absent (a fresh clone) rather than failing, but must
pass wherever the graph exists -- including every deploy target, whose graph is built
separately.
"""
from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"


@unittest.skipUnless(GRAPH_FILE.exists(), "sports_kg_v3.graphml not built in this checkout")
class TestOhpLockoutNode(unittest.TestCase):
    def _context(self, query: str) -> dict:
        from src.knowledge.graph_retrieval import retrieve_graph_context

        return retrieve_graph_context(
            query, graph_file=GRAPH_FILE, hops=1, max_seeds=3, movement="Overhead Press"
        )

    def test_incomplete_elbow_lockout_resolves(self) -> None:
        ctx = self._context("Incomplete Elbow Lockout")
        self.assertIn("Overhead Press:Incomplete Elbow Lockout", ctx["matched_nodes"])

    def test_lockout_seed_carries_coaching_content(self) -> None:
        """A node that resolves but has no causes/corrections is no better than the dashes it
        replaces -- assert the buckets the chat prompt actually renders."""
        ctx = self._context("Incomplete Elbow Lockout")
        seed = next(
            r for r in ctx["results"]
            if r["seed"]["node_id"] == "Overhead Press:Incomplete Elbow Lockout"
        )
        summary = seed["summary"]
        self.assertTrue(summary.get("causes"), "no CAUSED_BY -> Cause edges")
        self.assertTrue(summary.get("corrections"), "no CORRECTED_BY -> Cue edges")
        names = {c["name"] for c in summary["causes"]}
        self.assertIn("Weak Triceps Brachii", names)
        cues = {c["name"] for c in summary["corrections"]}
        self.assertIn("Full Elbow Extension At Lockout", cues)

    def test_no_fabricated_risk_edge(self) -> None:
        """The rule's citation establishes the mechanism, not an injury outcome. An
        INCREASES_RISK_OF edge here would be an uncited claim."""
        ctx = self._context("Incomplete Elbow Lockout")
        seed = next(
            r for r in ctx["results"]
            if r["seed"]["node_id"] == "Overhead Press:Incomplete Elbow Lockout"
        )
        self.assertFalse(seed["summary"].get("risks"))
