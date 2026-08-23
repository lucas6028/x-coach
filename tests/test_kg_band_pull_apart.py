"""The authored Band Pull Apart fault nodes must resolve for the detector's kg_queries.

This is a data test against data/kg/sports_kg_v3.graphml, which is a gitignored build
artifact. It SKIPS when the graph is absent (a fresh clone) rather than failing, but must
pass wherever the graph exists -- including every deploy target, whose graph is built
separately by scripts/knowledge/author_band_pull_apart_kg_v3.py.
"""
from __future__ import annotations

import unittest
from pathlib import Path

# IMPORTED FOR ITS SIDE EFFECT, BEFORE ANY `src.pose.movements.band_pull_apart` import below.
# `registry.py` registers the 14 detectors by importing them in order at its own footer, and
# `list_detectors()` returns that insertion order -- which `/api/movements` serves and
# `tests/test_movements_endpoint.py` pins. Importing `band_pull_apart` FIRST makes registry's
# re-import of the partially-initialized module a no-op, so Band Pull Apart registers LAST and
# that endpoint test fails when the two files share a pytest process. Importing registry here
# fixes the order for this file. NOTE: `tests/test_band_pull_apart.py` trips the same wire
# (`pytest tests/test_band_pull_apart.py tests/test_movements_endpoint.py` fails today, without
# any change from this file); that one is left alone as a separate, pre-existing defect.
import src.pose.movements.registry  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = REPO_ROOT / "data" / "kg" / "sports_kg_v3.graphml"

BENT_ELBOWS = "Band Pull Apart:Bent Elbows"
TRUNK_COMP = "Band Pull Apart:Trunk Extension Compensation"


@unittest.skipUnless(GRAPH_FILE.exists(), "sports_kg_v3.graphml not built in this checkout")
class TestBandPullApartKgNodes(unittest.TestCase):
    def _context(self, query: str) -> dict:
        from src.knowledge.graph_retrieval import retrieve_graph_context

        # hops=1 and max_seeds=3 mirror pose_rule_detector.py:775, the path that actually
        # renders a FaultCard. Asserting at any other depth would test a configuration
        # production does not use.
        return retrieve_graph_context(
            query, graph_file=GRAPH_FILE, hops=1, max_seeds=3, movement="Band Pull Apart"
        )

    def _seed(self, query: str, node_id: str) -> dict:
        ctx = self._context(query)
        self.assertIn(node_id, ctx["matched_nodes"], f"{query!r} did not resolve to {node_id!r}")
        return next(r for r in ctx["results"] if r["seed"]["node_id"] == node_id)

    def test_the_detector_constants_are_the_queries_under_test(self) -> None:
        """Pin the module constants themselves, so repointing a kg_query without re-checking
        the graph turns this file red instead of silently emptying a card."""
        from src.pose.movements.band_pull_apart import (
            BPA_ROM_KG_QUERY,
            BPA_SHRUG_KG_QUERY,
            BPA_TRUNK_KG_QUERY,
        )

        self.assertEqual(BPA_ROM_KG_QUERY, "Bent Elbows")
        self.assertEqual(BPA_TRUNK_KG_QUERY, "Trunk Extension Compensation")
        self.assertEqual(BPA_SHRUG_KG_QUERY, "Shoulder Shrugging")

    def test_every_firing_rule_query_resolves_to_a_scoped_fault(self) -> None:
        from src.pose.movements.band_pull_apart import (
            BPA_ROM_KG_QUERY,
            BPA_SHRUG_KG_QUERY,
            BPA_TRUNK_KG_QUERY,
        )

        for query, expected in (
            (BPA_SHRUG_KG_QUERY, "Band Pull Apart:Shoulder Shrugging"),
            (BPA_ROM_KG_QUERY, BENT_ELBOWS),
            (BPA_TRUNK_KG_QUERY, TRUNK_COMP),
        ):
            with self.subTest(query=query):
                seed = self._seed(query, expected)
                # The scoped Fault must be the FIRST seed -- a shared node outranking it would
                # put generic content at the top of the card.
                self.assertEqual(self._context(query)["matched_nodes"][0], expected)
                self.assertEqual(seed["seed"]["label"], "Fault")

    def test_bent_elbows_names_what_the_fault_costs(self) -> None:
        """Before 2026-08-23 this node had connectivity 0 and rendered an all-dashes card."""
        summary = self._seed("Bent Elbows", BENT_ELBOWS)["summary"]
        self.assertTrue(summary.get("quality_impacts"), "no AFFECTS_QUALITY edge")
        self.assertIn("Range Of Motion", {q["name"] for q in summary["quality_impacts"]})

    def test_trunk_compensation_names_what_the_fault_costs(self) -> None:
        summary = self._seed("Trunk Extension Compensation", TRUNK_COMP)["summary"]
        self.assertTrue(summary.get("quality_impacts"), "no AFFECTS_QUALITY edge")
        self.assertIn(
            "No Compensatory Trunk Movement", {q["name"] for q in summary["quality_impacts"]}
        )

    def test_no_fabricated_cause_risk_or_cue_edges(self) -> None:
        """DELIBERATE ABSENCES, not an unfinished card. This movement's only citation is
        Fukunaga (PMC8975561): it never mentions the elbow, and on the trunk it runs the other
        way -- "performing strengthening exercises with hip and trunk extension may be
        beneficial". A Cause, Risk or Cue edge on either fault would attach that citation to a
        claim the paper does not make. Padding these buckets means finding a source first.
        """
        for query, node_id in (("Bent Elbows", BENT_ELBOWS),
                               ("Trunk Extension Compensation", TRUNK_COMP)):
            summary = self._seed(query, node_id)["summary"]
            for bucket in ("causes", "risks", "corrections"):
                with self.subTest(node=node_id, bucket=bucket):
                    self.assertFalse(summary.get(bucket), f"uncited {bucket} edge on {node_id}")

    def test_the_rejected_stray_cue_stays_out_of_the_bent_elbows_card(self) -> None:
        """`Range Of Motion` carries `CORRECTED_BY -> Wrapping Surface Adjustment`, meaningless
        for this movement -- which is why the design doc rejected SEEDING the query there. The
        AFFECTS_QUALITY edge added instead must not drag that cue back in at production depth.
        """
        ctx = self._context("Bent Elbows")
        names = {n["name"] for n in ctx["subgraph"]["nodes"]}
        self.assertNotIn("Wrapping Surface Adjustment", names)

    def test_the_action_owns_every_scoped_fault(self) -> None:
        import networkx as nx

        graph = nx.read_graphml(GRAPH_FILE)
        owned = {
            v for _, v, d in graph.out_edges("Band Pull Apart", data=True)
            if d.get("type") == "HAS_FAULT"
        }
        self.assertIn(TRUNK_COMP, owned)
        scoped_faults = {
            n for n, d in graph.nodes(data=True)
            if str(d.get("movement")) == "Band Pull Apart" and str(d.get("label")) == "Fault"
        }
        self.assertEqual(scoped_faults, owned, "a scoped Fault is not reachable from the Action")


if __name__ == "__main__":  # pragma: no cover - convenience runner
    unittest.main()
