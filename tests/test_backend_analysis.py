from __future__ import annotations

import unittest

from backend.app.services import knowledge, library


def _first_library_video(min_faults: int = 1) -> str:
    page = library.list_videos(limit=20)
    for item in page["items"]:
        if item["fault_count"] >= min_faults:
            return item["video_id"]
    # Fall back to any available clip.
    return page["items"][0]["video_id"]


class LibraryServiceTests(unittest.TestCase):
    def test_list_videos_returns_items(self) -> None:
        page = library.list_videos(limit=5)
        self.assertGreater(page["total"], 0)
        self.assertLessEqual(len(page["items"]), 5)
        item = page["items"][0]
        for key in ("video_id", "split", "view_type", "fault_count", "faults"):
            self.assertIn(key, item)

    def test_fault_filter_restricts_results(self) -> None:
        page = library.list_videos(limit=10, fault="knees_inward")
        for item in page["items"]:
            self.assertIn("knees_inward", item["faults"])

    def test_load_analysis_contract(self) -> None:
        video_id = _first_library_video()
        result = library.load_analysis(video_id)

        # Core analysis keys present and frame_metrics stripped in favour of the slim pose block.
        for key in ("video_id", "metadata", "view", "quality", "detections", "retrievals", "pose"):
            self.assertIn(key, result)
        self.assertNotIn("frame_metrics", result)
        self.assertEqual(result["source"], "library")

        # pose block: one entry per frame, each landmark a [x, y, visibility] triple.
        pose = result["pose"]
        self.assertEqual(len(pose["frames"]), result["metadata"]["total_frames"])
        for frame in pose["frames"]:
            if frame["lm"] is not None:
                self.assertEqual(len(frame["lm"]), 33)
                self.assertEqual(len(frame["lm"][0]), 3)
                break

        # detections are typed dicts with the fields the UI relies on.
        for det in result["detections"]:
            for key in ("fault_id", "fault_name", "severity", "start_frame", "end_frame", "phase"):
                self.assertIn(key, det)

        # retrieval enrichment is attached when faults exist.
        if result["detections"]:
            self.assertTrue(result["retrievals"])

    def test_missing_video_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            library.load_analysis("definitely_not_a_real_id_xyz")


class KnowledgeServiceTests(unittest.TestCase):
    def test_graph_context_returns_subgraph(self) -> None:
        ctx = knowledge.graph_context("Knee Valgus")
        self.assertIn("subgraph", ctx)
        self.assertIn("nodes", ctx["subgraph"])
        self.assertTrue(ctx["matched_nodes"])

    def test_rag_snippets_return_results(self) -> None:
        out = knowledge.rag_snippets("knee valgus correction")
        self.assertEqual(out["query"], "knee valgus correction")
        self.assertIsInstance(out["results"], list)


if __name__ == "__main__":
    unittest.main()
