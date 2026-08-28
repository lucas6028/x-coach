"""Guard the two numbers the Lumen run cycle needs the CSS and the sprite sheet to agree on.

The waiting-state mascot is stepped through frontend/public/lumen/lumen-run.webp by a plain CSS
background-position animation, which has to be told the frame count twice -- once as
`background-size: N00%` and once as `steps(N, ...)`. Neither the sheet nor the stylesheet can tell
the other it changed. Regenerate the sheet with a different --frames and, with no test here,
nothing fails: the loader keeps running, showing a sliding window over two half-frames at a time.

So this is the same bargain tests/test_movement_mistakes_roster.py makes for the fault roster --
a fact that has to be written down in two places gets a test rather than a comment.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHEET = REPO_ROOT / "frontend" / "public" / "lumen" / "lumen-run.webp"
CSS = REPO_ROOT / "frontend" / "src" / "index.css"


class LumenRunSheetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.css = CSS.read_text(encoding="utf-8")

    def test_sheet_is_a_row_of_square_frames(self) -> None:
        from PIL import Image

        self.assertTrue(SHEET.exists(), f"missing {SHEET}; run scripts/prep_lumen_run_frames.py")
        with Image.open(SHEET) as sheet:
            width, height = sheet.size
        self.assertEqual(
            width % height,
            0,
            "the sheet must be a whole number of square frames laid out left to right",
        )

    def test_css_frame_count_matches_the_sheet(self) -> None:
        from PIL import Image

        with Image.open(SHEET) as sheet:
            width, height = sheet.size
        frames = width // height

        sizes = re.findall(r"lumen-run\.webp\"\) 0 0 / (\d+)% 100%", self.css)
        self.assertTrue(sizes, "no background/mask sizing for the run sheet found in index.css")
        for size in sizes:
            self.assertEqual(
                int(size) // 100,
                frames,
                f"index.css sizes the run sheet for {int(size) // 100} frames; it has {frames}",
            )

        steps = re.findall(r"lm-frames var\(--lm-cycle\) steps\((\d+), jump-none\)", self.css)
        self.assertTrue(steps, "no steps() timing for the run sheet found in index.css")
        for step in steps:
            self.assertEqual(
                int(step),
                frames,
                f"index.css steps the run sheet {step} ways; it has {frames} frames",
            )

    def test_the_analysing_band_still_exists(self) -> None:
        # It did not, for a while. Rewriting this block to step through the sprite sheet dropped
        # `.lm-scan::after` -- the violet band that sweeps down Lumen while the pipeline runs -- and
        # nothing failed: the loader kept animating, the markup tests kept passing, and the only
        # symptom was a waiting state that had quietly stopped saying it was analysing anything.
        # `.lm-scan` without its ::after is an empty masked box, so pin the pair.
        self.assertIn(".lm-scan::after {", self.css)
        band = self.css[self.css.index(".lm-scan::after {") :]
        band = band[: band.index("}")]
        self.assertIn('content: ""', band)
        self.assertIn("animation: lm-sweep", band)

    def test_the_stage_uses_the_app_palette(self) -> None:
        # The card Lumen stands on follows the muse-spark tokens; only the mascot's own gold is
        # hard-coded. A hex here means somebody re-pinned the stage to a fixed colour again, which
        # is how it ended up as a navy tile in a lavender app the first time.
        wrap = self.css[self.css.index(".lm-scan-wrap {") :]
        wrap = wrap[: wrap.index("\n}")]
        self.assertIn("var(--ms-violet-soft)", wrap)
        self.assertNotIn("#20223f", wrap.lower())

    def test_the_sweep_runs_to_the_far_end_of_the_sheet(self) -> None:
        # A percentage background-position is measured against how far the sheet can slide, so the
        # last frame is at +100%, not -100%. Getting the sign wrong slides it the wrong way and the
        # stage renders empty for every frame but the first -- which is not a visible failure in
        # any test that only checks the markup.
        self.assertIn("to { background-position: 100% 0;", self.css)
        self.assertNotIn("background-position: -100% 0", self.css)


if __name__ == "__main__":
    unittest.main()
