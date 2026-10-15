"""The committed rep-signal golden file still says what the Python detectors compute.

frontend/src/test/lib.repSignal.golden.test.ts pins the browser's rep signals to
tests/fixtures/rep_signal_golden.json, but that only proves TS agrees with the file. This test
proves the file agrees with Python TODAY: change a ported movement's metric, gate or rep knobs
without regenerating, and it fails here, not silently in production where the backend trusts
whatever windows the browser computed.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from scripts.pose.capture_rep_signal_golden import OUTPUT, PORTED_MOVEMENTS, capture, load_frames

REGENERATE = "regenerate with scripts/pose/capture_rep_signal_golden.py " + str(
    Path("tests") / "fixtures" / OUTPUT.name
)


class RepSignalGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(OUTPUT.read_text(encoding="utf-8"))
        _, fps, frames = load_frames(cls.golden)
        cls.movements, cls.cases = capture(frames, fps)

    def test_covers_every_ported_movement(self) -> None:
        self.assertEqual(sorted(self.golden["movements"]), sorted(PORTED_MOVEMENTS), REGENERATE)

    def test_rep_knobs_match_the_detectors(self) -> None:
        self.assertEqual(self.golden["movements"], self.movements, REGENERATE)

    def test_signals_match_the_detectors(self) -> None:
        for index, (stored, fresh) in enumerate(zip(self.golden["cases"], self.cases)):
            for movement in PORTED_MOVEMENTS:
                expected = stored["signals"].get(movement)
                actual = fresh["signals"][movement]
                with self.subTest(frame=index, movement=movement):
                    if expected is None:
                        self.assertIsNone(actual, REGENERATE)
                    else:
                        self.assertIsNotNone(actual, REGENERATE)
                        self.assertTrue(math.isclose(actual, expected, abs_tol=1e-9), REGENERATE)


if __name__ == "__main__":
    unittest.main()
