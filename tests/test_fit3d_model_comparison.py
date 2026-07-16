"""Self-tests for the cross-model comparison formatting (synthetic, no dataset/GPU needed)."""

from __future__ import annotations

import unittest

from src.fit3d import decision_eval as dec
from src.fit3d.model_comparison import ROTATION_INVARIANT, format_comparison


def _synthetic_result(names=("NLF", "HMR2")):
    cues = [c.name for c in dec.CUES]
    models = {}
    for i, n in enumerate(names):
        models[n] = {
            "n_seq": 32, "n_pairs": 160, "swap_lr": True,
            "mpjpe": 78.0 + 20 * i, "pa_mpjpe": 65.0 + 15 * i,
            "ez": 42.0 + 20 * i, "exy": 36.0, "ez_exy": (42.0 + 20 * i) / 36.0,
            "cue_err": {c: 7.0 + i for c in cues},
            "verdict_flip_deb": {c: 0.11 + 0.05 * i for c in cues},
            "verdict_flip_raw": {c: 0.26 for c in cues},
            "knee_at_thr": {"deb2d": {"flip": 0.16}},
        }
    return {
        "action": "squat", "split": "train", "models": models,
        "projection_2d": {
            "cue_err": {c.name: 18.0 for c in dec.CUES},
            "verdict_flip_deb": {c.name: 0.20 for c in dec.CUES},
        },
    }


class FormatComparisonTests(unittest.TestCase):
    def test_renders_all_models_and_sections(self):
        txt = format_comparison(_synthetic_result(("NLF", "HMR2")))
        self.assertIn("DEPTH PATTERN", txt)
        self.assertIn("CUE RECOVERY", txt)
        self.assertIn("VERDICT FLIP", txt)
        self.assertIn("NLF", txt)
        self.assertIn("HMR2", txt)
        # rotation-invariant cues are flagged with '*'
        for c in ROTATION_INVARIANT:
            self.assertIn(c, txt)

    def test_skips_missing_models(self):
        res = _synthetic_result(("NLF",))
        res["models"]["BROKEN"] = {"n_seq": 0, "missing": True}
        txt = format_comparison(res)
        self.assertIn("NLF", txt)
        self.assertNotIn("BROKEN", txt.split("models:")[1].split("\n")[0])  # not in the model list line


if __name__ == "__main__":
    unittest.main()
