"""Unit tests for the REHAB24-6 SSv2-vs-Kinetics checkpoint report.

The report trains nothing; it reads summary JSONs and evaluates the plan's rule tables.
The fixtures below build minimal summaries in the exact shapes the framing report, the
identity control and the position control write, so each rule row can be driven
deliberately: adopt / undetermined / worse for the primary, the H1 / H2 / H3 patterns
for the secondary reading table, and the missing-input path for every section.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scipy.stats import wilcoxon

from src.rehab24 import videomae_checkpoint_report as report

SUBJECTS = [str(i) for i in range(1, 10)]


def framing_summary(kin: dict[str, float], ssv2: dict[str, float], background: dict[str, float] | None = None) -> dict:
    def arm(values: dict[str, float]) -> dict:
        mean = sum(values.values()) / len(values)
        return {
            "seed_averaged_by_subject": values,
            "balanced_accuracy_no_p10": {"mean": mean},
            "balanced_accuracy_with_p10_sensitivity": {"mean": mean + 0.01},
        }

    arms = {"kin": arm(kin), "ssv2": arm(ssv2)}
    if background is not None:
        arms["ssv2_background_only"] = arm(background)
    return {
        "arms": arms,
        "primary": {
            "comparison": "ssv2-kin",
            "by_camera": {
                "cam17": {
                    "baseline_mean": 0.64,
                    "candidate_mean": 0.66,
                    "delta": {"mean": 0.02},
                    "n_positive": 6,
                    "n_subjects": 9,
                    "wilcoxon": {"p_value": 0.3},
                }
            },
            "by_exercise": {},
        },
    }


def within_summary(per_subject: dict[str, float], p_value: float = 1e-4, include_p10: bool = False) -> dict:
    values = list(per_subject.values())
    mean = sum(values) / len(values)
    block = {
        "mean": mean,
        "sd": 0.04,
        "n_subjects": len(values),
        "n_subjects_above_chance": sum(v > 0.5 for v in values),
        "per_subject_auc": per_subject,
        "permutation": {"p_value": p_value},
        "bootstrap_95ci_over_subjects": [mean - 0.02, mean + 0.02],
    }
    summary = {"primary": block, "secondary": {}}
    if include_p10:
        p10 = dict(per_subject, **{"10": 0.9})
        summary["secondary"]["p10_inclusive_sensitivity"] = {"per_subject_auc": p10, "mean": sum(p10.values()) / 10}
    return summary


def probe(per_subject: dict[str, float], inside_null: bool) -> dict:
    return {
        "per_subject": per_subject,
        "subject_macro_mean": sum(per_subject.values()) / len(per_subject),
        "median_inside_null_95_interval": inside_null,
        "null": {
            "median_null_95_interval": [-0.08, 0.08],
            "p_value_median": 0.2 if inside_null else 1e-4,
            "p_value_mean": 0.3,
        },
    }


def constant(value: float, step: float = 0.0) -> dict[str, float]:
    return {s: value + step * i for i, s in enumerate(SUBJECTS)}


class Fixture:
    """Writes a full set of summaries into a temp dir and returns the paths dict."""

    def __init__(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.paths = {key: self.root / f"{key}.json" for key in report.DEFAULT_PATHS}

    def write(self, key: str, payload: dict) -> None:
        self.paths[key].write_text(json.dumps(payload), encoding="utf-8")

    def close(self) -> None:
        self.dir.cleanup()


def kin_side(fixture: Fixture) -> None:
    fixture.write("kin_within", within_summary(constant(0.85, 0.01)))
    fixture.write("kin_probe_k0", probe(constant(0.40, 0.01), inside_null=False))
    fixture.write("kin_probe_k16", probe(constant(0.02, 0.005), inside_null=True))
    fixture.write("kin_within_k16", within_summary(constant(0.83, 0.01)))


class PairedDeltaTest(unittest.TestCase):
    def test_two_sided_p_matches_scipy_exact_two_sided(self) -> None:
        cand = constant(0.70, 0.01)
        offsets = [0.03, -0.01, 0.05, 0.02, 0.04, -0.02, 0.06, 0.01, 0.03]
        base = {s: cand[s] - d for s, d in zip(SUBJECTS, offsets)}
        result = report.paired_subject_delta(cand, base, n_bootstrap=200, bootstrap_seed=1)
        _, expected = wilcoxon(offsets, method="exact", alternative="two-sided")
        self.assertAlmostEqual(result["two_sided_p"], float(expected), places=12)
        self.assertEqual(result["n_positive"], 7)
        self.assertEqual(result["n_subjects"], 9)
        self.assertAlmostEqual(result["mean_delta"], sum(offsets) / 9)
        low, high = result["bootstrap_95ci_over_subjects"]
        self.assertLess(low, result["mean_delta"])
        self.assertGreater(high, result["mean_delta"])

    def test_unmatched_subject_is_dropped_and_listed(self) -> None:
        cand = constant(0.7)
        base = dict(constant(0.6), **{"10": 0.5})
        result = report.paired_subject_delta(cand, base, 100, 1)
        self.assertEqual(result["n_subjects"], 9)
        self.assertEqual(result["dropped_subjects"], ["10"])

    def test_all_zero_deltas_gives_no_test(self) -> None:
        result = report.paired_subject_delta(constant(0.6), constant(0.6), 100, 1)
        self.assertIsNone(result["two_sided_p"])
        self.assertEqual(report.primary_rule(result)["row"], "undetermined")


class PrimaryRuleTest(unittest.TestCase):
    def rule_for(self, deltas: list[float]) -> dict:
        cand = {s: 0.65 + d for s, d in zip(SUBJECTS, deltas)}
        return report.primary_rule(report.paired_subject_delta(cand, constant(0.65), 100, 1))

    def test_adopt_row(self) -> None:
        rule = self.rule_for([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.02, 0.05])
        self.assertEqual(rule["row"], "ssv2_adopted")

    def test_small_mean_is_undetermined_even_if_concordant(self) -> None:
        rule = self.rule_for([0.01] * 9)
        self.assertEqual(rule["row"], "undetermined")
        self.assertFalse(rule["conditions"]["mean_delta_at_least_+0.02"])

    def test_large_mean_with_high_p_is_undetermined(self) -> None:
        rule = self.rule_for([0.10, -0.08, 0.09, -0.07, 0.08, -0.06, 0.10, 0.11, 0.05])
        self.assertEqual(rule["row"], "undetermined")

    def test_worse_row(self) -> None:
        rule = self.rule_for([-0.05, -0.04, -0.06, -0.03, -0.05, -0.04, -0.07, -0.02, -0.05])
        self.assertEqual(rule["row"], "ssv2_worse")


class ShareAndProbeTest(unittest.TestCase):
    def test_share_definition(self) -> None:
        self.assertAlmostEqual(report.position_share(0.8741, 0.8556), (0.8741 - 0.8556) / 0.3741)

    def test_probe_summary_median_and_flags(self) -> None:
        summary = report.probe_summary(probe({"1": 0.1, "2": 0.5, "3": 0.3}, inside_null=False))
        self.assertAlmostEqual(summary["median"], 0.3)
        self.assertEqual(summary["n_positive"], 3)
        self.assertFalse(summary["median_inside_null_95_interval"])


class ReportSectionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        kin_side(self.fixture)

    def tearDown(self) -> None:
        self.fixture.close()

    def build(self) -> dict:
        return report.build_report(self.fixture.paths, n_bootstrap=200, bootstrap_seed=1)

    def test_missing_ssv2_inputs_leave_sections_unavailable_and_gates_readable(self) -> None:
        out = self.build()
        self.assertFalse(out["primary"]["available"])
        self.assertFalse(out["s1_within_session"]["available"])
        self.assertFalse(out["s2_position"]["available"])
        self.assertFalse(out["s3_background_only"]["available"])
        self.assertFalse(out["reading"]["rows"]["H2_motion_centric"])
        gate = out["gates"]["G4_reproduction"]
        self.assertIsNotNone(gate["values"]["kin_auc"])
        self.assertFalse(gate["all_present"])
        self.assertEqual(out["secondary_holm"], {})

    def test_g4_passes_on_the_published_numbers(self) -> None:
        kin_ba = dict(zip(SUBJECTS, [0.6119, 0.6812, 0.5797, 0.5949, 0.6497, 0.7380, 0.6804, 0.7541, 0.6609]))
        summary = framing_summary(kin_ba, constant(0.66))
        summary["arms"]["kin"]["balanced_accuracy_no_p10"]["mean"] = 0.66120807
        self.fixture.write("framing_summary", summary)
        kin_auc = dict(zip(SUBJECTS, [0.8368, 0.8945, 0.8745, 0.8166, 0.8350, 0.9446, 0.8847, 0.9132, 0.8673]))
        self.fixture.write("kin_within", within_summary(kin_auc))
        kin_probe = dict(zip(SUBJECTS, [0.229, 0.254, 0.320, 0.501, 0.691, 0.4324, 0.170, 0.571, 0.458]))
        self.fixture.write("kin_probe_k0", probe(kin_probe, False))
        gate = self.build()["gates"]["G4_reproduction"]
        self.assertTrue(gate["checks"]["kin_letterbox_ba_is_0.6612"])
        self.assertTrue(gate["checks"]["kin_within_session_auc_is_0.8741"])
        self.assertTrue(gate["checks"]["kin_probe_median_is_0.4324"])
        self.assertTrue(gate["pass"])

    def h2_inputs(self) -> None:
        """SSv2 ranks better within session, encodes less position, background at chance."""
        self.fixture.write(
            "framing_summary", framing_summary(constant(0.65, 0.01), constant(0.66, 0.01), constant(0.50, 0.002))
        )
        self.fixture.write("ssv2_within", within_summary(constant(0.90, 0.01)))
        self.fixture.write("ssv2_probe_k0", probe(constant(0.20, 0.01), inside_null=False))
        self.fixture.write("ssv2_probe_k16", probe(constant(0.01, 0.005), inside_null=True))
        self.fixture.write("ssv2_within_k0", within_summary(constant(0.90, 0.01)))
        self.fixture.write("ssv2_within_k16", within_summary(constant(0.895, 0.01)))
        self.fixture.write("ssv2_background_within", within_summary(constant(0.52, 0.005), p_value=0.2))

    def test_h2_pattern(self) -> None:
        self.h2_inputs()
        out = self.build()
        self.assertTrue(out["primary"]["available"])
        self.assertEqual(out["primary"]["rule"]["row"], "undetermined")  # +0.01 is inside the band
        self.assertTrue(out["s1_within_session"]["available"])
        self.assertTrue(out["s2_position"]["ssv2_probe_below_kin"])
        self.assertTrue(out["s2_position"]["ssv2_share_at_or_below_kin"])
        self.assertFalse(out["s2_position"]["checkpoints"]["ssv2"]["share_is_floor"])
        self.assertFalse(out["s3_background_only"]["informative"])
        rows = out["reading"]["rows"]
        self.assertTrue(rows["H2_motion_centric"])
        self.assertFalse(rows["H3_drift_reading"])
        self.assertFalse(rows["H1_checkpoint_general"])
        self.assertFalse(out["reading"]["fusion_gate_closed_for_ssv2"])
        self.assertIn("s1_delta_auc_two_sided", out["secondary_holm"])
        self.assertIn("s3_background_within_session_permutation", out["secondary_holm"])

    def test_h3_pattern_when_background_becomes_informative(self) -> None:
        self.h2_inputs()
        self.fixture.write("ssv2_background_within", within_summary(constant(0.70, 0.01), p_value=1e-4))
        out = self.build()
        self.assertTrue(out["s3_background_only"]["within_session"]["informative"])
        self.assertTrue(out["s3_background_only"]["informative"])
        rows = out["reading"]["rows"]
        self.assertTrue(rows["H3_drift_reading"])
        self.assertFalse(rows["H2_motion_centric"])
        self.assertTrue(rows["S3_checkpoint_specific_non_person_path"])
        self.assertTrue(out["reading"]["fusion_gate_closed_for_ssv2"])

    def test_h3_pattern_when_probe_rises(self) -> None:
        self.h2_inputs()
        self.fixture.write("ssv2_probe_k0", probe(constant(0.60, 0.01), inside_null=False))
        out = self.build()
        self.assertFalse(out["s2_position"]["ssv2_probe_below_kin"])
        self.assertTrue(out["reading"]["rows"]["H3_drift_reading"])

    def test_h1_pattern(self) -> None:
        self.h2_inputs()
        # S1 undetermined: alternate the sign of the within-session delta.
        kin_auc = json.loads(self.fixture.paths["kin_within"].read_text())["primary"]["per_subject_auc"]
        ssv2_auc = {s: v + (0.01 if i % 2 else -0.01) for i, (s, v) in enumerate(kin_auc.items())}
        self.fixture.write("ssv2_within", within_summary(ssv2_auc))
        out = self.build()
        self.assertTrue(out["reading"]["inputs"]["s1_undetermined"])
        self.assertTrue(out["reading"]["rows"]["H1_checkpoint_general"])
        self.assertFalse(out["reading"]["rows"]["H2_motion_centric"])

    def test_background_ba_informative_via_balanced_accuracy(self) -> None:
        self.h2_inputs()
        self.fixture.write(
            "framing_summary", framing_summary(constant(0.65, 0.01), constant(0.66, 0.01), constant(0.60, 0.01))
        )
        out = self.build()
        self.assertTrue(out["s3_background_only"]["balanced_accuracy"]["informative"])
        self.assertTrue(out["s3_background_only"]["informative"])

    def test_share_is_floor_when_k16_probe_stays_outside_null(self) -> None:
        self.h2_inputs()
        self.fixture.write("ssv2_probe_k16", probe(constant(0.25, 0.01), inside_null=False))
        out = self.build()
        self.assertTrue(out["s2_position"]["checkpoints"]["ssv2"]["share_is_floor"])

    def test_s4_and_s5(self) -> None:
        self.h2_inputs()
        self.fixture.write("kin_within", within_summary(constant(0.85, 0.01), include_p10=True))
        self.fixture.write("ssv2_within", within_summary(constant(0.90, 0.01), include_p10=True))
        out = self.build()
        self.assertIn("by_camera:cam17", out["s4_strata"]["strata"])
        self.assertAlmostEqual(out["s5_p10_sensitivity"]["balanced_accuracy_with_p10"]["mean_delta"], 0.01)
        self.assertEqual(out["s5_p10_sensitivity"]["within_session_with_p10"]["n_subjects"], 10)


class CliTest(unittest.TestCase):
    def test_main_writes_output(self) -> None:
        fixture = Fixture()
        try:
            kin_side(fixture)
            output = fixture.root / "out" / "report.json"
            argv: list[str] = []
            for key, path in fixture.paths.items():
                argv += [f"--{key.replace('_', '-')}", str(path)]
            argv += ["--bootstrap", "50", "--output", str(output)]
            report.main(argv)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["bootstrap"]["n_resamples"], 50)
            self.assertFalse(written["primary"]["available"])
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
