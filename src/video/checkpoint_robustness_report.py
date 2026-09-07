"""Compute the primary statistic of `notes/videomae_ssv2_checkpoint_validation_plan.md`.

That plan asks one narrow question: does the Kinetics-finetuned VideoMAE's
athlete-deletion retention (Stage B: 91% of the above-chance signal survives
painting the athlete out of every frame) reproduce with an SSv2-finetuned
checkpoint, or is it a property of the Kinetics fine-tune specifically? This
module assembles the arms (`full_frame`, `background_only`, `person_crop`, per
checkpoint) with `src.video.stage_b_report.load_single_arm`, so the same
validation-selected thresholds and the same balanced-accuracy implementation
back every number here as back Stage B's.

Everything downstream of the arms reuses `src.video.late_fusion.align` (to put
every arm on one shared video ordering before any cross-arm comparison) and
`src.video.classification_metrics.compute_metrics` (so BA is computed exactly
once, at the plan's validation-selected threshold). The one piece of new logic
is `paired_multi_arm_bootstrap`: the plan's primary statistic, delta A = A_ssv2
- A_kin, is a difference of differences across four arms (kin/ssv2 x
full_frame/background_only), and `late_fusion.paired_bootstrap_delta` only
pairs two. The generalization here draws one set of video indices per
resample and scores every requested arm on that draw before handing the
resulting per-arm seed-mean BA vector to each of several statistic callables,
so delta A, R_kin, R_ssv2 and both `person_crop - full_frame` secondaries all
share the same resampling noise (the plan's "both checkpoints and both arms
are resampled on the same video draw" requirement) with a single bootstrap
pass.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.video.classification_metrics import compute_metrics
from src.video.late_fusion import SplitPredictions, align
from src.video.stage_b_report import ArmRuns

#: The plan's arm vocabulary; every checkpoint contributes exactly these three.
ARM_KINDS = ("full_frame", "background_only", "person_crop")

#: Gate G4 (plan section "Gates"): the report script, run on the existing Kinetics
#: prediction dirs, must reproduce these to 4 decimals before any SSv2 BA is read.
G4_TARGETS = {"full_frame": 0.6401, "background_only": 0.6271, "person_crop": 0.6662}
G4_R_TARGET = 0.907
G4_BA_TOLERANCE = 0.0001
G4_R_TOLERANCE = 0.0005

#: The plan's rule table (section "Primary comparison"), keyed by which branch fired.
READING_SUPPORTED = "H2 supported"
READING_UNDETERMINED = "undetermined"
READING_REVERSED = "ssv2 needs the athlete less"

READING_DETAIL = {
    READING_SUPPORTED: (
        "H2 supported: the SSv2 fine-tune needs the athlete more than the Kinetics "
        "fine-tune does. The Stage B caveat is rewritten as partly checkpoint-specific "
        "and R_ssv2 replaces R_kin as the number to quote."
    ),
    READING_UNDETERMINED: (
        "undetermined at n = 244. The Stage B caveat stands as written and gains one "
        "sentence: the athlete-deletion arm was replicated with an SSv2 checkpoint at "
        "R = R_ssv2 (with its CI). No claim of equivalence is made."
    ),
    READING_REVERSED: (
        "SSv2 needs the athlete less than Kinetics. Reported as such; H1 is not the "
        "explanation either, and the note says the shortcut is not reducible to the "
        "fine-tune label set."
    ),
}


def arm_name(checkpoint: str, kind: str) -> str:
    return f"{checkpoint}_{kind}"


def athlete_contribution(full_frame: ArmRuns, background_only: ArmRuns) -> float:
    """A = BA_full_frame - BA_background_only, seed means."""
    return (
        full_frame.summary()["balanced_accuracy"]["mean"]
        - background_only.summary()["balanced_accuracy"]["mean"]
    )


def retention(full_frame: ArmRuns, background_only: ArmRuns) -> float:
    """R = (BA_background_only - 0.5) / (BA_full_frame - 0.5), seed means."""
    ba_full = full_frame.summary()["balanced_accuracy"]["mean"]
    ba_background = background_only.summary()["balanced_accuracy"]["mean"]
    return (ba_background - 0.5) / (ba_full - 0.5)


#: Below this, a bootstrap resample's full_frame BA is treated as indistinguishable
#: from chance and R is undefined for that resample -- see `safe_ratio`.
RATIO_DENOMINATOR_EPS = 1e-9


def safe_ratio(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or NaN when the denominator is ~0.

    R's denominator is BA_full_frame - 0.5. On the full 244-video test set this is
    never near zero, but a single bootstrap resample over a small or unlucky draw
    can land a checkpoint's full_frame BA exactly at chance, which would otherwise
    blow the whole interval up to +/-inf for one resample. NaN is dropped by
    `paired_multi_arm_bootstrap`'s percentile/fraction-positive computation instead.
    """
    if abs(denominator) < RATIO_DENOMINATOR_EPS:
        return float("nan")
    return numerator / denominator


def g4_gate(arms: dict[str, ArmRuns], checkpoint: str = "kin") -> dict[str, object]:
    """Reproduce the published Kinetics numbers before any SSv2 BA is read."""
    full_frame = arms[arm_name(checkpoint, "full_frame")]
    background_only = arms[arm_name(checkpoint, "background_only")]
    ba = {kind: arms[arm_name(checkpoint, kind)].summary()["balanced_accuracy"]["mean"] for kind in ARM_KINDS}
    ba_deltas = {kind: ba[kind] - target for kind, target in G4_TARGETS.items()}
    r = retention(full_frame, background_only)
    r_delta = r - G4_R_TARGET

    return {
        "balanced_accuracy": ba,
        "targets": dict(G4_TARGETS),
        "deltas_vs_target": ba_deltas,
        "R": r,
        "R_target": G4_R_TARGET,
        "R_delta_vs_target": r_delta,
        "passed": all(abs(delta) <= G4_BA_TOLERANCE for delta in ba_deltas.values())
        and abs(r_delta) <= G4_R_TOLERANCE,
    }


def _align_arm(reference: ArmRuns, arm: ArmRuns) -> ArmRuns:
    """Put `arm` on `reference`'s video ordering, seed by seed, via `late_fusion.align`.

    Every arm in this report comes from a different predictions dir (a different
    checkpoint x variant combination), so nothing guarantees they iterate the split
    file in the same row order the way Stage B's within-arm seeds do. Reusing
    `align` (rather than re-deriving the permutation here) means a mismatched or
    relabeled video set is refused exactly the way late fusion refuses it.
    """
    if arm is reference:
        return arm

    reference_predictions = SplitPredictions(
        video_ids=reference.video_ids, probabilities=reference.probabilities[0], labels=reference.labels
    )
    aligned_probabilities = []
    labels = reference.labels
    video_ids = reference.video_ids
    for probabilities in arm.probabilities:
        seed_predictions = SplitPredictions(video_ids=arm.video_ids, probabilities=probabilities, labels=arm.labels)
        _, reordered, labels, video_ids = align(reference_predictions, seed_predictions)
        aligned_probabilities.append(reordered)

    return ArmRuns(
        name=arm.name,
        seeds=list(arm.seeds),
        probabilities=aligned_probabilities,
        thresholds=list(arm.thresholds),
        labels=labels,
        video_ids=video_ids,
        metrics=arm.metrics,  # aggregate metrics are order-independent; no need to recompute.
    )


def align_arms(arms: dict[str, ArmRuns]) -> dict[str, ArmRuns]:
    """Align every arm to the first arm's video ordering so all can be paired-bootstrapped."""
    names = list(arms)
    reference = arms[names[0]]
    return {name: _align_arm(reference, arm) for name, arm in arms.items()}


def _seed_mean_balanced_accuracy(arm: ArmRuns, index: np.ndarray, drawn_labels: np.ndarray) -> float:
    return float(
        np.mean(
            [
                compute_metrics(probabilities[index], drawn_labels, threshold=threshold)["balanced_accuracy"]
                for probabilities, threshold in zip(arm.probabilities, arm.thresholds)
            ]
        )
    )


def paired_multi_arm_bootstrap(
    arms: dict[str, ArmRuns],
    statistics: dict[str, Callable[[dict[str, float]], float]],
    resamples: int = 10000,
    seed: int = 20260907,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap over test videos, scoring every arm on the same draw.

    One resample draws video indices with replacement, once, and scores every arm
    in `arms` on exactly that draw (seed-mean balanced accuracy at each seed's own
    validation-selected threshold, same as `late_fusion.paired_bootstrap_delta`).
    That per-arm BA vector is then handed to each callable in `statistics`, so a
    difference-of-differences like delta A = A_ssv2 - A_kin, a ratio like R_kin,
    and a two-arm delta like person_crop - full_frame all share the exact same
    resampling noise across the same call -- the plan's pairing requirement.

    `arms` must already be aligned to one video ordering (see `align_arms`); this
    is checked by comparing label vectors rather than re-deriving alignment here.
    """
    if not arms:
        raise ValueError("No arms supplied.")

    names = list(arms)
    reference_labels = arms[names[0]].labels
    n = reference_labels.size
    for name in names[1:]:
        if not np.array_equal(arms[name].labels, reference_labels):
            raise ValueError(f"Arm {name!r} is not aligned to a common label vector; call align_arms first.")

    rng = np.random.default_rng(seed)

    def ba_vector(index: np.ndarray) -> dict[str, float]:
        drawn_labels = reference_labels[index]
        return {name: _seed_mean_balanced_accuracy(arm, index, drawn_labels) for name, arm in arms.items()}

    observed_vector = ba_vector(np.arange(n))
    observed = {stat_name: fn(observed_vector) for stat_name, fn in statistics.items()}

    samples: dict[str, list[float]] = {stat_name: [] for stat_name in statistics}
    for _ in range(resamples):
        vector = ba_vector(rng.integers(0, n, size=n))
        for stat_name, fn in statistics.items():
            samples[stat_name].append(fn(vector))

    result: dict[str, dict[str, float]] = {}
    for stat_name, values in samples.items():
        values_array = np.asarray(values, dtype=np.float64)
        finite = values_array[np.isfinite(values_array)]
        dropped = int(values_array.size - finite.size)
        result[stat_name] = {
            "observed": observed[stat_name],
            # NaN resamples (see `safe_ratio`) are dropped before the percentiles and
            # the positive fraction, and their count is reported rather than silently
            # absorbed -- an interval computed over fewer than `resamples` draws.
            "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
            "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
            "resamples": int(resamples),
            "dropped_resamples": dropped,
            "fraction_positive": float((finite > 0).mean()) if finite.size else float("nan"),
        }
    return result


def primary_reading(ci_low: float, ci_high: float) -> str:
    """Apply the plan's rule table (section "Primary comparison") to delta A's CI."""
    if ci_low > 0.0:
        return READING_SUPPORTED
    if ci_high < 0.0:
        return READING_REVERSED
    return READING_UNDETERMINED
