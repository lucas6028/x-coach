"""Repeated stratified video-level splits for the B1 F2 contrast.

``notes/videomae_person_crop_validation_plan.md`` measured why the fixed split cannot
answer F2: the manipulation only reaches the 613 videos ``full_frame`` truncates, and
only 90 of those are in the 244-video test split, which extrapolates to a CI half-width
of 0.086 against an effect of +0.026. Repeated splits are the plan's option 1: score
every one of the 1623 videos out-of-fold, so the F2 subset returns to 613 scoreable
videos.

Design, and why each part is the way it is:

*Stratified, video-level.* Fitness-AQA ships no participant mapping and all 1623 ids
carry distinct prefixes, so grouping by athlete is not available -- videos are the only
unit. That is a real threat rather than a footnote: if one athlete contributed several
clips, a random video-level split puts them on both sides. The check is to re-derive
the pose-only denominator on these folds FIRST and compare it against the fixed split's
0.650. Each fold trains on ~1103 videos against the fixed split's 1136, so a materially
HIGHER score cannot be explained by more training data and points at leakage.

*Test folds partition the corpus.* Within a repeat every video sits in exactly one test
fold, so a repeat yields one out-of-fold prediction per video and one full-corpus score.
Repeats replace seeds: five repeats, five scores, the same shape the fixed-split arms
report.

*Validation is carved out of the non-test remainder, not shared.* Threshold and
checkpoint selection stay inside the fold, which is what keeps "threshold from
validation only" true fold by fold. 15% is chosen to leave the training size close to
the fixed split's, so the arms stay comparable to the archived numbers.

The five repeats are NOT independent -- the same videos recur in all of them, so a
bootstrap must run over videos within a repeat and be averaged across repeats, never
pooled as 25 independent draws.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPEATS = 5
DEFAULT_FOLDS = 5
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 20260812


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: list[str]
    val: list[str]
    test: list[str]

    @property
    def name(self) -> str:
        return f"r{self.repeat}f{self.fold}"

    def sizes(self) -> tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))


def stratified_folds(video_ids: list[str], labels: dict[str, int], n_folds: int, rng: random.Random) -> list[list[str]]:
    """Deal ids into ``n_folds`` groups, keeping the label ratio even across them.

    Each label's ids are shuffled and dealt round-robin, so a fold's positive rate
    tracks the corpus rate even when positives are scarce.
    """
    by_label: dict[int, list[str]] = {}
    for video_id in video_ids:
        by_label.setdefault(labels[video_id], []).append(video_id)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        # The starting fold rotates per label so the remainder of an uneven deal does
        # not always land on fold 0, which would skew fold 0's positive rate.
        offset = label % n_folds
        for index, video_id in enumerate(ids):
            folds[(index + offset) % n_folds].append(video_id)
    return folds


def make_folds(
    video_ids: list[str],
    labels: dict[str, int],
    n_repeats: int = DEFAULT_REPEATS,
    n_folds: int = DEFAULT_FOLDS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> list[Fold]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for a train/test partition.")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must leave a training set behind.")

    missing = [video_id for video_id in video_ids if video_id not in labels]
    if missing:
        raise ValueError(f"{len(missing)} ids have no label ({missing[:5]}); folds cannot be stratified.")

    folds: list[Fold] = []
    for repeat in range(1, n_repeats + 1):
        # Seeded per repeat, so a repeat's partition is reproducible on its own and
        # adding repeats never changes the earlier ones.
        rng = random.Random(seed + repeat)
        groups = stratified_folds(list(video_ids), labels, n_folds, rng)

        for fold_index in range(n_folds):
            test = sorted(groups[fold_index])
            remainder = sorted(video_id for index, group in enumerate(groups) if index != fold_index for video_id in group)

            val_rng = random.Random(seed + repeat * 100 + fold_index)
            val_count = max(int(round(len(remainder) * val_fraction)), 1)
            val_groups = stratified_folds(remainder, labels, max(round(1 / val_fraction), 2), val_rng)
            val = sorted(val_groups[0])[:val_count]
            val_set = set(val)
            train = [video_id for video_id in remainder if video_id not in val_set]

            folds.append(Fold(repeat=repeat, fold=fold_index, train=train, val=val, test=test))

    verify_folds(folds, video_ids, n_repeats)
    return folds


def verify_folds(folds: list[Fold], video_ids: list[str], n_repeats: int) -> None:
    """Refuse a fold set that cannot support an out-of-fold score.

    Two failures are silent otherwise: a video in both train and test inside one fold
    (leakage that inflates every arm equally, so no comparison reveals it), and a video
    missing from -- or repeated in -- a repeat's test partition, which makes the
    out-of-fold vector quietly shorter or double-counted.
    """
    expected = set(video_ids)
    for fold in folds:
        overlap = (set(fold.train) & set(fold.test)) | (set(fold.val) & set(fold.test)) | (set(fold.train) & set(fold.val))
        if overlap:
            raise ValueError(f"Fold {fold.name} has {len(overlap)} ids in more than one split ({sorted(overlap)[:5]}).")

    for repeat in range(1, n_repeats + 1):
        seen: list[str] = []
        for fold in folds:
            if fold.repeat == repeat:
                seen.extend(fold.test)
        if len(seen) != len(set(seen)):
            raise ValueError(f"Repeat {repeat} tests some videos more than once.")
        if set(seen) != expected:
            missing = expected - set(seen)
            raise ValueError(f"Repeat {repeat} never tests {len(missing)} videos ({sorted(missing)[:5]}).")


def write_fold_keys(fold: Fold, root: Path) -> Path:
    """Write one fold's three key files, in the layout the classifier already reads."""
    fold_dir = root / fold.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in (("train", fold.train), ("val", fold.val), ("test", fold.test)):
        with (fold_dir / f"{split_name}_keys.json").open("w", encoding="utf-8") as f:
            json.dump(ids, f)
    return fold_dir


def write_all_folds(folds: list[Fold], root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        write_fold_keys(fold, root)

    manifest = {
        "n_folds": len(folds),
        "repeats": sorted({fold.repeat for fold in folds}),
        "folds": [
            {"name": fold.name, "repeat": fold.repeat, "fold": fold.fold, "sizes": list(fold.sizes())}
            for fold in folds
        ],
    }
    with (root / "folds.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
