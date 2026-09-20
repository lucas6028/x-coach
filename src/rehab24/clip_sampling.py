"""Repetition-bounded clip sampling, shared by the VideoMAE and V-JEPA 2 extractors.

``notes/rehab24_videomae_vjepa2_validation_plan.md`` registers a defect in the
historical VideoMAE extractor: ``sample_clip_starts`` (moved here unchanged from
``src.rehab24.videomae_features``) already respects repetition boundaries when
picking *clip starts*, but the old ``read_clip_frames`` only stopped decoding at
the *video's* end, not the repetition's. A short repetition could therefore leak
frames from whatever comes after it. Extending that sampler to 64 frames without a
boundary guard would make the leak worse, not better -- hence this module.

``build_bounded_clips`` is the single index-generation path both extractors call,
so ``vm16`` and ``vj16`` (same ``clip_length``/``frame_stride``/rep bounds) draw
byte-identical ``frame_indices`` by construction, and every arm's indices are
provably confined to ``[first_index, min(last_index, total_frames - 1)]``.

Why ``padding_fraction == 0`` is a valid reproducibility gate (used by
``check_vm16_against_history`` in ``src.rehab24.vjepa2_features``): with
``frame_stride >= 1`` the raw, unclamped indices ``start + stride*k`` for
``k in 0..clip_length-1`` are strictly increasing, hence pairwise distinct. A
repeated index in the stored ``frame_indices`` can therefore only come from
clamping to the rep/video bound. So ``padding_fraction == 0`` for every clip of a
repetition implies clamping never fired for that repetition, which implies the
old sampler (bounded only by clip length, not by the repetition) would have
produced the exact same indices. Those repetitions are the ones where the old,
defective ``read_clip_frames`` happened to do the right thing anyway, and are the
only ones where an old and a new bundle can be compared as if they were the same
extraction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def sample_clip_starts(first_frame: int, last_frame: int, clip_length: int, frame_stride: int, num_clips: int) -> list[int]:
    """Evenly spaced clip starts within one repetition's decoder-index frame range.

    Unchanged from the historical ``src.rehab24.videomae_features`` implementation
    (moved here so both extractors share it verbatim): ``vm16`` must keep drawing
    the exact historical starts, and ``vj16`` must match ``vm16`` exactly, which
    only holds if both call the same function with the same clip length and stride.
    """
    start = max(first_frame - 1, 0)
    stop = max(last_frame, start + 1)
    effective_length = 1 + frame_stride * (clip_length - 1)
    max_start = max(stop - effective_length, start)
    if num_clips <= 1:
        return [(start + max_start) // 2]
    return np.linspace(start, max_start, num=num_clips, dtype=int).tolist()


def rep_index_bounds(first_frame: int, last_frame: int) -> tuple[int, int]:
    """Convert the manifest's 1-based, inclusive frame numbers to decoder indices.

    ``first_frame`` is 1-based, so its decoder index is ``first_frame - 1``.
    ``last_frame`` is 1-based and inclusive, so its decoder index is
    ``last_frame - 1`` too. ``last_index`` is floored at ``first_index`` so a
    malformed one-frame-or-negative-length row never yields an empty range.
    """
    first_index = max(first_frame - 1, 0)
    last_index = max(last_frame - 1, first_index)
    return first_index, last_index


def clip_frame_indices(
    start: int,
    clip_length: int,
    frame_stride: int,
    first_index: int,
    last_index: int,
    total_frames: int,
) -> np.ndarray:
    """One clip's frame indices, clamped to the repetition AND the decoded video.

    Raises if the repetition's own bounds are inconsistent with the video (the
    plan requires any such disagreement to block, not to be padded past silently).
    A short repetition pads by repeating the last in-repetition frame -- never a
    frame beyond ``last_index``, never beyond ``total_frames - 1``.
    """
    if total_frames > 0 and last_index > total_frames - 1 and first_index > total_frames - 1:
        raise ValueError(
            f"Repetition bounds [{first_index}, {last_index}] fall entirely after "
            f"the decoded video ({total_frames} frames); refusing to sample."
        )
    bound = min(last_index, total_frames - 1) if total_frames > 0 else last_index
    if bound < first_index:
        raise ValueError(
            f"Repetition bound {bound} is before its own start {first_index} "
            f"(total_frames={total_frames}); refusing to sample."
        )
    raw = start + frame_stride * np.arange(clip_length, dtype=np.int64)
    return np.clip(raw, first_index, bound)


def build_bounded_clips(
    first_frame: int,
    last_frame: int,
    total_frames: int,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
) -> tuple[list[dict[str, object]], int, int]:
    """All ``num_clips`` clips for one repetition: indices, uniqueness, padding.

    Returns ``(clips, first_index, last_index)``. Each clip dict holds ``start``
    (int), ``frame_indices`` (int64 array of length ``clip_length``),
    ``unique_frame_count`` (int) and ``padding_fraction`` (float, ``1 -
    unique/clip_length``). Duplicate clips in short repetitions are retained and
    counted, never discarded (plan, Dataset/sampling section).
    """
    starts = sample_clip_starts(first_frame, last_frame, clip_length, frame_stride, num_clips)
    first_index, last_index = rep_index_bounds(first_frame, last_frame)
    clips: list[dict[str, object]] = []
    for start in starts:
        indices = clip_frame_indices(start, clip_length, frame_stride, first_index, last_index, total_frames)
        unique = int(np.unique(indices).size)
        clips.append(
            {
                "start": int(start),
                "frame_indices": indices,
                "unique_frame_count": unique,
                "padding_fraction": float(1.0 - unique / clip_length),
            }
        )
    return clips, first_index, last_index


def decode_frames_at_indices(
    cap: cv2.VideoCapture, indices: Sequence[int], total_frames: int, context: str = ""
) -> dict[int, np.ndarray]:
    """Decode exactly the needed 0-based frame indices from an open capture, once each.

    Shared by both extractors: a repetition's clips can overlap (short reps under
    ``vj64``'s 127-frame span especially), so decoding the *union* of indices with
    one forward walk avoids decoding the same frame twice and avoids the repeated
    keyframe re-seeks a per-clip ``cap.set`` would cost on long H.264 clips (the
    reasoning already recorded on ``read_clip_frames``). ``indices`` need not be
    sorted or unique. Returned frames are RGB uint8.

    Every requested index is already clamped to ``[first_index, total_frames - 1]``
    by :func:`clip_frame_indices`, so a decode failure here always means a corrupt
    video or container/decoder disagreement, never a sampling bug -- and the plan
    requires any such decode error to block, not to be silently patched over.
    ``cap.read()`` failing is therefore raised EVERY time it happens, not only on
    the first requested index: an earlier version of this function padded any
    failure *after* the first one with the previous frame, which would have
    quietly let a mid-repetition decode gap through as if every frame decoded.
    ``context`` (e.g. ``"sample=... video=..."``) is folded into the message so a
    failure names the repetition and file it came from, not just the frame index.
    """
    needed = sorted({int(index) for index in indices})
    if not needed:
        return {}
    floor = max(total_frames - 1, 0) if total_frames > 0 else needed[-1]
    seek_to = min(needed[0], floor)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)

    decoded: dict[int, np.ndarray] = {}
    cursor = seek_to
    for index in needed:
        while cursor < index:
            cap.grab()
            cursor += 1
        ok, frame = cap.read()
        cursor += 1
        if not ok:
            where = f" ({context})" if context else ""
            raise RuntimeError(f"Could not decode frame {index}{where} (total_frames={total_frames}).")
        decoded[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return decoded


def sha256_of_files(paths: Sequence[Path]) -> str:
    """A single hex digest over an explicit, fixed-order list of source files.

    Stamped as ``provenance_code_fingerprint`` so a later edit to the sampler,
    backbone or extractor is visible in every bundle written afterwards, and so
    the resume-provenance check (below) refuses to silently mix bundles from two
    versions of this code. Reads in binary mode deliberately: a text-mode read
    on Windows normalizes line endings, which would make the hash depend on the
    checkout's line-ending settings rather than on the file's actual bytes.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def provenance_mismatches(existing: dict[str, str], current: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Keys present in both dicts whose values disagree, as ``{key: (old, new)}``."""
    return {key: (existing[key], current[key]) for key in current if key in existing and existing[key] != current[key]}


def read_bundle_provenance(path: Path) -> dict[str, str]:
    """The ``provenance_*`` fields of one ``.npz`` bundle, stripped of the prefix."""
    with np.load(path, allow_pickle=False) as data:
        return {key[len("provenance_") :]: str(data[key]) for key in data.files if key.startswith("provenance_")}


def assert_resume_provenance_matches(arm_dir: Path, provenance: dict[str, str]) -> None:
    """Refuse to add bundles to ``arm_dir`` if its existing provenance disagrees.

    Checked against the first existing bundle found, the same pattern as
    ``videomae_features.assert_output_dir_matches_variant``: nothing downstream
    would otherwise notice a mixed-provenance directory until a much later audit,
    and the resume path (skip whatever already exists) means a differently
    configured re-run would silently write the remainder under the wrong identity.
    """
    if not arm_dir.exists():
        return
    existing_path = next(arm_dir.rglob("*.npz"), None)
    if existing_path is None:
        return
    existing = read_bundle_provenance(existing_path)
    mismatches = provenance_mismatches(existing, provenance)
    if mismatches:
        raise RuntimeError(
            f"{arm_dir} already holds bundles with different provenance ({existing_path.name}): "
            f"{mismatches}. Use a new --run-dir, a new arm directory, or --overwrite."
        )
