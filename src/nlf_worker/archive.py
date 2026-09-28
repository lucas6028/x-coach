"""Patch the stock NLF TorchScript archive to run fully in float32 (stdlib only).

The stock ``nlf_l_multi.torchscript`` hard-codes two half-precision casts. On the home PC's GTX
1660 Ti (TU116) fp16 is broken: the pose stage returns NaN boxes/uncertainties, so every pose
fails the model's own plausibility filter, and the fp16 detector returns NaN or garbage boxes.
Rewriting the two casts' dtype code from 5 (``torch.half``) to 6 (``torch.float``) inside the
archive's serialized TorchScript code fixes both (plan: "Phase 0 results", 0.1-0.2).

A TorchScript file is a zip; only two ``code/`` entries change, and every other entry (the
weights under ``data/``) is copied through with its original ``ZipInfo``, so each entry keeps its
own compression type -- the archive mixes stored and deflated entries.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

# (entry-name suffix, needle, replacement). Each needle must occur exactly once in the whole
# archive: zero means this is not the archive the patch was written for (a silent unpatched copy
# would bring back the NaN outputs), more than one means the patch is ambiguous.
FP32_PATCHES: tuple[tuple[str, bytes, bytes], ...] = (
    (
        "code/__torch__.py",
        b"predict_multi_same_weights(torch.to(crops_flat, 5)",
        b"predict_multi_same_weights(torch.to(crops_flat, 6)",
    ),
    (
        "code/__torch__/nlf/pt/multiperson/person_detector.py",
        b"preds = (model).forward(torch.to(images, 5), )",
        b"preds = (model).forward(torch.to(images, 6), )",
    ),
)


def patch_to_fp32(
    src: str | Path,
    dst: str | Path,
    patches: tuple[tuple[str, bytes, bytes], ...] = FP32_PATCHES,
) -> None:
    """Write ``dst``: a copy of the TorchScript archive ``src`` with every patch applied.

    Raises ``ValueError`` (and writes nothing at ``dst``) unless each patch's needle occurs
    exactly once across all entries whose name ends with that patch's suffix. The output is
    written to ``dst + '.partial'`` and moved into place only on success, so an interrupted run
    never leaves a truncated archive that a later load would mistake for a finished one.
    """
    src, dst = Path(src), Path(dst)
    partial = dst.with_name(dst.name + ".partial")
    hits = [0] * len(patches)
    try:
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                for i, (suffix, old, new) in enumerate(patches):
                    if info.filename.endswith(suffix):
                        count = data.count(old)
                        hits[i] += count
                        if count:
                            data = data.replace(old, new)
                # writestr(ZipInfo, ...) reuses the entry's own compress_type and metadata.
                zout.writestr(info, data)
        bad = [
            f"{suffix!r}: needle found {hits[i]} time(s), expected exactly 1"
            for i, (suffix, _, _) in enumerate(patches)
            if hits[i] != 1
        ]
        if bad:
            raise ValueError(f"patch_to_fp32: {src} is not the expected NLF archive; " + "; ".join(bad))
        os.replace(partial, dst)
    finally:
        if partial.exists():
            partial.unlink()
