"""NLF-L on CUDA, fully in float32 -- ``Model`` for ``loop.run_one``.

Promotes the Phase 0 recipe (``data/models/nlf/nlf_fp32_loader.py``): the stock model's fp16
path is broken on the GTX 1660 Ti, so the worker loads the fp32-patched archive (created from the
stock one on first use by ``archive.patch_to_fp32``), converts the crop model, the detector and
the canonical points to float32, and precomputes the SMPL canonical weights in float32. Every
call runs under ``torch.jit.optimized_execution(False)`` by default, which Phase 0 measured as
bit-identical across repeats (on: at most 0.5 mm jitter, about 2x faster).

The model is loaded lazily on the first ``infer``, so an idle worker holds no GPU memory.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Imported for its side effect: it registers the torchvision ops (e.g. NMS) that the archive's
# TorchScript detector code calls, which must exist before torch.jit.load (Phase 0 recipe).
import torchvision  # noqa: F401
import torch

from src.nlf_worker.archive import patch_to_fp32

_EMPTY: dict[str, Any] = {"joints3d": None, "joint_uncertainties": None, "vertices3d": None}


def cuda_available() -> bool:
    return torch.cuda.is_available()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class NlfModel:
    """``Model.infer`` over NLF-L's ``detect_smpl_batched``, keeping the largest person per frame.

    Outputs are NLF's native units: ``joints3d`` (24, 3) SMPL-24 and ``vertices3d`` (6890, 3) in
    camera-coordinate mm, ``joint_uncertainties`` (24,) in mm. A frame with no detection gets
    ``None`` for all three.
    """

    def __init__(
        self,
        model_path: str | Path,
        fp32_path: str | Path,
        *,
        optimized_execution: bool = False,
    ) -> None:
        self._fp32_path = Path(fp32_path)
        if not self._fp32_path.exists():
            self._fp32_path.parent.mkdir(parents=True, exist_ok=True)
            patch_to_fp32(model_path, self._fp32_path)
        self._optimized = optimized_execution
        self._model = None
        self._sha256: str | None = None
        self._lock = threading.Lock()

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the fp32 archive actually loaded (computed once, then cached)."""
        if self._sha256 is None:
            self._sha256 = _sha256_file(self._fp32_path)
        return self._sha256

    @property
    def torch_version(self) -> str:
        return torch.__version__

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            if not torch.cuda.is_available():
                raise RuntimeError("NlfModel: CUDA is not available")
            with torch.jit.optimized_execution(self._optimized):
                m = torch.jit.load(str(self._fp32_path)).cuda().eval()
                m.crop_model.float()
                m.detector.float()
                cano = {k: v.float() for k, v in m.cano_all.items()}
                m.cano_all = cano
                weights = m.weights
                with torch.inference_mode(), torch.device("cuda"):
                    w = m.get_weights_for_canonical_points(cano["smpl"])
                    weights["smpl"] = {k: v.float() for k, v in w.items()}
                m.weights = weights
            self._model = m
            return m

    def infer(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """One result per RGB uint8 (H, W, 3) frame, in order. Consecutive frames of the same
        size run as one batch (a WebM may change resolution mid-stream; a batch must not)."""
        results: list[dict[str, Any]] = []
        start = 0
        while start < len(frames):
            end = start + 1
            while end < len(frames) and frames[end].shape == frames[start].shape:
                end += 1
            results.extend(self._infer_same_size(frames[start:end]))
            start = end
        return results

    def _infer_same_size(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        m = self._load()
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous().cuda()
        with torch.jit.optimized_execution(self._optimized), torch.inference_mode(), torch.device("cuda"):
            pred = m.detect_smpl_batched(x, num_aug=1)
        boxes = pred.get("boxes")
        out: list[dict[str, Any]] = []
        for fi in range(len(frames)):
            joints = pred["joints3d"][fi]
            if joints.shape[0] == 0:
                out.append(dict(_EMPTY))
                continue
            # Largest box = the person being filmed. Boxes are (x, y, w, h, score).
            k = int(torch.argmax(boxes[fi][:, 2] * boxes[fi][:, 3])) if boxes is not None else 0
            out.append(
                {
                    "joints3d": joints[k].float().cpu().numpy(),
                    "joint_uncertainties": pred["joint_uncertainties"][fi][k].float().cpu().numpy(),
                    "vertices3d": pred["vertices3d"][fi][k].float().cpu().numpy(),
                }
            )
        return out
