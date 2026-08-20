"""Loading and running the VideoMAE backbone, shared by both extractors.

Kept apart from ``src.video.videomae_pooling`` on purpose: the pooling arithmetic is
numpy-only so it stays unit-testable on machines with no GPU build of torch, while
everything here needs torch and transformers at import time.

Both extractors (REHAB24-6 repetitions, Fitness-AQA videos) compute the legacy and
the corrected token pooling from *one* forward pass, so the two arms share frames,
weights, clip sampling and library version by construction and a paired delta
measures the pooling fix and nothing else.
"""

from __future__ import annotations

import numpy as np

from src.video.videomae_pooling import layer_norm

try:
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
except ImportError as exc:  # pragma: no cover - imported at runtime on GPU/Colab/Kaggle
    raise SystemExit(
        "VideoMAE extraction requires `torch` and `transformers`.\n"
        "Install them with: pip install torch transformers accelerate timm"
    ) from exc


def assert_fc_norm_pretrained(weight: np.ndarray, bias: np.ndarray, model_name: str) -> None:
    """Fail loudly if ``fc_norm`` is still at LayerNorm default init.

    ``from_pretrained`` warns about missing keys but does not raise, so a checkpoint
    without ``fc_norm`` yields weight=1/bias=0 -- an identity-ish normalization that
    would look exactly like "corrected pooling doesn't help".
    """
    if np.allclose(weight, 1.0) and np.allclose(bias, 0.0):
        raise SystemExit(
            f"fc_norm of `{model_name}` is at LayerNorm default init (weight=1, bias=0), "
            "so it was NOT loaded from the checkpoint. Refusing to extract features that "
            "would silently misrepresent the classification path."
        )


def load_backbone(model_name: str, device: "torch.device") -> tuple["torch.nn.Module", np.ndarray, np.ndarray, float]:
    """Load the VideoMAE backbone plus the *pretrained* ``fc_norm`` parameters.

    ``fc_norm`` exists only on ``VideoMAEForVideoClassification``. Loading
    ``VideoMAEModel`` instead would leave no way to reproduce the classification
    representation, because ``use_mean_pooling=True`` checkpoints set the backbone's
    own ``layernorm`` to ``None``.
    """
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    if model.fc_norm is None:
        raise SystemExit(
            f"`{model_name}` has use_mean_pooling=False, so it has no fc_norm and the "
            "mean_pool_fc_norm mode is undefined for it. Use a mean-pooling checkpoint."
        )

    weight = model.fc_norm.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.fc_norm.bias.detach().cpu().numpy().astype(np.float32)

    assert_fc_norm_pretrained(weight, bias, model_name)

    backbone = model.videomae.to(device)
    backbone.eval()
    return backbone, weight, bias, float(model.config.layer_norm_eps)


def resolve_device(requested: str | None) -> "torch.device":
    """Pick a device, refusing a CUDA device whose kernels this torch build lacks.

    Kaggle hands out Tesla P100s (sm_60) whose torch build ships sm_70+ kernels only.
    The failure surfaces as ``no kernel image is available for execution on the
    device`` deep inside the first conv3d -- *after* the model has loaded and the run
    looks healthy. A real strided conv3d up front turns a 6-hour wasted run into a
    two-second fallback, so the probe uses the op that actually blew up rather than a
    cheap tensor allocation, which succeeds even on an unsupported architecture.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            print("Requested cuda but torch.cuda.is_available() is False; falling back to CPU.")
        return torch.device("cpu")

    try:
        probe = torch.nn.Conv3d(3, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)).to("cuda")
        with torch.no_grad():
            probe(torch.zeros(1, 3, 4, 8, 8, device="cuda"))
        torch.cuda.synchronize()
    except RuntimeError as exc:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(
            f"CUDA device {name} (sm_{capability[0]}{capability[1]}) failed a conv3d probe "
            f"with torch {torch.__version__}: {exc}\nFalling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")


def encode_clip(
    backbone: "torch.nn.Module",
    processor: "VideoMAEImageProcessor",
    frames: list[np.ndarray],
    device: "torch.device",
    fc_norm_weight: np.ndarray,
    fc_norm_bias: np.ndarray,
    fc_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass -> ``(legacy_first_token, mean_pool_fc_norm)`` for one clip."""
    inputs = processor(frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        hidden = backbone(pixel_values=pixel_values).last_hidden_state

    # Index 0 is the first patch token, not a CLS token (VideoMAE has none) --
    # reproduced only so the paired comparison can isolate the pooling fix.
    legacy = hidden[:, 0, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    mean_pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    corrected = layer_norm(mean_pooled, fc_norm_weight, fc_norm_bias, fc_norm_eps)
    return legacy, corrected
