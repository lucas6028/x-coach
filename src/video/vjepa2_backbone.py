"""Loading and running the V-JEPA 2 ViT-L encoder, pinned for the frozen-feature
comparison in ``notes/rehab24_videomae_vjepa2_validation_plan.md``.

Mirrors the split already used for VideoMAE (``src.video.videomae_backbone``):
model loading and the forward pass live here, dependency-heavy (torch,
transformers); the pure numpy preprocessing math is kept separately testable.
Unlike VideoMAE's ``fc_norm`` fix, there is no pooling correction to make here --
the plan asks for the encoder's own final hidden state, mean-pooled over tokens,
and nothing from the (22M-parameter) predictor head. ``skip_predictor=True`` means
the predictor never runs, so there is nothing to accidentally pool from it; the
assertion in :func:`encode_clip` is a second, load-bearing check of that fact
rather than a redundant one, because a future transformers upgrade could change
what ``skip_predictor`` returns without changing this file.

Checkpoint identity is pinned to the exact revision smoke-tested on this
machine's GPU (GTX 1660 Ti, transformers 5.5.0): a moving ``main`` branch would
make the comparison silently re-run against a different checkpoint on any later
Kaggle session, and the plan explicitly forbids that (Kaggle execution, step 2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
    from transformers import VJEPA2Model
except ImportError as exc:  # pragma: no cover - imported at runtime on GPU/Colab/Kaggle
    raise SystemExit(
        "V-JEPA 2 extraction requires `torch` and `transformers`.\nInstall them with: pip install torch transformers"
    ) from exc

#: The original V-JEPA 2 ViT-L checkpoint (NOT V-JEPA 2.1, NOT an action-conditioned
#: or action-classification variant) -- plan, Models and comparisons.
DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
#: Pinned to the revision smoke-tested locally, so a re-run never silently picks up
#: a moved `main` on the Hub.
DEFAULT_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
HIDDEN_SIZE = 1024
RESOLUTION = 256
#: `sum(p.numel() for p in model.parameters())` on the pinned revision, including
#: the ~22M-parameter predictor that this file's forward path never calls.
EXPECTED_PARAMETER_COUNT = 325_971_328

#: Same RGB scaling as VideoMAE's processor and as `VJEPA2VideoProcessor`'s own
#: `image_mean`/`image_std` for this checkpoint (checked directly against the
#: from_pretrained processor; no discrepancy from the plan's pinned values).
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_clip(frames: Sequence[np.ndarray]) -> np.ndarray:
    """RGB uint8 ``256x256`` frames -> a normalized ``(T, 3, 256, 256)`` float32 array.

    Pure numpy so it is unit-testable without touching torch. Geometry (letterbox
    + resize to 256x256) happens in the caller, immediately after decode -- this
    function only rescales to [0, 1], applies the ImageNet-style normalization the
    checkpoint expects, and moves channels to the front.
    """
    stacked = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames], axis=0) / 255.0
    normalized = (stacked - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32)


def pool_tokens(last_hidden_state: "torch.Tensor") -> np.ndarray:
    """Mean over the TOKEN axis (dim 1) of one clip's encoder output, not the batch axis.

    ``last_hidden_state`` is ``(batch, num_tokens, hidden)`` -- ``(1, 8192, 1024)``
    at 64 frames, ``(1, 2048, 1024)`` at 16 (``T/2 * 16 * 16`` tokens from the
    tubelet/patch grid). Batch size here is always 1; pooling the wrong axis would
    silently return something of the right shape after ``squeeze(0)`` only by
    accident, which is why this is its own tested function rather than inlined.
    """
    if last_hidden_state.dim() != 3:
        raise ValueError(f"Expected a (batch, tokens, hidden) tensor, got shape {tuple(last_hidden_state.shape)}.")
    pooled = last_hidden_state.mean(dim=1)
    return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def assert_expected_parameter_count(model: "torch.nn.Module", expected: int = EXPECTED_PARAMETER_COUNT) -> None:
    """Fail loudly if the loaded module's parameter count drifts from the pinned checkpoint.

    Catches both a wrong revision (e.g. V-JEPA 2.1, or an action-conditioned
    variant with extra heads) and a `from_pretrained` that silently drops or
    randomly initializes a submodule.
    """
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != expected:
        raise SystemExit(
            f"Loaded model has {actual:,} parameters, expected {expected:,}. This is not the pinned "
            f"`{DEFAULT_MODEL_NAME}`@`{DEFAULT_REVISION}` checkpoint; refusing to extract features from it."
        )


def load_backbone(
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str = DEFAULT_REVISION,
    device: "torch.device" = None,
    attn_implementation: str = "sdpa",
) -> "torch.nn.Module":
    """Load the pinned, frozen V-JEPA 2 encoder.

    Frozen for both grad and mode: `requires_grad_(False)` on every parameter,
    `eval()`, and the caller runs the forward pass under `torch.inference_mode()`
    (plan, Kaggle execution step 3: "Run inference mode ... one encoder at a time").
    """
    model = VJEPA2Model.from_pretrained(model_name, revision=revision, attn_implementation=attn_implementation)
    assert_expected_parameter_count(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def encode_clip(model: "torch.nn.Module", frames: Sequence[np.ndarray], device: "torch.device") -> np.ndarray:
    """One forward pass -> a ``(1024,)`` pooled feature vector for one clip.

    FP32 throughout: measured on this machine's GTX 1660 Ti, FP32 is *faster* than
    FP16 here (T=16: 0.76s vs 2.6s; T=64: 7.9s vs 14.8s) and the FP16-vs-FP32
    pooled-feature cosine similarity is 1.0, so there is no accuracy reason to
    prefer FP16 and a clear speed reason not to.

    ``skip_predictor=True`` means the ~22M-parameter predictor never runs; the
    assertion below is a second check of that (see module docstring) rather than
    a no-op, in case a future transformers version changes what an
    output-with-`skip_predictor` carries.
    """
    array = preprocess_clip(frames)
    tensor = torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(pixel_values_videos=tensor, skip_predictor=True)
    if getattr(outputs, "predictor_output", None) is not None:
        raise RuntimeError(
            "V-JEPA 2 forward pass produced a predictor_output despite skip_predictor=True; "
            "refusing to pool a representation that may include predictor activity."
        )
    return pool_tokens(outputs.last_hidden_state)
