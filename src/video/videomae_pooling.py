"""Pooling primitives for VideoMAE clip embeddings.

Two independent pooling axes turn a VideoMAE forward pass into one feature vector,
and both were previously wrong or unexamined in this repo:

**Token pooling** (over the 1568 patch tokens of one clip). The extractors used
``last_hidden_state[:, 0, :]`` and called it a CLS token. VideoMAE has no CLS token
-- ``transformers.models.videomae.modeling_videomae`` contains no ``cls_token`` at
all, and the encoder emits ``(batch, 1568, 768)`` = 8 tubelets x 196 patches with no
prepended summary token. Index 0 is therefore the top-left patch of the first
tubelet, not a clip representation. The classification path instead does
``sequence_output.mean(1)`` followed by ``fc_norm`` (a LayerNorm). Note that
``fc_norm`` lives only on ``VideoMAEForVideoClassification``: for checkpoints with
``use_mean_pooling=True`` the backbone's own ``layernorm`` is ``None``, so
``VideoMAEModel`` alone cannot reproduce the classification representation.

**Clip aggregation** (over the N clips sampled from one repetition). Both extractors
used element-wise ``max``. That is defensible for raw backbone activations but not
after ``fc_norm``: LayerNorm makes each clip vector zero-mean/unit-variance, so a
max over clips biases every dimension positive and discards the normalization that
was just applied. Aggregation is kept as an explicit, offline-selectable axis rather
than baked into extraction so that a change in token pooling and a change in clip
aggregation never land in the same measured delta.

This module is deliberately numpy-only: the torch forward pass belongs to the
extractor, and keeping the pooling arithmetic dependency-light makes it unit-testable
on machines with no GPU build of torch.
"""

from __future__ import annotations

import numpy as np

#: Token-pooling modes. ``legacy_first_token`` reproduces the historical (incorrect)
#: extraction so a paired comparison can isolate the pooling fix; it is not a
#: baseline anyone should adopt.
LEGACY_FIRST_TOKEN = "legacy_first_token"
MEAN_POOL_FC_NORM = "mean_pool_fc_norm"
TOKEN_POOLING_MODES = (LEGACY_FIRST_TOKEN, MEAN_POOL_FC_NORM)

#: Clip-aggregation modes. ``max`` is what the historical extraction used.
CLIP_AGGREGATIONS = ("max", "mean")


def layer_norm(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """Apply LayerNorm over the last axis, matching ``torch.nn.LayerNorm``.

    Used to apply the pretrained ``fc_norm`` to a mean-pooled clip vector without
    dragging torch into this module. torch normalizes with the *biased* variance,
    hence ``ddof=0``.
    """
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=-1, keepdims=True)
    variance = features.var(axis=-1, keepdims=True)  # ddof=0, as torch does
    normalized = (features - mean) / np.sqrt(variance + eps)
    return (normalized * weight + bias).astype(np.float32, copy=False)


def aggregate_clips(clip_features: np.ndarray, aggregation: str) -> np.ndarray:
    """Reduce a ``(num_clips, dim)`` stack to one ``(dim,)`` repetition vector."""
    if aggregation not in CLIP_AGGREGATIONS:
        raise ValueError(f"Unknown clip aggregation {aggregation!r}; expected one of {CLIP_AGGREGATIONS}.")
    stack = np.asarray(clip_features, dtype=np.float32)
    if stack.ndim != 2:
        raise ValueError(f"Expected a (num_clips, dim) stack, got shape {stack.shape}.")
    if stack.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty clip stack.")
    reduced = stack.max(axis=0) if aggregation == "max" else stack.mean(axis=0)
    return reduced.astype(np.float32, copy=False)


def feature_dir_name(token_pooling: str, aggregation: str) -> str:
    """Directory name for one (token pooling, clip aggregation) combination.

    Each such directory holds ``video_feature`` under the sample id, which is the
    contract ``videomae_video_classifier.build_samples`` already expects -- so the
    existing LOSO drivers consume these without modification.
    """
    if token_pooling not in TOKEN_POOLING_MODES:
        raise ValueError(f"Unknown token pooling {token_pooling!r}; expected one of {TOKEN_POOLING_MODES}.")
    if aggregation not in CLIP_AGGREGATIONS:
        raise ValueError(f"Unknown clip aggregation {aggregation!r}; expected one of {CLIP_AGGREGATIONS}.")
    return f"videomae_{token_pooling}_{aggregation}"


def build_provenance(
    model_name: str,
    clip_length: int,
    frame_stride: int,
    num_clips: int,
    transformers_version: str,
    token_pooling: str | None = None,
    clip_aggregation: str | None = None,
) -> dict[str, str]:
    """Provenance stamped into every ``.npz`` so features can never be silently mixed.

    Extraction writes the shared fields; the materialize step adds ``token_pooling``
    and ``clip_aggregation`` once it has committed to a combination.
    """
    record = {
        "model_name": model_name,
        "clip_length": str(clip_length),
        "frame_stride": str(frame_stride),
        "num_clips": str(num_clips),
        "transformers_version": transformers_version,
    }
    if token_pooling is not None:
        record["token_pooling"] = token_pooling
    if clip_aggregation is not None:
        record["clip_aggregation"] = clip_aggregation
    return record
