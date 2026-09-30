"""Read the SMPL triangle list out of the old SMPLify pickle without installing ``chumpy``.

The stock SMPL "basic model" pickle stores most of its content as ``chumpy`` objects, which
pulls in a large, unmaintained dependency. Only the face/triangle array (key ``'f'``, a plain
numpy array) is needed to render a mesh, so every ``chumpy`` class is stubbed out to a no-op
during unpickling; the stub's ``__setstate__`` silently discards whatever state chumpy would
have restored, and only ``'f'`` is read out of the resulting dict.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class _ChumpyStub:
    """A stand-in for any ``chumpy`` class. Never called except by ``pickle``, which restores
    state through ``__setstate__``; this discards it rather than trying to reconstruct it."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _SmplUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def load_smpl_faces(path: str | Path) -> np.ndarray:
    """Load the (N, 3) int64 triangle index array from an SMPL basic-model pickle at ``path``."""
    with open(path, "rb") as fh:
        data = _SmplUnpickler(fh, encoding="latin1").load()
    return np.asarray(data["f"], dtype=np.int64)
