"""Loads the worker's own credentials from ``data/models/nlf/worker.env`` (gitignored under
``data/*``; see the plan's "Worker credentials" note). Plain ``KEY=VALUE`` lines, blanks and
``#`` comments ignored, matching values may be quoted; real environment variables always win
over the file, so a one-off override never needs editing the file on disk.

Two layers:

* ``load_worker_settings`` -- the OPTIONAL runtime knobs (model paths, poll interval, sample
  rate, batch size, temp dir), each with a default. Needs no credentials, so the offline
  ``--dry-run`` uses it on its own.
* ``load_worker_env`` -- those settings plus the 8 REQUIRED credential keys, which the real
  worker loop needs; it raises naming every missing one.

Secret-bearing fields are excluded from the dataclass ``repr`` so a stray print or traceback
never shows them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_PATH = "data/models/nlf/worker.env"

_REQUIRED_KEYS = (
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "NLF_WORKER_EMAIL",
    "NLF_WORKER_PASSWORD",
    "NLF_R2_ACCOUNT_ID",
    "NLF_R2_ACCESS_KEY_ID",
    "NLF_R2_SECRET_ACCESS_KEY",
    "NLF_R2_BUCKET",
)


@dataclass(frozen=True, kw_only=True)
class WorkerSettings:
    """Optional runtime knobs. Relative paths are relative to the repository root."""

    NLF_MODEL_PATH: str = "data/models/nlf/nlf_l_multi.torchscript"
    NLF_MODEL_FP32_PATH: str = "data/models/nlf/nlf_l_multi_fp32.torchscript"
    NLF_SMPL_PKL: str = "data/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    NLF_POLL_SECONDS: float = 60.0
    NLF_SAMPLE_FPS: float = 10.0
    NLF_BATCH: int = 8
    NLF_TMP_DIR: str = "data/runtime/nlf_worker_tmp"


@dataclass(frozen=True, kw_only=True)
class WorkerEnv(WorkerSettings):
    """Credentials plus settings. ``kw_only`` lets these required fields follow the
    defaulted ones inherited from ``WorkerSettings``."""

    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str = field(repr=False)
    NLF_WORKER_EMAIL: str
    NLF_WORKER_PASSWORD: str = field(repr=False)
    NLF_R2_ACCOUNT_ID: str
    NLF_R2_ACCESS_KEY_ID: str = field(repr=False)
    NLF_R2_SECRET_ACCESS_KEY: str = field(repr=False)
    NLF_R2_BUCKET: str


_OPTIONAL_KEYS = tuple(f.name for f in fields(WorkerSettings))
# Numeric settings: parsed with this type and required to be positive.
_NUMERIC = {"NLF_POLL_SECONDS": float, "NLF_SAMPLE_FPS": float, "NLF_BATCH": int}


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = _strip_quotes(value.strip())
    return values


def _overlay_environ(values: dict[str, str], keys: tuple[str, ...]) -> dict[str, str]:
    for key in keys:
        env_value = os.environ.get(key)
        if env_value:
            values[key] = env_value
    return values


def _settings_from(values: dict[str, str]) -> dict[str, object]:
    """Typed optional settings from raw strings; a missing/blank key keeps its default."""
    out: dict[str, object] = {}
    for key in _OPTIONAL_KEYS:
        raw = values.get(key)
        if not raw:
            continue
        caster = _NUMERIC.get(key)
        if caster is None:
            out[key] = raw
            continue
        try:
            number = caster(raw)
        except ValueError:
            raise ValueError(f"{key} must be a {caster.__name__}, got {raw!r}") from None
        if number <= 0:
            raise ValueError(f"{key} must be positive, got {raw!r}")
        out[key] = number
    return out


def load_worker_settings(path: str | Path = DEFAULT_PATH) -> WorkerSettings:
    """Only the optional settings: the file at ``path`` if it exists, real environment variables
    on top, defaults for the rest. Never requires a credential. Raises ``ValueError`` for a
    numeric setting that isn't a positive number."""
    values = _overlay_environ(_parse_file(path), _OPTIONAL_KEYS)
    return WorkerSettings(**_settings_from(values))


def load_worker_env(path: str | Path = DEFAULT_PATH) -> WorkerEnv:
    """Parse ``path``, then apply any matching real environment variables on top. Raises
    ``ValueError`` naming every required key still missing after that (never their values)."""
    values = _overlay_environ(_parse_file(path), _REQUIRED_KEYS + _OPTIONAL_KEYS)

    missing = [key for key in _REQUIRED_KEYS if not values.get(key)]
    if missing:
        raise ValueError(f"Missing required worker env vars: {', '.join(missing)}")

    return WorkerEnv(
        **{key: values[key] for key in _REQUIRED_KEYS},
        **_settings_from(values),
    )
