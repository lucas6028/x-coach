"""The body of ``scripts/nlf_worker/run_worker.py``: builds the heavy adapters and runs the worker
in one of three modes (see ``cli.build_parser``):

* default -- poll Supabase every ``NLF_POLL_SECONDS`` until Ctrl+C;
* ``--once`` -- one poll, print its outcome;
* ``--dry-run CLIP --out DIR`` -- one synthetic job on a local clip with NO network: an in-memory
  Db, a local-directory store under DIR and a copying fetcher, with the real decoder, model and
  renderer. Writes DIR/meta.json plus each strip's 0 and 90 degree tiles as JPEGs. Needs no
  worker.env (optional settings still apply if the file exists).

Relative paths from worker.env resolve against the repo root, so a Task Scheduler start with a
different working directory still finds the model files.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.nlf_worker import cli, config, loop, result
from src.nlf_worker.loop import run_one

ROOT = Path(__file__).resolve().parents[3]
SIDE_TILE = loop.ANGLES // 4  # the 90-degree view


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def peak_memory_mib() -> float | None:
    """This process's peak working set (Windows) or peak RSS (elsewhere), in MiB; None if the
    platform won't say. Measured in-process because a venv's python.exe on Windows is only a
    launcher, so the real interpreter is a child that an outside observer would have to find."""
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage",
                    )
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            get_current = ctypes.windll.kernel32.GetCurrentProcess
            get_current.restype = wt.HANDLE
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            # Explicit argtypes: the pseudo-handle (-1) overflows ctypes' default int conversion.
            get_info.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]
            get_info.restype = wt.BOOL
            if not get_info(get_current(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PeakWorkingSetSize / 2**20
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux, bytes on macOS
        return peak / (2**20 if sys.platform == "darwin" else 2**10)
    except Exception:  # noqa: BLE001 -- a diagnostic only; never fail the run over it
        return None


def _log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def _build_compute(settings: config.WorkerSettings):
    from src.nlf_worker.runtime.decoder import PyAvDecoder
    from src.nlf_worker.runtime.model import NlfModel, cuda_available
    from src.nlf_worker.runtime.renderer import TurntableRenderer

    if not cuda_available():
        raise SystemExit("NLF worker: CUDA is not available (run under .venv-nlf on the GPU PC)")
    model = NlfModel(_repo_path(settings.NLF_MODEL_PATH), _repo_path(settings.NLF_MODEL_FP32_PATH))
    renderer = TurntableRenderer(
        _repo_path(settings.NLF_SMPL_PKL), tile_px=loop.TILE_PX, angles=loop.ANGLES
    )
    return model, PyAvDecoder(), renderer


def _run_kwargs(settings: config.WorkerSettings, model) -> dict:
    tmp_dir = _repo_path(settings.NLF_TMP_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sample_fps": settings.NLF_SAMPLE_FPS,
        "batch": settings.NLF_BATCH,
        "tmp_dir": str(tmp_dir),
        "model_sha256": model.model_sha256,
        "torch_version": model.torch_version,
    }


def _dry_run(args, settings: config.WorkerSettings) -> int:
    clip = Path(args.dry_run)
    if not clip.is_file():
        print(f"dry-run: clip not found: {clip}")
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    job = cli.dry_run_job(clip, args.movement, args.segment)
    db = cli.DryRunDb(job)
    store = cli.LocalDirStore(out)
    model, decoder, renderer = _build_compute(settings)
    timed_decoder, timed_model, timed_renderer = (
        cli.TimedDecoder(decoder), cli.TimedModel(model), cli.TimedRenderer(renderer)
    )
    try:
        outcome = run_one(
            db, store, cli.CopyFetcher(clip), timed_decoder, timed_model, timed_renderer,
            **_run_kwargs(settings, model),
        )
    finally:
        renderer.close()
    total = time.perf_counter() - t0

    print(f"outcome: {outcome}")
    if outcome != "done":
        for _, error in db.failed:
            print(f"error: {error}")
        return 1

    _, pose3d_key, meta = db.completed[0]
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pose3d_bytes = store.get(pose3d_key)
    pose3d = result.decode_pose3d(pose3d_bytes)
    up = np.asarray(pose3d["up_axis"], dtype=float)
    print(
        f"timing: total {total:.1f}s | decode {timed_decoder.seconds:.1f}s | "
        f"infer {timed_model.seconds:.1f}s ({timed_model.seconds / max(timed_model.frames, 1) * 1000:.0f} ms/frame) | "
        f"render {timed_renderer.seconds:.1f}s for {timed_renderer.strips} strip(s)"
    )
    print(
        f"samples: {timed_model.frames} at {settings.NLF_SAMPLE_FPS:g} fps, detected "
        f"{timed_model.detected}/{timed_model.frames}; up axis {np.round(up, 3).tolist()} "
        f"(tilt {np.degrees(np.arccos(min(1.0, abs(up[1])))):.1f} deg from image vertical)"
    )
    print(f"pose3d: {out / pose3d_key} ({len(pose3d_bytes) / 1024:.0f} KB)")
    print(f"key frames: {len(meta['key_frames'])}")
    tile = loop.TILE_PX
    for row in cli.describe_key_frames(pose3d, meta):
        i = row["index"]
        strip_key = meta["key_frames"][i]["strip_key"]
        data = store.get(strip_key)
        strip = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out / f"k{i:02d}_000.jpg"), strip[:, :tile])
        cv2.imwrite(str(out / f"k{i:02d}_090.jpg"), strip[:, SIDE_TILE * tile : (SIDE_TILE + 1) * tile])
        print(
            f"  k{i:02d} {row['kind']} rep={row['rep_index']} t={row['t_s']:.2f}s "
            f"knee={row['knee_deg']:.0f}deg hip-knee={row['hip_minus_knee_cm']:+.1f}cm "
            f"strip={len(data) / 1024:.0f}KB {strip.shape[1]}x{strip.shape[0]}"
        )
    print(f"meta: {out / 'meta.json'}")
    peak = peak_memory_mib()
    print(f"peak memory (working set): {peak:.0f} MiB" if peak is not None else "peak memory: unknown")
    return 0


def _worker(args, env: config.WorkerEnv) -> int:
    from src.nlf_worker.runtime.r2_store import R2Fetcher, R2Store, make_s3_client
    from src.nlf_worker.runtime.supabase_db import SupabaseDb

    db = cli.LoggingDb(
        SupabaseDb(env.SUPABASE_URL, env.SUPABASE_ANON_KEY, env.NLF_WORKER_EMAIL, env.NLF_WORKER_PASSWORD),
        log=_log,
    )
    s3 = make_s3_client(env.NLF_R2_ACCOUNT_ID, env.NLF_R2_ACCESS_KEY_ID, env.NLF_R2_SECRET_ACCESS_KEY)
    store, fetcher = R2Store(s3, env.NLF_R2_BUCKET), R2Fetcher(s3, env.NLF_R2_BUCKET)
    model, decoder, renderer = _build_compute(env)
    kwargs = _run_kwargs(env, model)

    def once() -> str:
        return run_one(db, store, fetcher, decoder, model, renderer, **kwargs)

    try:
        if args.once:
            outcome = once()
            print(outcome)
            return 1 if outcome == "failed" else 0
        _log(f"NLF worker started; polling every {env.NLF_POLL_SECONDS:g}s (Ctrl+C to stop)")
        cli.serve(once, env.NLF_POLL_SECONDS, log=_log)
        return 0
    except KeyboardInterrupt:
        _log("stopped")
        return 0
    finally:
        renderer.close()


def main(argv: list[str] | None = None) -> int:
    args = cli.parse_args(argv)
    env_path = _repo_path(args.env)
    try:
        settings = config.load_worker_settings(env_path) if args.dry_run else config.load_worker_env(env_path)
    except ValueError as exc:  # names missing/invalid keys only, never values of secrets
        print(f"config error ({env_path}): {exc}")
        return 2
    if args.dry_run:
        return _dry_run(args, settings)
    return _worker(args, settings)
