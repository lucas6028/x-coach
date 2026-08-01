"""Gate experiment -- is there ANY headroom in combining the direct image->3D models?

Every fusion idea (averaging, uncertainty weighting, per-frame routing, hybrid skeletons)
presupposes that the models fail on *different* frames. If NLF, HMR2.0, Multi-HMR and MeTRAbs
are all wrong on the same frames, no combination rule can help and the question is closed
before any fusion is built. This module measures that directly, on predictions already on
disk -- no new inference.

What is compared
----------------
Only the **rotation-invariant cue angles** (knee, hip). That is not a convenience: the models
use different body conventions whose joints sit at different anatomical points (offsets up to
176 mm on thorax), HMR2.0 regresses orientation in the *crop* frame, and Multi-HMR/NLF ran on
assumed FOVs while MeTRAbs got Fit3D's real intrinsics. Coordinates, per-axis depth and
gravity-dependent cues all carry those parity terms, so a "fusion gain" measured on them could
be a calibration artifact. The knee/hip angles are invariant to all of it (see
``notes/fit3d_model_comparison_summary.md``), so they are the only safe currency here.

Frames are intersected across models before anything is computed: the sparse models are stored
at full array length with NaN off-grid (strides: NLF/HMR2 1, Multi-HMR 6, MeTRAbs/MediaPipe 15,
all starting at frame 0), so every model is read on exactly the same frames.

What is reported, and how to read it
------------------------------------
Error is signed (``pred_cue - gt_cue``) because the two fusion families need different things:

* **Averaging** helps only if the *signed* errors are decorrelated (or oppositely biased) --
  they cancel. Measured by ``corr_signed`` and by actually computing the mean/median-fused
  error. Because the cue is a scalar and error is linear in it, the fused error is exactly the
  mean/median of the per-model errors -- no re-derivation needed.
* **Routing** helps only if the *absolute* errors are decorrelated -- a different model is best
  on different frames. Measured by ``corr_abs`` (Spearman) and by the oracle upper bounds.

``oracle_frame`` / ``oracle_seq`` pick the best model per frame / per sequence **using the GT**.
They are unreachable ceilings, not methods: a real router has no GT. If even the oracle barely
beats the best single model, every router loses too, and the fusion question is answered
negatively. ``switch_rate`` says whether an oracle router would even change its mind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fit3d import dataset as ds
from src.fit3d import decision_eval as dec
from src.fit3d import depth_eval as de
from src.fit3d.biomech import IMAGE2D, WORLD3D, frame_metrics

# Rotation-invariant cues only -- the parity-safe currency (see module docstring).
CUE_NAMES: tuple[str, ...] = ("knee_angle", "hip_angle")

PROJ2D = "proj2d"


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank of ``x`` (ties share their mean rank).

    ``argsort(argsort(x))`` is NOT a rank function under ties -- it breaks them arbitrarily and
    fabricates correlation on constant/degenerate columns. That bug produced a spurious rho in
    the uncertainty experiment; this is the corrected helper (cross-checked against
    ``scipy.stats.rankdata`` in the tests).
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i + 1
        while j < len(sx) and sx[j] == sx[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN-safe and degenerate-safe."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    x, y = a[m], b[m]
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return _corr(rankdata(a[m]), rankdata(b[m]))


def _cue_error_series(pred_world: np.ndarray, gt_metrics: dict[str, np.ndarray],
                      mode: str, n: int) -> dict[str, np.ndarray]:
    """Signed per-frame cue error (pred - gt) for the rotation-invariant cues."""
    pm = frame_metrics(pred_world[:n], mode)
    return {c: pm[c][:n] - gt_metrics[c][:n] for c in CUE_NAMES}


def collect_signed_errors(
    models: dict[str, str | Path],
    action: str = "squat",
    split: str = "train",
    source: str = "smpl3d",
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    include_proj2d: bool = True,
    subjs: list[str] | None = None,
) -> dict:
    """Per-frame signed cue error for every model, on the frames ALL of them cover.

    ``include_proj2d`` adds the GT-projected single-view 2D reading as an extra arm -- a
    *perfect* 2D detector (``notes/fit3d_2d_vs_3d_summary.md`` showed a real one, RTMPose, is
    indistinguishable from it on the knee cue), so it tests whether 2D is complementary to the
    3D models without needing the detector pipeline.
    """
    names = list(models)
    arms = names + ([PROJ2D] if include_proj2d else [])
    per_cue: dict[str, list[np.ndarray]] = {c: [] for c in CUE_NAMES}
    seq_ids: list[np.ndarray] = []
    cam_ids: list[np.ndarray] = []
    seq_names: list[str] = []

    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        j3d_m = ds.load_joints3d(split, subj, action, root)      # world metres
        j3d_mm = j3d_m * 1000.0
        gm = frame_metrics(j3d_m, WORLD3D)
        for cam in ds.cameras(split, subj, root):
            paths = {n: Path(models[n]) / de.pred_npz_name(subj, action, cam) for n in names}
            if not all(p.exists() for p in paths.values()):
                continue
            cp = ds.read_cam_params(split, subj, cam, action, root)
            gt_cam = de.gt_in_camera_frame(j3d_mm, cp)

            worlds: dict[str, np.ndarray] = {}
            for n in names:
                pred_cam, _ = de.load_prediction_h36m17(paths[n], gt_cam[:, :17, :], source=source)
                worlds[n] = dec.nlf_world_points(pred_cam, cp)
            n_frames = min([len(j3d_m)] + [len(w) for w in worlds.values()])

            errs = {n: _cue_error_series(worlds[n], gm, WORLD3D, n_frames) for n in names}
            if include_proj2d:
                proj = ds.project_world_to_image(j3d_m, cp)
                errs[PROJ2D] = _cue_error_series(proj, gm, IMAGE2D, min(n_frames, len(proj)))

            # Intersect: keep only frames every arm (and the GT) reads finitely, for every cue.
            keep = np.ones(n_frames, dtype=bool)
            for c in CUE_NAMES:
                keep &= np.isfinite(gm[c][:n_frames])
                for a in arms:
                    e = errs[a][c]
                    keep &= np.isfinite(np.pad(e, (0, n_frames - len(e)), constant_values=np.nan))
            if not keep.any():
                continue
            for c in CUE_NAMES:
                per_cue[c].append(np.stack([errs[a][c][:n_frames][keep] for a in arms], axis=1))
            seq_ids.append(np.full(int(keep.sum()), len(seq_names), dtype=np.int64))
            cam_ids.append(np.full(int(keep.sum()), cam, dtype=object))
            seq_names.append(f"{subj}__{cam}")

    return {
        "action": action,
        "split": split,
        "arms": arms,
        "models": names,
        "seq_names": seq_names,
        "seq_id": np.concatenate(seq_ids) if seq_ids else np.zeros(0, dtype=np.int64),
        "cam_id": np.concatenate(cam_ids) if cam_ids else np.zeros(0, dtype=object),
        "err": {c: (np.concatenate(v, axis=0) if v else np.zeros((0, len(arms)))) for c, v in per_cue.items()},
    }


def debias_per_camera(err: np.ndarray, cam_id: np.ndarray) -> np.ndarray:
    """Remove each (arm, camera) constant offset -- the oracle per-view calibration control.

    THE decisive control for this experiment. The arms carry very different constant biases
    (the metric-3D models read the knee ~6 deg deep, a projected 2D view reads it ~18 deg
    shallow), so *any* averaging or per-frame oracle picks up a large gain purely from those
    biases cancelling / one arm's bias happening to sit near zero. That gain is a
    **calibration** effect: a single constant per arm buys it without any fusion at all.

    Only the gain that survives this debiasing is genuine complementarity. Per-camera (not
    per-sequence) matches ``decision_eval._debias``, the convention the whole project reports.
    """
    out = np.array(err, dtype=np.float64, copy=True)
    for c in np.unique(cam_id):
        m = cam_id == c
        out[m] -= np.nanmean(out[m], axis=0, keepdims=True)
    return out


def _oracle_per_sequence(abs_err: np.ndarray, seq_id: np.ndarray) -> float:
    """Mean abs error if an oracle picked one model per SEQUENCE (pooled over frames)."""
    total, count = 0.0, 0
    for s in np.unique(seq_id):
        m = seq_id == s
        per_model = abs_err[m].mean(axis=0)
        total += float(per_model.min()) * int(m.sum())
        count += int(m.sum())
    return total / count if count else float("nan")


def shuffled_oracle(err: np.ndarray, n_rep: int = 20, seed: int = 0) -> float:
    """Per-frame oracle after independently permuting each arm's errors across frames.

    Motivation: taking a min over M arms lowers the mean *by construction* -- an
    order-statistic effect that appears even when no arm is genuinely better on any particular
    frame. Shuffling each column independently preserves every arm's marginal error
    distribution while destroying the joint structure.

    **CONFOUNDED -- read the result narrowly.** A global column-wise shuffle destroys two
    things at once: cross-model dependence (the target) and *shared per-frame difficulty* (that
    the bottom of a squat is hard for every arm). Shared difficulty alone drives
    ``shuffled < oracle_frame`` with ZERO cross-model dependence: with ``e_fm = d_f * z_fm``
    (``z`` independent), a hard frame's real min stays ``d_f * min(z)``, but after shuffling
    that row is redrawn from the pooled marginal and mostly lands on easy frames. Fit3D's
    difficulty *is* shared (|error| rank corr 0.28-0.52), so this statistic cannot separate the
    two, and ``shuffled < oracle_frame`` must NOT be read as "routing is unexploitable" --
    "hard for everyone" is orthogonal to "arm A predictably beats arm B here".

    Use it only as a reminder that raw oracle headroom is partly order-statistic luck. The
    decisive routing test is a GT-free router evaluated under LOSO against the best single arm;
    the unconfounded descriptive statistic here is ``per_arm[*]['frac_best']`` (near-uniform =>
    no FIXED choice captures the headroom).
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        shuffled = np.stack([rng.permutation(err[:, j]) for j in range(err.shape[1])], axis=1)
        vals.append(float(np.abs(shuffled).min(axis=1).mean()))
    return float(np.mean(vals))


def analyse_cue(err: np.ndarray, arms: list[str], seq_id: np.ndarray) -> dict:
    """Complementarity + fusion ceilings for one cue. ``err`` is (n_frames, n_arms), signed."""
    if err.size == 0:
        return {"n_frames": 0}
    abs_err = np.abs(err)
    mae = abs_err.mean(axis=0)
    best_i = int(np.argmin(mae))
    oracle_frame = float(abs_err.min(axis=1).mean())
    argmin = abs_err.argmin(axis=1)

    return {
        "n_frames": int(err.shape[0]),
        "n_seq": int(len(np.unique(seq_id))),
        "per_arm": {a: {"mae": float(mae[i]), "bias": float(err[:, i].mean()),
                        "sd": float(err[:, i].std()),
                        "frac_best": float((argmin == i).mean())} for i, a in enumerate(arms)},
        "best_single": {"arm": arms[best_i], "mae": float(mae[best_i])},
        # Fusion ceilings (need GT -> upper bounds, not methods).
        "oracle_frame": oracle_frame,
        # Luck-only ceiling: what min-over-arms buys with NO frame-level structure at all.
        "shuffled_oracle": shuffled_oracle(err),
        "oracle_seq": _oracle_per_sequence(abs_err, seq_id),
        "headroom_frame": float(mae[best_i] - oracle_frame),
        "headroom_frame_pct": float(100.0 * (mae[best_i] - oracle_frame) / mae[best_i])
        if mae[best_i] > 1e-12 else 0.0,
        # Would an oracle router even change its mind? (0 => one model always wins => no routing)
        "switch_rate": float((argmin != best_i).mean()),
        # Realisable, GT-free fusions.
        "mean_fusion_mae": float(np.abs(err.mean(axis=1)).mean()),
        "median_fusion_mae": float(np.abs(np.median(err, axis=1)).mean()),
        "corr_signed": [[_corr(err[:, i], err[:, j]) for j in range(len(arms))] for i in range(len(arms))],
        "corr_abs": [[_spearman(abs_err[:, i], abs_err[:, j]) for j in range(len(arms))]
                     for i in range(len(arms))],
    }


def analyse_cue_both(err: np.ndarray, arms: list[str], seq_id: np.ndarray,
                     cam_id: np.ndarray) -> dict:
    """The same analysis on raw errors and on per-camera-debiased errors.

    ``raw`` answers "does combining help as deployed"; ``debiased`` answers "does combining
    still help once each arm's constant offset is removed" -- i.e. is there complementarity
    beyond bias diversity. Only the debiased track licenses a fusion claim.
    """
    if err.size == 0:
        return {"raw": {"n_frames": 0}, "debiased": {"n_frames": 0}}
    return {
        "raw": analyse_cue(err, arms, seq_id),
        "debiased": analyse_cue(debias_per_camera(err, cam_id), arms, seq_id),
    }


def analyse(collected: dict) -> dict:
    arms, seq_id, cam_id = collected["arms"], collected["seq_id"], collected["cam_id"]
    return {
        "action": collected["action"],
        "split": collected["split"],
        "arms": arms,
        "n_seq": len(collected["seq_names"]),
        "per_cue": {c: analyse_cue_both(collected["err"][c], arms, seq_id, cam_id)
                    for c in CUE_NAMES},
    }


def run(models: dict[str, str | Path], action: str = "squat", split: str = "train",
        source: str = "smpl3d", root: Path = ds.DEFAULT_FIT3D_ROOT,
        include_proj2d: bool = True, subjs: list[str] | None = None) -> dict:
    return analyse(collect_signed_errors(models, action, split, source, root,
                                         include_proj2d, subjs))


def format_report(result: dict) -> str:
    arms = result["arms"]
    lines = [
        f"Fit3D model-fusion gate -- action={result['action']} split={result['split']}",
        f"  {result['n_seq']} sequences; rotation-invariant cues only (parity-safe across body conventions);",
        "  all arms read on the SAME frames (intersection of each model's stride grid).",
    ]
    for cue, both in result["per_cue"].items():
        raw, deb = both["raw"], both["debiased"]
        if not raw.get("n_frames"):
            lines += ["", f"  [{cue}] no common frames"]
            continue
        lines += [
            "",
            f"  [{cue}]  n_frames={raw['n_frames']}  n_seq={raw['n_seq']}",
            f"  {'arm':<12}{'MAE':>8}{'bias':>9}{'sd':>8}{'best%':>8}"
            f"{'| MAE-deb':>11}{'best%-deb':>11}",
            "  " + "-" * 68,
        ]
        for a in arms:
            s, d = raw["per_arm"][a], deb["per_arm"][a]
            lines.append(f"  {a:<12}{s['mae']:>8.2f}{s['bias']:>9.2f}{s['sd']:>8.2f}"
                         f"{s['frac_best'] * 100:>7.0f}%{d['mae']:>11.2f}{d['frac_best'] * 100:>10.0f}%")
        lines += [
            "  (MAE-deb = after removing each arm's per-camera constant offset: what ONE calibration",
            "   constant per arm buys, with no fusion at all -- the control every fusion number must beat)",
            "",
            f"  {'':<24}{'RAW':>10}{'DEBIASED':>12}",
            "  " + "-" * 48,
        ]
        for label, key, fmt in [
            ("best single arm", "best_single", "arm+mae"),
            ("mean-fusion (all arms)", "mean_fusion_mae", "f"),
            ("median-fusion", "median_fusion_mae", "f"),
            ("ORACLE per-sequence", "oracle_seq", "f"),
            ("ORACLE per-frame", "oracle_frame", "f"),
            ("  ...shuffled (luck only)", "shuffled_oracle", "f"),
        ]:
            if fmt == "arm+mae":
                lines.append(f"  {label:<24}{raw[key]['mae']:>10.2f}{deb[key]['mae']:>12.2f}"
                             f"   ({raw[key]['arm']} / {deb[key]['arm']})")
            else:
                lines.append(f"  {label:<24}{raw[key]:>10.2f}{deb[key]:>12.2f}")
        lines += [
            f"  {'oracle headroom':<24}{raw['headroom_frame_pct']:>9.0f}%{deb['headroom_frame_pct']:>11.0f}%",
            f"  {'oracle switch rate':<24}{raw['switch_rate'] * 100:>9.0f}%{deb['switch_rate'] * 100:>11.0f}%",
            "",
            "  DEBIASED signed-error correlation (high => shared errors => averaging cannot cancel them):",
            "  " + "".join(f"{a:>11}" for a in [""] + arms),
        ]
        for i, a in enumerate(arms):
            lines.append(f"  {a:>10} " + "".join(f"{deb['corr_signed'][i][j]:>11.2f}" for j in range(len(arms))))
        lines += ["", "  DEBIASED |error| rank correlation (high => the same frames are hard for everyone",
                  "  => routing is futile):", "  " + "".join(f"{a:>11}" for a in [""] + arms)]
        for i, a in enumerate(arms):
            lines.append(f"  {a:>10} " + "".join(f"{deb['corr_abs'][i][j]:>11.2f}" for j in range(len(arms))))
    return "\n".join(lines)


def to_json(result: dict) -> dict:
    return result


# --------------------------------------------------------------------------------------
# Rep-extreme decomposition -- WHY a model with the better cue MAE can lose the verdict.
# --------------------------------------------------------------------------------------

def decompose_rep_extreme(series_pred: np.ndarray, series_gt: np.ndarray,
                          reducer: str) -> dict[str, float] | None:
    """Split a model's rep reading into "accurate at the extreme" vs "picks the right frame".

    The verdict thresholds ``nanmin``/``nanmax`` over a rep, not the frame mean, so a model can
    have the lower mean-over-frames error and still read the rep worse. Two distinct ways:

    * ``point_err``   -- error ON the frame the GT calls the extreme. Large => the model is bad
      exactly where the coach looks, regardless of frame selection.
    * ``extreme_err`` -- error of the reported extreme *value* (what the verdict thresholds).
    * ``frame_offset``-- distance between the model's own extreme frame and the GT's. Large
      ``extreme_err`` with small ``point_err`` means the model reads the right frame fine but
      *selects the wrong frame* -- a selection effect, not an accuracy one.

    ``pooled_err`` is the published cue MAE restricted to the same frames, so all four are
    directly comparable. Returns None if the window has no usable frames.

    Everything is computed on the frames the model actually SAMPLED (both series restricted to
    where each is finite). Searching the GT extreme at full rate instead would almost never land
    on a subsampled model's grid, leaving ``point_err`` undefined -- and would silently fold the
    sampling penalty back in, which is the confound experiment 0.5 exists to remove.
    """
    both = np.isfinite(series_gt) & np.isfinite(series_pred)
    if not both.any():
        return None
    idx = np.flatnonzero(both)
    gt, pr = series_gt[idx], series_pred[idx]
    take = np.argmin if reducer == "min" else np.argmax

    i_gt, i_pr = int(take(gt)), int(take(pr))
    return {
        "point_err": float(abs(pr[i_gt] - gt[i_gt])),
        "extreme_err": float(abs(pr[i_pr] - gt[i_gt])),
        "frame_offset": float(abs(idx[i_pr] - idx[i_gt])),
        "pooled_err": float(np.mean(np.abs(pr - gt))),
    }


def _discrimination(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """How well a model's rep readings ORDER reps -- which is what a verdict actually needs.

    A verdict thresholds the reading, so what matters near the threshold is whether reps keep
    their true order, not how small the average error is. The two come apart in a specific,
    measurable way: a model that shrinks its predictions toward the population mean earns a
    *lower* MAE while compressing the very spread the threshold has to cut through.

    * ``slope``     -- OLS slope of reading on truth. 1 = faithful spread; < 1 = shrunk.
    * ``spread_ratio`` -- sd(reading) / sd(truth). Same story, scale-free.
    * ``r``         -- Pearson correlation with the truth: pure ordering fidelity, immune to
      both offset and scale, so it isolates discrimination from calibration.
    """
    if len(pairs) < 3:
        return {"slope": float("nan"), "spread_ratio": float("nan"), "r": float("nan"), "n_rep": 0}
    g = np.array([p[0] for p in pairs], dtype=np.float64)
    p = np.array([p[1] for p in pairs], dtype=np.float64)
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if g.size < 3 or g.std() < 1e-9:
        return {"slope": float("nan"), "spread_ratio": float("nan"), "r": float("nan"),
                "n_rep": int(g.size)}
    return {
        "slope": float(np.cov(p, g, bias=True)[0, 1] / g.var()),
        "spread_ratio": float(p.std() / g.std()),
        "r": _corr(p, g),
        "n_rep": int(g.size),
    }


def rep_extreme_decomposition(
    models: dict[str, str | Path],
    action: str = "squat",
    cue: str = "knee_angle",
    split: str = "train",
    source: str = "smpl3d",
    root: Path = ds.DEFAULT_FIT3D_ROOT,
    frame_stride: int = 15,
    min_rep_frames: int = 5,
    subjs: list[str] | None = None,
) -> dict:
    """Run :func:`decompose_rep_extreme` over every (subject, camera, rep) for each model.

    ``frame_stride`` puts every model on the same grid (default 15 = MeTRAbs/MediaPipe's), so
    the comparison is not confounded by sample count -- the point of experiment 0.5.
    """
    from src.fit3d.biomech import REP_REDUCERS

    reducer = REP_REDUCERS[cue]
    acc: dict[str, dict[str, list[float]]] = {
        n: {k: [] for k in ("point_err", "extreme_err", "frame_offset", "pooled_err")}
        for n in models
    }
    # Paired rep-level extremes, for the DISCRIMINATION metrics (see _discrimination).
    pairs: dict[str, list[tuple[float, float]]] = {n: [] for n in models}
    n_reps = 0
    for subj in subjs or ds.subjects(split, root):
        if action not in ds.actions(split, subj, root):
            continue
        rep_ann = ds.load_rep_ann(split, subj, root).get(action)
        if not rep_ann:
            continue
        segments = ds.rep_segments(rep_ann)
        j3d_m = ds.load_joints3d(split, subj, action, root)
        gm = frame_metrics(j3d_m, WORLD3D)[cue]
        j3d_mm = j3d_m * 1000.0
        for cam in ds.cameras(split, subj, root):
            paths = {n: Path(models[n]) / de.pred_npz_name(subj, action, cam) for n in models}
            if not all(p.exists() for p in paths.values()):
                continue
            cp = ds.read_cam_params(split, subj, cam, action, root)
            gt_cam = de.gt_in_camera_frame(j3d_mm, cp)
            series: dict[str, np.ndarray] = {}
            for n in models:
                pred_cam, _ = de.load_prediction_h36m17(paths[n], gt_cam[:, :17, :], source=source)
                world = dec.mask_to_stride(dec.nlf_world_points(pred_cam, cp), frame_stride)
                series[n] = frame_metrics(world, WORLD3D)[cue]
            n_frames = min([len(gm)] + [len(s) for s in series.values()])
            for start, end in segments:
                e = min(end, n_frames)
                if e - start < min_rep_frames:
                    continue
                counted = False
                red = np.nanmin if reducer == "min" else np.nanmax
                gt_ext = float(red(gm[start:e])) if np.isfinite(gm[start:e]).any() else float("nan")
                for n in models:
                    seg = series[n][start:e]
                    d = decompose_rep_extreme(seg, gm[start:e], reducer)
                    if d is None:
                        continue
                    counted = True
                    for k, v in d.items():
                        acc[n][k].append(v)
                    if np.isfinite(seg).any() and np.isfinite(gt_ext):
                        pairs[n].append((gt_ext, float(red(seg[np.isfinite(seg)]))))
                n_reps += int(counted)
    return {
        "action": action, "cue": cue, "frame_stride": frame_stride, "reducer": reducer,
        "n_rep_camera": n_reps,
        "per_model": {n: {**{k: float(np.nanmean(v)) if v else float("nan") for k, v in d.items()},
                          **_discrimination(pairs[n])}
                      for n, d in acc.items()},
        "pairs": {n: [[float(a), float(b)] for a, b in v] for n, v in pairs.items()},
    }


def shrinkage_sweep(pairs: list[tuple[float, float]], fault_when: str = "high",
                    lambdas: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                                 1.1, 1.2, 1.3, 1.5)) -> list[dict]:
    """Shrink a model's readings toward their own mean and watch MAE and verdict-flip diverge.

    Demonstrates the mechanism behind "the verdict follows ordering fidelity, not error
    magnitude" *directly*, instead of inferring it from a three-model ranking. Each reading
    becomes ``mean + lam * (reading - mean)``; the result is then re-centred on the truth so
    only the SPREAD varies (otherwise shrinkage would also move the bias and confound the
    sweep). Shrinking toward the mean is the classic way to buy a lower MAE, and it is exactly
    what destroys the spread a verdict threshold has to cut through.

    Returns one row per ``lam`` with the resulting slope, MAE and swept verdict-flip. The
    signature of the mechanism is MAE bottoming out at ``lam < 1`` while flip keeps rising as
    ``lam`` falls.
    """
    from src.fit3d import decision_eval as _dec

    g = np.array([p[0] for p in pairs], dtype=np.float64)
    p = np.array([p[1] for p in pairs], dtype=np.float64)
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if g.size < 5:
        return []
    centre = float(p.mean())
    out = []
    for lam in lambdas:
        r = centre + lam * (p - centre)
        r = r - (r.mean() - g.mean())          # re-centre: vary spread only, not bias
        out.append({
            "lam": float(lam),
            "slope": float(np.cov(r, g, bias=True)[0, 1] / g.var()),
            "mae": float(np.mean(np.abs(r - g))),
            "swept_flip": float(_dec._swept_flip(g, r, fault_when)),
        })
    return out
