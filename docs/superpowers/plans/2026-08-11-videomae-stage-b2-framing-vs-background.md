# Stage B-2: Separating Framing from Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether person-crop's advantage over full-frame comes from removing the background or from showing the model a complete, larger athlete — and only then decide whether a person-crop fusion stage is worth running.

**Architecture:** Two new zero-cost-to-design video variants (`full_frame_letterbox`, `person_crop_fixed_scale`) added to the existing in-memory variant transform, extracted on Kaggle with the pipeline Stage B already built, then evaluated through the existing `run_stage_b_report.py`. No new analysis machinery — the point is to reuse Stage B's arms so the numbers are directly comparable.

**Tech Stack:** Python 3.12, numpy, OpenCV, PyTorch + transformers (Kaggle CPU), pytest. Interpreter is always `.venv\Scripts\python.exe` from the repo root.

## Global Constraints

- Interpreter: `.venv\Scripts\python.exe`. Never bare `python`/`pip`. Run everything from the repo root.
- Tests: `.venv\Scripts\python.exe -m pytest tests/` — always scope to `tests/`.
- New test files that import torch/cv2/mediapipe MUST be added to the `--ignore` list in `.github/workflows/ci.yml`.
- Seeds are `1, 2, 3, 4, 5` for every arm. Label mode is `combined`. All grid-runner hyperparameters stay at their defaults (epochs 20 / batch 32 / lr 3e-4 / hidden 128 / dropout 0.4 / weight decay 0.01 / patience 5 / threshold objective `balanced_accuracy`).
- Threshold and checkpoint are selected on **validation** only. Test is evaluated once per (arm, seed).
- Every new arm must pass `scripts/video/audit_videomae_features.py` at 1623/1623 before it is trained on.
- Every control arm must be verified to differ from `full_frame`: 0 identical feature bundles, except videos whose box legitimately covers the whole frame.
- The denominator is the **re-derived** normalized pose-only value `0.650 ± 0.012`, not the published 0.635. See `notes/videomae_stage_b_results.md` §2.1.

---

## Pre-registration (write down before any Stage B-2 number exists)

**This section is committed before Task 4 runs. It is the point of the plan.**

### The question

Stage B measured `person_crop` 0.666 against `full_frame` 0.640. `person_crop` changes four things at once (`notes/videomae_stage_b_results.md` §4.3):

1. background removed (100% → 31% of frame area kept)
2. body recovered from the processor's centre crop (53% of videos are non-square; 38% have part of the athlete cut off in the full-frame path, p90 14.6%)
3. effective body resolution 34.0% → 43.8% of the 224² input (~1.14× linear)
4. geometric normalisation (centred, canonical scale, grey letterbox bars)

**Primary question:** how much of the +0.026 is item 1, and how much is items 2–4?

### The primary comparison

| arm | background | complete body | canonical scale |
| --- | --- | --- | --- |
| `full_frame` (Stage B, 0.640) | yes | no — 38% truncated | no |
| **`full_frame_letterbox`** (new) | **yes** | **yes** | no |
| `person_crop` (Stage B, 0.666) | no | yes | no |
| **`person_crop_fixed_scale`** (new) | no | yes | **yes** |

**Primary metric:** `full_frame_letterbox` minus `full_frame`, test-split selected-threshold balanced accuracy, mean over seeds 1–5.

This is the framing/truncation effect **with background held constant**. The background effect is then `person_crop` minus `full_frame_letterbox`.

### Decision rules, fixed in advance

Let `L` = `full_frame_letterbox`, `F` = `full_frame` (0.640), `P` = `person_crop` (0.666).

| Outcome | Reading | Consequence |
| --- | --- | --- |
| `L − F ≥ +0.020` and `P − L ≤ +0.010` | The gain is framing/truncation. Background removal contributes nothing. | §4.1's fusion hypothesis is **wrong** and is rewritten. Do not run a person-crop fusion stage; run a **letterbox** fusion stage instead — it is cheaper and keeps the full frame. |
| `L − F ≤ +0.010` and `P − L ≥ +0.020` | The gain is background removal. | §4.1's hypothesis **survives**. Proceed to a pre-registered person-crop fusion stage. |
| both deltas in `(0.010, 0.020)` or both ≥ +0.020 | Both contribute, neither dominates. | Report the decomposition; a fusion stage must control for framing, i.e. use `person_crop_fixed_scale` and compare against `full_frame_letterbox`, not against `full_frame`. |
| `L − F < 0` | Letterboxing hurts — the grey bars cost more than the truncation. | Report and stop; the person-crop story needs a different control. |

**Nothing here is renegotiated after seeing the numbers.** If the result falls between rows, the third row applies.

### The secondary question, and why `person_crop_fixed_scale` exists

`person_crop` still leaks framing geometry: the letterbox bar proportion encodes the athlete's bounding-box aspect ratio, and `box_geometry` alone scores 0.578 (`notes/videomae_stage_b_results.md` §2.6). `person_crop_fixed_scale` resizes every crop to a fixed square before letterboxing, so box size no longer reaches the model — only its aspect ratio does.

**Secondary check:** re-run the `box_geometry` zero-parameter control on the features that actually reach the model in `person_crop_fixed_scale`. It should stay at 0.578 for the aspect-only subset; the *size* columns become uninformative by construction.

### What this stage does NOT do

- It does **not** revisit the Stage B retention verdict. Conditions 1, 2 and 5 failed on the pre-registered full-frame arm; that stands regardless of any result here.
- It does **not** run a fusion arm. Fusion is the *next* stage, and only if a decision rule above says so.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/video/squat_video_variants.py` (modify) | Add `full_frame_letterbox` and `person_crop_fixed_scale` to `VARIANTS` and to `apply_variant`; add `resize_square()` |
| `src/video/videomae_feature_extraction.py` (modify) | Add both names to its own `VARIANTS`; allow `full_frame_letterbox` to run without a manifest |
| `tests/test_squat_video_variants.py` (modify) | Behaviour tests for both new variants |
| `tests/test_videomae_squat_extraction.py` (modify) | The no-manifest rule for `full_frame_letterbox` |
| `.kaggle_tmp/fitaqa_videomae_extract_lbox/` (create) | Kernel for `full_frame_letterbox` |
| `.kaggle_tmp/fitaqa_videomae_extract_pcfs/` (create) | Kernel for `person_crop_fixed_scale` |
| `notes/videomae_stage_b2_results.md` (create) | Pre-registration + results |

---

### Task 1: Two new variants in the transform

**Files:**
- Modify: `src/video/squat_video_variants.py:60` (`VARIANTS`), `:224-239` (`apply_variant`)
- Test: `tests/test_squat_video_variants.py`

**Interfaces:**
- Consumes: `letterbox_to_square(frame, fill=LETTERBOX_FILL)`, `Box`, `apply_variant(frames, variant, box)` — all existing.
- Produces: `resize_square(frame, side)`; `VARIANTS` gains `"full_frame_letterbox"` and `"person_crop_fixed_scale"`; `apply_variant` handles both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_squat_video_variants.py`, inside a new class:

```python
class NewVariantTests(unittest.TestCase):
    """Stage B-2's separating controls. See docs/superpowers/plans/2026-08-11-videomae-stage-b2-framing-vs-background.md."""

    def test_full_frame_letterbox_keeps_all_pixels_and_squares_the_frame(self) -> None:
        """The whole point: background intact, but the processor's centre crop
        becomes a no-op so the athlete is no longer truncated."""
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        frame[10:30, 20:40] = 200
        out = apply_variant([frame], "full_frame_letterbox", None)[0]

        self.assertEqual(out.shape[:2], (64, 64))
        # every original pixel survives, just re-centred
        self.assertEqual(int(out[10 + 8, 20, 0]), 200)
        self.assertTrue((out[0:8] == 114).all())

    def test_full_frame_letterbox_ignores_the_box_entirely(self) -> None:
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        with_box = apply_variant([frame], "full_frame_letterbox", Box(10, 10, 20, 20))[0]
        without_box = apply_variant([frame], "full_frame_letterbox", None)[0]
        self.assertTrue(np.array_equal(with_box, without_box))

    def test_person_crop_fixed_scale_removes_box_size_from_the_output(self) -> None:
        """Two athletes at different distances must produce the same output size,
        so box SIZE can no longer reach the model -- only its aspect ratio."""
        frame = np.full((200, 200, 3), 30, dtype=np.uint8)
        frame[20:180, 60:140] = 200
        near = apply_variant([frame], "person_crop_fixed_scale", Box(60, 20, 140, 180))[0]
        far = apply_variant([frame], "person_crop_fixed_scale", Box(80, 60, 120, 140))[0]

        self.assertEqual(near.shape, far.shape)
        self.assertEqual(near.shape[0], near.shape[1])

    def test_person_crop_fixed_scale_preserves_aspect_ratio(self) -> None:
        """Aspect must survive: squashing a tall athlete into a square would
        destroy the very geometry the arm is meant to preserve."""
        frame = np.full((200, 200, 3), 30, dtype=np.uint8)
        frame[20:180, 90:110] = 200
        out = apply_variant([frame], "person_crop_fixed_scale", Box(90, 20, 110, 180))[0]

        column_is_body = (out[out.shape[0] // 2] >= 150).sum()
        row_is_body = (out[:, out.shape[1] // 2] >= 150).sum()
        self.assertLess(column_is_body, row_is_body)

    def test_resize_square_is_exact(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(resize_square(frame, 64).shape[:2], (64, 64))
```

Add `resize_square` to the import block at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_squat_video_variants.py -k NewVariant -v`
Expected: FAIL — `ImportError: cannot import name 'resize_square'`.

- [ ] **Step 3: Implement**

In `src/video/squat_video_variants.py`, extend the tuple:

```python
VARIANTS = (
    "person_crop",
    "background_only",
    "reencoded",
    "full_frame_letterbox",
    "person_crop_fixed_scale",
)
```

Add near `letterbox_to_square`:

```python
#: Side length every fixed-scale crop is resized to before letterboxing. 224 is what
#: the processor feeds the model, so resizing here neither adds nor destroys detail
#: relative to what VideoMAE actually sees.
FIXED_CROP_SIDE = 224


def resize_square(frame: np.ndarray, side: int = FIXED_CROP_SIDE) -> np.ndarray:
    """Resize the longest edge to ``side``, keeping aspect ratio."""
    height, width = frame.shape[:2]
    scale = side / max(height, width)
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
```

Rewrite the tail of `apply_variant` (currently lines 224-239):

```python
def apply_variant(frames: list[np.ndarray], variant: str, box: Box | None) -> list[np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    if variant == "reencoded":
        return frames

    if variant == "full_frame_letterbox":
        # Background untouched; only the framing changes. This is the arm that
        # separates "the background was noise" from "the model finally saw a whole
        # athlete": the processor's shortest-edge-224 + centre-crop-224 truncates
        # 38% of this dataset's videos, and squaring the frame makes it a no-op.
        return [letterbox_to_square(frame) for frame in frames]

    if box is None:
        # No person was ever visible. Both box variants degrade to the untouched
        # video; the manifest records these so they are reported, not silently counted.
        return frames

    if variant == "person_crop":
        return [letterbox_to_square(frame[box.y0 : box.y1, box.x0 : box.x1]) for frame in frames]

    if variant == "person_crop_fixed_scale":
        # As person_crop, but every crop is resized to a common side first, so the
        # athlete's box SIZE no longer reaches the model. Box geometry alone scores
        # 0.578 on this dataset, so leaving size in the frame leaks a known predictor.
        return [
            letterbox_to_square(resize_square(frame[box.y0 : box.y1, box.x0 : box.x1]))
            for frame in frames
        ]

    # Filled per frame rather than from one static plate, so lighting changes and
    # camera drift stay consistent with the untouched part of the scene.
    return [fill_box_from_surroundings(frame, box) for frame in frames]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_squat_video_variants.py -v`
Expected: PASS, all pre-existing tests included.

- [ ] **Step 5: Commit**

```bash
git add src/video/squat_video_variants.py tests/test_squat_video_variants.py
git commit -m "feat(variants): add full_frame_letterbox and person_crop_fixed_scale"
```

---

### Task 2: Let the extractor run the two new variants

**Files:**
- Modify: `src/video/videomae_feature_extraction.py:45` (`VARIANTS`), and the `--variant-manifest` requirement in `main()`
- Test: `tests/test_videomae_squat_extraction.py`

**Interfaces:**
- Consumes: `load_variant_boxes(manifest_path) -> dict[str, Box | None]` (existing, raises `SystemExit` on a box-less row).
- Produces: `VARIANTS_NEEDING_BOXES: tuple[str, ...]` — the variants for which `--variant-manifest` is mandatory.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_videomae_squat_extraction.py`:

```python
class VariantsNeedingBoxesTests(unittest.TestCase):
    def test_box_variants_are_listed_and_frame_only_variants_are_not(self) -> None:
        """full_frame_letterbox transforms the whole frame, so demanding a manifest
        for it would block the one arm that needs no athlete information at all."""
        self.assertIn("person_crop", VARIANTS_NEEDING_BOXES)
        self.assertIn("background_only", VARIANTS_NEEDING_BOXES)
        self.assertIn("person_crop_fixed_scale", VARIANTS_NEEDING_BOXES)
        self.assertNotIn("full_frame_letterbox", VARIANTS_NEEDING_BOXES)
        self.assertNotIn("full_frame", VARIANTS_NEEDING_BOXES)
```

Add `VARIANTS_NEEDING_BOXES` to the import block.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_videomae_squat_extraction.py -k VariantsNeedingBoxes -v`
Expected: FAIL — `ImportError: cannot import name 'VARIANTS_NEEDING_BOXES'`.

- [ ] **Step 3: Implement**

In `src/video/videomae_feature_extraction.py`, replace the `VARIANTS` line:

```python
VARIANTS = (
    "full_frame",
    "person_crop",
    "background_only",
    "reencoded",
    "full_frame_letterbox",
    "person_crop_fixed_scale",
)

#: Variants whose transform needs one box per video. The others act on the whole
#: frame and must NOT require a manifest.
VARIANTS_NEEDING_BOXES = ("person_crop", "background_only", "person_crop_fixed_scale")
```

In `main()`, replace the manifest check:

```python
    boxes: dict[str, Box | None] = {}
    if args.variant in VARIANTS_NEEDING_BOXES:
        if args.variant_manifest is None:
            raise SystemExit(f"--variant {args.variant} needs --variant-manifest to supply its boxes.")
        boxes = load_variant_boxes(args.variant_manifest)
        uncovered = [request.video_id for request in requests if request.video_id not in boxes]
        if uncovered:
            raise SystemExit(
                f"{len(uncovered)} videos are missing from {args.variant_manifest} "
                f"({uncovered[:5]}). A control arm covering fewer videos than the main arm "
                "is not a paired comparison."
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_videomae_squat_extraction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video/videomae_feature_extraction.py tests/test_videomae_squat_extraction.py
git commit -m "feat(videomae): only box variants require a manifest"
```

---

### Task 3: Local smoke test on real video, and a visual check

**Files:**
- Read only: `data/Fitness-AQA/Squat/Labeled_Dataset/videos/33048_1.mp4`

No new code. This is the step that caught both broken control designs in Stage B; skipping it is how a 3.4-hour extraction gets wasted.

- [ ] **Step 1: Extract 2 videos per new variant to the scratchpad**

```bash
R=data/Fitness-AQA/Squat/Labeled_Dataset
.venv/Scripts/python.exe scripts/video/run_videomae_feature_extraction.py \
  --splits test --limit 2 --device cpu --variant full_frame_letterbox \
  --output-dir <scratch>/b2_lbox
.venv/Scripts/python.exe scripts/video/run_videomae_feature_extraction.py \
  --splits test --limit 2 --device cpu --variant person_crop_fixed_scale \
  --variant-manifest $R/videos_person_crop/manifest.json \
  --output-dir <scratch>/b2_pcfs
```

Expected: `fc_norm loaded from checkpoint (weight mean=0.6832, bias mean=0.0083)` — the same values Stage A and Stage B saw. A different value means the checkpoint did not load and the run must be stopped.

- [ ] **Step 2: Dump one frame of each variant and LOOK at it**

```bash
.venv/Scripts/python.exe -c "
import cv2, json
from pathlib import Path
from src.video.squat_video_variants import apply_variant, Box
cap = cv2.VideoCapture('data/Fitness-AQA/Squat/Labeled_Dataset/videos/33048_1.mp4')
for _ in range(40): ok, frame = cap.read()
cap.release()
rows = {r['video_id']: r for r in json.load(open('data/Fitness-AQA/Squat/Labeled_Dataset/videos_person_crop/manifest.json'))['rows']}
box = Box(*rows['33048_1']['box'])
cv2.imwrite('<scratch>/b2_lbox.png', apply_variant([frame], 'full_frame_letterbox', None)[0])
cv2.imwrite('<scratch>/b2_pcfs.png', apply_variant([frame], 'person_crop_fixed_scale', box)[0])
"
```

Then `Read` both PNGs. Confirm: the letterbox image shows the **whole original scene** with grey bars and nothing cropped; the fixed-scale image shows the athlete only, undistorted.

- [ ] **Step 3: Verify the variants actually differ from full_frame**

```bash
.venv/Scripts/python.exe -c "
import numpy as np, glob, os
a = {os.path.basename(p): p for p in glob.glob(r'<scratch>/b2_lbox/**/*.npz', recursive=True)}
b = {os.path.basename(p): p for p in glob.glob(r'<scratch>/b2_pcfs/**/*.npz', recursive=True)}
for name in sorted(a):
    x = np.load(a[name])['clip_features_mean_pool_fc_norm']
    y = np.load(b[name])['clip_features_mean_pool_fc_norm']
    print(name, 'lbox vs pcfs cos', float((x*y).sum()/(np.linalg.norm(x)*np.linalg.norm(y))))
"
```

Expected: cosine clearly below 1.0. Identical vectors mean a transform silently did nothing — the Stage B manifest defect.

- [ ] **Step 4: Commit nothing, record the observation**

No code changed. Note the two `fc_norm` values and the visual result in the running log of `notes/videomae_stage_b2_results.md` §1 (created in Task 5).

---

### Task 4: Two Kaggle kernels

**Files:**
- Create: `.kaggle_tmp/fitaqa_videomae_extract_lbox/{fitaqa-videomae-extract-lbox.py,kernel-metadata.json}`
- Create: `.kaggle_tmp/fitaqa_videomae_extract_pcfs/{fitaqa-videomae-extract-pcfs.py,kernel-metadata.json}`
- Modify: `.kaggle_tmp/fitaqa_videomae_src/src.zip` (rebuild)

**Interfaces:**
- Consumes: datasets `haoping6028/fitaqa-videomae-src` and `haoping6028/fitaqa-squat-videos` (already uploaded).
- Produces: `/kaggle/working/videomae_raw_<variant>.zip`, one bundle per split video.

- [ ] **Step 1: Generate both kernel dirs from the existing template**

```bash
.venv/Scripts/python.exe -c "
import json, shutil
from pathlib import Path
template = Path('.kaggle_tmp/fitaqa_videomae_extract/fitaqa-videomae-extract.py').read_text(encoding='utf-8')
for suffix, variant, needs_box in (('lbox','full_frame_letterbox',False), ('pcfs','person_crop_fixed_scale',True)):
    out = Path(f'.kaggle_tmp/fitaqa_videomae_extract_{suffix}'); out.mkdir(parents=True, exist_ok=True)
    body = template.replace('VARIANT = \"full_frame\"', f'VARIANT = \"{variant}\"')
    if needs_box:
        body = body.replace('    split_dir = split_file.parent\n',
            '    split_dir = split_file.parent\n    variant_manifest = find_one(\"variant_manifests/person_crop.json\", \"the person_crop boxes\")\n')
        body = body.replace('        \"--variant\", VARIANT,\n',
            '        \"--variant\", VARIANT,\n        \"--variant-manifest\", str(variant_manifest),\n')
    (out / f'fitaqa-videomae-extract-{suffix}.py').write_text(body, encoding='utf-8')
    (out / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'haoping6028/fitaqa-videomae-extract-{suffix}',
        'title': f'fitaqa-videomae-extract-{suffix}',
        'code_file': f'fitaqa-videomae-extract-{suffix}.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': True, 'keywords': [],
        'dataset_sources': ['haoping6028/fitaqa-videomae-src', 'haoping6028/fitaqa-squat-videos'],
        'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
    }, indent=2), encoding='utf-8')
    print('staged', out)
"
```

`enable_gpu` is **False**: Kaggle's P100 is sm_60, this torch build ships sm_70+, and Stage B's probe fell back to CPU on every GPU kernel anyway. Requesting a GPU only consumes one of the two concurrent GPU slots.

- [ ] **Step 2: Rebuild and re-upload src.zip**

```bash
.venv/Scripts/python.exe -c "
import zipfile
from pathlib import Path
out = Path('.kaggle_tmp/fitaqa_videomae_src/src.zip'); out.unlink(missing_ok=True)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path('src').rglob('*.py')):
        if '__pycache__' in p.parts: continue
        z.write(p, p.as_posix())
src = zipfile.ZipFile(out).read('src/video/videomae_feature_extraction.py').decode()
assert 'VARIANTS_NEEDING_BOXES' in src, 'stale src.zip -- rebuild after Task 2'
print('src.zip rebuilt and verified')
"
(cd .kaggle_tmp/fitaqa_videomae_src && uv run --with kaggle kaggle datasets version -p . -m "Stage B-2 variants")
```

**`-p .` from inside the directory is mandatory** — the Kaggle CLI splices the path into a temp filename and any path containing a slash fails with `[Errno 2]`. Stage B lost two attempts to a stale `src.zip`; the assertion above is what stops a third.

- [ ] **Step 3: Push both kernels**

```bash
for d in fitaqa_videomae_extract_lbox fitaqa_videomae_extract_pcfs; do
  (cd .kaggle_tmp/$d && uv run --with kaggle kaggle kernels push -p .)
done
```

- [ ] **Step 4: Monitor to completion**

Use the Monitor tool with a poll loop over `kaggle kernels status`, emitting on every terminal state (not only success). Expect 3.4–5.7 h per kernel.

- [ ] **Step 5: Download, verify, materialize, audit**

```bash
R=data/Fitness-AQA/Squat/Labeled_Dataset
for pair in "lbox:full_frame_letterbox" "pcfs:person_crop_fixed_scale"; do
  suffix="${pair%%:*}"; variant="${pair##*:}"
  uv run --with kaggle kaggle kernels output haoping6028/fitaqa-videomae-extract-$suffix -p <scratch>/k_$suffix
  # unzip to $R/videomae_raw_$variant, then:
  .venv/Scripts/python.exe scripts/video/materialize_videomae_features.py \
    --raw-dir $R/videomae_raw_$variant --output-parent $R/$variant \
    --token-pooling mean_pool_fc_norm --aggregation mean
  .venv/Scripts/python.exe scripts/video/audit_videomae_features.py \
    $R/$variant/videomae_mean_pool_fc_norm_mean
done
```

Expected: `1623/1623 ... => PASS` for both, with `variant` correct in the printed provenance.

- [ ] **Step 6: Verify each new arm differs from full_frame**

```bash
.venv/Scripts/python.exe -c "
import numpy as np
from pathlib import Path
R = Path('data/Fitness-AQA/Squat/Labeled_Dataset')
ff = {p.stem: p for p in (R/'videomae_raw_full_frame').rglob('*.npz')}
for variant in ('full_frame_letterbox','person_crop_fixed_scale'):
    other = {p.stem: p for p in (R/f'videomae_raw_{variant}').rglob('*.npz')}
    same = sum(1 for k in ff if np.array_equal(np.load(ff[k])['clip_features_mean_pool_fc_norm'],
                                               np.load(other[k])['clip_features_mean_pool_fc_norm']))
    print(variant, 'identical to full_frame:', same, '/', len(ff))
"
```

Expected: `full_frame_letterbox` identical only for already-square videos (768 of 1623 are 480×480, and for those the letterbox is genuinely a no-op — that is correct, not a defect). `person_crop_fixed_scale` should be near 0.

- [ ] **Step 7: Commit the kernels**

```bash
git add .kaggle_tmp/fitaqa_videomae_extract_lbox .kaggle_tmp/fitaqa_videomae_extract_pcfs
git commit -m "chore(kaggle): Stage B-2 extraction kernels"
```

---

### Task 5: Pre-registration document, committed before any result is read

**Files:**
- Create: `notes/videomae_stage_b2_results.md`

- [ ] **Step 1: Write §0 with the decision rules from this plan verbatim**

Copy the "Pre-registration" section above into `notes/videomae_stage_b2_results.md` §0, in Traditional Chinese to match `notes/videomae_stage_b_results.md`. Include: the four-way confound table, the primary comparison, the four decision rows, the denominator (0.650), the seeds, and the "does not revisit Stage B" clause.

- [ ] **Step 2: Commit before running Task 6**

```bash
git add notes/videomae_stage_b2_results.md
git commit -m "docs(videomae): pre-register Stage B-2 before any result exists"
```

This commit MUST land before Task 6 runs. Its whole value is the timestamp.

---

### Task 6: Train both arms and apply the decision rules

**Files:**
- Modify: `notes/videomae_stage_b2_results.md` (results section)

- [ ] **Step 1: Train both arms, seeds 1-5, label mode combined**

```bash
R=data/Fitness-AQA/Squat/Labeled_Dataset
for variant in full_frame_letterbox person_crop_fixed_scale; do
  .venv/Scripts/python.exe scripts/video/run_videomae_experiment_grid.py \
    --feature-dir $R/$variant/videomae_mean_pool_fc_norm_mean \
    --train-keys $R/Splits/train_keys.json --val-keys $R/Splits/val_keys.json \
    --test-keys $R/Splits/test_keys.json \
    --forward-labels $R/Labels/error_knees_forward.json \
    --inward-labels $R/Labels/error_knees_inward.json \
    --output-root data/Fitness-AQA/Squat/experiments/videomae_$variant \
    --label-modes combined --normalize-features
done
```

- [ ] **Step 2: Read the two numbers and apply the pre-registered rule**

```bash
.venv/Scripts/python.exe -c "
import csv, statistics
def read(n):
    rows=[r for r in csv.DictReader(open(f'data/Fitness-AQA/Squat/experiments/{n}/metrics/experiment_summary.csv'))
          if r['split']=='test' and r['threshold_kind']=='selected_threshold' and r['label_mode']=='combined']
    return [float(r['balanced_accuracy']) for r in rows]
F = statistics.mean(read('videomae_corrected'))
L = statistics.mean(read('videomae_full_frame_letterbox'))
P = statistics.mean(read('videomae_person_crop'))
S = statistics.mean(read('videomae_person_crop_fixed_scale'))
print(f'full_frame            {F:.4f}')
print(f'full_frame_letterbox  {L:.4f}   L-F = {L-F:+.4f}  (framing/truncation)')
print(f'person_crop           {P:.4f}   P-L = {P-L:+.4f}  (background)')
print(f'person_crop_fixed     {S:.4f}   S-P = {S-P:+.4f}  (removing box size)')
"
```

Match the result against the four decision rows in §0. **Do not reinterpret.**

- [ ] **Step 3: Run the full report so every arm carries a paired CI**

```bash
E=data/Fitness-AQA/Squat/experiments
.venv/Scripts/python.exe scripts/video/run_stage_b_report.py \
  --pose-predictions $E/pose_only/predictions \
  --videomae-predictions $E/videomae_full_frame_letterbox/predictions \
  --arm full_frame=$E/videomae_corrected/predictions \
  --arm person_crop=$E/videomae_person_crop/predictions \
  --arm person_crop_fixed_scale=$E/videomae_person_crop_fixed_scale/predictions \
  --output $E/stage_b2_report.json
```

The denominator gate will print FAIL (0.650 vs the published 0.635) and the script will exit non-zero **after** saving the report. That is expected and documented in `notes/videomae_stage_b_results.md` §2.1 — the report is complete.

- [ ] **Step 4: Write the results section and the consequence**

Fill `notes/videomae_stage_b2_results.md` §2 with the table, and §3 with **whichever decision row fired**, stated as the row was written in §0. If the outcome contradicts §4.1 of the Stage B note, say so explicitly and amend that section.

- [ ] **Step 5: Run the full suite and commit**

```bash
.venv\Scripts\python.exe -m pytest tests/
git add notes/videomae_stage_b2_results.md
git commit -m "docs(videomae): Stage B-2 result -- <the decision row that fired>"
```

---

## Self-Review

**Spec coverage.** The question is "is the +0.026 background or framing?" — Task 1 builds `full_frame_letterbox` (the separator) and `person_crop_fixed_scale` (the size-leak control), Task 2 lets the extractor run them without a manifest where none is needed, Task 3 catches a broken transform before 3.4 h of compute, Task 4 extracts, Task 5 fixes the rules in advance, Task 6 applies them. The secondary question (box-size leak) is covered by `person_crop_fixed_scale` in Tasks 1 and 6.

**Placeholders.** None: every step has the command or the code it needs. `<scratch>` is a path the executing session substitutes.

**Type consistency.** `resize_square(frame, side=FIXED_CROP_SIDE)` is defined in Task 1 and used only there. `VARIANTS_NEEDING_BOXES` is defined in Task 2 and used in Task 2's test and `main()`. `apply_variant` keeps its signature `(frames, variant, box)`. Arm directory names (`videomae_full_frame_letterbox`, `videomae_person_crop_fixed_scale`) match between Task 4's materialize step, Task 6's training loop and Task 6's read function.

**One known asymmetry, deliberately not fixed.** `full_frame_letterbox` shrinks the athlete relative to `full_frame` (the whole scene is squeezed into 224² instead of a 224² centre crop), so it does not hold effective resolution constant — it holds *background* constant while removing *truncation*. Resolution is therefore still bundled with the `P − L` term, and §3 must say so rather than claim a clean two-way split.
