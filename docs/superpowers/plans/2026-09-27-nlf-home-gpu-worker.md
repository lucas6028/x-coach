# NLF Home-GPU Worker Implementation Plan

> **Status (2026-09-28): Phase 0 passed (GO); Phase 1 in progress** on branch
> `feat/nlf-3d-view`, cut from `origin/main` at `6b7468a7`. The user chose not to wait for the
> InnoServe deadline. Phase 1 step-level tasks are in "Phase 1 — step-level tasks" below.

**Goal:** Let an owner of an already-analysed video ask for a **3D view**. A worker on the
home PC (GTX 1660 Ti, 6 GB) picks the request up whenever the PC is on, runs NLF
(Neural Localizer Fields) on the clip, and uploads a 3D result. The results page shows it
as a rotatable 3D figure, so a squat filmed from the front can be turned to the side to see
depth. The MediaPipe analysis stays instant and stays the source of every verdict (Phase 1).
NLF-driven verdicts are Phase 2, behind a fidelity check.

**Why NLF, and why only on the home PC:**
- It is the only monocular model with measured evidence that it fixes the depth verdict.
  Fit3D: raw single-view 2D wrongly fails 82% of good squats on depth; NLF 7%
  (`notes/fit3d_decision_fidelity_summary.md`).
- MediaPipe's own model card lists "applications requiring metric accurate depth" as out of
  scope.
- NLF needs a GPU. The Azure deployment is CPU-only and scales to zero (about $7/month).

**License:** NLF weights ("noncommercial research use") and SMPL (non-commercial research)
are acceptable because the app is a non-commercial research prototype. Keep it free, no ads
or paid pilots. If that changes, the commercially usable swap is SAM 3D Body + MHR, which
needs its own Phase 0.

## Decisions baked into this plan

1. **The worker pulls; nothing connects in to the PC.** It polls Supabase over outbound
   HTTPS. No port forwarding, no tunnel. While the PC is off, requests simply wait.
2. **The worker polls Supabase, never the Azure backend.** The backend is scale-to-zero;
   polling it every minute would keep a replica warm and cost roughly the always-on rate.
3. **The PC never holds the service_role key.** The backend does hold one
   (`backend/app/settings.py:46`; used by the LINE bridge, the LINE bot and the daily jobs).
   On an unattended home machine it would be full database access.
   - The worker signs in as a dedicated Supabase user holding a `worker` DB role.
   - It touches jobs only through `SECURITY DEFINER` functions granted to `authenticated`,
     each with an internal role gate. This is the admin pattern: `admin_list_users()`
     checks `is_admin(auth.uid())` and raises 42501 otherwise
     (`db/migrations/20260713000200_admin_user_overview.sql`).
   - The repo's other machine-caller pattern, an `X-Job-Token` header plus service_role-only
     RPCs (`backend/app/routers/jobs.py`, `db/migrations/20260927000000_daily_jobs.sql`),
     is rejected here. It runs through the backend, so it collides with decision 2.
4. **3D is opt-in per video** ("Get 3D view" button), not automatic for every analysis. It
   bounds GPU load and makes the privacy consent explicit at the moment a clip is sent to
   the PC.
5. **The body mesh is the default view, delivered as pre-rendered turntables** (user
   decisions, 2026-09-27).
   - **How it works.** At each key frame the worker renders the mesh on the PC from a ring of
     angles (proposed 24). It uploads the frames as one image strip, and the page rotates the
     body as the user drags across it.
   - **Why images.** The SMPL license says the model "shall not be copied, shared,
     distributed … in whole or in part". Drawing a mesh surface needs the SMPL triangle list,
     which is part of the model file. With turntables it never leaves the PC; only
     renderings do, and the license does not restrict those. This holds for any audience,
     including a public launch.
   - **Size.** Proposed at about 0.3–0.6 MB per key frame, at most 12 key frames per video,
     loaded only when the user opens a key frame.
   - **Scrubbing the whole clip.** A light 3D skeleton handles that. It is drawn from NLF
     joints on a 2D canvas with a rotation matrix, so it needs no SMPL assets and no 3D
     library.
   - **Gated on Phase 0** confirming the model build returns the full SMPL vertex set. The
     copy tested on 2026-09-27 returned 1,024 points.
6. **NLF runs on the whole clip at a reduced rate (proposed 10 fps), not on a few frames.**
   Joints for every sampled frame are small (about 86 KB for 30 s). They allow a scrubbable
   3D animation, and Phase 2 needs full sequences anyway. Clips are capped at 90 s
   (`frontend/src/lib/poseExtract.ts:20`), so the worst case is 900 samples.
7. **Jobs are keyed per video (`user_id`, `video_id`), not per analysis.** Re-analysing a clip
   inserts a second `analyses` row with the same `video_id` (`backend/app/services/store.py:314-318`),
   and `analyses` joins `videos` by (`user_id`, `video_id`) with no foreign key. The 3D
   belongs to the footage.
8. **The worker picks the key frames**, because the turntables are rendered on the PC. It
   reads them from the video's latest analysis, which `claim_nlf_job()` returns, and records
   which analysis it used.
   - Every movement: each detection's `peak_frame`.
   - Squat only: the bottom of each rep, taken as the minimum 3D knee angle inside each
     `reps.segments` span. Knee angle does not depend on camera tilt, whereas pelvis height
     in NLF camera coordinates does.
   - Non-squat movements score the whole clip on the browser path, so their rep segments
     are not relied on.
   - Key frames closer than 0.3 s are merged, and the total is capped at 12. The selection
     logic is pure and unit-tested.
9. **No `videos.status` reuse.** The column allows `pending/processing/done/failed`, but only
   `done` is ever written (`store.py:217`). NLF state gets its own table so the MediaPipe
   lifecycle is untouched.
10. **Signed-in users only.** Anonymous uploads live under `uploads/anon/` and get no database
    row (`backend/app/routers/analyze.py:166-168`), so there is nothing to attach a job to.

## Phase 0 — Feasibility on this PC (go/no-go)

No app code. Output: a short entry in this plan's "Phase 0 results" section plus a pinned
environment recipe.

Known state (2026-09-27):
- Every NLF run so far was a Kaggle P100 notebook. This PC never ran NLF; the old
  blocker was driver 512.78. The driver is now 610.88 and `.venv-cuda` (torch 2.13+cu126)
  sees the GPU.
- The model loads (about 11 s) but its detector returns zero people on a clear Fitness-AQA
  frame, even at threshold 0.01. Its output has no `boxes` key and a 1,024-point vertex
  array, where the Kaggle runs had `boxes` and 6,890.
- **The file is not the cause.** A fresh download of `https://bit.ly/nlf_l_pt` (2026-09-27,
  now at `data/models/nlf/nlf_l_multi.torchscript`) is byte-identical to the
  `.kaggle_tmp/nlf_smoke_out/models/` copy: SHA256 `2090B041…82CDAD`, 495,696,900 bytes.
  - The cause turned out to be half precision on this GPU; see Phase 0 results.
- Seven real recorded WebM clips sit in `data/runtime/uploads/` for testing.

Tasks:
- [x] **0.1 Pin the Kaggle-proven environment.** `.venv-nlf` with
  `torch==2.5.1 torchvision==0.20.1` (cu121), plus numpy, opencv-python and av. Built by
  `data/models/nlf/setup_venv_nlf.cmd`. The model file hash matches the Kaggle copy.
- [x] **0.2 Isolate the zero-detection cause.** Done 2026-09-28; see Phase 0 results.
  Requires the all-float32 model `data/models/nlf/nlf_l_multi_fp32.torchscript`, loaded
  through `data/models/nlf/nlf_fp32_loader.py`.
- [ ] **0.3 Measure on real app clips.** Use all seven WebM clips plus at least one portrait
  phone MP4 with rotation metadata. There are no MP4s in `data/runtime/uploads/`, so the user
  supplies one. At 540p and 720p, with batch sizes 8 and 16, record: ms per frame, peak VRAM,
  and per-frame detection rate. Sanity-check scale: pelvis-to-neck should be about 500–540 mm.
- [ ] **0.4 Prove the decoding path.**
  - Decode sequentially and take timestamps from the stream (PyAV `frame.time`), never from
    frame counts or seeking. Recorded WebM has no Duration element, and OpenCV reports a
    garbage frame count (INT64_MIN) for it.
  - Apply the display rotation explicitly. The browser honours an MP4's rotation metadata;
    PyAV does not, and OpenCV's behaviour depends on version and backend. A sideways frame
    means no detection, or a 3D view with the wrong "up". Verify on the portrait MP4 rather
    than assume.
- [ ] **0.5 Prove the frame-to-seconds mapping.** Decision 8 and the scrub sync need the
  worker's stream timestamps to equal the browser's `video.currentTime`, and phone MP4s
  often have a non-zero stream start or edit lists. At several times `t` on each clip,
  project NLF `joints2d` onto the frame and compare with the MediaPipe landmarks at frame
  `round(t * 30)` in that clip's pose JSON (`data/runtime/pose_json/<video_id>.json`). If they
  are offset in time, subtract the stream start time and re-check.
- [ ] **0.6 Prove the turntable render on this PC.**
  - Pick a renderer that works on Windows with this GPU: pyrender offscreen first, a plain
    numpy rasteriser as the fallback. Render 24 angles at one squat bottom and one fault
    peak.
  - Define "up" so the side view is truly sideways. NLF output is in camera coordinates, and
    a tilted phone would tilt the body and distort the depth reading. Candidates: the
    ankle-to-shoulder direction over standing frames, or NLF's `world_up_vector` argument.
  - Record the time per key frame and the strip's file size. Check visually on two clips that
    the 90° view is upright and shows hip versus knee height clearly.

**Gate (proposed thresholds; adjust if Phase 0 shows why):**
- Person detected on ≥95% of sampled frames across the seven clips.
- A 30 s clip at 10 fps finishes in ≤2 minutes end to end.
- Peak VRAM stays under 5.5 GB.
- The portrait MP4 decodes upright, and the 0.5 mapping check agrees to within one sampled
  frame.
- A 24-angle turntable renders in ≤10 s per key frame, at ≤0.6 MB per strip, with an
  upright side view.

If the gate fails, stop and report the numbers. Fallbacks, in order: lower the sample rate,
lower the resolution, then NLF-S if a compatible build exists.

### Phase 0 results

**0.1–0.2 (2026-09-28): NLF runs on the GTX 1660 Ti, but only fully in float32.**

- **Two causes, both confirmed.**
  - Under torch 2.13 (`.venv-cuda`), the fp16 person detector finds nobody, even at
    threshold 0.01.
  - Under torch 2.5.1, the stock model's hard-coded fp16 casts break on this card (TU116, a
    known GTX 16-series fp16 problem). The pose stage returns NaN boxes and uncertainties,
    so every pose fails the model's own plausibility filter. The detector returns NaN or
    about 10 garbage boxes per frame at batch size 8.
  - The missing `boxes` key and the 1,024-point vertex array are the model's
    `_predict_empty_smpl` path, i.e. zero surviving detections.
- **Fix.** `data/models/nlf/make_fp32_model.py` patches both casts in the TorchScript archive
  to float32: `torch.to(crops_flat, 5)` and `torch.to(images, 5)` become `6`.
  `nlf_fp32_loader.py` then converts `crop_model`, `detector` and `cano_all` to float32 and
  precomputes the `weights['smpl']` canonical tensors in float32.
- **Graph optimisation setting.** Every Phase 0 run used `torch.jit.optimized_execution(False)`.
  Re-checked with it on, on the same 8 frames × 4 repeats:
  - All criteria except bit-identity still pass.
  - Repeats differ by at most 0.2 mm on joints and 0.5 mm on vertices, with boxes identical.
  - About 2× faster: 0.12 vs 0.23 s per frame at 960×540.
  - Either setting works. **Default off** for reproducible output; speed is already inside
    the gate.
- **Check result: PASS** (`data/models/nlf/phase0_2_check.py`; 8 frames of Fitness-AQA
  `32903_8.mp4`, 3 repeats at batch sizes 4 and 8):
  - Exactly 1 detection per frame and no NaN.
  - Identical across repeats.
  - All uncertainties under the filter's 0.25 m.
  - Full 6,890-vertex mesh.
  - Pelvis-to-neck 515–520 mm.
- **Speed.** 0.23 s per frame steady state at 960×540, batch 8 (about 3× the P100's 79 ms
  in fp16). A 30 s clip at 10 fps is about 70 s; the 90 s cap about 3.5 min.
- **Measurement trap.** The model's output `vertex_uncertainties` and `joint_uncertainties`
  are multiplied by 1000 (mm). The plausibility filter compares the raw values (m) with
  0.25. The first version of the check forgot this and reported a false FAIL.
**0.3–0.4 on a real phone clip (2026-09-28): PASS for in-app recordings.**
- **Clip.** The user supplied a phone recording made in the app: portrait VP9 WebM,
  720×1280, 26.6 s, about 30 fps, timestamps starting at 0, no rotation metadata.
- **Method.** `data/models/nlf/phase0_3_clip.py`: PyAV sequential decode, sampled at 10 fps
  by stream time, all-float32 NLF with the model's plausibility filter on.
- **Results** (266 sampled frames per configuration):

  | Short side, batch | Detected | Speed | Peak VRAM |
  |---|---|---|---|
  | 540, 8 | 266/266 | 121 ms/frame | 1.65 GB |
  | 720, 8 | 266/266 | 123 ms/frame | 1.66 GB |
  | 540, 16 | 266/266 | 102 ms/frame | 2.25 GB |

  - No multi-person frames. Pelvis-to-neck median 494 mm (IQR 465–508).
  - About 27–33 s of inference for 26.5 s of video, well inside the gate.
- **Orientation.** The overlay frame is upright and the 2D joints land on the body. The
  in-app recorder writes upright pixels, so there is no rotation to apply.
- **Gallery video with rotation metadata: PASS.**
  - Clip: iPhone `IMG_9233.mov`, HEVC stored landscape at 3840×2160 with a −90° display
    matrix, 24 fps, 15.4 s.
  - PyAV exposes the matrix as `frame.rotation`. Applying `np.rot90(img, k=round(rotation/90) % 4)`
    gives an upright 2160×3840 frame; the overlay checked at 7.7 s is upright with the joints
    on the body.
  - Detected 154/154, 134 ms/frame, pelvis-to-neck median 504 mm.
  - Without this step NLF would receive a sideways person, so rotation is a required
    decode step in the worker.
- **Decode cost is small.** With PyAV `thread_type = "AUTO"`, 4.9 s for the 15 s 4K HEVC clip
  and 3.3 s for the 26.6 s WebM. End to end, including inference, that is about 26 s and
  about 35 s.
- **Only one clip so far.** The seven local recordings in `data/runtime/uploads/` are
  users' uploads. Reading them was blocked as production user data, and they stay untested
  unless the user allows it.

**0.5 time mapping (2026-09-28): PASS.**
- Method: `data/models/nlf/phase0_5_mapping.py` on the WebM and its app pose JSON (supplied by
  the user). For each shift δ, it compares NLF 2D joints at stream time `t` with the browser's
  MediaPipe landmarks at frame `round((t + δ) * 30)`, using 12 matched joints.
- Result: the error curve is V-shaped over ±0.3 s with its minimum at δ = 0 (20.8 px median
  at 720×1280; 22.7 px at −1 frame, 21.5 px at +1 frame). So stream time equals the app's
  `frame_index / fps`, with no offset for in-app recordings.
- Untested: the gallery case, since the MOV never went through the app and has no pose JSON.
  Its stream does start at 0.

**0.6 turntable render (2026-09-28): PASS.**
- **Tool.** `data/models/nlf/phase0_6_turntable.py`. pyrender 0.1.45 with pyglet 1.5.31
  (pyglet must be below 2) and trimesh, drawing offscreen through a hidden window, so the
  worker must run in the logged-in desktop session.
- **SMPL triangles.** Read from the local SMPLify model
  (`data/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) through an
  unpickler that stubs out `chumpy`. They are used only on the PC.
- **"Up" axis.** The shoulder-over-ankle direction on standing frames measured a 5.1° camera
  tilt. The 90° view of the standing key frame is upright and truly side-on.
- **Output.** 24 orthographic views per key frame, one WebP strip each, in 0.24–0.47 s per key
  frame at 66–114 KB per strip. Both are far inside the ≤10 s and ≤0.6 MB gate.
- **Depth reading.** The squat bottom (t = 11.12 s, 3D knee angle 51°) seen side-on shows the
  hip and knee joint centres level (hip 1.0 cm lower), i.e. the thigh roughly horizontal,
  read from a front-facing phone clip.
  - This is **not** a "parallel" verdict. Coaches judge parallel by the hip crease against the
    top of the knee, and joint centres at 0 are not that cutoff (the Fit3D depth-ratio note).
- **The MOV renders cleanly too.**
  - Rep window 2.5–13 s, taken from its knee-angle trace.
  - The "up" axis measured a 1.7° tilt after rotation. The 90° standing view is upright and
    side-on.
  - Bottom at 7.33 s, knee 49°, joint centres level (hip 1.2 cm lower). 0.28 s and 66–90 KB
    per strip.
- **Key-frame lesson.** Picking "minimum knee angle over the whole clip" chose a set-up crouch
  at 0.12 s (knee 17°) before the first rep. The worker must take bottoms only inside the
  analysis's `reps.segments` (decision 8). The check used the span where the app found a pose
  (3.6–25.2 s) as a stand-in.

**Phase 0 verdict: GO**, on the evidence of two real phone clips (an in-app WebM and an iPhone
HEVC MOV). Not yet covered:
- the seven local user recordings (blocked; need the user's permission);
- a fault-peak key frame (no analysis result available locally);
- time mapping for gallery uploads.

## Phase 1 — Worker plus 3D view (verdicts unchanged)

Detailed step-level tasks get written after Phase 0 passes. The task list and acceptance
criteria are fixed now.

- [ ] **1.1 Migration: jobs table, worker role, functions.** New file in `db/migrations/`.
  - Table `nlf_jobs`: one row per video (unique), plus `owner_id`, `status` (`queued`,
    `claimed`, `done`, `failed`), `claimed_at`, `attempts`, `error`, `result_key`,
    timestamps.
  - A composite foreign key to `videos (user_id, video_id)` with `on delete cascade`, so
    deleting a video removes its job.
  - RLS: the owner may insert a request for their own video and read its status; nothing
    else.
  - A `worker` role (an `is_worker(uid)` helper modelled on `is_admin`), a one-row heartbeat
    table (none exists today), and these `SECURITY DEFINER` functions, each gated on
    `is_worker(auth.uid())`:
    - `claim_nlf_job()`: uses `FOR UPDATE SKIP LOCKED`. Returns:
      - the job id;
      - the video's `videos.storage_key` (the prefix `uploads/{owner}/{video_id}`). The
        worker then reads `{prefix}/source`, which has no file extension; the content type
        was set at upload;
      - from the video's latest `analyses` row: its id, `movement`, `pose.fps`,
        `reps.segments`, and `detections[].{fault_id, peak_frame}`. That is enough for key
        frames (decision 8) and nothing more.
    - `complete_nlf_job(job_id, result_key, meta)`: returns false if the job or its video no
      longer exists. The worker then deletes the result it just uploaded. Otherwise a video
      deleted mid-job would leave a patient's 3D data orphaned in R2.
    - `fail_nlf_job(job_id, error)`.
    - `worker_heartbeat()`.
  - A claimed job whose `claimed_at` is older than 30 minutes goes back to `queued`, so a PC
    shut down mid-job does not strand it. After 3 attempts it becomes `failed`.
  - Acceptance: `node db/checks/run_pglite.mjs` passes, with new RLS checks showing that (a)
    a normal user cannot read another user's job, (b) a normal user cannot call the worker
    functions, and (c) the worker cannot select from `nlf_jobs` or the video tables directly.
- [ ] **1.2 Backend endpoints.**
  - `POST /api/analyses/{id}/nlf` queues a request (owner only, idempotent).
  - `GET /api/analyses/{id}/nlf` returns the status plus a presigned URL for the result
    when done.
  - The worker is offline if the last heartbeat is older than 10 minutes. Expose that as a
    `worker_online` flag in the status response and on the admin panel. Template:
    `GET /api/admin/line/status` (`backend/app/routers/admin.py:292-328`), which pairs each
    nullable value with a `*_error` field. Do not add it to the unauthenticated `/api/health`.
  - Acceptance: tests under `tests/`, new modules added to the explicit list in
    `scripts/run_backend_coverage.py`, and
    `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` passes.
- [ ] **1.3 Object access for the worker.** Chosen option _(open decision, see below)_.
  - Results go under the same prefix as the video: `{storage_key}/pose3d.v1.json.gz` plus
    one strip per key frame at `{storage_key}/turntable.v1/k{NN}.webp`. `delete_prefix`
    (`store.py:269-287`) already removes everything under the prefix, so the existing delete
    paths clean them up with no extra code.
  - Keys are write-once with `Cache-Control: immutable` (`backend/app/services/storage.py:40-48`),
    so a re-run or schema change writes a new versioned name and never overwrites.
  - Acceptance: tests prove that deleting the video deletes its NLF result, including the
    in-flight case (delete after claim, before complete, leaves no object behind).
- [ ] **1.4 The worker.**
  - Pure logic in `src/nlf_worker/`: frame sampling by stream timestamp, choice of the
    largest person, result serialisation. Thin CLI at `scripts/nlf_worker/run_worker.py`.
  - Loop: poll every 60 s → claim → fetch the video → decode sequentially → infer → pick
    key frames → render turntables → upload → complete. Heartbeat on each poll. Local
    copies of the video are deleted after each job, success or failure.
  - The renderer from 0.6 runs on the PC. The SMPL triangle list stays there.
  - Runs under `.venv-nlf`. Started by Windows Task Scheduler at logon.
  - Supabase access tokens expire after about an hour. The worker refreshes its session, or
    signs in again, before each poll that would otherwise fail.
  - The NLF call sits behind a small interface, so tests use a fake and never import torch
    (CI has no torch).
  - The Phase 0 recipe moves into the repo here. It currently lives only in gitignored
    `data/models/nlf/`: the archive patch (`make_fp32_model.py`), the float32 loader, the
    rotation rule, and the SMPL-faces unpickler. It becomes code in `src/nlf_worker/`.
  - Acceptance: unit tests for the pure logic pass under `.venv`, and a manual end-to-end
    run on one real clip is recorded in the plan.
- [ ] **1.5 Result format** (versioned JSON, gzip):
  - `schema_version`, model file hash, torch version.
  - `sample_times_s`.
  - Per sampled frame: `joints3d` (SMPL-24, mm, camera coordinates),
    `joint_uncertainties`, and a `detected` flag.
  - `analysis_id` of the analysis the key frames came from.
  - `key_frames`: each has `t_s`, `kind` (`rep_bottom` or `fault_peak`), `fault_ids`,
    `rep_index`, `sprite_key` and `angles`. Angle 0 is the camera's own view and angle 90 is
    the side view.
  - Everything is keyed by seconds. The worker converts analysis frame numbers
    (`reps.segments[].start_frame/end_frame`, `detections[].peak_frame`) to seconds with
    `analysis.pose.fps`, the same way `SkeletonOverlay` does (`round(currentTime * fps)`).
    - On the live browser path that fps is a fixed 30 Hz virtual timeline, not the video's
      own rate (`CANONICAL_FPS = 30`, `frontend/src/lib/repSpans.ts:21`).
    - The legacy server path stored the native fps instead
      (`src/pose/process_videos.py`). So always read the fps from the analysis; never
      assume 30.
- [ ] **1.6 Results page.**
  - "Get 3D view" button, then status (queued / processing / ready / failed / worker
    offline).
  - **Turntable viewer (the main view).** A key-frame list ("Rep 2 bottom", "Knees caving:
    peak"). Picking one loads its strip, and dragging rotates the body. A "Side view"
    button snaps to 90°. Picking a key frame also seeks the video to `t_s`.
  - **Skeleton scrub (secondary).** The NLF skeleton for the whole clip, synced to the video,
    drawn on a 2D canvas with a user-set rotation. No three.js.
  - Two mount points, because mobile is a separate tree.
    - Desktop: the card row under the video (`frontend/src/App.tsx:521-525`).
    - Mobile: `frontend/src/components/mobile/StudioMobile.tsx`.
  - Strings added to `frontend/src/lib/i18n.tsx`.
  - Pure, tested logic: drag distance to angle index, the 3D-to-2D projection, and key-frame
    labels. Only the thin canvas/pointer glue is coverage-excluded, like the existing
    camera/WASM glue.
  - Acceptance: `yarn test:coverage` passes from `frontend/`, and a manual browser check
    with a real result is recorded.
- [ ] **1.7 Turntable rendering.** Requires Phase 0 to show the build returns the full SMPL
  vertex set, and 0.6 to pass.
  - Fixed camera distance, a floor line, and the "up" axis chosen in 0.6.
  - Faint horizontal lines at hip and knee **joint-centre** height, so depth reads at a
    glance in the side view.
    - The page must not label their crossing "parallel". Parallel is judged by the hip crease
      against the top of the knee, and joint centres at 0 are not that cutoff.
    - A label like that would invite a depth verdict the rules don't make, the same
      disagreement risk as MediaPipe's shallow-depth fault.
  - Optional: tint the body parts involved in the key frame's faults, using the SMPL
    part segmentation on the PC only.
  - Nothing SMPL-derived leaves the PC except rendered images.
  - Acceptance: at a squat's lowest point, the 90° image agrees with the 3D knee angle at
    that time, checked on at least three reps.
- [ ] **1.8 Consent and runbook.**
  - One line in the consent text: requested clips are processed on a research PC and
    deleted after processing.
  - `docs/nlf-worker.md`: start/stop, credentials, rotating the worker login, what "worker
    offline" means.

## Phase 1 — step-level tasks (2026-09-28)

These refine 1.1–1.8 using the two codebase surveys of 2026-09-28. Where this section and the
task-level list above disagree, this section wins.

### Corrections from the surveys

- **"Worker role" and "selected users" are rows in `public.user_roles`, not Postgres roles.**
  The table is keyed `(user_id, role)` with `user_roles_role_check` allowing
  `('admin', 'clinician')` (`db/migrations/20260916000000_clinic.sql:34-39`). Widen the check to
  add `'nlf_user'` (may request 3D) and `'nlf_worker'` (the worker's login).
  - Model the gate functions on `is_admin()` (`20260713000000_admin_roles.sql:30-46`).
- **Admin role writes are direct table writes** gated by `is_admin` RLS
  (`backend/app/services/store.py:142-179`). Cross-user reads use the SECURITY DEFINER
  `admin_list_users()`, redefined in `clinic.sql:476-526`. Changing its columns means
  `drop function`, then `create`, then re-grant.
- **The newest migration is applied twice by the checks** (`db/checks/run_pglite.mjs:94-96`), so
  the migration must be idempotent.
- **No user-scoped client dependency exists.** Routers take `get_current_user`
  (`backend/app/auth.py:109`) and pass `user.token` to `store.*`, which calls
  `store._user_client(token)` (`store.py:24-33`).
- **Account deletion reaps no R2 objects** (the route does not exist;
  `AccountPane.tsx:90-92` says "not wired"). This is a pre-existing gap, not introduced here:
  NLF results follow the video's lifetime like `source` and `pose.json` do.
- **No consent UI exists.** The privacy line goes next to the "Get 3D view" button as a new
  i18n string. The nearest precedent is `settings.therapist.privacy`.
- **Worker credentials** live in `data/models/nlf/worker.env`, which is gitignored under
  `data/*`; `.env.nlf-worker` would *not* be ignored:
  - `SUPABASE_URL` and `SUPABASE_ANON_KEY`;
  - `NLF_WORKER_EMAIL` and `NLF_WORKER_PASSWORD`;
  - `NLF_R2_ACCOUNT_ID`, `NLF_R2_ACCESS_KEY_ID`, `NLF_R2_SECRET_ACCESS_KEY` and
    `NLF_R2_BUCKET`, for the dedicated token.

### Contract, fixed before any task starts

**Database** (migration `db/migrations/20260928000000_nlf_3d_view.sql`)

- **`user_roles.role`** gains `'nlf_user'` and `'nlf_worker'`.
- **Gate functions.** `has_nlf_access(uid uuid) → boolean` and
  `is_nlf_worker(uid uuid) → boolean`, both `security definer`, granted to `authenticated`.
- **Table `nlf_jobs`.** Columns:
  - `id uuid pk default gen_random_uuid()`;
  - `user_id uuid not null`;
  - `video_id text not null`;
  - `status text not null default 'queued' check (status in ('queued','claimed','done','failed'))`;
  - `attempts int not null default 0`;
  - `error text`;
  - `result_key text`;
  - `meta jsonb`;
  - `analysis_id uuid`;
  - `claimed_at timestamptz`;
  - `created_at` and `updated_at`, both `timestamptz not null default now()`.

  Constraints: `unique (user_id, video_id)`, and
  `foreign key (user_id, video_id) references public.videos (user_id, video_id) on delete cascade`.
  RLS: owner `select` only, with no direct insert, update or delete for anyone.
- **Table `nlf_worker_heartbeat`.** `(id int primary key check (id = 1), last_seen timestamptz)`
  with RLS on and no policies.
- **`request_nlf_job(p_video_id text) → public.nlf_jobs`** (security definer)
  - Requires `has_nlf_access(auth.uid())` and a `videos` row owned by the caller; otherwise
    it raises 42501 or P0002.
  - Inserts a queued job, or re-queues a `failed` one (`attempts = 0`, `error = null`).
    Otherwise it returns the existing row unchanged.
- **`nlf_worker_last_seen() → timestamptz`** (security definer, `authenticated`).
- **Worker functions** (security definer, each raising 42501 unless
  `is_nlf_worker(auth.uid())`):
  - **`nlf_claim_job()`** returns 0 or 1 rows of `(job_id uuid, owner_id uuid, video_id text,
    attempts int, storage_key text, analysis_id uuid, movement text, pose_fps double precision,
    rep_segments jsonb, detections jsonb)`.
    1. Stale claims (`claimed_at < now() - interval '30 minutes'`) go back to `queued`, or to
       `failed` once `attempts >= 3`.
    2. It picks the oldest `queued` job with `for update skip locked` and sets `claimed`,
       `attempts + 1`, `claimed_at = now()`. The returned `attempts` is the new value.
    3. It reads the video's latest `analyses` row (by `created_at`) for `id`, `movement`,
       `result->'pose'->'fps'`, `result->'reps'->'segments'`, and `detections` reduced to
       `[{fault_id, peak_frame}]`.
  - **`nlf_complete_job(p_job_id uuid, p_result_key text, p_meta jsonb) → boolean`** sets
    `done`. It returns `false` if the job is missing or not `claimed`, and the worker then
    deletes what it uploaded.
  - **`nlf_fail_job(p_job_id uuid, p_error text)`** sets `failed` if `attempts >= 3`, else
    `queued`, and stores `error`.
  - **`nlf_worker_heartbeat()`** upserts row 1 with `now()`.
- **`admin_list_users()`** gains a `has_nlf boolean` column (drop, create, re-grant, keeping
  every existing column).

**Result objects** (written by the worker under `{storage_key}/nlf.v1.a{attempts}/`, so a re-claim never
overwrites a write-once, immutable-cached key, and "delete what I uploaded" is one exact prefix)

- **`pose3d.json.gz`** holds:
  - `schema_version: 1`, `model_sha256`, `torch`, `sample_fps`;
  - `sample_times_s: [..]`;
  - `joints3d`: per sample, 24×3 mm in camera coordinates, `null` if undetected;
  - `joint_uncertainties`: per sample, 24, mm;
  - `up_axis: [x, y, z]`.
- **`turntable/k{NN}.webp`**: one strip per key frame, 24 tiles of 384×384, angle 0 = the
  camera's view, clockwise seen from above.
- **`nlf_jobs.meta`** holds:
  - `schema_version: 1`, `analysis_id`, `angles: 24`, `tile_px: 384`, `prefix` (the
    `nlf.v1.a{n}` folder);
  - `key_frames: [{t_s, kind: 'rep_bottom' | 'fault_peak', fault_ids: [..],
    rep_index: int | null, strip_key}]`.

**API**

- **`GET /api/nlf/status`** returns `{"enabled": bool}`, i.e. whether the caller has
  `nlf_user`.
- **`POST /api/analyses/{analysis_id}/nlf`** returns 202 plus an `NlfJob`.
  - 403 if the caller is not enabled; 404 if the analysis is not the caller's.
  - The UUID is normalised as in DELETE (`routers/analyses.py:66-69`).
- **`GET /api/analyses/{analysis_id}/nlf`** returns 200 plus an `NlfJob`, or 404 if no job
  exists.
- **`NlfJob`** has:
  - `status`, `worker_online` (last heartbeat within 10 min), `error`;
  - `created_at`, `updated_at`;
  - `key_frames: [{t_s, kind, fault_ids, rep_index, strip_url}]`, with presigned URLs and
    only when `done`;
  - `angles`, `tile_px`, `expires_in`.
  - No `pose3d_url` in Phase 1: nothing reads it yet. A future browser fetch of the `.json.gz`
    needs R2 CORS and `Content-Encoding: gzip` on the object.
- **`PUT /api/admin/users/{user_id}/role`** accepts an optional `enable_nlf: bool` alongside
  `make_admin` and `make_clinician`. `AdminUserRow` gains `has_nlf`.

### Tasks

**Delete path (verified).** `store.delete_analysis` deletes the `videos` row when the last
analysis of a video goes (`store.py:356-373`). That cascades the job, so an in-flight
`nlf_complete_job` returns false and the worker deletes its own prefix.
The coverage gate measures only `backend.app` (`scripts/run_backend_coverage.py:79`), so
`src/nlf_worker/` is not in the 95% gate; its tests still run under the CI `pytest tests/`.

Each task runs in its own implementation worker, followed by an independent verifier.

- [x] **T1: Migration and checks.**
  - Write `db/migrations/20260928000000_nlf_3d_view.sql` (idempotent) exactly to the contract.
  - Write `db/checks/nlf_3d_view_check.sql` with fresh UUIDs (`…2a1`, `…2b1`, `…2c1`,
    `…2d1`; not colliding with `clinic_rls_check.sql:21-26` or `daily_jobs_check.sql:16-21`).
  - The checks must prove:
    - (a) a user without `nlf_user` gets 42501 from `request_nlf_job`;
    - (b) an enabled user can request only their own video;
    - (c) a user cannot read another user's job;
    - (d) a normal user gets 42501 from every worker function;
    - (e) the worker cannot select `nlf_jobs`, `videos` or `analyses` directly;
    - (f) claim returns the storage key and the latest analysis fields;
    - (g) deleting the video cascades the job, and `nlf_complete_job` then returns false;
    - (h) a stale claim is re-queued, and fails after 3 attempts;
    - (i) re-requesting a failed job re-queues it;
    - (j) `admin_list_users()` returns `has_nlf`.
  - Acceptance: `node db/checks/run_pglite.mjs` exits 0.
- [x] **T2: Backend.**
  - `backend/app/services/store.py`: `get_nlf_enabled`, `request_nlf_job`, `get_nlf_job`,
    `nlf_worker_last_seen`, and `set_nlf_access` (direct upsert or delete of the
    `nlf_user` row, like `set_clinician_role`).
  - A new `backend/app/routers/nlf.py` with the three endpoints, registered in `main.py`.
  - `admin.py`: `RoleUpdate.enable_nlf`, and `has_nlf` passed through.
  - Presign `key_frames[].strip_key` with `storage.get_object_store()`. A
    `StorageError` becomes 503, as in `videos.py:23-34`.
  - Tests in a new `tests/test_nlf_api.py`, with the fakes extended as needed, added to
    `_DEFAULT_TESTS` in `scripts/run_backend_coverage.py`.
  - Acceptance:
    - `.venv\Scripts\python.exe -m pytest tests/` passes;
    - `.venv\Scripts\python.exe scripts/run_backend_coverage.py --fail-under 95` passes.
- [x] **T3: Worker, pure logic (CI-tested, no torch import at module load).** New package
  `src/nlf_worker/`:
  - `keyframes.py`: squat rep bottoms from the minimum 3D knee angle inside each rep segment;
    every movement's detection `peak_frame`; merge within 0.3 s; cap 12.
  - `timeline.py`: sampling by stream time; the rotation rule `k = round(rotation/90) % 4`;
    frame index to seconds via `pose_fps`.
  - `geometry.py`: the up axis from standing frames, the body frame, orthographic pixel
    mapping, knee angle.
  - `result.py`: builds `pose3d.v1` and `meta`.
  - `smpl_faces.py`: the chumpy-stub unpickler.
  - `loop.py`: one job end to end, over injected `Db`, `Store`, `Decoder`, `Model` and
    `Renderer` interfaces. The order is claim → fetch → decode → infer → key frames →
    render → upload → complete. A false `complete` deletes the uploads. Heartbeat on every
    poll; local files are removed in `finally`.
  - `config.py`: env loading from `data/models/nlf/worker.env`.
  - Tests: `tests/test_nlf_worker.py`, `unittest`, with fakes for every interface, added to
    the pytest run.
  - Acceptance: `.venv\Scripts\python.exe -m pytest tests/test_nlf_worker.py` passes, and
    the modules import without torch, pyrender or av installed.
- [x] **T4: Worker, runtime adapters (excluded from CI).**
  - `src/nlf_worker/runtime/`:
    - `model.py`: the archive patch plus the float32 loader, i.e. `make_fp32_model.py` and
      `nlf_fp32_loader.py` promoted, run under `optimized_execution(False)`;
    - `decoder.py`: PyAV, `thread_type = "AUTO"`, rotation applied;
    - `renderer.py`: pyrender turntable, from `phase0_6_turntable.py`;
    - `supabase_db.py`: password sign-in, refresh before expiry, RPCs only;
    - `r2_store.py`: boto3 against the dedicated token.
  - `scripts/nlf_worker/run_worker.py`: a thin CLI with `--once` and a loop every 60 s.
  - Acceptance: `.venv-nlf\Scripts\python.exe scripts/nlf_worker/run_worker.py --once
    --dry-run CLIP` renders the phone clip's turntables locally with no network calls, and
    the output matches Phase 0.
- [x] **T5: Frontend data and admin.**
  - `api.ts`: `nlfStatus`, `requestNlf`, `getNlf` and the `NlfJob` type; `AdminUserRow.has_nlf`;
    `setUserNlf`.
  - `lib/auth.tsx`: a `canUse3d` probe mirroring `isClinician` (added to the value and to
    `deps`).
  - `AdminUsers.tsx`: a 3D toggle in the Roles cell, the `rowError.kind` union and the
    `admin.users.desc` copy.
  - i18n keys in both `en` and `zhHant`.
  - Tests: `api.test.ts`, `lib.auth.nlf.test.tsx`, `pages.admin.test.tsx`.
  - Acceptance: `yarn test:coverage` passes from `frontend/` (thresholds in
    `vite.config.ts:65-70`).
- [x] **T6: Frontend panel.**
  - `lib/turntable.ts` (pure): drag distance to angle index, the side-view index, key-frame
    labels.
  - `components/nlf/Nlf3dPanel.tsx`:
    - hidden unless `canUse3d` and `analysis.analysis_id`;
    - the button, the privacy line and the status;
    - polls `getNlf` every 15 s only while `claimed`, or while `queued` and
      `worker_online`, and only while the page is visible; otherwise it shows a manual
      refresh. Polling is cancelled on unmount (the `cancelled`-flag idiom from
      `useVideoSrc.ts:96-117`).
    - Reason: a tab left open while the PC is off must not keep the scale-to-zero backend
      awake (decision 2).
  - `components/nlf/TurntableViewer.tsx`: shows the strip as a CSS background offset, with
    pointer drag, a "Side view" button and the key-frame list. An optional `onSeek(t_s)`
    moves the video.
  - Mounts:
    - desktop: a full-width block in the `App.tsx:511` column after the card grid at
      `:521-525`;
    - mobile: a card in `StudioMobile.tsx` placed *outside* the collapsible `Row`, which
      renders children only while open.
  - Tests: `components.nlf.Nlf3dPanel.test.tsx` and `lib.turntable.test.ts`.
  - Acceptance: `yarn test:coverage` passes.
- [x] **T7: Runbook and setup (user steps flagged).** `docs/nlf-worker.md`:
  1. Apply the migration in the Supabase SQL editor.
  2. Check that the email/password auth provider is enabled (the app itself signs in with
     Google). Create the worker's Supabase user and insert `(uid, 'nlf_worker')` into
     `user_roles`.
  3. Create the dedicated R2 token.
  4. Fill `data/models/nlf/worker.env`.
  5. Build `.venv-nlf` and the float32 model.
  6. Register the Task Scheduler entry at logon (it needs the desktop session for pyrender).
  7. Troubleshooting.
- [ ] **T8: End-to-end on the real stack (user-assisted).**
  - Enable one user.
  - Request 3D on the phone-clip analysis.
  - Run the worker `--once`.
  - Check the page shows the turntable, and that deleting the **last analysis** of that
    video, which is the app's only delete action, removes the `nlf.v1.a*` objects.

**Order:** T1, then T2 and T3 in parallel, then T4, T5 and T6 (T6 after T5), then T7, then T8.

**Status 2026-09-28: T1–T7 done and independently verified; T8 waits on the user's setup steps.**

- **T1.** `run_pglite` OK. Checks (a)–(k), plus: fail-job acts only on claimed jobs; a
  non-numeric fps can't jam the queue.
- **T2.** 52 API tests; backend coverage 98.1%.
- **T3–T4.** 105 worker tests.
  - Inference streams (peak about 1.8 GB), with one shared turntable framing per job.
  - Dry run on the phone clip: `done` in about 46 s.
- **T5–T6.**
  - Admin toggle, the `canUse3d` probe, and the panel on desktop and mobile, with `onSeek`
    wired.
  - Polling is scheduled from fetch completion, with a 60 s back-off after an error. There is
    no polling while offline or hidden; a Refresh button covers those cases.
  - Works under StrictMode. The tile is a slider with ARIA values.
  - 1518 frontend tests pass, and the build succeeds.
- **T7.** `docs/nlf-worker.md`, checked against the code.
- **Final combined run:** db checks OK; coverage 98.1%; `pytest tests/` 3493 passed. The only 5
  failures are the known local-data segmentation-corpus tests (memory
  `local-test-live-supabase-dependency`).
- **Follow-ups, not blocking:**
  - Feet heights differ slightly between key frames, because the shared view is centred on the
    pelvis.
  - A 90 s clip hasn't been timed.
  - Account deletion reaping objects is a pre-existing gap.

## Phase 2 — NLF verdicts (outline only)

Not started until Phase 1 is live and a fidelity check passes.
- Rule inputs: either map SMPL-24 to MediaPipe-33 (no face points, no heels, hip is a joint
  centre, so depth thresholds shift) or compute the rule features directly from SMPL-24.
- Fidelity check first: the app's actual thresholds, NLF vs MediaPipe, on Fit3D (mocap
  truth) and Fitness-AQA labels. The Lite→Heavy switch alone changed 50% of squat verdicts,
  so no new source drives verdicts without this.
- UI rule for when the later NLF verdict disagrees with the instant MediaPipe one.

## Decisions taken (user, 2026-09-28)

- **R2 access:** a dedicated R2 API token for the bucket, stored on the PC; option (a) below.
  It is a new token, not the one in `.env`, so it can be revoked on its own.
- **Who may request 3D:** selected users only. An admin enables it per user; everyone else
  never sees the button.
- **Clinicians:** owner only in Phase 1. Clinician access is a later, separate step with its
  own RLS checks.

## Open decisions (history)

1. **How the worker reads videos and writes results.**
   - (a) An R2 API token scoped to the one bucket, stored on the PC. Least code; the token
     is revocable on its own. R2 tokens scope to a bucket, not a prefix, so the worker can
     technically read every user's video whatever the database allows. (Full R2 credentials
     already sit in this PC's `.env`.)
   - (b) The worker calls a backend endpoint once per job for presigned GET/PUT URLs. No R2
     secret on the PC, but it wakes the backend once per job, adds an endpoint, and needs a
     presigned PUT that `storage.py` does not have yet (it signs GET only, 3600 s TTL).
   - Recommendation: (a) for the prototype, with a dedicated token (not the one in `.env`).
2. **Sample rate.** 10 fps is proposed; Phase 0 timing may change it.
3. ~~How the mesh reaches the browser.~~ **Decided 2026-09-27: pre-rendered turntables**
   (decision 5). Live 3D in the browser was rejected because it would ship the SMPL
   triangle list.
4. **Do linked clinicians see the 3D view?** They are the likeliest audience for a depth
   view. They can already read a patient's analyses (`analyses_clinician_select`,
   `db/migrations/20260916000000_clinic.sql:375-379`), but `nlf_jobs` as planned is owner-only,
   and clinicians have no policy on `videos`.

## Out of scope

LINE bot and LIFF paths, the library/demo clips, live recording, and any change to existing
verdicts in Phase 1.
