# NLF 3D-view worker: setup and runbook

The 3D view shows the user's body as a rotatable mesh at key frames: each squat's lowest point
and each fault's worst moment. Turned to the side view, a squat filmed from the front shows its
depth.

The web app only queues requests. A worker on a home PC with a GPU does the work while that PC is
on:

1. It polls Supabase for a queued job.
2. It downloads the video from R2.
3. It runs the NLF body model and renders 24-angle turntable images.
4. It uploads them next to the video and marks the job done.

Design and measurements: `docs/superpowers/plans/2026-09-27-nlf-home-gpu-worker.md`.

**Licence.** The NLF weights and the SMPL body model are licensed for non-commercial research
only. The app is a research prototype, so this is fine; keep it free, with no ads and no paid
pilots. Only rendered images leave the PC; the SMPL model file and its triangle list never do.

## One-time setup

Steps marked **(you)** need a dashboard or a password, so only the project owner can do them.

### 1. Database (you)

Run `db/migrations/20260928000000_nlf_3d_view.sql` in the Supabase SQL editor. It adds:

- the `nlf_jobs` and `nlf_worker_heartbeat` tables;
- the `nlf_user` and `nlf_worker` roles;
- the job functions;
- a `has_nlf` column on `admin_list_users()`.

It is safe to run twice. Validate it locally first with `node db/checks/run_pglite.mjs`.

### 2. The worker's login (you)

1. Supabase → Authentication → Providers: make sure **Email** is enabled. The app itself signs
   in with Google, but the worker signs in with email and password.
2. Authentication → Users → **Add user** with an email you control and a long random password.
   Tick auto-confirm.
3. In the SQL editor, with that user's id:

   ```sql
   insert into public.user_roles (user_id, role) values ('<worker-uid>', 'nlf_worker')
   on conflict do nothing;
   ```

This login can only claim, complete and fail jobs and send heartbeats. It cannot read any table
directly, and it never holds the service-role key.

### 3. A dedicated R2 token (you)

In Cloudflare → R2 → **Manage API tokens**, create a token with **Object Read & Write** on the
app's bucket only. Keep it separate from the backend's token, so you can revoke it on its own.

R2 tokens are scoped to a bucket, not a folder, so this token can technically read every
user's upload. Treat this PC accordingly.

### 4. `data/models/nlf/worker.env`

This file is gitignored (everything under `data/` is).

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<anon key, same as the frontend's>
NLF_WORKER_EMAIL=<worker email>
NLF_WORKER_PASSWORD=<worker password>
NLF_R2_ACCOUNT_ID=<cloudflare account id>
NLF_R2_ACCESS_KEY_ID=<dedicated token key id>
NLF_R2_SECRET_ACCESS_KEY=<dedicated token secret>
NLF_R2_BUCKET=<bucket name>
```

Optional keys and their defaults:

| Key | Default | Meaning |
|---|---|---|
| `NLF_MODEL_PATH` | `data/models/nlf/nlf_l_multi.torchscript` | Stock NLF model |
| `NLF_MODEL_FP32_PATH` | `data/models/nlf/nlf_l_multi_fp32.torchscript` | Patched float32 copy |
| `NLF_SMPL_PKL` | `data/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` | SMPL triangle list |
| `NLF_POLL_SECONDS` | `60` | Poll interval |
| `NLF_SAMPLE_FPS` | `10` | Frames sampled per second of video |
| `NLF_BATCH` | `8` | Frames per GPU batch |
| `NLF_TMP_DIR` | `data/runtime/nlf_worker_tmp` | Scratch for downloaded clips |

Environment variables override the file. The worker never prints secret values.

### 5. Python environment and model files

The worker needs its own venv. The model's fixes are tied to **torch 2.5.1**: the half-precision
detector finds nobody under torch 2.13.

```
uv venv --python <path to a Python 3.12 python.exe> .venv-nlf
uv pip install --python .venv-nlf\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-nlf\Scripts\python.exe numpy opencv-python av pyrender trimesh "pyglet<2" boto3 "supabase>=2,<3"
```

Model files:

- **NLF-L.** `https://bit.ly/nlf_l_pt` saved as `data/models/nlf/nlf_l_multi.torchscript`
  (495,696,900 bytes, SHA256 `2090B041…82CDAD`).
  - On first run the worker writes the float32 copy next to it.
  - Why: this GPU (GTX 1660 Ti) produces NaNs in the model's hard-coded half precision.
- **SMPL faces.** The SMPLify download from the Max Planck Institute, unpacked to
  `data/smplify_public/`. It has the same non-commercial licence.

### 6. Smoke test, with no network

```
.venv-nlf\Scripts\python.exe scripts\nlf_worker\run_worker.py --dry-run <clip> --out data\runtime\nlf_dry_run --movement squat --segment <start_s> <end_s>
```

- The outcome should be `done`.
- `--out` gets `pose3d.json.gz`, the turntable strips, `meta.json`, and 0°/90° JPEGs to look at.
- Measured on the phone test clip: about 50 s for 26.5 s of video.

## Enabling users

Admin panel → **Users** → **Enable 3D view** on each person who should get the feature. Only
enabled users see the "Get 3D view" button, on the results page of their own analyses.

## Running the worker

```
.venv-nlf\Scripts\python.exe scripts\nlf_worker\run_worker.py          # poll every NLF_POLL_SECONDS
.venv-nlf\Scripts\python.exe scripts\nlf_worker\run_worker.py --once   # one poll, print the outcome
```

**Start at logon.** In Task Scheduler, create a task with:

- **Trigger:** at log on of your user.
- **Action:** program `<repo>\.venv-nlf\Scripts\python.exe`, arguments
  `scripts\nlf_worker\run_worker.py`, **Start in** `<repo>` (the repository root).
- **General:** *Run only when user is logged on*. The renderer draws through a hidden window
  and needs your desktop session; "run whether logged on or not" has no desktop and rendering
  fails.

## What users see

| State | Meaning |
|---|---|
| **Queued** | Waiting for the PC. If the PC's last heartbeat is older than 10 minutes, the page says the processing PC is offline. It stops polling and offers a manual refresh, so an idle tab doesn't keep the backend awake. |
| **Processing** | The worker has claimed the job. The page polls every 15 s while visible. |
| **Done** | The turntable viewer: drag to rotate, "Side view" jumps to 90°, and the key-frame list switches between rep bottoms and fault peaks. |
| **Failed** | The error, plus "Try again", which re-queues the job. |

The two faint lines in the images are hip and knee **joint-centre** heights. They are not a
"parallel" verdict: coaches judge that by the hip crease against the top of the knee.

## Timing and recovery

- **Typical run:** about 2 s of work per second of video, measured on the GTX 1660 Ti with
  float32 at 10 fps.
- **A PC shut down mid-job:** the claim expires after 30 minutes and the job is re-queued.
  After 3 attempts it becomes `failed` with "claim timed out".
- **Results:** each attempt uploads under its own folder, `{video prefix}/nlf.v1.a{attempt}/`.
- **A video deleted mid-job:** if the user deletes the last analysis of that video, the job
  disappears with it. The worker's `complete` then returns false and it deletes what it just
  uploaded.
- **Normal deletion:** deleting a video's last analysis removes its 3D files along with the
  clip. Account deletion does not yet remove any stored objects. That gap predates this
  feature.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `worker.env` missing, the worker exits with code 2 | Create the file (step 4). |
| `poll error: …` repeating in the log | No network, or a wrong `SUPABASE_URL`. The worker keeps retrying. |
| Sign-in fails | The Email provider is disabled, or the password is wrong (step 2). |
| Claim raises "permission denied" (42501) | The worker's user is missing its `nlf_worker` row in `user_roles`. |
| R2 403 on download or upload | The dedicated token lacks Read & Write on this bucket. |
| Every frame "no detection", or NaN | Running under the wrong venv or torch version; use `.venv-nlf` (torch 2.5.1). |
| Render fails with an OpenGL or pyglet error | The task is not running in your desktop session; see "Start at logon". |
| Jobs stuck on Queued with the PC offline | The worker is not running, or its heartbeat can't reach Supabase. |
