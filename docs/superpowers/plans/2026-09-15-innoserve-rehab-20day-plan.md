# InnoServe 2026 — 復健照護層 20 天實作計畫

Written 2026-09-15. Deadline: InnoServe registration closes **2026-10-05 (Mon) 16:00**
(both 大會專題類 資訊應用組 and 指定專題類 聯新國際智慧健康照護組, same work, two 系統概述文件).
Preliminary judging 10/13–10/23 is document + 3-minute video only; finals 11/7 at NTU.

Goal of the 20 days: turn x-coach from a self-serve fitness coach into a
**therapist → patient home-exercise loop** (assign plan, patient trains with form feedback and
pain check-ins, red flags surface to the therapist), without adding any new ML. Everything
below is scoped to what the Landseed rubric rewards (初賽 市場性 40 / 創新 40 / 完整 20; 決賽
可行性 35) and to what the 5-page 概述文件 and the video can show.

## 0. Starting point (verified 2026-09-15)

- All 14 detectors are on `main` and `GET /api/movements` derives from the registry, so every
  registered movement is already analyzable in the web app (only Squat is `validated=True`;
  the rest carry a Beta tag). No detector work is needed.
- Plans (`/api/plans`, Day 1–7 templates, tick via `PATCH .../items/{id}` with
  `completed_at` + `analysis_id`), Lumen plan agent, admin roles (`user_roles`,
  `is_admin()`), LINE login + reply bot, analyses history are all on `main`.
- Things that do **not** exist yet and that this plan adds: any clinician/patient notion,
  any cross-user read except the two SECURITY DEFINER admin/LINE functions, any persisted
  form score (`formScore()` is client-side only, `100 − 25·Σseverity`), pain/RPE, red flags,
  LINE **push** (only reply exists), any scheduler/cron anywhere in the repo.
- `POST /plans/{id}/start` wipes every item's tick and `analysis_id` by design, so adherence
  history cannot be rebuilt from `plan_items`. This plan stores adherence in a new
  append-only table instead of changing that behaviour.
- Migrations are hand-applied from `db/migrations/` in filename order (Supabase SQL editor
  or `psql`). There is no CI migration step; the user applies them before deploy.

## 1. Product decisions (defaults chosen; change here, not mid-build)

| Decision | Choice | Why |
|---|---|---|
| Who is a clinician | A row in `user_roles` with `role='clinician'` (the column is already free text). `is_clinician(uid)` mirrors `is_admin()`. Admins grant it from `/admin/users`. | Zero new auth concepts; reuses the admin grant UI. |
| Linking a patient | Clinician generates a short invite code; patient enters it under Settings → 我的治療師. Row in `care_links (clinician_id, patient_id, status)`. | Consent is explicit and patient-initiated; no email lookup, no PHI exposure. |
| Assignment | `training_plans.assigned_by uuid null`. Plan owner (`user_id`) stays the patient, so every existing owner-only path keeps working. | Smallest change to a table with 20+ tests behind it. |
| Clinician cross-user read | RLS policies on `training_plans`, `plan_items`, `analyses`, `session_checkins`: allowed when an **active** `care_links` row joins `auth.uid()` to the row's `user_id`. | Backend keeps using the user's own JWT (no service_role), same as the admin panel. |
| Adherence + pain record | New append-only `session_checkins`: `pain_nrs 0–10`, `rpe 0–10`, `note`, `plan_id`, `plan_item_id`, `analysis_id`, `form_score`, `flagged`, `flag_reasons jsonb`. | Survives plan restart; one table feeds adherence, trend, and red flags. |
| Form score server-side | Port `frontend/src/lib/formScore.ts` formula into `backend/app/services/scoring.py`; stored on the check-in, not on `analyses`. | Trend needs it queryable; a drift test keeps the two formulas equal. |
| Red-flag rules (v1, deterministic) | (a) `pain_nrs ≥ 6`; (b) pain up by ≥ 3 vs the previous check-in on the same plan; (c) mean form score of the last 3 sessions ≥ 20 below the previous 3; (d) 7 days without a check-in on an assigned plan. | Explainable, defensible in Q&A, no model. |
| Reminders | GitHub Actions `schedule:` cron (Asia/Taipei 20:00) calls `POST /api/jobs/daily` with a shared `JOB_TOKEN` header; endpoint pushes LINE messages. | Repo already runs GHA; no new infra. Container Apps Jobs is the fallback. |
| Scope of "rehab" wording | Health promotion and exercise assistance. The app never diagnoses or changes a prescription; it reports deviation from what the therapist set and tells the patient to contact them. | Keeps the demo out of SaMD territory; judges will ask. |

Out of scope (say so in the document rather than half-build): new detectors, clinical
validation claims, ROM-based rules beyond what the existing analysis already stores,
EHR/HIS integration, video retention policy changes.

## 2. Calendar

Dev window 9/16–9/30 (15 days). Submission window 10/1–10/5. Feature freeze **9/30 EOD**.

| Days | Work package | Cut line |
|---|---|---|
| 9/16 | WP0 spike + migration | — |
| 9/16–9/18 | WP1 roles, care links, clinic API | must ship |
| 9/19–9/21 | WP2 check-ins, scoring, red flags, patient UI | must ship |
| 9/22–9/24 | WP3 clinician dashboard | must ship |
| 9/25–9/26 | WP4 rehab templates, Lumen rehab mode, disclaimers | must ship |
| 9/27–9/29 | WP5 LINE push + daily job | cut first if behind |
| 9/30 | Freeze: full suites, coverage, deploy, migration applied | — |
| 10/1–10/3 | Field trial (5% bonus), screenshots, 3-min video | — |
| 9/22→10/4 | Two 系統概述文件 in parallel (outline 9/22, draft 9/28, final 10/4) | — |
| 10/4 | Upload everything; 10/5 is buffer only | — |

## 3. Work packages

Each WP is a dispatchable unit: branch, files, acceptance, tests. Backend tests follow the
`_FakeDb` pattern from `tests/test_plans_store.py:128` + `app.dependency_overrides`; frontend
tests use `renderWithProviders` + `vi.mocked(api.X)`. Coverage gate stays 95%
(`scripts/run_backend_coverage.py --fail-under 95`); frontend `yarn test:coverage` from
`frontend/` (serial run reports thresholds, see memory `frontend-coverage-flakes`).

### WP0 — Spike and migration (9/16, half day)

- Confirm what per-analysis numeric metrics are persisted in `analyses.result` (the
  `quality` map, `detections[].evidence`, `reps.segments`). Output: a 10-line note in this
  file listing which numbers a trend chart can use without re-running analysis. If a joint
  angle per rep is not persisted, the trend is form score + adherence only (decision, not a
  blocker).
- Write `db/migrations/20260916000000_clinic.sql`:
  - `is_clinician(uid)` SECURITY DEFINER, same shape as `is_admin`.
  - `care_links(id, clinician_id, patient_id, invite_code text unique, status
    pending|active|revoked, created_at, accepted_at)`; RLS: clinician sees own rows,
    patient sees rows where they are the patient; accept = patient updates status.
  - `training_plans.assigned_by uuid null references auth.users`.
  - `session_checkins` as in §1, owner-only RLS plus clinician read policy.
  - Clinician read policies on `training_plans`, `plan_items`, `analyses` (`select` only)
    and insert/update on `training_plans`/`plan_items` where the patient is actively linked.
  - Policy rule from the admin migration applies: a policy on a table must not select from
    itself (recursion), so the link check is a SECURITY DEFINER `is_linked(clinician, patient)`.
- Acceptance: migration applies cleanly on a fresh local Postgres or the Supabase SQL editor
  with no error; `db/migrations/README.md` gains the new file in the ordered list.

#### WP0 findings (2026-09-15, done)

Metrics a trend can use without re-running analysis (`analyses.result`, one write site,
`routers/analyze.py:240-250`):
- Always there for a registered movement: `quality.*` (8 numbers; `wasMeasured` is
  `valid_frame_ratio > 0`), `view.view_confidence` and scores, `reps.detected`,
  `len(reps.analyzed)`, `movement`, `created_at`. So form score is always computable.
- Only when a fault fires, worst rep only: `detections[].severity` and `evidence` (one
  `max_avg_knee_angle` for squat depth, `max_lead_knee_angle_deg` for lunge). Missing means "no
  fault", not 0.
- No per-rep joint angle is persisted (`frame_metrics` is dropped before persist); it could be
  recomputed from the stored `result.pose` landmarks, which is re-analysis in all but name.
- **Decision: trend = form score + pain NRS + adherence.** No ROM line.
- `fault_count` counts the squat's severity-0 "side view required" notice; don't chart it.

Schema deviations from §1/WP0, all forced by what the tables and RLS allow:
- `user_roles.user_id` was the PRIMARY KEY (one role per user). Re-keyed to `(user_id, role)`
  so an admin can also be a clinician; `store.set_user_role` must upsert on `user_id,role` and
  delete by role, and has to ship with the migration.
- Accepting an invite cannot be "patient updates status": on a pending row `patient_id` is null,
  so no policy lets the patient see it without making codes enumerable. Accept, revoke and flag
  acknowledgement are SECURITY DEFINER functions (`accept_care_invite`, `revoke_care_link`,
  `ack_checkin_flag`); `care_links` has no UPDATE policy, since one would let a clinician rewrite
  `patient_id` to any user. Codes expire after 7 days (`expires_at`).
- Names for the other party come from definer functions `clinic_patients()` / `my_clinicians()`
  (auth.users is not readable with a user JWT).
- `store.list_analyses` relies on RLS alone for "mine"; with the clinician policy it would list a
  clinician's patients' analyses in their own history. WP1 adds an explicit `user_id` filter.
- Clinicians get no policy on `videos`: the dashboard shows results, never plays the clip.
- Verified: `db/checks/clinic_rls_check.sql` (13 groups, run as each role) passes on a fresh
  database with every migration applied and the clinic migration re-applied; two deliberately
  broken policies each make it fail.

### WP1 — Roles, links, clinic API (9/16–9/18)

Backend, branch `feat/clinic-core`.

- `backend/app/auth.py`: `get_clinician_user()` (403 otherwise), mirroring `get_admin_user`.
- `backend/app/services/store.py`: `is_clinician(token, user_id)`.
- New `backend/app/services/clinic.py` + `backend/app/routers/clinic.py` (`/api/clinic`):
  - `GET /clinic/status` → `{is_clinician}` (uses `get_current_user`, like admin status).
  - `POST /clinic/invites` → creates a `care_links` row with a 6-char code, `pending`.
  - `POST /clinic/links/accept {code}` (patient) → sets `patient_id`, `active`.
  - `GET /clinic/patients` → linked patients with `last_checkin_at`, `adherence_7d`,
    `open_flags`.
  - `GET /clinic/patients/{id}` → plans (with `assigned_by`), last 30 check-ins, trend
    series (dates, form_score, pain_nrs), open flags.
  - `POST /clinic/patients/{id}/plans` → same body as `POST /plans` (template_key or items)
    but `user_id = patient`, `assigned_by = clinician`. Reuses `services/plans.create_plan`
    with an explicit owner parameter (one new kwarg, default = caller).
  - `DELETE /clinic/links/{id}` → revoke.
- Admin: `/admin/users` gains a "clinician" toggle next to the admin toggle (backend
  `PATCH /api/admin/users/{id}/roles` extended; frontend `pages/admin/Users`).
- Tests: `tests/test_clinic_api.py`, `tests/test_clinic_store.py` (FakeDb), extend
  `tests/test_backend_admin*.py` for the role toggle. Acceptance: a non-clinician gets 403 on
  every `/api/clinic/*` except status; a clinician cannot read a patient who has not
  accepted; assigned plan appears in the patient's `GET /plans` with `assigned_by` set.

### WP2 — Check-ins, scoring, red flags, patient UI (9/19–9/21)

Branch `feat/clinic-checkins`.

- `backend/app/services/scoring.py`: `form_score(detections, quality) -> int | None`, same
  formula and `wasMeasured` gate as `frontend/src/lib/formScore.ts`. Drift test parses the
  TS constants off disk, same as `tests/test_movement_muscles.py` does for muscles.
- `backend/app/services/redflags.py`: pure function `evaluate(new_checkin, history) ->
  list[reason]` implementing rules (a)–(c) of §1; rule (d) lives in the daily job (WP5) and
  degrades to "shown on dashboard only" if WP5 is cut.
- Router: `POST /api/checkins` (body: plan_id, plan_item_id?, analysis_id?, pain_nrs, rpe,
  note) → computes `form_score` from the linked analysis, runs red flags, stores;
  `GET /api/checkins?plan_id=`; `PATCH /api/clinic/flags/{checkin_id}/ack` (clinician).
- Frontend (patient):
  - `components/checkin/CheckinDialog.tsx`: NRS 0–10 slider + RPE 0–10 + note. Opens after
    the studio ticks a plan item (`App.tsx:219` path) and from `PlanItemRow` (「回報今天狀況」).
  - Settings → 「我的治療師」: enter invite code, show linked clinician, unlink.
  - Red-flag banner on `/plans/:id` when the latest check-in is flagged: fixed copy,
    "請聯絡你的治療師", no advice generation.
  - i18n keys in both languages.
- Tests: `tests/test_checkins_api.py`, `tests/test_redflags.py` (table-driven, every rule
  has a positive and a negative case), `tests/test_scoring_drift.py`; frontend
  `components.CheckinDialog.test.tsx`, `pages.Settings.clinician.test.tsx`,
  `App.checkinAfterTick.test.tsx`.
- Acceptance: pain 7 flags; pain 3→6 flags; three 90s then three 60s flags; 5→5 does not.

### WP3 — Clinician dashboard (9/22–9/24)

Branch `feat/clinic-dashboard`. Pattern: copy `AdminLayout` gating
(`adminState === "ready" && isAdmin`) into `ClinicLayout` with `isClinician` resolved from
`GET /api/clinic/status` in `lib/auth.tsx`.

- Routes in `AppRoutes.tsx`: `/clinic` (patients list), `/clinic/patients/:id`,
  `/clinic/invite`. Sidebar link gated on `isClinician`.
- `/clinic`: table of patients — name/email, last check-in, 7-day adherence, open flags
  (red badge), 「指派菜單」 button.
- `/clinic/patients/:id`: three panels — assigned plans (reuse `PlanPreview`/`WeekStrip`,
  read-only), trend chart (form score + pain over last 30 days; use the existing chart
  approach in the app, no new library), check-in list with flags and 「已處理」.
- Assign flow: `CreatePlanDialog` reused with a `forPatient` prop; rehab templates (WP4)
  listed first.
- Tests: `pages.Clinic.test.tsx`, `pages.ClinicPatient.test.tsx`, `lib.auth.clinician.test.tsx`,
  `components.Sidebar.clinician.test.tsx`, `api.clinic.test.ts`.
- Acceptance: a clinician with two linked patients sees both, one flagged; a non-clinician
  hitting `/clinic` is redirected; screenshots of both pages captured with the headless-Edge
  recipe (memory `frontend-visual-check-recipe`) for the 概述文件.

### WP4 — Rehab templates, Lumen rehab mode, disclaimers (9/25–9/26)

Branch `feat/rehab-templates`.

- `routers/plans.py TEMPLATES` gains three rehab templates, all built from **registered**
  movements only: `knee_rehab` (squat, lunge, leg abduction, shoulder bridge),
  `shoulder_rehab` (arm abduction, arm vw, band pull apart, row), `low_back_core`
  (shoulder bridge, sit-up, torso twist, deadlift light). Add `category: "rehab" | "fitness"`
  to `PlanTemplate`; frontend groups by category. Sets/reps conservative (2×10).
  `tests/test_plans_api.py` template tests extended; the 16-name catalog guard must stay green.
- Lumen: `plan_agent._system_prompt` and `chat._system_preamble` take `rehab: bool`, true
  when the plan has `assigned_by` (plan agent) or the analysis is linked to such a plan
  (chat, via `ChatContext`). Rehab block: never diagnose, never change sets/reps/movements of
  an assigned plan (plan tools `add_item/remove_item/update_item` refuse with a fixed message
  when `assigned_by` is set), encourage, point to the therapist on pain.
  `tests/test_plan_agent.py`: assigned plan → mutation tools return the refusal payload.
- Disclaimer copy (i18n) on `/clinic`, `CheckinDialog`, and the analysis report footer when
  the plan is assigned.
- Movement Beta tag: keep. The 概述文件 explains it honestly (only Squat validated on
  labeled data; REHAB24-6 experiments cover the rehab movements' feasibility).

### WP5 — LINE push + daily job (9/27–9/29, cut first)

Branch `feat/line-push-reminders`.

- `line_bot.push(line_sub, text)` → `POST https://api.line.me/v2/bot/message/push`; the LINE
  user id comes from `auth.users.email = line_<sub>@line.invalid` (the existing identity key),
  resolved in a SECURITY DEFINER `line_sub_for_user(uid)` granted to service_role, or by the
  existing service-role client used for the summary RPC.
- `POST /api/jobs/daily` guarded by `JOB_TOKEN` (settings + Azure secret): for every active
  assigned plan with no check-in today → push reminder to the patient; for every clinician
  with linked patients → push one summary line (adherence + open flags); apply red-flag rule
  (d). Idempotent per day (record in `app_settings` key `jobs.daily.last_run`).
- `.github/workflows/daily-jobs.yml`: `schedule: cron: '0 12 * * *'` (20:00 Taipei) +
  `workflow_dispatch`, `curl` with the token from repo secrets.
- Tests: `tests/test_line_push.py` (httpx patched, like `test_backend_admin_line.py`),
  `tests/test_jobs_daily.py` (FakeDb; asserts one push per patient per day, none when a
  check-in exists, 401 without token).
- Acceptance: `workflow_dispatch` against the Azure deployment produces one real LINE push
  to a test account. If this is not green by 9/29 EOD, ship without it and describe reminders
  as "planned" in the document — do not let it slip into the freeze day.

### 9/30 — Freeze and deploy

- Full backend suite, coverage gate, frontend coverage (serial), `graphify update .`.
- User applies `20260916000000_clinic.sql` on Supabase, grants one clinician role to a demo
  account, sets `JOB_TOKEN` in Container Apps if WP5 shipped.
- Deploy via the existing GHA → GHCR → Container Apps path. Set `minReplicas=1` for
  10/1–10/5 (video recording) and again for 11/7; scale-to-zero cold starts would show in
  the video.
- Seed two demo patients with 2–3 weeks of check-ins and analyses so the trend chart and a
  red flag are visible in the video. Seeding script under `scripts/demo/seed_clinic.py`
  (hits the real API with two demo accounts; never runs in CI).

## 4. Submission work (non-code, runs in parallel from 9/22)

- **Field trial for the Landseed +5%** (10/1–10/3): one physiotherapy clinic, campus
  athletic trainer, or sports-medicine department; 3–5 users do one assigned session each with
  a check-in. Keep a dated log (who/where/how many/feedback) and one photo without faces or
  logos; describe it in the 概述文件 前言 or 結語. Consent form for any recorded video.
- **Two 系統概述文件** (Word, A4, 標楷體 14, ≤5 pages, ≤4 MB, fixed eight headings):
  - 資訊應用組: lead with explainable feedback (rule-based faults with citations, KG/RAG
    grounding, per-rep attribution), the registry-driven 14-movement architecture, and
    extensibility; the care loop is the applied example.
  - 聯新組: lead with the care loop (therapist assigns → patient trains at home → check-in
    → red flag → therapist acts), the four Landseed sub-domains it touches (運動醫學, 智慧照護
    early warning, 智慧健康), market and workflow fit for a 無圍牆醫院, the REHAB24-6
    evidence, and the honest limits (2D monocular depth, Beta movements, no diagnosis).
  - Both: disclose generative-AI use (OpenRouter coaching chat, Gemini KG extraction) and
    what the team built on top of it.
- **Video 3 min** (10/2–10/3): record on the Azure deployment. Storyboard: 0:00 problem
  and users (20 s) → therapist assigns knee_rehab (30 s) → patient records lunge, gets
  per-rep explainable feedback (60 s) → check-in with pain 7 → red flag on the therapist
  dashboard, LINE message if WP5 shipped (40 s) → architecture and limits (30 s). YouTube
  unlisted.
- **Anonymity sweep** before recording: no school name/logo/advisor name in the UI (login
  page, footer, about, admin pages, demo emails), file names, or the video.
- **Forms**: 附件 5 切結書 signed by every member (PDF ≤2 MB), student IDs (PDF ≤2 MB; add
  在學證明 if the IC card lacks department/registration stamp), 16:9 group photo >1280×720 with
  the advisor, 1–2 advisors registered, ≤8 students, 資管系 ≤ half if any.
- **Upload 10/4**, not 10/5. Both groups need their own 概述文件 upload.

## 5. Risks and pre-decided responses

- **RLS recursion or policy bugs** show up only against real Postgres. Mitigation: WP0
  applies the migration on a local Postgres (docker) or a throwaway Supabase project before
  any backend code, and `tests/` never pretend to test policies.
- **Coverage gate 95%** with ~6 new modules: every WP ships tests in the same PR; no "tests
  later".
- **Frontend coverage flakes** grow with tree size: use the serial run for thresholds.
- **LINE push needs a real channel**: WP5 is the only package with an external dependency,
  hence last and cut first.
- **Judges ask "is this a medical device?"**: answer is in §1 (health promotion, no
  diagnosis, deviation reporting only) and repeated in the document.
- **Two 5-page documents plus a video in the last 4 days is too tight** if started then:
  the outline is due 9/22 and drafts 9/28, written by whoever is not on the critical WP.

## 6. Definition of done for the whole programme (10/4)

1. A clinician account can invite, a patient can accept, the clinician assigns a rehab
   template, the patient trains a movement and checks in, a red flag appears on `/clinic`.
2. Backend and frontend suites green locally at CI settings; coverage ≥95%.
3. Deployed on Azure with the migration applied; demo data seeded; anonymity sweep done.
4. Two 概述文件, one video link, forms, photo uploaded for both groups by 10/4.
