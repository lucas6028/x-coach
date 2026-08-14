-- Training plans ("訓練菜單"): a user-authored weekly routine over the movement catalog.
--
-- Design notes:
--   * A plan is a REUSABLE TEMPLATE, not a dated schedule. Items sit in relative day slots
--     (Day 1..7), and "start this plan" stamps `started_at` and clears every item's progress.
--     There is deliberately no calendar date anywhere: the user picks up a plan whenever they
--     train, and the same plan is meant to be run again next week without being copied.
--   * There is NO `day_count` column. The number of days a plan spans is derived from its items
--     (the UI always offers the seven slots and renders the used ones), so a plan can never hold
--     an item on a day that its own length says does not exist -- a drift a stored count would
--     have allowed the moment the two were edited independently.
--   * `movement` is a plain text column, NOT a foreign key: the movement catalog lives in code
--     (src/pose/movements/catalog.py, 16 names) and the API validates against it. A DB enum would
--     mean a migration every time the catalog changes, and a lookup table would duplicate a list
--     the Python side already owns.
--   * All sixteen catalog movements are plannable, but only the fourteen with a registered
--     detector can be ANALYSED (GET /api/movements). The frontend renders the other two as
--     manual-tick-only. That asymmetry is intentional and lives in the app layer, not here.
--
-- WHAT RESTARTING A PLAN THROWS AWAY, stated here because it reads like a bug otherwise:
-- `POST /api/plans/{id}/start` nulls `completed_at` and `analysis_id` on every item. A plan item
-- completed in a previous run therefore reverts to unticked and loses its link to the analysis it
-- produced -- while the analysis itself survives untouched in `analyses` and stays visible in
-- 我的紀錄. The plan tracks the CURRENT run only; the training record is the analysis history.
-- Keeping per-run history instead would need a third table (plan_runs) and buys the user nothing
-- that /history does not already show.

-- ---------------------------------------------------------------------------
-- training_plans: one row per user-authored routine
-- ---------------------------------------------------------------------------
create table if not exists public.training_plans (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users (id) on delete cascade,
    name        text not null check (char_length(name) between 1 and 80),
    notes       text check (notes is null or char_length(notes) <= 500),
    -- Which built-in template this plan was copied from, or NULL when built from scratch.
    -- Provenance only: the copy is independent, so editing a template in code never mutates a
    -- plan a user already owns.
    template_key text,
    -- NULL until the user presses "開始這份菜單". Set (and item progress cleared) on every start,
    -- so it reads as "this run began at".
    started_at  timestamptz,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists training_plans_user_created_idx
    on public.training_plans (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- plan_items: one exercise slot inside a plan's day
-- ---------------------------------------------------------------------------
create table if not exists public.plan_items (
    id          uuid primary key default gen_random_uuid(),
    plan_id     uuid not null references public.training_plans (id) on delete cascade,
    -- Denormalized from the parent plan so the RLS policy is a column comparison rather than a
    -- subquery on every row read. The API only ever writes it from the authenticated user's id.
    user_id     uuid not null references auth.users (id) on delete cascade,
    day_index   int  not null check (day_index between 1 and 7),
    -- Order within the day. Ties are broken by created_at, so a client that never sets it still
    -- gets a stable, sensible order.
    position    int  not null default 0,
    movement    text not null,               -- canonical catalog spelling, e.g. "Overhead Press"
    sets        int  not null default 3  check (sets between 1 and 20),
    reps        int  not null default 10 check (reps between 1 and 200),
    notes       text check (notes is null or char_length(notes) <= 200),
    -- Progress for the CURRENT run (see the restart note at the top of this file).
    completed_at timestamptz,
    -- The analysis the user produced for this item, when they trained it through the studio
    -- rather than just ticking it off. `on delete set null` -- NOT a bare uuid column -- because
    -- deleting an analysis from 我的紀錄 is a supported action, and a dangling id would leave the
    -- plan card offering a 看報告 link that 404s.
    analysis_id uuid references public.analyses (id) on delete set null,
    created_at  timestamptz not null default now()
);

create index if not exists plan_items_plan_day_idx
    on public.plan_items (plan_id, day_index, position);

-- Keep training_plans.updated_at fresh as the plan is edited. `touch_updated_at()` already exists
-- from the initial videos/analyses migration; `create or replace` there means it is safe to
-- assume, and this migration only attaches the trigger.
drop trigger if exists training_plans_touch_updated_at on public.training_plans;
create trigger training_plans_touch_updated_at
    before update on public.training_plans
    for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: a user can only see/write their own plans and items.
-- ---------------------------------------------------------------------------
alter table public.training_plans enable row level security;
alter table public.plan_items     enable row level security;

drop policy if exists "training_plans_owner_all" on public.training_plans;
create policy "training_plans_owner_all" on public.training_plans
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists "plan_items_owner_all" on public.plan_items;
create policy "plan_items_owner_all" on public.plan_items
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
