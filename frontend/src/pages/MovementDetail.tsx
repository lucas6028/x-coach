import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowsLeftRight,
  CaretRight,
  Check,
  Crosshair,
  Lightbulb,
  MapPin,
  Path,
  Record as RecordIcon,
  ShieldCheck,
  Timer,
  UploadSimple,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { Link, Navigate, useNavigate, useParams, useSearchParams } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import MovementArt from "../components/movements/MovementArt";
import MuscleMap from "../components/movements/MuscleMap";
import { api, type HistoryItem, type Retrieval } from "../api";
import { useAuth } from "../lib/auth";
import { movementLabel, useI18n } from "../lib/i18n";
import { MOVEMENT_GROUPS, type AnalyzableMovement } from "../lib/movements";
import { movementDetail, type Muscle, type MovementDetail as Detail } from "../lib/movementDetail";
import { movementMistakes, type Mistake } from "../lib/movementMistakes";
import { summaryCategory } from "../lib/retrieval";

// One movement, in full: what it is, how it is performed, which muscles it trains, the faults the
// knowledge graph defines for it, and the user's own history of it. Reached from the "View details"
// button on every library card — including the thirteen that cannot be analysed yet, which is the
// point: without this page they were dead tiles.
//
// Built against the exercise-detail reference mock (~/Downloads/detail_page). It follows the
// mock's Overview layout closely; where the mock states a fact this app does not have, the card is
// dropped rather than filled with a plausible number. Specifically NOT ported:
//
//  * "Average duration" and "Calories (est.)" in the details table. Both are invented in the mock,
//    and neither the pipeline nor lib/calories.ts (whose METs are keyed to the games, not to
//    exercises) can produce them. The table carries five taxonomy rows plus the one fact we
//    genuinely own: whether this movement can be analysed at all.
//  * The four-up "Demo videos" grid of stock footage. Four movements have a real clip under
//    public/demo; they get a single player. The other twelve get no card, rather than another
//    movement's footage relabelled.
//  * The mock's own top bar (search + "All categories"). That is its shell, and it duplicates the
//    library page this one is opened from.

const TABS = ["overview", "howTo", "mistakes", "muscles", "records"] as const;
type Tab = (typeof TABS)[number];

const TAB_KEY: Record<Tab, string> = {
  overview: "detail.tabOverview",
  howTo: "detail.tabHowTo",
  mistakes: "detail.tabMistakes",
  muscles: "detail.tabMuscles",
  records: "detail.tabRecords",
};

const CARD = "rounded-[22px] border border-border-dark bg-surface p-4 shadow-card";

export default function MovementDetail() {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const { movement: rawParam = "" } = useParams();
  // The route key is the canonical movement name, percent-encoded — the same identity the
  // catalog, the KG and ?movement= use. No slug table: a second naming scheme is a second thing
  // to keep in sync (public/movements/ already proves it, with "pushup.png" for "Push-up").
  const movement = decodeURIComponent(rawParam);
  const detail = movementDetail(movement);

  // The open tab lives in the URL so a link can point at one ("the squat's common mistakes"), and
  // so the browser's back button steps between tabs the way it looks like it should. Replaced
  // rather than pushed on click: five tabs pushing history would bury the page you arrived from.
  const [searchParams, setSearchParams] = useSearchParams();
  const requested = searchParams.get("tab");
  const tab: Tab = TABS.includes(requested as Tab) ? (requested as Tab) : "overview";
  const setTab = (next: Tab) =>
    setSearchParams(next === "overview" ? {} : { tab: next }, { replace: true });

  // Same source and same fallback as the library and the studio: on failure assume Squat only,
  // because the alternative — offering every movement — sends the user to record a clip we would
  // then grade with the wrong rules.
  const [live, setLive] = useState<AnalyzableMovement[]>([{ name: "Squat", validated: true }]);
  useEffect(() => {
    let cancelled = false;
    api
      .getMovements()
      .then((ms) => {
        if (!cancelled && ms.length) setLive(ms);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const group = MOVEMENT_GROUPS.find((g) => (g.items as readonly string[]).includes(movement));
  const related = useMemo(
    () => (group?.items ?? []).filter((m) => m !== movement).slice(0, 4),
    [group, movement]
  );

  // A movement outside the catalog is a mistyped or stale URL, not an error worth a page: send it
  // back to the library, which lists what does exist.
  if (!detail || !group) return <Navigate to="/movements" replace />;

  const entry = live.find((m) => m.name === movement);
  const label = movementLabel(t, movement);

  return (
    <AppLayout title={label}>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <main className="mx-auto max-w-6xl px-4 py-8 lg:px-6 lg:py-10">
          <nav className="mb-2 flex flex-wrap items-center gap-x-1.5 text-[12px] font-medium text-faint">
            <Link to="/" className="transition-colors hover:text-primary">
              {t("studio.crumbHome")}
            </Link>
            <span className="opacity-60">/</span>
            <Link to="/movements" className="transition-colors hover:text-primary">
              {t("detail.crumbLibrary")}
            </Link>
            <span className="opacity-60">/</span>
            <span className="text-muted">{label}</span>
          </nav>

          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="font-display text-[28px] font-bold leading-none tracking-tight text-content lg:text-[32px]">
              {label}
            </h1>
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-semibold text-primary">
              <MapPin size={12} weight="fill" />
              {t(group.key)}
            </span>
            {entry && !entry.validated && (
              <span
                title={t("movements.betaNote")}
                className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warning ring-1 ring-warning/40"
              >
                {t("movements.beta")}
              </span>
            )}
          </div>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted">
            {detail.description[lang]}
          </p>

          {/* Tabs on the left, the way out on the right — the mock's row, minus its search bar. */}
          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-b border-border-dark">
            <div role="tablist" className="scrollbar-none flex min-w-0 gap-6 overflow-x-auto">
              {TABS.map((x) => (
                <button
                  key={x}
                  role="tab"
                  aria-selected={tab === x}
                  onClick={() => setTab(x)}
                  className={`relative shrink-0 pb-3 text-[13.5px] transition-colors ${
                    tab === x
                      ? "font-semibold text-primary"
                      : "font-medium text-muted hover:text-content"
                  }`}
                >
                  {t(TAB_KEY[x])}
                  {tab === x && (
                    <span className="absolute inset-x-0 -bottom-px h-[2.5px] rounded-full bg-primary" />
                  )}
                </button>
              ))}
            </div>
            <Link
              to="/movements"
              className="mb-2 inline-flex h-10 shrink-0 items-center gap-2 rounded-2xl border border-border-dark bg-surface px-4 text-[13px] font-semibold text-muted transition-colors hover:text-content"
            >
              <ArrowLeft size={15} weight="bold" />
              {t("detail.back")}
            </Link>
          </div>

          {tab === "overview" && (
            <Overview
              movement={movement}
              detail={detail}
              entry={entry}
              related={related}
              onAnalyze={(mode) =>
                navigate(
                  `/app?movement=${encodeURIComponent(movement)}${
                    mode === "record" ? "&capture=record" : ""
                  }`
                )
              }
            />
          )}
          {tab === "howTo" && <HowToTab movement={movement} detail={detail} />}
          {tab === "mistakes" && <MistakesTab movement={movement} />}
          {tab === "muscles" && <MusclesTab detail={detail} />}
          {tab === "records" && <RecordsTab movement={movement} label={label} />}
        </main>
      </div>
    </AppLayout>
  );
}

// ---------------------------------------------------------------------------------------------
// Overview — the tab the reference screenshot shows, and the one this page is judged on. Three
// rows of cards, in the mock's own column ratios.
// ---------------------------------------------------------------------------------------------

function Overview({
  movement,
  detail,
  entry,
  related,
  onAnalyze,
}: {
  movement: string;
  detail: Detail;
  entry?: AnalyzableMovement;
  related: readonly string[];
  onAnalyze: (mode: "upload" | "record") => void;
}) {
  return (
    <div className="mt-4 space-y-3.5">
      <div className="grid gap-3.5 xl:grid-cols-[1.42fr_1fr]">
        <HowToCard movement={movement} detail={detail} />
        <MusclesCard detail={detail} />
      </div>

      <div className="grid gap-3.5 xl:grid-cols-[1.2fr_0.88fr_0.82fr]">
        <AnalyseCard live={Boolean(entry)} onAnalyze={onAnalyze} />
        <WhatWeAnalyse />
        <DetailsCard detail={detail} entry={entry} />
      </div>

      <div className="grid gap-3.5 xl:grid-cols-[1.42fr_1fr]">
        {detail.demo ? <DemoCard clip={detail.demo.clip} /> : <span className="hidden xl:block" />}
        <RelatedCard related={related} />
      </div>
    </div>
  );
}

// The step strip. Every movement in the catalog now has real figures, so the card-art fallback
// below is currently unreachable — it stays for the next movement added without a set, which would
// otherwise render five blanks.
//
// The max-width is what stops a figure from spilling into its neighbour's column: the strip gives
// each step about that much room. It binds only on the sets wider than the box (Shoulder Bridge at
// ~2.6:1, Sit-up at ~1.7:1 and Push-up at ~1.6:1), which then read shorter than the 132px box —
// correct, since all three are performed on the floor, and consistent within a set because its
// five share a canvas.
function StepFigure({ movement, image }: { movement: string; image?: string }) {
  if (image) {
    return (
      <img
        src={image}
        alt=""
        loading="lazy"
        decoding="async"
        className="h-full w-auto max-w-[118px] object-contain object-bottom"
      />
    );
  }
  return (
    <span className="flex h-full w-[92px] items-end justify-center opacity-70">
      <MovementArt movement={movement} />
    </span>
  );
}

function HowToCard({ movement, detail }: { movement: string; detail: Detail }) {
  const { t, lang } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.howTo")}</h2>
      <div className="scrollbar-none mt-3 flex items-start justify-between gap-1 overflow-x-auto pb-1">
        {detail.steps.map((step, i) => (
          <div key={i} className="flex min-w-0 flex-1 items-start">
            <div className="flex min-w-0 flex-1 flex-col items-center text-center">
              <div className="flex h-[132px] w-full items-end justify-center">
                <StepFigure movement={movement} image={step.image} />
              </div>
              <div className="mt-3 flex items-start gap-2 px-1">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-content">
                  {i + 1}
                </span>
                <p className="text-left text-[11.5px] leading-snug text-muted">{step.text[lang]}</p>
              </div>
            </div>
            {i < detail.steps.length - 1 && (
              <CaretRight size={14} className="mt-14 shrink-0 text-faint" />
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function MuscleLegend({ muscles, tone }: { muscles: Muscle[]; tone: "primary" | "secondary" }) {
  const { t } = useI18n();
  return (
    <div>
      <p className="mb-2 flex items-center gap-2 text-[12px] font-semibold text-content">
        <span
          className={`h-2.5 w-2.5 rounded-full ${tone === "primary" ? "bg-primary" : "bg-primary/30"}`}
        />
        {t(tone === "primary" ? "detail.primary" : "detail.secondary")}
      </p>
      {muscles.map((m) => (
        <div key={m} className="mb-1.5 flex items-center gap-2 pl-[18px] text-[12.5px] text-muted">
          <span
            className={`h-2 w-2 shrink-0 rounded-full ${
              tone === "primary" ? "bg-primary" : "bg-primary/30"
            }`}
          />
          {t(`muscle.${m}`)}
        </div>
      ))}
    </div>
  );
}

// The figures: the movement's illustrated plate where one has been drawn, and the schematic body
// map everywhere else. Both carry the same information — which regions this movement trains — so
// the legend beside them needs no branch.
function MuscleFigures({
  detail,
  size,
  plateSize,
}: {
  detail: Detail;
  /** Height of ONE drawn figure; two are rendered side by side. */
  size: string;
  /** Height of the plate, which carries both views in one image and so runs taller than a
   *  single drawn figure to put the two bodies at a comparable size. */
  plateSize: string;
}) {
  const { t } = useI18n();
  if (detail.plate) {
    return (
      <img
        src={detail.plate}
        // The plate shows anterior and posterior side by side, and the groups it highlights are
        // named in the legend next to it, so it carries no alt of its own.
        alt=""
        loading="lazy"
        decoding="async"
        className={`${plateSize} w-auto object-contain`}
      />
    );
  }
  return (
    <>
      <figure className="text-center">
        <MuscleMap
          side="front"
          primary={detail.primary}
          secondary={detail.secondary}
          className={`${size} w-auto`}
        />
        <figcaption className="mt-2 text-[12px] font-medium text-muted">
          {t("detail.anterior")}
        </figcaption>
      </figure>
      <figure className="text-center">
        <MuscleMap
          side="back"
          primary={detail.primary}
          secondary={detail.secondary}
          className={`${size} w-auto`}
        />
        <figcaption className="mt-2 text-[12px] font-medium text-muted">
          {t("detail.posterior")}
        </figcaption>
      </figure>
    </>
  );
}

function MusclesCard({ detail }: { detail: Detail }) {
  const { t } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.muscles")}</h2>
      <div className="mt-2 flex items-center gap-2">
        <div className="flex flex-1 items-end justify-evenly pt-1">
          <MuscleFigures detail={detail} size="h-[208px]" plateSize="h-[250px]" />
        </div>
        <div className="w-[124px] shrink-0 space-y-4 pt-3">
          <MuscleLegend muscles={detail.primary} tone="primary" />
          <MuscleLegend muscles={detail.secondary} tone="secondary" />
        </div>
      </div>
    </section>
  );
}

function AnalyseCard({
  live,
  onAnalyze,
}: {
  live: boolean;
  onAnalyze: (mode: "upload" | "record") => void;
}) {
  const { t } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.analyse")}</h2>
      <p className="mt-1 text-[13px] text-muted">{t("detail.analyseSub")}</p>

      {/* The two actions are hidden, not disabled, when no detector is registered: the studio
          would refuse the upload anyway, and a button that opens a refusal is worse than a
          sentence saying so. */}
      {live ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <div className="rounded-[20px] bg-content/[0.03] px-5 py-6 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
              <RecordIcon size={20} weight="fill" />
            </div>
            <h3 className="mt-3 text-[14px] font-bold text-content">{t("detail.recordLive")}</h3>
            <p className="mx-auto mt-1 max-w-[190px] text-[12px] leading-relaxed text-muted">
              {t("detail.recordLiveSub")}
            </p>
            <button
              type="button"
              onClick={() => onAnalyze("record")}
              className="mt-4 inline-flex h-10 items-center gap-2 rounded-full bg-primary px-5 text-[13px] font-semibold text-primary-content shadow-accent transition-colors hover:bg-primary/90"
            >
              <RecordIcon size={14} weight="fill" />
              {t("detail.startRecording")}
            </button>
          </div>
          <div className="rounded-[20px] bg-content/[0.03] px-5 py-6 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
              <UploadSimple size={20} weight="bold" />
            </div>
            <h3 className="mt-3 text-[14px] font-bold text-content">{t("detail.uploadVideo")}</h3>
            <p className="mx-auto mt-1 max-w-[190px] text-[12px] leading-relaxed text-muted">
              {t("detail.uploadVideoSub")}
            </p>
            <button
              type="button"
              onClick={() => onAnalyze("upload")}
              className="mt-4 inline-flex h-10 items-center gap-2 rounded-full border border-border-dark bg-surface px-5 text-[13px] font-semibold text-content transition-colors hover:border-primary/40"
            >
              <UploadSimple size={14} weight="bold" />
              {t("movements.uploadCta")}
            </button>
          </div>
        </div>
      ) : (
        <p className="mt-4 rounded-[20px] border border-warning/40 bg-warning/10 px-4 py-3 text-[12.5px] leading-relaxed text-content">
          {t("detail.notAnalyzable")}
        </p>
      )}

      {/* The clip constraints, in the words the dropzone itself uses — one string, so the two can
          never claim different limits. */}
      <div className="mt-3 flex items-center gap-2 rounded-2xl bg-content/[0.03] px-4 py-2.5 text-[12px] text-muted">
        <ShieldCheck size={14} className="shrink-0 text-faint" />
        {t("upload.hint")}
      </div>
    </section>
  );
}

// What the pipeline actually does, in five lines. Authored copy about OUR analyzer, not about the
// movement — so unlike the mock's version it says nothing squat-specific.
const ANALYSE_ITEMS = [
  { Icon: Crosshair, key: "wa1" },
  { Icon: ArrowsLeftRight, key: "wa2" },
  { Icon: Timer, key: "wa3" },
  { Icon: Path, key: "wa4" },
  { Icon: ShieldCheck, key: "wa5" },
];

function WhatWeAnalyse() {
  const { t } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.whatWeAnalyse")}</h2>
      <div className="mt-4 space-y-3.5">
        {ANALYSE_ITEMS.map(({ Icon, key }) => (
          <div key={key} className="flex gap-3">
            <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Icon size={16} weight="duotone" />
            </span>
            <div className="min-w-0">
              <p className="text-[13px] font-semibold text-content">{t(`detail.${key}.title`)}</p>
              <p className="text-[11.5px] leading-snug text-muted">{t(`detail.${key}.text`)}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function DetailsCard({ detail, entry }: { detail: Detail; entry?: AnalyzableMovement }) {
  const { t } = useI18n();

  const analysisState = !entry
    ? t("detail.analysisNone")
    : entry.validated
      ? t("detail.analysisLive")
      : t("detail.analysisBeta");

  const rows: Array<[string, string, boolean]> = [
    [t("detail.difficulty"), t(`difficulty.${detail.difficulty}`), true],
    [t("detail.equipment"), t(`equipment.${detail.equipment}`), false],
    [t("detail.type"), t(`exerciseType.${detail.type}`), false],
    [t("detail.mechanic"), t(`mechanic.${detail.mechanic}`), false],
    [t("detail.force"), t(`force.${detail.force}`), false],
    // The one row here that is a fact about THIS app rather than about the exercise, and the
    // reason the mock's invented "calories" row is not missed.
    [t("detail.analysis"), analysisState, false],
  ];

  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.details")}</h2>
      <div className="mt-2">
        {rows.map(([label, value, badge]) => (
          <div
            key={label}
            className="flex items-center justify-between border-b border-border-dark py-[11px] last:border-0"
          >
            <span className="text-[13px] text-muted">{label}</span>
            {badge ? (
              <span className="rounded-full bg-primary/10 px-2.5 py-0.5 text-[12px] font-semibold text-primary">
                {value}
              </span>
            ) : (
              <span className="text-[13px] font-medium text-content">{value}</span>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function DemoCard({ clip }: { clip: string }) {
  const { t } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.demo")}</h2>
      <p className="mt-0.5 text-[12.5px] text-muted">{t("detail.demoSub")}</p>
      {/* Capped rather than full-bleed: at the column's full width this is a 640px hero, and it is
          a reference clip beside the page's real actions, not the subject of the page. */}
      <video
        className="mt-3 aspect-video w-full max-w-[440px] rounded-xl bg-content/[0.04] object-cover"
        src={`/demo/${clip}.mp4`}
        poster={`/demo/${clip}.jpg`}
        controls
        preload="none"
        playsInline
      />
    </section>
  );
}

function RelatedCard({ related }: { related: readonly string[] }) {
  const { t } = useI18n();
  return (
    <section className={CARD}>
      <h2 className="text-[15.5px] font-bold text-content">{t("detail.related")}</h2>
      <ul className="mt-5 flex items-start justify-between gap-2">
        {related.map((m) => (
          <li key={m} className="flex-1">
            <Link to={`/movements/${encodeURIComponent(m)}`} className="group flex flex-col items-center">
              <span className="flex h-[84px] w-[84px] items-center justify-center overflow-hidden rounded-full bg-gradient-to-b from-[#f7f5ff] to-[#eceefb] transition-transform group-hover:-translate-y-0.5">
                <MovementArt movement={m} />
              </span>
              <span className="mt-2 text-center text-[12px] font-medium text-muted group-hover:text-primary">
                {movementLabel(t, m)}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ---------------------------------------------------------------------------------------------
// The other four tabs. They exist so the row is not decoration; the Overview above is where the
// design work went.
// ---------------------------------------------------------------------------------------------

function HowToTab({ movement, detail }: { movement: string; detail: Detail }) {
  const { t, lang } = useI18n();
  const [active, setActive] = useState(0);
  const step = detail.steps[active];

  return (
    <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <section className="rounded-[24px] border border-border-dark bg-surface p-6 shadow-card">
        <p className="text-[12px] font-semibold uppercase tracking-wide text-primary">
          {t("detail.stepOf", { n: active + 1, total: detail.steps.length })}
        </p>
        <p className="mt-3 text-[17px] font-semibold leading-relaxed text-content">
          {step.text[lang]}
        </p>
        {/* max-w-full on the figure because a wide one (Shoulder Bridge is ~2.6:1) comes to
            ~730px at this height, and an img sized by `h-full w-auto` will not shrink to fit its
            flex parent. It never binds on the sets drawn standing. */}
        <div className="mt-6 flex h-[280px] items-center justify-center rounded-[24px] bg-content/[0.03] p-4">
          {step.image ? (
            <img src={step.image} alt="" className="h-full w-auto max-w-full object-contain" />
          ) : (
            <span className="h-full w-[220px]">
              <MovementArt movement={movement} />
            </span>
          )}
        </div>
      </section>
      <section className="space-y-2">
        {detail.steps.map((s, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setActive(i)}
            className={`flex w-full items-start gap-3 rounded-2xl border p-4 text-left transition-colors ${
              active === i
                ? "border-primary bg-surface shadow-card"
                : "border-transparent bg-surface/70 hover:border-border-dark"
            }`}
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-[11px] font-bold text-primary-content">
              {i + 1}
            </span>
            <p className="text-[13.5px] leading-snug text-content">{s.text[lang]}</p>
          </button>
        ))}
      </section>
    </div>
  );
}

// Common mistakes. Built against the reference mock (~/Downloads/squat-mistakes-react): a numbered
// row per fault, a wrong/correct illustration pair, the fault stated and argued, and a boxed list
// of corrective cues.
//
// WHAT THE MOCK'S SHELL CONTRIBUTED: nothing. Its sidebar, breadcrumbs, "Watch tutorial" /
// "Start analysis" buttons and "Back to Squat" link all already exist on this page, one level up.
// Its body-region tab strip (All / Lower body / Upper body / Core / Full body) is dropped outright
// rather than ported: it filters a list of movements, and this page is ONE movement, so every tab
// but that movement's own would render empty. Only the card, and the tip strip under it, are new.
//
// WHERE THE LIST COMES FROM, and this is the change from the version this replaced: the faults are
// the ones this movement's rule detector can actually report, authored in lib/movementMistakes.ts
// and pinned to src/pose/movements/<movement>.py by tests/test_movement_mistakes_roster.py. The
// previous version listed every Fault node the knowledge graph defines for the movement, which for
// a flagship is dozens of concepts the analyzer will never mention -- a browsable index of the
// graph rather than an answer to "what will this app tell me I did wrong".
//
// The graph is still here, one hop deeper. Expanding a card fetches that fault's causes / risks /
// cues by its detector's own `kg_query`, so the concepts a card shows are exactly the ones a real
// detection of that fault would retrieve.
//
// WHAT IS NOT SHARED, and it is worth stating rather than discovering: the NAME. A card's heading
// is authored for a reader ("Knees caving in"), while the studio renders a detection through
// `faultLabel(t, d.fault_name)`, which for this fault comes out as its detector's own
// "Knees Inward / Knee Valgus". Same `fault_id`, same retrieval, two headings -- deliberate here,
// because the mock's copy is the friendlier one and the studio's is the clinical one, but if the
// two are ever to be unified it is the studio that should read from lib/movementMistakes.ts, not
// this page that should go back to reciting detector strings.

// Cause -> risk -> fix, the same three buckets and the same labels the studio's FaultCard uses,
// so a fault read about here and the same fault detected in a clip are described identically.
const CHAIN = [
  { cat: "causes", labelKey: "feedback.cause" },
  { cat: "risks", labelKey: "feedback.risk" },
  { cat: "corrections", labelKey: "feedback.cue" },
] as const;

/** One half of the wrong/correct pair.
 *
 *  The slot is always drawn, the image is optional, and neither half of that is arbitrary. The
 *  slot is always drawn so the pair reads as part of the card from day one and so dropping the
 *  art in later reflows nothing. The image is optional because it is the thing that carries the
 *  fault: with none, the panel does NOT fall back to the movement's generic card art, since the
 *  same drawing captioned "wrong" on the left and "correct" on the right is a picture that lies.
 *  It shows its verdict badge over a tinted ground and says which side it is, and that is all. */
function MistakePanel({ tone, src, alt }: { tone: "wrong" | "correct"; src?: string; alt: string }) {
  const { t } = useI18n();
  const wrong = tone === "wrong";
  return (
    <div
      className={`relative flex h-[152px] items-center justify-center overflow-hidden rounded-[14px] ${
        wrong ? "bg-danger/[0.06]" : "bg-secondary/[0.07]"
      }`}
    >
      <span
        className={`absolute left-2.5 top-2.5 z-10 flex h-6 w-6 items-center justify-center rounded-full text-white ${
          wrong ? "bg-danger" : "bg-secondary"
        }`}
      >
        {wrong ? <X size={13} weight="bold" /> : <Check size={13} weight="bold" />}
      </span>
      {src ? (
        <img src={src} alt={alt} loading="lazy" decoding="async" className="h-full w-full object-contain" />
      ) : (
        <span
          className={`px-3 text-center text-[11px] font-semibold ${
            wrong ? "text-danger/70" : "text-secondary/80"
          }`}
        >
          {t(wrong ? "detail.mistakeWrong" : "detail.mistakeCorrect")}
        </span>
      )}
    </div>
  );
}

function MistakeCard({
  mistake,
  index,
  open,
  onToggle,
  loaded,
}: {
  mistake: Mistake;
  index: number;
  open: boolean;
  onToggle: () => void;
  loaded?: Retrieval | "loading" | "error";
}) {
  const { t, lang } = useI18n();
  const title = mistake.title[lang];
  const retrieval = typeof loaded === "string" ? undefined : loaded;
  const panelId = `mistake-${mistake.id}`;

  // Three regions on a wide screen, in the mock's own order: the pair, the fault stated and
  // argued, the fixes.
  const columns = "xl:grid-cols-[296px_minmax(0,1fr)_minmax(0,336px)]";

  return (
    <article className="rounded-[22px] border border-border-dark bg-surface p-4 shadow-card sm:p-5">
      <div className="flex items-start gap-3.5 xl:items-center xl:gap-4">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-danger text-[14px] font-bold text-white ring-4 ring-danger/15">
          {index + 1}
        </span>

        <div className={`grid min-w-0 flex-1 items-center gap-4 ${columns}`}>
          {/* Capped below xl, where this region spans the whole card: at full width each panel
              becomes a 490x152 letterbox, and the figures going in here are standing bodies. The
              cap keeps both halves near the square the xl column already gives them. */}
          <div className="grid w-full max-w-[296px] grid-cols-2 gap-1.5 xl:max-w-none">
            <MistakePanel
              tone="wrong"
              src={mistake.art?.wrong}
              alt={t("detail.mistakeWrongAlt", { fault: title })}
            />
            <MistakePanel
              tone="correct"
              src={mistake.art?.correct}
              alt={t("detail.mistakeCorrectAlt", { fault: title })}
            />
          </div>

          <div className="min-w-0">
            <h3 className="text-[15.5px] font-bold leading-snug text-content">{title}</h3>
            <p className="mt-1 text-[12.5px] leading-relaxed text-muted">{mistake.subtitle[lang]}</p>
            <span className="mt-3 inline-block rounded-md bg-primary/10 px-2 py-1 text-[11px] font-semibold text-primary">
              {t("detail.mistakeWhy")}
            </span>
            <p className="mt-2 text-[12.5px] leading-relaxed text-muted">{mistake.why[lang]}</p>
          </div>

          <div className="rounded-[14px] border border-primary/15 bg-primary/[0.05] p-4">
            <h4 className="text-[12px] font-semibold text-primary">{t("detail.mistakeFix")}</h4>
            <ul className="mt-2.5 space-y-2">
              {mistake.fixes[lang].map((fix) => (
                <li key={fix} className="flex items-start gap-2 text-[12.2px] leading-snug text-muted">
                  <span className="mt-[3px] flex h-3 w-3 shrink-0 items-center justify-center rounded-full bg-secondary text-white">
                    <Check size={8} weight="bold" />
                  </span>
                  {fix}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* The mock's detail arrow, given the job the old list's row-button had: one graph
            traversal, on demand, for this fault's causes / risks / cues. */}
        <button
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={onToggle}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-faint transition-colors hover:bg-content/[0.04] hover:text-content"
        >
          <span className="sr-only">{t("detail.mistakeMore", { fault: title })}</span>
          <CaretRight size={18} className={`transition-transform ${open ? "rotate-90" : ""}`} />
        </button>
      </div>

      {open && (
        <div id={panelId} className="mt-4 border-t border-border-dark pt-4 text-[12.5px]">
          {loaded === "loading" ? (
            <p className="text-muted">{t("detail.mistakesLoading")}</p>
          ) : loaded === "error" ? (
            <p className="flex items-center gap-2 text-muted">
              <WarningCircle size={14} className="shrink-0 text-danger" />
              {t("detail.mistakesError")}
            </p>
          ) : (
            <div className="space-y-3">
              {CHAIN.map(({ cat, labelKey }) => {
                const items = summaryCategory(retrieval, cat);
                if (!items.length) return null;
                return (
                  <div key={cat}>
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-faint">
                      {t(labelKey)}
                    </p>
                    <ul className="mt-1 flex flex-wrap gap-1.5">
                      {items.map((x) => (
                        <li key={x} className="rounded-full bg-content/[0.05] px-2.5 py-1 text-muted">
                          {x}
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
              {!CHAIN.some(({ cat }) => summaryCategory(retrieval, cat).length) && (
                <p className="text-muted">{t("detail.mistakesNoLinks")}</p>
              )}
            </div>
          )}
        </div>
      )}

    </article>
  );
}

function MistakesTab({ movement }: { movement: string }) {
  const { t } = useI18n();
  const mistakes = movementMistakes(movement);
  const [open, setOpen] = useState<string | null>(null);
  const [detailByFault, setDetailByFault] = useState<
    Record<string, Retrieval | "loading" | "error">
  >({});

  // The list is local, so switching movements must not leave the previous one's expansion open.
  useEffect(() => {
    setOpen(null);
    setDetailByFault({});
  }, [movement]);

  const expand = useCallback(
    async (mistake: Mistake) => {
      const known = detailByFault[mistake.id];
      // A failed traversal keeps the card OPEN and retries in place. Toggling it shut on the click
      // that retries would land the result behind a closed card, so the reader would have to click
      // a third time to see whether the retry worked.
      const closing = open === mistake.id && known !== "error";
      setOpen(closing ? null : mistake.id);
      if (closing || (known && known !== "error")) return;
      setDetailByFault((prev) => ({ ...prev, [mistake.id]: "loading" }));
      try {
        // Queried by the detector's own kg_query, not by the card's title: the title is written
        // for a reader, the query has to be the string the analyzer would send.
        const context = await api.graph(mistake.kgQuery, movement);
        // summaryCategory reads a Retrieval, which is how the studio already derives causes /
        // risks / cues from exactly this payload. Wrapping the context rather than re-implementing
        // the walk keeps one derivation in the app (lib/retrieval.ts).
        setDetailByFault((prev) => ({
          ...prev,
          [mistake.id]: {
            fault_id: mistake.id,
            fault_name: mistake.kgQuery,
            query_text: mistake.kgQuery,
            retrieval_mode: "kg",
            context,
          },
        }));
      } catch {
        setDetailByFault((prev) => ({ ...prev, [mistake.id]: "error" }));
      }
    },
    [detailByFault, movement, open]
  );

  // Catalog movements with no registered detector (Jumping Jacks, High Knee) land here, and this
  // is the honest answer for them: nothing is authored because nothing can be detected.
  if (mistakes.length === 0) return <Empty text={t("detail.mistakesEmpty")} />;

  return (
    <div className="mt-5">
      <p className="max-w-2xl text-[13px] leading-relaxed text-muted">{t("detail.mistakesSub")}</p>

      {/* In the detector's own rule order, which is what the numbers count. Not sorted by anything
          derived: a stable number per fault is what lets someone say "mistake 3". */}
      <div className="mt-4 space-y-3">
        {mistakes.map((mistake, index) => (
          <MistakeCard
            key={mistake.id}
            mistake={mistake}
            index={index}
            open={open === mistake.id}
            onToggle={() => void expand(mistake)}
            loaded={detailByFault[mistake.id]}
          />
        ))}
      </div>

      <div className="mt-3.5 flex items-center gap-3 rounded-[18px] border border-primary/15 bg-primary/[0.05] px-4 py-3.5">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Lightbulb size={18} weight="duotone" />
        </span>
        <div className="min-w-0">
          <h4 className="text-[13.5px] font-bold text-content">{t("detail.mistakeTipTitle")}</h4>
          <p className="mt-0.5 text-[12px] leading-relaxed text-muted">{t("detail.mistakeTip")}</p>
        </div>
      </div>
    </div>
  );
}

function MusclesTab({ detail }: { detail: Detail }) {
  const { t } = useI18n();
  return (
    <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <section className="flex items-center justify-evenly rounded-[24px] border border-border-dark bg-surface p-6 shadow-card">
        <MuscleFigures detail={detail} size="h-[300px]" plateSize="h-[360px]" />
      </section>
      <section className="rounded-[24px] border border-border-dark bg-surface p-6 shadow-card">
        <h2 className="text-[17px] font-bold text-content">{t("detail.muscles")}</h2>
        <p className="mt-5 text-[12px] font-semibold uppercase tracking-wide text-faint">
          {t("detail.primary")}
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {detail.primary.map((m) => (
            <span
              key={m}
              className="rounded-full bg-primary px-3 py-1 text-[12px] font-semibold text-primary-content"
            >
              {t(`muscle.${m}`)}
            </span>
          ))}
        </div>
        <p className="mt-5 text-[12px] font-semibold uppercase tracking-wide text-faint">
          {t("detail.secondary")}
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {detail.secondary.map((m) => (
            <span
              key={m}
              className="rounded-full bg-primary/10 px-3 py-1 text-[12px] font-semibold text-primary"
            >
              {t(`muscle.${m}`)}
            </span>
          ))}
        </div>
        {/* Only for the drawn map. The illustrated plates ARE anatomical, so the caveat would be
            an apology for a picture that does not need one. */}
        {!detail.plate && (
          <p className="mt-6 text-[12.5px] leading-relaxed text-muted">{t("detail.mapNote")}</p>
        )}
      </section>
    </div>
  );
}

// The user's own analyses of this movement. `listAnalyses` has no server-side movement filter, so
// the page fetches a window and filters it client-side; rows saved before per-movement selection
// carry no `movement` and are counted as Squat, the same fallback History uses.
//
// The window is finite, so when the account holds more sessions than it covers the card SAYS so
// rather than presenting a silent subset as "your history of this movement".
const RECORDS_WINDOW = 100;

function RecordsTab({ movement, label }: { movement: string; label: string }) {
  const { t, lang } = useI18n();
  const { user } = useAuth();
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [total, setTotal] = useState(0);
  const [scanned, setScanned] = useState(0);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    setItems(null);
    setError(false);
    api
      .listAnalyses(RECORDS_WINDOW)
      .then((page) => {
        if (cancelled) return;
        setTotal(page.total);
        setScanned(page.items.length);
        setItems(page.items.filter((it) => (it.movement ?? "Squat") === movement));
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [user, movement]);

  if (!user) return <Empty text={t("detail.recordsSignIn")} />;
  if (error) return <Empty icon="warn" text={t("detail.recordsError")} />;
  if (items === null) return <Empty text={t("detail.mistakesLoading")} />;

  return (
    <div className="mt-5 overflow-hidden rounded-[24px] border border-border-dark bg-surface shadow-card">
      <div className="flex items-center justify-between gap-3 border-b border-border-dark px-5 py-4">
        <div>
          <h2 className="text-[16px] font-bold text-content">
            {t("detail.recordsTitle", { movement: label })}
          </h2>
          <p className="text-[12.5px] text-muted">{t("detail.recordsSub")}</p>
        </div>
        <Link to="/history" className="shrink-0 text-[12.5px] font-semibold text-primary">
          {t("detail.recordsAll")}
        </Link>
      </div>

      {total > scanned && (
        <p className="border-b border-border-dark px-5 py-2.5 text-[12px] text-faint">
          {t("detail.recordsPartial", { n: scanned, total })}
        </p>
      )}

      {items.length === 0 ? (
        <p className="px-5 py-8 text-center text-[13px] text-muted">{t("detail.recordsEmpty")}</p>
      ) : (
        items.map((it) => (
          <Link
            key={it.id}
            to={`/app?analysis=${encodeURIComponent(it.id)}`}
            className="flex items-center justify-between border-b border-border-dark px-5 py-4 last:border-0 hover:bg-content/[0.02]"
          >
            <span className="text-[13.5px] font-medium text-content">
              {new Date(it.created_at).toLocaleString(lang, {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            </span>
            <span
              className={`text-[13px] font-semibold ${
                it.fault_count === 0 ? "text-secondary" : "text-danger"
              }`}
            >
              {it.fault_count === 0
                ? t("detail.recordsClean")
                : t("detail.recordsFaults", { n: it.fault_count })}
            </span>
          </Link>
        ))
      )}
    </div>
  );
}

function Empty({ text, icon }: { text: string; icon?: "warn" }) {
  return (
    <div className="mt-5 flex items-center justify-center gap-2 rounded-[24px] border border-border-dark bg-surface px-6 py-12 text-center text-sm text-muted">
      {icon === "warn" && <WarningCircle size={16} className="shrink-0 text-danger" />}
      {text}
    </div>
  );
}
