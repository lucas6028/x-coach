import { ArrowCounterClockwise, MagnifyingGlass } from "@phosphor-icons/react";
import MenuCard from "../MenuCard";
import { movementLabel, useI18n } from "../../lib/i18n";

export type ResultFilter = "all" | "clean" | "faults";
export type RangeFilter = "all" | "today" | "7d" | "30d";

export interface HistoryFilterState {
  search: string;
  movement: string;
  result: ResultFilter;
  range: RangeFilter;
}

export const EMPTY_FILTERS: HistoryFilterState = {
  search: "",
  movement: "all",
  result: "all",
  range: "all",
};

export function filtersActive(f: HistoryFilterState): boolean {
  return f.search !== "" || f.movement !== "all" || f.result !== "all" || f.range !== "all";
}

// How far back each preset reaches, in days. `today` is handled separately (local midnight, not a
// rolling 24 hours) so it agrees with the "Today" day separator in the grid.
const RANGE_DAYS: Record<Exclude<RangeFilter, "all" | "today">, number> = { "7d": 7, "30d": 30 };

/** The cutoff a range preset implies, or null for "all time". */
export function rangeStart(range: RangeFilter, now = new Date()): Date | null {
  if (range === "all") return null;
  const d = new Date(now);
  d.setHours(0, 0, 0, 0);
  if (range === "today") return d;
  d.setDate(d.getDate() - (RANGE_DAYS[range] - 1));
  return d;
}

interface Props {
  value: HistoryFilterState;
  onChange: (next: HistoryFilterState) => void;
  /** Movement keys present in the loaded rows — the type dropdown only offers what exists. */
  movements: string[];
}

// The filter row, built from the same MenuCard the studio header uses for movement and extraction
// tier. Deliberately not a native <select>: the studio abandoned those because the browser draws
// their list itself, dropping a system-white menu on top of the glass, and a filter row on the same
// canvas would have reintroduced exactly that. The reference's date-RANGE picker becomes four
// presets — the mock's button opens nothing, and a real range picker is a component this app does
// not have.
//
// No leading icons here, unlike the studio's two: caption + value already name each control, and
// three tinted circles in a row read as status badges rather than as filters.
export default function HistoryFilters({ value, onChange, movements }: Props) {
  const { t } = useI18n();
  const set = (patch: Partial<HistoryFilterState>) => onChange({ ...value, ...patch });

  // Each list leads with its "no filter" entry, so clearing one control is a choice inside that
  // control rather than something only the row-level reset can do.
  const movementOptions = [
    { value: "all", label: t("history.filterAllMovements") },
    ...movements.map((m) => ({ value: m, label: movementLabel(t, m) })),
  ];
  const resultOptions = [
    { value: "all", label: t("history.filterAllResults") },
    { value: "clean", label: t("history.filterClean") },
    { value: "faults", label: t("history.filterFaults") },
  ];
  const rangeOptions = [
    { value: "all", label: t("history.rangeAll") },
    { value: "today", label: t("history.rangeToday") },
    { value: "7d", label: t("history.range7") },
    { value: "30d", label: t("history.range30") },
  ];
  const shown = (opts: { value: string; label: string }[], v: string) =>
    opts.find((o) => o.value === v)?.label ?? v;

  return (
    <div className="flex flex-wrap items-center gap-2 sm:gap-3">
      {/* Matched to the MenuCard's height and radius so the row reads as one set of controls. */}
      <div className="relative min-w-[190px] max-w-[260px] flex-1">
        <MagnifyingGlass
          size={15}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[#63709f]"
        />
        <input
          type="search"
          aria-label={t("history.searchPlaceholder")}
          placeholder={t("history.searchPlaceholder")}
          value={value.search}
          onChange={(e) => set({ search: e.target.value })}
          className="glass-control h-[52px] w-full rounded-2xl pl-10 pr-4 text-[13px] font-medium text-[#1e2142] outline-none transition-colors placeholder:font-normal placeholder:text-[#8b93bb] hover:border-[#c9bcff] focus:border-[#c9bcff] [&::-webkit-search-cancel-button]:appearance-none"
        />
      </div>

      <MenuCard
        label={t("history.filterMovement")}
        value={value.movement}
        display={shown(movementOptions, value.movement)}
        options={movementOptions}
        onChange={(v) => set({ movement: v })}
        align="left"
      />

      <MenuCard
        label={t("history.filterStatus")}
        value={value.result}
        display={shown(resultOptions, value.result)}
        options={resultOptions}
        onChange={(v) => set({ result: v as ResultFilter })}
        align="left"
      />

      <MenuCard
        label={t("history.filterRange")}
        value={value.range}
        display={shown(rangeOptions, value.range)}
        options={rangeOptions}
        onChange={(v) => set({ range: v as RangeFilter })}
        align="left"
      />

      {/* Only offered once something is actually filtered — a permanent reset on an unfiltered
          list is a control that can only do nothing. */}
      {filtersActive(value) && (
        <button
          type="button"
          onClick={() => onChange(EMPTY_FILTERS)}
          className="glass-control flex h-[52px] items-center gap-2 rounded-2xl px-4 text-[13px] font-semibold text-[#59648f] transition-colors hover:border-[#c9bcff] hover:text-[#1e2142]"
        >
          <ArrowCounterClockwise size={14} weight="bold" />
          {t("history.clearFilters")}
        </button>
      )}
    </div>
  );
}
