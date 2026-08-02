import { useEffect, useMemo, useRef, useState } from "react";
import {
  GearSix,
  MagnifyingGlass,
  Sparkle,
  UserCircle,
  X,
  type Icon,
} from "@phosphor-icons/react";
import { useAuth } from "../../lib/auth";
import { useI18n } from "../../lib/i18n";
import GeneralPane from "./GeneralPane";
import ModelPane from "./ModelPane";
import AccountPane from "./AccountPane";

type SectionId = "general" | "account" | "model";

// Two groups, as in the reference rail: the account's own settings, then what the user customises
// about the app. Every x-coach setting fits one of the two — we don't carry Claude's Billing/Usage
// categories, and an empty pane reads worse than a shorter rail.
const GROUPS: { labelKey: string; items: { id: SectionId; labelKey: string; Icon: Icon }[] }[] = [
  {
    labelKey: "settings.groupSettings",
    items: [
      { id: "general", labelKey: "settings.general", Icon: GearSix },
      { id: "account", labelKey: "settings.account", Icon: UserCircle },
    ],
  },
  {
    labelKey: "settings.groupCustomize",
    items: [{ id: "model", labelKey: "settings.model", Icon: Sparkle }],
  },
];

// The settings popup: a searchable category rail on the left, one scrollable pane on the right.
//
// Follows ConfirmDialog's overlay idiom (`fixed inset-0 z-50`, no portal) rather than introducing
// a portal layer. Only the selected pane is mounted, so each pane owns its own state and side
// effects (ModelPane's /api/health fetch, AccountPane's clear flow) without the shell knowing.
//
// The shell is presentational: it never navigates. Whoever renders it decides what closing means —
// the /settings route goes back, the account menu just unmounts it.
export default function SettingsDialog({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const { user } = useAuth();
  const [section, setSection] = useState<SectionId>("general");
  const [query, setQuery] = useState("");
  const cardRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // Escape closes. Bound only while mounted, so nothing swallows the key once the popup is gone.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Focus moves into the dialog on open and goes back to whatever opened it on close, so keyboard
  // users aren't dumped at the top of the page behind the overlay.
  useEffect(() => {
    restoreRef.current = document.activeElement as HTMLElement | null;
    cardRef.current?.focus();
    return () => restoreRef.current?.focus();
  }, []);

  // Search filters the rail by category name. Groups left with no match disappear entirely rather
  // than lingering as a bare heading. The selected pane is deliberately NOT reset — narrowing the
  // rail shouldn't yank the user out of the settings they are reading.
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return GROUPS;
    return GROUPS.map((g) => ({
      ...g,
      items: g.items.filter((i) => t(i.labelKey).toLowerCase().includes(q)),
    })).filter((g) => g.items.length > 0);
  }, [query, t]);

  if (!user) return null;

  return (
    <div
      // mousedown, not click: a drag that starts inside the card and releases on the backdrop
      // would otherwise read as a backdrop click and dismiss the popup.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 sm:p-4"
    >
      <div
        ref={cardRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        // Named directly rather than via a visually-hidden heading: the rail already carries a
        // "Settings" group label, and a third one would just repeat itself to a screen reader.
        aria-label={t("settings.title")}
        // Full-bleed on phones, a centred card from `sm` up — roughly the reference's 3:2 shape.
        className="flex h-full w-full flex-col overflow-hidden bg-surface-dark outline-none sm:h-[min(42rem,88vh)] sm:max-w-5xl sm:flex-row sm:rounded-2xl sm:border sm:border-border-dark sm:shadow-2xl sm:shadow-black/40"
      >
        {/* Category rail. Vertical from `sm` up; a horizontally scrollable tab strip on phones. */}
        <nav
          aria-label={t("settings.nav")}
          // ~21% of the card, matching the reference's rail-to-dialog ratio rather than its
          // absolute width — our card is narrower, so the same pixel width reads as too heavy.
          className="flex shrink-0 flex-col gap-3 border-b border-border-dark p-3 sm:w-[13.5rem] sm:border-b-0 sm:border-r"
        >
          <div className="relative shrink-0">
            <MagnifyingGlass
              size={18}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint"
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t("settings.search")}
              aria-label={t("settings.search")}
              // `type="search"` for the searchbox role, minus Chrome's blue clear glyph — the
              // reference field has no such decoration.
              className="h-10 w-full rounded-xl border border-border-dark bg-content/[0.03] pl-10 pr-3 text-sm text-content outline-none transition-colors placeholder:text-faint focus:border-primary/40 [&::-webkit-search-cancel-button]:appearance-none"
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            {groups.length === 0 ? (
              <p className="px-2 py-3 text-sm text-faint">{t("settings.searchEmpty")}</p>
            ) : (
              groups.map((group) => (
                <div key={group.labelKey} className="mb-2 last:mb-0">
                  <h3 className="hidden px-2 pb-1 pt-2 text-sm text-faint sm:block">
                    {t(group.labelKey)}
                  </h3>
                  <ul className="flex gap-1 overflow-x-auto sm:block sm:space-y-0.5 sm:overflow-x-visible">
                    {group.items.map(({ id, labelKey, Icon: Ico }) => {
                      const active = section === id;
                      return (
                        <li key={id}>
                          <button
                            type="button"
                            aria-current={active ? "true" : undefined}
                            onClick={() => setSection(id)}
                            className={`flex w-full items-center gap-3 whitespace-nowrap rounded-lg px-2.5 py-2 text-[15px] transition-colors ${
                              active
                                ? "bg-content/[0.08] text-content"
                                : "text-muted hover:bg-content/5 hover:text-content"
                            }`}
                          >
                            <Ico size={20} className="shrink-0" />
                            <span className="flex-1 truncate text-left">{t(labelKey)}</span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))
            )}
          </div>
        </nav>

        {/* Pane */}
        <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
          <button
            type="button"
            onClick={onClose}
            aria-label={t("settings.close")}
            className="absolute right-4 top-4 z-10 rounded-lg p-2 text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <X size={20} />
          </button>
          {/* Rows run the full width of the pane; the close button sits ABOVE them rather than
              beside them, which is why the top padding is so much larger than the rest. */}
          <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-10 pt-16 sm:px-8 sm:pb-12 sm:pt-20">
            {section === "general" && <GeneralPane user={user} />}
            {section === "account" && <AccountPane />}
            {section === "model" && <ModelPane />}
          </div>
        </div>
      </div>
    </div>
  );
}
