import { useEffect, useRef, useState } from "react";
import { CaretDown, Check } from "@phosphor-icons/react";
import { useI18n, movementLabel } from "../lib/i18n";
import { FLAGSHIP_MOVEMENTS, GENERAL_MOVEMENTS } from "../lib/movements";

interface Props {
  value: string;
  onChange: (m: string) => void;
}

const GROUPS: { key: string; items: readonly string[] }[] = [
  { key: "explore.groupFlagship", items: FLAGSHIP_MOVEMENTS },
  { key: "explore.groupGeneral", items: GENERAL_MOVEMENTS },
];

// Grouped movement picker. Mirrors LanguageToggle's dropdown mechanics (outside-click + Escape close,
// menu semantics), but the trigger is a real labelled control showing the selected movement's name.
export default function MovementSelector({ value, onChange }: Props) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex h-10 items-center gap-2 rounded-lg border border-border-dark bg-surface px-3 text-sm font-medium text-content transition-colors hover:bg-content/5"
      >
        <span className="truncate">{movementLabel(t, value)}</span>
        <CaretDown size={12} weight="bold" className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute left-0 z-50 mt-1.5 max-h-[60vh] min-w-[13rem] overflow-y-auto rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20"
        >
          {GROUPS.map((group) => (
            <div key={group.key}>
              <p className="px-2.5 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider text-faint">
                {t(group.key)}
              </p>
              {group.items.map((m) => {
                const active = value === m;
                return (
                  <button
                    key={m}
                    role="menuitemradio"
                    aria-checked={active}
                    onClick={() => {
                      onChange(m);
                      setOpen(false);
                    }}
                    className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                      active ? "bg-primary/10 text-primary" : "text-muted hover:bg-content/5 hover:text-content"
                    }`}
                  >
                    <span className="flex-1 text-left">{movementLabel(t, m)}</span>
                    {active && <Check size={16} weight="bold" />}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
