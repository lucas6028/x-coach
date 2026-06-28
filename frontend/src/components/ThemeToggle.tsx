import { useEffect, useRef, useState } from "react";
import { CaretDown, Check, Desktop, Moon, Sun, type Icon } from "@phosphor-icons/react";
import { useTheme, type Theme } from "../lib/theme";
import { useI18n } from "../lib/i18n";

const OPTIONS: { value: Theme; Icon: Icon; labelKey: string }[] = [
  { value: "light", Icon: Sun, labelKey: "theme.light" },
  { value: "system", Icon: Desktop, labelKey: "theme.system" },
  { value: "dark", Icon: Moon, labelKey: "theme.dark" },
];

// Theme picker as a dropdown menu. Trigger shows the active mode's icon; the
// menu lists all three with the current one checked.
export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
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

  const cur = OPTIONS.find((o) => o.value === theme)!;
  const CurIcon = cur.Icon;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("theme.aria", { name: t(cur.labelKey) })}
        title={t("theme.label", { name: t(cur.labelKey) })}
        className="flex h-10 items-center gap-1 rounded-lg px-2 text-muted transition-colors hover:bg-content/5 hover:text-content"
      >
        <CurIcon size={20} />
        <CaretDown size={12} weight="bold" className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-50 mt-1.5 min-w-[9rem] overflow-hidden rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20"
        >
          {OPTIONS.map((o) => {
            const Ico = o.Icon;
            const active = theme === o.value;
            return (
              <button
                key={o.value}
                role="menuitemradio"
                aria-checked={active}
                onClick={() => {
                  setTheme(o.value);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                  active ? "bg-primary/10 text-primary" : "text-muted hover:bg-content/5 hover:text-content"
                }`}
              >
                <Ico size={18} />
                <span className="flex-1 text-left">{t(o.labelKey)}</span>
                {active && <Check size={16} weight="bold" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
