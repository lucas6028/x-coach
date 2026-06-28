import { useEffect, useRef, useState } from "react";
import { CaretDown, Check, Translate } from "@phosphor-icons/react";
import { LANGS, useI18n } from "../lib/i18n";

// Language picker as a dropdown menu. Trigger is a translate glyph; the menu
// lists each language by name with the current one checked.
export default function LanguageToggle() {
  const { lang, setLang, t } = useI18n();
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
        aria-label={`${t("lang.label")}: ${t(`lang.${lang}`)}`}
        title={`${t("lang.label")}: ${t(`lang.${lang}`)}`}
        className="flex h-10 items-center gap-1 rounded-lg px-2 text-muted transition-colors hover:bg-content/5 hover:text-content"
      >
        <Translate size={20} />
        <CaretDown size={12} weight="bold" className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-50 mt-1.5 min-w-[9rem] overflow-hidden rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20"
        >
          {LANGS.map((l) => {
            const active = lang === l.value;
            return (
              <button
                key={l.value}
                role="menuitemradio"
                aria-checked={active}
                onClick={() => {
                  setLang(l.value);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors ${
                  active ? "bg-primary/10 text-primary" : "text-muted hover:bg-content/5 hover:text-content"
                }`}
              >
                <span className="w-5 text-center text-xs font-semibold">{l.short}</span>
                <span className="flex-1 text-left">{t(`lang.${l.value}`)}</span>
                {active && <Check size={16} weight="bold" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
