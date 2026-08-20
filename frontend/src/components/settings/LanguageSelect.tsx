import { useEffect, useRef, useState } from "react";
import { CaretDown, Check } from "@phosphor-icons/react";
import { LANGS, useI18n } from "../../lib/i18n";

// Language picker styled as the reference popup's borderless "value + caret" select, rather than
// the navbar's icon-only trigger (components/LanguageToggle) — in a settings row there is room to
// show the current value as text.
export default function LanguageSelect() {
  const { lang, setLang, t } = useI18n();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    // Escape dismisses the menu ONLY. SettingsDialog listens for Escape on window, and this
    // listener is on document — one hop earlier in the bubble path — so stopping propagation here
    // keeps the first Escape from tearing down the whole popup out from under an open menu.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      setOpen(false);
    };
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
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${t("lang.label")}: ${t(`lang.${lang}`)}`}
        className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[15px] text-muted transition-colors hover:bg-content/5 hover:text-content"
      >
        {t(`lang.${lang}`)}
        <CaretDown size={14} weight="bold" className={open ? "rotate-180" : undefined} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-10 mt-1.5 min-w-[9rem] overflow-hidden rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20"
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
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted hover:bg-content/5 hover:text-content"
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
