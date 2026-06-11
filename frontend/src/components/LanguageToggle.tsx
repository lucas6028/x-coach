import { LANGS, useI18n, type Lang } from "../lib/i18n";

interface Props {
  // Segmented control when expanded; a single cycling button in the slim rail.
  expanded: boolean;
}

export default function LanguageToggle({ expanded }: Props) {
  const { lang, setLang, t } = useI18n();

  if (!expanded) {
    const order = LANGS.map((l) => l.value);
    const cur = LANGS.find((l) => l.value === lang)!;
    const next: Lang = order[(order.indexOf(lang) + 1) % order.length];
    return (
      <button
        onClick={() => setLang(next)}
        title={`${t("lang.label")}: ${t(`lang.${lang}`)}`}
        aria-label={`${t("lang.label")}: ${t(`lang.${lang}`)}`}
        className="w-10 h-10 mx-auto flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors text-sm font-semibold"
      >
        {cur.short}
      </button>
    );
  }

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-content/5 border border-border-dark">
      {LANGS.map((l) => (
        <button
          key={l.value}
          onClick={() => setLang(l.value)}
          title={t(`lang.${l.value}`)}
          aria-label={t(`lang.${l.value}`)}
          aria-pressed={lang === l.value}
          className={`flex-1 flex items-center justify-center py-1.5 rounded-md text-xs font-semibold transition-colors ${
            lang === l.value
              ? "bg-surface text-primary shadow-sm"
              : "text-muted hover:text-content"
          }`}
        >
          {l.short}
        </button>
      ))}
    </div>
  );
}
