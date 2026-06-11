import { useTheme, type Theme } from "../lib/theme";
import { useI18n } from "../lib/i18n";

const OPTIONS: { value: Theme; icon: string; labelKey: string }[] = [
  { value: "light", icon: "light_mode", labelKey: "theme.light" },
  { value: "system", icon: "computer", labelKey: "theme.system" },
  { value: "dark", icon: "dark_mode", labelKey: "theme.dark" },
];

interface Props {
  // Segmented control when expanded; a single cycling button in the slim rail.
  expanded: boolean;
}

export default function ThemeToggle({ expanded }: Props) {
  const { theme, setTheme } = useTheme();
  const { t } = useI18n();

  if (!expanded) {
    const order: Theme[] = ["light", "system", "dark"];
    const cur = OPTIONS.find((o) => o.value === theme)!;
    const curLabel = t(cur.labelKey);
    const next = order[(order.indexOf(theme) + 1) % order.length];
    return (
      <button
        onClick={() => setTheme(next)}
        title={t("theme.label", { name: curLabel })}
        aria-label={t("theme.aria", { name: curLabel })}
        className="w-10 h-10 mx-auto flex items-center justify-center rounded-lg text-muted hover:bg-content/5 hover:text-content transition-colors"
      >
        <span className="material-symbols-outlined">{cur.icon}</span>
      </button>
    );
  }

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-content/5 border border-border-dark">
      {OPTIONS.map((o) => (
        <button
          key={o.value}
          onClick={() => setTheme(o.value)}
          title={t(o.labelKey)}
          aria-label={t(o.labelKey)}
          aria-pressed={theme === o.value}
          className={`flex-1 flex items-center justify-center py-1.5 rounded-md transition-colors ${
            theme === o.value
              ? "bg-surface text-primary shadow-sm"
              : "text-muted hover:text-content"
          }`}
        >
          <span className="material-symbols-outlined text-lg">{o.icon}</span>
        </button>
      ))}
    </div>
  );
}
