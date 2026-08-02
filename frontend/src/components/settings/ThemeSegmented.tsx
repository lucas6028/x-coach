import { Desktop, Moon, Sun, type Icon } from "@phosphor-icons/react";
import { useTheme, type Theme } from "../../lib/theme";
import { useI18n } from "../../lib/i18n";

// Order matches the reference control: follow-the-system first, then the two explicit modes.
const OPTIONS: { value: Theme; Icon: Icon; labelKey: string }[] = [
  { value: "system", Icon: Desktop, labelKey: "theme.system" },
  { value: "light", Icon: Sun, labelKey: "theme.light" },
  { value: "dark", Icon: Moon, labelKey: "theme.dark" },
];

// Theme picker as an inline three-way segmented control — all options visible at once, which is
// how the reference popup does it. The navbar keeps its compact dropdown (components/ThemeToggle);
// this variant exists because a settings row has the width to show the whole choice.
export default function ThemeSegmented() {
  const { theme, setTheme } = useTheme();
  const { t } = useI18n();

  return (
    <div
      role="radiogroup"
      aria-label={t("settings.theme")}
      className="flex items-center gap-1 rounded-xl border border-border-dark bg-content/[0.03] p-1"
    >
      {OPTIONS.map(({ value, Icon: Ico, labelKey }) => {
        const active = theme === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={t(labelKey)}
            title={t(labelKey)}
            onClick={() => setTheme(value)}
            className={`flex h-8 w-9 items-center justify-center rounded-lg transition-colors ${
              active
                ? "bg-content/10 text-content"
                : "text-muted hover:bg-content/5 hover:text-content"
            }`}
          >
            <Ico size={18} />
          </button>
        );
      })}
    </div>
  );
}
