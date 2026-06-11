import { useTheme, type Theme } from "../lib/theme";

const OPTIONS: { value: Theme; icon: string; label: string }[] = [
  { value: "light", icon: "light_mode", label: "Light" },
  { value: "system", icon: "computer", label: "System" },
  { value: "dark", icon: "dark_mode", label: "Dark" },
];

interface Props {
  // Segmented control when expanded; a single cycling button in the slim rail.
  expanded: boolean;
}

export default function ThemeToggle({ expanded }: Props) {
  const { theme, setTheme } = useTheme();

  if (!expanded) {
    const order: Theme[] = ["light", "system", "dark"];
    const cur = OPTIONS.find((o) => o.value === theme)!;
    const next = order[(order.indexOf(theme) + 1) % order.length];
    return (
      <button
        onClick={() => setTheme(next)}
        title={`Theme: ${cur.label} (click to change)`}
        aria-label={`Theme: ${cur.label}`}
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
          title={o.label}
          aria-label={o.label}
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
