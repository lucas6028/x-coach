import type { Analysis } from "../api";
import { useI18n, viewLabel } from "../lib/i18n";

interface Props {
  analysis: Analysis | null;
  loading: boolean;
}

export default function Header({ analysis, loading }: Props) {
  const { t } = useI18n();
  return (
    <header className="h-16 shrink-0 border-b border-border-dark bg-background-dark/95 backdrop-blur flex items-center justify-between px-4 lg:px-6">
      <div className="flex flex-col min-w-0">
        <h1 className="text-content text-base lg:text-lg font-bold tracking-tight truncate">
          {analysis ? t("header.session", { id: analysis.video_id }) : t("header.title")}
        </h1>
        <div className="flex items-center gap-2 text-[11px] text-muted font-mono">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              loading ? "bg-yellow-400 animate-pulse" : analysis ? "bg-green-500" : "bg-faint"
            }`}
          />
          {loading ? t("header.processing") : analysis ? t("header.complete") : t("header.awaiting")}
          {analysis && (
            <>
              <span className="text-faint">|</span>
              <span>{t("header.view", { type: viewLabel(t, analysis.view.view_type) })}</span>
              <span className="text-faint hidden sm:inline">|</span>
              <span className="hidden sm:inline uppercase">{analysis.source}</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
