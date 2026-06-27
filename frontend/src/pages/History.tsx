import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type HistoryItem } from "../api";
import { useAuth } from "../lib/auth";
import { useI18n, viewLabel } from "../lib/i18n";

type Status = "loading" | "ready" | "error";

// "我的紀錄": the signed-in user's saved analyses. Each row replays into the studio via
// /app?analysis=<id>. Product UI — kept in the app's token system, with loading/empty/error states.
export default function History() {
  const { t, lang } = useI18n();
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setStatus("loading");
    setError("");
    try {
      const page = await api.listAnalyses();
      setItems(page.items);
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime())
      ? iso
      : d.toLocaleString(lang, { dateStyle: "medium", timeStyle: "short" });
  };

  return (
    <div className="min-h-[100dvh] bg-background-dark text-content">
      <header className="sticky top-0 z-10 border-b border-border-dark bg-background-dark/95 px-4 backdrop-blur lg:px-6">
        <div className="mx-auto flex h-16 max-w-3xl items-center justify-between gap-3">
          <Link to="/app" className="flex items-center gap-2.5">
            <img src="/icon.svg" alt="" className="h-8 w-8 rounded" />
            <span className="font-display font-bold tracking-tight">X-Coach</span>
          </Link>
          <div className="flex items-center gap-2">
            <Link
              to="/app"
              className="flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-muted transition-colors hover:bg-content/5 hover:text-content"
            >
              <span className="material-symbols-outlined text-lg">add</span>
              <span className="hidden sm:inline">{t("history.newAnalysis")}</span>
            </Link>
            <button
              onClick={async () => {
                await signOut();
                navigate("/");
              }}
              className="flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-muted transition-colors hover:bg-content/5 hover:text-content"
            >
              <span className="material-symbols-outlined text-lg">logout</span>
              <span className="hidden sm:inline">{t("account.signout")}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-8 lg:px-6 lg:py-12">
        <h1 className="font-display text-2xl font-bold tracking-tight">{t("history.title")}</h1>
        <p className="mt-1.5 text-sm text-muted">
          {user?.email ? t("history.subtitle", { email: user.email }) : t("history.subtitleAnon")}
        </p>

        {status === "loading" && (
          <ul className="mt-8 flex flex-col gap-2" aria-hidden="true">
            {[0, 1, 2, 3].map((i) => (
              <li
                key={i}
                className="flex items-center gap-4 rounded-2xl border border-border-dark bg-surface-dark p-4"
              >
                <span className="h-10 w-10 shrink-0 animate-pulse rounded-lg bg-content/10" />
                <span className="flex-1">
                  <span className="block h-3.5 w-32 animate-pulse rounded bg-content/10" />
                  <span className="mt-2 block h-3 w-24 animate-pulse rounded bg-content/5" />
                </span>
              </li>
            ))}
          </ul>
        )}

        {status === "error" && (
          <div className="mt-8 flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
            <span className="material-symbols-outlined text-lg leading-none">error</span>
            <div className="min-w-0 flex-1">
              <p className="font-medium">{t("history.errorTitle")}</p>
              <p className="mt-0.5 break-words text-danger/80">{error}</p>
            </div>
            <button
              onClick={() => void load()}
              className="shrink-0 rounded-lg border border-danger/30 px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10"
            >
              {t("history.retry")}
            </button>
          </div>
        )}

        {status === "ready" && items.length === 0 && (
          <div className="mt-8 flex flex-col items-center gap-4 rounded-2xl border border-dashed border-border-dark bg-content/[0.02] px-6 py-16 text-center">
            <span className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary">
              <span className="material-symbols-outlined text-3xl">video_library</span>
            </span>
            <div>
              <p className="font-medium text-content">{t("history.empty")}</p>
              <p className="mt-1 text-sm text-muted">{t("history.emptyHint")}</p>
            </div>
            <Link
              to="/app"
              className="mt-1 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
            >
              <span className="material-symbols-outlined text-lg">videocam</span>
              {t("history.startCta")}
            </Link>
          </div>
        )}

        {status === "ready" && items.length > 0 && (
          <ul className="mt-8 flex flex-col gap-2">
            {items.map((it) => {
              const clean = it.fault_count === 0;
              return (
                <li key={it.id}>
                  <Link
                    to={`/app?analysis=${it.id}`}
                    className="group flex items-center gap-4 rounded-2xl border border-border-dark bg-surface-dark p-4 transition-colors hover:border-primary/40 hover:bg-content/[0.03]"
                  >
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <span className="material-symbols-outlined">directions_run</span>
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium text-content">
                        {t("history.rowTitle", { view: viewLabel(t, it.view_type ?? "unknown") })}
                      </p>
                      <p className="mt-0.5 font-mono text-xs text-muted">{fmtDate(it.created_at)}</p>
                    </div>
                    <span
                      className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${
                        clean
                          ? "bg-secondary/15 text-secondary"
                          : "bg-[rgb(var(--c-fault))]/15 text-[rgb(var(--c-fault))]"
                      }`}
                    >
                      {clean
                        ? t("history.clean")
                        : it.fault_count === 1
                          ? t("history.faultOne")
                          : t("history.faultMany", { count: it.fault_count })}
                    </span>
                    <span className="material-symbols-outlined shrink-0 text-muted transition-transform group-hover:translate-x-0.5">
                      chevron_right
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </main>
    </div>
  );
}
