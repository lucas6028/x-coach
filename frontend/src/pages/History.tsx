import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CaretRight,
  FilmSlate,
  PersonSimpleRun,
  Trash,
  VideoCamera,
  WarningCircle,
} from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import { api, type HistoryItem } from "../api";
import AppLayout from "../components/AppLayout";
import ConfirmDialog from "../components/ConfirmDialog";
import { useAuth } from "../lib/auth";
import { movementLabel, useI18n, viewLabel } from "../lib/i18n";

type Status = "loading" | "ready" | "error";

// "我的紀錄": the signed-in user's saved analyses. Each row replays into the studio via
// /app?analysis=<id>. Product UI — kept in the app's token system, with loading/empty/error states.
export default function History() {
  const { t, lang } = useI18n();
  const { user } = useAuth();

  const [items, setItems] = useState<HistoryItem[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState("");

  // Per-row deletion. Only one row can be in the confirm state at a time; `deleteError` is keyed by
  // row id so the message appears under the row it belongs to, not at the top of the page.
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null);

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

  // Splice the row out locally rather than refetching: `groups` is derived, so an emptied day
  // header disappears on its own and deleting the last row falls back to the empty state.
  const runDelete = async (id: string) => {
    setDeletingId(id);
    // Only clear this row's own error, if any -- a different row's still-unresolved failure
    // must not be silently forgotten just because this row's delete was confirmed.
    setDeleteError((prev) => (prev?.id === id ? null : prev));
    try {
      await api.deleteAnalysis(id);
      setItems((prev) => prev.filter((it) => it.id !== id));
      setPendingId(null);
    } catch (e) {
      setDeleteError({ id, message: e instanceof Error ? e.message : String(e) });
      setPendingId(null);
    } finally {
      setDeletingId(null);
    }
  };

  // The date now lives in the group header, so each row shows only its time.
  const fmtTime = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString(lang, { timeStyle: "short" });
  };

  // The record the confirm dialog is about. Derived rather than stored, so a row that disappears
  // (deleted, or a reload that no longer returns it) closes the dialog instead of stranding it.
  const pendingItem = items.find((it) => it.id === pendingId);
  const deletingPending = pendingItem !== undefined && deletingId === pendingItem.id;

  // What the dialog echoes back: the same title the row shows, plus its time.
  const rowLabel = (it: HistoryItem) =>
    `${t("history.rowTitle", {
      view: viewLabel(t, it.view_type ?? "unknown"),
      movement: movementLabel(t, it.movement ?? "Squat"),
    })} · ${fmtTime(it.created_at)}`;

  // Group the (newest-first) rows into day sections, preserving order — so the list reads as a
  // reverse-chronological timeline with a date header separating each day. Rows with an unparseable
  // timestamp fall into one trailing "unknown" group rather than being dropped.
  const groups = useMemo(() => {
    const out: { key: string; label: string; items: HistoryItem[] }[] = [];
    const index = new Map<string, number>();
    for (const it of items) {
      const d = new Date(it.created_at);
      const valid = !Number.isNaN(d.getTime());
      const key = valid ? d.toDateString() : "unknown";
      let i = index.get(key);
      if (i === undefined) {
        i = out.length;
        index.set(key, i);
        out.push({
          key,
          label: valid ? d.toLocaleDateString(lang, { dateStyle: "long" }) : it.created_at,
          items: [],
        });
      }
      out[i].items.push(it);
    }
    return out;
  }, [items, lang]);

  return (
    <AppLayout title={t("history.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-3xl px-4 py-8 lg:px-6 lg:py-12">
          <p className="text-sm text-muted">
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
            <WarningCircle size={18} className="shrink-0" />
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
              <FilmSlate size={30} weight="duotone" />
            </span>
            <div>
              <p className="font-medium text-content">{t("history.empty")}</p>
              <p className="mt-1 text-sm text-muted">{t("history.emptyHint")}</p>
            </div>
            <Link
              to="/app"
              className="mt-1 inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.99]"
            >
              <VideoCamera size={18} weight="fill" />
              {t("history.startCta")}
            </Link>
          </div>
        )}

        {status === "ready" && items.length > 0 && (
          <div className="mt-8 flex flex-col gap-6">
            {groups.map((g) => (
              <section key={g.key}>
                {/* Date separator: one header per day, above that day's rows. */}
                <h2 className="mb-2 flex items-center gap-3 text-xs font-medium uppercase tracking-wider text-muted">
                  <span>{g.label}</span>
                  <span className="h-px flex-1 bg-border-dark" />
                </h2>
                <ul className="flex flex-col gap-2">
                  {g.items.map((it) => {
                    const clean = it.fault_count === 0;
                    return (
                      <li key={it.id} className="group relative">
                        <Link
                          to={`/app?analysis=${it.id}`}
                          className="flex items-center gap-4 rounded-2xl border border-border-dark bg-surface-dark p-4 pr-14 transition-colors hover:border-primary/40 hover:bg-content/[0.03]"
                        >
                          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                            <PersonSimpleRun size={22} weight="duotone" />
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="truncate font-medium text-content">
                              {t("history.rowTitle", {
                                view: viewLabel(t, it.view_type ?? "unknown"),
                                movement: movementLabel(t, it.movement ?? "Squat"),
                              })}
                            </p>
                            <p className="mt-0.5 flex items-center gap-2 font-mono text-xs text-muted">
                              {fmtTime(it.created_at)}
                              <span className="rounded bg-content/5 px-1.5 py-0.5 text-[11px] font-medium text-muted">
                                {/* The promoted column, then Squat. `HistoryItem` carries no
                                    `result` -- list_analyses selects only the promoted columns, not
                                    the heavy document -- so there is no per-row echo to fall back
                                    to here. Rows predating the column are Squat by construction:
                                    every analysis before this change was pinned to it. */}
                                {movementLabel(t, it.movement ?? "Squat")}
                              </span>
                            </p>
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
                          <CaretRight
                            size={18}
                            className="shrink-0 text-muted transition-transform group-hover:translate-x-0.5"
                          />
                        </Link>

                        {/* Sibling of the Link, not a child: a <button> inside an <a> is invalid
                            HTML and its click bubbles into navigation. Hidden until the row is
                            hovered or the button is focused; always visible on touch. */}
                        <button
                          type="button"
                          aria-label={t("history.deleteAria")}
                          title={t("history.deleteCta")}
                          onClick={() => {
                            setPendingId(it.id);
                            // Only clear this row's own error, if any -- a different row's
                            // still-unresolved failure must not be silently forgotten just
                            // because the user opened another row's confirm.
                            setDeleteError((prev) => (prev?.id === it.id ? null : prev));
                          }}
                          className="absolute right-3 top-9 -translate-y-1/2 rounded-lg p-2 text-muted opacity-0 transition-opacity hover:bg-danger/10 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
                        >
                          <Trash size={16} weight="duotone" />
                        </button>

                        {deleteError?.id === it.id && (
                          <p className="mt-1.5 flex items-center gap-1.5 px-1 text-xs text-danger">
                            <WarningCircle size={14} weight="fill" className="shrink-0" />
                            {t("history.deleteError")}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))}
          </div>
        )}
        </main>
      </div>

      {/* One dialog for the whole page, not one per row: `pendingId` says which record it is
          about, and `detail` echoes that record back so the user can see they picked the right one. */}
      <ConfirmDialog
        open={pendingItem !== undefined}
        title={t("history.deleteTitle")}
        description={t("history.deleteDesc")}
        detail={pendingItem && rowLabel(pendingItem)}
        confirmLabel={deletingPending ? t("history.deleting") : t("history.deleteConfirm")}
        cancelLabel={t("history.deleteCancel")}
        busy={deletingPending}
        onConfirm={() => pendingItem && void runDelete(pendingItem.id)}
        onCancel={() => setPendingId(null)}
      />
    </AppLayout>
  );
}
