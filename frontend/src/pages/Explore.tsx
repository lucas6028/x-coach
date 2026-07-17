import { useCallback, useEffect, useMemo, useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { api, type Retrieval, type RetrievalContext } from "../api";
import AppLayout from "../components/AppLayout";
import MovementSelector from "../components/MovementSelector";
import { GraphScene, Legend, collect, KIND_ORDER } from "../components/KnowledgeGraphWidget";
import { useI18n, faultLabel } from "../lib/i18n";

// Two-level browse of the movement knowledge graph: pick a movement -> its faults load as chips ->
// picking a fault renders that fault's cause/correction/risk subgraph, reusing the exact GraphScene
// visual from the analysis widget. Every fetch goes through api.graph(query, movement).
export default function Explore() {
  const { t } = useI18n();
  const [movement, setMovement] = useState("Squat");
  const [faults, setFaults] = useState<string[]>([]);
  const [activeFault, setActiveFault] = useState<string | null>(null);
  const [context, setContext] = useState<RetrievalContext | null>(null);
  const [search, setSearch] = useState("");
  // Starts true so the very first commit shows the skeleton, never a flash of the empty state.
  const [loadingFaults, setLoadingFaults] = useState(true);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the movement's fault list. The movement is queried against itself so the backend returns the
  // movement-scoped subgraph; we keep only its Fault nodes. Also the retry target.
  const loadFaults = useCallback(async () => {
    setLoadingFaults(true);
    setError(null);
    try {
      const ctx = await api.graph(movement, movement);
      const seen = new Set<string>();
      const list = (ctx.subgraph?.nodes ?? [])
        .filter((n) => n.label === "Fault")
        .map((n) => n.name ?? n.node_id)
        .filter((name) => (seen.has(name) ? false : seen.add(name)));
      setFaults(list);
      const first = list[0] ?? null;
      setActiveFault(first);
      // Bridge straight into the graph fetch's loading state, so the commit between "faults ready"
      // and the graph effect running shows the skeleton, not a flash of chips + empty message. The
      // graph effect (which always runs when a fault is active) drives loadingGraph back to false.
      if (first) setLoadingGraph(true);
    } catch {
      setError(t("explore.error"));
      setFaults([]);
      setActiveFault(null);
    } finally {
      setLoadingFaults(false);
    }
  }, [movement, t]);

  useEffect(() => {
    void loadFaults();
  }, [loadFaults]);

  // Whenever the active fault changes, pull its scoped subgraph. `active` guards against a superseded
  // request (movement/fault switched again before this resolves) writing stale state.
  useEffect(() => {
    if (!activeFault) {
      setContext(null);
      return;
    }
    let active = true;
    setLoadingGraph(true);
    setError(null);
    api
      .graph(activeFault, movement)
      .then((ctx) => {
        if (active) setContext(ctx);
      })
      .catch(() => {
        if (active) setError(t("explore.error"));
      })
      .finally(() => {
        if (active) setLoadingGraph(false);
      });
    return () => {
      active = false;
    };
  }, [activeFault, movement, t]);

  // Reuse collect()/GraphScene from the analysis widget so the visual is byte-identical.
  const graph = useMemo(() => {
    if (!context || !activeFault) return null;
    const retrieval: Retrieval = {
      fault_id: "",
      fault_name: activeFault,
      query_text: activeFault,
      retrieval_mode: "graph",
      context,
    };
    return collect(retrieval);
  }, [context, activeFault]);

  const neighbors = graph?.neighbors ?? [];
  const hasGraph = !!graph && neighbors.length > 0;
  const presentKinds = KIND_ORDER.filter((k) => neighbors.some((n) => n.kind === k));

  const needle = search.trim().toLowerCase();
  const visibleFaults = faults.filter((f) => faultLabel(t, f).toLowerCase().includes(needle));

  return (
    <AppLayout title={t("explore.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-5xl px-4 py-8 lg:px-6 lg:py-12">
          <h1 className="font-display text-2xl font-bold text-content">{t("explore.title")}</h1>
          <p className="mt-1.5 text-sm text-muted">{t("explore.subtitle")}</p>

          {/* Controls: movement picker + client-side fault search. */}
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <MovementSelector value={movement} onChange={setMovement} />
            <div className="relative">
              <MagnifyingGlass
                size={18}
                weight="duotone"
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint"
              />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label={t("explore.search")}
                placeholder={t("explore.search")}
                className="h-10 rounded-lg border border-border-dark bg-surface pl-9 pr-3 text-sm text-content placeholder:text-faint focus:border-primary/40 focus:outline-none"
              />
            </div>
          </div>

          {/* Fault chips for the selected movement. */}
          {!loadingFaults && !error && visibleFaults.length > 0 && (
            <div className="mt-5 flex flex-wrap gap-2">
              {visibleFaults.map((f) => {
                const isActive = f === activeFault;
                return (
                  <button
                    key={f}
                    onClick={() => setActiveFault(f)}
                    className={`rounded-full px-3 py-1 text-sm transition-colors ${
                      isActive
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "border border-border-dark text-muted hover:bg-content/5 hover:text-content"
                    }`}
                  >
                    {faultLabel(t, f)}
                  </button>
                );
              })}
            </div>
          )}

          {/* Graph panel: skeleton while loading, error with retry, empty message, else the scene. */}
          <div className="relative mt-5 h-[520px] rounded-xl border border-border-dark bg-surface">
            {loadingFaults || loadingGraph ? (
              <div className="h-full animate-pulse rounded-xl bg-content/5" />
            ) : error ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center">
                <p className="text-sm text-danger">{error}</p>
                <button
                  onClick={() => void loadFaults()}
                  className="rounded-lg border border-border-dark px-3 py-1.5 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
                >
                  {t("history.retry")}
                </button>
              </div>
            ) : visibleFaults.length === 0 || !hasGraph ? (
              <div className="flex h-full items-center justify-center px-4 text-center">
                <p className="text-sm text-muted">{t("explore.empty")}</p>
              </div>
            ) : (
              <GraphScene
                centerLabel={faultLabel(t, graph!.center)}
                neighbors={neighbors}
                large
                animateKey={movement + ":" + activeFault}
              />
            )}
          </div>

          {!loadingFaults && !loadingGraph && !error && hasGraph && (
            <div className="mt-3">
              <Legend kinds={presentKinds} t={t} />
            </div>
          )}
        </main>
      </div>
    </AppLayout>
  );
}
