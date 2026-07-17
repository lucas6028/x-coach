import { useEffect, useMemo, useState } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { api, type MovementFault, type Retrieval, type RetrievalContext } from "../api";
import AppLayout from "../components/AppLayout";
import MovementSelector from "../components/MovementSelector";
import { GraphScene, Legend, collect, KIND_ORDER } from "../components/KnowledgeGraphWidget";
import { useI18n, faultLabel } from "../lib/i18n";

// Two-level browse of the movement knowledge graph: pick a movement -> its complete fault list
// loads as chips (via /api/knowledge/faults, which enumerates by the movement attribute so no
// fault is hidden) -> picking a fault renders its cause/correction/risk subgraph, reusing the
// exact GraphScene visual from the analysis widget.
export default function Explore() {
  const { t } = useI18n();
  const [movement, setMovement] = useState("Squat");
  const [faults, setFaults] = useState<MovementFault[]>([]);
  const [activeFault, setActiveFault] = useState<string | null>(null);
  const [context, setContext] = useState<RetrievalContext | null>(null);
  const [search, setSearch] = useState("");
  // Starts true so the first commit shows the skeleton, never a flash of the empty state.
  const [loadingFaults, setLoadingFaults] = useState(true);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Bumped by Retry to re-run the fetch that failed (the fault name can be unchanged, so a plain
  // re-select would be a no-op that never clears the skeleton).
  const [reloadNonce, setReloadNonce] = useState(0);

  // Switching movement resets the selection synchronously (same render), so the graph effect can
  // never fire a stale fault against the new movement scope.
  const pickMovement = (m: string) => {
    if (m === movement) return;
    setActiveFault(null);
    setContext(null);
    setMovement(m);
  };

  // Load the movement's complete fault list; auto-select the first fault that actually has a graph
  // (connectivity > 0) so the page never opens on an empty panel, falling back to the first fault.
  useEffect(() => {
    let active = true;
    setLoadingFaults(true);
    setError(null);
    api
      .movementFaults(movement)
      .then((res) => {
        if (!active) return;
        setFaults(res.faults);
        const pick = res.faults.find((f) => f.connectivity > 0) ?? res.faults[0];
        setActiveFault(pick ? pick.name : null);
      })
      .catch(() => {
        if (!active) return;
        setError(t("explore.error"));
        setFaults([]);
        setActiveFault(null);
      })
      .finally(() => {
        if (active) setLoadingFaults(false);
      });
    return () => {
      active = false;
    };
  }, [movement, reloadNonce, t]);

  // Pull the active fault's scoped subgraph. `active` guards a superseded request from writing
  // stale state; reloadNonce lets Retry re-run this even when the fault is unchanged.
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
  }, [activeFault, movement, reloadNonce, t]);

  // Reuse collect()/GraphScene from the analysis widget so the visual is identical.
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
  const visibleFaults = faults.filter((f) => faultLabel(t, f.name).toLowerCase().includes(needle));
  const activeVisible = activeFault != null && visibleFaults.some((f) => f.name === activeFault);

  const busy = loadingFaults || loadingGraph;
  const showChips = !loadingFaults && !error && visibleFaults.length > 0;
  const showGraph = !busy && !error && activeVisible && hasGraph;

  // The panel below the chips has one clear message per state (distinct empty vs. no-graph vs.
  // filtered-out), never the contradictory "no faults" over visible chips.
  let panelMessage: string | null = null;
  if (!busy && !error) {
    if (visibleFaults.length === 0) panelMessage = t("explore.empty");
    else if (!activeVisible) panelMessage = t("explore.pick");
    else if (!hasGraph) panelMessage = t("explore.noGraph");
  }

  return (
    <AppLayout title={t("explore.title")}>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto max-w-5xl px-4 py-8 lg:px-6 lg:py-12">
          <h1 className="font-display text-2xl font-bold text-content">{t("explore.title")}</h1>
          <p className="mt-1.5 text-sm text-muted">{t("explore.subtitle")}</p>

          {/* Controls: movement picker + client-side fault search. */}
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <MovementSelector value={movement} onChange={pickMovement} />
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
          {showChips && (
            <div className="mt-5 flex flex-wrap gap-2">
              {visibleFaults.map((f) => {
                const isActive = f.name === activeFault;
                return (
                  <button
                    key={f.name}
                    onClick={() => setActiveFault(f.name)}
                    className={`rounded-full px-3 py-1 text-sm transition-colors ${
                      isActive
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "border border-border-dark text-muted hover:bg-content/5 hover:text-content"
                    }`}
                  >
                    {faultLabel(t, f.name)}
                  </button>
                );
              })}
            </div>
          )}

          {/* Graph panel: skeleton while loading, error with retry, a state message, else the scene. */}
          <div className="relative mt-5 h-[520px] rounded-xl border border-border-dark bg-surface">
            {busy ? (
              <div className="h-full animate-pulse rounded-xl bg-content/5" />
            ) : error ? (
              <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center">
                <p className="text-sm text-danger">{error}</p>
                <button
                  onClick={() => setReloadNonce((n) => n + 1)}
                  className="rounded-lg border border-border-dark px-3 py-1.5 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
                >
                  {t("history.retry")}
                </button>
              </div>
            ) : showGraph ? (
              <GraphScene
                centerLabel={faultLabel(t, graph!.center)}
                neighbors={neighbors}
                large
                animateKey={movement + ":" + activeFault}
              />
            ) : (
              <div className="flex h-full items-center justify-center px-4 text-center">
                <p className="text-sm text-muted">{panelMessage}</p>
              </div>
            )}
          </div>

          {showGraph && (
            <div className="mt-3">
              <Legend kinds={presentKinds} t={t} />
            </div>
          )}
        </main>
      </div>
    </AppLayout>
  );
}
