import { useCallback, useEffect, useMemo, useState } from "react";
import { MagnifyingGlass, WarningCircle } from "@phosphor-icons/react";
import AppLayout from "../components/AppLayout";
import { api, type FullGraphEdge, type FullGraphNode, type FullGraphResponse } from "../api";
import { useI18n, type TFunc } from "../lib/i18n";

// One page to browse the WHOLE knowledge graph (2000+ nodes, 3000+ edges) — filterable, with
// counts, table views, and a graph view of the filtered subset. Everything the per-fault 1-hop
// widgets (KnowledgeGraphWidget) never show: the complete node/edge index behind GET
// /api/knowledge/full. The whole graph is fetched ONCE, unfiltered; every filter below (movement,
// label, relation, search) is applied client-side so toggling one is instant.

type Status = "loading" | "ready" | "error";
type Tab = "nodes" | "edges" | "graph";

const PAGE_SIZE = 100;
const MAX_GRAPH_NODES = 300;
const GRAPH_WIDTH = 760;
const GRAPH_HEIGHT = 480;

const TABS: Tab[] = ["nodes", "edges", "graph"];
const TAB_KEY: Record<Tab, string> = {
  nodes: "kg.tabNodes",
  edges: "kg.tabEdges",
  graph: "kg.tabGraph",
};

// One colour per KG label — covers all 8 schema labels. `cause`/`risk`/`cue`/`evidence` reuse the
// exact hexes KnowledgeGraphWidget uses for the same concepts so the two views read consistently;
// the other four labels (never shown there) get their own.
const LABEL_COLOR: Record<string, string> = {
  Fault: "#f5b945",
  EvidenceSignal: "#94a3b8",
  Cause: "#0f758a",
  Risk: "#ef4444",
  Cue: "#42d159",
  Phase: "#8b5cf6",
  Action: "#f97316",
  QualityDimension: "#3b82f6",
};
const DEFAULT_LABEL_COLOR = "#64748b";
const labelColor = (label: string) => LABEL_COLOR[label] ?? DEFAULT_LABEL_COLOR;

interface Point {
  x: number;
  y: number;
}

/**
 * A small, deterministic force-directed layout: repulsion between every pair, a spring along each
 * edge, and a gentle pull to center. Positions are seeded from node INDEX (not Math.random()), so
 * the same filtered subset always draws in the same place — required for the layout to be pure and
 * for DOM snapshots/tests to be stable. Runs synchronously; callers cap input to MAX_GRAPH_NODES so
 * the O(n^2) repulsion pass (~150 iterations) stays cheap.
 */
function layoutGraph(
  nodes: FullGraphNode[],
  edges: FullGraphEdge[],
  width: number,
  height: number
): Record<string, Point> {
  const n = nodes.length;
  if (n === 0) return {};
  const indexOf = new Map(nodes.map((node, i) => [node.node_id, i]));
  const px = new Array<number>(n);
  const py = new Array<number>(n);
  const radius = Math.min(width, height) * 0.38;
  for (let i = 0; i < n; i++) {
    const angle = (i / n) * Math.PI * 2;
    px[i] = width / 2 + Math.cos(angle) * radius;
    py[i] = height / 2 + Math.sin(angle) * radius;
  }

  const edgeIdx: Array<[number, number]> = [];
  for (const e of edges) {
    const a = indexOf.get(e.source);
    const b = indexOf.get(e.target);
    if (a !== undefined && b !== undefined && a !== b) edgeIdx.push([a, b]);
  }

  const REPULSION = 14000;
  const SPRING = 0.02;
  const SPRING_LEN = 56;
  const CENTER_PULL = 0.01;
  const ITERATIONS = 150;

  for (let iter = 0; iter < ITERATIONS; iter++) {
    const dx = new Array<number>(n).fill(0);
    const dy = new Array<number>(n).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let ddx = px[i] - px[j];
        let ddy = py[i] - py[j];
        let distSq = ddx * ddx + ddy * ddy;
        if (distSq < 0.0001) {
          // Deterministic nudge for exactly-overlapping seeds, never random.
          ddx = 0.1 + (i - j) * 0.001;
          ddy = 0.1;
          distSq = ddx * ddx + ddy * ddy;
        }
        const dist = Math.sqrt(distSq);
        const force = REPULSION / distSq;
        const fx = (ddx / dist) * force;
        const fy = (ddy / dist) * force;
        dx[i] += fx;
        dy[i] += fy;
        dx[j] -= fx;
        dy[j] -= fy;
      }
    }

    for (const [a, b] of edgeIdx) {
      const ddx = px[a] - px[b];
      const ddy = py[a] - py[b];
      const dist = Math.sqrt(ddx * ddx + ddy * ddy) || 0.1;
      const force = SPRING * (dist - SPRING_LEN);
      const fx = (ddx / dist) * force;
      const fy = (ddy / dist) * force;
      dx[a] -= fx;
      dy[a] -= fy;
      dx[b] += fx;
      dy[b] += fy;
    }

    for (let i = 0; i < n; i++) {
      dx[i] += (width / 2 - px[i]) * CENTER_PULL;
      dy[i] += (height / 2 - py[i]) * CENTER_PULL;
    }
    for (let i = 0; i < n; i++) {
      px[i] += dx[i];
      py[i] += dy[i];
    }
  }

  const result: Record<string, Point> = {};
  nodes.forEach((node, i) => {
    result[node.node_id] = { x: px[i], y: py[i] };
  });
  return result;
}

function paginate<T>(items: T[], page: number, pageSize: number): T[] {
  return items.slice(page * pageSize, page * pageSize + pageSize);
}

function Pagination({
  page,
  pages,
  total,
  pageSize,
  onPrev,
  onNext,
  t,
}: {
  page: number;
  pages: number;
  total: number;
  pageSize: number;
  onPrev: () => void;
  onNext: () => void;
  t: TFunc;
}) {
  if (total === 0) return null;
  const from = page * pageSize + 1;
  const to = Math.min(total, (page + 1) * pageSize);
  return (
    <div className="mt-3 flex items-center justify-end gap-3 text-xs text-muted">
      <span>{t("kg.rangeOfTotal", { from, to, total })}</span>
      <button
        type="button"
        onClick={onPrev}
        disabled={page <= 0}
        className="rounded-full border border-border-dark px-3 py-1 font-medium text-content transition-colors hover:border-primary/40 disabled:cursor-default disabled:opacity-40"
      >
        {t("kg.prevPage")}
      </button>
      <button
        type="button"
        onClick={onNext}
        disabled={page >= pages - 1}
        className="rounded-full border border-border-dark px-3 py-1 font-medium text-content transition-colors hover:border-primary/40 disabled:cursor-default disabled:opacity-40"
      >
        {t("kg.nextPage")}
      </button>
    </div>
  );
}

function NodesTable({
  rows,
  onPick,
  t,
}: {
  rows: FullGraphNode[];
  onPick: (nodeId: string) => void;
  t: TFunc;
}) {
  if (rows.length === 0) {
    return <p className="py-10 text-center text-sm text-muted">{t("kg.noResults")}</p>;
  }
  return (
    <table className="w-full text-left text-[13px]">
      <thead>
        <tr className="text-[11px] font-semibold uppercase tracking-wide text-faint">
          <th className="px-3 py-2">{t("kg.colId")}</th>
          <th className="px-3 py-2">{t("kg.colName")}</th>
          <th className="px-3 py-2">{t("kg.colLabel")}</th>
          <th className="px-3 py-2">{t("kg.colMovement")}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((n) => (
          <tr
            key={n.node_id}
            onClick={() => onPick(n.node_id)}
            className="cursor-pointer border-t border-border-dark transition-colors hover:bg-content/[0.03]"
          >
            <td className="px-3 py-2 font-mono text-[11px] text-content">{n.node_id}</td>
            <td className="px-3 py-2 text-content">{n.name}</td>
            <td className="px-3 py-2">
              <span
                className="rounded-full px-2 py-0.5 text-[11px] font-medium text-white"
                style={{ backgroundColor: labelColor(n.label) }}
              >
                {n.label}
              </span>
            </td>
            <td className="px-3 py-2 text-muted">{n.movement}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EdgesTable({ rows, t }: { rows: FullGraphEdge[]; t: TFunc }) {
  if (rows.length === 0) {
    return <p className="py-10 text-center text-sm text-muted">{t("kg.noEdges")}</p>;
  }
  return (
    <table className="w-full text-left text-[13px]">
      <thead>
        <tr className="text-[11px] font-semibold uppercase tracking-wide text-faint">
          <th className="px-3 py-2">{t("kg.colSource")}</th>
          <th className="px-3 py-2">{t("kg.colRelation")}</th>
          <th className="px-3 py-2">{t("kg.colTarget")}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((e, i) => (
          <tr key={`${e.source}-${e.relation}-${e.target}-${i}`} className="border-t border-border-dark">
            <td className="px-3 py-2 font-mono text-[11px] text-content">{e.source}</td>
            <td className="px-3 py-2 text-muted">{e.relation}</td>
            <td className="px-3 py-2 font-mono text-[11px] text-content">{e.target}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Legend({ t }: { t: TFunc }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-wide text-faint">
        {t("kg.legendTitle")}
      </span>
      {Object.entries(LABEL_COLOR).map(([label, color]) => (
        <span key={label} className="flex items-center gap-1.5 text-[11px] text-muted">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}

function GraphView({
  nodes,
  edges,
  t,
}: {
  nodes: FullGraphNode[];
  edges: FullGraphEdge[];
  t: TFunc;
}) {
  const layout = useMemo(
    () => layoutGraph(nodes, edges, GRAPH_WIDTH, GRAPH_HEIGHT),
    [nodes, edges]
  );
  return (
    <div className="flex flex-col gap-3">
      <svg
        viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
        className="w-full rounded-2xl border border-border-dark bg-surface"
        style={{ height: GRAPH_HEIGHT }}
      >
        {edges.map((e, i) => {
          const a = layout[e.source];
          const b = layout[e.target];
          if (!a || !b) return null;
          return (
            <line
              key={`e${i}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke="#94a3b8"
              strokeOpacity={0.5}
              strokeWidth={1}
            >
              <title>{e.relation}</title>
            </line>
          );
        })}
        {nodes.map((n) => {
          const p = layout[n.node_id];
          if (!p) return null;
          return (
            <circle key={n.node_id} cx={p.x} cy={p.y} r={6} fill={labelColor(n.label)}>
              <title>{`${n.name} (${n.label})`}</title>
            </circle>
          );
        })}
      </svg>
      <Legend t={t} />
    </div>
  );
}

export default function KnowledgeGraph() {
  const { t } = useI18n();

  const [data, setData] = useState<FullGraphResponse | null>(null);
  const [status, setStatus] = useState<Status>("loading");

  const [movement, setMovement] = useState("all");
  const [labels, setLabels] = useState<Set<string>>(new Set());
  const [relations, setRelations] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<Tab>("nodes");
  const [nodePage, setNodePage] = useState(0);
  const [edgePage, setEdgePage] = useState(0);

  const load = useCallback(() => {
    setStatus("loading");
    api
      .fullGraph()
      .then((res) => {
        setData(res);
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => load(), [load]);

  const toggleLabel = (label: string) =>
    setLabels((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });

  const toggleRelation = (relation: string) =>
    setRelations((prev) => {
      const next = new Set(prev);
      if (next.has(relation)) next.delete(relation);
      else next.add(relation);
      return next;
    });

  // Movement scope mirrors the backend's dump_full_graph(movement=...) semantics: the nodes
  // tagged with that movement PLUS any `shared` node one hop away from them. Plain equality
  // would silently drop most of a movement's cause/cue/risk chains (they live on shared nodes).
  const movementScope = useMemo(() => {
    if (!data || movement === "all") return null;
    const scoped = new Set(data.nodes.filter((n) => n.movement === movement).map((n) => n.node_id));
    if (movement === "shared") return scoped;
    const sharedIds = new Set(
      data.nodes.filter((n) => n.movement === "shared").map((n) => n.node_id)
    );
    for (const e of data.edges) {
      if (scoped.has(e.source) && sharedIds.has(e.target)) scoped.add(e.target);
      else if (scoped.has(e.target) && sharedIds.has(e.source)) scoped.add(e.source);
    }
    return scoped;
  }, [data, movement]);

  const filteredNodes = useMemo(() => {
    if (!data) return [];
    const q = search.trim().toLowerCase();
    return data.nodes.filter((n) => {
      if (movementScope && !movementScope.has(n.node_id)) return false;
      if (labels.size > 0 && !labels.has(n.label)) return false;
      if (q && !n.node_id.toLowerCase().includes(q) && !n.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, movementScope, labels, search]);

  const filteredNodeIds = useMemo(
    () => new Set(filteredNodes.map((n) => n.node_id)),
    [filteredNodes]
  );

  const filteredEdges = useMemo(() => {
    if (!data) return [];
    return data.edges.filter((e) => {
      if (!filteredNodeIds.has(e.source) || !filteredNodeIds.has(e.target)) return false;
      if (relations.size > 0 && !relations.has(e.relation)) return false;
      return true;
    });
  }, [data, filteredNodeIds, relations]);

  // A filter change can leave the current page past the end of the new (shorter) result set —
  // reset both tables back to the top rather than showing an empty page.
  useEffect(() => setNodePage(0), [data, movement, labels, search]);
  useEffect(() => setEdgePage(0), [data, movement, labels, search, relations]);

  const movementOptions = useMemo(
    () => (data ? Object.keys(data.counts.movements).sort((a, b) => a.localeCompare(b)) : []),
    [data]
  );
  const labelOptions = useMemo(
    () => (data ? Object.keys(data.counts.labels).sort() : []),
    [data]
  );
  const relationOptions = useMemo(
    () => (data ? Object.keys(data.counts.relations).sort() : []),
    [data]
  );

  const nodePages = Math.max(1, Math.ceil(filteredNodes.length / PAGE_SIZE));
  const edgePages = Math.max(1, Math.ceil(filteredEdges.length / PAGE_SIZE));
  const clampedNodePage = Math.min(nodePage, nodePages - 1);
  const clampedEdgePage = Math.min(edgePage, edgePages - 1);
  const nodeRows = paginate(filteredNodes, clampedNodePage, PAGE_SIZE);
  const edgeRows = paginate(filteredEdges, clampedEdgePage, PAGE_SIZE);

  return (
    <AppLayout>
      <div className="flex-1 min-h-0 overflow-y-auto">
        <main className="mx-auto flex max-w-6xl flex-col px-4 py-8 lg:px-6 lg:py-12">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-2xl font-bold text-content">{t("kg.pageTitle")}</h1>
              <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">
                {t("kg.pageSubtitle")}
              </p>
            </div>
            {data && (
              <div className="flex flex-wrap gap-4 text-xs font-medium text-muted">
                <span>{t("kg.nodesShown", { shown: filteredNodes.length, total: data.total.nodes })}</span>
                <span>{t("kg.edgesShown", { shown: filteredEdges.length, total: data.total.edges })}</span>
              </div>
            )}
          </div>

          {status === "loading" && (
            <div
              role="status"
              className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-14 text-center"
            >
              <p className="text-sm text-muted">{t("kg.pageLoading")}</p>
            </div>
          )}

          {status === "error" && (
            <div className="mt-8 flex flex-col items-center gap-3 rounded-2xl border border-border-dark bg-surface px-6 py-14 text-center">
              <WarningCircle size={22} className="text-danger" weight="duotone" />
              <p className="text-sm text-muted">{t("kg.pageError")}</p>
              <button
                type="button"
                onClick={load}
                className="mt-1 rounded-full border border-border-dark px-4 py-1.5 text-xs font-medium text-content transition-colors hover:border-primary/40 active:scale-[0.98]"
              >
                {t("kg.pageRetry")}
              </button>
            </div>
          )}

          {status === "ready" && data && (
            <>
              <div className="mt-6 flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="relative sm:w-[260px]">
                    <MagnifyingGlass
                      size={15}
                      className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-faint"
                    />
                    <input
                      type="search"
                      aria-label={t("kg.searchPlaceholder")}
                      placeholder={t("kg.searchPlaceholder")}
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="glass-control h-[40px] w-full rounded-full pl-10 pr-4 text-[13px] font-medium text-content outline-none transition-colors placeholder:font-normal placeholder:text-faint hover:border-primary/40 focus:border-primary/40 [&::-webkit-search-cancel-button]:appearance-none"
                    />
                  </div>

                  <label className="flex items-center gap-2 text-xs font-medium text-muted">
                    {t("kg.filterMovement")}
                    <select
                      value={movement}
                      onChange={(e) => setMovement(e.target.value)}
                      className="glass-control h-[36px] rounded-full px-3 text-[13px] text-content outline-none"
                    >
                      <option value="all">{t("kg.allMovements")}</option>
                      {movementOptions.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-faint">
                    {t("kg.filterLabel")}
                  </span>
                  {labelOptions.map((l) => {
                    const active = labels.has(l);
                    return (
                      <button
                        key={l}
                        type="button"
                        aria-pressed={active}
                        onClick={() => toggleLabel(l)}
                        style={active ? { backgroundColor: labelColor(l) } : undefined}
                        className={`h-[28px] rounded-full px-3 text-[12px] font-medium transition-colors ${
                          active ? "text-white" : "glass-control text-muted hover:text-content"
                        }`}
                      >
                        {l} <span className="opacity-70">({data.counts.labels[l] ?? 0})</span>
                      </button>
                    );
                  })}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-faint">
                    {t("kg.filterRelation")}
                  </span>
                  {relationOptions.map((r) => {
                    const active = relations.has(r);
                    return (
                      <button
                        key={r}
                        type="button"
                        aria-pressed={active}
                        onClick={() => toggleRelation(r)}
                        className={`h-[28px] rounded-full px-3 text-[12px] font-medium transition-colors ${
                          active
                            ? "bg-primary text-primary-content"
                            : "glass-control text-muted hover:text-content"
                        }`}
                      >
                        {r} <span className="opacity-70">({data.counts.relations[r] ?? 0})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div role="tablist" className="mt-6 flex gap-1 border-b border-border-dark">
                {TABS.map((tb) => (
                  <button
                    key={tb}
                    type="button"
                    role="tab"
                    aria-selected={tab === tb}
                    onClick={() => setTab(tb)}
                    className={`px-4 py-2 text-sm font-semibold transition-colors ${
                      tab === tb
                        ? "border-b-2 border-primary text-content"
                        : "border-b-2 border-transparent text-muted hover:text-content"
                    }`}
                  >
                    {t(TAB_KEY[tb])}
                  </button>
                ))}
              </div>

              <div className="mt-4 flex-1 min-h-0 overflow-auto">
                {tab === "nodes" && (
                  <NodesTable rows={nodeRows} onPick={(id) => setSearch(id)} t={t} />
                )}
                {tab === "edges" && <EdgesTable rows={edgeRows} t={t} />}
                {tab === "graph" &&
                  (filteredNodes.length > MAX_GRAPH_NODES ? (
                    <p className="py-10 text-center text-sm text-muted">{t("kg.tooManyNodes")}</p>
                  ) : (
                    <GraphView nodes={filteredNodes} edges={filteredEdges} t={t} />
                  ))}
              </div>

              {tab === "nodes" && (
                <Pagination
                  page={clampedNodePage}
                  pages={nodePages}
                  total={filteredNodes.length}
                  pageSize={PAGE_SIZE}
                  onPrev={() => setNodePage((p) => Math.max(0, p - 1))}
                  onNext={() => setNodePage((p) => Math.min(nodePages - 1, p + 1))}
                  t={t}
                />
              )}
              {tab === "edges" && (
                <Pagination
                  page={clampedEdgePage}
                  pages={edgePages}
                  total={filteredEdges.length}
                  pageSize={PAGE_SIZE}
                  onPrev={() => setEdgePage((p) => Math.max(0, p - 1))}
                  onNext={() => setEdgePage((p) => Math.min(edgePages - 1, p + 1))}
                  t={t}
                />
              )}
            </>
          )}
        </main>
      </div>
    </AppLayout>
  );
}
