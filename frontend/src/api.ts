// Typed client for the x-coach FastAPI backend. URLs are relative; Vite proxies /api -> :8000.

import { supabase } from "./lib/supabase";

export interface VideoMeta {
  fps: number;
  width: number;
  height: number;
  total_frames: number;
}

export interface ViewInfo {
  view_type: string;
  view_confidence: number;
  [k: string]: unknown;
}

export interface Detection {
  fault_id: string;
  fault_name: string;
  kg_query: string;
  retrieval_mode: string;
  severity: number;
  confidence: number;
  observability: string;
  start_time: number;
  end_time: number;
  start_frame: number;
  end_frame: number;
  peak_frame: number;
  phase: string;
  evidence: Record<string, number | string>;
}

export interface SubgraphNode {
  node_id: string;
  label: string;
}
export interface SubgraphEdge {
  source: string;
  target: string;
  relation: string;
  direction?: string;
}
export interface RagResult {
  rank: number;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
}

export interface RetrievalContext {
  // KG mode
  matched_nodes?: string[];
  results?: Array<Record<string, unknown>> | RagResult[];
  subgraph?: { nodes: SubgraphNode[]; edges: SubgraphEdge[] };
  query?: string;
}

export interface Retrieval {
  fault_id: string;
  fault_name: string;
  query_text: string;
  retrieval_mode: string;
  context: RetrievalContext;
}

export interface PoseFrame {
  i: number;
  lm: [number, number, number][] | null; // [x, y, visibility] x 33
}
export interface PoseBlock {
  fps: number;
  width: number;
  height: number;
  frames: PoseFrame[];
}

export interface Analysis {
  video_id: string;
  metadata: VideoMeta;
  view: ViewInfo;
  quality: Record<string, number>;
  detections: Detection[];
  retrievals: Retrieval[];
  pose: PoseBlock;
  ground_truth?: Record<string, number[][]>;
  source: "library" | "upload";
  // Present when an authenticated upload was persisted to the user's history (null if the
  // save failed). Absent for anonymous uploads and library clips.
  analysis_id?: string | null;
}

// A row in the user's history list (the promoted columns, no heavy result payload).
export interface HistoryItem {
  id: string;
  video_id: string;
  source: string;
  view_type: string | null;
  fault_count: number;
  created_at: string;
}
export interface HistoryPage {
  total: number;
  items: HistoryItem[];
}

// A single stored analysis row; `result` is the full Analysis document for replay.
export interface StoredAnalysis {
  id: string;
  video_id: string;
  source: string;
  view_type: string | null;
  fault_count: number;
  created_at: string;
  result: Analysis;
}

export interface LibraryItem {
  video_id: string;
  split: string;
  view_type: string;
  fault_count: number;
  faults: string[];
}
export interface LibraryPage {
  total: number;
  items: LibraryItem[];
}

// Bearer header for the current Supabase session, or {} when logged out / not configured.
// Returning {} (not a header with an empty token) keeps anonymous requests single-arg.
async function authHeader(): Promise<Record<string, string>> {
  if (!supabase) return {};
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function getJSON<T>(url: string): Promise<T> {
  const headers = await authHeader();
  // Only pass an init object when we actually have a token, so public reads stay header-free.
  const res = Object.keys(headers).length ? await fetch(url, { headers }) : await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return (await res.json()) as T;
}

export const api = {
  health: () => getJSON<{ status: string }>("/api/health"),

  listVideos: (limit = 50, offset = 0, fault?: string) =>
    getJSON<LibraryPage>(
      `/api/videos?limit=${limit}&offset=${offset}` + (fault ? `&fault=${fault}` : "")
    ),

  getAnalysis: (videoId: string) => getJSON<Analysis>(`/api/analysis/${videoId}`),

  graph: (query: string) =>
    getJSON<RetrievalContext>(`/api/knowledge/graph?query=${encodeURIComponent(query)}`),

  videoFileUrl: (videoId: string) => `/api/video-file/${videoId}`,

  // The caller's saved analyses, newest first (requires a signed-in session).
  listAnalyses: (limit = 50, offset = 0) =>
    getJSON<HistoryPage>(`/api/analyses?limit=${limit}&offset=${offset}`),

  // One saved analysis row, including the full `result` for replay (requires a session).
  getStoredAnalysis: (id: string) => getJSON<StoredAnalysis>(`/api/analyses/${id}`),

  // Delete all of the caller's saved analyses (requires a session). Returns the count removed.
  async deleteAnalyses(): Promise<{ deleted: number }> {
    const res = await fetch("/api/analyses", { method: "DELETE", headers: await authHeader() });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for /api/analyses`);
    return (await res.json()) as { deleted: number };
  },

  async analyzeUpload(file: File): Promise<Analysis> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      headers: await authHeader(),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error((detail as { detail?: string }).detail || `Analyze failed (${res.status})`);
    }
    return (await res.json()) as Analysis;
  },
};
