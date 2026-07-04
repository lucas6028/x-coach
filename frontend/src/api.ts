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

// ---- Conversational coaching (LLM chat, grounded in an analysis) --------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// One detected fault plus its retrieved knowledge, as the frontend already derives it for the
// ReasoningLog view. Sent to the backend so it can build a grounded system prompt server-side.
export interface ChatFaultContext {
  fault_name: string;
  phase?: string;
  severity?: number;
  start_time?: number;
  end_time?: number;
  evidence?: string;
  causes: string[];
  risks: string[];
  corrections: string[];
  rag_snippet?: string | null;
}

export interface ChatContext {
  video_id?: string;
  view_type?: string;
  view_confidence?: number;
  fault_count: number;
  quality: Record<string, number>;
  faults: ChatFaultContext[];
}

// Callbacks the streaming chat client drives as SSE frames arrive. `onError` carries an *in-band*
// failure (OpenRouter connect/mid-stream/empty) — the stream already returned 200, so it is not a
// thrown ChatError. A pre-flight failure (401/422/503) is thrown as a ChatError before any of these
// fire, so the two failure modes stay distinguishable to the caller.
export interface ChatStreamHandlers {
  onDelta: (text: string) => void;
  onDone: (model: string) => void;
  onError: (detail: string) => void;
}

// A persisted chat thread for one analysed video (one per user+video_id). Restored on history-replay.
export interface Conversation {
  video_id: string;
  messages: ChatMessage[];
}

// Parse one SSE frame ("event: <e>\ndata: <json>") and dispatch it to the handlers. A frame with no
// event line, or an unparseable data payload, is ignored (keep-alives / partial writes).
function dispatchSSE(frame: string, handlers: ChatStreamHandlers): void {
  let event = "";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!event) return;
  let data: { text?: string; model?: string; detail?: string };
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }
  if (event === "delta") handlers.onDelta(data.text ?? "");
  else if (event === "done") handlers.onDone(data.model ?? "");
  else if (event === "error") handlers.onError(data.detail ?? "Chat failed");
}

// Carries the HTTP status so the UI can tell an expired session (401) apart from an LLM outage
// (502/503) and message the user accordingly, instead of a single undifferentiated failure.
export class ChatError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "ChatError";
  }
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
  health: () =>
    getJSON<{ status: string; auth_configured?: boolean; chat_configured?: boolean }>(
      "/api/health"
    ),

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

  // Grounded follow-up chat about an analysis, streamed as Server-Sent Events (requires a signed-in
  // session; 401 otherwise). `messages` is the conversation so far, oldest first, with the new user
  // turn last; `context` is the compact grounding blob from buildChatContext(analysis). Deltas,
  // completion, and in-band errors are delivered via `handlers`; a pre-flight failure (before the
  // stream opens) throws a ChatError carrying the HTTP status so the caller can tell an expired
  // session (401) from an LLM outage. `model` is the user's chosen OpenRouter slug (validated
  // server-side); omit it to use the server default.
  async chatStream(
    messages: ChatMessage[],
    context: ChatContext,
    handlers: ChatStreamHandlers,
    model?: string
  ): Promise<void> {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      // The server validates `model` against its allowlist; omit it to use the server default.
      body: JSON.stringify(model ? { messages, context, model } : { messages, context }),
    });
    if (!res.ok || !res.body) {
      const detail = await res.json().catch(() => ({}));
      throw new ChatError(
        (detail as { detail?: string }).detail || `Chat failed (${res.status})`,
        res.status
      );
    }

    // Read the byte stream, splitting on the blank-line frame boundary. A frame can straddle two
    // chunks, so buffer until a full "\n\n"-terminated frame is available before dispatching.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        dispatchSSE(buffer.slice(0, sep), handlers);
        buffer = buffer.slice(sep + 2);
      }
    }
  },

  // Restore the caller's saved chat thread for a video ({messages: []} when none). Requires a
  // session; the tray only calls it for a signed-in, chat-configured user.
  getConversation: (videoId: string) =>
    getJSON<Conversation>(`/api/conversations/${encodeURIComponent(videoId)}`),

  // Save the caller's chat thread for a video (idempotent upsert of the whole thread).
  async putConversation(videoId: string, messages: ChatMessage[]): Promise<void> {
    const url = `/api/conversations/${encodeURIComponent(videoId)}`;
    const res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify({ messages }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
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
