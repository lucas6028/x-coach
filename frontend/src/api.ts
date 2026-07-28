// Typed client for the x-coach FastAPI backend. URLs are relative; Vite proxies /api -> :8000.

import { supabase } from "./lib/supabase";
import type { AnalyzableMovement } from "./lib/movements";
import type { PoseJson, RepsPlan } from "./lib/poseExtract";

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
  name?: string;
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

// One fault a movement defines, with its 1-hop graph connectivity (0 = no linked
// causes/corrections/risks to render yet).
export interface MovementFault {
  name: string;
  connectivity: number;
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
  /** Which detector produced this analysis. Absent on analyses predating per-movement
   *  selection; consumers fall back to "Squat". */
  movement?: string;
  // Present only when the request carried a client-planned `reps` (see `analyzePose`'s `reps`
  // param). NOT the same shape as the request's RepsPlan: the backend reports which segments it
  // actually analyzed (`analyzed` indices) and adds `start_time`/`end_time` per segment, derived
  // server-side from fps — the browser's plan only knows frame indices.
  reps?: {
    detected: number;
    analyzed: number[];
    max_reps: number | null;
    fallback: string | null;
    segments: {
      index: number; start_frame: number; end_frame: number;
      start_time: number; end_time: number; analyzed: boolean; partial: boolean;
    }[];
  };
}

// A row in the user's history list (the promoted columns, no heavy result payload).
export interface HistoryItem {
  id: string;
  video_id: string;
  source: string;
  view_type: string | null;
  fault_count: number;
  created_at: string;
  movement?: string | null;
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
  movement?: string;
}

// Callbacks the streaming chat client drives as SSE frames arrive. `onError` carries an *in-band*
// failure (LLM provider connect/mid-stream/empty) — the stream already returned 200, so it is not a
// thrown ChatError. A pre-flight failure (401/422/503) is thrown as a ChatError before any of these
// fire, so the two failure modes stay distinguishable to the caller.
export interface ChatStreamHandlers {
  onDelta: (text: string) => void;
  onDone: (model: string) => void;
  onError: (detail: string) => void;
}

// A persisted chat thread for one analysed video (one per user+video_id). Restored on history-replay.
// `followups` is the latest answer's grounded next-question chips, persisted so a reload restores the
// chips too (not just the answer). Optional — absent/empty for pre-followups rows or a cleared thread.
export interface Conversation {
  video_id: string;
  messages: ChatMessage[];
  followups?: string[];
}

// The coach-model picker is server-driven (from /api/health): the authoritative list of selectable
// model ids + which is the default. Display names/logos are a frontend concern (see ModelIcon).
export interface HealthResponse {
  status: string;
  auth_configured?: boolean;
  chat_configured?: boolean;
  // Whether POST /api/auth/line (the in-LIFF silent login bridge) is configured server-side.
  line_login_configured?: boolean;
  chat_models?: string[];
  chat_default?: string;
}

// A minted Supabase session from the LINE bridge — handed straight to supabase.auth.setSession.
export interface LineSession {
  access_token: string;
  refresh_token: string;
}

// Non-chat endpoint failure that still needs its HTTP status (e.g. the LINE bridge: 401 means
// "stale LINE token, re-run liff.login()", anything else is a real error to surface).
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ---- Admin: runtime settings (admin-only; GET/PUT /api/admin/settings) --------------------

export interface AdminLlmSettings {
  llm_models: string[];
  llm_followup_model: string;
  llm_base_url: string;
  chat_temperature: number | null;
  chat_timeout: number;
  followup_timeout: number;
}
export interface AdminRagKgSettings {
  rag_top_k: number;
  kg_hops: number;
  kg_seeds: number;
}
export interface AdminAnalyzeSettings {
  allowed_upload_suffixes: string[];
  max_concurrent_analyses: number;
}
export interface AdminSettingsGroups {
  llm: AdminLlmSettings;
  rag_kg: AdminRagKgSettings;
  analyze: AdminAnalyzeSettings;
}
// GET/PUT both return the currently-effective knobs plus their env/constant defaults, grouped so the
// form can show "current vs default". No secret (API key / Supabase creds) ever appears here.
export interface AdminSettingsResponse {
  effective: AdminSettingsGroups;
  defaults: AdminSettingsGroups;
}
// A partial update — only the provided knobs are persisted. Ranges are validated server-side (422).
export interface AdminSettingsUpdate {
  llm_models?: string[];
  llm_followup_model?: string;
  llm_base_url?: string;
  chat_temperature?: number | null;
  chat_timeout?: number;
  followup_timeout?: number;
  rag_top_k?: number;
  kg_hops?: number;
  kg_seeds?: number;
  // max_concurrent_analyses is intentionally omitted: it's read-only (sourced from the
  // XCOACH_MAX_CONCURRENT_ANALYSES env var, applied at startup) and must never be sent in an update.
  allowed_upload_suffixes?: string[];
}

// ---- Admin: user oversight + system overview (admin-only; P3) -----------------------------

// One row of the admin users table: identity, activity counts, and whether they hold the admin role.
export interface AdminUserRow {
  id: string;
  email: string | null;
  created_at: string;
  last_sign_in_at: string | null;
  analyses_count: number;
  conversations_count: number;
  is_admin: boolean;
}
export interface AdminUsersResponse {
  users: AdminUserRow[];
}
// The admin dashboard: the same health flags as /api/health plus user/analysis totals. No secret here.
export interface AdminOverview {
  auth_configured: boolean;
  chat_configured: boolean;
  chat_models: string[];
  chat_default: string;
  stores: Record<string, boolean>;
  total_users: number;
  total_analyses: number;
}

// LINE connection status + this month's push-message quota (admin-only; read-only). No secret here.
export interface LineQuota {
  type: "limited" | "none";
  used: number;
  value?: number; // present only when type === "limited"
  remaining?: number; // present only when type === "limited"
}
export interface LineStatus {
  messaging_configured: boolean;
  login_configured: boolean;
  // The LINE *Login* channel id (non-secret). NOTE: this is a DIFFERENT channel from the Messaging
  // bot — do not render it under the bot-status card. Currently surfaced for status only, not displayed.
  channel_id: string;
  quota: LineQuota | null;
  quota_error: "unreachable" | null;
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
  health: () => getJSON<HealthResponse>("/api/health"),

  // Exchange a LINE (LIFF) ID token for a Supabase session (see backend routers/auth_line).
  // Unauthenticated by design — the ID token itself is the proof of identity. Throws ApiError
  // carrying the HTTP status so the caller can tell a stale LINE token (401) from an outage.
  async lineLogin(idToken: string): Promise<LineSession> {
    const res = await fetch("/api/auth/line", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token: idToken }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new ApiError(
        (detail as { detail?: string }).detail || `LINE login failed (${res.status})`,
        res.status
      );
    }
    return (await res.json()) as LineSession;
  },

  // Whether the signed-in caller holds the admin role. Any signed-in user may ask (the backend
  // returns their own flag, not a 403); used to gate the Admin nav link and page. Auto-attaches
  // the bearer token via getJSON.
  adminStatus: () => getJSON<{ is_admin: boolean }>("/api/admin/status"),

  // The effective runtime knobs + their defaults (admin-only; 403 for a non-admin). Auto-attaches
  // the bearer token via getJSON.
  getAdminSettings: () => getJSON<AdminSettingsResponse>("/api/admin/settings"),

  // Persist a partial settings update (admin-only). Only the provided knobs are written; the backend
  // validates ranges (422 on a bad value) and returns the new effective state. Auth header auto-attached.
  async updateAdminSettings(payload: AdminSettingsUpdate): Promise<AdminSettingsResponse> {
    const res = await fetch("/api/admin/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for /api/admin/settings`);
    return (await res.json()) as AdminSettingsResponse;
  },

  // The read-only users overview (admin-only; 403 for a non-admin). Auth header auto-attached.
  listAdminUsers: () => getJSON<AdminUsersResponse>("/api/admin/users"),

  // The system-status dashboard: health flags + user/analysis totals (admin-only). Auth auto-attached.
  getAdminOverview: () => getJSON<AdminOverview>("/api/admin/overview"),

  // LINE connection status + push-quota usage (admin-only). Auth header auto-attached.
  getLineStatus: () => getJSON<LineStatus>("/api/admin/line/status"),

  // Grant/revoke another user's admin role (admin-only). The backend rejects self-demotion (400).
  async setUserRole(userId: string, makeAdmin: boolean): Promise<{ ok: boolean }> {
    const url = `/api/admin/users/${encodeURIComponent(userId)}/role`;
    const res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify({ make_admin: makeAdmin }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
    return (await res.json()) as { ok: boolean };
  },

  // The movements the pipeline can actually analyse, derived server-side from the detector
  // registry. Backs the /movements cards and the studio selector.
  getMovements: () =>
    getJSON<{ movements: AnalyzableMovement[] }>("/api/movements").then((r) => r.movements),

  listVideos: (limit = 50, offset = 0, fault?: string) =>
    getJSON<LibraryPage>(
      `/api/videos?limit=${limit}&offset=${offset}` + (fault ? `&fault=${fault}` : "")
    ),

  getAnalysis: (videoId: string) => getJSON<Analysis>(`/api/analysis/${videoId}`),

  graph: (query: string, movement?: string) =>
    getJSON<RetrievalContext>(
      `/api/knowledge/graph?query=${encodeURIComponent(query)}` +
        (movement ? `&movement=${encodeURIComponent(movement)}` : "")
    ),

  // The complete, movement-scoped fault list (name + connectivity), enumerated by the graph's
  // `movement` node attribute so no fault is hidden. Backs GET /api/knowledge/faults.
  movementFaults: (movement: string) =>
    getJSON<{ movement: string; faults: MovementFault[] }>(
      `/api/knowledge/faults?movement=${encodeURIComponent(movement)}`
    ),

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
  // session (401) from an LLM outage. `model` is the user's chosen model slug (validated
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

  // Two grounded next-question suggestions for a completed turn (a separate, best-effort call the
  // client fires *after* the answer renders — fire-and-forget, so it never blocks the answer).
  // `messages` is the thread ending on the assistant answer; `context` is the same grounding blob.
  // Resolves to the questions (or `[]` on any non-ok response, since a missing chip isn't worth
  // surfacing an error). `model` is the user's chosen slug (validated server-side); omit for default.
  async chatFollowups(
    messages: ChatMessage[],
    context: ChatContext,
    model?: string
  ): Promise<string[]> {
    const res = await fetch("/api/chat/followups", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify(model ? { messages, context, model } : { messages, context }),
    });
    if (!res.ok) return [];
    const data = (await res.json().catch(() => ({}))) as { questions?: string[] };
    return data.questions ?? [];
  },

  // Restore the caller's saved chat thread for a video ({messages: []} when none). Requires a
  // session; the tray only calls it for a signed-in, chat-configured user.
  getConversation: (videoId: string) =>
    getJSON<Conversation>(`/api/conversations/${encodeURIComponent(videoId)}`),

  // Save the caller's chat thread for a video (idempotent upsert of the whole thread). `followups`
  // is the latest answer's chips; omit (or []) to clear them — matching the "clear on new send" flow.
  async putConversation(
    videoId: string,
    messages: ChatMessage[],
    followups: string[] = []
  ): Promise<void> {
    const url = `/api/conversations/${encodeURIComponent(videoId)}`;
    const res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...(await authHeader()) },
      body: JSON.stringify({ messages, followups }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  },

  async analyzeUpload(file: File, movement: string): Promise<Analysis> {
    const form = new FormData();
    form.append("file", file);
    // Which detector runs. The backend rejects an unregistered value with 400 before it spends
    // a MediaPipe pass, and echoes the canonical spelling back as `movement` on the result.
    form.append("movement", movement);
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

  async analyzePose(movement: string, pose: PoseJson, video: Blob, reps?: RepsPlan): Promise<Analysis> {
    const form = new FormData();
    form.append("movement", movement);
    form.append("pose", JSON.stringify(pose));
    // Only sent when the browser actually planned the extraction. Omitting it keeps the endpoint
    // on its pre-SP2 path, which is what the CLI and any old client rely on.
    if (reps) form.append("reps", JSON.stringify(reps));
    const ext = video.type.includes("mp4") ? "mp4" : "webm";
    form.append("file", video, `capture.${ext}`);
    const res = await fetch("/api/analyze/pose", {
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
