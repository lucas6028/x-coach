import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, CheckCircle, PaperPlaneTilt, SignIn, Warning } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { api, ChatError, type Analysis, type ChatMessage, type LiveToolRun, type ToolRun } from "../api";
import { buildChatContext } from "../lib/grounding";
import { retrievalByFault } from "../lib/retrieval";
import { wasMeasured } from "../lib/quality";
import { getStoredModel } from "../lib/model";
import { useAuth } from "../lib/auth";
import { movementLabel, useI18n } from "../lib/i18n";
import FaultCard from "./FaultCard";
import KnowledgeGraphWidget from "./KnowledgeGraphWidget";
import { LumenAvatar, LumenLoader } from "./LumenLoader";
import Markdown from "./Markdown";
import { ToolRunList } from "./ToolRunList";

interface Props {
  analysis: Analysis;
  currentTime: number;
  onSeek: (t: number) => void;
  // Which fault the playhead is currently inside — drives the KG's centered node. Defaults to
  // null (no active fault → the graph seeds from the first retrieval).
  activeFaultId?: string | null;
}

// The coaching tray, unified as one "chat room": the grounded rule+GraphRAG analysis (fault cards
// / clean-rep) is the coach's opening, and the LLM follow-up conversation continues in the SAME
// scroll thread below it, with a single composer pinned at the foot. The feedback always renders
// (the client already holds the analysis); only the composer adapts to the three honest states —
// working chat (signed in + server-configured), a sign-in invite, or the disabled "coming soon"
// affordance when auth or the LLM provider key is absent.
export default function CoachTray({
  analysis,
  currentTime,
  onSeek,
  activeFaultId = null,
}: Props) {
  const { t } = useI18n();
  const { configured, user } = useAuth();
  const reduce = useReducedMotion();

  const byFault = useMemo(() => retrievalByFault(analysis.retrievals), [analysis]);
  const detections = analysis.detections;
  // An empty `detections` list means BOTH "no faults found" and "no frame was ever measurable".
  // Same SHARED criterion MetricsCards uses (src/lib/quality.ts) — the HUD and this banner
  // co-render on one screen (App.tsx), so a second, independently-drifting definition would let
  // them contradict each other, which is worse than the bug being fixed.
  const measured = wasMeasured(analysis.quality);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // The in-progress assistant turn, accumulated from streamed deltas. Rendered as a live block
  // below the committed thread while loading; committed into `messages` on a clean `done`, or
  // discarded on an error (the optimistic user turn is rolled back alongside it).
  const [streaming, setStreaming] = useState("");
  // Every tool call this turn, in order. Unlike v3's single transient line these are NOT cleared
  // when the answer starts: the record is the answer's provenance and belongs beside it.
  const [toolRuns, setToolRuns] = useState<LiveToolRun[]>([]);
  // Two grounded next-question suggestions the coach offers after an answer. Captured per answer and
  // persisted alongside the thread (only the latest set), so a reload restores the chips, not just
  // the response; cleared on a new send / analysis.
  const [followups, setFollowups] = useState<string[]>([]);
  // Whether the *server* has an LLM provider key. Independent of Supabase auth. null = not yet
  // checked (assume available so the common path is instant).
  const [chatOnServer, setChatOnServer] = useState<boolean | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Monotonic token for the fire-and-forget follow-up fetch: a suggestion only lands if its turn is
  // still the latest (guards against a slow fetch resolving under a newer turn or a switched analysis).
  const followupSeq = useRef(0);
  // Sticky-scroll intent: true while the user sits at/near the bottom (auto-follow new content), flips
  // to false the moment they scroll up to read earlier messages (so streaming/chips don't yank them
  // back down). A fresh send re-engages it. Default true = follow until the user says otherwise.
  const stickToBottom = useRef(true);

  // Whether the composer is a live chat (signed in + server-configured). Also gates thread
  // restore/persist. Declared before the effects that depend on it (avoids a TDZ ref in deps).
  const isWorking = configured && !!user && chatOnServer !== false;
  const canSend = !!input.trim() && !loading;

  // Suggestion-chip styling, shared by the empty-state starters and the per-answer follow-ups so the
  // two read as the same affordance ("和一開始的選項一樣").
  const chipClass =
    "glass-control flex items-center justify-between gap-2 rounded-2xl px-3.5 py-2.5 text-left text-[12.5px] text-[#3a3d5a] transition-colors";

  // A new analysis starts fresh, then restores its saved thread if the session can persist: a
  // history-replay of a saved analysis brings its conversation back, while a fresh upload (no saved
  // thread yet) simply stays empty. A failed/absent fetch leaves the empty thread — never blocks.
  useEffect(() => {
    setMessages([]);
    setError("");
    setFollowups([]);
    setToolRuns([]);
    followupSeq.current++; // invalidate any in-flight suggestion from the previous analysis
    if (!isWorking) return;
    let active = true;
    api
      .getConversation(analysis.video_id)
      .then((c) => {
        if (!active || !c.messages?.length) return;
        setMessages(c.messages);
        // Restore the latest answer's chips too, so a reload brings the suggestions back — not just
        // the response. Empty for pre-followups threads (they simply restore without chips).
        if (c.followups?.length) setFollowups(c.followups);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- isWorking gates restore; not identity.
  }, [analysis.video_id, isWorking]);

  // Ask the backend once whether the LLM is configured, so a key-less server shows the disabled
  // affordance rather than a live input that 503s. Only a definitive chat_configured=false disables
  // the chat; a transient health-check failure stays optimistic (a real misconfig then surfaces as
  // a recoverable 503 on send, unlike the permanent fallback a naive false would cause).
  useEffect(() => {
    if (!configured || !user) return;
    let active = true;
    api
      .health()
      .then((h) => active && setChatOnServer(!!h.chat_configured))
      .catch(() => active && setChatOnServer(true));
    return () => {
      active = false;
    };
  }, [configured, user]);

  // Keep the newest turn in view as the conversation grows — but only while the user is following the
  // bottom (see `stickToBottom`), and never on the initial render, so the coach's analysis (top of the
  // thread) is what the user sees first. (Guard scrollTo — jsdom.) `streaming` tracks the answer as
  // tokens arrive; `followups` follows the chips down when they land (they arrive async, after the turn
  // commits, so without this dep they'd render below the fold).
  useEffect(() => {
    if (messages.length === 0) return;
    const el = scrollRef.current;
    if (stickToBottom.current && el && typeof el.scrollTo === "function")
      el.scrollTo({ top: el.scrollHeight });
  }, [messages, loading, streaming, followups]);

  // Track whether the user is following the bottom: within a small threshold of the foot ⇒ keep
  // auto-scrolling; scrolled up ⇒ stop, and stay put until they return to the bottom (or send again).
  function onThreadScroll() {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 80;
  }

  // `textArg` lets a starter-suggestion chip send its prompt directly; otherwise we send the input.
  async function send(textArg?: string) {
    const text = (textArg ?? input).trim();
    if (!text || loading) return;
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    setError("");
    setLoading(true);
    setStreaming("");
    setToolRuns([]);
    setFollowups([]); // drop the previous answer's suggestions while this one streams
    stickToBottom.current = true; // a fresh send re-engages auto-follow (user is acting at the foot)
    const mySeq = ++followupSeq.current; // this turn owns the next suggestion result
    let acc = "";
    let runs: LiveToolRun[] = [];
    let inbandError = "";
    // True only once the stream's `done` frame has actually been seen. `api.chatStream` resolves
    // the instant the reader drains, regardless of which frame (if any) was last — so a connection
    // that dies mid-flight with neither `done` nor `error` looks exactly like success here. That
    // window is widest right after `onReset` clears `acc` to "": the very next thing the server
    // does is `_dispatch_tool`, a graphml load plus a vector query, the slowest blocking work in
    // the request, and a proxy idle-timeout there would otherwise commit and persist an EMPTY
    // assistant turn. That is unrecoverable — `ChatRequest.messages` requires `content` to be
    // non-empty, so the thread could never be posted to again. Checked before `inbandError` so an
    // unterminated stream is never mistaken for clean just because no error frame happened to fire.
    let finished = false;
    try {
      await api.chatStream(
        next,
        buildChatContext(analysis),
        {
          onDelta: (tkn) => {
            acc += tkn;
            setStreaming(acc);
            // NOTE: unlike v3, the tool records are deliberately NOT cleared here.
          },
          onDone: () => {
            finished = true;
          },
          onTool: (id, name, query) => {
            runs = [...runs, { id, name, query, pending: true }];
            setToolRuns(runs);
          },
          // Match on `id`, and only while still pending: a duplicate or replayed frame must not
          // overwrite a run that has already settled.
          onToolDone: (id, sources) => {
            runs = runs.map((r) =>
              r.id === id && r.pending
                ? { ...r, pending: false, ...(sources.length ? { sources } : {}) }
                : r,
            );
            setToolRuns(runs);
          },
          // The round that produced this text also called a tool, so it was narration, not the
          // answer. Drop the text — but NOT `runs`: those calls really happened and really fed the
          // answer, so erasing them would misreport the reasoning chain.
          onReset: () => {
            acc = "";
            setStreaming("");
          },
          // An in-band error (LLM provider connect/mid-stream/empty) isn't thrown — capture it and
          // rethrow below so success and failure share one rollback path.
          onError: (detail) => {
            inbandError = detail;
          },
        },
        getStoredModel(), // the user's Settings choice; server validates against its allowlist
      );
      if (!finished) throw new ChatError("The coach connection ended unexpectedly.", 502);
      if (inbandError) throw new ChatError(inbandError, 502);
      // Strip the in-memory transport/UI fields by REBUILDING each run from an allow-list rather
      // than destructuring them away: a field added to LiveToolRun later must not silently ride
      // along into stored jsonb. This also settles any run still pending because its `tool_done`
      // was lost or uncorrelatable — the committed record simply shows no sources, which is the
      // truth, instead of a row that claims to still be running.
      const committed: ToolRun[] = runs.map((r) => ({
        name: r.name,
        query: r.query,
        ...(r.sources?.length ? { sources: r.sources } : {}),
      }));
      const thread: ChatMessage[] = [
        ...next,
        { role: "assistant", content: acc, ...(committed.length ? { tools: committed } : {}) },
      ];
      setMessages(thread);
      // Persist the completed turn (fire-and-forget — a save failure must not disrupt the chat). Chips
      // are written empty here (clearing the previous answer's), then re-persisted below once this
      // turn's chips land — so the message survives even if the follow-up fetch fails or hangs.
      void api.putConversation(analysis.video_id, thread).catch(() => undefined);
      // Fire-and-forget the follow-up chips: the answer is already on screen, so we fetch two grounded
      // next-questions in the background and drop them in when they arrive — unless a newer turn or a
      // switched analysis has since bumped `followupSeq`, in which case this stale result is ignored.
      // When they land for the still-latest turn, re-persist the thread *with* the chips so a reload
      // restores them (persist the captured `thread`, not the `messages` state — no stale closure).
      void api
        .chatFollowups(thread, buildChatContext(analysis), getStoredModel())
        .then((qs) => {
          if (mySeq !== followupSeq.current) return;
          setFollowups(qs);
          if (qs.length)
            void api.putConversation(analysis.video_id, thread, qs).catch(() => undefined);
        })
        .catch(() => undefined);
    } catch (e) {
      // Roll back the optimistic user turn (the partial assistant text lives in `streaming`, which
      // `finally` clears — nothing to slice) and restore the text so a retry doesn't duplicate it.
      setMessages((m) => m.slice(0, -1));
      setInput(text);
      const expired = e instanceof ChatError && e.status === 401;
      setError(expired ? t("chat.sessionExpired") : t("chat.error"));
    } finally {
      setLoading(false);
      setStreaming("");
      setToolRuns([]);
    }
  }

  // --- Composer (foot of the tray): one of three honest states -----------------------------
  const disabledComposer = (
    <div className="relative" title={t("chat.title")}>
      <input
        disabled
        className="glass-control w-full cursor-not-allowed rounded-full py-2.5 pl-4 pr-10 text-[12.5px] text-[#63709f] placeholder-[#b8bcd3]"
        placeholder={t("chat.placeholder")}
      />
      <PaperPlaneTilt size={18} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#63709f]" />
    </div>
  );

  const signInComposer = (
    <div className="flex items-center gap-3 rounded-2xl border border-[#ece8ff] bg-white p-3">
      <LumenAvatar size={36} className="shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed text-[#59648f]">{t("chat.signIn")}</p>
      <Link
        to="/login"
        className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-gradient-to-r from-[#a48bff] to-[#7b5cff] px-3 py-1.5 text-xs font-medium text-white shadow-[0_4px_12px_rgba(123,92,255,0.3)] transition-transform active:scale-[0.98]"
      >
        <SignIn size={15} weight="bold" />
        {t("account.signin")}
      </Link>
    </div>
  );

  const workingComposer = (
    // The reference's pill composer: a white capsule with the send action as a violet disc.
    <div className="glass-control flex items-center gap-2 rounded-full px-2 py-2 transition-all focus-within:border-[#c9bcff]">
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void send();
          }
        }}
        disabled={loading}
        aria-label={t("chat.heading")}
        className="min-w-0 flex-1 bg-transparent px-2.5 text-[12.5px] text-[#3a3d5a] placeholder-[#b8bcd3] focus:outline-none disabled:opacity-60"
        placeholder={t("chat.placeholderActive")}
      />
      <button
        type="button"
        onClick={() => void send()}
        disabled={!canSend}
        aria-label={t("chat.send")}
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-r from-[#a48bff] to-[#7b5cff] text-white shadow-[0_4px_12px_rgba(123,92,255,0.3)] transition enabled:active:scale-95 disabled:from-[#e6e6f2] disabled:to-[#e6e6f2] disabled:text-[#63709f] disabled:shadow-none"
      >
        <PaperPlaneTilt size={15} weight={canSend ? "fill" : "regular"} />
      </button>
    </div>
  );

  const composer = !configured
    ? disabledComposer
    : !user
      ? signInComposer
      : chatOnServer === false
        ? disabledComposer
        : workingComposer;

  // The byline above each of Lumen's answers (committed and streaming share it): her avatar, her
  // name, and the "grounded in your analysis" provenance tag.
  const coachTag = (
    <div className="mb-1.5 flex items-center gap-2">
      <LumenAvatar size={18} />
      <span className="text-[11px] font-semibold uppercase tracking-wide text-primary">
        {t("chat.coach")}
      </span>
      <span
        title={t("chat.grounded")}
        className="inline-flex items-center gap-1 text-[10px] font-medium text-secondary"
      >
        <CheckCircle size={12} weight="fill" />
        {t("chat.groundedShort")}
      </span>
    </div>
  );

  return (
    <div className="glass-panel flex min-h-0 w-full flex-1 flex-col overflow-hidden rounded-[24px]">
      {/* One header for the whole coaching conversation. */}
      <div className="flex shrink-0 items-center justify-between gap-2 px-4 pb-3 pt-4">
        <h2 className="flex items-center gap-2.5 text-[15px] font-bold tracking-tight text-[#1e2142]">
          <LumenAvatar size={26} />
          {t("chat.heading")}
        </h2>
        <span
          title={t("chat.grounded")}
          className="flex items-center gap-1.5 rounded-full border border-[#d0f0dc] bg-[#e6f7ed] px-2.5 py-1 text-[11px] font-semibold text-[#1a9e5a]"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-[#22c55e]" />
          {t("feedback.badge")}
        </span>
      </div>

      {/* One scroll thread: the grounded analysis first, then the conversation. */}
      <div
        ref={scrollRef}
        onScroll={onThreadScroll}
        className="scrollbar-thin flex-1 overflow-y-auto"
      >
        {detections.length === 0 ? (
          // No fault cards to show. WHICH banner depends on whether anything was measured: a warm
          // green "clean rep" only when it was, and a neutral "could not be measured" note when it
          // was not — congratulating someone on form nothing measured is a claim, not encouragement.
          <div className="space-y-4 p-4">
            {measured ? (
              <div className="flex items-center gap-3 rounded-[16px] border border-[#d0f0dc] bg-[#e6f7ed]/85 p-4">
                <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-white">
                  <CheckCircle size={26} weight="fill" className="text-[#22c55e]" />
                </span>
                <p className="text-[13px] leading-relaxed text-[#1e2142]">
                  {t("feedback.noFaults", {
                    movement: movementLabel(t, analysis.movement ?? "Squat"),
                  })}
                </p>
              </div>
            ) : (
              <div className="flex items-center gap-3 rounded-[16px] border border-[#ece8ff] bg-white p-4">
                <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#f5f6fb]">
                  <Warning size={26} weight="fill" className="text-[#63709f]" />
                </span>
                <p className="text-[13px] leading-relaxed text-[#1e2142]">
                  {t("feedback.notMeasured")}
                </p>
              </div>
            )}
            <KnowledgeGraphWidget analysis={analysis} activeFaultId={activeFaultId} />
          </div>
        ) : (
          <div className="space-y-4 p-4">
            {detections.map((d, i) => (
              <motion.div
                key={i}
                initial={reduce ? false : { opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }}
              >
                <FaultCard
                  d={d}
                  retrieval={byFault.get(d.fault_id)}
                  active={currentTime >= d.start_time && currentTime <= d.end_time}
                  onSeek={onSeek}
                />
              </motion.div>
            ))}
            {/* Knowledge graph — the last item in the fault-card stack, same full width. */}
            <KnowledgeGraphWidget analysis={analysis} activeFaultId={activeFaultId} />
          </div>
        )}

        {/* Follow-up conversation — same thread, below the analysis. Only when chat is usable. */}
        {isWorking && (
          <div className="border-t border-[#ececf8] px-4 pb-4 pt-3">
            <div className="mb-2 flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#63709f]">
                {t("coach.followUp")}
              </span>
              <span className="h-px flex-1 bg-[#ececf8]" />
            </div>

            {messages.length === 0 && !loading ? (
              // Empty state: the intro line plus starter-suggestion chips that send on click. These
              // are the static entry points; per-answer dynamic follow-ups would need model support.
              <div>
                <p className="text-xs leading-relaxed text-[#59648f]">{t("chat.intro")}</p>
                <div className="mt-3 flex flex-col gap-2">
                  {["chat.suggestFix", "chat.suggestDrill", "chat.suggestWhy"].map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => void send(t(key))}
                      className={chipClass}
                    >
                      <span>{t(key)}</span>
                      <ArrowRight size={15} className="shrink-0 text-faint" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              // Editorial thread (design 06): the user's turn is a quiet right-aligned line, the
              // coach's answer is an airy labelled block tagged as grounded in the analysis.
              <div className="space-y-5">
                {messages.map((m, i) =>
                  m.role === "user" ? (
                    // The reference's user turn: a right-aligned tinted bubble with a clipped
                    // top-right corner.
                    <motion.div
                      key={i}
                      initial={reduce ? false : { opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                      className="flex justify-end"
                    >
                      <p className="max-w-[85%] rounded-[16px] rounded-tr-[4px] border border-[#e0e3ff] bg-[#eef0ff]/85 px-3.5 py-2.5 text-[12.5px] font-medium leading-relaxed text-[#3a3d5a]">
                        {m.content}
                      </p>
                    </motion.div>
                  ) : (
                    <motion.div
                      key={i}
                      initial={reduce ? false : { opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                    >
                      {coachTag}
                      {m.role === "assistant" && m.tools && m.tools.length > 0 && (
                        <ToolRunList runs={m.tools} />
                      )}
                      <div className="rounded-[16px] rounded-tl-[4px] border border-[#ece8ff] bg-[#f3f0ff]/85 px-3.5 py-3 text-[12.5px] leading-relaxed text-[#3a3d5a]">
                        <Markdown>{m.content}</Markdown>
                      </div>
                    </motion.div>
                  ),
                )}
                {/* The turn in flight shares ONE byline block with tool records and the streamed
                    answer — coachTag, then ToolRunList, then content — the same order as a committed
                    message above, so nothing shifts when the turn commits. Rendered as soon as EITHER
                    exists, so a tool record shows the coach's byline from the moment it lands, not
                    just once the first token streams (which is exactly when a record is most likely
                    to be the only thing on screen). */}
                {(toolRuns.length > 0 || streaming) && (
                  <div>
                    {coachTag}
                    {toolRuns.length > 0 && <ToolRunList runs={toolRuns} />}
                    {streaming && (
                      <div className="rounded-[16px] rounded-tl-[4px] border border-[#ece8ff] bg-[#f3f0ff]/85 px-3.5 py-3 text-[12.5px] leading-relaxed text-[#3a3d5a]">
                        <Markdown>{streaming}</Markdown>
                      </div>
                    )}
                  </div>
                )}
                {/* Lumen's dots only until either a tool record or the first token lands; then the
                    byline block above carries it. */}
                {loading && !streaming && toolRuns.length === 0 && (
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <LumenLoader variant="dots" />
                    {t("chat.thinking")}
                  </div>
                )}
                {/* Grounded next-question suggestions under the latest answer — same chips as the
                    opening starters, but generated from this answer. Hidden while a turn is in flight. */}
                {!loading && !streaming && followups.length > 0 && (
                  <div className="flex flex-col gap-2">
                    {followups.map((q, i) => (
                      <button key={i} type="button" onClick={() => void send(q)} className={chipClass}>
                        <span>{q}</span>
                        <ArrowRight size={15} className="shrink-0 text-faint" />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error + composer, pinned at the foot of the tray. */}
      {error && (
        <p className="px-4 pt-2 text-xs text-[#e05252]" role="alert">
          {error}
        </p>
      )}
      <div className="mt-auto shrink-0 p-3">{composer}</div>
    </div>
  );
}
