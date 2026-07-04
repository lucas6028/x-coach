import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Brain, CheckCircle, CircleNotch, PaperPlaneTilt, SignIn } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { api, ChatError, type Analysis, type ChatMessage } from "../api";
import { buildChatContext } from "../lib/grounding";
import { retrievalByFault } from "../lib/retrieval";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";
import FaultCard from "./FaultCard";
import KnowledgeGraphWidget from "./KnowledgeGraphWidget";

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
// affordance when auth or the OpenRouter key is absent.
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

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // The in-progress assistant turn, accumulated from streamed deltas. Rendered as a live block
  // below the committed thread while loading; committed into `messages` on a clean `done`, or
  // discarded on an error (the optimistic user turn is rolled back alongside it).
  const [streaming, setStreaming] = useState("");
  // Whether the *server* has an OpenRouter key. Independent of Supabase auth. null = not yet
  // checked (assume available so the common path is instant).
  const [chatOnServer, setChatOnServer] = useState<boolean | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // A new analysis starts a fresh conversation — old turns no longer describe the current rep.
  useEffect(() => {
    setMessages([]);
    setError("");
  }, [analysis.video_id]);

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

  // Keep the newest turn in view as the conversation grows — but never on the initial render, so
  // the coach's analysis (top of the thread) is what the user sees first. (Guard scrollTo — jsdom.)
  // `streaming` is a dep so the view tracks the answer as tokens arrive.
  useEffect(() => {
    if (messages.length === 0) return;
    const el = scrollRef.current;
    if (el && typeof el.scrollTo === "function") el.scrollTo({ top: el.scrollHeight });
  }, [messages, loading, streaming]);

  const isWorking = configured && !!user && chatOnServer !== false;
  const canSend = !!input.trim() && !loading;

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
    let acc = "";
    let inbandError = "";
    try {
      await api.chatStream(next, buildChatContext(analysis), {
        onDelta: (tkn) => {
          acc += tkn;
          setStreaming(acc);
        },
        onDone: () => undefined,
        // An in-band error (OpenRouter connect/mid-stream/empty) isn't thrown — capture it and
        // rethrow below so success and failure share one rollback path.
        onError: (detail) => {
          inbandError = detail;
        },
      });
      if (inbandError) throw new ChatError(inbandError, 502);
      setMessages((m) => [...m, { role: "assistant", content: acc }]);
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
    }
  }

  // --- Composer (foot of the tray): one of three honest states -----------------------------
  const disabledComposer = (
    <div className="relative" title={t("chat.title")}>
      <input
        disabled
        className="w-full cursor-not-allowed rounded-2xl border border-border-dark bg-background py-2.5 pl-3 pr-10 text-sm text-muted placeholder-faint"
        placeholder={t("chat.placeholder")}
      />
      <PaperPlaneTilt size={18} className="absolute right-2 top-1/2 -translate-y-1/2 text-faint" />
    </div>
  );

  const signInComposer = (
    <div className="flex items-center gap-3 rounded-2xl border border-border-dark bg-background/60 p-3">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10">
        <Brain size={18} weight="duotone" className="text-primary" />
      </span>
      <p className="min-w-0 flex-1 text-xs leading-relaxed text-muted">{t("chat.signIn")}</p>
      <Link
        to="/login"
        className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-content transition-transform active:scale-[0.98]"
      >
        <SignIn size={15} weight="bold" />
        {t("account.signin")}
      </Link>
    </div>
  );

  const workingComposer = (
    // Pill composer (design 06). Voice + attach are the multimodal slots this design anticipates;
    // they slot in beside the input once those features exist, so no dead controls ship now.
    <div className="flex items-center gap-2 rounded-2xl border border-border-dark bg-background px-2 py-1.5 focus-within:ring-2 focus-within:ring-primary/40">
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
        className="min-w-0 flex-1 bg-transparent px-1.5 text-sm text-content placeholder-faint focus:outline-none disabled:opacity-60"
        placeholder={t("chat.placeholderActive")}
      />
      <button
        type="button"
        onClick={() => void send()}
        disabled={!canSend}
        aria-label={t("chat.send")}
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-content transition enabled:active:scale-95 disabled:bg-content/10 disabled:text-faint"
      >
        <PaperPlaneTilt size={16} weight={canSend ? "fill" : "regular"} />
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

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-background">
      {/* One header for the whole coaching conversation. */}
      <div className="flex items-center justify-between gap-2 border-b border-border-dark bg-surface-dark px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-content">
          <Brain size={17} weight="duotone" className="text-primary" />
          {t("chat.heading")}
        </h2>
        <span className="rounded-full border border-primary/20 bg-primary/10 px-2.5 py-1 font-mono text-[10px] text-primary">
          {t("feedback.badge")}
        </span>
      </div>

      {/* One scroll thread: the grounded analysis first, then the conversation. */}
      <div ref={scrollRef} className="scrollbar-thin flex-1 overflow-y-auto">
        {detections.length === 0 ? (
          // Clean rep — a compact, warm banner, with the KG card flush below it (same stack).
          <div className="space-y-4 p-4">
            <div className="flex items-center gap-3 rounded-xl border border-secondary/20 bg-secondary/[0.06] p-4">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-secondary/10">
                <CheckCircle size={26} weight="fill" className="text-secondary" />
              </span>
              <p className="text-sm leading-relaxed text-content">{t("feedback.noFaults")}</p>
            </div>
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
          <div className="border-t border-border-dark/60 px-4 pb-4 pt-3">
            <div className="mb-2 flex items-center gap-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-faint">
                {t("coach.followUp")}
              </span>
              <span className="h-px flex-1 bg-border-dark/60" />
            </div>

            {messages.length === 0 && !loading ? (
              // Empty state: the intro line plus starter-suggestion chips that send on click. These
              // are the static entry points; per-answer dynamic follow-ups would need model support.
              <div>
                <p className="text-xs leading-relaxed text-muted">{t("chat.intro")}</p>
                <div className="mt-3 flex flex-col gap-2">
                  {["chat.suggestFix", "chat.suggestDrill", "chat.suggestWhy"].map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => void send(t(key))}
                      className="flex items-center justify-between gap-2 rounded-2xl border border-border-dark bg-surface px-3.5 py-2.5 text-left text-[13px] text-content transition-colors hover:bg-content/[0.03]"
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
                    <motion.p
                      key={i}
                      initial={reduce ? false : { opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                      className="text-right text-sm leading-relaxed text-muted"
                    >
                      {m.content}
                    </motion.p>
                  ) : (
                    <motion.div
                      key={i}
                      initial={reduce ? false : { opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                    >
                      <div className="mb-1.5 flex items-center gap-2">
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
                      <p className="text-[15px] leading-relaxed text-content">{m.content}</p>
                    </motion.div>
                  ),
                )}
                {/* Live assistant turn: the streamed answer as tokens arrive, same styling as a
                    committed coach turn so it doesn't jump on completion. */}
                {streaming && (
                  <div>
                    <div className="mb-1.5 flex items-center gap-2">
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
                    <p className="text-[15px] leading-relaxed text-content">{streaming}</p>
                  </div>
                )}
                {/* Spinner only until the first token lands; then the streaming text carries it. */}
                {loading && !streaming && (
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <CircleNotch size={14} className="animate-spin" />
                    {t("chat.thinking")}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error + composer, pinned at the foot of the tray. */}
      {error && (
        <p className="px-4 pt-2 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
      <div className="border-t border-border-dark bg-surface-dark p-3">{composer}</div>
    </div>
  );
}
