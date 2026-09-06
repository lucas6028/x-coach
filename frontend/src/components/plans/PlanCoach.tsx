import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, PaperPlaneTilt, SignIn } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import {
  api,
  ChatError,
  type ChatMessage,
  type LiveToolRun,
  type Plan,
  type ToolRun,
} from "../../api";
import { getStoredModel } from "../../lib/model";
import { useAuth } from "../../lib/auth";
import { useI18n } from "../../lib/i18n";
import { LumenAvatar, LumenLoader } from "../LumenLoader";
import Markdown from "../Markdown";
import { ToolRunList } from "../ToolRunList";

interface Props {
  /** The plan being edited, or null for builder mode (Lumen may create one). */
  planId: string | null;
  /** A tool wrote the plan: here is the full fresh plan, so the caller can render it in place. */
  onPlan?: (plan: Plan) => void;
  /** A plan was created during this conversation. Fires at most once, and never when `planId`
   *  was already set at mount. */
  onCreated?: (planId: string) => void;
  /** Quick-prompt chips, shown only while the thread is empty. Clicking one sends its text. */
  suggestions?: string[];
  /** A thread to open with (wins over the persisted one). */
  initialMessages?: ChatMessage[];
  /** Lumen's opening line, rendered client-side — never sent to the API, never persisted. */
  greeting?: string;
  className?: string;
}

/**
 * Lumen's plan conversation: the chat surface shared by the builder page (`/plans/new`) and the
 * plan detail panel. Same three honest composer states as `CoachTray` — a working chat, a sign-in
 * invite, or the disabled affordance when the server has no LLM key — but grounded in a PLAN
 * rather than an analysis, so the interesting output is the plan a tool wrote, handed up through
 * `onPlan` for the caller to render beside this component.
 *
 * WHY THE PLAN ID LIVES IN A REF, not in props alone: in builder mode the id is born mid-turn, and
 * the parent will hand it straight back as `planId`. Keying the restore effect on the live prop
 * would then re-run it and wipe the conversation the user is having. So the id the component
 * *uses* is a ref (seeded from the prop, adopted from the stream), and the restore runs once
 * against the id present at mount.
 */
export default function PlanCoach({
  planId,
  onPlan,
  onCreated,
  suggestions,
  initialMessages,
  greeting,
  className = "",
}: Props) {
  const { t, lang } = useI18n();
  const { configured, user } = useAuth();
  const reduce = useReducedMotion();

  // Mount-time snapshots. Both are read once on purpose (see the note above): the restore is a
  // mount concern, and a later prop identity change must not restart it.
  const mountPlanId = useRef(planId).current;
  const seedMessages = useRef(initialMessages ?? []).current;

  const [messages, setMessages] = useState<ChatMessage[]>(seedMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState("");
  const [toolRuns, setToolRuns] = useState<LiveToolRun[]>([]);
  const [chatOnServer, setChatOnServer] = useState<boolean | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  // The plan this conversation is actually bound to. Starts as the prop, becomes the created plan's
  // id the moment the stream reveals it — mid-turn, so the same turn's persist writes to the right
  // key and the next turn scopes its tools to the new plan.
  const idRef = useRef<string | null>(planId);
  // Whether `onCreated` has been (or should never be) fired. A plan passed in at mount is not a
  // creation, so it counts as already announced.
  const announced = useRef(planId !== null);

  const isWorking = configured && !!user && chatOnServer !== false;
  const canSend = !!input.trim() && !loading;

  // Adopt an id the parent supplies later, but never overwrite one the stream already gave us.
  useEffect(() => {
    if (planId && !idRef.current) idRef.current = planId;
  }, [planId]);

  // Restore the saved thread. Precedence, decided once: a caller-supplied `initialMessages` wins
  // (it is the caller stating what this thread is), otherwise the persisted `plan:<id>` thread,
  // and the greeting shows only when both leave the thread empty. Best-effort — a failed or absent
  // fetch simply leaves the empty thread, and builder mode (no id yet) has nothing to restore.
  useEffect(() => {
    if (!isWorking || !mountPlanId || seedMessages.length) return;
    let active = true;
    api
      .getConversation(`plan:${mountPlanId}`)
      .then((c) => {
        if (!active || !c.messages?.length) return;
        setMessages(c.messages);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [isWorking, mountPlanId, seedMessages]);

  // Ask the backend once whether the LLM is configured (same optimism as CoachTray: only a
  // definitive `false` disables the composer; a failed health check stays hopeful).
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

  // Follow the newest turn, unless the user has scrolled up to read. (Guard scrollTo — jsdom.)
  useEffect(() => {
    const el = scrollRef.current;
    if (stickToBottom.current && el && typeof el.scrollTo === "function")
      el.scrollTo({ top: el.scrollHeight });
  }, [messages, loading, streaming, toolRuns]);

  function onThreadScroll() {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 80;
  }

  // Learn the plan's id from whichever frame reveals it first. `tool_done.plan` usually beats
  // `done.plan_id`, and taking the earlier one matters: a stream that dies after create_plan but
  // before `done` has still created a plan the user would otherwise have no way back to.
  function adoptPlanId(id: string) {
    if (!idRef.current) idRef.current = id;
    if (announced.current) return;
    announced.current = true;
    onCreated?.(id);
  }

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
    stickToBottom.current = true;
    let acc = "";
    let runs: LiveToolRun[] = [];
    let inbandError = "";
    // Only a real `done` frame counts as success — a stream that dies mid-flight resolves the same
    // way here, and committing that would persist a truncated (or empty) assistant turn.
    let finished = false;
    let donePlanId: string | undefined;
    try {
      await api.planChatStream(
        next,
        idRef.current,
        {
          onDelta: (tkn) => {
            acc += tkn;
            setStreaming(acc);
          },
          onTool: (run) => {
            runs = [...runs, run];
            setToolRuns(runs);
          },
          // Match on id, and only while pending: a replayed frame must not reopen a settled run.
          onToolDone: (id) => {
            runs = runs.map((r) => (r.id === id && r.pending ? { ...r, pending: false } : r));
            setToolRuns(runs);
          },
          // The plan really changed, so it is handed up immediately — and it is NOT rolled back if
          // the turn later fails. A tool that reported a write did write.
          onPlan: (plan) => {
            adoptPlanId(plan.id);
            onPlan?.(plan);
          },
          onReset: () => {
            acc = "";
            setStreaming("");
          },
          onDone: (info) => {
            finished = true;
            donePlanId = info.plan_id;
          },
          onError: (detail) => {
            inbandError = detail;
          },
        },
        { model: getStoredModel(), lang }
      );
      if (donePlanId) adoptPlanId(donePlanId);
      if (!finished) throw new ChatError("The coach connection ended unexpectedly.", 502);
      if (inbandError) throw new ChatError(inbandError, 502);
      // An empty answer cannot be committed: the backend requires non-empty `content`, so storing
      // one would make the thread un-postable forever. Treated as a failed turn, which it is.
      if (!acc.trim()) throw new ChatError("The coach returned an empty answer.", 502);
      // Rebuild each run from an allow-list rather than destructuring the live fields away, so a
      // field added to LiveToolRun later cannot ride along into stored jsonb.
      const committed: ToolRun[] = runs.map((r) => ({ name: r.name, query: r.query }));
      const thread: ChatMessage[] = [
        ...next,
        { role: "assistant", content: acc, ...(committed.length ? { tools: committed } : {}) },
      ];
      setMessages(thread);
      // Fire-and-forget: a save failure must never disrupt the conversation. Nothing to persist
      // until a plan exists — the key is the plan.
      const id = idRef.current;
      if (id) void api.putConversation(`plan:${id}`, thread).catch(() => undefined);
    } catch (e) {
      // Roll back the optimistic user turn and restore the text so a retry does not duplicate it.
      // The PLAN state is deliberately left alone (see onPlan above).
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

  // --- Composer: the same three honest states as the analysis tray -------------------------
  const disabledComposer = (
    <div className="relative" title={t("chat.title")}>
      <input
        disabled
        aria-label={t("plans.coach.inputLabel")}
        className="w-full cursor-not-allowed rounded-full border border-border-dark bg-content/[0.03] py-2.5 pl-4 pr-10 text-[13px] text-muted"
        placeholder={t("chat.placeholder")}
      />
      <PaperPlaneTilt size={16} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-faint" />
    </div>
  );

  const signInComposer = (
    <div className="flex items-center gap-3 rounded-2xl border border-border-dark bg-surface p-3">
      <LumenAvatar size={32} className="shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed text-muted">{t("plans.coach.signIn")}</p>
      <Link
        to="/login"
        className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-primary px-3 py-1.5 text-xs font-semibold text-primary-content transition-colors hover:bg-primary/90 active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
      >
        <SignIn size={14} weight="bold" />
        {t("account.signin")}
      </Link>
    </div>
  );

  const workingComposer = (
    <div className="flex items-center gap-2 rounded-full border border-border-dark bg-surface px-2 py-1.5 transition-colors focus-within:border-primary/50">
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
        aria-label={t("plans.coach.inputLabel")}
        className="min-w-0 flex-1 bg-transparent px-2.5 text-[13px] text-content placeholder-faint focus:outline-none disabled:opacity-60"
        placeholder={t("plans.coach.placeholder")}
      />
      <button
        type="button"
        onClick={() => void send()}
        disabled={!canSend}
        aria-label={t("chat.send")}
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-content shadow-accent transition enabled:hover:bg-primary/90 enabled:active:scale-95 disabled:bg-content/[0.08] disabled:text-faint disabled:shadow-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
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

  const coachTag = (
    <div className="mb-1.5 flex items-center gap-2">
      <LumenAvatar size={18} />
      <span className="text-[11px] font-semibold uppercase tracking-wide text-primary">
        {t("chat.coach")}
      </span>
    </div>
  );

  // Tinted, not outlined: the panel is already `bg-surface`, so a white bubble with a hairline
  // border on a white panel is a rectangle the eye has to hunt for. The fill does the grouping.
  const bubble =
    "rounded-2xl rounded-tl-md border border-transparent bg-content/[0.03] px-3.5 py-3 text-[13px] leading-relaxed text-content";

  // Quick prompts: the entry points into a conversation nobody has started yet, so they go away as
  // soon as there is one. They belong UNDER the greeting when there is one — chips pinned above the
  // composer read as a toolbar, while chips under Lumen's question read as answers to it.
  const chips = isWorking &&
    messages.length === 0 &&
    !loading &&
    suggestions &&
    suggestions.length > 0 && (
      <div className="flex flex-col gap-2">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => void send(s)}
            className="flex items-center justify-between gap-2 rounded-2xl border border-border-dark bg-surface px-3.5 py-2.5 text-left text-[12.5px] text-content transition-colors hover:border-primary/40 hover:bg-primary/[0.04] active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
          >
            <span>{s}</span>
            <ArrowRight size={15} className="shrink-0 text-faint" />
          </button>
        ))}
      </div>
    );

  // Tailwind emits utilities in a fixed order, so a class appended later does NOT win a conflict
  // (`rounded-2xl` and `shadow-card` both land after their `-none` counterparts). The shell drops
  // its own shape when the caller states a replacement — that is how the phone sheet hosts this
  // component edge-to-edge instead of as a card inside a card.
  const shell = [
    /\brounded-none\b/.test(className) ? null : "rounded-2xl",
    /\bborder-0\b/.test(className) ? null : "border border-border-dark",
    /\bshadow-none\b/.test(className) ? null : "shadow-card",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={`flex min-h-0 w-full flex-col overflow-hidden bg-surface ${shell} ${className}`}
    >
      <div className="flex shrink-0 items-center gap-2.5 border-b border-border-dark px-4 py-3">
        <LumenAvatar size={26} />
        <div className="min-w-0">
          <p className="font-display text-sm font-bold text-content">{t("chat.coach")}</p>
          <p className="truncate text-[11px] text-muted">{t("plans.coach.role")}</p>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={onThreadScroll}
        className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-4 py-4"
      >
        {/* Lumen's opening line. Rendered, never stored: a state entry would be uploaded on the
            next turn and written into the persisted thread, and it is neither the model's words
            nor the user's. It disappears the moment a real thread exists. */}
        {messages.length === 0 && greeting && (
          <div>
            {coachTag}
            <div className={bubble}>{greeting}</div>
            {chips && <div className="mt-3">{chips}</div>}
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <motion.div
              key={i}
              initial={reduce ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="flex justify-end"
            >
              <p className="max-w-[85%] rounded-2xl rounded-tr-md border border-primary/20 bg-primary/[0.06] px-3.5 py-2.5 text-[13px] font-medium leading-relaxed text-content">
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
              {m.tools && m.tools.length > 0 && <ToolRunList runs={m.tools} />}
              <div className={bubble}>
                <Markdown>{m.content}</Markdown>
              </div>
            </motion.div>
          )
        )}

        {/* The turn in flight: one byline block carrying the tool records and the streamed text,
            in the same order a committed message uses, so nothing shifts when it commits. */}
        {(toolRuns.length > 0 || streaming) && (
          <div>
            {coachTag}
            {toolRuns.length > 0 && <ToolRunList runs={toolRuns} />}
            {streaming && (
              <div className={bubble}>
                <Markdown>{streaming}</Markdown>
              </div>
            )}
          </div>
        )}
        {loading && !streaming && toolRuns.length === 0 && (
          <div className="flex items-center gap-2 text-xs text-muted">
            <LumenLoader variant="dots" />
            {t("chat.thinking")}
          </div>
        )}
      </div>

      {/* No greeting to hang them off (the detail panel opens straight into the plan), so the chips
          stay where they were: the last thing above the composer. */}
      {!greeting && chips && <div className="shrink-0 px-4 pb-1">{chips}</div>}

      {error && (
        <p className="px-4 pt-2 text-xs text-danger" role="alert">
          {error}
        </p>
      )}
      <div className="mt-auto shrink-0 px-4 pb-4 pt-2">{composer}</div>
    </div>
  );
}
