import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Brain, CircleNotch, PaperPlaneTilt, SignIn } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { api, ChatError, type Analysis, type ChatMessage } from "../api";
import { buildChatContext } from "../lib/grounding";
import { useAuth } from "../lib/auth";
import { useI18n } from "../lib/i18n";

// The conversational-coaching panel. Grounded: it only ever sends the analysis's detected faults
// and retrieved cues (via buildChatContext) to the backend, which owns the system prompt. Three
// honest states:
//   • no auth backend configured  → disabled input ("coming with the LLM layer"), unchanged copy;
//   • configured but signed out    → an invitation to sign in (the endpoint is gated);
//   • signed in                    → a working, grounded chat.
export default function ChatInput({ analysis }: { analysis?: Analysis }) {
  const { t } = useI18n();
  const { configured, user } = useAuth();
  const reduce = useReducedMotion();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Whether the *server* has an OpenRouter key. Independent of Supabase auth: a signed-in user
  // whose backend lacks OPENROUTER_API_KEY must see the honest "coming soon" state, not a working
  // input that always 503s. null = not yet checked (assume available so the common path is instant).
  const [chatOnServer, setChatOnServer] = useState<boolean | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // A new analysis starts a fresh conversation — old turns no longer describe the current rep.
  useEffect(() => {
    setMessages([]);
    setError("");
  }, [analysis?.video_id]);

  // Ask the backend once (only when we'd otherwise show the working chat) whether the LLM is
  // configured, so a key-less server shows the disabled affordance rather than a broken input.
  // Only a *definitive* chat_configured=false disables the chat; a transient health-check failure
  // stays optimistic (assume available) — otherwise one network blip would wrongly lock a fully
  // configured chat into the "coming soon" state with no retry. A real misconfig then surfaces as
  // a 503 on send, which is recoverable, unlike the permanent fallback.
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

  // Keep the newest turn in view as the transcript grows. (Guard scrollTo — jsdom lacks it.)
  useEffect(() => {
    const el = scrollRef.current;
    if (el && typeof el.scrollTo === "function") el.scrollTo({ top: el.scrollHeight });
  }, [messages, loading]);

  // --- Honest fallbacks -------------------------------------------------------------------
  // Shown when the LLM layer isn't usable yet — either the auth backend isn't wired up, or the
  // server has no OpenRouter key. Deliberately a disabled "coming soon" affordance, never a live
  // input that would just error.
  const disabledFallback = (
    <div className="border-t border-border-dark bg-surface-dark p-3">
      <div className="relative" title={t("chat.title")}>
        <input
          disabled
          className="w-full cursor-not-allowed rounded-md border border-border-dark bg-background py-2.5 pl-3 pr-10 text-sm text-muted placeholder-faint"
          placeholder={t("chat.placeholder")}
        />
        <PaperPlaneTilt size={18} className="absolute right-2 top-1/2 -translate-y-1/2 text-faint" />
      </div>
    </div>
  );

  if (!configured) return disabledFallback;

  // Configured but signed out — the chat endpoint requires a session.
  if (!user) {
    return (
      <div className="border-t border-border-dark bg-surface-dark p-4">
        <div className="flex items-center gap-3 rounded-xl border border-border-dark bg-background/60 p-3">
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
      </div>
    );
  }

  // Signed in, but the server confirmed it has no OpenRouter key — honest fallback, not a live
  // input that would 503 on every send.
  if (chatOnServer === false) return disabledFallback;

  // --- Working chat -----------------------------------------------------------------------
  const canSend = !!input.trim() && !loading && !!analysis;

  async function send() {
    const text = input.trim();
    if (!text || loading || !analysis) return;
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    setError("");
    setLoading(true);
    try {
      const { reply } = await api.chat(next, buildChatContext(analysis));
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch (e) {
      // Roll back the optimistic user turn and restore the text, so a retry doesn't duplicate it
      // (in the transcript or in the history re-sent to the LLM).
      setMessages((m) => m.slice(0, -1));
      setInput(text);
      // Distinguish an expired session (401 → re-auth) from a transient LLM/backend outage.
      const expired = e instanceof ChatError && e.status === 401;
      setError(expired ? t("chat.sessionExpired") : t("chat.error"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col border-t border-border-dark bg-surface-dark">
      <div className="flex items-center justify-between gap-2 px-4 pt-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-content">
          <Brain size={16} weight="duotone" className="text-primary" />
          {t("chat.heading")}
        </h3>
        <span className="rounded-full border border-secondary/20 bg-secondary/10 px-2 py-0.5 font-mono text-[10px] text-secondary">
          {t("chat.grounded")}
        </span>
      </div>

      {messages.length === 0 ? (
        <p className="px-4 pb-3 pt-2 text-xs leading-relaxed text-muted">{t("chat.intro")}</p>
      ) : (
        <div ref={scrollRef} className="scrollbar-thin max-h-56 space-y-2.5 overflow-y-auto px-4 py-3">
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={reduce ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              <div
                className={`max-w-[85%] rounded-xl px-3 py-2 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-primary text-primary-content"
                    : "border border-border-dark bg-background text-content"
                }`}
              >
                <span className="mb-0.5 block text-[10px] font-medium uppercase tracking-wider opacity-60">
                  {m.role === "user" ? t("chat.you") : t("chat.coach")}
                </span>
                {m.content}
              </div>
            </motion.div>
          ))}
          {loading && (
            <div className="flex items-center gap-2 text-xs text-muted">
              <CircleNotch size={14} className="animate-spin" />
              {t("chat.thinking")}
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="px-4 pb-1 text-xs text-danger" role="alert">
          {error}
        </p>
      )}

      <div className="p-3">
        <div className="relative">
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
            className="w-full rounded-md border border-border-dark bg-background py-2.5 pl-3 pr-10 text-sm text-content placeholder-faint focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 disabled:opacity-60"
            placeholder={t("chat.placeholderActive")}
          />
          <button
            type="button"
            onClick={() => void send()}
            disabled={!canSend}
            aria-label={t("chat.send")}
            className="absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-primary transition-transform enabled:hover:bg-primary/10 enabled:active:scale-[0.95] disabled:text-faint"
          >
            <PaperPlaneTilt size={18} weight={canSend ? "fill" : "regular"} />
          </button>
        </div>
      </div>
    </div>
  );
}
