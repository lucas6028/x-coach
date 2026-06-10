// Placeholder for the follow-up chat — the LLM reasoning layer is deferred to a later iteration.
export default function ChatInput() {
  return (
    <div className="p-3 border-t border-border-dark bg-surface-dark">
      <div className="relative" title="Conversational coaching arrives with the LLM layer.">
        <input
          disabled
          className="w-full bg-[#111214] border border-border-dark rounded-md py-2.5 pl-3 pr-10 text-sm text-gray-500 placeholder-gray-600 cursor-not-allowed"
          placeholder="Ask the AI Coach… (LLM layer coming soon)"
        />
        <span className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 text-gray-700 text-lg">
          send
        </span>
      </div>
    </div>
  );
}
