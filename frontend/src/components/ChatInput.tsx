import { PaperPlaneTilt } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";

// Placeholder for the follow-up chat — the LLM reasoning layer is deferred to a later iteration.
export default function ChatInput() {
  const { t } = useI18n();
  return (
    <div className="p-3 border-t border-border-dark bg-surface-dark">
      <div className="relative" title={t("chat.title")}>
        <input
          disabled
          className="w-full bg-background border border-border-dark rounded-md py-2.5 pl-3 pr-10 text-sm text-muted placeholder-faint cursor-not-allowed"
          placeholder={t("chat.placeholder")}
        />
        <PaperPlaneTilt
          size={18}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-faint"
        />
      </div>
    </div>
  );
}
