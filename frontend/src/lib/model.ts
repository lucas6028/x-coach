export interface ChatModel {
  id: string;
  label: string;
}

// The models a hosted user can pick in Settings (OpenRouter "vendor/model" slugs). Kept in sync with
// backend/app/settings.py ALLOWED_CHAT_MODELS — the backend validates the chosen id.
export const CHAT_MODELS: ChatModel[] = [
  { id: "deepseek/deepseek-v4-flash", label: "DeepSeek V4 Flash" },
  { id: "xiaomi/mimo-v2.5", label: "MiMo V2.5" },
  { id: "minimax/minimax-m3", label: "MiniMax M3" },
  { id: "tencent/hy3-preview", label: "Hy3 Preview" },
];

// "" = use the server's configured OPENROUTER_MODEL. Picking this sends no model, so a self-hoster
// who sets OPENROUTER_MODEL to ANY model gets it — without touching the curated list above. This is
// the default (like the theme toggle's "system"), so a fresh user runs whatever the operator set.
export const SERVER_DEFAULT = "";

const STORAGE_KEY = "chat_model";

// The user's chosen model id, or "" (server default) when unset or not one of the offered models.
export function getStoredModel(): string {
  const m = (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) || "";
  return CHAT_MODELS.some((x) => x.id === m) ? m : SERVER_DEFAULT;
}

export function setStoredModel(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}
