export interface ChatModel {
  id: string;
  label: string;
}

// The chat models a user can pick in Settings (OpenRouter "vendor/model" slugs). Kept in sync with
// backend/app/settings.py ALLOWED_CHAT_MODELS — the backend validates the chosen id and falls back
// to its default for anything else, so a drift here only ever costs the offered option, never
// safety. Labels are brand names, not translated.
export const CHAT_MODELS: ChatModel[] = [
  { id: "deepseek/deepseek-v4-flash", label: "DeepSeek V4 Flash" },
  { id: "xiaomi/mimo-v2.5", label: "MiMo V2.5" },
  { id: "minimax/minimax-m3", label: "MiniMax M3" },
  { id: "tencent/hy3-preview", label: "Hy3 Preview" },
];

export const DEFAULT_CHAT_MODEL = CHAT_MODELS[0].id;

const STORAGE_KEY = "chat_model";

// The user's chosen model id, or the default when unset / not a recognized option.
export function getStoredModel(): string {
  const m = (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) || "";
  return CHAT_MODELS.some((x) => x.id === m) ? m : DEFAULT_CHAT_MODEL;
}

export function setStoredModel(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}
