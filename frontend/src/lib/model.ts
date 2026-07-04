// The user's chat-model preference. The list of *selectable* models is server-driven (fetched from
// /api/health), so this module only persists the chosen id. An empty string means "follow the
// server default" — the client then sends no model and the backend uses OPENROUTER_MODEL.

const STORAGE_KEY = "chat_model";

// The stored model id, or "" when the user hasn't pinned one (follow the server default).
export function getStoredModel(): string {
  return (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) || "";
}

export function setStoredModel(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}
