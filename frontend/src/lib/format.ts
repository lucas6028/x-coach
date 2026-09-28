export function fmtTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const MB = 1024 * 1024;
const GB = 1024 * MB;

/**
 * A storage size for display ("37 MB", "4.2 MB", "1.5 GB"). Binary units, like the backend's
 * `_as_mb`: the quota the upload 413 reports as "500 MB" must read "500 MB" in settings too, and
 * decimal units would show that same 500 MiB quota as "524 MB". One decimal only below 10, where
 * it still carries information.
 */
export function fmtStorage(bytes: number): string {
  if (!isFinite(bytes) || bytes < 0) bytes = 0;
  const [value, unit] = bytes >= GB ? [bytes / GB, "GB"] : [bytes / MB, "MB"];
  const shown = value >= 10 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${shown} ${unit}`;
}
