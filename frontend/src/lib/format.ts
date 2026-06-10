export function fmtTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Severity -> a label and an opacity for timeline/cards.
export function severityLabel(sev: number): string {
  if (sev >= 0.75) return "High";
  if (sev >= 0.4) return "Moderate";
  return "Mild";
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
