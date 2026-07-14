import { type ReactNode } from "react";
import { CheckCircle, FloppyDisk, WarningCircle } from "@phosphor-icons/react";
import type { TFunc } from "../../lib/i18n";

// Shared form primitives for the three admin settings pages (LLM / RAG-KG / Analyze). Each page keeps
// its OWN narrow FormState + toPayload (sending only its group's keys); these bits are the common shell.

export type SaveState =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "done" }
  | { kind: "error"; message: string };

export const inputClass =
  "w-full rounded-xl border border-border-dark bg-content/[0.02] px-3 py-2 text-sm text-content outline-none transition-colors focus:border-primary/50 focus:bg-content/[0.04]";
export const textareaClass = `${inputClass} font-mono resize-y`;

// Split a newline/comma-separated raw string into a trimmed, non-empty list.
export const splitList = (raw: string): string[] =>
  raw
    .split(/[\n,]/)
    .map((x) => x.trim())
    .filter(Boolean);

// Parse a numeric form field. Returns null for a blank OR non-finite value (so the caller can reject
// it) instead of silently coercing to NaN (→ JSON null → stored default) or 0 (→ backend 422) the way
// bare Number() does.
export const parseNumber = (raw: string): number | null => {
  if (raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
};

// Validate + parse a group of REQUIRED numeric fields in one pass. Returns the parsed numbers keyed
// the same, or null if ANY field is blank/non-numeric — letting the caller show one honest error and
// skip the submit rather than corrupting the group with NaN/0.
export function parseRequiredNumbers<K extends string>(
  fields: Record<K, string>
): Record<K, number> | null {
  const out = {} as Record<K, number>;
  for (const key of Object.keys(fields) as K[]) {
    const n = parseNumber(fields[key]);
    if (n === null) return null;
    out[key] = n;
  }
  return out;
}

export function defaultHint(t: TFunc, value: string | number): string {
  return t("admin.settings.defaultLabel", { value: String(value) });
}

export function SettingsCard({
  icon,
  title,
  desc,
  children,
}: {
  icon: ReactNode;
  title: string;
  desc: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-border-dark bg-surface-dark p-5">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-sm font-semibold text-content">{title}</h2>
      </div>
      <p className="mt-1 text-xs text-muted">{desc}</p>
      <div className="mt-4 space-y-4">{children}</div>
    </div>
  );
}

export function Field({
  id,
  label,
  hint,
  hintDanger,
  children,
}: {
  /** Omit for a read-only display (no associated form control to point `htmlFor` at). */
  id?: string;
  label: string;
  hint?: string;
  hintDanger?: boolean;
  children: ReactNode;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-xs font-medium text-faint">
        {label}
      </label>
      <div className="mt-1.5">{children}</div>
      {hint && <p className={`mt-1 text-xs ${hintDanger ? "text-danger/80" : "text-faint"}`}>{hint}</p>}
    </div>
  );
}

// The save button + inline success/error state, shared across the settings pages.
export function SaveBar({ t, save, onSave }: { t: TFunc; save: SaveState; onSave: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        onClick={onSave}
        disabled={save.kind === "saving"}
        className="inline-flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary/90 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
      >
        <FloppyDisk size={16} weight="fill" />
        {save.kind === "saving" ? t("admin.settings.saving") : t("admin.settings.save")}
      </button>
      {save.kind === "done" && (
        <p className="flex items-center gap-1.5 text-sm text-secondary">
          <CheckCircle size={16} weight="fill" />
          {t("admin.settings.saved")}
        </p>
      )}
      {save.kind === "error" && (
        <p className="flex items-center gap-1.5 text-sm text-danger">
          <WarningCircle size={16} weight="fill" />
          {save.message}
        </p>
      )}
    </div>
  );
}

// Loading + load-error states shared by the settings pages.
export function SettingsLoading({ t }: { t: TFunc }) {
  return <p className="text-sm text-muted">{t("admin.settings.loading")}</p>;
}

export function SettingsLoadError({ t }: { t: TFunc }) {
  return (
    <div className="flex items-start gap-2.5 rounded-2xl border border-danger/30 bg-danger/[0.06] p-4 text-sm text-danger">
      <WarningCircle size={18} className="shrink-0" />
      <p className="font-medium">{t("admin.settings.loadError")}</p>
    </div>
  );
}
