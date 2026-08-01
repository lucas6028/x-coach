import type { ReactNode } from "react";

// Shared layout atoms for the settings popup's right-hand pane, matched to the reference layout:
// a section heading, then hairline-separated rows with the label left and the control right.

/** A section heading inside a pane ("Profile", "Preferences", …). */
export function PaneTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-xl font-semibold text-content">{children}</h3>;
}

/** The rows under a PaneTitle. Owns the gap to the heading so panes don't restate it. */
export function PaneRows({ children }: { children: ReactNode }) {
  return <div className="mt-7">{children}</div>;
}

/**
 * One settings row. When there is no `hint` the label is a bare <span> rather than a wrapper —
 * that keeps the row element itself the nearest <div> around the label, which is what both the
 * flex layout and the tests rely on to pair a label with its control.
 */
export function SettingRow({
  label,
  hint,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-dark py-5 last:border-b-0">
      {hint ? (
        <div className="min-w-0 flex-1">
          <p className="text-[15px] text-content">{label}</p>
          {/* A div, not a <p>: callers pass block content here (status lines, links). */}
          <div className="mt-1 text-sm text-muted">{hint}</div>
        </div>
      ) : (
        <span className="text-[15px] text-content">{label}</span>
      )}
      {children != null && <div className="shrink-0">{children}</div>}
    </div>
  );
}
