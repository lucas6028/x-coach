// The nav rail's shared visual language, stated once. Both rails use it: the app's Sidebar
// (components/Sidebar.tsx) and the admin console's AdminNav (pages/admin/AdminLayout.tsx).
//
// It lives here rather than being exported from Sidebar.tsx because Sidebar owns behaviour the
// admin rail must NOT inherit — the New-analysis CTA, the LINE auto-login branch, and the app's
// own hardcoded destinations. Sharing the classes keeps the two rails looking identical without
// making one a special case of the other.

// One rail row: icon beside its label, the whole row a rounded target. Collapsed, the label is
// gone and the icon centres itself in the 76px strip — the row keeps its height either way, so
// toggling only moves things horizontally.
export const railCell = (open: boolean) =>
  `w-full flex items-center gap-3 min-h-[46px] px-3 rounded-[14px] transition-colors ${
    open ? "justify-start" : "justify-center"
  }`;

// `primary` is the reference's violet, so primary/10 over white lands on its #f3f0ff pill —
// using the token keeps the active state one definition instead of two. The lift under the
// selected row is the reference's own, and it is what separates "selected" from a plain hover.
export const RAIL_CELL_ACTIVE = "bg-primary/10 text-primary shadow-[0_14px_34px_rgba(112,70,255,0.14)]";
export const RAIL_CELL_IDLE = "text-[#59648f] hover:bg-[#f8f8fb] hover:text-[#1e2142]";
export const RAIL_LABEL = "text-sm leading-none tracking-tight truncate";

// The rail's single primary action, sitting at the top of the nav list above the destinations.
// In the app that is "New analysis"; in the admin console it is the way back out to the app.
export const RAIL_CTA =
  "bg-gradient-to-br from-[#a48bff] to-[#7b5cff] text-white shadow-[0_8px_20px_rgba(123,92,255,0.3)] hover:from-[#9a80ff] hover:to-[#6e4bff] active:scale-[0.98] mb-1";

// The rail's own frame: a floating frosted card, not a flush bordered column. `glass-rail` is
// defined at top level in index.css (not scoped to `.ms-shell`), so it resolves in either shell.
export const RAIL_FRAME = "glass-rail h-full shrink-0 flex flex-col rounded-[28px]";

// The app's brand mark. The artwork carries its own rounded violet plate, so the ring and the
// shadow have real edges to trace.
export function RailMark({ className = "" }: { className?: string }) {
  return (
    <img
      src="/icon.svg"
      alt=""
      className={`h-10 w-10 rounded-xl shadow-accent ring-1 ring-black/5 ${className}`}
    />
  );
}
