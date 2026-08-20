import type { ReactNode } from "react";

// The reference design's small dashboard panel: a near-white rounded card with a tinted
// icon bubble beside its title, and an optional action on the right of the header row.
export default function StudioCard({
  icon,
  title,
  action,
  children,
  className = "",
  /** Stagger index for the theme's `pop` entrance — 0, 1, 2 across the dashboard row. */
  index = 0,
}: {
  icon: ReactNode;
  title: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  index?: number;
}) {
  return (
    // `glass-panel` (translucent over the shell's gradient, no blur) + the theme's staggered
    // `pop` entrance. Both live in index.css; `pop` collapses under reduced motion.
    <section
      style={{ animationDelay: `${index * 90}ms` }}
      className={`glass-panel xc-pop rounded-[18px] p-4 ${className}`}
    >
      <div className="mb-3.5 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[#f0eaff] text-primary">
            {icon}
          </span>
          <h3 className="truncate text-[13px] font-bold text-[#1e2142]">{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
