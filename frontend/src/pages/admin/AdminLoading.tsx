import { useI18n } from "../../lib/i18n";

// The admin console's one loading state, shared by the shell's access check and by every page that
// waits on a read (overview, LINE, the settings forms). It replaces the bare line of grey text
// those used to render: a static sentence gives no signal that anything is actually in flight, so
// a slow LINE call read as a page that had simply stopped.
//
// The label stays in the DOM as real text (not `sr-only`) so it is both visible and assertable;
// the ring and the dots are decorative and hidden from assistive tech, which gets the label alone.
export default function AdminLoading({ labelKey = "admin.loading" }: { labelKey?: string }) {
  const { t } = useI18n();
  return (
    <div role="status" className="flex flex-col items-center justify-center gap-4 py-20">
      {/* Two counter-rotating arcs on a soft violet track. `motion-reduce` drops the spin rather
          than the element — a reduced-motion user still sees the indicator, just not moving. */}
      <span className="relative flex h-14 w-14 items-center justify-center" aria-hidden="true">
        <span className="absolute inset-0 rounded-full border-4 border-primary/10" />
        <span className="absolute inset-0 animate-spin rounded-full border-4 border-transparent border-t-primary motion-reduce:animate-none" />
        <span
          className="absolute inset-[7px] animate-spin rounded-full border-[3px] border-transparent border-b-primary/40 motion-reduce:animate-none"
          style={{ animationDirection: "reverse", animationDuration: "1.4s" }}
        />
      </span>

      {/* No animated dots beside this: every one of these labels already ends in an ellipsis
          ("Checking your access…"), and a second set of dots reads as a typo rather than motion. */}
      <p className="text-sm font-medium text-muted">{t(labelKey)}</p>
    </div>
  );
}
