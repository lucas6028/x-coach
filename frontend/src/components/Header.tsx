import { List } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { useI18n } from "../lib/i18n";

interface Props {
  onMenu?: () => void;
  /** The page's own header (breadcrumb, title, its controls), rendered beside the drawer button. */
  children?: ReactNode;
}

// The top row of the content card: the page's own header, and the drawer button that opens the
// rail below `lg`. Everything else has moved to the rail — the brand is its mark, every
// destination is an entry, and the account cluster (avatar, sign-in, the in-flight LINE login)
// sits at its foot. Language lives in the settings dialog behind that avatar; there is no theme
// picker any more, the app is light-only.
// Colours are the reference's own fixed hexes.
export default function Header({ onMenu, children }: Props) {
  const { t } = useI18n();

  return (
    // `items-start`: the page header can be three lines tall (breadcrumb, title, subtitle) and the
    // drawer button belongs at the top of that band, not centred against it.
    <header className="relative z-30 flex shrink-0 items-start gap-2 sm:gap-3">
      {/* Mobile only: open the off-canvas drawer. */}
      <button
        onClick={onMenu}
        aria-label={t("nav.show")}
        className="lg:hidden shrink-0 w-10 h-10 flex items-center justify-center rounded-2xl text-[#59648f] hover:bg-[#f5f6fb] hover:text-[#1e2142] transition-colors"
      >
        <List size={22} />
      </button>

      {/* The page's own header, or just a spacer when a page supplies none. */}
      <div className="min-w-0 flex-1">{children}</div>
    </header>
  );
}
