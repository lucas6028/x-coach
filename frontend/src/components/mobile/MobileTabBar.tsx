import {
  Barbell,
  ClipboardText,
  ClockCounterClockwise,
  GameController,
  Plus,
} from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { useI18n } from "../../lib/i18n";

interface Props {
  /** Start a fresh session — the raised centre action. */
  onNewAnalysis: () => void;
}

/**
 * The mock's five-slot bottom bar, shared by the mobile web shell and the in-LINE shell so both
 * phones get the same navigation.
 *
 * Its four destinations mirror the desktop rail's, minus the studio: Movements, Plans, My
 * records, Games — the library is where you pick what to train, a plan is when you train it, and
 * the history is what came out. The centre slot is a raised action rather than a destination, so
 * the four tabs split two either side of it — which is also what keeps the FAB on the bar's
 * centre line. The studio has no tab of its own: that raised action IS the way into it.
 */
export default function MobileTabBar({ onNewAnalysis }: Props) {
  const { t } = useI18n();
  const { pathname } = useLocation();

  const cell = "flex flex-col items-center gap-1 py-2.5 text-[10px] font-medium transition-colors";
  const idle = "text-[#8b8fa8]";
  const on = "text-primary";

  const left = [
    {
      to: "/movements",
      label: t("nav.movements"),
      Icon: Barbell,
      // A movement's detail page belongs to the library tab. (The desktop rail matches only the
      // index; on a phone the bar is the whole navigation, so a detail page lighting nothing
      // would leave the user with no sense of where they are.)
      active: pathname === "/movements" || pathname.startsWith("/movements/"),
    },
    {
      to: "/plans",
      label: t("nav.plans"),
      Icon: ClipboardText,
      active: pathname === "/plans" || pathname.startsWith("/plans/"),
    },
  ];
  const right = [
    {
      to: "/history",
      label: t("nav.history"),
      Icon: ClockCounterClockwise,
      active: pathname === "/history",
    },
    {
      to: "/games",
      label: t("nav.games"),
      Icon: GameController,
      // The rail's own definition of "on Games": the hub and both game routes.
      active: pathname === "/games" || pathname === "/67" || pathname === "/ninja",
    },
  ];

  return (
    // The rail floats rather than sitting flush: `mx-2` and the bottom margin keep its rounded
    // edge off the screen edge, so the labels aren't reading against the bezel. The inset goes on
    // the MARGIN, not the padding — padding here would eat into the five columns, and "My
    // records" at 10px is already close to truncating.
    <nav
      aria-label={t("nav.tabBar")}
      className="glass-rail relative z-30 mx-3 mb-[max(env(safe-area-inset-bottom),0.75rem)] grid shrink-0 grid-cols-5 items-end rounded-[26px] px-1 pb-1.5 pt-2"
    >
      {left.map(({ to, label, Icon, active }) => (
        <Link
          key={to}
          to={to}
          aria-current={active ? "page" : undefined}
          className={`${cell} ${active ? on : idle}`}
        >
          <Icon size={21} weight={active ? "fill" : "duotone"} />
          <span className="max-w-full truncate px-0.5">{label}</span>
        </Link>
      ))}

      {/* The raised centre action. It sits proud of the bar, so the cell reserves its height and
          the button overhangs upward — the bar itself keeps a normal row height. */}
      <div className="flex justify-center">
        <button
          onClick={onNewAnalysis}
          aria-label={t("nav.newAnalysis")}
          title={t("nav.newAnalysis")}
          className="-mt-6 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-[#a48bff] to-[#7b5cff] text-white shadow-[0_10px_24px_rgba(123,92,255,0.45)] ring-4 ring-white/70 transition-transform active:scale-95"
        >
          <Plus size={26} weight="bold" />
        </button>
      </div>

      {right.map(({ to, label, Icon, active }) => (
        <Link
          key={to}
          to={to}
          aria-current={active ? "page" : undefined}
          className={`${cell} ${active ? on : idle}`}
        >
          <Icon size={21} weight={active ? "fill" : "duotone"} />
          <span className="max-w-full truncate px-0.5">{label}</span>
        </Link>
      ))}
    </nav>
  );
}
