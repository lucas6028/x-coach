import { ClockCounterClockwise, Folders, GearSix, Plus, VideoCamera } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { useI18n } from "../../lib/i18n";

interface Props {
  /** Start a fresh session — the raised centre action. */
  onNewAnalysis: () => void;
  /** Open the sample-clip picker. A tab rather than a route: the picker is a modal. */
  onOpenLibrary: () => void;
}

/**
 * The mock's five-slot bottom bar, shared by the mobile web shell and the in-LINE shell so both
 * phones get the same navigation.
 *
 * Its slots carry real destinations rather than the mock's Home / Progress / Library / Profile:
 * Analyse, My records, the new-analysis FAB, Library, Settings. Games loses the tab it had in the
 * old four-tab LIFF bar — the centre slot is a raised action in this design, so five slots buy
 * four destinations, and Games is the one of the five that is not part of the core analyse loop.
 * Its route still resolves and the desktop rail still links it.
 */
export default function MobileTabBar({ onNewAnalysis, onOpenLibrary }: Props) {
  const { t } = useI18n();
  const { pathname } = useLocation();

  const cell = "flex flex-col items-center gap-1 py-2.5 text-[10px] font-medium transition-colors";
  const idle = "text-[#8b8fa8]";
  const on = "text-primary";

  const left = [
    { to: "/app", label: t("nav.analyse"), Icon: VideoCamera, active: pathname === "/app" },
    {
      to: "/history",
      label: t("nav.history"),
      Icon: ClockCounterClockwise,
      active: pathname === "/history",
    },
  ];
  const right = [
    {
      to: "/settings",
      label: t("nav.settings"),
      Icon: GearSix,
      active: pathname === "/settings",
    },
  ];

  return (
    <nav
      aria-label={t("nav.tabBar")}
      className="glass-rail relative z-30 grid shrink-0 grid-cols-5 items-end rounded-[26px] px-1 pb-[max(env(safe-area-inset-bottom),0.25rem)] pt-1"
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

      <button onClick={onOpenLibrary} className={`${cell} ${idle}`}>
        <Folders size={21} weight="duotone" />
        <span className="max-w-full truncate px-0.5">{t("nav.library")}</span>
      </button>

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
