import { CaretLeft, UploadSimple } from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import { useI18n } from "../../lib/i18n";
import { useAuth } from "../../lib/auth";
import AccountMenu from "../AccountMenu";

interface Props {
  title: string;
  /** The mock's upload affordance — starts a fresh session. */
  onNewAnalysis: () => void;
}

/**
 * The mock's phone header: a round back button, the page title centred, and round actions on the
 * right. The mock's "…" overflow is the account menu here, which is what that slot would hold —
 * language, theme and sign-out already live behind that avatar.
 */
export default function MobileTopBar({ title, onNewAnalysis }: Props) {
  const { t } = useI18n();
  const { user } = useAuth();
  const navigate = useNavigate();

  const round =
    "flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-white text-[#2d2a5a] shadow-[0_4px_16px_rgba(110,90,180,0.12)] transition-transform active:scale-95";

  return (
    <header className="flex shrink-0 items-center gap-2 px-3 py-3">
      <button onClick={() => navigate(-1)} aria-label={t("mobile.back")} className={round}>
        <CaretLeft size={18} weight="bold" />
      </button>

      <h1 className="min-w-0 flex-1 truncate text-center font-display text-[17px] font-bold tracking-tight text-[#1e2142]">
        {title}
      </h1>

      <button onClick={onNewAnalysis} aria-label={t("nav.newAnalysis")} className={round}>
        <UploadSimple size={18} weight="bold" />
      </button>
      {/* Only once there is an account behind it; signed out, the tab bar's Settings is the way in. */}
      {user && <AccountMenu />}
    </header>
  );
}
