import { useEffect, useRef, useState } from "react";
import { CaretDown, GearSix, SignOut } from "@phosphor-icons/react";
import { useLocation, useNavigate } from "react-router-dom";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import { avatarUrl, displayName, initial } from "../lib/profile";
import SettingsDialog from "./settings/SettingsDialog";

interface Props {
  /** Omitted: the compact top-row trigger — avatar alone, menu opening downward from the right.
   *  Set to place the control in the sidebar's footer instead, where it is a full-width row and
   *  the menu has to open UPWARD (there is nothing below it). "closed" is the 76px rail, which
   *  has room for the avatar but not the name. */
  rail?: "open" | "closed";
}

// Account control as an avatar-triggered dropdown. The trigger shows the user's
// profile image (or an initial fallback); the menu opens settings and signs out.
export default function AccountMenu({ rail }: Props) {
  const { t } = useI18n();
  const { user, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [imgError, setImgError] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!user) return null;

  const url = imgError ? null : avatarUrl(user);
  const name = displayName(user);

  const trigger = rail
    ? `w-full rounded-[14px] px-2 py-2 text-muted transition-colors hover:bg-content/5 hover:text-content ${
        rail === "open" ? "flex items-center gap-2.5" : "flex justify-center"
      }`
    : "flex h-10 items-center gap-1 rounded-lg pl-1 pr-1.5 text-muted transition-colors hover:bg-content/5 hover:text-content";

  // Up from the rail's footer, down from a top row. `left-0` rather than `right-0` in the rail:
  // the menu is narrower than the open rail, so anchoring it left keeps it over the rail instead
  // of hanging across the content card.
  const menuPos = rail ? "bottom-full left-0 mb-1.5" : "right-0 mt-1.5";

  return (
    <div ref={ref} className={rail ? "relative w-full" : "relative"}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("account.menu")}
        title={name}
        className={trigger}
      >
        {url ? (
          <img
            src={url}
            alt=""
            referrerPolicy="no-referrer"
            onError={() => setImgError(true)}
            className="h-7 w-7 shrink-0 rounded-full object-cover ring-1 ring-border-dark"
          />
        ) : (
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary ring-1 ring-border-dark">
            {initial(user)}
          </span>
        )}
        {rail === "open" && (
          <span className="min-w-0 flex-1 truncate text-left text-sm font-medium">{name}</span>
        )}
        {rail !== "closed" && (
          <CaretDown
            size={12}
            weight="bold"
            // In the rail the menu comes out of the top, so the resting caret points up and the
            // open state rotates it back down — the mirror of the top row's behaviour.
            className={`shrink-0 transition-transform ${
              rail ? (open ? "rotate-0" : "rotate-180") : open ? "rotate-180" : ""
            }`}
          />
        )}
      </button>
      {open && (
        <div
          role="menu"
          className={`absolute z-50 min-w-[10rem] overflow-hidden rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20 ${menuPos}`}
        >
          {/* Opens the popup in place, leaving the URL alone — the /settings route renders the
              same component, so on that route the popup is already up and this is a no-op. */}
          <button
            role="menuitem"
            onClick={() => {
              setOpen(false);
              if (location.pathname !== "/settings") setSettingsOpen(true);
            }}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <GearSix size={18} />
            <span className="flex-1 text-left">{t("account.settings")}</span>
          </button>
          <div className="my-1 h-px bg-border-dark" />
          <button
            role="menuitem"
            onClick={() => {
              setOpen(false);
              void signOut().then(() => {
                navigate(location.pathname.startsWith("/admin") ? "/admin/login" : "/login");
              });
            }}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <SignOut size={18} />
            <span className="flex-1 text-left">{t("account.signout")}</span>
          </button>
        </div>
      )}
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
    </div>
  );
}
