import { useEffect, useRef, useState } from "react";
import { CaretDown, GearSix, SignOut } from "@phosphor-icons/react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useI18n } from "../lib/i18n";
import { useAuth } from "../lib/auth";
import { avatarUrl, displayName, initial } from "../lib/profile";

// Account control as an avatar-triggered dropdown. The trigger shows the user's
// profile image (or an initial fallback); the menu links to settings and signs out.
export default function AccountMenu() {
  const { t } = useI18n();
  const { user, signOut } = useAuth();
  const [open, setOpen] = useState(false);
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

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("account.menu")}
        title={name}
        className="flex h-10 items-center gap-1 rounded-lg pl-1 pr-1.5 text-muted transition-colors hover:bg-content/5 hover:text-content"
      >
        {url ? (
          <img
            src={url}
            alt=""
            referrerPolicy="no-referrer"
            onError={() => setImgError(true)}
            className="h-7 w-7 rounded-full object-cover ring-1 ring-border-dark"
          />
        ) : (
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary ring-1 ring-border-dark">
            {initial(user)}
          </span>
        )}
        <CaretDown size={12} weight="bold" className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 z-50 mt-1.5 min-w-[10rem] overflow-hidden rounded-xl border border-border-dark bg-surface-dark p-1 shadow-lg shadow-black/20"
        >
          <Link
            to="/settings"
            role="menuitem"
            onClick={() => setOpen(false)}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-muted transition-colors hover:bg-content/5 hover:text-content"
          >
            <GearSix size={18} />
            <span className="flex-1 text-left">{t("account.settings")}</span>
          </Link>
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
    </div>
  );
}
