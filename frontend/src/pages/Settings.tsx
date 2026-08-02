import { useLocation, useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import SettingsDialog from "../components/settings/SettingsDialog";

// The /settings route: the app shell with the settings popup open over it.
//
// Settings are a popup everywhere (the account menu opens the same component in place, without
// touching the URL), but the route survives because things point at it: RequireAuth gates it,
// AppRoutes.test pins it, and the LINE in-app shell has a bottom tab for it — inside LINE this is
// the ONLY way to reach the language and theme controls, since there is no navbar there.
export default function Settings() {
  const navigate = useNavigate();
  const location = useLocation();

  // Closing should return you where you came from. React Router stamps `key: "default"` on the
  // first entry of a history stack — a deep link or a fresh tab, where there is nothing to go
  // back to — so those land in the studio instead of leaving the browser.
  const close = () => {
    if (location.key === "default") navigate("/app");
    else navigate(-1);
  };

  return (
    <AppLayout>
      <SettingsDialog onClose={close} />
    </AppLayout>
  );
}
