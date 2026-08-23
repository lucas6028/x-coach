import { useLocation, useNavigate } from "react-router-dom";
import AppLayout from "../components/AppLayout";
import SettingsDialog from "../components/settings/SettingsDialog";

// The /settings route: the app shell with the settings popup open over it.
//
// Settings are a popup everywhere (the account menu opens the same component in place, without
// touching the URL), but the route survives because things point at it: RequireAuth gates it and
// AppRoutes.test pins it. On a phone there is no Settings tab; the header's account menu opens the
// same popup, and inside LINE the silent auto-login means that avatar is always present.
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
