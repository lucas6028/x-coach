import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import AppRoutes from "./AppRoutes";
import { I18nProvider } from "./lib/i18n";
import { AuthProvider } from "./lib/auth";
import { LiffProvider } from "./lib/liffContext";
import { initLiff } from "./lib/liff";
import "./index.css";

// Kick off LIFF init at bootstrap (fire-and-forget, cached) so the SDK load + LINE
// callback token-exchange (~1-1.5s) overlaps first paint instead of running only once the
// auto-login effect fires after getSession. On a LINE redirect-return this is squarely on
// the login critical path; a no-op when VITE_LIFF_ID is unset. AuthProvider's effect
// awaits the very same cached promise, so nothing runs twice.
void initLiff();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <AppRoutes />
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </BrowserRouter>
  </React.StrictMode>
);
