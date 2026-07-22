import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import Landing from "./landing/Landing";
import Login from "./pages/Login";
import History from "./pages/History";
import Movements from "./pages/Movements";
import Settings from "./pages/Settings";
import AdminLayout from "./pages/admin/AdminLayout";
import AdminLogin from "./pages/admin/AdminLogin";
import AdminOverview from "./pages/admin/AdminOverview";
import AdminLine from "./pages/admin/AdminLine";
import AdminUsers from "./pages/admin/AdminUsers";
import AdminSettingsLlm from "./pages/admin/AdminSettingsLlm";
import AdminSettingsRag from "./pages/admin/AdminSettingsRag";
import AdminSettingsAnalyze from "./pages/admin/AdminSettingsAnalyze";
import Games from "./pages/Games";
import LiffDiag from "./pages/LiffDiag";
import RequireAuth from "./components/RequireAuth";
// Lazily loaded so the ~800 kB MediaPipe bundle only downloads when a player opens a game route.
const SixSeven = lazy(() => import("./pages/SixSeven"));
const FruitNinja = lazy(() => import("./pages/FruitNinja"));
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
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/app" element={<App />} />
              <Route path="/games" element={<Games />} />
              <Route
                path="/67"
                element={
                  <Suspense fallback={null}>
                    <SixSeven />
                  </Suspense>
                }
              />
              <Route
                path="/ninja"
                element={
                  <Suspense fallback={null}>
                    <FruitNinja />
                  </Suspense>
                }
              />
              <Route path="/login" element={<Login />} />
              {/* LIFF device check (Phase 0 of the LINE rollout): open inside LINE on a phone. */}
              <Route path="/liff/diag" element={<LiffDiag />} />
              <Route
                path="/history"
                element={
                  <RequireAuth>
                    <History />
                  </RequireAuth>
                }
              />
              <Route
                path="/movements"
                element={
                  <RequireAuth>
                    <Movements />
                  </RequireAuth>
                }
              />
              <Route
                path="/settings"
                element={
                  <RequireAuth>
                    <Settings />
                  </RequireAuth>
                }
              />
              <Route path="/admin/login" element={<AdminLogin />} />
              <Route
                path="/admin"
                element={
                  <RequireAuth redirectTo="/admin/login">
                    <AdminLayout />
                  </RequireAuth>
                }
              >
                <Route index element={<AdminOverview />} />
                <Route path="line" element={<AdminLine />} />
                <Route path="users" element={<AdminUsers />} />
                <Route path="settings/llm" element={<AdminSettingsLlm />} />
                <Route path="settings/rag" element={<AdminSettingsRag />} />
                <Route path="settings/analyze" element={<AdminSettingsAnalyze />} />
              </Route>
            </Routes>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </BrowserRouter>
  </React.StrictMode>
);
