import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import Landing from "./landing/Landing";
import Login from "./pages/Login";
import History from "./pages/History";
import Settings from "./pages/Settings";
import AdminLayout from "./pages/admin/AdminLayout";
import AdminLogin from "./pages/admin/AdminLogin";
import AdminOverview from "./pages/admin/AdminOverview";
import AdminUsers from "./pages/admin/AdminUsers";
import AdminSettingsLlm from "./pages/admin/AdminSettingsLlm";
import AdminSettingsRag from "./pages/admin/AdminSettingsRag";
import AdminSettingsAnalyze from "./pages/admin/AdminSettingsAnalyze";
import Games from "./pages/Games";
import RequireAuth from "./components/RequireAuth";
// Lazily loaded so the ~800 kB MediaPipe bundle only downloads when a player opens a game route.
const SixSeven = lazy(() => import("./pages/SixSeven"));
const FruitNinja = lazy(() => import("./pages/FruitNinja"));
import { I18nProvider } from "./lib/i18n";
import { AuthProvider } from "./lib/auth";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
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
            <Route
              path="/history"
              element={
                <RequireAuth>
                  <History />
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
              <Route path="users" element={<AdminUsers />} />
              <Route path="settings/llm" element={<AdminSettingsLlm />} />
              <Route path="settings/rag" element={<AdminSettingsRag />} />
              <Route path="settings/analyze" element={<AdminSettingsAnalyze />} />
            </Route>
          </Routes>
        </I18nProvider>
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>
);
