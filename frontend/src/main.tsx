import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import Landing from "./landing/Landing";
import Login from "./pages/Login";
import History from "./pages/History";
import Settings from "./pages/Settings";
import RequireAuth from "./components/RequireAuth";
// Lazily loaded so the ~800 kB MediaPipe bundle only downloads when players open /duel.
const PoseDuel = lazy(() => import("./pages/PoseDuel"));
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
            <Route
              path="/duel"
              element={
                <Suspense fallback={null}>
                  <PoseDuel />
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
          </Routes>
        </I18nProvider>
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>
);
