import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import Landing from "./landing/Landing";
import Login from "./pages/Login";
import History from "./pages/History";
import Settings from "./pages/Settings";
import RequireAuth from "./components/RequireAuth";
// Lazily loaded so the ~800 kB MediaPipe bundle only downloads when a player opens a game route.
const MemeBlast = lazy(() => import("./pages/MemeBlast"));
const SixSeven = lazy(() => import("./pages/SixSeven"));
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
              path="/blast"
              element={
                <Suspense fallback={null}>
                  <MemeBlast />
                </Suspense>
              }
            />
            <Route
              path="/67"
              element={
                <Suspense fallback={null}>
                  <SixSeven />
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
