import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import App from "./App";
import Landing from "./landing/Landing";
import Login from "./pages/Login";
import History from "./pages/History";
import Movements from "./pages/Movements";
import PlanDetail from "./pages/PlanDetail";
import Plans from "./pages/Plans";
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
const WebSlinger = lazy(() => import("./pages/WebSlinger"));

// The route table, extracted from main.tsx so WHICH ROUTES ARE PUBLIC is testable rather than
// being untested configuration. main.tsx still owns bootstrap (createRoot, providers, LIFF init);
// this owns only the mapping from path to element.
//
// What is gated and what is not: `RequireAuth` guards routes that show a specific user's own data
// (/history, /settings) or privileged tooling (/admin). Everything else is open, including /app —
// the anonymous public demo, where an upload is analysed but nothing is saved.
export default function AppRoutes() {
  return (
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
      <Route
        path="/web-slinger"
        element={
          <Suspense fallback={null}>
            <WebSlinger />
          </Suspense>
        }
      />
      <Route path="/login" element={<Login />} />
      {/* LIFF device check (Phase 0 of the LINE rollout): open inside LINE on a phone. */}
      <Route path="/liff/diag" element={<LiffDiag />} />
      {/* PUBLIC on purpose. This is a catalog of the 16 movement names, their body-region
          grouping, and which are analyzable — the same information GET /api/movements serves
          unauthenticated and the landing page markets openly. Nothing here is user-specific.
          It was gated only because it inherited the wrapper from the /explore knowledge-graph
          browser it replaced (commit 9058abe9 swapped the path and component and left the guard),
          which left a signed-out visitor seeing "Movements" in the sidebar, clicking it, and
          bouncing to /login — for a menu whose live cards hand off to /app, which is itself
          public. Pinned by src/test/AppRoutes.test.tsx. */}
      <Route path="/movements" element={<Movements />} />
      {/* GATED, unlike /movements above. A plan is one user's own data — its name, its exercises
          and its progress — so both the list and the detail sit behind RequireAuth alongside
          /history, and the API answers 401 without a session regardless. */}
      <Route
        path="/plans"
        element={
          <RequireAuth>
            <Plans />
          </RequireAuth>
        }
      />
      <Route
        path="/plans/:planId"
        element={
          <RequireAuth>
            <PlanDetail />
          </RequireAuth>
        }
      />
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
        <Route path="line" element={<AdminLine />} />
        <Route path="users" element={<AdminUsers />} />
        <Route path="settings/llm" element={<AdminSettingsLlm />} />
        <Route path="settings/rag" element={<AdminSettingsRag />} />
        <Route path="settings/analyze" element={<AdminSettingsAnalyze />} />
      </Route>
    </Routes>
  );
}
