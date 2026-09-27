import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import { AnalysisJobProvider } from "../lib/analysisJob";

// `route` seeds the MemoryRouter so a component can be rendered at a URL that carries query
// params (the studio reads ?movement=). Defaults to "/" so every existing caller is unaffected.
//
// AnalysisJobProvider is mounted here too — same position as in main.tsx (inside AuthProvider and
// I18nProvider, wrapping the rendered UI) — so every caller of this helper gets a real job runner
// rather than the inert outside-the-provider default. It also mounts BackgroundAnalysisPill, which
// calls `useLocation()`; that is why this wrapper needs `route` to already be inside a Router,
// which it is (MemoryRouter is the outermost element here).
export function renderWithProviders(
  ui: ReactElement,
  options?: RenderOptions & { route?: string }
) {
  const { route = "/", ...renderOptions } = options ?? {};
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>
        <I18nProvider>
          <AnalysisJobProvider>{ui}</AnalysisJobProvider>
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>,
    renderOptions
  );
}
