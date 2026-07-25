import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

// `route` seeds the MemoryRouter so a component can be rendered at a URL that carries query
// params (the studio reads ?movement=). Defaults to "/" so every existing caller is unaffected.
export function renderWithProviders(
  ui: ReactElement,
  options?: RenderOptions & { route?: string }
) {
  const { route = "/", ...renderOptions } = options ?? {};
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>
        <I18nProvider>{ui}</I18nProvider>
      </AuthProvider>
    </MemoryRouter>,
    renderOptions
  );
}
