import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

export function renderWithProviders(ui: ReactElement, options?: RenderOptions) {
  return render(
    <MemoryRouter>
      <I18nProvider>{ui}</I18nProvider>
    </MemoryRouter>,
    options
  );
}
