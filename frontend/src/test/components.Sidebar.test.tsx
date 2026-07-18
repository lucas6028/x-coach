import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import userEvent from "@testing-library/user-event";
import Sidebar from "../components/Sidebar";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import { renderWithProviders } from "./renderWithProviders";

describe("Sidebar — desktop rail (open)", () => {
  const renderRail = () =>
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={vi.fn()} />
    );

  it("shows the nav labels when open", () => {
    renderRail();
    expect(screen.getByText("Analyse")).toBeInTheDocument();
    expect(screen.getByText("Library")).toBeInTheDocument();
  });

  it("shows the version and tagline when open", () => {
    renderRail();
    expect(screen.getByText("Prototype v0.1")).toBeInTheDocument();
    expect(screen.getByText("Pose · Rules · GraphRAG")).toBeInTheDocument();
  });

  it("omits the brand and collapse toggle — those live in the top navbar", () => {
    renderRail();
    expect(screen.queryByText("X-Coach")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /navigation/i })).not.toBeInTheDocument();
  });

  it("calls onOpenLibrary when the Library button is clicked", async () => {
    const user = userEvent.setup();
    const onOpenLibrary = vi.fn();
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onOpenLibrary={onOpenLibrary} onNewAnalysis={vi.fn()} />
    );
    await user.click(screen.getByRole("button", { name: /Library/i }));
    expect(onOpenLibrary).toHaveBeenCalledOnce();
  });

  it("calls onNewAnalysis when the New analysis button is clicked", async () => {
    const user = userEvent.setup();
    const onNewAnalysis = vi.fn();
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={onNewAnalysis} />
    );
    await user.click(screen.getByRole("button", { name: /New analysis/i }));
    expect(onNewAnalysis).toHaveBeenCalledOnce();
  });
});

describe("Sidebar — games hub active state", () => {
  // The single Games entry highlights on the hub route AND on either individual game route it
  // links into; render at each so every branch of the onGames check is exercised.
  const renderAt = (path: string) =>
    render(
      <MemoryRouter initialEntries={[path]}>
        <AuthProvider>
          <I18nProvider>
            <Sidebar
              open
              width={240}
              animate={false}
              onOpenLibrary={vi.fn()}
              onNewAnalysis={vi.fn()}
            />
          </I18nProvider>
        </AuthProvider>
      </MemoryRouter>
    );

  it.each(["/games", "/67", "/ninja"])("highlights the Games entry on %s", (path) => {
    renderAt(path);
    const link = screen.getByRole("link", { name: /Games/i });
    expect(link.className).toContain("text-primary");
  });

  it("does not highlight the Games entry on an unrelated route", () => {
    renderAt("/history");
    const link = screen.getByRole("link", { name: /Games/i });
    expect(link.className).not.toContain("text-primary");
  });
});

describe("Sidebar — collapsed rail", () => {
  it("hides the nav labels when collapsed", () => {
    renderWithProviders(
      <Sidebar open={false} width={64} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={vi.fn()} />
    );
    expect(screen.queryByText("Analyse")).not.toBeInTheDocument();
    expect(screen.queryByText("Prototype v0.1")).not.toBeInTheDocument();
  });
});

describe("Sidebar — mobile drawer (onClose)", () => {
  it("shows the brand when rendered as a drawer", () => {
    renderWithProviders(
      <Sidebar open width={270} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.getByText("X-Coach")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderWithProviders(
      <Sidebar open width={270} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={vi.fn()} onClose={onClose} />
    );
    await user.click(screen.getByRole("button", { name: /Hide navigation/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
