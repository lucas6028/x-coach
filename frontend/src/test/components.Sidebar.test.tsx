import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import userEvent from "@testing-library/user-event";
import Sidebar from "../components/Sidebar";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import { renderWithProviders } from "./renderWithProviders";

describe("Sidebar — desktop rail (open)", () => {
  const renderRail = (onToggle = vi.fn()) =>
    renderWithProviders(
      <Sidebar
        open={true}
        width={236}
        animate
        onToggle={onToggle}
        onOpenLibrary={vi.fn()}
        onNewAnalysis={vi.fn()}
      />
    );

  it("shows the nav labels when open", () => {
    renderRail();
    expect(screen.getByText("Analyse")).toBeInTheDocument();
    expect(screen.getByText("Library")).toBeInTheDocument();
  });

  // The footer used to carry a version tag and a tagline ("Prototype v0.1" / "Pose · Rules ·
  // GraphRAG"). Both were removed; this pins the removal so the footer can't creep back.
  it("carries no version tag or tagline footer", () => {
    renderRail();
    expect(screen.queryByText("Prototype v0.1")).not.toBeInTheDocument();
    expect(screen.queryByText("Pose · Rules · GraphRAG")).not.toBeInTheDocument();
  });

  // This used to pin the opposite — the rail showed the mark alone. It now carries the same
  // mark + wordmark lockup the landing nav does, so the shell names itself the way the page
  // that links into it does.
  it("shows the brand wordmark beside the mark", () => {
    renderRail();
    expect(screen.getByText("X-Coach")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "X-Coach" })).toBeInTheDocument();
  });

  // The rail owns its width toggle again (the top row still carries none — see
  // components.Header.test.tsx). Its name must NOT collide with the drawer's ✕ / the navbar's ☰,
  // both of which are "Hide/Show navigation": two controls under one accessible name makes every
  // by-name query in the layout tests ambiguous.
  it("carries its own collapse toggle, named apart from the drawer's close button", () => {
    renderRail();
    expect(screen.getByRole("button", { name: /collapse navigation/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /hide navigation/i })).not.toBeInTheDocument();
  });

  it("calls onToggle when the collapse button is clicked", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderRail(onToggle);
    await user.click(screen.getByRole("button", { name: /collapse navigation/i }));
    expect(onToggle).toHaveBeenCalledOnce();
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
  const renderCollapsed = () =>
    renderWithProviders(
      <Sidebar
        open={false}
        width={76}
        animate
        onToggle={vi.fn()}
        onOpenLibrary={vi.fn()}
        onNewAnalysis={vi.fn()}
      />
    );

  it("hides the nav labels when collapsed", () => {
    renderCollapsed();
    expect(screen.queryByText("Analyse")).not.toBeInTheDocument();
    expect(screen.queryByText("Library")).not.toBeInTheDocument();
  });

  // The wordmark is the one piece of the brand lockup that cannot survive the 76px strip. Once
  // the text is gone the link's aria-label is the only thing naming the mark.
  it("drops the brand wordmark but keeps the mark named", () => {
    renderCollapsed();
    expect(screen.queryByText("X-Coach")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "X-Coach" })).toBeInTheDocument();
  });

  // With the labels gone the `title` attributes are the only thing naming the destinations —
  // an icon strip whose icons are unnamed is unusable with a screen reader and unguessable
  // with a mouse. Querying by role+name is exactly what a user would rely on.
  it("keeps every destination named by its tooltip", () => {
    renderCollapsed();
    expect(screen.getByRole("link", { name: "Analyse" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Library" })).toBeInTheDocument();
  });

  it("flips the toggle's label to expand", () => {
    renderCollapsed();
    expect(screen.getByRole("button", { name: /expand navigation/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /collapse navigation/i })).not.toBeInTheDocument();
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

  // AppLayout hands the drawer no onToggle: a drawer that shrank to a 76px strip floating over
  // the page would be a second, worse way to dismiss it. It closes outright instead.
  it("renders no collapse toggle", () => {
    renderWithProviders(
      <Sidebar open width={270} animate={false} onOpenLibrary={vi.fn()} onNewAnalysis={vi.fn()} onClose={vi.fn()} />
    );
    expect(screen.queryByRole("button", { name: /collapse navigation/i })).not.toBeInTheDocument();
  });
});
