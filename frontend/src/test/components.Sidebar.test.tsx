import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Sidebar from "../components/Sidebar";
import { renderWithProviders } from "./renderWithProviders";

describe("Sidebar — open", () => {
  it("shows the X-Coach brand name when open", () => {
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.getByText("X-Coach")).toBeInTheDocument();
  });

  it("shows the nav labels when open", () => {
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.getByText("Analyse")).toBeInTheDocument();
    expect(screen.getByText("Library")).toBeInTheDocument();
  });

  it("shows the version and tagline when open", () => {
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.getByText("Prototype v0.1")).toBeInTheDocument();
    expect(screen.getByText("Pose · Rules · GraphRAG")).toBeInTheDocument();
  });

  it("has a 'Hide navigation' toggle button when open", () => {
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: /Hide navigation/i })).toBeInTheDocument();
  });

  it("calls onToggle when the toggle button is clicked", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={onToggle} onOpenLibrary={vi.fn()} />
    );
    await user.click(screen.getByRole("button", { name: /Hide navigation/i }));
    expect(onToggle).toHaveBeenCalledOnce();
  });

  it("calls onOpenLibrary when the Library button is clicked", async () => {
    const user = userEvent.setup();
    const onOpenLibrary = vi.fn();
    renderWithProviders(
      <Sidebar open={true} width={240} animate={false} onToggle={vi.fn()} onOpenLibrary={onOpenLibrary} />
    );
    await user.click(screen.getByRole("button", { name: /Library/i }));
    expect(onOpenLibrary).toHaveBeenCalledOnce();
  });
});

describe("Sidebar — collapsed", () => {
  it("hides the brand name when collapsed", () => {
    renderWithProviders(
      <Sidebar open={false} width={64} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.queryByText("X-Coach")).not.toBeInTheDocument();
  });

  it("has a 'Show navigation' toggle button when collapsed", () => {
    renderWithProviders(
      <Sidebar open={false} width={64} animate={false} onToggle={vi.fn()} onOpenLibrary={vi.fn()} />
    );
    expect(screen.getByRole("button", { name: /Show navigation/i })).toBeInTheDocument();
  });
});
