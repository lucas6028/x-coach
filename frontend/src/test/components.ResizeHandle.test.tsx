import { describe, it, expect, vi } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import ResizeHandle from "../components/ResizeHandle";
import { renderWithProviders } from "./renderWithProviders";

describe("ResizeHandle", () => {
  it("renders a separator element", () => {
    renderWithProviders(<ResizeHandle onResize={vi.fn()} />);
    expect(screen.getByRole("separator")).toBeInTheDocument();
  });

  it("calls onResizeStart on mouse down", () => {
    const onResizeStart = vi.fn();
    renderWithProviders(<ResizeHandle onResize={vi.fn()} onResizeStart={onResizeStart} />);
    fireEvent.mouseDown(screen.getByRole("separator"), { clientX: 100 });
    expect(onResizeStart).toHaveBeenCalledOnce();
  });

  it("calls onResize with the delta on mouse move after press", () => {
    const onResize = vi.fn();
    renderWithProviders(<ResizeHandle onResize={onResize} />);
    fireEvent.mouseDown(screen.getByRole("separator"), { clientX: 100 });
    fireEvent.mouseMove(window, { clientX: 120 });
    expect(onResize).toHaveBeenCalledWith(20);
  });

  it("calls onResize with negative delta when moving left", () => {
    const onResize = vi.fn();
    renderWithProviders(<ResizeHandle onResize={onResize} />);
    fireEvent.mouseDown(screen.getByRole("separator"), { clientX: 200 });
    fireEvent.mouseMove(window, { clientX: 170 });
    expect(onResize).toHaveBeenCalledWith(-30);
  });

  it("calls onResizeEnd on mouse up", () => {
    const onResizeEnd = vi.fn();
    renderWithProviders(<ResizeHandle onResize={vi.fn()} onResizeEnd={onResizeEnd} />);
    fireEvent.mouseDown(screen.getByRole("separator"), { clientX: 100 });
    fireEvent.mouseUp(window);
    expect(onResizeEnd).toHaveBeenCalledOnce();
  });

  it("stops firing onResize after mouse up", () => {
    const onResize = vi.fn();
    renderWithProviders(<ResizeHandle onResize={onResize} />);
    fireEvent.mouseDown(screen.getByRole("separator"), { clientX: 100 });
    fireEvent.mouseUp(window);
    fireEvent.mouseMove(window, { clientX: 150 });
    // Should only have been called during the active drag, not after mouseup
    expect(onResize).not.toHaveBeenCalled();
  });
});
