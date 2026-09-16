import { describe, it, expect, vi, afterEach } from "vitest";
import { act, screen, within } from "@testing-library/react";
import { renderWithProviders } from "./renderWithProviders";
import TrendChart from "../components/clinic/TrendChart";
import type { ClinicTrendPoint } from "../api";

// WP4: `flagged` is now computed server-side per point (services/clinic.py::trend) and carried on
// the point itself, rather than derived here by matching `created_at` against the check-in/
// open-flag rows — this suite proves the chart reads `point.flagged` directly.
function points(overrides: Partial<ClinicTrendPoint>[] = []): ClinicTrendPoint[] {
  const base: ClinicTrendPoint[] = [
    { created_at: "2026-09-01T00:00:00Z", form_score: 70, pain_nrs: 3, flagged: false },
    { created_at: "2026-09-08T00:00:00Z", form_score: 40, pain_nrs: 8, flagged: true },
  ];
  return base.map((p, i) => ({ ...p, ...overrides[i] }));
}

describe("TrendChart", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Regression guard for a defect only a screenshot showed: with a constant `viewBox="0 0 320 72"`
  // on a `w-full` svg, the browser preserves the 320:72 ratio, so in a ~700px panel the whole plot
  // scaled to the 72px height and floated as a 320px island in the middle. The viewBox width must
  // follow the measured element width (1 user unit = 1 px), so the plot fills the panel with round
  // dots and unstretched strokes.
  it("tracks the measured width in the viewBox, falling back to 320 before measurement", () => {
    const callbacks: ResizeObserverCallback[] = [];
    class FakeResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        callbacks.push(cb);
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);

    renderWithProviders(<TrendChart trend={points()} />);
    const form = screen.getByLabelText("Form score over the last 30 days");
    const pain = screen.getByLabelText("Pain over the last 30 days");
    // jsdom lays nothing out, so clientWidth is 0 -> the fallback, never a 0-wide viewBox.
    expect(form).toHaveAttribute("viewBox", "0 0 320 72");
    expect(pain).toHaveAttribute("viewBox", "0 0 320 40");

    act(() => {
      callbacks[0]([{ contentRect: { width: 812 } } as ResizeObserverEntry], {} as ResizeObserver);
    });
    expect(form).toHaveAttribute("viewBox", "0 0 812 72");
    expect(pain).toHaveAttribute("viewBox", "0 0 812 40");
    // The last point sits at the right edge (width - PAD_X), i.e. the plot really is stretched.
    const dots = form.querySelectorAll("circle");
    expect(dots[dots.length - 1].getAttribute("cx")).toBe("806");
  });

  it("shows the empty state with fewer than two points", () => {
    renderWithProviders(<TrendChart trend={[{ ...points()[0] }]} />);
    expect(screen.getByText("Not enough check-ins yet to chart a trend.")).toBeInTheDocument();
  });

  it("marks the accessible table row 'Flagged' for a flagged point and 'Not flagged' otherwise", () => {
    renderWithProviders(<TrendChart trend={points()} />);
    const table = screen.getByRole("table", { hidden: true });
    const rows = within(table).getAllByRole("row").slice(1); // drop the header row
    expect(within(rows[0]).getByText("Not flagged")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Flagged")).toBeInTheDocument();
  });

  it("reads flagged straight off each point, not by matching timestamps against other rows", () => {
    // Two points share one created_at (a same-instant duplicate is exactly the case the old
    // timestamp-matching approach could not disambiguate); only the SECOND is flagged. If the
    // chart still matched by timestamp instead of reading `point.flagged`, this would flag both
    // (a Set has no notion of "the second occurrence") or neither.
    const shared = "2026-09-10T00:00:00Z";
    const trend: ClinicTrendPoint[] = [
      { created_at: shared, form_score: 70, pain_nrs: 3, flagged: false },
      { created_at: shared, form_score: 40, pain_nrs: 8, flagged: true },
    ];
    renderWithProviders(<TrendChart trend={trend} />);
    const table = screen.getByRole("table", { hidden: true });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("Not flagged")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Flagged")).toBeInTheDocument();
  });
});
