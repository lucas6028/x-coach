import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import App from "../App";
import { mockAnalysis } from "./fixtures";

function renderApp() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <I18nProvider>
          <App />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

// Exposes the current location.search so a test can assert the URL the app navigated to.
function LocationProbe() {
  const loc = useLocation();
  return <div data-testid="loc-search">{loc.search}</div>;
}

function renderAppWithLocation() {
  return render(
    <MemoryRouter initialEntries={["/app"]}>
      <AuthProvider>
        <I18nProvider>
          <App />
          <LocationProbe />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

function uploadAClip() {
  const input = document.querySelector("input[type=file]") as HTMLInputElement | null;
  expect(input).not.toBeNull();
  fireEvent.change(input!, {
    target: { files: [new File(["data"], "squat.mp4", { type: "video/mp4" })] },
  });
}

afterEach(() => vi.restoreAllMocks());

describe("App — initial state", () => {
  it("renders the demo intro (no analysis)", () => {
    renderApp();
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Analyze a squat in about 20 seconds."
    );
  });

  it("renders the header with AWAITING INPUT status", () => {
    renderApp();
    expect(screen.getByText("AWAITING INPUT")).toBeInTheDocument();
  });

  it("renders the sidebar", () => {
    renderApp();
    expect(screen.getAllByText("X-Coach").length).toBeGreaterThan(0);
  });
});

describe("App — library picker", () => {
  it("opens the library picker when 'Open a sample clip' is clicked", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ total: 0, items: [] }),
    } as Response);

    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole("button", { name: /Open a sample clip/i }));
    expect(screen.getByText("Sample Library")).toBeInTheDocument();
  });

  it("closes the library picker when the close button is clicked", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ total: 0, items: [] }),
    } as Response);

    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole("button", { name: /Open a sample clip/i }));
    await waitFor(() => expect(screen.getByText("Sample Library")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(screen.queryByText("Sample Library")).not.toBeInTheDocument();
  });
});

describe("App — analysis loaded", () => {
  it("shows analysis results when API returns data", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ total: 1, items: [{ video_id: "vid_001", split: "test", view_type: "side", fault_count: 1, faults: ["knees_inward"] }] }),
    } as Response);

    const user = userEvent.setup();
    renderApp();
    await user.click(screen.getByRole("button", { name: /Open a sample clip/i }));
    await waitFor(() => screen.getByText("vid_001"));

    // Now mock getAnalysis and click the video
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockAnalysis,
    } as Response);

    await user.click(screen.getByText("vid_001").closest("button")!);
    await waitFor(() => expect(screen.getByText("ANALYSIS COMPLETE")).toBeInTheDocument());
  });

  it("shows an error message when an upload fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Error",
      json: async () => ({ detail: "Server error" }),
    } as Response);

    renderApp();

    // The hidden file input drives uploads — assert it exists so this test can't
    // silently no-op if the dropzone stops rendering it.
    const input = document.querySelector("input[type=file]") as HTMLInputElement | null;
    expect(input).not.toBeNull();

    const file = new File(["data"], "squat.mp4", { type: "video/mp4" });
    fireEvent.change(input!, { target: { files: [file] } });

    await waitFor(() =>
      expect(screen.getByText("That clip did not go through")).toBeInTheDocument()
    );
    // The backend's detail message is surfaced to the user.
    expect(screen.getByText("Server error")).toBeInTheDocument();
  });
});

describe("App — upload reflects the analysis in the URL", () => {
  it("updates the URL to ?analysis=<id> after a persisted upload, without re-fetching", async () => {
    // A signed-in upload comes back with a persisted analysis_id.
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ...mockAnalysis, analysis_id: "bb718ecf" }),
    } as Response);

    renderAppWithLocation();
    uploadAClip();

    await waitFor(() => expect(screen.getByText("ANALYSIS COMPLETE")).toBeInTheDocument());
    // The URL now carries the id (shareable + refresh-survivable). Poll it: the router's location
    // update can settle a tick after the analysis render.
    await waitFor(() =>
      expect(screen.getByTestId("loc-search").textContent).toBe("?analysis=bb718ecf")
    );
    // The replay effect is guarded, so the analysis is NOT re-fetched — if it were, GET
    // /api/analyses/<id> would return this same (result-less) shape and drop "ANALYSIS COMPLETE".
    expect(screen.getByText("ANALYSIS COMPLETE")).toBeInTheDocument();
  });

  it("leaves the URL untouched for an anonymous upload (no analysis_id)", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockAnalysis, // anonymous: nothing persisted, no analysis_id
    } as Response);

    renderAppWithLocation();
    uploadAClip();

    await waitFor(() => expect(screen.getByText("ANALYSIS COMPLETE")).toBeInTheDocument());
    expect(screen.getByTestId("loc-search").textContent).toBe("");
  });
});

describe("App — sidebar toggle", () => {
  it("expands the desktop sidebar when the toggle button is clicked", async () => {
    const user = userEvent.setup();
    renderApp();
    // The desktop sidebar defaults to collapsed, so only the (always-open) mobile
    // drawer shows the brand initially.
    expect(screen.getAllByText("X-Coach").length).toBe(1);
    // The desktop rail's show-navigation toggle (first in DOM order).
    const showBtn = screen.getAllByRole("button", { name: /Show navigation/i })[0];
    await user.click(showBtn);
    // After expanding, both desktop and mobile sidebars show the brand.
    expect(screen.getAllByText("X-Coach").length).toBe(2);
  });
});
