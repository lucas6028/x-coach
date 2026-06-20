import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import App from "../App";
import { mockAnalysis } from "./fixtures";

function renderApp() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <App />
      </I18nProvider>
    </MemoryRouter>
  );
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

describe("App — sidebar toggle", () => {
  it("collapses the desktop sidebar when the toggle button is clicked", async () => {
    const user = userEvent.setup();
    renderApp();
    // Both desktop and mobile sidebars render, so expect 2 X-Coach elements initially.
    expect(screen.getAllByText("X-Coach").length).toBe(2);
    // The desktop sidebar's toggle button (first in DOM order)
    const hideBtn = screen.getAllByRole("button", { name: /Hide navigation/i })[0];
    await user.click(hideBtn);
    // After collapsing, only the mobile sidebar (always open=true) still shows X-Coach
    expect(screen.getAllByText("X-Coach").length).toBe(1);
  });
});
