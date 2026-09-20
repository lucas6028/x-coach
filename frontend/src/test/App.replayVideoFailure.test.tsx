import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";
import App from "../App";
import { api } from "../api";
import { mockAnalysis } from "./fixtures";

function renderAppAt(entry: string) {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <AuthProvider>
        <I18nProvider>
          <App />
        </I18nProvider>
      </AuthProvider>
    </MemoryRouter>
  );
}

afterEach(() => vi.restoreAllMocks());

// A clinician can open a linked patient's saved report (GET /api/analyses/{id} works for them per
// backend/app/routers/clinic.py), but the clip's signed-URL endpoint (GET /api/uploads/{id}/url)
// 404s for them — clinicians have no DB access to videos, by design. This pins that App.tsx's
// replay path (via lib/useVideoSrc.ts) already degrades quietly when that happens: no crash, no
// blocking error screen, and the analysis results still render. See lib/useVideoSrc.ts:102-114 —
// the failure is swallowed there (`.catch` sets `src` to null) and never reaches App's own error
// state, so this test needed no change to App.tsx itself, only this verification.
describe("App replay — the clip's signed URL 404s (clinician viewing a patient's report)", () => {
  it("still renders the analysis results, with no crash and no blocking error screen", async () => {
    vi.spyOn(api, "getStoredAnalysis").mockResolvedValue({
      id: "a1",
      video_id: "vid_history",
      source: "upload",
      view_type: "side",
      fault_count: 1,
      created_at: "2026-09-14T00:00:00Z",
      result: { ...mockAnalysis, source: "upload", video_id: "vid_history", analysis_id: "a1" },
    });
    vi.spyOn(api, "uploadMedia").mockRejectedValue(new Error("404 Not Found for /api/uploads/vid_history/url"));

    renderAppAt("/app?analysis=a1");

    // The analysis rendered fully: the video panel's own fault-count pill (derived from
    // `analysis.detections`, not from the clip) is on screen.
    expect(await screen.findByText("1 fault detected")).toBeInTheDocument();
    // No blocking error screen from App's own error state.
    expect(screen.queryByText(/couldn't/i)).not.toBeInTheDocument();

    // The <video> element renders with no `src` — the failed re-sign left it unresolved rather
    // than pointing at a broken URL — confirming the quiet-degrade path, not a lucky non-crash.
    await waitFor(() => {
      const video = document.querySelector("video");
      expect(video).not.toBeNull();
      expect(video).not.toHaveAttribute("src");
    });
  });
});
