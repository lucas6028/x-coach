import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";

const { liffState, mockProbe } = vi.hoisted(() => ({
  liffState: { liff: null as unknown },
  mockProbe: vi.fn(),
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => true,
  initLiff: vi.fn(() => Promise.resolve(liffState.liff)),
}));

vi.mock("../lib/camera", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/camera")>();
  return { ...actual, probeCamera: mockProbe };
});

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import LiffDiag, { collectLiffFacts } from "../pages/LiffDiag";
import type { Liff } from "@line/liff";

const mockUseAuth = vi.mocked(useAuth);

function fakeLiff(overrides: Record<string, unknown> = {}) {
  return {
    isInClient: vi.fn().mockReturnValue(true),
    isLoggedIn: vi.fn().mockReturnValue(true),
    getIDToken: vi.fn().mockReturnValue("tok"),
    getOS: vi.fn().mockReturnValue("ios"),
    getLineVersion: vi.fn().mockReturnValue("14.0.0"),
    getVersion: vi.fn().mockReturnValue("2.29.1"),
    getLanguage: vi.fn().mockReturnValue("zh-TW"),
    ...overrides,
  } as unknown as Liff;
}

function renderDiag() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <LiffDiag />
      </I18nProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  liffState.liff = fakeLiff();
  mockUseAuth.mockReturnValue({ user: null } as unknown as ReturnType<typeof useAuth>);
});

describe("collectLiffFacts", () => {
  it("reports a live SDK's facts", () => {
    const facts = Object.fromEntries(collectLiffFacts(fakeLiff()).map((f) => [f.label, f.value]));
    expect(facts["liff.init"]).toBe("ok");
    expect(facts["isInClient"]).toBe("true");
    expect(facts["ID token"]).toBe("present");
    expect(facts["OS"]).toBe("ios");
    expect(facts["LINE version"]).toBe("14.0.0");
  });

  it("degrades every SDK-derived fact when init failed", () => {
    const facts = Object.fromEntries(collectLiffFacts(null).map((f) => [f.label, f.value]));
    expect(facts["liff.init"]).toBe("failed"); // isLiffConfigured is mocked true
    expect(facts["isInClient"]).toBe("—");
    expect(facts["ID token"]).toBe("—");
  });

  it("shows — when an SDK getter throws", () => {
    const throwing = fakeLiff({
      getLineVersion: vi.fn(() => {
        throw new Error("not in client");
      }),
    });
    const facts = Object.fromEntries(collectLiffFacts(throwing).map((f) => [f.label, f.value]));
    expect(facts["LINE version"]).toBe("—");
  });
});

describe("LiffDiag page", () => {
  it("renders the environment facts", async () => {
    renderDiag();
    await waitFor(() => expect(screen.getByText("ios")).toBeInTheDocument());
    expect(screen.getByText("VITE_LIFF_ID")).toBeInTheDocument();
    expect(screen.getByText("2.29.1")).toBeInTheDocument();
  });

  it("shows the signed-out session state, and signed-in when a user exists", async () => {
    renderDiag();
    expect(await screen.findByText(/Not signed in|未登入/)).toBeInTheDocument();
    mockUseAuth.mockReturnValue({
      user: { id: "u1", email: "line_u1@line.invalid" },
    } as unknown as ReturnType<typeof useAuth>);
    renderDiag();
    expect(await screen.findByText("line_u1@line.invalid")).toBeInTheDocument();
  });

  it("runs the camera probe and reports success", async () => {
    mockProbe.mockResolvedValue({ ok: true, reason: "ok", message: "fine" });
    renderDiag();
    await userEvent.click(screen.getByRole("button", { name: /Test camera|測試相機/ }));
    await waitFor(() =>
      expect(screen.getByText(/Camera works|可正常使用相機/)).toBeInTheDocument()
    );
    expect(mockProbe).toHaveBeenCalled();
  });

  it("shows the LIFF escape-hatch hint when the probe times out inside LINE", async () => {
    mockProbe.mockResolvedValue({ ok: false, reason: "timeout", message: "hung" });
    renderDiag();
    // Wait for init to resolve (inClient=true) before probing.
    await waitFor(() => expect(screen.getByText("ios")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /Test camera|測試相機/ }));
    await waitFor(() =>
      expect(screen.getByText(/Open in browser|用瀏覽器開啟/)).toBeInTheDocument()
    );
  });

  it("does not show the hint for a plain permission denial", async () => {
    mockProbe.mockResolvedValue({
      ok: false,
      reason: "denied",
      message: "NotAllowedError: permission was denied.",
    });
    renderDiag();
    await waitFor(() => expect(screen.getByText("ios")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /Test camera|測試相機/ }));
    await waitFor(() =>
      expect(screen.getByText(/permission was denied/)).toBeInTheDocument()
    );
    expect(screen.queryByText(/Open in browser|用瀏覽器開啟/)).not.toBeInTheDocument();
  });
});
