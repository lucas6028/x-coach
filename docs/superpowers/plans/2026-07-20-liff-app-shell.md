# LIFF App Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the SPA is opened inside the LINE app, render an app-like shell — a bottom tab bar instead of the web sidebar/marketing header — instead of the web layout.

**Architecture:** A `LiffProvider` resolves `liff.isInClient()` once (async) and exposes it through `useLiffContext()`. `AppLayout` branches on it at the top: in-client renders the new sibling `LiffAppShell`, otherwise the existing Header+Sidebar layout is untouched. No page component changes its own structure — every page already renders through `AppLayout`.

**Tech Stack:** React 18 + TypeScript + Vite + Tailwind, react-router-dom v6, `@line/liff`, `@phosphor-icons/react`, vitest + @testing-library/react.

Spec: `docs/superpowers/specs/2026-07-20-liff-app-shell-design.md`

## Global Constraints

- **Every frontend command runs with cwd = `frontend/`.** The Bash and PowerShell tools share one cwd; a stray `cd` to the repo root mass-fails vitest. Run `cd frontend` first in each shell you use.
- Test runner: `yarn test` (= `vitest run`). Single file: `yarn test src/test/<file>`. Coverage gate: `yarn test:coverage` (lines 70 / functions 70 / branches 60 / statements 68).
- Tests are vitest files under `frontend/src/test/`, named `<area>.<Subject>[.<facet>].test.tsx`. Use the shared `renderWithProviders` helper (`src/test/renderWithProviders.tsx`) where a router + auth + i18n context is enough.
- Any LIFF-side failure degrades to plain web. `initLiff()` already catches init failures and resolves `null`; `null` must read as `isInClient: false`.
- Every new user-visible string needs BOTH an `en` and a `zh-Hant` entry in `frontend/src/lib/i18n.tsx` (the `en` dict starts at line 18, `zhHant` at line 618).
- Do not modify `lib/liff.ts`, `lib/auth.tsx`, `lib/camera.ts`, `pages/SixSeven.tsx`, `pages/FruitNinja.tsx`, or any backend code.
- Commit after every task.

### Two deliberate deviations from the spec (decided while reading the code)

1. **Studio title.** The spec said the analysis tab's nav label. `Header.tsx:55` already falls back to `t("header.title")` ("Squat Analysis" / 「深蹲分析」) when no `title` prop is given, so `LiffAppShell` uses the *same* fallback — one less concept, consistent copy.
2. **Tab bar is in normal flow, not `fixed`.** The spec described a fixed bar plus compensating bottom padding on `main`. A flex-column shell with the `<nav>` as the last child gets the same visual result with no overlap and no magic padding number. Safe-area insets still apply (`pb-[env(safe-area-inset-bottom)]` on the nav).

---

### Task 1: `useLiffContext` — the in-client detection hook

**Files:**
- Create: `frontend/src/lib/liffContext.tsx`
- Modify: `frontend/src/main.tsx`
- Test: `frontend/src/test/lib.liffContext.test.tsx`

**Interfaces:**
- Consumes: `initLiff()` and `isLiffConfigured()` from `src/lib/liff.ts` (existing; `initLiff` memoizes its promise, so calling it here does NOT trigger a second init).
- Produces:
  - `export function LiffProvider({ children }: { children: ReactNode }): JSX.Element`
  - `export function useLiffContext(): { ready: boolean; isInClient: boolean }`
  - (`guessInClient()` stays module-private — the provider tests cover every branch of it.)

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/lib.liffContext.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// The provider is the unit under test; lib/liff is its (already-tested) dependency.
const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import { LiffProvider, useLiffContext } from "../lib/liffContext";

function Probe() {
  const { ready, isInClient } = useLiffContext();
  return <span>{`ready=${ready} inClient=${isInClient}`}</span>;
}

const renderProbe = () =>
  render(
    <LiffProvider>
      <Probe />
    </LiffProvider>
  );

// jsdom's navigator.userAgent is read-only; redefine it per test.
function stubUserAgent(ua: string) {
  Object.defineProperty(window.navigator, "userAgent", { value: ua, configurable: true });
}

const REAL_UA = window.navigator.userAgent;

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => true };
  stubUserAgent(REAL_UA);
  window.history.replaceState({}, "", "/app");
});

afterEach(() => {
  stubUserAgent(REAL_UA);
});

describe("LiffProvider — optimistic guess before init resolves", () => {
  it("guesses in-client from the LINE user agent", () => {
    stubUserAgent("Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 Line/14.2.0");
    renderProbe();
    // First paint, before the init promise resolves: the guess is already applied.
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses in-client from a liff.state query param", () => {
    window.history.replaceState({}, "", "/app?liff.state=%2Fhistory");
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses in-client from a liff-referrer query param", () => {
    window.history.replaceState({}, "", "/app?liff-referrer=https%3A%2F%2Fliff.line.me");
    renderProbe();
    expect(screen.getByText("ready=false inClient=true")).toBeInTheDocument();
  });

  it("guesses web for a plain browser", () => {
    renderProbe();
    expect(screen.getByText("ready=false inClient=false")).toBeInTheDocument();
  });
});

describe("LiffProvider — correction once the SDK answers", () => {
  it("corrects an optimistic in-client guess that was wrong", async () => {
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0");
    liffState.sdk = { isInClient: () => false };
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });

  it("promotes a web guess to in-client when the SDK says so", async () => {
    liffState.sdk = { isInClient: () => true };
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=true")).toBeInTheDocument()
    );
  });

  it("reads a failed init (null) as not in-client", async () => {
    liffState.sdk = null;
    renderProbe();
    await waitFor(() =>
      expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument()
    );
  });
});

describe("LiffProvider — unconfigured", () => {
  it("is immediately ready and never in-client without VITE_LIFF_ID", () => {
    liffState.configured = false;
    stubUserAgent("Mozilla/5.0 (iPhone) Line/14.2.0");
    renderProbe();
    // No LIFF id: the LINE user agent is irrelevant, and there is nothing to wait for.
    expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument();
  });
});

describe("useLiffContext — outside a provider", () => {
  it("defaults to the web shell", () => {
    render(<Probe />);
    expect(screen.getByText("ready=true inClient=false")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/lib.liffContext.test.tsx`
Expected: FAIL — `Failed to resolve import "../lib/liffContext"`.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/lib/liffContext.tsx`:

```tsx
// LIFF situation awareness for the layout layer. `lib/liff.isInLiffClient()` is async (it
// awaits the SDK init), but a layout has to decide which shell to render on its very first
// paint — so this provider resolves the real answer once and shares it with the whole app.
//
// The pending window is real: on a LINE redirect-return the SDK load + init takes ~1-1.5s.
// Rendering the web shell for that second and then swapping to the app shell is a visible
// flash of the wrong UI, so the initial state is a SYNCHRONOUS guess from two signals LINE
// leaves lying around (the in-app browser's user agent, and the liff.state/liff-referrer
// params LINE appends when it opens a LIFF URL). The guess is corrected — silently, and
// almost never in practice — once the SDK answers.

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { initLiff, isLiffConfigured } from "./liff";

export interface LiffContextValue {
  /** Whether the SDK has answered yet (false while `isInClient` is still the guess). */
  ready: boolean;
  /** Running inside the LINE app's in-app browser. */
  isInClient: boolean;
}

// Default for consumers rendered outside the provider (every existing unit test): plain web.
const LiffCtx = createContext<LiffContextValue>({ ready: true, isInClient: false });

/** The synchronous first-paint guess. */
function guessInClient(): boolean {
  // No LIFF id configured — the app is a plain web page and no signal can change that.
  if (!isLiffConfigured()) return false;
  if (/\bLine\//i.test(navigator.userAgent || "")) return true;
  const query = window.location.search;
  return query.includes("liff.state") || query.includes("liff-referrer");
}

export function LiffProvider({ children }: { children: ReactNode }) {
  const [value, setValue] = useState<LiffContextValue>(() => ({
    // Nothing to wait for when LIFF is unconfigured: that state is final.
    ready: !isLiffConfigured(),
    isInClient: guessInClient(),
  }));

  useEffect(() => {
    if (!isLiffConfigured()) return;
    let cancelled = false;
    // Awaits the promise main.tsx already kicked off — initLiff() memoizes, so this is a
    // subscription to the in-flight init, not a second one.
    void initLiff().then((liff) => {
      if (cancelled) return;
      setValue({ ready: true, isInClient: Boolean(liff?.isInClient()) });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return <LiffCtx.Provider value={value}>{children}</LiffCtx.Provider>;
}

/** `{ ready, isInClient }` — safe to call anywhere, including outside the provider. */
export function useLiffContext(): LiffContextValue {
  return useContext(LiffCtx);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/lib.liffContext.test.tsx`
Expected: PASS — 9 tests.

- [ ] **Step 5: Mount the provider in `main.tsx`**

In `frontend/src/main.tsx`, add the import next to the other lib imports:

```tsx
import { LiffProvider } from "./lib/liffContext";
```

Then wrap `AuthProvider` — the provider must sit OUTSIDE it, because Landing and Login read
the context and neither is rendered under auth:

```tsx
  <React.StrictMode>
    <BrowserRouter>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <Routes>
              {/* …every existing route, unchanged… */}
            </Routes>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </BrowserRouter>
  </React.StrictMode>
```

(Re-indent the existing `AuthProvider` subtree by two spaces; do not change any route.)

- [ ] **Step 6: Run the full suite**

Run: `cd frontend && yarn test`
Expected: PASS — every existing test still green (`main.tsx` has no test of its own and is excluded from coverage).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/liffContext.tsx frontend/src/main.tsx frontend/src/test/lib.liffContext.test.tsx
git commit -m "feat(liff): add the in-client detection context"
```

---

### Task 2: `LiffAppShell` — the bottom-tab app shell

**Files:**
- Create: `frontend/src/components/LiffAppShell.tsx`
- Test: `frontend/src/test/components.LiffAppShell.test.tsx`

**Interfaces:**
- Consumes: `useI18n()` from `src/lib/i18n.tsx`; existing i18n keys `nav.analyse`, `nav.history`, `nav.games`, `settings.title`, `header.title` (all already present in both dictionaries — no new keys in this task).
- Produces: `export default function LiffAppShell({ children, title }: { children: ReactNode; title?: string }): JSX.Element`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/components.LiffAppShell.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import LiffAppShell from "../components/LiffAppShell";
import { I18nProvider } from "../lib/i18n";

const renderAt = (path: string, title?: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <I18nProvider>
        <LiffAppShell title={title}>
          <p>page body</p>
        </LiffAppShell>
      </I18nProvider>
    </MemoryRouter>
  );

describe("LiffAppShell — structure", () => {
  it("renders its children", () => {
    renderAt("/app");
    expect(screen.getByText("page body")).toBeInTheDocument();
  });

  it("shows the four tabs and nothing else in the tab bar", () => {
    renderAt("/app");
    const bar = screen.getByRole("navigation");
    const links = screen.getAllByRole("link");
    expect(links).toHaveLength(4);
    expect(bar).toContainElement(links[0]);
    ["Analyse", "My records", "Games", "Settings"].forEach((label) => {
      expect(screen.getByRole("link", { name: new RegExp(label, "i") })).toBeInTheDocument();
    });
  });

  it("omits the web chrome — no brand lockup, no sidebar toggle, no sign-in", () => {
    renderAt("/app");
    expect(screen.queryByText("X-Coach")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /navigation/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /sign in/i })).not.toBeInTheDocument();
  });

  it("insets the tab bar for the home indicator", () => {
    renderAt("/app");
    expect(screen.getByRole("navigation").className).toContain("safe-area-inset-bottom");
  });
});

describe("LiffAppShell — title", () => {
  it("shows the page title when given one", () => {
    renderAt("/history", "My records");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("My records");
  });

  it("falls back to the studio title when untitled (the studio passes no title)", () => {
    renderAt("/app");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Squat Analysis");
  });
});

describe("LiffAppShell — active tab", () => {
  it.each([
    ["/app", "Analyse"],
    ["/history", "My records"],
    ["/games", "Games"],
    ["/settings", "Settings"],
  ])("highlights %s", (path, label) => {
    renderAt(path);
    const link = screen.getByRole("link", { name: new RegExp(label, "i") });
    expect(link.className).toContain("text-primary");
  });

  it.each(["/67", "/ninja"])("highlights the Games tab on the game route %s", (path) => {
    renderAt(path);
    expect(screen.getByRole("link", { name: /Games/i }).className).toContain("text-primary");
  });

  it("highlights nothing on a tab-less route", () => {
    renderAt("/movements");
    screen.getAllByRole("link").forEach((link) => {
      expect(link.className).not.toContain("text-primary");
    });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/components.LiffAppShell.test.tsx`
Expected: FAIL — `Failed to resolve import "../components/LiffAppShell"`.

- [ ] **Step 3: Write the implementation**

Create `frontend/src/components/LiffAppShell.tsx`:

```tsx
import { ClockCounterClockwise, GameController, GearSix, VideoCamera } from "@phosphor-icons/react";
import { Link, useLocation } from "react-router-dom";
import { type ReactNode } from "react";
import { useI18n } from "../lib/i18n";

// The in-LINE shell: what AppLayout renders instead of the navbar + sidebar when the SPA is
// running inside the LINE app. A bottom tab bar is the one thing that makes a web app read as
// a native one on a phone, so the marketing header, the brand lockup, the sign-in button and
// the sidebar all go away — inside LINE the user is already signed in silently (see
// lib/auth's LINE auto-login) and the four destinations below are the whole app.
//
// Movements and Admin deliberately have no tab: their routes still resolve (rich menu, direct
// link, or a link from the studio), they just don't earn one of four thumb-reachable slots.

interface Props {
  children: ReactNode;
  // The page name, same prop AppLayout takes. The studio passes none and gets the studio title
  // — mirroring Header's own fallback, so the copy matches the web build.
  title?: string;
}

export default function LiffAppShell({ children, title }: Props) {
  const { t } = useI18n();
  const { pathname } = useLocation();

  const tabs = [
    { to: "/app", label: t("nav.analyse"), Icon: VideoCamera, active: pathname === "/app" },
    {
      to: "/history",
      label: t("nav.history"),
      Icon: ClockCounterClockwise,
      active: pathname === "/history",
    },
    {
      to: "/games",
      label: t("nav.games"),
      Icon: GameController,
      // One tab for the hub and both individual games (mirrors Sidebar's onGames check).
      active: pathname === "/games" || pathname === "/67" || pathname === "/ninja",
    },
    { to: "/settings", label: t("settings.title"), Icon: GearSix, active: pathname === "/settings" },
  ];

  return (
    // 100dvh (not 100vh) so the LINE in-app browser's collapsing toolbar can't clip the tab
    // bar; the top inset covers a notch when LINE renders the LIFF view full-bleed.
    <div className="h-[100dvh] w-full flex flex-col overflow-hidden bg-background-dark text-content pt-[env(safe-area-inset-top)]">
      <header className="h-12 shrink-0 border-b border-border-dark bg-surface flex items-center px-4">
        <h1 className="text-sm font-semibold tracking-tight truncate">{title ?? t("header.title")}</h1>
      </header>

      {/* Same contract as AppLayout's main: bounded height, pages own their own scrolling. */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0">{children}</main>

      {/* In normal flow, not fixed — the flex column already keeps it pinned to the bottom, and
          nothing can slide under it, so no compensating padding is needed. */}
      <nav className="shrink-0 grid grid-cols-4 border-t border-border-dark bg-surface pb-[env(safe-area-inset-bottom)]">
        {tabs.map(({ to, label, Icon, active }) => (
          <Link
            key={to}
            to={to}
            className={`flex flex-col items-center gap-0.5 py-2 text-[11px] font-medium transition-colors ${
              active ? "text-primary" : "text-muted"
            }`}
          >
            <Icon size={22} weight={active ? "fill" : "duotone"} />
            <span className="truncate max-w-full px-1">{label}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/components.LiffAppShell.test.tsx`
Expected: PASS — 12 tests.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/LiffAppShell.tsx frontend/src/test/components.LiffAppShell.test.tsx
git commit -m "feat(liff): add the in-client bottom-tab app shell"
```

---

### Task 3: Branch `AppLayout` on the LIFF context

**Files:**
- Modify: `frontend/src/components/AppLayout.tsx`
- Test: `frontend/src/test/components.AppLayout.liff.test.tsx`

**Interfaces:**
- Consumes: `useLiffContext()` (Task 1), `LiffAppShell` (Task 2).
- Produces: no new exports — `AppLayout`'s props are unchanged, so no page component needs editing.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/components.AppLayout.liff.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Drive the branch through the real provider, with lib/liff (the SDK edge) faked.
const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import AppLayout from "../components/AppLayout";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

const renderLayout = () =>
  render(
    <MemoryRouter initialEntries={["/app"]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <AppLayout title="My records">
              <p>page body</p>
            </AppLayout>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => false };
});

describe("AppLayout — inside the LINE app", () => {
  beforeEach(() => {
    liffState.sdk = { isInClient: () => true };
  });

  it("renders the tab bar and drops the sidebar", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    // The web sidebar's signature entries are gone.
    expect(screen.queryByText("Prototype v0.1")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /New analysis/i })).not.toBeInTheDocument();
    // The tab bar's four destinations are present.
    expect(screen.getByRole("link", { name: /Analyse/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Settings/i })).toBeInTheDocument();
  });

  it("drops the web navbar's brand lockup", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    expect(screen.queryByLabelText("X-Coach")).not.toBeInTheDocument();
  });

  it("still renders the page body", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByText("page body")).toBeInTheDocument());
  });
});

describe("AppLayout — on the web (regression guard)", () => {
  it("keeps the existing navbar + sidebar shell", async () => {
    renderLayout();
    await waitFor(() => expect(screen.getByText("Prototype v0.1")).toBeInTheDocument());
    expect(screen.getByLabelText("X-Coach")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /New analysis/i })).toBeInTheDocument();
    expect(screen.getByText("page body")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/components.AppLayout.liff.test.tsx`
Expected: FAIL — the in-client cases fail because `AppLayout` always renders the sidebar (`Prototype v0.1` is found where the test asserts it is absent).

- [ ] **Step 3: Write the implementation**

In `frontend/src/components/AppLayout.tsx`, add the two imports:

```tsx
import LiffAppShell from "./LiffAppShell";
import { useLiffContext } from "../lib/liffContext";
```

Inside the component, add the hook alongside the existing ones (it must be called before the
early return so hook order stays stable across renders):

```tsx
  const navigate = useNavigate();
  const { isInClient } = useLiffContext();
  // Desktop sidebar is a fixed-width rail; it only toggles between expanded and the icon rail.
  const [sidebarOpen, setSidebarOpen] = useState(initialSidebarOpen);
  const [mobileNav, setMobileNav] = useState(false);
```

Then, immediately after the `openLibrary` / `newAnalysis` fallbacks and before the returned
JSX, add the branch:

```tsx
  // Inside the LINE app the whole web chrome is replaced by the app shell: bottom tabs instead
  // of a sidebar, no marketing navbar. Every page renders through here, so this one branch
  // converts the entire app without any page knowing about LIFF.
  if (isInClient) return <LiffAppShell title={title}>{children}</LiffAppShell>;
```

Also extend the component's doc comment above `export default function AppLayout` with a final
sentence:

```
// Inside the LINE in-app browser this delegates to LiffAppShell instead (see lib/liffContext).
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/components.AppLayout.liff.test.tsx`
Expected: PASS — 4 tests.

- [ ] **Step 5: Run the full suite (the branch touches every page)**

Run: `cd frontend && yarn test`
Expected: PASS — all existing tests stay green. They render without `LiffProvider`, and the
context default is `{ ready: true, isInClient: false }`, so they take the web branch.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/AppLayout.tsx frontend/src/test/components.AppLayout.liff.test.tsx
git commit -m "feat(liff): render the app shell instead of the web chrome in LINE"
```

---

### Task 4: Land the LIFF entry point on `/app`

**Files:**
- Modify: `frontend/src/landing/Landing.tsx:564`
- Modify: `frontend/src/pages/Login.tsx:87`
- Test: `frontend/src/test/pages.liffEntry.test.tsx`

**Interfaces:**
- Consumes: `useLiffContext()` (Task 1).
- Produces: nothing new — behavioural change only.

Context: the real entry point is the LIFF endpoint URL configured in the LINE console
(`https://<host>/app`). These redirects are the safety net for a user who reaches `/` or
`/login` inside LINE — the marketing page and the sign-in form are both meaningless there,
because `lib/auth`'s LINE auto-login is already exchanging a token in the background.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/pages.liffEntry.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import Landing from "../landing/Landing";
import Login from "../pages/Login";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

// Render the real route table for the two entry routes plus a studio stand-in, so a redirect
// is observable as "the studio marker is on screen".
const renderAt = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/app" element={<p>studio</p>} />
            </Routes>
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => true };
});

describe("entry points inside the LINE app", () => {
  it("redirects the landing page to the studio", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.getByText("studio")).toBeInTheDocument());
  });

  it("redirects the login page to the studio", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.getByText("studio")).toBeInTheDocument());
  });
});

describe("entry points on the web (regression guard)", () => {
  beforeEach(() => {
    liffState.sdk = { isInClient: () => false };
  });

  it("keeps the landing page", async () => {
    renderAt("/");
    await waitFor(() => expect(screen.queryByText("studio")).not.toBeInTheDocument());
    expect(screen.getAllByRole("link", { name: /open|開始|studio/i }).length).toBeGreaterThan(0);
  });

  it("keeps the login form", async () => {
    renderAt("/login");
    await waitFor(() => expect(screen.queryByText("studio")).not.toBeInTheDocument());
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/pages.liffEntry.test.tsx`
Expected: FAIL — the two in-client cases time out waiting for "studio"; the landing/login pages render instead.

- [ ] **Step 3: Redirect the landing page**

In `frontend/src/landing/Landing.tsx`, add `Navigate` to the existing router import and the
context import:

```tsx
import { Link, Navigate } from "react-router-dom";
import { useLiffContext } from "../lib/liffContext";
```

Then change the top-level component (line 564) to:

```tsx
export default function Landing() {
  const { isInClient } = useLiffContext();
  // Inside LINE the marketing page is dead weight — the user arrived from a rich menu to use
  // the app, and the LIFF endpoint URL points at /app anyway. This covers a stray "/" hit.
  if (isInClient) return <Navigate to="/app" replace />;
  return (
    <Layout>
      {/* …unchanged… */}
```

- [ ] **Step 4: Redirect the login page**

In `frontend/src/pages/Login.tsx`, add the context import next to the other lib imports:

```tsx
import { useLiffContext } from "../lib/liffContext";
```

Add the hook next to the existing ones (before any early return) and the guard immediately
after the existing `if (user)` guard at line 87:

```tsx
  const { isInClient } = useLiffContext();
```

```tsx
  if (user) return <Navigate to={from} replace />;
  // Inside LINE the silent LIFF token exchange (see lib/auth's auto-login effect) is already
  // running — showing a sign-in form would suggest it failed.
  if (isInClient) return <Navigate to="/app" replace />;
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/pages.liffEntry.test.tsx`
Expected: PASS — 4 tests.

- [ ] **Step 6: Run the full suite**

Run: `cd frontend && yarn test`
Expected: PASS — `landing.test.tsx` and the login page tests render without `LiffProvider`, so they take the web branch and stay green.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/landing/Landing.tsx frontend/src/pages/Login.tsx frontend/src/test/pages.liffEntry.test.tsx
git commit -m "feat(liff): land the LINE entry point on the studio"
```

---

### Task 5: Move language + theme into Settings

**Files:**
- Modify: `frontend/src/pages/Settings.tsx:118-121` (insert a section after Profile)
- Modify: `frontend/src/lib/i18n.tsx` (3 new keys × 2 dictionaries)
- Test: `frontend/src/test/pages.Settings.test.tsx` (append — the file already has the auth mock, the `api.health` stub and a `renderSettings()` helper; a second file would duplicate all three)

**Interfaces:**
- Consumes: `LanguageToggle` (`src/components/LanguageToggle.tsx`) and `ThemeToggle` (`src/components/ThemeToggle.tsx`) — both are self-contained default-export buttons taking no props.
- Produces: new i18n keys `settings.appearance`, `settings.language`, `settings.theme`.

Context: inside LINE there is no navbar, so the language and theme toggles that live in
`Header.tsx:83-84` have nowhere to render. Settings gets its own copy. The web navbar keeps
its toggles — a duplicate control that stays in sync (both read the same i18n/theme context)
is better than a control that only exists on one surface.

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/test/pages.Settings.test.tsx` (the existing `beforeEach` already
mocks `useAuth` and stubs `api.health`, and `renderSettings()` is defined at the top of the
file — reuse both):

```tsx
describe("Settings — appearance", () => {
  it("shows the appearance section with language and theme controls", async () => {
    renderSettings();
    expect(await screen.findByText("Appearance")).toBeInTheDocument();
    expect(screen.getByText("Language")).toBeInTheDocument();
    expect(screen.getByText("Theme")).toBeInTheDocument();
  });

  it("switches language from within Settings", async () => {
    const user = userEvent.setup();
    renderSettings();
    await user.click(await screen.findByRole("button", { name: "中" }));
    expect(screen.getByText("外觀")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/pages.Settings.test.tsx`
Expected: FAIL — `Unable to find an element with the text: Appearance`.

- [ ] **Step 3: Add the i18n keys**

In `frontend/src/lib/i18n.tsx`, in the `en` dict immediately after `"settings.subtitle"` (line 405):

```ts
  "settings.appearance": "Appearance",
  "settings.language": "Language",
  "settings.theme": "Theme",
```

And in `zhHant` immediately after its `"settings.subtitle"` (line 1015):

```ts
  "settings.appearance": "外觀",
  "settings.language": "語言",
  "settings.theme": "主題",
```

- [ ] **Step 4: Add the section**

In `frontend/src/pages/Settings.tsx`, add the two component imports next to the existing ones:

```tsx
import LanguageToggle from "../components/LanguageToggle";
import ThemeToggle from "../components/ThemeToggle";
```

Insert this section between the Profile `</section>` (line 118) and the `{/* Coach model … */}`
comment (line 121):

```tsx
        {/* Appearance — language and theme. The web navbar carries the same two toggles, but
            inside the LINE app there is no navbar (see components/LiffAppShell), so this is
            the only place they exist there. */}
        <section className="mt-10">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-faint">
            {t("settings.appearance")}
          </h2>
          <div className="mt-3 divide-y divide-border-dark overflow-hidden rounded-2xl border border-border-dark bg-surface-dark">
            <div className="flex items-center justify-between gap-4 p-4">
              <span className="text-sm text-content">{t("settings.language")}</span>
              <LanguageToggle />
            </div>
            <div className="flex items-center justify-between gap-4 p-4">
              <span className="text-sm text-content">{t("settings.theme")}</span>
              <ThemeToggle />
            </div>
          </div>
        </section>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/pages.Settings.test.tsx`
Expected: PASS — 2 tests.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/Settings.tsx frontend/src/lib/i18n.tsx frontend/src/test/pages.Settings.test.tsx
git commit -m "feat(settings): surface language and theme in the settings page"
```

---

### Task 6: Pre-emptive camera hint on the games hub

**Files:**
- Modify: `frontend/src/pages/Games.tsx:70-88` (insert below the heading block)
- Modify: `frontend/src/lib/i18n.tsx` (1 new key × 2 dictionaries)
- Test: `frontend/src/test/pages.Games.liffHint.test.tsx`

**Interfaces:**
- Consumes: `useLiffContext()` (Task 1).
- Produces: new i18n key `games.liffCameraHint`.

Context: the *reactive* degradation already exists and is not being touched — `lib/camera.ts`
turns the LIFF-on-iOS `getUserMedia` hang into a catchable `CameraError`, and both games
already show `camera.liffHint` when it fires (`SixSeven.tsx:255`, `FruitNinja.tsx:330`). This
task only warns *before* the player taps Play, so a broken camera isn't a surprise.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/test/pages.Games.liffHint.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const { liffState } = vi.hoisted(() => ({
  liffState: { configured: true, sdk: null as unknown },
}));

vi.mock("../lib/liff", () => ({
  isLiffConfigured: () => liffState.configured,
  initLiff: () => Promise.resolve(liffState.sdk),
}));

import Games from "../pages/Games";
import { LiffProvider } from "../lib/liffContext";
import { I18nProvider } from "../lib/i18n";
import { AuthProvider } from "../lib/auth";

const HINT = /camera may not work inside LINE/i;

const renderGames = () =>
  render(
    <MemoryRouter initialEntries={["/games"]}>
      <LiffProvider>
        <AuthProvider>
          <I18nProvider>
            <Games />
          </I18nProvider>
        </AuthProvider>
      </LiffProvider>
    </MemoryRouter>
  );

beforeEach(() => {
  liffState.configured = true;
  liffState.sdk = { isInClient: () => false };
});

describe("Games — camera hint", () => {
  it("warns about the LINE in-app camera when running in-client", async () => {
    liffState.sdk = { isInClient: () => true };
    renderGames();
    await waitFor(() => expect(screen.getByText(HINT)).toBeInTheDocument());
  });

  it("shows no hint on the web", async () => {
    renderGames();
    await waitFor(() => expect(screen.getByText("Pose Arcade")).toBeInTheDocument());
    expect(screen.queryByText(HINT)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && yarn test src/test/pages.Games.liffHint.test.tsx`
Expected: FAIL — the in-client case can't find the hint text.

- [ ] **Step 3: Add the i18n key**

In `frontend/src/lib/i18n.tsx`, in the `en` dict next to the other `games.*` keys (after
`"games.totalEmpty"`):

```ts
  "games.liffCameraHint":
    "The live camera may not work inside LINE. If a game won't start, tap ⋮ at the top right and choose \"Open in browser\".",
```

And in `zhHant`, in the matching position:

```ts
  "games.liffCameraHint":
    "在 LINE 內即時相機可能無法使用。若遊戲開不起來，請點右上角 ⋮ 選「用其他瀏覽器開啟」。",
```

- [ ] **Step 4: Add the hint**

In `frontend/src/pages/Games.tsx`, add the imports:

```tsx
import { Warning } from "@phosphor-icons/react";
import { useLiffContext } from "../lib/liffContext";
```

(`Warning` joins the existing `@phosphor-icons/react` import — merge it into that statement
rather than adding a second one.)

Add the hook next to the page's existing hooks:

```tsx
  const { isInClient } = useLiffContext();
```

Insert the banner immediately after the `{t("games.sub")}` paragraph:

```tsx
          {/* Inside LINE, getUserMedia can hang on iOS (see lib/camera). The games already
              recover from that after the fact; this warns before the player commits. */}
          {isInClient && (
            <div className="mt-5 flex items-start gap-3 rounded-2xl border border-amber-500/25 bg-amber-500/[0.07] p-4">
              <Warning size={20} weight="fill" className="mt-0.5 shrink-0 text-amber-400" />
              <p className="text-sm leading-relaxed text-muted">{t("games.liffCameraHint")}</p>
            </div>
          )}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && yarn test src/test/pages.Games.liffHint.test.tsx`
Expected: PASS — 2 tests.

- [ ] **Step 6: Run the full suite and the coverage gate**

Run: `cd frontend && yarn test`
Expected: PASS — everything green.

Run: `cd frontend && yarn test:coverage`
Expected: PASS — thresholds met (lines 70 / functions 70 / branches 60 / statements 68).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/Games.tsx frontend/src/lib/i18n.tsx frontend/src/test/pages.Games.liffHint.test.tsx
git commit -m "feat(liff): warn about the in-app camera before a game starts"
```

---

### Task 7: Verify in a real LIFF browser and document the console setup

**Files:**
- Modify: `docs/line-login-liff-setup.md` (append a section)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Build to catch type errors the test run can miss**

Run: `cd frontend && yarn build`
Expected: exit 0, no TypeScript errors.

- [ ] **Step 2: Point the LIFF app at the studio**

In the LINE Developers console, open the LIFF app under the LINE Login channel and set:
- Endpoint URL: `https://<ngrok-or-prod-host>/app`
- Size: `Full`

No new scopes are needed for this plan.

- [ ] **Step 3: Verify on a real phone**

Open the LIFF URL from LINE and confirm, on both iOS and Android if available:
1. The bottom tab bar appears; the sidebar and marketing navbar do not.
2. All four tabs navigate, and the active tab is highlighted.
3. The tab bar clears the home indicator (nothing is cut off at the bottom).
4. Opening the LIFF root (`https://<host>/`) lands on the studio, not the landing page.
5. The games hub shows the camera warning.
6. Language and theme can be changed from Settings.

Record the outcome (device, OS version, what worked) — if the tab bar is clipped or a tab
fails to navigate, stop and report rather than patching blind.

- [ ] **Step 4: Document the setup**

Append to `docs/line-login-liff-setup.md`:

```markdown
## LIFF app shell (2026-07-20)

Inside the LINE app the SPA renders `LiffAppShell` (bottom tabs: 分析 / 歷史 / 遊戲 / 設定)
instead of the web navbar + sidebar. Detection lives in `frontend/src/lib/liffContext.tsx`;
`AppLayout` branches on it, so no page component knows about LIFF.

Console setup this requires:

- LIFF app **Endpoint URL** → `https://<host>/app` (the shell also redirects `/` and `/login`
  to `/app` when in-client, but the endpoint URL is the real entry point).
- LIFF app **Size** → `Full`.
- No additional scopes. `liff.sendMessages()` / share target picker are a later phase and DO
  need `chat_message.write` plus the share-target-picker toggle.

Rich menu items can deep-link to any of `/app`, `/history`, `/games`, `/settings` — the SPA's
existing routes resolve them directly.
```

- [ ] **Step 5: Commit**

```bash
git add docs/line-login-liff-setup.md
git commit -m "docs: record the LIFF app shell console setup"
```
