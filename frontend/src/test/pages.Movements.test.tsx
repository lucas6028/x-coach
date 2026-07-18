import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { ALL_MOVEMENTS, ANALYZABLE_MOVEMENTS } from "../lib/movements";

// The global setup freezes requestAnimationFrame, so motion's reveal would never settle under test.
// The stagger is declarative config; mount the primitives synchronously and pass DOM props through.
vi.mock("motion/react", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  const cache: Record<string, React.ComponentType<Record<string, unknown>>> = {};
  const motion = new Proxy({} as Record<string, unknown>, {
    get: (_t, tag: string) => {
      if (!cache[tag]) {
        cache[tag] = React.forwardRef<unknown, Record<string, unknown>>((props, ref) => {
          const { initial, animate, exit, transition, variants, children, ...rest } = props;
          void initial; void animate; void exit; void transition; void variants;
          return React.createElement(tag, { ...rest, ref }, children as React.ReactNode);
        }) as unknown as React.ComponentType<Record<string, unknown>>;
      }
      return cache[tag];
    },
  });
  return {
    __esModule: true,
    motion,
    AnimatePresence: ({ children }: { children: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    useReducedMotion: () => true,
  };
});

const navigate = vi.fn();
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigate };
});

import Movements from "../pages/Movements";

beforeEach(() => {
  navigate.mockClear();
  localStorage.clear();
});
afterEach(() => vi.clearAllMocks());

describe("Movements menu", () => {
  it("lists every movement in the catalog at once, with no picker to open first", () => {
    renderWithProviders(<Movements />);
    // The whole menu is on the page: 16 movements, nothing hidden behind a dropdown.
    for (const m of ALL_MOVEMENTS) expect(screen.getByText(m)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(ALL_MOVEMENTS.length);
  });

  it("groups the catalog by body region", () => {
    renderWithProviders(<Movements />);
    for (const label of ["Lower body", "Upper body", "Core", "Full body"]) {
      expect(screen.getByRole("heading", { name: label })).toBeInTheDocument();
    }
    expect(screen.getAllByRole("list")).toHaveLength(4);
  });

  it("makes exactly the analyzable movements clickable; every other card is inert", () => {
    renderWithProviders(<Movements />);
    const buttons = screen.getAllByRole("button", { name: /Analyze a video/ });
    expect(buttons).toHaveLength(ANALYZABLE_MOVEMENTS.length);
    // The single live card is the Squat one, not just "some" card.
    expect(within(buttons[0]).getByText("Squat")).toBeInTheDocument();
  });

  it("marks the movements that have no detector as Soon rather than as broken buttons", () => {
    renderWithProviders(<Movements />);
    const locked = ALL_MOVEMENTS.length - ANALYZABLE_MOVEMENTS.length;
    expect(screen.getAllByText("Soon")).toHaveLength(locked);
    // A "Soon" movement must not be reachable as a control at all (a disabled <button> would still
    // announce itself as an action that exists).
    expect(screen.queryByRole("button", { name: /Deadlift/ })).not.toBeInTheDocument();
  });

  it("hands off to the analysis studio when a live movement is picked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Movements />);
    await user.click(screen.getByRole("button", { name: /Squat/ }));
    expect(navigate).toHaveBeenCalledWith("/app");
  });

  it("renders localised movement names in Traditional Chinese", () => {
    localStorage.setItem("lang", "zh-Hant");
    renderWithProviders(<Movements />);
    expect(screen.getByText("深蹲")).toBeInTheDocument();
    expect(screen.getByText("硬舉")).toBeInTheDocument();
    expect(screen.getByText("開合跳")).toBeInTheDocument();
    // The English canonical name is the data key, never shown to a zh reader.
    expect(screen.queryByText("Deadlift")).not.toBeInTheDocument();
  });
});
