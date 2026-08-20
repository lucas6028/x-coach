import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import { ALL_MOVEMENTS } from "../lib/movements";
import { api } from "../api";

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

// AppLayout renders real chrome (Header + desktop/mobile Sidebar), which contributes its own
// <button> elements (nav toggles, "New analysis", language/theme menus). Scoping every query to
// the "Analyze" accessible name is what the pre-existing tests below already did, and it is what
// keeps the movement-card assertions from being polluted by that chrome. The rail's own item is
// spelled "Analyse", so the American spelling here matches only the cards.
const liveButtons = () => screen.getAllByRole("button", { name: /Analyze/ });

// The card is a <div> inside the section's <li>, not one big button: it has two destinations now
// (the studio and the movement's detail page), so queries reach a card through its <li> and then
// ask for the control they mean.
const card = (name: string) => screen.getByText(name).closest("li")!;
const analyzeIn = (name: string) =>
  within(card(name)).getByRole("button", { name: /Analyze/ });

const LIVE = [
  { name: "Squat", validated: true },
  { name: "Overhead Press", validated: false },
  { name: "Push-up", validated: false },
];

beforeEach(() => {
  navigate.mockClear();
  localStorage.clear();
  vi.spyOn(api, "getMovements").mockResolvedValue(LIVE);
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

describe("Movements page", () => {
  it("makes exactly the analyzable movements actionable", async () => {
    renderWithProviders(<Movements />);
    await waitFor(() => expect(liveButtons()).toHaveLength(LIVE.length));
  });

  it("lists the rest as inert, not as disabled buttons", async () => {
    renderWithProviders(<Movements />);
    await waitFor(() => expect(liveButtons()).toHaveLength(LIVE.length));
    expect(screen.getAllByText(/Soon|即將開放/).length).toBe(ALL_MOVEMENTS.length - LIVE.length);
  });

  it("tags unvalidated movements Beta and leaves Squat untagged", async () => {
    renderWithProviders(<Movements />);
    const betas = await screen.findAllByText("Beta");
    expect(betas).toHaveLength(2);
    expect(within(card("Squat")).queryByText("Beta")).toBeNull();
    // Named, not just counted: Overhead Press specifically carries the tag Squat does not.
    expect(within(card("Overhead Press")).getByText("Beta")).toBeInTheDocument();
  });

  it("navigates to the studio with the chosen movement", async () => {
    renderWithProviders(<Movements />);
    await waitFor(() => expect(liveButtons()).toHaveLength(LIVE.length));
    await userEvent.click(analyzeIn("Push-up"));
    expect(navigate).toHaveBeenCalledWith("/app?movement=Push-up");
  });

  it("navigates to the studio with a space-containing movement name, percent-encoded", async () => {
    renderWithProviders(<Movements />);
    await waitFor(() => expect(liveButtons()).toHaveLength(LIVE.length));
    // Names Overhead Press explicitly (the count-only checks above would also pass if this card
    // were actually some other movement, e.g. Row, misrouted by a typo'd name match) and pins the
    // one interesting case for encodeURIComponent: a movement name containing a space.
    await userEvent.click(analyzeIn("Overhead Press"));
    expect(navigate).toHaveBeenCalledWith("/app?movement=Overhead%20Press");
  });

  it("gives every card a details link, analyzable or not", async () => {
    renderWithProviders(<Movements />);
    await waitFor(() => expect(liveButtons()).toHaveLength(LIVE.length));
    // All sixteen, because the detail page is knowledge about the movement rather than a result
    // of analysing one — the thirteen inert cards are exactly the ones that had no destination
    // before this link existed.
    expect(screen.getAllByRole("link", { name: /View details/ })).toHaveLength(
      ALL_MOVEMENTS.length
    );
    // The href carries the canonical name, percent-encoded — the same identity /app?movement= uses.
    expect(
      within(card("Overhead Press")).getByRole("link", { name: /View details/ })
    ).toHaveAttribute("href", "/movements/Overhead%20Press");
    // A locked movement links out too.
    expect(within(card("Bicep Curl")).getByRole("link", { name: /View details/ })).toHaveAttribute(
      "href",
      "/movements/Bicep%20Curl"
    );
  });

  it("narrows to one body region when a filter pill is picked", async () => {
    renderWithProviders(<Movements />);
    await userEvent.click(screen.getByRole("button", { name: "Upper body" }));
    // One section left, and it is the right one: an upper-body movement is present and a
    // lower-body one is gone.
    expect(screen.getAllByRole("list")).toHaveLength(1);
    expect(screen.getByRole("heading", { name: "Upper body" })).toBeInTheDocument();
    expect(screen.getByText("Row")).toBeInTheDocument();
    expect(screen.queryByText("Squat")).toBeNull();

    // And the whole catalog comes back — a filter that cannot be undone is a dead end.
    await userEvent.click(screen.getByRole("button", { name: "All" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(ALL_MOVEMENTS.length);
  });

  it("searches across sections and drops the ones left empty", async () => {
    renderWithProviders(<Movements />);
    await userEvent.type(screen.getByRole("searchbox"), "row");
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByText("Row")).toBeInTheDocument();
    // The heading of a section with no surviving card is gone too, not left over nothing.
    expect(screen.queryByRole("heading", { name: "Lower body" })).toBeNull();
  });

  it("searches the label the reader can see, not the English key", async () => {
    localStorage.setItem("lang", "zh-Hant");
    renderWithProviders(<Movements />);
    await userEvent.type(screen.getByRole("searchbox"), "深蹲");
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.getByText("深蹲")).toBeInTheDocument();
  });

  it("says nothing matched rather than showing an empty grid", async () => {
    renderWithProviders(<Movements />);
    await userEvent.type(screen.getByRole("searchbox"), "zzzz");
    expect(screen.queryAllByRole("listitem")).toHaveLength(0);
    expect(screen.getByText(/No movements match/)).toBeInTheDocument();
  });

  it("falls back to Squat-only when the list cannot be fetched", async () => {
    vi.spyOn(api, "getMovements").mockRejectedValue(new Error("offline"));
    renderWithProviders(<Movements />);
    // Positive control: wait for the fetch to actually settle, so this isn't just asserting the
    // initial useState value before the effect has had a chance to run (or fail to run at all).
    await waitFor(() => expect(api.getMovements).toHaveBeenCalled());
    await waitFor(() =>
      expect(screen.getAllByText(/Soon|即將開放/)).toHaveLength(ALL_MOVEMENTS.length - 1)
    );
    expect(liveButtons()).toHaveLength(1);
    // And it is Squat's card that carries it, not just "some card".
    expect(within(card("Squat")).getByRole("button", { name: /Analyze/ })).toBeTruthy();
    expect(within(card("Push-up")).queryByRole("button", { name: /Analyze/ })).toBeNull();
  });
});
