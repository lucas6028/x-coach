import { describe, it, expect, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import WhileYouWait, { hasMovementGuide } from "../components/WhileYouWait";
import { renderWithProviders } from "./renderWithProviders";
import { movementDetail } from "../lib/movementDetail";
import { movementMistakes } from "../lib/movementMistakes";

// A pass-through spy, so ONE test can stand in for a movement with steps but no rules. Since
// 2026-09-26 no catalog movement is like that (Jumping Jacks and High Knee were the last), but
// the branch is still reachable by a movement designed ahead of its detector. Every other test
// in this file reads the real content through it.
vi.mock("../lib/movementMistakes", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/movementMistakes")>();
  return { ...actual, movementMistakes: vi.fn(actual.movementMistakes) };
});

// The panel reads the real content modules rather than a fixture: its whole point is that it shows
// the SAME steps and mistakes the movement detail page shows, so pinning it against invented data
// would pass while the two had drifted apart.
describe("WhileYouWait", () => {
  it("opens on the movement's steps, in the authored order", () => {
    renderWithProviders(<WhileYouWait movement="Squat" />);
    const steps = movementDetail("Squat")!.steps;
    expect(screen.getByText(steps[0].text.en)).toBeInTheDocument();
    expect(screen.getByText(steps[steps.length - 1].text.en)).toBeInTheDocument();
  });

  it("switches to the common mistakes the detector can actually report", () => {
    renderWithProviders(<WhileYouWait movement="Squat" />);
    fireEvent.click(screen.getByRole("tab", { name: /common mistakes/i }));

    const first = movementMistakes("Squat")[0];
    expect(screen.getByText(first.title.en)).toBeInTheDocument();
    expect(screen.getByText(first.subtitle.en)).toBeInTheDocument();
    // One cue only — the rest of the fix list stays behind the full guide.
    expect(screen.getByText(first.fixes.en[0])).toBeInTheDocument();
    expect(screen.queryByText(first.fixes.en[1])).not.toBeInTheDocument();
    // The steps tab's content is gone, not merely hidden.
    expect(screen.queryByText(movementDetail("Squat")!.steps[0].text.en)).not.toBeInTheDocument();
  });

  // A movement with steps but no registered rules has nothing to warn about. The panel drops the
  // tab strip rather than offering a tab that opens an empty list.
  it("drops the tab strip for a movement with steps but no registered rules", () => {
    const spy = vi.mocked(movementMistakes);
    const real = spy.getMockImplementation()!;
    spy.mockImplementation(() => []);
    try {
      renderWithProviders(<WhileYouWait movement="Jumping Jacks" />);
      expect(screen.queryAllByRole("tab")).toHaveLength(0);
      expect(
        screen.getByText(movementDetail("Jumping Jacks")!.steps[0].text.en)
      ).toBeInTheDocument();
    } finally {
      spy.mockImplementation(real);
    }
    // Non-vacuity: without the stub, the same movement now DOES show the tabs.
    expect(movementMistakes("Jumping Jacks").length).toBeGreaterThan(0);
  });

  it("links out to the movement's own detail page", () => {
    renderWithProviders(<WhileYouWait movement="Push-up" />);
    expect(screen.getByRole("link", { name: /full guide/i })).toHaveAttribute(
      "href",
      "/movements/Push-up"
    );
  });

  // An uncatalogued movement has no steps and no rules, so the caller must be able to keep the
  // default "what comes back" panel instead of rendering an empty shell.
  it("reports no guide, and renders nothing, for a movement outside the catalog", () => {
    expect(hasMovementGuide("Cartwheel")).toBe(false);
    expect(hasMovementGuide("Squat")).toBe(true);
    const { container } = renderWithProviders(<WhileYouWait movement="Cartwheel" />);
    expect(container).toBeEmptyDOMElement();
  });
});
