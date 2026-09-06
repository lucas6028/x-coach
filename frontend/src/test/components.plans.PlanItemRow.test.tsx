import { describe, it, expect, vi } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "./renderWithProviders";
import PlanItemRow from "../components/plans/PlanItemRow";
import type { PlanItem } from "../api";

function item(overrides: Partial<PlanItem> = {}): PlanItem {
  return {
    id: "i1",
    plan_id: "p1",
    day_index: 1,
    position: 0,
    movement: "Squat",
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-08-13T00:00:00Z",
    ...overrides,
  };
}

function renderRow(overrides: Partial<PlanItem> = {}, props: Partial<{ analyzable: boolean; busy: boolean; onToggle: () => void; onRemove: () => void }> = {}) {
  return renderWithProviders(
    <ul>
      <PlanItemRow
        item={item(overrides)}
        planId="p1"
        analyzable={props.analyzable ?? true}
        busy={props.busy ?? false}
        onToggle={props.onToggle ?? (() => {})}
        onRemove={props.onRemove ?? (() => {})}
      />
    </ul>
  );
}

describe("PlanItemRow — the figure", () => {
  it("leads with the movements-library art, not a glyph", async () => {
    // The card is recognised by its picture, and it must be the SAME asset the library page uses —
    // a second set of figures for the plan would be a second visual language for one movement.
    // Queried by src because MovementArt is decoration (`alt=""`), so it has no accessible role.
    const { container } = renderRow();
    expect(container.querySelector('img[src="/movements/squat.png"]')).toBeInTheDocument();
  });

  it("falls back to the icon when a movement has no art", () => {
    // Guard for a movement added to the catalog before its PNG ships: the card must still render
    // rather than draw a broken image. MovementIcon draws inline SVG, so no <img> appears at all.
    const { container } = renderRow({ movement: "Not A Real Movement" });
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("gives the art a square stage", () => {
    // Not cosmetic. The source PNGs are opaque pre-matted squares, so a non-square stage leaves a
    // hard-edged white box floating over the gradient (see the comments in MovementCard.tsx).
    const { container } = renderRow();
    const img = container.querySelector('img[src="/movements/squat.png"]') as HTMLElement;
    expect(img.parentElement?.className).toContain("aspect-square");
  });
});

describe("PlanItemRow — three separate controls", () => {
  it("keeps the tick, the studio link and remove as three independently operable controls", async () => {
    // The point of the split: "I trained but didn't film it" has to be recordable, so ticking off
    // and recording cannot be the same control. Removing is a third thing again.
    const onToggle = vi.fn();
    const onRemove = vi.fn();
    renderRow({}, { onToggle, onRemove });

    const tick = screen.getByRole("button", { name: /mark squat as done/i });
    const studio = screen.getByRole("link", { name: /record & analyse/i });
    const remove = screen.getByRole("button", { name: /remove squat/i });
    expect(studio).toHaveAttribute("href", "/app?movement=Squat&plan=p1&plan_item=i1");

    await userEvent.click(tick);
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(onRemove).not.toHaveBeenCalled();

    await userEvent.click(remove);
    expect(onRemove).toHaveBeenCalledTimes(1);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("still says the short word on the chip and the full phrase to assistive tech", () => {
    renderRow();
    const studio = screen.getByRole("link", { name: /record & analyse/i });
    expect(studio).toHaveTextContent("Analyse");
    expect(studio).not.toHaveTextContent(/record/i);
  });

  it("locks every control while a write for this card is in flight", () => {
    renderRow({}, { busy: true });
    expect(screen.getByRole("button", { name: /mark squat as done/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /remove squat/i })).toBeDisabled();
  });

  it("keeps the tick and remove on a movement with no detector", () => {
    // A card with no analysis is still a plannable exercise; only its video analysis is missing.
    renderRow({ movement: "Jumping Jacks" }, { analyzable: false });
    expect(screen.queryByRole("link", { name: /record & analyse/i })).toBeNull();
    expect(screen.getByText("No analysis")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /mark jumping jacks as done/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /remove jumping jacks/i })).toBeInTheDocument();
  });
});

describe("PlanItemRow — the done state", () => {
  it("is announced on the tick and drawn on the card", () => {
    // Two carriers, because the grid is scanned visually and read by name: `aria-pressed` for the
    // reader, the struck-through name plus the filled green tick for the eye.
    const { container } = renderRow({ completed_at: "2026-08-13T01:00:00Z" });
    const tick = screen.getByRole("button", { name: /mark squat as not done/i });
    expect(tick).toHaveAttribute("aria-pressed", "true");
    expect(tick.className).toContain("bg-secondary");

    expect(screen.getByText("Squat").className).toContain("line-through");
    const card = container.querySelector("li") as HTMLElement;
    expect(card.className).toContain("border-secondary");
  });

  it("is not claimed on an unticked card", () => {
    const { container } = renderRow();
    const tick = screen.getByRole("button", { name: /mark squat as done/i });
    expect(tick).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByText("Squat").className).not.toContain("line-through");
    expect((container.querySelector("li") as HTMLElement).className).not.toContain("border-secondary");
  });

  it("shows the report instead of the studio once the item carries an analysis", () => {
    renderRow({ analysis_id: "an-9", completed_at: "2026-08-13T01:00:00Z" });
    const report = screen.getByRole("link", { name: /view report/i });
    expect(report).toHaveAttribute("href", "/app?analysis=an-9");
    expect(screen.queryByRole("link", { name: /record & analyse/i })).toBeNull();
    // Still removable and still untickable-back — the report replaces the studio link only.
    expect(screen.getByRole("button", { name: /mark squat as not done/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /remove squat/i })).toBeInTheDocument();
  });
});

describe("PlanItemRow — the text under the figure", () => {
  it("truncates a long movement name instead of widening the card", () => {
    // A card that grows to fit its longest name breaks the grid it sits in.
    renderRow({ movement: "Band Pull Apart" });
    expect(screen.getByText("Band Pull Apart").className).toContain("truncate");
  });

  it("shows sets, reps and the note", () => {
    renderRow({ notes: "slow eccentric" });
    expect(screen.getByText(/3 × 10/)).toBeInTheDocument();
    expect(screen.getByText("slow eccentric")).toBeInTheDocument();
  });
});

describe("PlanItemRow — in the day panel", () => {
  it("renders inside a list item so a day is still a list of exercises", () => {
    const { container } = renderRow();
    const li = container.querySelector("li") as HTMLElement;
    expect(within(li).getByText("Squat")).toBeInTheDocument();
  });
});
