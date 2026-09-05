import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nProvider } from "../lib/i18n";
import WeekStrip from "../components/plans/WeekStrip";
import { itemsByDay } from "../lib/plans";
import type { PlanItem } from "../api";

function item(overrides: Partial<PlanItem> & { id: string; day_index: number }): PlanItem {
  return {
    plan_id: "p1",
    position: 0,
    movement: "Squat",
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-09-01T00:00:00Z",
    ...overrides,
  };
}

// Day 1: two exercises, both done. Day 3: two exercises, one done. Everything else is a rest day.
const ITEMS: PlanItem[] = [
  item({ id: "a", day_index: 1, completed_at: "2026-09-01T10:00:00Z" }),
  item({ id: "b", day_index: 1, movement: "Push-up", completed_at: "2026-09-01T10:05:00Z" }),
  item({ id: "c", day_index: 3, movement: "Row" }),
  item({ id: "d", day_index: 3, movement: "Sit-up", completed_at: "2026-09-03T10:00:00Z" }),
];

function renderStrip(props: Partial<Parameters<typeof WeekStrip>[0]> = {}) {
  const onSelect = vi.fn();
  render(
    <I18nProvider>
      <WeekStrip
        days={itemsByDay(ITEMS)}
        selected={3}
        today={3}
        onSelect={onSelect}
        panelId="panel"
        tabId={(d) => `tab-${d}`}
        {...props}
      />
    </I18nProvider>
  );
  return { onSelect };
}

// A tab's accessible name is its whole summary run together — jsdom has no layout, so
// dom-accessibility-api joins the inline spans without spaces ("Day 3Today1/2Abdominals+2"). Hence
// a prefix match and no trailing word boundary.
const tab = (day: number) => screen.getByRole("tab", { name: new RegExp(`^day ${day}`, "i") });

describe("WeekStrip — the week as one object", () => {
  it("draws all seven days as tabs of one tablist", async () => {
    renderStrip();
    const strip = screen.getByRole("tablist");
    const tabs = within(strip).getAllByRole("tab");
    expect(tabs).toHaveLength(7);
    tabs.forEach((el, i) => expect(el).toHaveTextContent(`Day ${i + 1}`));
    expect(tabs[0]).toHaveAttribute("aria-controls", "panel");
    expect(tabs[0]).toHaveAttribute("id", "tab-1");
  });

  it("shows each training day's done-of-total count and top muscle group", async () => {
    renderStrip();
    // Day 3 is one of two done; day 1 is two of two.
    expect(within(tab(3)).getByText("1/2")).toBeInTheDocument();
    expect(within(tab(1)).getByText("2/2")).toBeInTheDocument();
    // Row + Sit-up prime abs, upper back and lats; the tile names the first and counts the rest.
    expect(within(tab(3)).getByText("Abdominals")).toBeInTheDocument();
    expect(within(tab(3)).getByText("+2")).toBeInTheDocument();
    expect(within(tab(3)).queryByText("Lats")).toBeNull();
  });

  it("marks a rest day as rest, with no count and no muscles", async () => {
    renderStrip();
    const rest = tab(2);
    expect(within(rest).getByText("Rest")).toBeInTheDocument();
    expect(within(rest).queryByText("0/0")).toBeNull();
    expect(rest.textContent).not.toMatch(/glutes|quadriceps/i);
  });

  it("says when a day is fully done", async () => {
    renderStrip();
    // The tick is decorative; the words are what a screen reader gets.
    expect(within(tab(1)).getByText("All done")).toBeInTheDocument();
    expect(within(tab(3)).queryByText("All done")).toBeNull();
  });

  it("marks today, and marks it separately from the selection", async () => {
    // Today and selected are DIFFERENT days here, so each must be identifiable on its own.
    renderStrip({ selected: 5, today: 2 });
    expect(within(tab(2)).getByText("Today")).toBeInTheDocument();
    expect(tab(2)).toHaveAttribute("aria-selected", "false");
    expect(tab(5)).toHaveAttribute("aria-selected", "true");
    expect(within(tab(5)).queryByText("Today")).toBeNull();
  });

  it("keeps both badges when today IS the selected day", async () => {
    renderStrip({ selected: 3, today: 3 });
    expect(within(tab(3)).getByText("Today")).toBeInTheDocument();
    expect(tab(3)).toHaveAttribute("aria-selected", "true");
  });

  it("says nothing is today on a finished plan", async () => {
    renderStrip({ today: null });
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("marks exactly one tab selected, and gives only that one a tab stop", async () => {
    renderStrip({ selected: 4 });
    const tabs = screen.getAllByRole("tab");
    expect(tabs.filter((el) => el.getAttribute("aria-selected") === "true")).toHaveLength(1);
    // Roving tabindex: the strip is one stop, then arrows inside it.
    expect(tabs.filter((el) => el.getAttribute("tabindex") === "0")).toHaveLength(1);
    expect(tab(4)).toHaveAttribute("tabindex", "0");
  });
});

describe("WeekStrip — choosing a day", () => {
  it("selects the day that was clicked", async () => {
    const { onSelect } = renderStrip();
    await userEvent.click(tab(6));
    expect(onSelect).toHaveBeenCalledWith(6);
  });

  it("moves with the arrow keys, taking focus with it", async () => {
    const { onSelect } = renderStrip({ selected: 3 });
    tab(3).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenCalledWith(4);
    // Focus follows selection — the next arrow press must act on the day now under the cursor,
    // not on the one the parent still calls `selected` until it re-renders.
    expect(tab(4)).toHaveFocus();

    onSelect.mockClear();
    tab(3).focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it("wraps at both ends, because a week is a cycle", async () => {
    const { onSelect } = renderStrip({ selected: 7 });
    tab(7).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(onSelect).toHaveBeenCalledWith(1);

    onSelect.mockClear();
    tab(1).focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(onSelect).toHaveBeenCalledWith(7);
  });

  it("jumps to the ends with Home and End", async () => {
    const { onSelect } = renderStrip({ selected: 4 });
    tab(4).focus();
    await userEvent.keyboard("{Home}");
    expect(onSelect).toHaveBeenCalledWith(1);

    onSelect.mockClear();
    await userEvent.keyboard("{End}");
    expect(onSelect).toHaveBeenCalledWith(7);
  });

  it("brings the selected tile into view without dragging the page with it", async () => {
    // The app shell is a fixed box with its own internal scroll region, so a default
    // scrollIntoView would scroll that region as well as the strip. `block: nearest` is the part
    // that keeps the scrolling inside the strip, which is why it is pinned here.
    const scrollIntoView = vi.fn();
    const real = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = scrollIntoView;
    try {
      renderStrip({ selected: 5 });
      expect(scrollIntoView).toHaveBeenCalledTimes(1);
      expect(scrollIntoView.mock.instances[0]).toBe(tab(5));
      expect(scrollIntoView).toHaveBeenCalledWith(
        expect.objectContaining({ block: "nearest", inline: "nearest" })
      );
    } finally {
      Element.prototype.scrollIntoView = real;
    }
  });

  it("leaves other keys alone", async () => {
    const { onSelect } = renderStrip({ selected: 4 });
    tab(4).focus();
    await userEvent.keyboard("{ArrowDown}");
    expect(onSelect).not.toHaveBeenCalled();
  });
});
