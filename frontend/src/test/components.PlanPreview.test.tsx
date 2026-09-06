import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { I18nProvider } from "../lib/i18n";
import PlanPreview from "../components/plans/PlanPreview";
import type { Plan, PlanItem } from "../api";

function item(day: number, movement: string): PlanItem {
  return {
    id: `${movement}-${day}`,
    plan_id: "p1",
    day_index: day,
    position: 0,
    movement,
    sets: 3,
    reps: 10,
    notes: null,
    completed_at: null,
    analysis_id: null,
    created_at: "2026-01-01T00:00:00Z",
  };
}

function plan(items: PlanItem[]): Plan {
  return {
    id: "p1",
    name: "zzq-plan",
    notes: null,
    template_key: null,
    started_at: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    items,
  };
}

function renderPreview(items: PlanItem[], compact?: boolean) {
  return render(
    <I18nProvider>
      <PlanPreview plan={plan(items)} compact={compact} />
    </I18nProvider>
  );
}

// The localStorage stub in src/test/setup.ts is one Map for the whole file, so the language a
// test chooses would otherwise leak into the next one.
beforeEach(() => localStorage.clear());

describe("PlanPreview rest days", () => {
  it("folds the trailing empty days of a three-day plan into one line", () => {
    // Four of seven bands saying nothing reads as a plan that failed to finish, not as a week
    // with rest in it. The days that DO have work still get their own band.
    renderPreview([item(1, "Squat"), item(2, "Push-up"), item(3, "Plank")]);

    expect(screen.getByText(/Day 4–7 · Rest/)).toBeInTheDocument();
    expect(screen.queryByText("Day 4")).toBeNull();
    expect(screen.queryByText("Day 7")).toBeNull();
    expect(screen.getByText("Day 1")).toBeInTheDocument();
    expect(screen.getByText("Day 3")).toBeInTheDocument();
  });

  it("leaves a lone empty day between two working days as its own band", () => {
    // "Day 2–2 · Rest" would be worse than the band that already says "Rest day", so a run of one
    // is not a run.
    renderPreview([item(1, "Squat"), item(3, "Plank")]);

    expect(screen.getByText("Day 2")).toBeInTheDocument();
    expect(screen.getByText("Rest day")).toBeInTheDocument();
    expect(screen.queryByText(/Day 2–/)).toBeNull();
    // The trailing run 4–7 still collapses, so exactly one range line exists.
    expect(screen.getByText(/Day 4–7 · Rest/)).toBeInTheDocument();
  });
});

describe("PlanPreview muscle coverage", () => {
  it("shows what the drafted plan trains, under the day bands", () => {
    // Squat (quads/glutes) plus a push-up (chest/triceps): the summary fills in from the same
    // streamed plan the bands are drawn from, so it appears as Lumen writes.
    renderPreview([item(1, "Squat"), item(2, "Push-up")]);

    const card = screen.getByRole("heading", { name: /what this plan trains/i }).closest("section");
    expect(card).not.toBeNull();
    expect(within(card as HTMLElement).getByText("Quadriceps")).toBeInTheDocument();
    expect(within(card as HTMLElement).getByText("Chest")).toBeInTheDocument();
    // Neither movement touches the back, so the week is not balanced yet and says so.
    expect(within(card as HTMLElement).getByText(/not trained this week/i)).toHaveTextContent(
      /lats/i
    );
  });

  it("says the week is balanced instead of showing an empty gap line", () => {
    const items = [
      item(1, "Squat"),
      item(2, "Deadlift"),
      item(3, "Push-up"),
      item(4, "Row"),
      item(5, "Overhead Press"),
    ];
    renderPreview(items);

    expect(screen.getByText(/covers every major muscle group/i)).toBeInTheDocument();
    expect(screen.queryByText(/not trained this week/i)).toBeNull();
  });

  it("renders nothing at all when no item's movement is in the catalog", () => {
    // An empty plan must not read "not trained this week: everything".
    renderPreview([item(1, "Plank")]);
    expect(screen.queryByRole("heading", { name: /what this plan trains/i })).toBeNull();
  });

  it("names the gaps in Traditional Chinese with a Chinese list separator", () => {
    // The joiner is language-dependent — "腿後肌、上背", not "腿後肌, 上背".
    localStorage.setItem("lang", "zh-Hant");
    renderPreview([item(1, "Squat"), item(2, "Push-up")]);

    expect(screen.getByRole("heading", { name: "這份菜單練到的部位" })).toBeInTheDocument();
    expect(screen.getByText(/這週還沒練到：/)).toHaveTextContent("上背、闊背肌");
  });

  it("drops the body maps but keeps the groups when compact", () => {
    const { container } = renderPreview([item(1, "Squat")], true);

    expect(screen.getByRole("heading", { name: /what this plan trains/i })).toBeInTheDocument();
    expect(screen.getByText("Quadriceps")).toBeInTheDocument();
    expect(screen.queryByText("Anterior")).toBeNull();
    expect(screen.queryByText("Posterior")).toBeNull();
    // The plate is the thing compact drops; the groups beside it are the thing it keeps.
    // Scoped to the anatomical plates rather than every `img` on the page: the exercise rows now
    // carry the movement-library figure too, and counting those would make this assertion fail for
    // a reason that has nothing to do with what compact drops.
    expect(container.querySelectorAll('img[src*="muscles-worked"]').length).toBe(0);
    // Still present, so the scoping above cannot pass by finding no images at all.
    expect(container.querySelectorAll('img[src^="/movements/"]').length).toBeGreaterThan(0);
  });

  it("layers one shipped plate per distinct movement", () => {
    // Squat twice: the union of two identical images is the one image, so the repeat must not
    // become a second layer to download and composite.
    renderPreview([item(1, "Squat"), item(2, "Squat"), item(3, "Push-up")]);

    const card = screen.getByRole("heading", { name: /what this plan trains/i }).closest("section");
    // `alt=""` makes these presentational, so they have no `img` ROLE to query by -- the legend
    // beside them is what names the muscles to a screen reader.
    const layers = Array.from((card as HTMLElement).querySelectorAll("img"));
    expect(layers.map((img) => img.getAttribute("src"))).toEqual([
      "/movements/muscles-worked/squat.webp",
      "/movements/muscles-worked/push-up.webp",
    ]);
    for (const layer of layers) {
      // `darken` (not multiply, which accumulates the grey of every layer) is what makes the
      // stack read as ONE body with both movements' muscles lit.
      expect(layer.style.mixBlendMode).toBe("darken");
      expect(layer.getAttribute("alt")).toBe("");
      expect(layer.getAttribute("loading")).toBe("lazy");
      expect(layer.getAttribute("decoding")).toBe("async");
    }
  });
});
