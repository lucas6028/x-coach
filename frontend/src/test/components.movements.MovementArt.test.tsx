import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { existsSync } from "node:fs";
import { join } from "node:path";
import MovementArt, { ART } from "../components/movements/MovementArt";
import { GLYPHS } from "../components/movements/MovementIcon";
import { ALL_MOVEMENTS } from "../lib/movements";

// The eight movements the reference actually drew.
const DRAWN = [
  "Squat",
  "Lunge",
  "Deadlift",
  "Push-up",
  "Overhead Press",
  "Row",
  "Sit-up",
  "Jumping Jacks",
];

describe("MovementArt", () => {
  it("draws the illustration for a movement that has one", () => {
    const { container } = render(<MovementArt movement="Squat" />);
    const img = container.querySelector("img")!;
    expect(img).toHaveAttribute("src", "/movements/squat.png");
    // Empty alt, not the movement name: the card names it in text right above the tile.
    expect(img).toHaveAttribute("alt", "");
  });

  it("falls back to that movement's own drawn figure, not a broken image", () => {
    const { container } = render(<MovementArt movement="Torso Twist" />);
    expect(container.querySelector("img")).toBeNull();
    // The same glyph the title carries, drawn large — a card with no illustration still shows the
    // movement it is for rather than a generic stand-in.
    const paths = container.querySelectorAll("path");
    expect(paths).toHaveLength(GLYPHS["Torso Twist"].strokes.length);
  });

  it("keys its figures to names the catalog actually uses", () => {
    // A near-miss key ("Jumping Jack" for "Jumping Jacks") is otherwise invisible: the card just
    // shows the placeholder and every other assertion still passes.
    for (const m of DRAWN) expect(ALL_MOVEMENTS as readonly string[], m).toContain(m);
    expect(Object.keys(ART).sort()).toEqual([...DRAWN].sort());
  });

  it("points every figure at a file that is actually in public/", () => {
    // Nothing else catches a typo'd or uncommitted filename -- React renders the <img> happily and
    // the page just shows a broken tile.
    for (const m of DRAWN) {
      const { container } = render(<MovementArt movement={m} />);
      const src = container.querySelector("img")!.getAttribute("src")!;
      expect(existsSync(join(process.cwd(), "public", src)), `${m} -> ${src}`).toBe(true);
    }
  });
});
