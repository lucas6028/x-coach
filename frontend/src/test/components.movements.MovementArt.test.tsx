import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { existsSync } from "node:fs";
import { join } from "node:path";
import MovementArt, { ART } from "../components/movements/MovementArt";
import { ALL_MOVEMENTS } from "../lib/movements";

describe("MovementArt", () => {
  it("draws the illustration for a movement that has one", () => {
    const { container } = render(<MovementArt movement="Squat" />);
    const img = container.querySelector("img")!;
    expect(img).toHaveAttribute("src", "/movements/squat.png");
    // Empty alt, not the movement name: the card names it in text right above the tile.
    expect(img).toHaveAttribute("alt", "");
  });

  it("falls back to a drawn figure, not a broken image, for a movement without art", () => {
    // No catalog movement is actually missing art today (see the test below) -- this guards the
    // fallback branch itself, which exists for a future movement added before its art ships.
    const { container } = render(<MovementArt movement="Not A Real Movement" />);
    expect(container.querySelector("img")).toBeNull();
  });

  it("keys its figures to names the catalog actually uses", () => {
    // A near-miss key ("Jumping Jack" for "Jumping Jacks") is otherwise invisible: the card just
    // shows the placeholder and every other assertion still passes.
    for (const m of ALL_MOVEMENTS) expect(ART, m).toHaveProperty(m);
    expect(Object.keys(ART).sort()).toEqual([...ALL_MOVEMENTS].sort());
  });

  it("points every figure at a file that is actually in public/", () => {
    // Nothing else catches a typo'd or uncommitted filename -- React renders the <img> happily and
    // the page just shows a broken tile.
    for (const m of ALL_MOVEMENTS) {
      const { container } = render(<MovementArt movement={m} />);
      const src = container.querySelector("img")!.getAttribute("src")!;
      expect(existsSync(join(process.cwd(), "public", src)), `${m} -> ${src}`).toBe(true);
    }
  });
});
