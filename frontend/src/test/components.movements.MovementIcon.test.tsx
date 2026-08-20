import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import MovementIcon, { GLYPHS } from "../components/movements/MovementIcon";
import { ALL_MOVEMENTS } from "../lib/movements";

describe("MovementIcon", () => {
  it("has a drawn glyph for every movement in the catalog", () => {
    for (const m of ALL_MOVEMENTS) expect(GLYPHS[m], m).toBeDefined();
  });

  it("is keyed only to names the catalog uses", () => {
    // A near-miss key ("Jumping Jack" for "Jumping Jacks") is otherwise invisible — the card
    // simply renders no icon and every other assertion still passes.
    for (const key of Object.keys(GLYPHS)) {
      expect(ALL_MOVEMENTS as readonly string[], key).toContain(key);
    }
  });

  it("draws a different figure for every movement", () => {
    // The whole point of drawing these is that each one shows its own movement. Two identical
    // glyphs would mean one of them is showing the wrong exercise.
    const shapes = Object.values(GLYPHS).map((g) => JSON.stringify(g));
    expect(new Set(shapes).size).toBe(ALL_MOVEMENTS.length);
  });

  it("keeps every figure inside the 24x24 frame", () => {
    // A stray coordinate draws a limb clipped at the icon's edge, which no render assertion sees.
    for (const [name, g] of Object.entries(GLYPHS)) {
      const nums = g.strokes
        .join(" ")
        .match(/-?\d+(\.\d+)?/g)!
        .map(Number)
        .concat(g.head);
      for (const n of nums) {
        expect(n, name).toBeGreaterThanOrEqual(0);
        expect(n, name).toBeLessThanOrEqual(24);
      }
    }
  });

  it("renders a decorative glyph, hidden from assistive tech", () => {
    const { container } = render(<MovementIcon movement="Squat" />);
    const svg = container.querySelector("svg")!;
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg.querySelectorAll("path")).toHaveLength(GLYPHS["Squat"].strokes.length);
    // The name beside it is the accessible text; the icon must add nothing to it.
    expect(container.textContent).toBe("");
  });
});
