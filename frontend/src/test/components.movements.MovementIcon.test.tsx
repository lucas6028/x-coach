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

  it("draws each figure with strokes or a fill, never neither and never both", () => {
    // The renderer draws whichever is present, so a glyph with both would come out as a
    // silhouette wearing its own outline, and one with neither as a bare head.
    for (const [name, g] of Object.entries(GLYPHS)) {
      expect(Boolean(g.strokes?.length) !== Boolean(g.fill), name).toBe(true);
    }
  });

  it("keeps every figure inside the 24x24 frame", () => {
    // A stray coordinate draws a limb clipped at the icon's edge, which no render assertion sees.
    for (const [name, g] of Object.entries(GLYPHS)) {
      // Path data is coordinate PAIRS, and both axes are bounded 0..24, so a pooled check reads
      // the same whichever one went out. Name the axis or the failure points at nothing.
      const coords: [string, number][] = [...(g.strokes ?? []), g.fill ?? ""]
        .join(" ")
        .match(/-?\d+(\.\d+)?/g)!
        .map((v, i) => [i % 2 === 0 ? "x" : "y", Number(v)]);
      // The head is a circle, so it is its centre plus and minus its radius that has to fit —
      // the centre alone passing says nothing once `headR` varies.
      const r = g.headR ?? 2;
      for (const [axis, c] of [
        ["head x", g.head[0]],
        ["head y", g.head[1]],
      ] as [string, number][]) {
        coords.push([axis, c - r], [axis, c + r]);
      }
      for (const [axis, n] of coords) {
        expect(n, `${name} ${axis}`).toBeGreaterThanOrEqual(0);
        expect(n, `${name} ${axis}`).toBeLessThanOrEqual(24);
      }
    }
  });

  it("fills the silhouettes even-odd, which is the only thing making their holes holes", () => {
    // Overhead Press closes a loop with the arms and the bar and drops the head inside it as an
    // island; Deadlift leaves a gap under the torso. Under SVG's default nonzero rule both fill
    // in solid — a silent visual regression no other assertion here would see.
    for (const m of ["Overhead Press", "Deadlift"]) {
      const { container } = render(<MovementIcon movement={m} />);
      expect(container.querySelector("path"), m).toHaveAttribute("fill-rule", "evenodd");
    }
  });

  it("renders a decorative glyph, hidden from assistive tech", () => {
    const { container } = render(<MovementIcon movement="Squat" />);
    const svg = container.querySelector("svg")!;
    expect(svg).toHaveAttribute("aria-hidden", "true");
    // Squat is the silhouette glyph: one filled path, no strokes.
    expect(svg.querySelectorAll("path")).toHaveLength(1);
    expect(svg.querySelector("path")).toHaveAttribute("fill", "currentColor");
    // The name beside it is the accessible text; the icon must add nothing to it.
    expect(container.textContent).toBe("");
  });
});
