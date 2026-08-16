import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect } from "vitest";

import { MOVEMENT_DETAIL } from "../lib/movementDetail";
import { ALL_MOVEMENTS } from "../lib/movements";

// Off the cwd rather than import.meta.url, which vitest serves as a non-file URL. Vitest's cwd is
// the project root, which for this repo means yarn has to be run from frontend/ anyway.
const PUBLIC = resolve(process.cwd(), "public");

// The authored content itself is prose and not worth asserting word by word — pages.MovementDetail
// covers that it reaches the screen. What is worth a guard is the step ART, because it is wired by
// hand: five WebPs listed in a const, produced by scripts/prep_movement_detail_art.py from PNGs
// that are gitignored. Listing four of the five, or listing them out of order, renders a strip
// that looks deliberate and is wrong — and neither the type checker nor a render test would object.

describe("the authored movement detail", () => {
  it("covers every movement in the catalog", () => {
    for (const movement of ALL_MOVEMENTS) {
      expect(MOVEMENT_DETAIL[movement], movement).toBeDefined();
    }
  });

  // The other direction. An entry here for a name the catalog does not list is unreachable content:
  // no card links to it, so nobody would notice it rot. It also makes the two files' counts agree,
  // which is what lets StepFigure's card-art fallback in pages/MovementDetail.tsx be described as
  // unreachable now that all sixteen have art — a claim worth a test rather than a comment.
  it("has no detail entry outside the catalog", () => {
    for (const name of Object.keys(MOVEMENT_DETAIL)) {
      expect(ALL_MOVEMENTS, name).toContain(name);
    }
  });

  it("gives each movement five numbered steps", () => {
    for (const [name, detail] of Object.entries(MOVEMENT_DETAIL)) {
      expect(detail.steps.length, name).toBe(5);
    }
  });

  it("illustrates a movement's steps all or not at all, in order", () => {
    for (const [name, detail] of Object.entries(MOVEMENT_DETAIL)) {
      const images = detail.steps.map((step) => step.image);
      if (images.every((image) => image === undefined)) continue;

      // A half-illustrated set is the failure this test exists for: the strip would mix real
      // figures with the fallback card art and read as five different drawings of one movement.
      expect(images.filter(Boolean).length, name).toBe(5);
      images.forEach((image, i) => {
        expect(image, name).toBe(`${images[0]!.slice(0, -"-1.webp".length)}-${i + 1}.webp`);
      });
    }
  });

  // The three tests above are entirely internal: a set that is five-long, numbered and consistent
  // still 404s if the slug is wrong ("sit-up" vs "situp", "band-pull-apart" vs "band-pullapart"),
  // and a slug is exactly the kind of thing that gets typed by hand. Nothing else in the suite
  // touches public/ — jsdom does not fetch these, so a missing file renders as an empty box in the
  // browser and a passing test here. So: go and look.
  it("points every image at a file that is actually in public/", () => {
    // First, so that a run started from the wrong directory says so once instead of reporting
    // every asset in the catalog as missing.
    expect(existsSync(PUBLIC), `no public/ under ${process.cwd()} — run yarn from frontend/`).toBe(
      true,
    );

    for (const [name, detail] of Object.entries(MOVEMENT_DETAIL)) {
      const assets = [...detail.steps.map((step) => step.image), detail.plate];
      for (const asset of assets.filter((a): a is string => a !== undefined)) {
        expect(asset, `${name} — must be an absolute public path`).toMatch(/^\//);
        expect(existsSync(resolve(PUBLIC, asset.slice(1))), `${name} — ${asset}`).toBe(true);
      }
    }
  });
});
