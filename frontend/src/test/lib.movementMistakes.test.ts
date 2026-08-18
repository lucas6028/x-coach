import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect } from "vitest";

import { MOVEMENT_MISTAKES, art, movementMistakes } from "../lib/movementMistakes";
import { ALL_MOVEMENTS } from "../lib/movements";

// Off the cwd rather than import.meta.url, which vitest serves as a non-file URL.
const PUBLIC = resolve(process.cwd(), "public");

// The half of this module a Python test cannot see. tests/test_movement_mistakes_roster.py owns the
// hard invariant — that the ids and kgQueries here equal the ones in src/pose/movements/*.py — but
// it stops at the roster: it cannot type-check the prose, and it would have to guess at whether an
// `art(...)` path points at a file that exists. Those are this file's job.

const ENTRIES = Object.entries(MOVEMENT_MISTAKES);
const ALL = ENTRIES.flatMap(([movement, list]) => list.map((m) => [movement, m] as const));

describe("the authored common mistakes", () => {
  it("only names movements the catalog knows", () => {
    // An entry keyed to a name no card links to is unreachable content, and would silently never
    // render — the same failure lib.movementDetail guards against on its own table.
    for (const [movement] of ENTRIES) {
      expect(ALL_MOVEMENTS, movement).toContain(movement);
    }
  });

  it("returns an empty list, not undefined, for a movement with no entry", () => {
    // The two unregistered movements take this path, and so does a stale URL. The tab branches on
    // `.length`, so an undefined here would throw rather than show the empty state.
    expect(movementMistakes("Jumping Jacks")).toEqual([]);
    expect(movementMistakes("Burpee")).toEqual([]);
  });

  it("fills both languages for every field", () => {
    // Record<Lang, …> makes a MISSING key a type error, but not an empty or copy-pasted one. A
    // zh-Hant string identical to its English twin is the failure mode that ships unnoticed.
    for (const [movement, m] of ALL) {
      for (const field of ["title", "subtitle", "why"] as const) {
        expect(m[field].en.trim(), `${movement}/${m.id}/${field}/en`).not.toBe("");
        expect(m[field]["zh-Hant"].trim(), `${movement}/${m.id}/${field}/zh`).not.toBe("");
        expect(m[field]["zh-Hant"], `${movement}/${m.id}/${field} is untranslated`).not.toBe(
          m[field].en
        );
      }
    }
  });

  it("gives every fault the same number of cues in both languages", () => {
    // The card renders one list per language; a mismatch means a reader in one language is told
    // one fewer thing to do than a reader in the other.
    for (const [movement, m] of ALL) {
      expect(m.fixes.en.length, `${movement}/${m.id}`).toBe(m.fixes["zh-Hant"].length);
      expect(m.fixes.en.length, `${movement}/${m.id}`).toBeGreaterThan(0);
      for (const fix of [...m.fixes.en, ...m.fixes["zh-Hant"]]) {
        expect(fix.trim(), `${movement}/${m.id}`).not.toBe("");
      }
    }
    // React keys the cue list by the cue text, so a duplicate inside one fault would collapse two
    // list items into one.
    for (const [movement, m] of ALL) {
      expect(new Set(m.fixes.en).size, `${movement}/${m.id}`).toBe(m.fixes.en.length);
      expect(new Set(m.fixes["zh-Hant"]).size, `${movement}/${m.id}`).toBe(
        m.fixes["zh-Hant"].length
      );
    }
  });

  it("points every declared illustration at a file that exists", () => {
    // Twenty-seven pairs declared and fifty-three still to come, so this runs against a moving
    // target:
    // a typo'd or not-yet-exported pair fails here rather than rendering as two broken images in
    // the middle of the page. It reads public/ off disk, so it also fails if a WebP is deleted
    // without its `art(...)` call going with it.
    for (const [movement, m] of ALL) {
      if (!m.art) continue;
      for (const path of [m.art.wrong, m.art.correct]) {
        expect(existsSync(resolve(PUBLIC, path.replace(/^\//, ""))), `${movement}/${path}`).toBe(
          true
        );
      }
    }
  });

  it("derives both illustration paths from the fault id, hyphenated like the rest of public/", () => {
    expect(art("knees_inward")).toEqual({
      wrong: "/movements/mistakes/knees-inward-wrong.webp",
      correct: "/movements/mistakes/knees-inward-correct.webp",
    });
  });
});
