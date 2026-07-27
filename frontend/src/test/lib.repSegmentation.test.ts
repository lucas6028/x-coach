import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import {
  DEFAULT_MIN_REP_SECONDS,
  ENTER_FRACTION,
  EXIT_FRACTION,
  PERCENTILE_HIGH,
  PERCENTILE_LOW,
  segmentReps,
  selectReps,
  type RepWindow,
} from "../lib/repSegmentation";

// The SAME file tests/test_rep_segmentation.py reads. Either implementation changing a threshold
// turns both suites red — that is the whole point of SP1 §7 having produced it. Resolved from
// import.meta.url, not cwd, so the test does not care where vitest was launched from.
//
// NOT `new URL("../../../tests/...", import.meta.url)`: Vite 6's asset plugin statically detects
// that exact `new URL(<literal>, import.meta.url)` AST shape and rewrites it at transform time
// into a dev-server `/@fs/...` URL (protocol `http:`), which `fs.readFileSync` then rejects with
// "The URL must be of scheme file" even under vitest. Building the path with `fileURLToPath` +
// `path.resolve` is opaque to that static analysis and keeps the actual intent: cwd-independent.
interface FixtureCase {
  name: string;
  signal: number[];
  fps: number;
  polarity: "min" | "max";
  rectify: boolean;
  rep_start: "extended" | "flexed";
  min_rep_seconds: number;
  expected: { index: number; start: number; end: number; partial: boolean }[];
}
const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../tests/fixtures/rep_segmentation_cases.json");
const fixture = JSON.parse(readFileSync(fixturePath, "utf-8")) as { cases: FixtureCase[] };

describe("segmentReps against the shared Python fixture", () => {
  it("has the fixture and it is not empty", () => {
    expect(fixture.cases.length).toBeGreaterThan(0);
  });

  for (const c of fixture.cases) {
    it(`matches Python on "${c.name}"`, () => {
      const got = segmentReps(c.signal, {
        fps: c.fps,
        polarity: c.polarity,
        rectify: c.rectify,
        repStart: c.rep_start,
        minRepSeconds: c.min_rep_seconds,
      });
      expect(got).toEqual(c.expected.map((e) => ({ ...e })));
    });
  }
});

describe("thresholds are named constants, matching rep_segmentation.py", () => {
  it("carries the Python values", () => {
    expect(PERCENTILE_LOW).toBe(5);
    expect(PERCENTILE_HIGH).toBe(95);
    expect(ENTER_FRACTION).toBe(0.35);
    expect(EXIT_FRACTION).toBe(0.65);
    expect(DEFAULT_MIN_REP_SECONDS).toBe(0.4);
  });
});

describe("segmentReps degenerate inputs", () => {
  it("returns [] for a flat signal (span == 0)", () => {
    expect(segmentReps(new Array(90).fill(5), { fps: 30 })).toEqual([]);
  });

  it("returns [] when there are fewer samples than two minimum reps", () => {
    expect(segmentReps([1, 2, 3], { fps: 30 })).toEqual([]);
  });

  it("rejects an unknown polarity rather than guessing", () => {
    // @ts-expect-error deliberately wrong, mirroring rep_segmentation.py:175-178
    expect(() => segmentReps([1, 2], { fps: 30, polarity: "sideways" })).toThrow(/polarity/);
  });
});

const win = (index: number, start: number, end: number, partial = false): RepWindow =>
  ({ index, start, end, partial });

describe("selectReps", () => {
  const five = [win(1, 0, 9), win(2, 10, 19), win(3, 20, 29), win(4, 30, 39), win(5, 40, 49)];

  it("takes first / middle / last from five reps", () => {
    expect(selectReps(five, 3).map((r) => r.index)).toEqual([1, 3, 5]);
  });

  it("takes everything when there are fewer than the cap", () => {
    expect(selectReps(five.slice(0, 2), 3).map((r) => r.index)).toEqual([1, 2]);
  });

  it("treats 0 and null as 'every rep'", () => {
    expect(selectReps(five, 0)).toHaveLength(5);
    expect(selectReps(five, null)).toHaveLength(5);
  });

  it("skips partial reps when complete ones exist", () => {
    const mixed = [win(1, 0, 9, true), win(2, 10, 19), win(3, 20, 29)];
    expect(selectReps(mixed, 3).map((r) => r.index)).toEqual([2, 3]);
  });

  it("keeps partial reps when they are all there is", () => {
    const allPartial = [win(1, 0, 9, true), win(2, 10, 19, true)];
    expect(selectReps(allPartial, 3).map((r) => r.index)).toEqual([1, 2]);
  });

  // THE TRAP. Python's int(round(...)) is banker's rounding; Math.round is not. Measured:
  // n=6,k=3 -> Python [0,2,5] but Math.round gives [0,3,5]; n=10,k=3 -> [0,4,9] vs [0,5,9].
  // Without half-to-even, a 6-rep and a 10-rep clip analyse DIFFERENT reps in the two languages.
  it("rounds half to even, like Python, on six reps", () => {
    const six = [...five, win(6, 50, 59)];
    expect(selectReps(six, 3).map((r) => r.index)).toEqual([1, 3, 6]);
  });

  it("rounds half to even, like Python, on ten reps", () => {
    const ten = Array.from({ length: 10 }, (_, i) => win(i + 1, i * 10, i * 10 + 9));
    expect(selectReps(ten, 3).map((r) => r.index)).toEqual([1, 5, 10]);
  });

  it("agrees with numpy.linspace on a tie that rounds UP", () => {
    // n=8,k=3 puts the middle at 3.5, which half-to-even sends to 4 -- the same way Math.round
    // would. Included because the two rules only differ in one direction, and a port that got
    // the direction backwards would still pass the two cases above.
    const eight = Array.from({ length: 8 }, (_, i) => win(i + 1, i * 10, i * 10 + 9));
    expect(selectReps(eight, 3).map((r) => r.index)).toEqual([1, 5, 8]);
  });

  it("returns [] for no reps", () => {
    expect(selectReps([], 3)).toEqual([]);
  });
});
