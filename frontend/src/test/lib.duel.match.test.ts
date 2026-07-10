import { describe, it, expect } from "vitest";
import {
  advanceHold,
  roundWinner,
  matchWinner,
  HOLD_MS,
  MATCH_POINTS,
  DECAY_FACTOR,
} from "../lib/duel/match";

describe("advanceHold", () => {
  it("builds up while matching", () => {
    expect(advanceHold(0, true, 100)).toBe(100);
  });

  it("decays faster than it builds when not matching", () => {
    expect(advanceHold(500, false, 100)).toBe(500 - 100 * DECAY_FACTOR);
  });

  it("clamps to [0, HOLD_MS]", () => {
    expect(advanceHold(HOLD_MS - 10, true, 1000)).toBe(HOLD_MS);
    expect(advanceHold(10, false, 1000)).toBe(0);
  });
});

describe("roundWinner", () => {
  it("is null while nobody has completed the hold", () => {
    expect(roundWinner(HOLD_MS - 1, HOLD_MS - 1)).toBeNull();
  });

  it("awards the round to whoever completes the hold", () => {
    expect(roundWinner(HOLD_MS, 0)).toBe("a");
    expect(roundWinner(0, HOLD_MS)).toBe("b");
  });

  it("breaks a same-frame double-complete toward the one further along", () => {
    expect(roundWinner(HOLD_MS + 5, HOLD_MS)).toBe("a");
    expect(roundWinner(HOLD_MS, HOLD_MS + 5)).toBe("b");
  });

  it("is null on an exact tie so the round keeps going", () => {
    expect(roundWinner(HOLD_MS, HOLD_MS)).toBeNull();
  });
});

describe("matchWinner", () => {
  it("is null before either side reaches MATCH_POINTS", () => {
    expect(matchWinner(MATCH_POINTS - 1, MATCH_POINTS - 1)).toBeNull();
  });

  it("declares the side that reaches MATCH_POINTS", () => {
    expect(matchWinner(MATCH_POINTS, 1)).toBe("a");
    expect(matchWinner(1, MATCH_POINTS)).toBe("b");
  });
});
