import { describe, it, expect } from "vitest";
import { stepCount, initialCount, COMBO_WINDOW_MS, type CountState } from "../lib/sixseven/counter";

describe("stepCount", () => {
  it("ignores neutral (dead-zone) frames", () => {
    const out = stepCount(initialCount, "neutral", 100);
    expect(out.scored).toBe(false);
    expect(out.state).toBe(initialCount);
  });

  it("arms on the first raised hand without scoring", () => {
    const out = stepCount(initialCount, "left", 100);
    expect(out.scored).toBe(false);
    expect(out.state.lastLead).toBe("left");
    expect(out.state.count).toBe(0);
    expect(out.state.lastSwitchAt).toBe(100);
  });

  it("does not score while the same hand stays up", () => {
    const armed = stepCount(initialCount, "left", 100).state;
    const out = stepCount(armed, "left", 200);
    expect(out.scored).toBe(false);
    expect(out.state.count).toBe(0);
  });

  it("scores a 67 each time the raised hand switches", () => {
    let s: CountState = stepCount(initialCount, "left", 100).state;
    const first = stepCount(s, "right", 400);
    expect(first.scored).toBe(true);
    expect(first.state.count).toBe(1);
    expect(first.state.lastLead).toBe("right");
  });

  it("grows the combo when switches stay in rhythm", () => {
    let s: CountState = stepCount(initialCount, "left", 0).state;
    s = stepCount(s, "right", 400).state; // count 1, combo 1
    s = stepCount(s, "left", 800).state; // count 2, combo 2
    s = stepCount(s, "right", 1200).state; // count 3, combo 3
    expect(s.count).toBe(3);
    expect(s.combo).toBe(3);
    expect(s.bestCombo).toBe(3);
  });

  it("resets the combo to 1 when the rhythm lapses", () => {
    let s: CountState = stepCount(initialCount, "left", 0).state;
    s = stepCount(s, "right", 400).state; // combo 1
    s = stepCount(s, "left", 800).state; // combo 2
    const late = stepCount(s, "right", 800 + COMBO_WINDOW_MS + 1); // too slow
    expect(late.state.count).toBe(3);
    expect(late.state.combo).toBe(1);
    expect(late.state.bestCombo).toBe(2); // best from the earlier streak
  });

  it("does not mutate the input state", () => {
    const armed = stepCount(initialCount, "left", 100).state;
    const before = { ...armed };
    stepCount(armed, "right", 400);
    expect(armed).toEqual(before);
  });
});
