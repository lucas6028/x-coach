import { describe, it, expect } from "vitest";
import { createRoundState, stepRound, type RoundState } from "../lib/duel/round";
import { HOLD_MS, MATCH_POINTS, ROUND_BREAK_MS } from "../lib/duel/match";
import { POSES } from "../lib/duel/poses";

function state(over: Partial<RoundState> = {}): RoundState {
  return {
    poseId: "t_pose",
    winsA: 0,
    winsB: 0,
    holdA: 0,
    holdB: 0,
    breakUntil: 0,
    roundFlash: null,
    ...over,
  };
}

const input = (
  over: Partial<{ matchedA: boolean; matchedB: boolean; dtMs: number; now: number; rng: () => number }> = {}
) => ({ matchedA: false, matchedB: false, dtMs: 100, now: 1000, rng: () => 0, ...over });

describe("createRoundState", () => {
  it("starts a fresh match with the rng-chosen first pose", () => {
    const s = createRoundState(() => 0);
    expect(s.winsA).toBe(0);
    expect(s.winsB).toBe(0);
    expect(s.breakUntil).toBe(0);
    expect(s.roundFlash).toBeNull();
    expect(POSES.some((p) => p.id === s.poseId)).toBe(true);
  });
});

describe("stepRound", () => {
  it("freezes scoring during the post-round break", () => {
    const s = state({ breakUntil: 2000, holdA: 120 });
    const out = stepRound(s, input({ matchedA: true, now: 1000 }));
    expect(out.state.holdA).toBe(120); // unchanged
    expect(out.matchOver).toBeNull();
  });

  it("accumulates a player's hold while matching", () => {
    const out = stepRound(state(), input({ matchedA: true, dtMs: 100 }));
    expect(out.state.holdA).toBe(100);
    expect(out.state.winsA).toBe(0);
  });

  it("awards a round, resets holds, flags the flash, and picks a fresh pose", () => {
    const out = stepRound(state({ holdA: HOLD_MS, poseId: "t_pose" }), input({ matchedA: true }));
    expect(out.state.winsA).toBe(1);
    expect(out.state.holdA).toBe(0);
    expect(out.state.holdB).toBe(0);
    expect(out.state.roundFlash).toBe("a");
    expect(out.state.breakUntil).toBe(1000 + ROUND_BREAK_MS);
    expect(out.state.poseId).not.toBe("t_pose"); // pickPose excludes the current pose
    expect(out.matchOver).toBeNull();
  });

  it("awards to player B when B completes the hold", () => {
    const out = stepRound(state({ holdB: HOLD_MS }), input({ matchedB: true }));
    expect(out.state.winsB).toBe(1);
    expect(out.state.roundFlash).toBe("b");
  });

  it("ends the match once a player reaches MATCH_POINTS, without starting a break", () => {
    const out = stepRound(
      state({ winsA: MATCH_POINTS - 1, holdA: HOLD_MS, poseId: "t_pose" }),
      input({ matchedA: true })
    );
    expect(out.matchOver).toBe("a");
    expect(out.state.winsA).toBe(MATCH_POINTS);
    expect(out.state.breakUntil).toBe(0); // no next round
    expect(out.state.poseId).toBe("t_pose"); // pose not advanced on match point
  });

  it("does not mutate the input state", () => {
    const s = state({ holdA: 40, winsA: 1 });
    stepRound(s, input({ matchedA: true }));
    expect(s.holdA).toBe(40);
    expect(s.winsA).toBe(1);
  });
});
