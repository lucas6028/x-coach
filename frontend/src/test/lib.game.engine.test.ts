import { describe, it, expect } from "vitest";
import {
  createGameState,
  stepGame,
  nextPoseId,
  GRADE_MS,
  DROP_FACTOR,
  type GameState,
} from "../lib/game/engine";
import { HIT_THRESHOLD, HOLD_MS } from "../lib/game/scoring";
import { POSES } from "../lib/game/poses";

function state(over: Partial<GameState> = {}): GameState {
  return { ...createGameState(), ...over };
}

const input = (
  over: Partial<{ quality: number; hasLandmarks: boolean; now: number; rng: () => number }> = {}
) => ({ quality: 0, hasLandmarks: true, now: 1000, rng: () => 0, ...over });

describe("createGameState", () => {
  it("starts empty on the first pose", () => {
    const s = createGameState();
    expect(s).toMatchObject({ score: 0, combo: 0, bestCombo: 0, poses: 0, holdStart: 0 });
    expect(s.targetId).toBe(POSES[0].id);
  });
});

describe("nextPoseId", () => {
  it("never returns the current pose", () => {
    for (let r = 0; r < 1; r += 0.09) expect(nextPoseId("t_pose", r)).not.toBe("t_pose");
  });

  it("clamps r = 1 to a valid pose", () => {
    expect(POSES.some((p) => p.id === nextPoseId("t_pose", 1))).toBe(true);
  });
});

describe("stepGame", () => {
  it("resets the hold when no body is detected", () => {
    const out = stepGame(state({ holdStart: 900 }), input({ hasLandmarks: false }));
    expect(out.state.holdStart).toBe(0);
    expect(out.grade).toBeNull();
  });

  it("starts the hold clock on the first matching frame", () => {
    const out = stepGame(state(), input({ quality: 0.7, now: 1000 }));
    expect(out.state.holdStart).toBe(1000);
    expect(out.state.poses).toBe(0);
    expect(out.grade).toBeNull();
  });

  it("keeps waiting while the hold is too short", () => {
    const out = stepGame(state({ holdStart: 1000 }), input({ quality: 0.7, now: 1000 + HOLD_MS - 1 }));
    expect(out.state.holdStart).toBe(1000);
    expect(out.state.poses).toBe(0);
  });

  it("locks the pose in once held long enough", () => {
    const out = stepGame(
      state({ holdStart: 1000, targetId: "t_pose" }),
      input({ quality: 0.95, now: 1000 + HOLD_MS })
    );
    expect(out.state.combo).toBe(1);
    expect(out.state.bestCombo).toBe(1);
    expect(out.state.poses).toBe(1);
    expect(out.state.score).toBeGreaterThan(0);
    expect(out.state.holdStart).toBe(0);
    expect(out.state.gradeUntil).toBe(1000 + HOLD_MS + GRADE_MS);
    expect(out.state.targetId).not.toBe("t_pose");
    expect(out.grade).toBe("perfect");
  });

  it("breaks the combo on a decisive drop mid-hold", () => {
    const out = stepGame(
      state({ holdStart: 1000, combo: 3 }),
      input({ quality: HIT_THRESHOLD * DROP_FACTOR - 0.01, now: 1500 })
    );
    expect(out.state.combo).toBe(0);
    expect(out.state.holdStart).toBe(0);
  });

  it("forgives a shallow dip without breaking the combo", () => {
    const out = stepGame(
      state({ holdStart: 1000, combo: 3 }),
      input({ quality: HIT_THRESHOLD - 0.05, now: 1500 })
    );
    expect(out.state.combo).toBe(3);
    expect(out.state.holdStart).toBe(0);
  });

  it("does not mutate the input state", () => {
    const s = state({ holdStart: 1000, combo: 2 });
    stepGame(s, input({ quality: 0.9, now: 1000 + HOLD_MS }));
    expect(s.holdStart).toBe(1000);
    expect(s.combo).toBe(2);
  });
});
