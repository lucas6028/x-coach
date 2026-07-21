import { describe, expect, it } from "vitest";
import { landmarksToFrame } from "../lib/poseExtract";

const lm = (n: number) => Array.from({ length: n }, (_, i) => ({ x: i / 100, y: i / 50, z: 0.1, visibility: 0.9 }));

describe("landmarksToFrame", () => {
  it("serializes 33 landmarks + world landmarks into the shared schema", () => {
    const frame = landmarksToFrame(7, lm(33), lm(33));
    expect(frame.frame_index).toBe(7);
    expect(frame.landmarks).toHaveLength(33);
    expect(frame.landmarks![0]).toEqual({ x: 0, y: 0, z: 0.1, visibility: 0.9 });
    expect(frame.world_landmarks).toHaveLength(33);
  });

  it("emits null landmarks when the frame has no full 33-point pose", () => {
    expect(landmarksToFrame(1, undefined, undefined).landmarks).toBeNull();
    expect(landmarksToFrame(2, lm(20), lm(20)).landmarks).toBeNull(); // detector needs >=33
  });
});
