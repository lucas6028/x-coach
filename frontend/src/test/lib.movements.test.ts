import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../api";
import { ALL_MOVEMENTS, MOVEMENT_GROUPS } from "../lib/movements";

afterEach(() => vi.restoreAllMocks());

// The sixteen-name catalog exists TWICE across the two languages: here, and in
// `src/pose/movements/catalog.py`, which the plan API validates every `movement` against. A name
// added to one and not the other means a movement the plan builder offers and the API rejects with
// a 400, so both sides pin the same list — `tests/test_movement_catalog.py` is this test's other
// half. Neither can import the other, which is exactly why both assert rather than derive.
describe("the movement catalog", () => {
  it("holds exactly sixteen movements", () => {
    expect(ALL_MOVEMENTS).toHaveLength(16);
  });

  it("matches src/pose/movements/catalog.py name for name, in order", () => {
    expect([...ALL_MOVEMENTS]).toEqual([
      "Squat", "Lunge", "Deadlift", "Leg Abduction", "Shoulder Bridge",
      "Push-up", "Overhead Press", "Row", "Bicep Curl", "Band Pull Apart",
      "Arm Abduction", "Arm VW",
      "Sit-up", "Torso Twist",
      "Jumping Jacks", "High Knee",
    ]);
  });

  it("has every movement in exactly one body-region group", () => {
    const grouped = MOVEMENT_GROUPS.flatMap((g) => [...g.items]);
    expect(grouped).toEqual([...ALL_MOVEMENTS]);
    expect(new Set(grouped).size).toBe(grouped.length);
  });
});

describe("api.getMovements", () => {
  it("returns the analyzable movements with their validation flags", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        movements: [
          { name: "Squat", validated: true },
          { name: "Overhead Press", validated: false },
          { name: "Push-up", validated: false },
        ],
      }),
    } as Response);

    const movements = await api.getMovements();
    expect(movements.map((m) => m.name)).toEqual(["Squat", "Overhead Press", "Push-up"]);
    expect(movements.find((m) => m.name === "Squat")?.validated).toBe(true);
    expect(movements.find((m) => m.name === "Push-up")?.validated).toBe(false);
  });
});

describe("api.analyzeUpload", () => {
  it("sends the chosen movement as a form field", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ video_id: "v1", movement: "Push-up" }),
    } as Response);

    await api.analyzeUpload(new File(["x"], "clip.mp4"), "Push-up");

    const body = fetchSpy.mock.calls[0][1]?.body as FormData;
    expect(body.get("movement")).toBe("Push-up");
    expect(body.get("file")).toBeInstanceOf(File);
  });
});
