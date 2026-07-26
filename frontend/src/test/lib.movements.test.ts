import { describe, it, expect, vi, afterEach } from "vitest";
import { api } from "../api";

afterEach(() => vi.restoreAllMocks());

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
