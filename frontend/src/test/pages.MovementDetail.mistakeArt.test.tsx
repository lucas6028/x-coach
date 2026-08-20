import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { I18nProvider } from "../lib/i18n";
import { api } from "../api";

// The UNillustrated card, which no squat fault takes any more. All five now declare a wrong/correct
// pair, so `mistake.art &&` and MistakePanel's `src ?` branch are only ever taken one way on the
// movement every other test renders — and the placeholder is not dead code. It is what three of
// the fourteen movements still show, what any movement shows while its sheets arrive one at a
// time, and what any newly registered fault shows on its first day. This file mocks the art back
// OFF one entry so both halves of that branch stay exercised.
//
// It ran the other way round until the art landed: it used to mock art ONTO an entry so the
// illustrated branch was covered before any file existed. Same purpose, opposite direction.
//
// Its own file because vi.mock is hoisted per module, and the sibling
// pages.MovementDetail.test.tsx deliberately renders the real roster.

vi.mock("../lib/movementMistakes", async () => {
  const actual = await vi.importActual<typeof import("../lib/movementMistakes")>(
    "../lib/movementMistakes"
  );
  const [first, ...rest] = actual.MOVEMENT_MISTAKES.Squat;
  const { art: _dropped, ...undrawn } = first;
  return {
    ...actual,
    movementMistakes: (movement: string) =>
      movement === "Squat" ? [undrawn, ...rest] : actual.movementMistakes(movement),
  };
});

vi.mock("../lib/auth", () => ({ useAuth: vi.fn() }));
import { useAuth } from "../lib/auth";
import MovementDetail from "../pages/MovementDetail";

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => vi.fn() };
});

beforeEach(() => {
  localStorage.clear();
  vi.mocked(useAuth).mockReturnValue({ user: null } as unknown as ReturnType<typeof useAuth>);
  vi.spyOn(api, "getMovements").mockResolvedValue([{ name: "Squat", validated: true }]);
});
afterEach(() => vi.restoreAllMocks());

function renderSquatMistakes() {
  render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/movements/Squat?tab=mistakes"]}>
        <Routes>
          <Route path="/movements/:movement" element={<MovementDetail />} />
        </Routes>
      </MemoryRouter>
    </I18nProvider>
  );
}

describe("MovementDetail — a common mistake with no illustration", () => {
  it("holds both slots open and labels each as itself", async () => {
    renderSquatMistakes();
    await screen.findByRole("heading", { name: "Knees caving in", level: 3 });
    // One card unillustrated, so exactly one pair of placeholders. The slot is drawn either way,
    // which is what lets art land later without reflowing anything around it.
    expect(screen.getAllByText("Common mistake")).toHaveLength(1);
    expect(screen.getAllByText("Correct form")).toHaveLength(1);
  });

  it("draws nothing rather than standing the movement's own art in for the fault", async () => {
    renderSquatMistakes();
    await screen.findByRole("heading", { name: "Knees caving in", level: 3 });
    // The card that lost its pair contributes no image at all — not the generic squat drawing, and
    // above all not one picture captioned both ways, which would be a lie about what the fault
    // looks like. The other four are untouched and still show their two files each.
    expect(screen.queryByAltText("Knees caving in — what it looks like")).not.toBeInTheDocument();
    expect(screen.queryByAltText("Knees caving in — corrected")).not.toBeInTheDocument();
    const drawn = screen.getAllByRole("img").map((img) => img.getAttribute("src"));
    expect(drawn).toHaveLength(8);
    expect(drawn.some((src) => src?.includes("knees-inward"))).toBe(false);
  });
});
