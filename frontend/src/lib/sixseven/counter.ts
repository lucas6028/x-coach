// The "67" rep counter, as a pure state machine. Each time the raised hand switches sides — a
// confirmed alternation of the 6-7 bob — one "67" is scored. Switching quickly keeps a rhythm
// combo alive. All timing is fed in by the caller's loop, so this is fully unit-tested.
import type { Lead } from "./gesture";

// Seconds in a round.
export const ROUND_SECONDS = 30;
// Alternate within this window to keep (and grow) the rhythm combo.
export const COMBO_WINDOW_MS = 1400;

export type CountState = {
  // The last hand that was definitely up ("neutral" frames don't change it).
  lastLead: "left" | "right" | null;
  // How many 67s completed this round.
  count: number;
  // Current rhythm streak.
  combo: number;
  bestCombo: number;
  // Timestamp (ms) of the last counted switch.
  lastSwitchAt: number;
};

export const initialCount: CountState = {
  lastLead: null,
  count: 0,
  combo: 0,
  bestCombo: 0,
  lastSwitchAt: 0,
};

export type CountStep = { state: CountState; scored: boolean };

// Advance the counter for one frame's lead reading. A rep scores only on a real switch from one
// raised hand to the other; a neutral (dead-zone) frame is ignored, and the first raised hand
// merely arms the counter. Returns a new state; the input is not mutated.
export function stepCount(state: CountState, lead: Lead, now: number): CountStep {
  if (lead === "neutral") return { state, scored: false };

  // First definite side arms the counter without scoring.
  if (state.lastLead === null) {
    return { state: { ...state, lastLead: lead, lastSwitchAt: now }, scored: false };
  }

  // Same hand still up — nothing new.
  if (lead === state.lastLead) return { state, scored: false };

  // A switch: that's one 67. Keep the combo if it came quickly enough.
  const inRhythm = now - state.lastSwitchAt <= COMBO_WINDOW_MS;
  const combo = inRhythm ? state.combo + 1 : 1;
  return {
    state: {
      lastLead: lead,
      count: state.count + 1,
      combo,
      bestCombo: Math.max(state.bestCombo, combo),
      lastSwitchAt: now,
    },
    scored: true,
  };
}
