// The charge → fire state machine for the "Kamehameha" gesture. Bring your hands
// together to fill the charge meter; once it's full, throw your arms apart to fire.
// A tiny pure reducer so the whole mechanic is testable without a camera.

// Wrist gap (in shoulder-widths) below which hands count as "together" (charging).
export const TOGETHER_GAP = 0.8;
// Gap above which hands count as "thrown apart" (a fire attempt).
export const FIRE_GAP = 1.7;
// Time (ms) of holding hands together to reach a full charge.
export const CHARGE_MS = 450;
// Time (ms) for an interrupted charge to bleed back to empty.
export const DECAY_MS = 700;

export type ChargeState = { charge: number };

export const initialCharge: ChargeState = { charge: 0 };

// A charge is ready to fire once it's full.
export function isArmed(state: ChargeState): boolean {
  return state.charge >= 1;
}

export type ChargeStep = { state: ChargeState; fired: boolean };

// Advance the charge by `dt` ms given the current wrist `gap`. Fires (and resets the
// meter) only on the transition to "apart" while fully charged — so a fire is always a
// deliberate charge-then-release, never a stray frame where the hands drift apart.
export function stepCharge(state: ChargeState, gap: number, dt: number): ChargeStep {
  let charge = state.charge;
  let fired = false;
  if (gap < TOGETHER_GAP) {
    charge = Math.min(1, charge + dt / CHARGE_MS);
  } else if (gap > FIRE_GAP) {
    if (charge >= 1) {
      fired = true;
      charge = 0;
    } else {
      charge = Math.max(0, charge - dt / DECAY_MS);
    }
  } else {
    // In the dead-zone between the two thresholds the charge bleeds slowly, so you
    // can't hold a full charge indefinitely — but a quick pass-through still fires.
    charge = Math.max(0, charge - dt / (DECAY_MS * 2));
  }
  return { state: { charge }, fired };
}
