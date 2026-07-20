import { describe, it, expect } from "vitest";
import {
  stepCharge,
  isArmed,
  initialCharge,
  TOGETHER_GAP,
  FIRE_GAP,
  CHARGE_MS,
  type ChargeState,
} from "../lib/blast/charger";

const together = TOGETHER_GAP - 0.1;
const apart = FIRE_GAP + 0.1;

describe("stepCharge", () => {
  it("fills the charge while hands are together", () => {
    let s: ChargeState = initialCharge;
    ({ state: s } = stepCharge(s, together, CHARGE_MS / 2));
    expect(s.charge).toBeCloseTo(0.5, 5);
    expect(isArmed(s)).toBe(false);
  });

  it("caps the charge at 1 and arms", () => {
    const { state, fired } = stepCharge({ charge: 0.9 }, together, CHARGE_MS);
    expect(state.charge).toBe(1);
    expect(isArmed(state)).toBe(true);
    expect(fired).toBe(false);
  });

  it("fires when armed hands are thrown apart, and resets the meter", () => {
    const { state, fired } = stepCharge({ charge: 1 }, apart, 16);
    expect(fired).toBe(true);
    expect(state.charge).toBe(0);
  });

  it("does not fire when hands go apart without a full charge", () => {
    const { state, fired } = stepCharge({ charge: 0.4 }, apart, 100);
    expect(fired).toBe(false);
    expect(state.charge).toBeLessThan(0.4); // decays instead
  });

  it("bleeds charge in the dead-zone between thresholds", () => {
    const mid = (TOGETHER_GAP + FIRE_GAP) / 2;
    const { state, fired } = stepCharge({ charge: 0.8 }, mid, 100);
    expect(fired).toBe(false);
    expect(state.charge).toBeLessThan(0.8);
    expect(state.charge).toBeGreaterThan(0);
  });

  it("never drops the charge below 0", () => {
    const { state } = stepCharge({ charge: 0.01 }, apart, 10000);
    expect(state.charge).toBe(0);
  });
});
