import { describe, it, expect, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import {
  I18nProvider,
  useI18n,
  getStoredLang,
  faultLabel,
  viewLabel,
  phaseLabel,
  severityText,
  DICTS,
  type TFunc,
} from "../lib/i18n";

// Use the REAL t function (with the real dictionaries) so these assertions catch
// drift in src/lib/i18n.tsx, rather than a hand-copied dict that can never fail.
function realT(lang: "en" | "zh-Hant" = "en"): TFunc {
  localStorage.setItem("lang", lang); // I18nProvider reads getStoredLang() on mount
  const { result } = renderHook(() => useI18n(), { wrapper: I18nProvider });
  return result.current.t;
}

// Both dicts are typed `Record<string, string>`, so a key added to one locale and forgotten in the
// other is a TYPE-CHECK PASS: `t()` falls back to returning the raw key, and the zh UI renders
// "feedback.notMeasured" as literal text. Nothing else in the suite can see that — English-locale
// tests pass, review does not catch it. This is the only guard.
describe("locale key parity", () => {
  // ONE-DIRECTIONAL ON PURPOSE, and the direction is the one that fails visibly. A key present in
  // `en` but missing from `zh-Hant` makes the zh UI render the raw key ("feedback.notMeasured") as
  // literal text. The reverse is a DELIBERATE pattern here, not drift: `dataLabel` falls back to
  // title-casing the raw value, which is already correct English, so the zh dict carries
  // `movement.*` / `fault.*` entries that `en` intentionally omits. Asserting strict two-way
  // equality would encode a rule this codebase does not follow — verified against the real dicts,
  // where the zh-only set is exactly those data-label families.
  it("every en key has a zh-Hant translation", () => {
    const missing = Object.keys(DICTS.en).filter((k) => !(k in DICTS["zh-Hant"]));
    expect(missing).toEqual([]);
  });

  it("no translation is an empty string", () => {
    for (const [lang, dict] of Object.entries(DICTS)) {
      for (const [key, value] of Object.entries(dict)) {
        expect(value, `${lang}:${key}`).not.toBe("");
      }
    }
  });
});

describe("getStoredLang", () => {
  beforeEach(() => localStorage.clear());

  it("returns 'en' by default when no browser/storage preference", () => {
    // navigator.language is 'en' in jsdom
    expect(getStoredLang()).toBe("en");
  });

  it("returns stored 'zh-Hant'", () => {
    localStorage.setItem("lang", "zh-Hant");
    expect(getStoredLang()).toBe("zh-Hant");
  });

  it("returns stored 'en'", () => {
    localStorage.setItem("lang", "en");
    expect(getStoredLang()).toBe("en");
  });

  it("falls back to 'en' for an invalid stored value", () => {
    localStorage.setItem("lang", "fr");
    expect(getStoredLang()).toBe("en");
  });
});

describe("faultLabel", () => {
  beforeEach(() => localStorage.clear());

  it("translates known fault keys", () => {
    const t = realT("en");
    expect(faultLabel(t, "knees_inward")).toBe("Knee Valgus");
    expect(faultLabel(t, "knees_forward")).toBe("Knees Forward");
  });

  it("title-cases unknown fault IDs", () => {
    const t = realT("en");
    expect(faultLabel(t, "some_new_fault")).toBe("Some New Fault");
  });

  // The label keys must be the fault_id the detector actually emits: pose_rule_detector
  // emits "heel_rise", and this key was "heel_lift" for a while — close enough to read as
  // correct, but it silently fell through to the title-cased fallback, so the zh UI showed
  // "Heel Rise" instead of 腳跟離地.
  it("keys the heel fault on the detector's fault_id in both languages", () => {
    expect(faultLabel(realT("en"), "heel_rise")).toBe("Heel Rise");
    expect(faultLabel(realT("zh-Hant"), "heel_rise")).toBe("腳跟離地");
  });
});

describe("viewLabel", () => {
  beforeEach(() => localStorage.clear());

  it("translates known view keys", () => {
    const t = realT("en");
    expect(viewLabel(t, "side")).toBe("Side");
    expect(viewLabel(t, "front")).toBe("Front");
  });

  it("title-cases unknown view IDs", () => {
    const t = realT("en");
    expect(viewLabel(t, "overhead_view")).toBe("Overhead View");
  });
});

describe("phaseLabel", () => {
  beforeEach(() => localStorage.clear());

  it("translates known phase keys", () => {
    const t = realT("en");
    expect(phaseLabel(t, "descent")).toBe("Descent");
    expect(phaseLabel(t, "bottom")).toBe("Bottom");
  });

  it("title-cases unknown phases", () => {
    const t = realT("en");
    expect(phaseLabel(t, "mid_air")).toBe("Mid Air");
  });
});

describe("severityText", () => {
  beforeEach(() => localStorage.clear());

  it("returns 'High' for severity ≥ 0.75", () => {
    const t = realT("en");
    expect(severityText(t, 0.75)).toBe("High");
    expect(severityText(t, 1.0)).toBe("High");
    expect(severityText(t, 0.9)).toBe("High");
  });

  it("returns 'Moderate' for severity in [0.4, 0.75)", () => {
    const t = realT("en");
    expect(severityText(t, 0.4)).toBe("Moderate");
    expect(severityText(t, 0.6)).toBe("Moderate");
    expect(severityText(t, 0.74)).toBe("Moderate");
  });

  it("returns 'Mild' for severity < 0.4", () => {
    const t = realT("en");
    expect(severityText(t, 0.0)).toBe("Mild");
    expect(severityText(t, 0.39)).toBe("Mild");
  });

  it("works in zh-Hant", () => {
    const tZh = realT("zh-Hant");
    expect(severityText(tZh, 0.9)).toBe("高");
    expect(severityText(tZh, 0.5)).toBe("中");
    expect(severityText(tZh, 0.1)).toBe("低");
  });
});
