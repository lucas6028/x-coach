/** @type {import('tailwindcss').Config} */
// Palette ported from the motion_analysis_muse-spark reference design: a lavender canvas
// (#eef0fb) under white rounded cards, violet #7B61FF as the single accent, green #22c55e for
// "good" and coral #ff5a5a for faults.
//
// Semantic surface/text tokens are driven by CSS variables (see index.css). The legacy "-dark"
// suffixed names are a naming accident kept as aliases to avoid churn across components — they
// are NOT a dark theme. The app is light-only: there is one token set and nothing ever puts a
// `dark` class on <html>.
const withVar = (name) => `rgb(var(${name}) / <alpha-value>)`;

export default {
  // Class-based, deliberately: Tailwind's DEFAULT is `media`, which would let a stray `dark:`
  // utility switch on the visitor's OS preference and half-darken a light-only app. Keyed to a
  // class nothing applies, such a utility simply never matches.
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#7b61ff",
        "primary-content": "#ffffff",
        secondary: "#22c55e",
        danger: "#ff5a5a",
        // `warning` was used by DemoIntro's Beta badge and the movement-unavailable panel but
        // never actually defined here, so those classes compiled to nothing. Amber, matching the
        // reference's "Form Score 68%" caution band.
        warning: "#e0a33a",

        // Theme-aware surfaces
        background: withVar("--c-bg"),
        "background-dark": withVar("--c-bg"),
        surface: withVar("--c-surface"),
        "surface-dark": withVar("--c-surface"),
        border: withVar("--c-border"),
        "border-dark": withVar("--c-border"),

        // Theme-aware text
        content: withVar("--c-content"),
        muted: withVar("--c-muted"),
        faint: withVar("--c-faint"),

        // Misc theme-aware accents
        track: withVar("--c-track"),
      },
      fontFamily: {
        // The reference pairs Plus Jakarta Sans (headings) with Inter (everything else).
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      // Theme-aware elevation driven by CSS vars (see index.css) so cards lift the same
      // way on the light and dark canvases.
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
        accent: "var(--shadow-accent)",
      },
    },
  },
  plugins: [],
};
