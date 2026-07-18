/** @type {import('tailwindcss').Config} */
// Palette ported from demo/index.html (the X-Coach dashboard mock-up).
// Semantic surface/text tokens are driven by CSS variables (see index.css) so the
// same class names adapt to light & dark themes. The legacy "-dark" suffixed names
// are kept as aliases to avoid churn across components.
const withVar = (name) => `rgb(var(${name}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#0f758a",
        "primary-content": "#ffffff",
        secondary: "#42d159",
        danger: "#ef4444",

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
        display: ["Space Grotesk", "sans-serif"],
        body: ["Noto Sans", "sans-serif"],
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
