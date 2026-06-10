/** @type {import('tailwindcss').Config} */
// Palette ported from demo/index.html (the X-Coach dashboard mock-up).
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
        "background-dark": "#121416",
        "surface-dark": "#1e2124",
        "border-dark": "#2a2e33",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["Noto Sans", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
