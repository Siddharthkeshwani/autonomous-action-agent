import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    colors: {
      paper: "var(--paper)",
      surface: "var(--surface)",
      track: "var(--track)",
      line: "var(--line)",
      ink: "var(--ink)",
      muted: "var(--muted)",
      accent: "var(--accent)",
      error: "var(--error)",
      ready: "var(--ready)",
    },
  },
} satisfies Config;
