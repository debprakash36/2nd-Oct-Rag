import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["**/*.test.ts", "**/*.test.tsx"],
    // The "no dangerouslySetInnerHTML" test scans the source tree; exclude node_modules
    // and build output so it does not walk 400MB of dependencies on every run.
    exclude: ["node_modules/**", ".next/**"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});