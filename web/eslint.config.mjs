/**
 * Flat ESLint config.
 *
 * `next lint` is deprecated and prompts interactively on first run, so linting is
 * driven by the ESLint CLI directly. `eslint-config-next` is still an eslintrc-style
 * config, so `FlatCompat` adapts it to the flat format.
 */

import js from "@eslint/js";
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({ baseDirectory: import.meta.dirname });

const config = [
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts", "coverage/**"],
  },
  js.configs.recommended,
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // Tests intentionally build partial API payloads and assert on them.
      "@typescript-eslint/no-explicit-any": "warn",
      // A leading underscore marks a parameter kept only to type a signature.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
];

export default config;