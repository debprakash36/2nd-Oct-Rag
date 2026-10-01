/**
 * Shared test setup.
 *
 * `jest-dom` matchers are registered globally rather than per-test so the DOM
 * assertions read as assertions instead of as library setup.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  // jsdom leaves fetch mocks behind between files; a leaked one silently satisfies
  // the next test's assertions about what was requested.
  vi.restoreAllMocks();
});