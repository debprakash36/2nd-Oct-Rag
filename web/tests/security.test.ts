/**
 * FR-22 enforcement: model and document output must never be rendered as markup.
 *
 * The real control for XSS in a chat UI is escaping at the render boundary — React
 * escapes text nodes by default, and `dangerouslySetInnerHTML` is the one escape
 * hatch that disables it. That hatch is easy to add for "just this one snippet" and
 * silently turns every answer into an injection surface, because the text being
 * rendered is model output over untrusted corpus content.
 *
 * A lint rule would be the natural guard, but there is no dependable ESLint rule for
 * "no `dangerouslySetInnerHTML`" and adding one is out of scope for this phase. A
 * test that scans the source tree is the version that cannot be bypassed by someone
 * who does not know the rule exists.
 */

import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const WEB_ROOT = process.cwd();
const SCAN_DIRS = ["app", "components", "lib"];
const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx"]);

function* sourceFiles(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      yield* sourceFiles(path);
    } else if (SOURCE_EXTENSIONS.has(path.slice(path.lastIndexOf(".")))) {
      yield path;
    }
  }
}

/**
 * Strip comments so a file may *discuss* the forbidden APIs — as this one and
 * `SourceViewer` both do — without tripping the scan.
 *
 * Not a real parser: string literals are not tracked, so a `//` inside a string
 * would truncate the rest of a line. That errs toward a false positive, never a
 * false negative, which is the right direction for a security check.
 */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

describe("FR-22: no dangerouslySetInnerHTML", () => {
  it("finds no dangerouslySetInnerHTML in application source", () => {
    const offenders: string[] = [];

    for (const dir of SCAN_DIRS) {
      for (const file of sourceFiles(join(WEB_ROOT, dir))) {
        const contents = stripComments(readFileSync(file, "utf8"));
        if (contents.includes("dangerouslySetInnerHTML")) {
          offenders.push(relative(WEB_ROOT, file));
        }
      }
    }

    expect(
      offenders,
      `dangerouslySetInnerHTML found in: ${offenders.join(", ")}. ` +
        "Render model and passage output as text; React escapes it by default.",
    ).toEqual([]);
  });

  it("finds no raw innerHTML or document.write", () => {
    // The other two ways to inject markup from a string in a browser.
    const offenders: string[] = [];
    const patterns = [/\.innerHTML\s*=/, /\.outerHTML\s*=/, /document\.write\s*\(/];

    for (const dir of SCAN_DIRS) {
      for (const file of sourceFiles(join(WEB_ROOT, dir))) {
        const contents = stripComments(readFileSync(file, "utf8"));
        if (patterns.some((pattern) => pattern.test(contents))) {
          offenders.push(relative(WEB_ROOT, file));
        }
      }
    }

    expect(offenders, `raw HTML assignment found in: ${offenders.join(", ")}`).toEqual([]);
  });

  it("renders passage text in SourceViewer as a text child", () => {
    // Positive assertion: the component that displays untrusted corpus text must
    // pass it as a child, not through a prop that implies markup.
    const source = readFileSync(
      join(WEB_ROOT, "components", "SourceViewer.tsx"),
      "utf8",
    );
    expect(source).toMatch(/<pre[^>]*>\{text\}<\/pre>/);
  });

  it("renders answer text as a text child in the chat thread", () => {
    const source = readFileSync(join(WEB_ROOT, "app", "chat", "page.tsx"), "utf8");
    // The answer is interpolated as a value, not wrapped in a tag string.
    expect(source).not.toMatch(/<span[^>]*dangerously/);
    expect(source).toMatch(/\{turn\.content\}/);
  });
});