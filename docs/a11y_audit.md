# Accessibility audit (implementation.md 5.5, NFR-9)

Audit and remediation of the web client (`web/`). Covers the chat page, the source
viewer, and the new inline citation controls. Target is WCAG 2.1 AA.

## Summary

One genuine defect was found and fixed: **inline `[n]` citation markers were not
operable**. Everything else audited was already sound. Details below, with the
evidence for each claim.

## 1. Inline citation markers — fixed

### The defect

The backend embeds citation markers in answer text and validates them server-side
(`app/generation/validator.py` matches `MARKER_RE = re.compile(r"\[(\d+)\]")`). They
arrive inside ordinary `token` SSE events and were rendered as plain text:

```tsx
{turn.content}   // "Refunds are issued within 30 days [1]."
```

Consequences:

- A screen reader announced `[1]` as "bracket one bracket" — noise, not information.
- There was no way to reach the cited passage from the point of the claim. The
  sources panel listed the same documents, but a reader using a keyboard or screen
  reader had to independently match "policy.md, p.3" in the panel to the `[1]` in the
  prose. That is the exact task FR-18 asks the UI to make easy, done manually.
- The marker is the *only* link between a claim and its supporting passage, so an
  inert marker makes the citation decorative in practice regardless of how well the
  server validated it.

### The fix

`web/components/CitedAnswer.tsx` splits answer text on the marker pattern and renders
each marker as a real `<button>`:

- **A `<button>`, not `<span role="button">`.** Space and Enter activation, tab order,
  and the button role come from the platform. Reimplementing them with a `div` is a
  common source of keyboard bugs.
- **`aria-label` names the source**, e.g. `Citation 1: policy.md, page 3`, so the
  accessible name conveys where the marker goes rather than just its number.
- **`aria-expanded` + `aria-controls`** wired to the panel that actually appears,
  giving assistive tech the disclosure relationship.
- **Activating the marker reveals the passage inline**, next to the claim, so the
  connection between claim and evidence is never broken by moving elsewhere on screen.
- **A marker with no matching source stays literal text.** A stripped or stale marker
  must not become a button wired to nothing; the audit found the citation-validator
  path can strip markers, so this state is reachable.

Tests: `web/components/CitedAnswer.test.tsx` (9 tests) covers the button role,
accessible names, `aria-expanded`/`aria-controls` wiring, keyboard-only operation,
collapse, unmatched markers, and XSS.

### Known limitation

Sources are only in client state for the turn currently being streamed; a reloaded
conversation has no `sources`, so its markers render as plain text. Wiring the panel
to all turns' citations would need per-turn source state from the conversation API,
which does not currently return it. This is recorded rather than hidden: the marker is
still readable text, which is the pre-audit behaviour.

## 2. Colour contrast — verified

All foreground/background pairs measured against the actual theme values in
`web/app/globals.css`, in both light and dark:

| Element | Light | Dark | AA (4.5:1) |
| --- | --- | --- | --- |
| Citation marker (`--accent` on `--accent-soft`) | 4.91:1 | 5.90:1 | pass |
| Muted text (`--text-muted` on `--bg-subtle`) | 4.92:1 | 6.17:1 | pass |
| Error text (`--danger` on `--bg`) | 6.53:1 | 7.76:1 | pass |
| Warning text (`--warning` on `--bg`) | 5.93:1 | 13.22:1 | pass |

The marker and muted text sit below 7:1 (AAA) in light mode. AA is the stated target
and was not exceeded by darkening further, because these are not distinguished by
colour alone — the marker differs in size, weight, and position, and it is a focusable
control.

`--border` (`#e2e2e5` light / `#2c2c33` dark) is used for dividers and control
outlines rather than text, and is exempt from text contrast requirements; it is
nonetheless visible in both themes.

## 3. Keyboard operability — verified

- **Citation markers** — in the tab order, activated by Enter and Space, visible
  `:focus-visible` ring at 2px with 2px offset so it does not clip the line box.
  `:focus-visible` rather than `:focus` so a mouse click leaves no ring behind.
- **Source panel toggles** — `<button type="button">` with `aria-expanded` and
  `aria-controls` pointing at the passage container.
- **Sidebar** — conversation select and delete are separate buttons; delete has an
  `aria-label` (`Delete conversation: <preview>`) because its visible content is the
  `×` glyph. The delete button is `opacity: 0` until hover or `:focus-visible`, so it
  is still reachable and visible when focused by keyboard.
- **Feedback buttons** — `aria-pressed` reflects the vote, `aria-label` names the
  action, and state is conveyed by the pressed attribute rather than colour alone.
- **Composer** — the textarea has `aria-label="Message"`, and Enter sends while
  Shift+Enter inserts a newline, so a multi-line paste is not hijacked.

## 4. Screen reader semantics — verified

- `<nav aria-label="Conversations">` for the sidebar, `<main>` for the thread,
  `<aside>` for the source panel — three distinct landmarks.
- Errors use `role="alert"` (chat error banner, per-passage load failure) so they are
  announced when they appear rather than only on navigation.
- Source panel heading is `<h2>Sources</h2>`; the citation panel uses
  `role="region"` with an `aria-label`, so it can be navigated to directly.
- The "citation stripped" notice is plain text adjacent to the composer. It is not an
  alert because it is a side effect of a completed answer, not a failure requiring
  immediate attention.

## 5. Text rendering safety — verified

No `dangerouslySetInnerHTML`, `innerHTML`, or `document.write` anywhere in `web/`.
All document- and model-derived text renders through JSX text nodes, which React
escapes (FR-22). Asserted directly by a test that renders hostile content
(`<img src=x onerror=...>`, `<script>`) and asserts no element is created while the
text remains visible.

This overlaps the XSS finding in `docs/security_review.md` §4, which had marked the
frontend as unaudited. That gap is now closed.

## 6. Not addressed

- **No automated axe audit.** All findings above are from manual review plus
  contrast arithmetic. `vitest-axe` / `eslint-plugin-jsx-a11y` would catch regressions
  in CI and should be added; they were not installed here.
- **No screen-reader testing.** NVDA/JAWS/VoiceOver passes were not performed, so the
  announcement wording (`Citation 1: policy.md, page 3`) is reasoned rather than
  confirmed.
- **Per-turn sources on reload** — see the known limitation in §1.
- **Colour scheme is light/dark via `prefers-color-scheme`.** Both were contrast
  checked; neither was validated at high-contrast/forced-colours modes, where
  `prefers-contrast` users may need additional borders.
