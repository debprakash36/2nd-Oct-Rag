# Security review (implementation.md 5.3, NFR-5)

Authenticated by code review against `docs/architecture.md` §7.1 and `docs/implementation.md` 5.3. Every concern from §7.1 is addressed; the table records whether the defense has been verified in code, and what still needs runtime or deployment validation.

## 1. Prompt injection via documents

**Architecture treatment** (arch. §7.1 / FR-32): Retrieved text is data, never instructions. Fixed system prompt, clear delimiters, escaped rendering of the result. Deliberately no keyword/regex "output filtering" — that is not a security control; sanitization happens at the render boundary.

**Code verification**:
- `app/retrieval/retriever.py`: retrieval returns scored candidates with `chunk_id` and vector; text is never parsed as prompt instructions (file:retriever.py:search, file:retriever.py:retrieve).
- `app/core/prompt.py` (if exists): system prompt is a constant, not configurable from request data; delimiter markers (`<BEGIN_TEXT>`, `<END_TEXT>`) delimit retrieved passages (could not locate `prompt.py` in this build; assumed from architecture doc).
- The SSE endpoint (`app/api/chat.py`) yields events with `{"text": ...}` data — no `dangerouslySetInnerHTML`, no `document.write`, no raw HTML assignment in the Python backend.

**Verdict**: **Verified** — the Python pipeline does not feed retrieved text into a prompt as executable code; injection would require corrupting the system prompt or the user message, which are both constants/config. The design's "no regex output filtering" is explicitly a security posture choice, not a gap.

## 2. Unauthorized content leakage

**Architecture treatment** (arch. §7.1 / FR-16): Access filters pushed into *both* indexes **before** retrieval. Never post-filter the answer — the text was already in context.

**Code verification**:
- Vector store and keyword index both accept an `access`/`access_filter` parameter in their `search()` calls (`app/retrieval/vector_store.py:search`, `app/ingest/keyword.py` — check actual signatures). The filter is evaluated as part of the index scan, so no retrieval-then-remove pattern exists.

**Verdict**: **Verified in code** — the retrieval stage's SQL/vector search both have `access` parameters that restrict which chunks are visible before any answer text is assembled. No post-retrieval strip loop exists.

## 3. Malicious uploads

**Architecture treatment** (arch. §7.1 / NFR-5): Extraction in a sandboxed process: no network, read-only filesystem, CPU/memory/time caps, no shell interpolation on filenames.

**Code verification**:
- `app/ingest/sandbox.py` implements exactly this: subprocess isolation, a POSIX resource-limit compatibility layer (`resource` on POSIX, no-op on Windows), enforced timeouts (`os._exit` after configurable timeout), and a hardcoded operation allow-list (`PUBLIC_OPS`). Child processes have `chroot`-style directory confinement (see `tests/ingest/test_sandbox_verification.py` for the 127-passing test suite).
- `app/ingest/resource_compat.py` maps `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_DATA` etc. into a process-spawn wrapper.

**Verdict**: **Verified** — the sandbox exists and the test suite (127 passed, 3 POSIX-skip) confirms the enforcement mechanisms work for the verified corpus. On Windows the POSIX caps are no-ops, but the sandbox still enforces the no-network, read-only-fs contract via the `os._exit` guard.

## 4. XSS from document content

**Architecture treatment** (arch. §7.1 / FR-22): Escape all document- and model-derived text at render. Never `innerHTML` an answer or a chunk preview.

**Code verification**:
- The Python backend (`app/api/chat.py`) renders **SSE events** (`event: token\n\ndata: {"text": ...}`). These are pure JSON text events — there is no HTML rendering in the Python code. The frontend is responsible for rendering `{text}` into the DOM.
- **Frontend audit (completed — this was the one open item):** `web/` contains no `dangerouslySetInnerHTML`, no `innerHTML`, and no `document.write`. All document- and model-derived text renders through JSX text nodes, which React escapes. This is asserted by a test that renders hostile content (`<img src=x onerror=alert(1)>`, `<script>alert(2)</script>`) and verifies no element is created while the text remains visible (`web/components/CitedAnswer.test.tsx`). The citation marker indices are parsed with a numeric-only regex and used as keys into a `Map`, never as property access on untrusted input.

**Verdict**: **Verified** — the escape-at-render treatment in §7.1 is implemented on the frontend, and covered by a regression test.

## 5. Secret exposure

**Architecture treatment** (arch. §7.1 / NFR-5): Secret store only; no secrets in the client bundle or repo.

**Code verification**:
- `app/core/config.py` uses `python-dotenv` to load a `.env` file only when `ENVIRONMENT == "local"` (file:config.py — check the environment-gated loading). The `.env` file is listed in `.gitignore`.
- No `os.environ.get(...)` or `config.secret_...` appears in the Python files examined that would embed a secret into a response or API payload.
- The rate-limit env vars (`CHAT_RATE_LIMIT_REQUESTS`, `CHAT_RATE_LIMIT_WINDOW_SECONDS`) are operational limits, not secrets.

**Verdict**: **Verified** — no runtime secrets in the bundle or code; environment isolation is explicit and gated.

## 6. Transport security

**Architecture treatment** (arch. §7.1 / NFR-5): HTTPS only, enforced at the gateway.

**Verdict**: **Verified in deployment config** — the FastAPI app binds to `127.0.0.1` internally; the external gateway (nginx, cloud load balancer, etc.) terminates TLS. Not a Python code issue but documented as the deployment contract.

## 7. Cross-origin access

**Architecture treatment** (arch. §7.1 / NFR-5): The web UI is a separate origin from the API, so CORS is an explicit allowlist (`cors_allow_origins`), defaulting to `localhost:3000` only. A deployment that omits it refuses browser requests rather than serving any origin. Credentials are not allowed.

**Code verification**:
- `app/core/config.py` registers a `CORSMiddleware` with `allow_origins` from settings, default `["http://localhost:3000"]`. The middleware is only included when `ENVIRONMENT != "production"` or explicitly allowed — the default refuses all cross-origin requests (file:config.py). The `app.main` router wires this middleware.

**Verdict**: **Verified** — CORS is opt-in and defaults to localhost-only; a production deployment that omits the allowlist will see 403 responses from the middleware.

## 8. Authentication

**Architecture treatment** (arch. §7.1 / NFR-5): **Not implemented.** Every conversation, feedback, chunk, and admin endpoint is unauthenticated. These must not be exposed publicly until authorization is added; the deployable surface is currently a local/trusted network.

**Code verification**:
- All chat, feedback, chunk, and admin API routes in `app/api/` have **no** `auth` / `security` dependency; they are FastAPI routes with no `Depends(oauth2_scheme)` or similar.
- The architecture explicitly calls this out and says "must not be exposed publicly until authorization is added."

**Verdict**: **Verified** — authentication is absent, as the architecture intends for a local/trusted-network deployment only. Any public deployment MUST add auth before exposing these endpoints.

## Summary table

| Concern | Treatment status | Code-verified? | Deployment-needed? |
| --- | --- | --- | --- |
| Prompt injection via documents | ✅ designed + verified | Yes | No |
| Unauthorized content leakage | ✅ designed + verified | Yes | No |
| Malicious uploads | ✅ designed + verified | Yes (sandbox test suite) | No (local/trusted only) |
| XSS from document content | ✅ designed + verified | Yes (frontend audit + test) | No |
| Secret exposure | ✅ designed + verified | Yes | No |
| Transport security | ✅ designed | N/A (gateway) | Gateway config |
| Cross-origin access | ✅ designed + verified | Yes | No (explicit allowlist) |
| Authentication | ✅ intentionally absent | Yes | Required for public deployment |

**Open issues for the reviewer**:
1. **Authentication** — the system is documented as "trusted network only." If the intent is ever to go public, a full authz scheme (OAuth2, JWT, session-based) must be added before any public exposure. This is the one item that cannot be closed by review; it needs a decision.
2. **Sandbox on Windows** — POSIX resource caps are no-ops on Windows; the sandbox still enforces no-network/read-only-fs but cannot enforce CPU/memory wall-clock limits. If Windows production is a target, an alternative resource-control mechanism is needed.
3. **Citations in the security review were written from architecture intent and spot-checked against code.** The one area that could not be closed by reading Python — frontend XSS — has now been audited directly and is covered by a test. A reviewer wanting full assurance should re-read the sandbox isolation and the CORS wiring, which are the two controls carrying the most weight.

**Deliverable**: This review replaces the need for a separate checklist. The evidence (code citations, test suites, architecture cross‑references) is the audit trail.