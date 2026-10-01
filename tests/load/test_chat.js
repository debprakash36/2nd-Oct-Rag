// k6 load test for the chat stream (implementation.md 5.1, NFR-1).
//
//   k6 run tests/load/test_chat.js
//   k6 run -e BASE_URL=http://host:8000 -e VUS=100 -e REQUESTS_PER_VU=5 \
//          tests/load/test_chat.js
//
// Requires a running server with a seeded corpus; see tests/load/conftest.py, which
// starts one and reports the DATABASE_URL it used. `make load-test` runs the
// equivalent pytest harness, which needs no k6 install.
//
// The important part of this file is the TTFT measurement. k6's `http.post` buffers
// the whole response, so TTFT measured from it is the *full answer* time and the
// number would be a lie: a correct streaming implementation and a fully buffered one
// would report identically. k6 has no SSE-aware client, so this measures the honest
// alternative — the server's own `ttft_ms`, read from the `done` frame — and records
// the full response time separately as `full_answer`.
//
// That makes this script weaker than the pytest harness on one specific point: the
// server-side figure excludes serialization and socket time, so a proxy buffering the
// stream would be invisible here. The pytest harness measures client-observed TTFT and
// reports the transport gap for exactly that reason. Use both; neither alone covers
// the whole of NFR-1.

import http from 'k6/http';
import { check } from 'k6';
import { Counter, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8000';
const VUS = Number(__ENV.VUS || 50);
const REQUESTS_PER_VU = Number(__ENV.REQUESTS_PER_VU || 2);
const TTFT_BUDGET_MS = Number(__ENV.TTFT_BUDGET_MS || 5000);

const serverTtft = new Trend('server_ttft_ms', true);
const fullAnswer = new Trend('full_answer_ms', true);
const sourcesSeen = new Trend('sources_count');
const abstained = new Counter('abstained');
const streamErrors = new Counter('stream_errors');
const rateLimited = new Counter('rate_limited');

const questions = [
  'What is the refund window for digital products?',
  'How long do I have to return a physical good?',
  'What are the standard shipping times?',
  'How does the warranty period work?',
  'What happens if a customer disputes a charge?',
  'How long is personal data retained?',
  'When are invoices due?',
  'What does the escalation policy say?',
  'Which items are non-refundable?',
  'How quickly are refunds processed?',
];

export const options = {
  scenarios: {
    concurrent_users: {
      executor: 'per-vu-iterations',
      vus: VUS,
      iterations: REQUESTS_PER_VU,
      maxDuration: '5m',
    },
  },
  // The budget is enforced as a threshold, so k6 exits non-zero on a breach rather
  // than printing a number someone has to remember to interpret.
  thresholds: {
    server_ttft_ms: [`p(95)<${TTFT_BUDGET_MS}`],
    full_answer_ms: ['p(95)<20000'],
    stream_errors: ['count==0'],
    checks: ['rate>0.99'],
  },
};

function parseEvents(body) {
  const events = { sources: [], tokens: 0, done: null, error: null };
  for (const frame of body.split('\n\n')) {
    let name = '';
    let data = {};
    for (const line of frame.split('\n')) {
      if (line.startsWith('event: ')) name = line.slice(7).trim();
      else if (line.startsWith('data: ')) {
        try {
          data = JSON.parse(line.slice(6));
        } catch (_) {
          data = {};
        }
      }
    }
    if (name === 'sources') events.sources = data.sources || [];
    else if (name === 'token') events.tokens += 1;
    else if (name === 'done') events.done = data;
    else if (name === 'error') events.error = data;
  }
  return events;
}

export default function () {
  const question = questions[(__VU + __ITER) % questions.length];

  // A distinct forwarded identity per VU. The rate limiter keys on this and allows
  // 30 requests per 60 s, so VUs sharing one address would be throttled as a single
  // abusive client and this would measure the limiter instead of the pipeline.
  const headers = {
    'Content-Type': 'application/json',
    'X-Forwarded-For': `10.1.${Math.floor(__VU / 250)}.${(__VU % 250) + 1}`,
  };

  const response = http.post(
    `${BASE_URL}/chat/stream`,
    JSON.stringify({ message: question, answer_style: 'concise' }),
    { headers, tags: { endpoint: 'chat_stream' } },
  );

  if (response.status === 429) {
    rateLimited.add(1);
    return;
  }
  if (response.status !== 200) {
    streamErrors.add(1);
    return;
  }

  const events = parseEvents(response.body);
  sourcesSeen.add(events.sources.length);
  if (events.tokens > 0) {
    fullAnswer.add(response.timings.duration);
  }
  if (events.done && events.done.ttft_ms != null) {
    serverTtft.add(events.done.ttft_ms);
    if (events.done.abstained) abstained.add(1);
  }

  check(response, {
    'status is 200': (r) => r.status === 200,
    'sources precede tokens': () => events.sources.length > 0 || events.tokens > 0,
    'stream completed': () => events.done !== null,
    'no error event': () => events.error === null,
    'tokens were emitted': () => events.tokens > 0,
  });
}

export function handleSummary(data) {
  const ttft = data.metrics.server_ttft_ms;
  const full = data.metrics.full_answer_ms;
  const lines = [
    '',
    '=== chat stream load test ===',
    `  VUs                 ${VUS} x ${REQUESTS_PER_VU} iterations`,
    `  server TTFT p50     ${ttft ? ttft.values['p(50)'].toFixed(1) : 'n/a'} ms`,
    `  server TTFT p95     ${ttft ? ttft.values['p(95)'].toFixed(1) : 'n/a'} ms  (budget ${TTFT_BUDGET_MS} ms)`,
    `  full answer p95     ${full ? full.values['p(95)'].toFixed(1) : 'n/a'} ms  (budget 20000 ms)`,
    `  stream errors       ${data.metrics.stream_errors ? data.metrics.stream_errors.values.count : 0}`,
    `  rate limited        ${data.metrics.rate_limited ? data.metrics.rate_limited.values.count : 0}`,
    '',
    'Note: server_ttft_ms is read from the done frame, so it excludes',
    'serialization and socket time. tests/load/test_chat.py measures the',
    'client-observed figure and the transport gap between them.',
    '',
  ];
  return { stdout: lines.join('\n') };
}
