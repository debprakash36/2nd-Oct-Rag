.PHONY: check lint format typecheck test corpus ingest migrate dev venv chroma-sync \
	load-test k6-load clean-cache \
	web-install web-lint web-typecheck web-test web-check web-dev web-build

PY := .venv/Scripts/python.exe
RUFF := .venv/Scripts/python.exe -m ruff
MYPY := .venv/Scripts/python.exe -m mypy
PYTEST := .venv/Scripts/python.exe -m pytest
export PYTHONPATH := $(CURDIR)

venv:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

lint:
	$(RUFF) check app tests scripts alembic

format:
	$(RUFF) format app tests

typecheck:
	$(MYPY) app

test:
	$(PYTEST)

check: lint typecheck test

# NFR-1 gate (implementation.md 5.1): 50 concurrent users, p95 TTFT <= 5 s.
# Needs a seeded corpus, which the harness builds once and caches in
# .load_cache/. Writes docs/load_test_results.json for docs/perf_report.md.
# The k6 variant is `make k6-load` against an already-running server.
load-test:
	$(PYTEST) -m load tests/load/ -v

k6-load:
	k6 run tests/load/test_chat.js

clean-cache:
	rm -rf .load_cache

# Generate a synthetic corpus. Needed before `ingest` on a fresh checkout, and
# `ingest` fails with a clear message if the directory is missing.
corpus:
	$(PY) scripts/make_corpus.py --out ./samples --count 100

ingest:
	$(PY) scripts/ingest_corpus.py --dir ./samples --stats

migrate:
	$(PY) -m alembic upgrade head

# Project authoritative chunks into the Chroma index. Ingestion does not write to
# Chroma, so retrieval against `vector_store=chroma` returns nothing until this
# has run. Safe to re-run.
chroma-sync:
	$(PY) scripts/sync_chroma_index.py

dev:
	$(PY) -m uvicorn app.main:app --reload

# --- Web UI (Next.js) ------------------------------------------------------
# Kept separate from the Python targets: the two toolchains have no shared install,
# and a failure here should not look like a backend failure.

web-install:
	cd web && npm install

web-lint:
	cd web && npm run lint

web-typecheck:
	cd web && npm run typecheck

web-test:
	cd web && npm test

web-build:
	cd web && npm run build

web-check: web-lint web-typecheck web-test

web-dev:
	cd web && npm run dev
