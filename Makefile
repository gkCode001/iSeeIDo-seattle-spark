.PHONY: up down ingest bench lint fmt doctor

PY ?= python3

up:                ## docker compose: milvus, vllm, nim, services
	docker compose up -d

down:
	docker compose down

ingest:            ## run M1 against the configured source
	$(PY) -m services.ingest

bench:             ## time a single caption — the number that governs everything (SPEC §9 block 0)
	$(PY) -m services.ingest.bench

lint:
	ruff check .

fmt:
	ruff format .

doctor:            ## check this box against the prerequisites in CLAUDE.md
	@$(PY) scripts/doctor.py
