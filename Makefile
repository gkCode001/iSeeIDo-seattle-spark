.PHONY: up down serve ingest bench test lint fmt doctor

PY ?= python3

up:                ## docker compose: milvus, vllm, nim, services
	docker compose up -d

down:
	docker compose down

serve:             ## start the one model process (SPEC §10 D1/D3) — run this first
	nohup ./scripts/serve_models.sh & echo "llama-server starting; log: /tmp/spark-llama-server.log"

ingest:            ## run M1 against the configured source
	$(PY) -m services.ingest

bench:             ## time a single caption — the number that governs everything (SPEC §9 block 0)
	$(PY) -m services.ingest.bench

test:              ## stdlib unittest — no third-party packages required
	$(PY) -m unittest discover -s tests -t . -v

lint:
	ruff check .

fmt:
	ruff format .

doctor:            ## check this box against the prerequisites in CLAUDE.md
	@$(PY) scripts/doctor.py
