.PHONY: help up down logs migrate seed ingest simulate_crowd train test lint dev

# Variables (override on command line: make seed ZONE=subramanyapura)
ZONE ?= peenya
CSV  ?=

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start all services
	docker compose up --build -d

down: ## Stop all services
	docker compose down -v

logs: ## Tail API logs
	docker compose logs -f api

migrate: ## Run Alembic head migration
	alembic upgrade head

seed: ## Full Step 2 pipeline: OSM ingest + crime seed + crowd simulate (ZONE=peenya|subramanyapura CSV=path)
	python -m data.seed $(ZONE) $(if $(CSV),--csv $(CSV),)

ingest: ## Download OSM graph and POIs for ZONE only
	python -m data.ingest_osm $(ZONE)

simulate_crowd: ## Simulate hourly H3 crowd counts for ZONE only
	python -m data.simulate_crowd $(ZONE)

train: ## Train the ML safety model
	python -m app.ml.train_risk_model

test: ## Run pytest suite
	pytest tests/ -v --tb=short

lint: ## Run ruff linter
	ruff check app/ tests/ data/

dev: ## Run API locally (no Docker)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
