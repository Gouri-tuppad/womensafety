.PHONY: help up down logs migrate seed train test lint dev

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

seed: ## Seed synthetic data
	python -m app.data.seed

train: ## Train the ML safety model
	python -m app.ml.model

test: ## Run pytest suite
	pytest tests/ -v --tb=short

lint: ## Run ruff linter
	ruff check app/ tests/

dev: ## Run API locally (no Docker)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
