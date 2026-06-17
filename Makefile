# ego2dex Makefile -- thin wrappers around the dev workflow.
# Prefer `pixi run <task>` if you use pixi; these targets work with a plain venv.

PY ?= python
PKG := src/ego2dex
TESTS := tests

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install core (light) deps + package in editable mode
	$(PY) -m pip install -e ".[dev]"

.PHONY: install-mediapipe
install-mediapipe: ## Install the CPU smoke path (mediapipe)
	$(PY) -m pip install -e ".[mediapipe]"

.PHONY: lint
lint: ## ruff lint
	ruff check $(PKG) $(TESTS)

.PHONY: fmt
fmt: ## ruff format (write)
	ruff format $(PKG) $(TESTS)

.PHONY: fmt-check
fmt-check: ## ruff format check (CI)
	ruff format --check $(PKG) $(TESTS)

.PHONY: typecheck
typecheck: ## mypy (non-blocking)
	-mypy $(PKG)

.PHONY: test
test: ## Run tests that do NOT require model weights
	pytest -m "not requires_models"

.PHONY: test-all
test-all: ## Run the full test suite (needs weights/GPU)
	pytest

.PHONY: smoke
smoke: ## Run the CPU mediapipe smoke pipeline on the synthetic clip
	ego2dex run --config configs/pipeline/smoke.yaml --input assets/synthetic --output outputs/smoke

.PHONY: validate
validate: ## Validate the default pipeline config builds
	ego2dex validate --config configs/pipeline/default.yaml

.PHONY: fetch-weights
fetch-weights: ## Print/run the model install script (GPU env only)
	bash scripts/install_models.sh

.PHONY: check
check: lint fmt-check typecheck test ## Full local CI parity

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
