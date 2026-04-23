RUN_COMMAND = poetry run

# Docker Configuration
PROJECT_NAME = service-core-ml
DOCKER_REGISTRY = ghcr.io
DOCKER_IMAGE_NAME = $(DOCKER_REGISTRY)/enlitia/$(PROJECT_NAME)
DOCKER_DEFAULT_PLATFORM = linux/amd64

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# CLEAN -------------------------------------------------------------------------------------------
pyc-clean: ## Remove python byte code files
	find . -iname "*.pyc" -delete

rm-venv-clean: ## Remove the virtual environment
	poetry env remove python || true

py-cache-clean: ## Remove all .pyc and .pyo files as well as __pycache__ directories
	find . | grep -E "(/__pycache__$$|\.pyc$$|\.pyo$$)" | xargs rm -rf

clean: pyc-clean rm-venv-clean py-cache-clean ## Cleans the environment

# DEV-SETUP ---------------------------------------------------------------------------------------
install: ## Install dependencies and set up pre-commit hooks
	poetry install
	$(RUN_COMMAND) pre-commit install
	$(RUN_COMMAND) pre-commit install --hook-type commit-msg

hooks:  ## Installs git hooks
	$(RUN_COMMAND) pre-commit install
	$(RUN_COMMAND) pre-commit install --hook-type commit-msg

lock: ## Update poetry lock file
	poetry lock

deps: lock ## Installs all project package dependencies
	poetry install

repo: ## Setup python packages private repo (not used in this project)
	@echo "Private repo setup not needed for this project"

dev-setup: deps hooks ## Sets up the development environment

# TESTS  ------------------------------------------------------------------------------------------

test: ## Run all tests (always passes if no tests)
	@echo "No tests to run. Passing."

# FORMAT ------------------------------------------------------------------------------------------
format:  ## Formats the code
	$(RUN_COMMAND) black -l 120 ./src
	$(RUN_COMMAND) isort ./src

# CODE ANALYSIS ---------------------------------------------------------------------------------
type-analysis: ## Checks the code regarding types
	$(RUN_COMMAND) mypy ./src

lint-analysis:  ## Lints the code
	$(RUN_COMMAND) ruff ./src

lint-fix:
	$(RUN_COMMAND) ruff --fix ./src

static-analysis: type-analysis lint-analysis ## Checks the code for errors and inconsistency

check: format lint-fix static-analysis test ## Run all checks before committing

# ALL ---------------------------------------------------------------------------------------------
poetry-all: clean dev-setup format static-analysis  ## Runs all development flow steps

# DOCKER ------------------------------------------------------------------------------------------
docker-build:
	docker build \
		--build-arg DB_USER="$(DB_USER)" \
		--build-arg DB_PASSWORD="$(DB_PASSWORD)" \
		--build-arg SM_SETTINGS_MODULE="production" \
		--platform $(DOCKER_DEFAULT_PLATFORM) \
		-t $(PROJECT_NAME) .

docker-run:
	docker run \
		-e DB_USER="$(DB_USER)" \
		-e DB_PASSWORD="$(DB_PASSWORD)" \
		-e SM_SETTINGS_MODULE="production" \
		--rm \
		$(PROJECT_NAME)

docker-login:
	docker login -u $(DOCKER_LOGIN) -p $(DOCKER_PASS) $(DOCKER_REGISTRY)

docker-push: docker-login
	docker tag $(PROJECT_NAME) $(DOCKER_IMAGE_NAME):latest
	docker push $(DOCKER_IMAGE_NAME):latest

docker-deploy: docker-build docker-push
