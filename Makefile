.PHONY: help install install-dev test test-cov lint format clean build publish serve docs

# Default target
help:
	@echo "SPADE - Makefile Commands"
	@echo "========================="
	@echo ""
	@echo "Development:"
	@echo "  make install        Install package in development mode"
	@echo "  make install-dev    Install with all development dependencies"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linters (ruff, mypy)"
	@echo "  make format         Format code with black"
	@echo "  make serve          Start API server (development mode)"
	@echo ""
	@echo "Build & Release:"
	@echo "  make clean          Remove build artifacts"
	@echo "  make build          Build distribution packages"
	@echo "  make publish        Publish to PyPI (requires credentials)"
	@echo "  make publish-test   Publish to TestPyPI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation (if available)"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,api,cli]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=spade --cov-report=html --cov-report=term

# Code quality
lint:
	@echo "Running ruff..."
	ruff check src/spade tests/ || true
	@echo ""
	@echo "Running mypy..."
	mypy src/spade --ignore-missing-imports || true

format:
	black src/spade tests/
	ruff check --fix src/spade tests/ || true

# API server
serve:
	python -m spade serve --port 8000

# Build & distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	python -m build

publish: build
	twine check dist/*
	twine upload dist/*

publish-test: build
	twine check dist/*
	twine upload --repository testpypi dist/*

# Docker
docker-build:
	docker build -t spade-forensics:latest .

docker-run:
	docker run -p 8000:8000 spade-forensics:latest

# Documentation
docs:
	@echo "Documentation target - add sphinx or mkdocs build here"

# Quick checks before commit
check: format lint test
	@echo " All checks passed!"

# Version bump helpers
version:
	@python -c "from spade import __version__; print(f'Current version: {__version__}')"
