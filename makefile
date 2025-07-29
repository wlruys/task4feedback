.PHONY: build-dev build-release clean test install-build-deps install-dev setup-dev format lint check-deps help

PYTHON := python3
PIP := $(PYTHON) -m pip

install-build-deps:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install "scikit-build-core>=0.10" "nanobind>=2.5" "numpy>=2.0,<3.0"
install-dev:
	$(PIP) install -e .[dev]
setup-dev: install-build-deps install-dev
	pre-commit install
	@echo "Development environment setup complete!"
build-dev: install-build-deps
	SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG CMAKE_BUILD_TYPE=Debug $(PIP) install -e .[dev] -v --no-build-isolation
build-release: install-build-deps
	CMAKE_BUILD_TYPE=Release $(PIP) install -e .[dev] -v --no-build-isolation -Ceditable.rebuild=true
build-deb: install-build-deps
	SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG CMAKE_BUILD_TYPE=RelWithDebInfo $(PIP) install -e .[dev] -v --no-build-isolation -Ceditable.rebuild=true
# Force rebuild (clean first)
rebuild-dev: clean build-dev
rebuild-release: clean build-release
clean:
	rm -rf build/
# 	rm -rf dist/
# 	rm -rf *.egg-info/
# 	find . -name "*.so" -delete
# 	find . -name "*.pyc" -delete
# 	find . -name "__pycache__" -type d -exec rm -rf {} +
# 	find . -name ".pytest_cache" -type d -exec rm -rf {} +
test:
	pytest test/ -v
test-cov:
	pytest test/ -v --cov=src/task4feedback --cov-report=html --cov-report=term
format:
	ruff format .
	ruff check --fix .
lint:
	ruff check .
check-deps:
	@echo "Checking build dependencies..."
	@$(PYTHON) -c "import scikit_build_core; print('✓ scikit-build-core available')" || echo "✗ scikit-build-core missing"
	@$(PYTHON) -c "import nanobind; print('✓ nanobind available')" || echo "✗ nanobind missing"
	@$(PYTHON) -c "import numpy; print('✓ numpy available')" || echo "✗ numpy missing"
	@cmake --version || echo "✗ cmake not found"
wheel: install-build-deps
	$(PIP) wheel . -w dist/ --no-build-isolation
sdist:
	$(PYTHON) -m build --sdist
dev: setup-dev build-dev test
	@echo "Development build and test complete!"
deb: clean build-deb test
	@echo "Release build and test complete!"
release: clean build-release test
	@echo "Release build and test complete!"
help:
	@echo "Available targets:"
	@echo "  setup-dev      - Set up complete development environment"
	@echo "  build-dev      - Build in debug mode"
	@echo "  build-release  - Build in release mode"
	@echo "  rebuild-dev    - Clean and build in debug mode"
	@echo "  rebuild-release- Clean and build in release mode"
	@echo "  test           - Run tests"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  format         - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  clean          - Remove build artifacts"
	@echo "  check-deps     - Check if build dependencies are available"
	@echo "  wheel          - Build wheel package"
	@echo "  sdist          - Build source distribution"
	@echo "  dev            - Complete dev setup + build + test"
	@echo "  release        - Clean + release build + test"
	@echo "  help           - Show this help message"
