.PHONY: build-dev build-release clean test install-build-deps install-dev setup-dev format lint check-deps help install-gmsh

PYTHON := python3
PIP := $(PYTHON) -m pip
NPROC := $(shell command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu)

install-build-deps:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install "scikit-build-core>=0.10" "nanobind>=2.5" "numpy>=2.0,<3.0"
install-dev:
	$(PIP) install -e .[dev]
setup-dev: install-build-deps install-dev install-gmsh
	pre-commit install
	@echo "Development environment setup complete!"
build-dev: install-build-deps install-gmsh
	CMAKE_BUILD_PARALLEL_LEVEL=$(NPROC) SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG CMAKE_BUILD_TYPE=Debug $(PIP) install -e .[dev] -v --no-build-isolation
build-release: install-build-deps install-gmsh
	CMAKE_BUILD_PARALLEL_LEVEL=$(NPROC) CMAKE_BUILD_TYPE=Release $(PIP) install -e .[dev] -v --no-build-isolation
build-deb: install-build-deps install-gmsh
	CMAKE_BUILD_PARALLEL_LEVEL=$(NPROC) SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG CMAKE_BUILD_TYPE=RelWithDebInfo $(PIP) install -e .[dev] -v --no-build-isolation
# Force rebuild (clean first)
rebuild-dev: clean build-dev
rebuild-release: clean build-release
clean:
	rm -rf build/
#	rm -rf dist/
#	rm -rf *.egg-info/
#	find . -name "*.so" -delete
#	find . -name "*.pyc" -delete
#	find . -name "__pycache__" -type d -exec rm -rf {} +
#	find . -name ".pytest_cache" -type d -exec rm -rf {} +
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
	@$(PYTHON) -c "import gmsh; print('✓ gmsh available')" || echo "✗ gmsh missing"
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
install-gmsh:
	@echo "Checking for gmsh Python module..."
	@if $(PYTHON) -c "import gmsh" >/dev/null 2>&1; then \
		echo "✓ Gmsh already installed."; \
	else \
		echo "gmsh not found. Trying pip install..."; \
		if $(PIP) install --no-cache-dir gmsh >/dev/null 2>&1; then \
			echo "✓ Installed gmsh via pip."; \
		else \
			echo "pip install failed. Building gmsh from source..."; \
			rm -rf build_gmsh; \
			git clone --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git build_gmsh; \
			cd build_gmsh && mkdir build && cd build && \
			cmake -DENABLE_BUILD_DYNAMIC=1 -DCMAKE_BUILD_TYPE=Release .. && \
			make -j$(NPROC); \
			SITE_PACKAGES=$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))'); \
			echo "Installing gmsh.py and libgmsh to $$SITE_PACKAGES"; \
			cp libgmsh* "$$SITE_PACKAGES/"; \
			cp ../api/gmsh.py "$$SITE_PACKAGES/"; \
			cd ../.. && rm -rf build_gmsh; \
			echo "✓ Gmsh installed from source."; \
		fi; \
	fi
help:
	@echo "Available targets:"
	@echo "  setup-dev      - Set up complete development environment"
	@echo "  build-dev      - Build in debug mode"
	@echo "  build-release  - Build in release mode"
	@echo "  rebuild-dev    - Clean and build in debug mode"
	@echo "  rebuild-release- Clean and build in release mode"
	@echo "  install-gmsh   - Build and install gmsh Python module from source"
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
