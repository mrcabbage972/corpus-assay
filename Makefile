.PHONY: dev release-dev wheel sdist test test-cov e2e lint fmt check clean

# Editable install with the Rust extension built (rerun after editing src/*.rs).
dev:
	uv sync
	uv run maturin develop --uv

# Same, with an optimized Rust build (for benchmarking or large scans).
release-dev:
	uv sync
	uv run maturin develop --uv --release

wheel:
	uv run maturin build --release --locked --out dist

sdist:
	uv run maturin sdist --out dist

test:
	uv run pytest

test-cov:
	uv run pytest --cov=corpus_assay --cov-report=term-missing --cov-report=xml

e2e:
	uv run pytest -m e2e

# Checks only; never modifies files (see `fmt`).
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	cargo fmt --all --check
	uv run cargo clippy --locked --all-targets -- -D warnings

fmt:
	uv run ruff check . --fix
	uv run ruff format .
	cargo fmt --all

check: lint test

clean:
	cargo clean
	rm -rf dist build .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find python \( -name '*.so' -o -name '*.pyd' \) -delete
