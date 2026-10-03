# Contributing to corpus-assay

Thanks for your interest! Bug reports, feature requests, and pull requests are welcome.

## Development setup

You need [uv](https://docs.astral.sh/uv/) and a stable [Rust toolchain](https://rustup.rs).

```bash
git clone https://github.com/mrcabbage972/corpus-assay
cd corpus-assay
make dev        # uv sync + build the Rust extension in place
make check      # lint (ruff, pyright, rustfmt, clippy) + tests
```

The package lives in `python/corpus_assay/` and the Rust extension in `src/`.
`uv sync` rebuilds the extension when Rust sources change. You can also rebuild
explicitly with `make dev` (or `make release-dev` for an optimized build). Tests always
run against the built extension, so rebuild after editing Rust.

Useful targets:

| Target | What it does |
| --- | --- |
| `make test` | Full test suite |
| `make e2e` | Offline end-to-end CLI tests only (`pytest -m e2e`) |
| `make test-cov` | Tests with coverage |
| `make lint` | Checks only; never modifies files |
| `make fmt` | Apply ruff and rustfmt formatting and safe fixes |
| `make wheel` | Build a release wheel into `dist/` |

## Pull requests

- Keep changes focused, and add or update tests for behavior changes.
- `make check` must pass. CI runs it on Linux, macOS, and Windows for Python 3.12 and 3.13.
- Update `CHANGELOG.md` under *Unreleased* for user-visible changes.
- Changing the normalization pipeline, hashing, or a binary format changes what existing
  indexes mean. Call it out in the PR and the changelog.
- The spaced-seed golden vectors (`tests/fixtures/spaced_golden.json`) should only change
  when spaced-seed behavior is meant to change. Regenerate them with
  `uv run python tests/test_spaced_golden.py`.

## Releasing (maintainers)

1. Bump `version` in `Cargo.toml` (the single source of the package version) and date
   the CHANGELOG section.
2. Optionally dry-run: run the *Release* workflow manually with `publish: testpypi`.
3. Tag and push: `git tag -a vX.Y.Z -m "corpus-assay X.Y.Z" && git push origin vX.Y.Z`.
   The workflow then:
   1. builds the wheels and sdist;
   2. smoke-tests them on every platform;
   3. publishes to TestPyPI and verifies the install;
   4. publishes to PyPI after approval of the `pypi` environment;
   5. creates the GitHub Release.
