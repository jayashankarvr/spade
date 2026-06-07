# Contributing

Thanks for your interest in SPADE! Contributions are welcome via pull request.

## Development setup

```bash
# Fork the repo on GitHub, then clone your fork:
git clone https://github.com/<your-username>/spade.git
cd spade
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Workflow

`main` is the integration branch and is protected. External contributors don't
push directly - open a pull request from a fork:

1. **Fork** this repository and clone your fork.
2. Create a topic branch: `git checkout -b my-change`.
3. Make your change, with tests.
4. Run the checks locally (these also run in CI on your PR):
   ```bash
   pytest
   ruff check src/
   black --check src/
   ```
5. Push to your fork and **open a pull request** against `main`.
6. CI must pass and the maintainer will review before merge.

## Code style

- **black** for formatting (`black src/`), line length 88.
- **ruff** for linting (`ruff check src/`).
- Type hints where they add clarity (not required everywhere).
- Docstrings for public APIs.

## Scope and direction

Before proposing a large feature, please read [ROADMAP.md](ROADMAP.md) - it
describes where the project is headed and, importantly, what is intentionally
**out of scope**. Issues and PRs that align with the roadmap are easiest to land.

Good places to start:
- Test coverage and edge cases
- Documentation and examples
- Benchmark harness and datasets (see ROADMAP §8)

## License

By contributing, you agree that your contributions are licensed under the
project's Apache 2.0 license.
