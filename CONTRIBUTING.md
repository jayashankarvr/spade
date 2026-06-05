# Contributing

## Setup

```bash
git clone https://github.com/spade-forensics/spade.git
cd spade
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

## Workflow

1. Create a branch
2. Make changes
3. Run `pytest` and `black .`
4. Submit PR

## Code style

- Black for formatting
- Type hints where helpful (not everywhere)
- Docstrings for public APIs

## Good first issues

- Add more descriptor types
- Improve CLI output
- Add visualization helpers
- Write more tests

## Future work

**Phase 2 (Completed):**
- ✅ Multi-scale pyramids, 256-dim descriptors, LSH, REST API

**Phase 3 (In Progress):**
- GPU acceleration (code ready, needs testing)
- Learned descriptors (code ready, needs training)
- Distributed search (sharded indexes implemented)
- Ray integration for massive scale
- Bayesian probability fusion
- Graph Neural Networks for spatial reasoning

## License

Apache 2.0 - contributions under same license.
