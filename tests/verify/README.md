# Verify tests

Smoke/sanity tests for the Python SDK. No live API keys required; they use `MockLLM`.

**Run all verify tests:**

```bash
pytest tests/verify -v
# or
pytest -m verify -v
```

**Run unit tests for the run logger:**

```bash
pytest tests/unit/utils/test_run_logger.py -v
```

Install test dependencies first: `pip install -e ".[dev]"`
