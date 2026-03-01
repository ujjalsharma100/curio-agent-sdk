.PHONY: test test-unit test-integration test-e2e test-perf test-all test-cov test-fast test-watch

test:                   ## Run all tests (except live and slow)
	pytest tests/ -v -m "not live and not slow"

test-unit:              ## Run unit tests only
	pytest tests/unit -v

test-integration:       ## Run integration tests only
	pytest tests/integration -v -m "integration"

test-e2e:               ## Run end-to-end tests only
	pytest tests/e2e -v -m "e2e"

test-perf:              ## Run performance tests
	pytest tests/performance -v -m "slow"

test-live:              ## Run live API tests (requires API keys)
	pytest tests/live -v -m "live"

test-all:               ## Run everything
	pytest tests/ -v

test-cov:               ## Run with coverage report
	pytest tests/ -v --cov=src/curio_agent_sdk --cov-report=html --cov-report=term -m "not live"

test-fast:              ## Run fast unit tests
	pytest tests/unit -v -x
