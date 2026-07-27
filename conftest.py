"""Global test configuration.

Test levels (cumulative):
    pytest tests/ -m smoke          → smoke: core math only (88 tests, ~1min)
    pytest tests/                   → quick: + fitting & analysis (203 tests, ~3min)
    pytest tests/ --full            → full: + integration pipelines (341 tests, ~10min)
    pytest tests/ --all             → all: + slow tier 3-4 heteroscedastic (394 tests, ~12min)
"""
import jax
jax.config.update("jax_enable_x64", True)

import pytest


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", default=False,
                     help="Run all tests including slow tier 3-4 tests")
    parser.addoption("--full", action="store_true", default=False,
                     help="Run full tests including integration (but not slow tier 3-4)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (tier 3-4 heteroscedastic tests)")
    config.addinivalue_line("markers", "integration: mark test as integration-level (fitting full pipelines)")
    config.addinivalue_line("markers", "smoke: mark test as fast smoke test (core math, no fitting)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return  # run everything

    skip_slow = pytest.mark.skip(reason="use --all to run slow tier 3-4 tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    if not config.getoption("--full"):
        skip_integration = pytest.mark.skip(reason="use --full to run integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
