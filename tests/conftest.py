"""Pytest configuration for the test suite.

Adds a ``slow`` marker (used by the S-1 cross-engine regression suite to gate the
large multi-hundred-page bodies) and a ``--runslow`` opt-in. By default slow tests
are skipped so ``make test`` stays fast; run the full corpus with::

    pytest tests/test_s1_regression.py --runslow
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (e.g. large S-1 bodies in the regression suite)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselected by default; enable with --runslow)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
