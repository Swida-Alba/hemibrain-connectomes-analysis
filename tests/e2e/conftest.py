"""Shared pytest configuration for the real-data end-to-end suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: real-data end-to-end tests (need the hemibrain vector cache "
        "and/or live data access; skip when unavailable)",
    )
