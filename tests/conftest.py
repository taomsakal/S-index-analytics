"""Pytest configuration: forces matplotlib to use non-interactive Agg backend."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure():
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass
