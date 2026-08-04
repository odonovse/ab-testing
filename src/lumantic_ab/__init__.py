"""A/B testing over raw or summary inputs.

Tests live in paradigm subpackages, e.g.:

    from lumantic_ab.frequentist import t_test_summary
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lumantic-ab")
except PackageNotFoundError:
    # Running from a source checkout without an install (e.g. PYTHONPATH=src).
    __version__ = "0.0.0.dev0"
