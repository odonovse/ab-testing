"""Frequentist A/B tests.

    from lumantic_ab.frequentist import t_test_summary

Re-exports only — no logic lives here. Anything shared between the test
modules belongs in results.py, not in this file: submodules import from
results.py, and a submodule importing from its own package's __init__
is a circular import.
"""

from .results import TTestResult
from .summary import t_test_summary

__all__ = ["TTestResult", "t_test_summary"]
