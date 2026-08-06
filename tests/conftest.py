"""
Shared fixtures for the frequentist test suite.

Everything is built from a fixed seed so the suite is deterministic — a
statistical test that fails one run in twenty is worse than no test at all.
"""

import numpy as np
import pandas as pd
import pytest

SEED = 20260806


@pytest.fixture(scope="session")
def continuous_arms():
    """
    Raw treatment/control arrays for a continuous metric.

    Deliberately unequal in both size (900 vs 300) and variance (sd 5 vs 1).
    A balanced, equal-variance fixture would still pass if Welch's test were
    silently replaced by pooled Student's, so it would not be testing the
    thing this package claims to do.
    """
    rng = np.random.default_rng(SEED)
    treatment = rng.normal(10.0, 5.0, 900)
    control = rng.normal(9.0, 1.0, 300)
    return treatment, control


@pytest.fixture(scope="session")
def binary_arms():
    """Raw treatment/control arrays for a binary metric, unequal sizes."""
    rng = np.random.default_rng(SEED)
    treatment = rng.binomial(1, 0.30, 5000).astype(float)
    control = rng.binomial(1, 0.25, 3000).astype(float)
    return treatment, control


def _unit_level(treatment, control, metric_name):
    """One row per unit, with a boolean assignment column."""
    return pd.DataFrame(
        {
            "assignment": np.concatenate(
                [np.ones(treatment.size, dtype=bool), np.zeros(control.size, dtype=bool)]
            ),
            metric_name: np.concatenate([treatment, control]),
        }
    )


@pytest.fixture
def continuous_df(continuous_arms):
    return _unit_level(*continuous_arms, metric_name="revenue")


@pytest.fixture
def binary_df(binary_arms):
    return _unit_level(*binary_arms, metric_name="converted")


@pytest.fixture
def summary_df(continuous_arms):
    """
    The same data as continuous_df, pre-aggregated to one row per metric-arm.
    Used to check the summary and raw paths agree.
    """
    treatment, control = continuous_arms
    return pd.DataFrame(
        {
            "metric": ["revenue", "revenue"],
            "assignment": [True, False],
            "sample": [treatment.size, control.size],
            "mean": [treatment.mean(), control.mean()],
            "var": [treatment.var(ddof=1), control.var(ddof=1)],
        }
    )
