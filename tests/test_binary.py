"""
t_test_binary: checked against scipy, plus the closed-form variance identity
it relies on.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_testing.frequentist import t_test_binary

# --- Oracle -----------------------------------------------------------------


def test_matches_scipy_welch(binary_df, binary_arms):
    treatment, control = binary_arms
    got = t_test_binary(binary_df, metric="converted")
    ref = stats.ttest_ind(treatment, control, equal_var=False)

    assert got.t_stat == pytest.approx(ref.statistic, rel=1e-12)
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-12)


def test_absolute_ci_matches_scipy(binary_df, binary_arms):
    treatment, control = binary_arms
    got = t_test_binary(binary_df, metric="converted", alpha=0.05)
    low, high = stats.ttest_ind(
        treatment, control, equal_var=False
    ).confidence_interval(0.95)

    assert got.ci_low == pytest.approx(low, rel=1e-12)
    assert got.ci_high == pytest.approx(high, rel=1e-12)


# --- The identity the binary path leans on ----------------------------------


def test_se_uses_the_closed_form_variance(binary_df, binary_arms):
    """
    For a 0/1 variable, p(1-p)*n/(n-1) is exactly the ddof=1 sample variance.
    binary.py uses the closed form rather than calling .var(), so check the
    standard error it actually produces against the ddof=1 route.

    Asserting the identity on its own would only be testing numpy — it has to
    go through t_test_binary to protect this package.
    """
    treatment, control = binary_arms
    got = t_test_binary(binary_df, metric="converted")

    expected_se = np.sqrt(
        treatment.var(ddof=1) / treatment.size + control.var(ddof=1) / control.size
    )
    assert got.se == pytest.approx(expected_se, rel=1e-12)


def test_proportions_are_recovered(binary_df, binary_arms):
    treatment, control = binary_arms
    got = t_test_binary(binary_df, metric="converted")

    assert got.mean_treatment == pytest.approx(treatment.mean(), rel=1e-12)
    assert got.mean_control == pytest.approx(control.mean(), rel=1e-12)


# --- Guards worth pinning ---------------------------------------------------


def test_non_binary_values_raise():
    """A 0.5 is not a conversion; silently averaging it would be wrong."""
    df = pd.DataFrame(
        {
            "assignment": [True] * 3 + [False] * 3,
            "converted": [0.0, 1.0, 0.5, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="binary"):
        t_test_binary(df, metric="converted")


def test_non_boolean_assignment_raises(binary_df):
    """0/1 ints are not booleans — reject rather than guess the encoding."""
    df = binary_df.assign(assignment=binary_df["assignment"].astype(int))
    with pytest.raises(ValueError, match="boolean"):
        t_test_binary(df, metric="converted")


def test_nan_in_metric_raises():
    """
    A NaN must be reported as missing data, not as a bad encoding. Match on
    the message: if the isfinite check ever moves back below the 0/1 check it
    becomes unreachable, and this test is what notices.
    """
    df = pd.DataFrame(
        {
            "assignment": [True] * 3 + [False] * 3,
            "converted": [0.0, 1.0, np.nan, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        t_test_binary(df, metric="converted")


def test_non_binary_values_still_raise_the_binary_error():
    """Finite but non-binary values must still be caught by the 0/1 check."""
    df = pd.DataFrame(
        {
            "assignment": [True] * 3 + [False] * 3,
            "converted": [0.0, 1.0, 0.5, 1.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="must be binary"):
        t_test_binary(df, metric="converted")
