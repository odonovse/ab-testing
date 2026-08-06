"""
t_test_continuous: checked against scipy, plus the invariants that must hold
whatever the data looks like.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_testing.frequentist import t_test_continuous

# --- Oracle: scipy is the reference implementation --------------------------


def test_matches_scipy_welch(continuous_df, continuous_arms):
    treatment, control = continuous_arms
    got = t_test_continuous(continuous_df, metric="revenue")
    ref = stats.ttest_ind(treatment, control, equal_var=False)

    assert got.t_stat == pytest.approx(ref.statistic, rel=1e-12)
    assert got.p_value == pytest.approx(ref.pvalue, rel=1e-12)


def test_absolute_ci_matches_scipy(continuous_df, continuous_arms):
    treatment, control = continuous_arms
    got = t_test_continuous(continuous_df, metric="revenue", alpha=0.05)
    low, high = stats.ttest_ind(
        treatment, control, equal_var=False
    ).confidence_interval(0.95)

    assert got.ci_low == pytest.approx(low, rel=1e-12)
    assert got.ci_high == pytest.approx(high, rel=1e-12)


def test_is_welch_not_pooled_student(continuous_df, continuous_arms):
    """
    The fixture has unequal variances, so Welch and pooled Student's must
    disagree. If this ever passes, equal_var=False has been lost somewhere.
    """
    treatment, control = continuous_arms
    got = t_test_continuous(continuous_df, metric="revenue")
    pooled = stats.ttest_ind(treatment, control, equal_var=True)

    assert not np.isclose(got.t_stat, pooled.statistic)


def test_unequal_group_sizes_are_supported(continuous_df, continuous_arms):
    """A 3:1 split is not a 50:50 split, and must still be exactly right."""
    treatment, control = continuous_arms
    assert treatment.size != control.size

    got = t_test_continuous(continuous_df, metric="revenue")
    ref = stats.ttest_ind(treatment, control, equal_var=False)
    assert got.t_stat == pytest.approx(ref.statistic, rel=1e-12)


# --- Properties that hold regardless of the data ----------------------------


def test_swapping_arms_flips_the_sign(continuous_df):
    got = t_test_continuous(continuous_df, metric="revenue")
    swapped = t_test_continuous(
        continuous_df.assign(assignment=~continuous_df["assignment"]),
        metric="revenue",
    )

    assert swapped.absolute_diff == pytest.approx(-got.absolute_diff, rel=1e-12)
    assert swapped.t_stat == pytest.approx(-got.t_stat, rel=1e-12)
    # magnitude-only quantities are unchanged by the relabelling
    assert swapped.p_value == pytest.approx(got.p_value, rel=1e-12)
    assert swapped.se == pytest.approx(got.se, rel=1e-12)


def test_larger_alpha_narrows_the_interval(continuous_df):
    tight = t_test_continuous(continuous_df, metric="revenue", alpha=0.01)
    loose = t_test_continuous(continuous_df, metric="revenue", alpha=0.10)

    assert (loose.ci_high - loose.ci_low) < (tight.ci_high - tight.ci_low)


def test_relative_ci_uses_the_delta_method(continuous_df, continuous_arms):
    """
    The relative CI is not the absolute CI divided by the control mean — it
    carries the uncertainty in that denominator too. This is the subtlest
    piece of maths in the package and the easiest thing to "simplify" wrongly.
    """
    treatment, control = continuous_arms
    n_t, n_c = treatment.size, control.size
    var_t, var_c = treatment.var(ddof=1), control.var(ddof=1)
    mean_t, mean_c = treatment.mean(), control.mean()

    dof = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    relative_se = np.sqrt(
        (var_t / n_t) / mean_c**2 + (mean_t**2 * (var_c / n_c)) / mean_c**4
    )
    margin = stats.t.ppf(0.975, dof) * relative_se
    relative_diff = (mean_t - mean_c) / mean_c

    got = t_test_continuous(continuous_df, metric="revenue", alpha=0.05)
    assert got.relative_ci_low == pytest.approx(relative_diff - margin, rel=1e-12)
    assert got.relative_ci_high == pytest.approx(relative_diff + margin, rel=1e-12)

    # and it is genuinely different from the naive scaling
    assert not np.isclose(got.relative_ci_low, got.ci_low / mean_c)


# --- Guards worth pinning ---------------------------------------------------


def test_zero_variance_arm_raises():
    df = pd.DataFrame(
        {"assignment": [True] * 3 + [False] * 3, "revenue": [5.0] * 3 + [3.0, 4.0, 5.0]}
    )
    with pytest.raises(ValueError, match="variance"):
        t_test_continuous(df, metric="revenue")


def test_zero_control_mean_raises():
    """A control mean of 0 makes the relative effect a division by zero."""
    df = pd.DataFrame(
        {
            "assignment": [True] * 3 + [False] * 4,
            "revenue": [2.0, 3.0, 4.0, -1.0, 1.0, -2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="relative effects undefined"):
        t_test_continuous(df, metric="revenue")


def test_non_boolean_assignment_raises(continuous_df):
    """0/1 ints are not booleans — reject rather than guess the encoding."""
    df = continuous_df.assign(assignment=continuous_df["assignment"].astype(int))
    with pytest.raises(ValueError, match="boolean"):
        t_test_continuous(df, metric="revenue")


def test_nan_in_metric_raises():
    """
    Unlike the binary path, this guard is reachable here: a NaN is a
    perfectly plausible continuous value until you check for it.
    """
    df = pd.DataFrame(
        {
            "assignment": [True] * 3 + [False] * 3,
            "revenue": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        }
    )
    with pytest.raises(ValueError, match="NaN or infinite"):
        t_test_continuous(df, metric="revenue")
