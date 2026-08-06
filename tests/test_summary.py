"""
t_test_summary: the summary path must agree with the raw path, since the two
are independent implementations of the same test.
"""

import dataclasses

import pandas as pd
import pytest

from ab_testing.frequentist import t_test_continuous, t_test_summary

# --- Cross-consistency ------------------------------------------------------


def test_reproduces_the_raw_path(summary_df, continuous_df):
    """
    Handed the summary statistics of the same data, every field must match
    what t_test_continuous computed from the raw rows. This is the strongest
    invariant in the suite: it catches the two paths drifting apart when only
    one of them gets edited.
    """
    raw = t_test_continuous(continuous_df, metric="revenue")
    aggregated = t_test_summary(summary_df, metric="revenue")

    for field in dataclasses.fields(raw):
        assert getattr(aggregated, field.name) == pytest.approx(
            getattr(raw, field.name), rel=1e-12
        ), field.name


def test_alpha_is_respected_the_same_way(summary_df, continuous_df):
    raw = t_test_continuous(continuous_df, metric="revenue", alpha=0.01)
    aggregated = t_test_summary(summary_df, metric="revenue", alpha=0.01)

    assert aggregated.ci_low == pytest.approx(raw.ci_low, rel=1e-12)
    assert aggregated.ci_high == pytest.approx(raw.ci_high, rel=1e-12)


# --- Guards worth pinning ---------------------------------------------------


def test_duplicate_metric_arm_rows_raise(summary_df):
    """
    Two treatment rows for one metric is ambiguous — silently taking the
    first would quietly analyse half the data.
    """
    duplicated = pd.concat([summary_df, summary_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one treatment row"):
        t_test_summary(duplicated, metric="revenue")


def test_unknown_metric_raises(summary_df):
    with pytest.raises(ValueError, match="not found"):
        t_test_summary(summary_df, metric="not_a_real_metric")


def test_missing_column_raises(summary_df):
    with pytest.raises(KeyError):
        t_test_summary(summary_df.drop(columns=["var"]), metric="revenue")
