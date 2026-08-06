"""
Welch's t-test on pre-aggregated, arm-level summary statistics.
For now we have assumed 2-groups (A/B), with a boolean
assignment column and one row per metric-arm. This function
should be used when the data has already been aggregated,
rather than provided in a unit-level format.
"""

# Import Necessary Packages
import numpy as np
from scipy import stats
from .results import TTestResult

# Define the Summary-Stats T-Test Function
def t_test_summary(
    df,
    metric="conversions",          
    metric_col="metric",
    assignment_col="assignment",
    alpha=0.05,
    n_col="sample",
    mean_col="mean",
    var_col="var",
):
    """
    This is a basic Welch's t-test function that takes pre-aggregated
    summary statistics (one row per metric-arm) rather than raw
    unit-level rows. It needs the following arguements:

    df: A dataframe with one row per metric-arm, containing at least the
                columns named by metric_col, assignment, n_col, mean_col, var_col.
    metric_col: The column where the metric name is stored, default is metric.
    metric: The exact metric used in the analysis, default is conversions
    assignment_col: Column of booleans identifying treatment (True) vs
                control (False) rows. Default is "assignment".
    alpha: The significance level (default is 0.05).
    n_col: Column name holding sample size per arm. Default "sample".
    mean_col: Column name holding the mean per arm. Default "mean".
    var_col: Column name holding the sample variance (ddof=1) per arm.
                Default "var".

    Returns a TTestResult object (see results.TTestResult for field
    definitions).

    Raises:
        KeyError: if any required column is missing.
        ValueError: if alpha is not strictly between 0 and 1; if the metric
                is not found, or found more than once, in either arm; if
                either arm's n is fewer than 2; if either arm's variance is
                zero or negative; or if mean_control is 0.
    """

    # Include Critical Error Checks
    if df.empty:
        raise ValueError("Dataframe is empty — no data to test")
    required = (metric_col, assignment_col, n_col, mean_col, var_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"df is missing required column(s): {missing}")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    if df[assignment_col].dtype != 'bool':
        raise ValueError(
            f"{assignment_col!r} must be a boolean column;" 
            f"got dtype {df[assignment_col].dtype}"
        )

    # Isolate the Treatment and Control Rows for This Metric
    df_metric = df.loc[df[metric_col] == metric]
    if df_metric.empty:
        raise ValueError(f"metric {metric!r} not found in {metric_col!r} column")

    treatment_rows = df_metric.loc[df_metric[assignment_col]]
    control_rows = df_metric.loc[~df_metric[assignment_col]]

    if len(treatment_rows) != 1 or len(control_rows) != 1:
        raise ValueError(
            f"expected exactly one treatment row and one control row for "
            f"metric={metric!r}, got {len(treatment_rows)} treatment "
            f"row(s) and {len(control_rows)} control row(s)"
        )

    # Pull Out the Scalars
    n_treatment = int(treatment_rows[n_col].iloc[0])
    n_control = int(control_rows[n_col].iloc[0])
    mean_treatment = float(treatment_rows[mean_col].iloc[0])
    mean_control = float(control_rows[mean_col].iloc[0])
    var_treatment = float(treatment_rows[var_col].iloc[0])
    var_control = float(control_rows[var_col].iloc[0])

    # var(ddof=1) Needs at Least 2 Observations per Group
    if n_treatment < 2 or n_control < 2:
        raise ValueError(
            f"each arm needs at least 2 observations to trust the variance "
            f"(got n_treatment={n_treatment}, n_control={n_control})"
        )

    # Error Check for Breaking Denominator / Degenerate Variance
    if mean_control == 0:
        raise ValueError(
            f"mean_control is 0 for {metric!r}; relative effects undefined"
        )
    if var_treatment <= 0:
        raise ValueError(f"{metric!r} has zero/negative variance in Treatment arm")
    if var_control <= 0:
        raise ValueError(f"{metric!r} has zero/negative variance in Control arm")

    # Calculate the Standard Error of the Treatment and Control Groups
    se = np.sqrt(var_treatment / n_treatment + var_control / n_control)

    # Calculate the Absolute and Relative Difference
    absolute_diff = mean_treatment - mean_control
    relative_diff = absolute_diff / mean_control

    # Calculate the Degrees of Freedom
    dof = (var_treatment / n_treatment + var_control / n_control) ** 2 / (
        (var_treatment / n_treatment) ** 2 / (n_treatment - 1)
        + (var_control / n_control) ** 2 / (n_control - 1)
    )

    # Run the t-test Manually (no raw arrays available for stats.ttest_ind)
    t_stat = absolute_diff / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)

    # Calculate the Margin of Error
    margin = stats.t.ppf(1 - alpha / 2, dof) * se

    # Delta-Method Standard Error for the Relative Difference
    var_mean_treatment = var_treatment / n_treatment
    var_mean_control = var_control / n_control
    relative_se = np.sqrt(
        var_mean_treatment / mean_control ** 2
        + (mean_treatment ** 2 * var_mean_control) / mean_control ** 4
    )
    relative_margin = stats.t.ppf(1 - alpha / 2, dof) * relative_se

    # Return the Results
    return TTestResult(
        mean_control=mean_control,
        mean_treatment=mean_treatment,
        absolute_diff=absolute_diff,
        relative_diff=relative_diff,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=absolute_diff - margin,
        ci_high=absolute_diff + margin,
        relative_ci_low=relative_diff - relative_margin,
        relative_ci_high=relative_diff + relative_margin,
    )
