"""
Welch's t-test on raw, unit-level rows for a binary metric.
For now we have assumed 2-groups (A/B) of equal size (50:50),
with a boolean assignment column and (in this function) a 
binary metric. This function should be used when the data
is provided in a unit-level format.
"""

# Import Necessary Packages
import numpy as np
from scipy import stats
from .results import TTestResult

# Define the Binary T-Test Function
def t_test_binary(
    df,
    metric="conversion_rate",
    assignment_col="assignment",
    alpha=0.05
):

    """
    This is a basic Welch's t-test function that takes raw unit-level data 
    with two groups of equal size, and a binary metric of interest. It needs
    the following arguements:

    df: A dataframe with one row per metric-arm, containing at least the
                columns named by metric_col and assignment
    metric: The metric used in the analysis, default here is conversion_rate
    assignment_col: Column of booleans identifying treatment (True) vs
                control (False) rows. Default is "assignment".
    alpha: The significance level (default is 0.05).

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
    required = (assignment_col, metric)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"df is missing required column(s): {missing}")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    if not np.isin(df[metric], (0.0, 1.0)).all():
        raise ValueError(f"{metric!r} must be binary (0/1)")
    if df[assignment_col].dtype != 'bool':
        raise ValueError(
            f"{assignment_col!r} must be a boolean column;" 
            f"got dtype {df[assignment_col].dtype}"
        )
    if not np.isfinite(df[metric]).all():
        raise ValueError(f"{metric!r} has NaN or infinite values in the Sample.")

    # Split the Datasets by Assignment
    treatment_values = df.loc[df[assignment_col], metric].to_numpy(dtype=float)
    control_values = df.loc[~df[assignment_col], metric].to_numpy(dtype=float)
 
    # Error Check for Missing Data
    if treatment_values.size == 0 or control_values.size == 0:
        raise ValueError(
            f"expected units in both arms, got {treatment_values.size}"
            f"treatment and {control_values.size} control"
        )
 
    # Compute the Summary Statistics That the Summary Version Is Handed
    n_treatment = treatment_values.size
    n_control = control_values.size
    mean_treatment = float(treatment_values.mean())
    mean_control = float(control_values.mean())
 
    # Variance Needs at Least 2 Observations per Group
    if n_treatment < 2 or n_control < 2:
        raise ValueError(
            f"each arm needs at least 2 observations to trust the variance "
            f"(got n_treatment={n_treatment}, n_control={n_control})"
        )
 
    # Estimate Variances
    var_treatment = (mean_treatment * (1 - mean_treatment) * n_treatment) / (n_treatment - 1)
    var_control = (mean_control * (1 - mean_control) * n_control) / (n_control - 1)
 
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
 
    # Run the t-test (raw arrays available, so hand it to scipy)
    t_stat, p_value = stats.ttest_ind(
        treatment_values, control_values, equal_var=False
    )
    t_stat = float(t_stat)
    p_value = float(p_value)
 
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
