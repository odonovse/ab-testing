"""
This defines the results type that every frequentist test function will
use. This module imports nothing from its siblings, so it can never take
part in a circular import, but everything else depends on it.
"""

# Import Necessary Packages
from dataclasses import dataclass

# Define the Results Class
@dataclass(frozen=True)
class TTestResult:
    mean_control: float
    mean_treatment: float
    absolute_diff: float
    relative_diff: float
    se: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    relative_ci_low: float
    relative_ci_high: float
