"""
Data models for t-test calculations.
"""

from dataclasses import dataclass
from typing import Literal

from verifyparams import (
    verify_decimals, verify_int_or_float, verify_numeric, verify_sta_conf_level
)

from stemfard.stats.means_ttests.TwoSamplesStatistics.models import verify_sta_decision


@dataclass(slots=True)
class ParamsStatistics:
    """Container for statistics parameters."""
    n: int
    sample_mean: int | float
    sample_std: int | float
    hypo_popn_mean: int | float
    conf_level: bool = 0.95
    decision: Literal["test-stat", "p-value", "graph"]
    decimals: int = 4
    
    def __post_init__(self):
        """Validate the params after initialization."""
        self.n = verify_numeric(
            value=self.n,
            is_integer=True,
            is_positive=True,
            allow_zero=False,
            param_name="n: Sample size"
        )
        
        self.sample_mean = verify_int_or_float(
            value=self.sample_mean,
            param_name="sample_mean: Sample mean"
        )
        
        self.sample_std = verify_numeric(
            value=self.sample_std,
            is_integer=False,
            is_positive=True,
            allow_zero=False,
            param_name="sample_std: Sample standard deviation"
        )
        
        self.hypo_popn_mean = verify_int_or_float(
            value=self.hypo_popn_mean,
            param_name="hypo_popn_mean: Hypothesized population mean"
        )
        
        self.conf_level = verify_sta_conf_level(value=self.conf_level)
        self.decision = verify_sta_decision(value=self.decision)
        self.decimals = verify_decimals(value=self.decimals)
    
    def __repr__(self) -> str:
        return f"ParamsStatistics(n={self.n})"