"""
Data models for t-test calculations.
"""

from dataclasses import dataclass
from typing import Literal

from verifyparams import (
    verify_boolean, verify_decimals, verify_sta_conf_level,
    verify_sta_decision, verify_numeric_arr
)


@dataclass(slots=True)
class ParamsStatistics:
    """Container for statistics parameters."""
    n: tuple[int, int]
    sample_means: tuple[int | float, int | float]
    sample_stds: tuple[int | float, int | float]
    var_equal: bool = False
    welch: bool = False
    conf_level: float = 0.95
    decision: Literal["test-stat", "p-value", "graph"]
    decimals: int = 4
    
    def __post_init__(self):
        """Validate the params after initialization."""
        self.n = verify_numeric_arr(
            value=self.n,
            n=2,
            all_integers=True,
            all_positive=True,
            allow_zero=False,
            param_name="n: Sample sizes"    
        )
        
        self.sample_means = verify_numeric_arr(
            value=self.sample_means,
            n=2,
            all_integers=False,
            all_positive=False,
            allow_zero=True,
            param_name="sample_means: Sample means"    
        )
        
        self.sample_means = verify_numeric_arr(
            value=self.sample_means,
            n=2,
            all_integers=False,
            all_positive=True,
            allow_zero=False,
            param_name="sample_stds: Sample standard deviations"
        )
        
        self.var_equal = verify_boolean(value=self.var_equal, default=False)
        self.welch = verify_boolean(value=self.welch, default=False)
        self.conf_level = verify_sta_conf_level(value=self.conf_level)
        self.decision = verify_sta_decision(value=self.decision)
        
        self.decimals = verify_decimals(value=self.decimals)
    
    def __repr__(self) -> str:
        return f"ParamsStatistics(...)"
    
    @property
    def n1(self) -> int:
        return self.n[0]
    
    @property
    def n2(self) -> int:
        return self.n[1]
    
    @property
    def mean1(self) -> int:
        return self.sample_means[0]
    
    @property
    def mean2(self) -> int:
        return self.sample_means[1]
    
    @property
    def std1(self) -> int:
        return self.sample_stds[0]
    
    @property
    def std2(self) -> int:
        return self.sample_stds[1]