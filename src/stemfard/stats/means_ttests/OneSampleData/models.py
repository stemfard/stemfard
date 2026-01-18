"""
Data models for t-test calculations.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from numpy import sqrt
from numpy.typing import NDArray
from scipy.stats import t

from stemcore import arr_to_numeric
from verifyparams import (
    verify_decimals, verify_int_or_float, verify_sta_alternative,
    verify_sta_conf_level
)

from stats.means_ttests.TwoSamplesStatistics.models import verify_sta_decision


@dataclass(slots=True)
class ParamsOneSampleTData:
    """Container for statistics parameters."""
    data: list[int | float] | NDArray
    popn_mean: int | float
    alternative: str
    conf_level: bool = 0.95
    decision: Literal["test-stat", "p-value", "graph"]
    decimals: int = 4
    
    def __post_init__(self):
        """Validate the params after initialization."""
        self.data = arr_to_numeric(
            data=self.data,
            kind="array",
            param_name="data"
        ).copy()
        
        self.popn_mean = verify_int_or_float(
            value=self.popn_mean,
            param_name="popn_mean: Hypothesized population mean"
        )
        
        self.alternative = verify_sta_alternative(value=self.alternative)
        self.conf_level = verify_sta_conf_level(value=self.conf_level)
        self.decision = verify_sta_decision(value=self.decision)
        self.decimals = verify_decimals(value=self.decimals)
    
    def __repr__(self) -> str:
        return f"ParamsOneSampleTData(...)"


class CalculatedOneSampleTData:
    
    params_data: ParamsOneSampleTData
    
    @property
    def n(self) -> int:
        return len(self.params_data)
    
    @property
    def dfn(self) -> int:
        return self.n - 1
    
    @cached_property
    def mean_data(self) -> float:
        return self.params_data.mean()
    
    @cached_property
    def std_data(self) -> float:
        return self.params_data.std()
    
    @cached_property
    def var_data(self) -> float:
        return self.params_data.var()
    
    @cached_property
    def samplemean_minus_popnmean(self) -> float:
        return self.mean_data - self.params_data.popn_mean
    
    @cached_property
    def standard_error(self) -> float:
        return self.std_data / sqrt(self.n)
    
    @cached_property
    def tstat(self) -> float:
        return self.samplemean_minus_popnmean / self.standard_error
    
    @cached_property
    def pvalue(self) -> float:
        """Calculate the p-value"""
        if self.alternative == "less":
            p = t.cdf(x=self.tstat, df=self.dfn)
        elif self.alternative == "greater":
            p = 1 - t.cdf(x=self.tstat, df=self.dfn)
        elif self.alternative == "two-sided":
            p = 2 * t.cdf(x=-abs(self.tstat), df=self.dfn)
        
        return p
    
    @property
    def sig_level(self) -> float:
        return 1 - self.params_data.conf_level
    
    @cached_property
    def tcrit(self) -> float:
        """
        Calculate critical value of t (i.e. value from t distribution
        table)
        """
        if self.alternative == "less":
            crit_value = abs(t.ppf(q=self.sig_level / 2, df=self.dfn))
        elif self.alternative == "greater":
            crit_value = abs(t.ppf(q=1 - self.sig_level / 2, df=self.dfn))
        elif self.alternative == "two-sided":
            crit_value = abs(t.ppf(q=1 - self.sig_level, df=self.dfn))
                
        if isinstance(crit_value, (tuple, list)):
            crit_value = crit_value[0]
        
    @cached_property
    def confidence_interval(self) -> tuple[float, float]:
        lower, upper = t.interval(
                confidence=self.conf_level,
                loc=self.mean_data,
                df=self.dfn,
                scale=self.standard_error
            )
        return lower, upper