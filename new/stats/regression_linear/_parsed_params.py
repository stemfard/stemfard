from dataclasses import dataclass
from typing import Literal
import warnings

from stemcore import arr_to_numeric
from verifyparams import (
    verify_boolean, verify_decimals, verify_membership, verify_sta_conf_level
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.warning_msgs import ActionTaken


@dataclass(slots=True, frozen=True)
class ParseLinearRegression:
    params: CoreParamsResult
    x: SequenceArrayLike
    y: SequenceArrayLike
    simple_linear_method: Literal["matrix", "normal"] | None = "normal"
    conf_level: float = 0.95
    coefficients: bool = True
    stderrors: bool = True
    tstats: bool = True
    pvalues: bool = True
    conf_intervals: bool = True
    anova: bool = True
    fitted_and_residuals: bool = False
    others: bool = False
    steps_compute: bool = True
    steps_detailed: bool = True
    show_bg: bool = True
    decimals: int = 12
    
    def __repr__(self) -> str:
        return (f"ParseLinalgIterative(A={self.A.shape}, b={len(self.b)})")
    

def parse_linear_regression(
    x: SequenceArrayLike,
    y: SequenceArrayLike,
    simple_linear_method: Literal["matrix", "normal"] | None = "normal",
    conf_level: float = 0.95,
    coefficients: bool = True,
    stderrors: bool = True,
    tstats: bool = True,
    pvalues: bool = True,
    conf_intervals: bool = True,
    anova: bool = True,
    fitted_and_residuals: bool = False,
    others: bool = False,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 12
) -> ParseLinearRegression:
    
    raw_params = {}
    parsed_params = {}
    
    # x: SequenceArrayLike,
    # y: SequenceArrayLike,
    
    x: SequenceArrayLike = arr_to_numeric(
        data=x,
        param_name="x",
        all_integers=False,
        all_positive=False,
        allow_zero=True
    )
    
    y: SequenceArrayLike = arr_to_numeric(
        data=y,
        param_name="y",
        all_integers=False,
        all_positive=False,
        allow_zero=True
    )
    
    if y.ndim != 1:
        y = y.flatten()
        
    y = y.reshape(-1, 1)
        
    if len(x) != len(y):
        if x.ndim == 2:
            x = x.T # attempt to transpose
            
        # check after transposing
        if len(x) != len(y):
            raise ValueError(
                "Expected length of 'y' to match the length of 'x', "
                f"got 'y' (shape = {y.shape}) vs 'x (shape = {x.shape})"
            )
        else:
            warnings.warn(
                "'x' was transposed to match the length of 'y'",
                category=ActionTaken,
                stacklevel=2
            )
            
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # simple_linear_method: Literal["matrix", "normal"] = "normal"
    
    simple_linear_method = verify_membership(
        value=simple_linear_method,
        valid_items=["matrix", "normal"],
        param_name="simple_linear_method"
    )
    
    if x.ndim == 2 and x.shape[1] > 1:
        simple_linear_method = None
        
    
    # conf_level: float = 0.95
    
    conf_level = verify_sta_conf_level(value=conf_level)
    
    # coefficients: bool = True
    # stderrors: bool = True
    # tstats: bool = True
    # pvalues: bool = True
    # conf_intervals: bool = True
    # anova: bool = True
    # fitted_and_residuals: bool = False
    # others: bool = False
    
    # steps_compute: bool = True
    # steps_detailed: bool = True
    # show_bg: bool = True
    
    coefficients = verify_boolean(coefficients, default=True)
    stderrors = verify_boolean(stderrors, default=True)
    tstats = verify_boolean(tstats, default=True)
    pvalues = verify_boolean(pvalues, default=True)
    conf_intervals = verify_boolean(conf_intervals, default=True)
    anova = verify_boolean(anova, default=True)
    fitted_and_residuals = verify_boolean(fitted_and_residuals, default=False)
    others = verify_boolean(others, default=False)
    
    steps_compute = verify_boolean(steps_compute, default=True)
    steps_detailed = verify_boolean(steps_detailed, default=True)
    show_bg = verify_boolean(show_bg, default=True)
    
    # decimals: int = 12
    
    decimals = verify_decimals(decimals, force_decimals=12)
       
    return ParseLinearRegression(
        x=x,
        y=y,
        simple_linear_method=simple_linear_method,
        conf_level=conf_level,
        coefficients=coefficients,
        stderrors=stderrors,
        tstats=tstats,
        pvalues=pvalues,
        conf_intervals=conf_intervals,
        anova=anova,
        fitted_and_residuals=fitted_and_residuals,
        others=others,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        show_bg=show_bg,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )