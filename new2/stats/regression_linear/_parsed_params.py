from dataclasses import dataclass
from typing import Any, Literal
import warnings

from numpy import float64
from numpy.typing import NDArray
from stemcore import arr_to_numeric
from verifyparams import (
    verify_boolean, verify_decimals, verify_membership, verify_sta_conf_level
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.warning_msgs import ActionTaken


@dataclass(frozen=True)
class LinearRegressionParsedParams:
    """Immutable container for all input parameters"""
    params: CoreParamsResult
    x: SequenceArrayLike
    y: SequenceArrayLike
    simple_linear_method: Literal["matrix", "normal"] | None
    conf_level: float
    coefficients: bool
    standard_errors: bool
    tstats: bool
    pvalues: bool
    confidence_intervals: bool
    anova: bool
    fitted_and_residuals: bool
    others: bool
    predict_at_x: NDArray[float64]
    steps_compute: bool
    steps_detailed: bool
    show_bg: bool
    decimals: int

def parse_linear_regression(
    x: SequenceArrayLike,
    y: SequenceArrayLike,
    simple_linear_method: Literal["matrix", "normal"] | None = "normal",
    conf_level: float = 0.95,
    coefficients: bool = True,
    standard_errors: bool = True,
    tstats: bool = True,
    pvalues: bool = True,
    confidence_intervals: bool = True,
    anova: bool = True,
    fitted_and_residuals: bool = False,
    others: bool = False,
    predict_at_x: NDArray[float64] | None = None,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 12,
    params: CoreParamsResult = ...
) -> LinearRegressionParsedParams:
    
    raw_params: dict[str, Any] = {}
    parsed_params: dict[str, Any] = {}
    
    # x: SequenceArrayLike,
    # y: SequenceArrayLike,
    
    x: NDArray[float64] = arr_to_numeric(
        data=x,
        all_integers=False,
        all_positive=False,
        param_name="x"
    )
    
    y: NDArray[float64] = arr_to_numeric(
        data=y,
        all_integers=False,
        all_positive=False,
        param_name="y"
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
    
    simple_linear_method: str = verify_membership(
        value=simple_linear_method,
        valid_items=["matrix", "normal"],
        param_name="simple_linear_method"
    )
    
    if x.ndim == 2 and x.shape[1] > 1:
        simple_linear_method = None
        
    
    # conf_level: float = 0.95
    
    conf_level: float = verify_sta_conf_level(value=conf_level)
    
    # coefficients: bool = True
    # standard_errors: bool = True
    # tstats: bool = True
    # pvalues: bool = True
    # confidence_intervals: bool = True
    # anova: bool = True
    # fitted_and_residuals: bool = False
    # others: bool = False
    
    coefficients: bool = verify_boolean(coefficients, default=True)
    standard_errors: bool = verify_boolean(standard_errors, default=True)
    tstats: bool = verify_boolean(tstats, default=True)
    pvalues: bool = verify_boolean(pvalues, default=True)
    confidence_intervals: bool = verify_boolean(
        confidence_intervals, default=True
    )
    anova: bool = verify_boolean(anova, default=True)
    fitted_and_residuals: bool = verify_boolean(
        fitted_and_residuals, default=False
    )
    others: bool = verify_boolean(others, default=False)
    
    # predict_at_x: NDArray[float64]
    
    if predict_at_x:
        predict_at_x: NDArray[float64] = arr_to_numeric(
            data=predict_at_x, param_name="predict_at_x"
        )
    
    # steps_compute: bool = True
    # steps_detailed: bool = True
    # show_bg: bool = True
    
    steps_compute: bool = verify_boolean(steps_compute, default=True)
    steps_detailed: bool = verify_boolean(steps_detailed, default=True)
    show_bg: bool = verify_boolean(show_bg, default=True)
    
    # decimals: int = 12
    
    decimals: int = verify_decimals(decimals, force_decimals=12)
       
    return LinearRegressionParsedParams(
        x=x,
        y=y,
        simple_linear_method=simple_linear_method,
        conf_level=conf_level,
        coefficients=coefficients,
        standard_errors=standard_errors,
        tstats=tstats,
        pvalues=pvalues,
        confidence_intervals=confidence_intervals,
        anova=anova,
        fitted_and_residuals=fitted_and_residuals,
        others=others,
        predict_at_x=predict_at_x,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        show_bg=show_bg,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )