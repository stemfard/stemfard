from dataclasses import dataclass
from typing import Any, Literal

from numpy import float64
from numpy.typing import NDArray
from stemcore import arr_to_numeric
from verifyparams import (
    verify_boolean, verify_decimals, verify_membership,
    verify_membership_iterable, verify_sta_conf_level
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._parse_compute import parse_compute

VALID_STATISTICS = (
    "all", "beta", "se", "t", "p", "ci", "ssr", "sse", "sst", "f", "anova",
    "fit", "rmse", "r2", "adj_r2", "loglk", "aic", "bic", "dw", "omnibus",
    "skew", "kurt", "jb", "cond"
)

@dataclass(frozen=True)
class LinearRegressionParsedParams:
    """Immutable container for all input parameters"""
    x: NDArray[float64]
    y: NDArray[float64]
    slinear_method: str
    slinear_formula: str
    conf_level: float
    statistics: list[str]
    predict_at_x: NDArray[float64] | None
    steps_compute: bool
    steps_detailed: bool
    steps_bg: bool
    decimals: int
    params: CoreParamsResult


def linear_regression_parser(
    x: NDArray[float64],
    y: NDArray[float64],
    slinear_method: Literal["matrix", "normal"] = "normal",
    slinear_formula: Literal["raw", "expanded"] = "expanded",
    conf_level: float = 0.95,
    statistics: list[str] = "beta",
    predict_at_x: NDArray[float64] | None = None,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 12,
) -> LinearRegressionParsedParams:
    
    raw_params: dict[str, Any] = {}
    parsed_params: dict[str, Any] = {}
    
    raw_params = {
        "x": x,
        "y": y,
        "slinear_method": slinear_method,
        "slinear_formula": slinear_formula,
        "conf_level": conf_level,
        "statistics": statistics,
        "predict_at_x": predict_at_x,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals
    }
    
    x: NDArray[float64] = arr_to_numeric(
        data=x,
        all_integers=False,
        all_positive=False,
        arr_1d_to_2d=True,
        param_name="x"
    )
    
    nrows, ncols = x.shape
    
    y: NDArray[float64] = arr_to_numeric(
        data=y,
        all_integers=False,
        all_positive=False,
        arr_1d_to_2d=True,
        param_name="y"
    )
    
    if y.shape[0] > 1 and y.shape[1] > 1:
        raise ValueError(f"Expected 'y' to be 1D, got ({y.shape})")
        
    if nrows != y.shape[0]:
        raise ValueError(
            "Expected number of rows of 'x' and 'y' to be equal, "
            f"got 'x' (shape = {x.shape}) vs 'y' (shape = {y.shape})"
        )
    
    if slinear_method is not None:
        slinear_method: str = verify_membership(
            value=slinear_method,
            valid_items=["matrix", "normal"],
            param_name="slinear_method"
        )
    
    if ncols > 1:
        slinear_method = None
    
    conf_level: float = verify_sta_conf_level(value=conf_level)
    
    statistics = parse_compute(param=statistics, valid=VALID_STATISTICS)
    statistics = verify_membership_iterable(
        value=statistics,
        valid_items=VALID_STATISTICS,
        param_name="statistics"
    )
    
    if predict_at_x is not None:
        predict_at_x: NDArray[float64] = arr_to_numeric(
            data=predict_at_x, arr_1d_to_2d=True, param_name="predict_at_x"
        )
    
    steps_compute: bool = verify_boolean(steps_compute, default=True)
    steps_detailed: bool = verify_boolean(steps_detailed, default=True)
    steps_bg: bool = verify_boolean(steps_bg, default=True)
    
    decimals: int = verify_decimals(decimals, force_decimals=12)
    
    parsed_params = {
        "x": x,
        "y": y,
        "slinear_method": slinear_method,
        "slinear_formula": slinear_formula,
        "conf_level": conf_level,
        "statistics": statistics,
        "predict_at_x": predict_at_x,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals,
    }

    return LinearRegressionParsedParams(
        x=x,
        y=y,
        slinear_method=slinear_method,
        slinear_formula=slinear_formula,
        conf_level=conf_level,
        statistics=statistics,
        predict_at_x=predict_at_x,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )