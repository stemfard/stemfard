from typing import Literal
from stemfard.core.utils_classes import ResultDict
from core._type_aliases import ArrayLike, ScalarArrayLike
from stats.eda_grouped_data.params_factory import create_grouped_params
from stats.eda_grouped_data.steps.means import sta_eda_grouped_mean_steps
from stats.eda_grouped_data.steps.stdev import sta_eda_grouped_std_steps

def sta_eda_grouped(
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    freq: ArrayLike,
    statistic: Literal["mean", "std", "percentiles", "all"] = "mean",
    assumed_mean: int | float | Literal["auto"] | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] = "(x-a)/w",
    formula_std: Literal[1, 2, 3] = 1,
    percentiles: ScalarArrayLike = 50,
    cumfreq_curve: bool = False,
    x_values: ArrayLike | None = None,
    decimals: int = 4,
    **kwargs,
) -> ResultDict:
    """
    Compute grouped data statistics.

    This function dispatches to the appropriate specialized function
    based on the `statistic` parameter. It uses the unified factory 
    `create_grouped_params` to build all necessary objects.

    Parameters
    ----------
    statistic : {"mean", "std", "percentiles", "all"}, default="mean"
        Type of statistic to compute.
    assumed_mean : int, float, "auto", or None
        Assumed mean for mean/std calculation.
    assumed_mean_formula : {"x-a", "x/w-a", "(x-a)/w"}, default="(x-a)/w"
        Formula for assumed mean calculations.
    formula_std : {1, 2, 3}, default=1
        Formula version for standard deviation.
    percentiles : int, float, list, or array, default=50
        Percentile values to compute.
    cumfreq_curve : bool, default=False
        Whether cumulative frequency curve is requested.
    x_values : list, array, or None, default=None
        Optional x-values for percentile calculations.
    decimals : int, default=4
        Rounding precision.
    **kwargs
        Additional parameters for plotting etc.

    Returns
    -------
    ResultDict
        Contains calculation results, tables, and step-by-step explanation.
    """
    if statistic == "mean":
        fparams = create_grouped_params(
            lower_limits, upper_limits, freq,
            statistic="mean",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            decimals=decimals,
            **kwargs
        )
        return sta_eda_grouped_mean_steps(fparams)

    elif statistic == "std":
        fparams = create_grouped_params(
            lower_limits, upper_limits, freq,
            statistic="std",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            formula_std=formula_std,
            decimals=decimals,
            **kwargs
        )
        return sta_eda_grouped_std_steps(fparams)

    elif statistic == "percentiles":
        fparams = create_grouped_params(
            lower_limits, upper_limits, freq,
            statistic="percentiles",
            percentiles=percentiles,
            cumfreq_curve=cumfreq_curve,
            x_values=x_values,
            decimals=decimals,
            **kwargs
        )
        return sta_eda_grouped_percentiles(fparams)

    elif statistic == "all":
        result = {}
        result["mean"] = sta_eda_grouped(
            lower_limits, upper_limits, freq,
            statistic="mean",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            decimals=decimals,
            **kwargs
        )
        result["std"] = sta_eda_grouped(
            lower_limits, upper_limits, freq,
            statistic="std",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            formula_std=formula_std,
            decimals=decimals,
            **kwargs
        )
        result["percentiles"] = sta_eda_grouped(
            lower_limits, upper_limits, freq,
            statistic="percentiles",
            percentiles=percentiles,
            cumfreq_curve=cumfreq_curve,
            x_values=x_values,
            decimals=decimals,
            **kwargs
        )
        return result

    else:
        raise ValueError(f"Unknown statistic: {statistic}")


def sta_eda_grouped_mean(
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    freq: ArrayLike,
    assumed_mean: int | float | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate the mean of grouped data using the unified engine.
    """
    return sta_eda_grouped(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        statistic="mean",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )


def sta_eda_grouped_std(
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    freq: ArrayLike,
    assumed_mean: int | float | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    formula_std: int = 1,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate the standard deviation of grouped data using the unified engine.
    """
    return sta_eda_grouped(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        statistic="std",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        formula_std=formula_std,
        decimals=decimals
    )


def sta_eda_grouped_percentiles(
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    freq: ArrayLike,
    percentiles: list[int | float] | int | float = 50,
    cumfreq_curve: bool = False,
    x_values: ArrayLike | None = None,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate percentiles of grouped data using the unified engine.
    """
    return sta_eda_grouped(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        statistic="percentiles",
        percentiles=percentiles,
        cumfreq_curve=cumfreq_curve,
        x_values=x_values,
        decimals=decimals
    )