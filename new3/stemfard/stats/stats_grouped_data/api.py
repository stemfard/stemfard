from typing import Literal

from stemfard.core.utils_classes import ResultDict
from stemfard.core._type_aliases import SequenceArrayLike, ScalarSequenceArrayLike
from stemfard.stats.stats_grouped_data.params_factory import create_grouped_params
from stemfard.stats.stats_grouped_data.steps.means import stats_grouped_mean_steps
from stemfard.stats.stats_grouped_data.steps.stdev import stats_grouped_std_steps

def stats_grouped(
    lower: SequenceArrayLike,
    upper: SequenceArrayLike,
    freq: SequenceArrayLike,
    statistic: Literal["mean", "std", "percentiles", "all"] = "mean",
    assumed_mean: int | float | Literal["auto"] | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] = "(x-a)/w",
    var_formula: Literal[1, 2, 3] = 1,
    percentiles: ScalarSequenceArrayLike = 50,
    cumfreq_curve: bool = False,
    x_values: SequenceArrayLike | None = None,
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
    var_formula : {1, 2, 3}, default=1
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
        Contains calculation results, tables, and step-by-step
        explanation.
    """
    if statistic == "mean":
        fparams = create_grouped_params(
            lower=lower,
            upper=upper,
            freq=freq,
            statistic="mean",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            decimals=decimals,
            **kwargs
        )
        return stats_grouped_mean_steps(fparams)

    elif statistic == "std":
        fparams = create_grouped_params(
            lower=lower,
            upper=upper,
            freq=freq,
            statistic="std",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            var_formula=var_formula,
            decimals=decimals,
            **kwargs
        )
        return stats_grouped_std_steps(fparams)

    elif statistic == "percentiles":
        fparams = create_grouped_params(
            lower=lower,
            upper=upper,
            freq=freq,
            statistic="percentiles",
            percentiles=percentiles,
            cumfreq_curve=cumfreq_curve,
            x_values=x_values,
            decimals=decimals,
            **kwargs
        )
        return stats_grouped_percentiles(fparams)

    elif statistic == "all":
        result = {}
        result["mean"] = stats_grouped(
            lower=lower,
            upper=upper,
            freq=freq,
            statistic="mean",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            decimals=decimals,
            **kwargs
        )
        result["std"] = stats_grouped(
            lower=lower,
            upper=upper,
            freq=freq,
            statistic="std",
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            var_formula=var_formula,
            decimals=decimals,
            **kwargs
        )
        result["percentiles"] = stats_grouped(
            lower=lower,
            upper=upper,
            freq=freq,
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


def stats_grouped_mean(
    lower: SequenceArrayLike,
    upper: SequenceArrayLike,
    freq: SequenceArrayLike,
    assumed_mean: int | float | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate the mean of grouped data using the unified engine.
    """
    return stats_grouped(
        lower=lower,
        upper=upper,
        freq=freq,
        statistic="mean",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )


def stats_grouped_std(
    lower: SequenceArrayLike,
    upper: SequenceArrayLike,
    freq: SequenceArrayLike,
    assumed_mean: int | float | None = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    var_formula: int = 1,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate the standard deviation of grouped data using the unified
    engine.
    """
    return stats_grouped(
        lower=lower,
        upper=upper,
        freq=freq,
        statistic="std",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        var_formula=var_formula,
        decimals=decimals
    )


def stats_grouped_percentiles(
    lower: SequenceArrayLike,
    upper: SequenceArrayLike,
    freq: SequenceArrayLike,
    percentiles: list[int | float] | int | float = [25, 50, 75],
    cumfreq_curve: bool = False,
    x_values: SequenceArrayLike | None = None,
    decimals: int = 4,
) -> ResultDict:
    """
    Calculate percentiles of grouped data using the unified engine.
    """
    return stats_grouped(
        lower=lower,
        upper=upper,
        freq=freq,
        statistic="percentiles",
        percentiles=percentiles,
        cumfreq_curve=cumfreq_curve,
        x_values=x_values,
        decimals=decimals
    )