from typing import Literal, Union
from numpy import float64
from numpy.typing import NDArray
from core.utils_classes import ResultDict
from stats.eda_grouped_data.models import create_grouped_params
from stats.eda_grouped_data.steps.means import sta_eda_grouped_std_steps

# Type aliases for clarity
GroupedDataArray = Union[list[Union[int, float]], NDArray[float64]]
PercentilesArray = Union[int, float, list[Union[int, float]], NDArray[float64]]


def sta_eda_grouped(
    lower_limits: GroupedDataArray,
    upper_limits: GroupedDataArray,
    freq: GroupedDataArray,
    statistic: Literal["mean", "std", "percentiles", "all"] = "mean",
    assumed_mean: Union[int, float, Literal["auto"], None] = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] = "(x-a)/w",
    formula_std: Literal[1, 2, 3] = 1,
    percentiles: PercentilesArray = 50,
    cumfreq_curve: bool = False,
    x_values: GroupedDataArray | None = None,
    decimals: int = 4,
    **kwargs
) -> Union[ResultDict, dict]:
    """
    Compute grouped data statistics: mean, std, percentiles, or all.

    Parameters
    ----------
    lower_limits : list | ndarray
        Lower class boundaries.
    upper_limits : list | ndarray
        Upper class boundaries.
    freq : list | ndarray
        Frequencies for each class.
    statistic : str
        "mean", "std", "percentiles", or "all".
    assumed_mean : int | float | "auto" | None
        Optional assumed mean.
    assumed_mean_formula : str
        Formula for assumed mean calculation.
    formula_std : int
        Formula type for standard deviation (1, 2, 3).
    percentiles : int, float, list, or ndarray
        Percentiles to calculate.
    cumfreq_curve : bool
        Whether to include cumulative frequency curve.
    x_values : list | ndarray | None
        Optional x-values for percentile calculations.
    decimals : int
        Decimal precision for results.
    **kwargs : dict
        Extra parameters passed to plotting or internal models.

    Returns
    -------
    ResultDict or dict
        - If statistic="all", returns a dict with keys "mean", "std", "percentiles".
        - Otherwise, returns a ResultDict for the requested statistic.
    """

    if statistic in {"mean", "std", "percentiles"}:
        # Build full parameter object
        fparams = create_grouped_params(
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            freq=freq,
            statistic=statistic,
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula,
            formula_std=formula_std,
            percentiles=percentiles,
            cumfreq_curve=cumfreq_curve,
            x_values=x_values,
            decimals=decimals,
            **kwargs
        )

        # Dispatch to the correct step engine
        if statistic == "mean":
            return sta_eda_grouped_std_steps(fparams)
        elif statistic == "std":
            return sta_eda_grouped_std_steps(fparams)
        elif statistic == "percentiles":
            return sta_eda_grouped_std_steps(fparams)

    elif statistic == "all":
        # Compute all three statistics and merge results
        return {
            "mean": sta_eda_grouped(
                lower_limits, upper_limits, freq,
                statistic="mean",
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                decimals=decimals,
                **kwargs
            ),
            "std": sta_eda_grouped(
                lower_limits, upper_limits, freq,
                statistic="std",
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                formula_std=formula_std,
                decimals=decimals,
                **kwargs
            ),
            "percentiles": sta_eda_grouped(
                lower_limits, upper_limits, freq,
                statistic="percentiles",
                percentiles=percentiles,
                cumfreq_curve=cumfreq_curve,
                x_values=x_values,
                decimals=decimals,
                **kwargs
            )
        }

    else:
        raise ValueError(f"Unknown statistic: {statistic}. Must be 'mean', 'std', 'percentiles', or 'all'.")