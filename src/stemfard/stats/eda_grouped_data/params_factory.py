from typing import Literal

from core._type_aliases import ArrayLike, ScalarArrayLike
from stats.eda_grouped_data.models import (
    ParamsData,
    ParamsMeanStd,
    ParamsPercentiles,
    ParamsPlot,
    GroupedStructure,
    GroupedTransform,
    GroupedStatistics,
    EDAGroupedDataParams,
)

def create_grouped_params(
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
    freq: ArrayLike,
    statistic: Literal["mean", "std", "percentiles"],
    assumed_mean: int | float | Literal["auto"] = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] = "(x-a)/w",
    formula_std: Literal[1, 2, 3] = 1,
    percentiles: ScalarArrayLike = 50,
    cumfreq_curve: bool = False,
    x_values: ArrayLike | None = None,
    fig_width: int = 6,
    fig_height: int = 4,
    line_color: Literal[
        "blue", "black", "green", "red", "purple", "orange"
    ] = "blue",
    xaxis_orientation: int = 0,
    x_label: str = "Data",
    y_label: str = "Cumulative Frequency",
    grid: bool = True,
    decimals: int = 4,
) -> EDAGroupedDataParams:
    """
    Factory function to create a complete EDAGroupedDataParams object.

    This builds all parameter objects and analysis layers required for
    grouped statistics calculations (mean, std, percentiles) while
    enforcing immutability and input validation.
    """

    # ---------- PARAMETER OBJECTS ----------
    
    data_params = ParamsData(
        lower=lower_limits,
        upper=upper_limits,
        freq=freq,
        statistic=statistic,
        decimals=decimals
    )

    mean_std_params = ParamsMeanStd(
        assumed_mean=assumed_mean,
        formula=assumed_mean_formula,
        formula_std=formula_std
    )

    percentiles_params = ParamsPercentiles(
        percentiles=percentiles,
        cumfreq_curve=cumfreq_curve,
        x_values=x_values
    )

    plot_params = ParamsPlot(
        fig_width=fig_width,
        fig_height=fig_height,
        line_color=line_color,
        xaxis_orientation=xaxis_orientation,
        x_label=x_label,
        y_label=y_label,
        grid=grid
    )

    # ---------- STRUCTURE + TRANSFORM + STATISTICS ----------
    
    structure = GroupedStructure(
        lower=data_params.lower,
        upper=data_params.upper,
        freq=data_params.freq,
        decimals=decimals
    )

    transform = GroupedTransform(
        structure=structure,
        params_mean=mean_std_params
    )

    statistics = GroupedStatistics(
        structure=structure,
        transform=transform
    )

    # ---------- RETURN FACADE OBJECT ----------
    
    return EDAGroupedDataParams(
        data=data_params,
        mean_std=mean_std_params,
        percentiles=percentiles_params,
        plot=plot_params,
        structure=structure,
        transform=transform,
        statistics=statistics
    )