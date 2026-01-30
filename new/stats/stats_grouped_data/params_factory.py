from typing import Literal

from stemfard.core._type_aliases import SequenceArrayLike, ScalarSequenceArrayLike
from stemfard.stats.stats_grouped_data.models import (
    ParamsData,
    ParamsMeanStd,
    ParamsPercentiles,
    ParamsPlot,
    EDAGroupedDataFacade,
)

def create_grouped_params(
    lower: SequenceArrayLike,
    upper: SequenceArrayLike,
    freq: SequenceArrayLike,
    statistic: Literal["mean", "std", "percentiles"],
    assumed_mean: int | float | Literal["auto"] = None,
    assumed_mean_formula: Literal["x-a", "x/w-a", "(x-a)/w"] = "(x-a)/w",
    var_formula: Literal[1, 2, 3] = 1,
    percentiles: ScalarSequenceArrayLike = 50,
    cumfreq_curve: bool = False,
    x_values: SequenceArrayLike | None = None,
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
) -> EDAGroupedDataFacade:
    """
    Factory function to create a complete EDAGroupedDataFacade object.
    The structure, transform, and statistics layers are built automatically.
    """

    # ---------- PARAMETER OBJECTS ----------
    
    data_params = ParamsData(
        lower=lower,
        upper=upper,
        freq=freq,
        statistic=statistic,
        decimals=decimals
    )

    mean_std_params = ParamsMeanStd(
        assumed_mean=assumed_mean,
        formula=assumed_mean_formula,
        var_formula=var_formula
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

    # ---------- RETURN FACADE OBJECT USING FACTORY METHOD ----------
    
    return EDAGroupedDataFacade.create(
        data=data_params,
        mean_std=mean_std_params,
        percentiles=percentiles_params,
        plot=plot_params
    )