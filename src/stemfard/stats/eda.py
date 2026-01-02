from typing import Literal

from numpy import array, cumsum, int64, ndarray
from pandas import DataFrame


def sta_eda_grouped(
    lc_limits: list[int | float] | ndarray,
    uc_limits: list[int | float] | ndarray,
    freq: list[int | float] | ndarray,
    statistic: Literal["mean", "sd", "percentiles"],
    decimals: int = 4
) -> float:
    
    if not isinstance(lc_limits, ndarray):
        lc_limits = array(lc_limits)
    
    if not isinstance(uc_limits, ndarray):
        uc_limits = array(uc_limits)
        
    if not isinstance(freq, ndarray):
        freq = array(freq, dtype=int64)
    
    n = len(lc_limits)
    class_limits = [
        f"{lc_limits[index]} - {uc_limits[index]}" for index in range(n)
    ]
    
    df_series = {
        "Class": class_limits,
        "Frequency": freq
    }
    
    if statistic == "percentiles":
        df_series.update(
            {"Cumulative frequency": cumsum(freq)}
        )
    
    dframe = DataFrame(data=df_series)
    
    return dframe


def sta_eda_ungrouped(
    data: list[int | float] | ndarray,
    assumed_mean: int | float,
    statistic: Literal["min", "max", "range"],
) -> float:
    
    return data


def sta_correlation(
    x: list[int | float] | ndarray,
    y: list[int | float] | ndarray,
    method: Literal["pearson", "spearman", "kendal"]
) -> float:
    
    return x