from typing import Literal

from numpy import array, cumsum, float64, int64, ndarray
from pandas import DataFrame

from stemfard.core.results import Result


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
        
    midpoint = array(
        object=[(lc_limits[k] + uc_limits[k]) / 2 for k in range(n)],
        dtype=float64
    )
    
    answer = sum(midpoint * freq) / sum(freq)
    
    dframe = DataFrame(data=df_series)
    
    return Result(table=dframe, answer=answer)
    

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