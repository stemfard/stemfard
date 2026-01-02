from typing import Literal

from numpy import ndarray


def sta_eda_grouped(
    lc_limits: list[int | float] | ndarray,
    uc_limits: list[int | float] | ndarray,
    freq: list[int | float] | ndarray,
    statistic: Literal["mean", "sd", "percentiles"]
) -> float:
    
    return freq


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