from typing import Literal
import warnings

from numpy import (
    arange, array, asarray, bincount, ceil, char, clip, cumsum, digitize,
    empty, float64, floor, mod, ndarray, min, max
)
from pandas import DataFrame

from stemfard.core.convert import to_numeric
from stemfard.core.results import Result
from stemfard.stats.utils import FrequencyTallyWarning


def sta_freq_tally(
    data: list[int | float] | ndarray,
    class_width: int | float,
    start_from: int | float | None  = None,
    show_values: bool = False,
    decimals: int = 4
) -> DataFrame:
    """
    Compute a statistical frequency tally.

    Parameters
    ----------
    data : array-like of int or float
        Input data. Must be one-dimensional.
    class_width : int or float
        Width of each class interval.
    start_from : int or float, optional
        Starting value for the first class.
    show_values : bool, default False
        Whether to display individual values per class.
    decimals : int, default 4
        Number of decimal places for formatting.

    Returns
    -------
    dframe : pandas.DataFrame
        Frequency tally table.

    Warns
    -----
    FrequencyTallyWarning
        If maximum frequency exceeds 50 and output is simplified.

    Raises
    ------
    ValueError
        If input data is not one-dimensional.
    """
    data_arr = asarray(data, dtype=float64)
    
    if data_arr.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
    
    if show_values and len(data_arr) > 250:
        raise ValueError(
            "'show_values' is only allowed if the number of elements in "
            "'data' is less than 250"
        )
    
    # Vectorized calculations
    min_val = min(data_arr)
    max_val = max(data_arr)
    
    if start_from is None:
        start_val = floor(min_val / class_width) * class_width
    else:
        start_val = start_from
    
    end_val = ceil(max_val / class_width) * class_width
    bins = arange(start_val, end_val + class_width, class_width)
    
    # Vectorized bin assignment
    bin_indices = digitize(data_arr, bins, right=False) - 1
    bin_indices = clip(bin_indices, 0, len(bins) - 2)
    
    # Count frequencies using bincount
    frequencies = bincount(bin_indices, minlength=len(bins) - 1)
    
    # Generate labels
    left_edges = bins[:-1]
    right_edges = bins[1:] - 1
    
    labels = []
    for left, right in zip(left_edges, right_edges):
        if left.is_integer() and right.is_integer():
            labels.append(f"{int(left)} - {int(right)}")
        else:
            # Clean up trailing zeros
            left_str = f"{left:.{decimals}f}".rstrip('0').rstrip('.')
            right_str = f"{right:.{decimals}f}".rstrip('0').rstrip('.')
            labels.append(f"{left_str} - {right_str}")
    
    # Fast tally marks
    def create_tally_fast(count: int) -> str:
        if count == 0:
            return ""
        
        groups = count // 5
        remainder = count % 5
        
        if groups > 0 and remainder > 0:
            return " ///// ." * groups + " " + "/" * remainder
        elif groups > 0:
            return (" ///// ." * groups).strip()
        else:
            return "/" * remainder
    
    tallies = [create_tally_fast(int(freq)) for freq in frequencies]
    
    # Build result
    result_dict = {
        "Class": labels,
        "Tally": tallies,
        "Frequency": frequencies
    }
    
    if show_values and max(frequencies) <= 50:
        values_by_bin = []
        for i in range(len(frequencies)):
            mask = bin_indices == i
            bin_values = data_arr[mask]
            
            if len(bin_values) > 0:
                # Format efficiently
                int_mask = mod(bin_values, 1) == 0
                formatted_vals = empty(len(bin_values), dtype=object)
                
                # Handle integers
                int_vals = bin_values[int_mask]
                if len(int_vals) > 0:
                    formatted_vals[int_mask] = int_vals.astype(int).astype(str)
                
                # Handle floats
                float_mask = ~int_mask
                float_vals = bin_values[float_mask]
                if len(float_vals) > 0:
                    formatted = char.mod(f"%.{decimals}f", float_vals)
                    formatted = char.rstrip(formatted, '0')
                    formatted = char.rstrip(formatted, '.')
                    formatted_vals[float_mask] = formatted
                
                values_by_bin.append(", ".join(formatted_vals))
            else:
                values_by_bin.append("")
        
        result_dict["Values"] = values_by_bin
        
    dframe = DataFrame(result_dict)
    
    if max(frequencies) > 50:
        warnings.warn(
            "Maximum frequency exceeds 50, 'Tally' and 'Values' columns were "
            "omitted for readability.",
            category=FrequencyTallyWarning,
            stacklevel=2
        )
        dframe = dframe[["Class", "Frequency"]]
    
    dframe.index = arange(1, len(dframe) + 1)
    
    return dframe


def sta_eda_grouped(
    lc_limits: list[int | float] | ndarray,
    uc_limits: list[int | float] | ndarray,
    freq: list[int | float] | ndarray,
    statistic: Literal["mean", "sd", "percentiles"] = "mean",
    decimals: int = 4
) -> float:
    
    lc_limits = to_numeric(data=lc_limits, kind=array, dtype=float64)
    uc_limits = to_numeric(data=uc_limits, kind=array, dtype=float64)
    freq = to_numeric(data=freq, kind=array, dtype=float64)
    
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
    
    dframe = DataFrame(data=df_series).round(decimals)
    
    return Result(table=dframe, answer=round(answer, decimals))
    

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