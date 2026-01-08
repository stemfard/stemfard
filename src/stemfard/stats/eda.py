from enum import Enum
import warnings

from numpy import (
    arange, asarray, bincount, ceil, char, clip, cumsum, digitize, empty,
    float64, floor, mod, nan, ndarray
)
from numpy.typing import NDArray
from pandas import DataFrame, concat
from stemcore import numeric_format, str_data_join_contd
from verifyparams import (
    verify_all_integers, verify_int_or_float, verify_len_equal,
    verify_membership
)

from stemfard.core.convert import to_numeric
from stemfard.core.utils_classes import FrequencyTallyWarning, ResultDict


class StatisticType(Enum):
    MEAN = "mean"
    STANDARD_DEVIATION = "sd"
    PERCENTILES = "percentiles"


class AssumedMeanFormulaType(Enum):
    X_MINUS_A = "x-a"
    X_OVER_W_MINUS_A = "x/w-a"
    X_MINUS_A_OVER_W = "(x-a)/w"


class GroupedStatisticsCalculator:

    def __init__(self):
        """
        Initialize the calculator.
        """
        
    
    def compute(
        self,
        lower_limits: list[int | float] | NDArray[float64],
        upper_limits: list[int | float] | NDArray[float64],
        freq: list[int | float] | NDArray[float64],
        class_width: int | float,
        statistic: StatisticType = StatisticType.MEAN,
        decimals: int = 4,
        **kwargs
    ) -> ResultDict:
        """
        Calculate grouped statistics with comprehensive error handling.
        
        Args:
            lower_limits: Lower class limits/boundaries
            upper_limits: Upper class limits/boundaries
            frequencies: Class frequencies
            statistic: Type of statistic to calculate
            decimals: Number of decimal places for rounding
            **kwargs: Additional parameters specific to each statistic
            
        Returns:
            ResultDict containing answer, table, and metadata
        """
        lower_limits, upper_limits, freq, class_width, statistic = self._validate_common_params(
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            freq=freq,
            class_width=class_width,
            statistic=statistic
        )
        
        class_labels = self._generate_class_labels(
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            decimals=decimals
        )
        
        # * 0.5 for better numerical accurace than /2 (i.e. division by 2)
        midpoints = lower_limits * 0.5 + upper_limits * 0.5
        
        params_dct = {
            "class_labels": class_labels,
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "midpoints": midpoints,
            "freq": freq,
            "decimals": decimals
        }
        
        if statistic in ["mean", "sd"]:
            params_dct.update(
                {
                "assumed_mean": kwargs.get("assumed_mean"),
                "assumed_mean_formula": (
                    kwargs.get("assumed_mean_formula", "(x-a)/w")
                )
            }
            )
        else:
            params_dct.update(
                {
                    "percentiles": kwargs.get("percentiles"),
                }
            )
            cumfreq_curve = kwargs.get("cumfreq_curve")
            
            if cumfreq_curve:
                params_dct = {
                    "x_values": kwargs.get("x_values"),
                    "fig_width": kwargs.get("fig_width"),
                    "fig_height": kwargs.get("fig_height"),
                    "line_color": kwargs.get("line_color"),
                    "xaxis_orientation": kwargs.get("xaxis_orientation"),
                    "x_title": kwargs.get("x_title")
                }
        
        # Perform calculations based on statistic type
        
        if statistic == "mean":
            return self._calculate_mean(**params_dct)
        elif statistic == "sd":
            return self._calculate_standard_deviation(**params_dct)
        elif statistic == "percentiles":
            return self._calculate_percentiles(**params_dct)
    
    
    def _validate_common_params(
        self,
        lower_limits: list[int | float] | NDArray[float64],
        upper_limits: list[int | float] | NDArray[float64],
        freq: list[int | float] | NDArray[float64],
        class_width: int | float,
        statistic: StatisticType
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Validate and convert input data to numpy arrays."""
        lower_limits = to_numeric(data=lower_limits, param_name="lower_limits")
        upper_limits = to_numeric(data=upper_limits, param_name="upper_limits")
        freq = to_numeric(freq, dtype=None)
        verify_all_integers(value=freq, param_name="freq")
        verify_len_equal(
            lower_limits,
            upper_limits,
            freq,
            param_names=["lower_limits", "upper_limits", "freq"]
        )
        verify_int_or_float(value=class_width, param_name="class_width")
        statistic = statistic.value
        verify_membership(
            user_input=statistic,
            valid_items=["mean", "variance", "percentiles"],
            param_name="statistic"
        )
        
        return lower_limits, upper_limits, freq, class_width, statistic
    
    
    def _generate_class_labels(
        self, 
        lower_limits: NDArray[float64], 
        upper_limits: NDArray[float64],
        decimals: int = 4
    ) -> list[str]:
        """Generate formatted class labels."""
        return [
            f"{round(lower, decimals)} - {round(upper, decimals)}"
            for lower, upper in zip(lower_limits, upper_limits)
        ]
    
    
    def _validate_assumed_mean(
        self, 
        assumed_mean: int | float, 
        assumed_mean_formula: AssumedMeanFormulaType = None
    ) -> None:
        """Validate assumed mean parameters."""
        verify_int_or_float(value=assumed_mean, param_name="assumed_mean")
        
        verify_membership(
            user_input=assumed_mean_formula,
            valid_items=["x-a", "x/w-a", "(x-a)/w"],
            param_name="assumed_mean_formula"
        )
    
    
    def _apply_assumed_mean_formula(
        self,
        midpoints: NDArray[float64],
        class_width: int | float,
        assumed_mean: int | float,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int = 4
    ) -> NDArray[float64]:
        """Apply assumed mean formula to midpoints."""
        if assumed_mean_formula == "x-a":
            t_values = midpoints - assumed_mean
            t_name = f"t = x - {round(assumed_mean, decimals)}"
            return t_values, t_name
        elif assumed_mean_formula == "x/w-a":
            t_values = midpoints / class_width - assumed_mean
            t_name = f"t = x / {round(class_width, decimals)} - {round(assumed_mean, decimals)}"
            return t_values, t_name
        elif assumed_mean_formula == "(x-a)/w":
            t_values = (midpoints - assumed_mean) / class_width
            t_name = f"t = (x - {round(assumed_mean, decimals)}) / {round(class_width, decimals)}"
            return t_values, t_name
    
    
    def _calculate_mean(
        self,
        class_labels: list[str],
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        class_width: int | float,
        midpoints: NDArray[float64],
        freq: NDArray[float64],
        assumed_mean: int | float | None,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int
    ) -> ResultDict:
        """Calculate mean for grouped data."""
        
        if assumed_mean is not None:
            if assumed_mean == -999:
                assumed_mean = 0.5 * lower_limits[0] + 0.5 * upper_limits[-1]
            else:
                assumed_mean = self._validate_assumed_mean(
                    assumed_mean=assumed_mean,
                    assumed_mean_formula=assumed_mean_formula
                )
            
            t_name, t_values = self._apply_assumed_mean_formula(
                midpoints=midpoints,
                class_width=class_width,
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                decimals=decimals
            )

            total_freq = sum(freq)
            weighted_sum = sum(t_values * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                t_name: t_values, 
                "Frequency (f)": freq,
                "ft": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                t_name: [nan], 
                "Frequency (f)": [total_freq],
                "ft": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return ResultDict(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": assumed_mean,
                    "formula": assumed_mean_formula,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum,
                }
            )
        else:
            total_freq = sum(freq)
            weighted_sum = sum(midpoints * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                "Frequency (f)": freq,
                "fx": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                "Frequency (f)": [total_freq],
                "fx": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return ResultDict(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": None,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum
                }
            )
            
    
    def _calculate_standard_deviation(
        self,
        class_labels: list[str],
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        class_width: int | float,
        midpoints: NDArray[float64],
        freq: NDArray[float64],
        assumed_mean: int | float | None,
        assumed_mean_formula: AssumedMeanFormulaType,
        decimals: int = 4
    ) -> ResultDict:
        """Calculate variance for grouped data."""
        
        if assumed_mean is not None:
            if assumed_mean == -999:
                assumed_mean = 0.5 * lower_limits[0] + 0.5 * upper_limits[-1]
            else:
                assumed_mean = self._validate_assumed_mean(
                    assumed_mean=assumed_mean,
                    assumed_mean_formula=assumed_mean_formula
                )
            
            t_name, t_values = self._apply_assumed_mean_formula(
                midpoints=midpoints,
                class_width=class_width,
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula,
                decimals=decimals
            )

            total_freq = sum(freq)
            weighted_sum = sum(t_values * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                t_name: t_values, 
                "Frequency (f)": freq,
                "ft": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                t_name: [nan], 
                "Frequency (f)": [total_freq],
                "ft": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return ResultDict(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": assumed_mean,
                    "formula": assumed_mean_formula,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum,
                }
            )
        else:
            total_freq = sum(freq)
            weighted_sum = sum(midpoints * freq)
            mean_value = weighted_sum / total_freq
            
            dframe = DataFrame({
                "Class": class_labels,
                "Midpoint (x)": round(midpoints, decimals),
                "Frequency (f)": freq,
                "fx": round(midpoints * freq, decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                "Frequency (f)": [total_freq],
                "fx": [round(weighted_sum, decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            
            return ResultDict(
                answer=round(mean_value, decimals),
                table=dframe,
                metadata={
                    "statistic": "mean",
                    "assumed_mean": None,
                    "total_frequency": total_freq,
                    "weighted_sum": weighted_sum
                }
            )
    
    
    def _calculate_percentiles(
        self,
        lower_limits: NDArray[float64],
        upper_limits: NDArray[float64],
        frequencies: NDArray[float64],
        class_labels: list[str],
        decimals: int,
        **kwargs
    ) -> ResultDict:
        """Calculate percentiles for grouped data."""
        pass


def sta_eda_grouped_mean(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    assumed_mean: int | float | None,
    assumed_mean_formula: AssumedMeanFormulaType = AssumedMeanFormulaType.X_MINUS_A_OVER_W,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    
    return result


def sta_eda_grouped_sd(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    assumed_mean: int | float | None,
    assumed_mean_formula: AssumedMeanFormulaType = AssumedMeanFormulaType.X_MINUS_A_OVER_W,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    
    return result


def sta_eda_grouped_percentiles(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    class_width: int | float,
    decimals: int = 4,
) -> float:
    """
    Convenience function to compute grouped mean.
    
    Args:
        lc_limits: Lower class limits
        uc_limits: Upper class limits
        freq: Frequencies
        decimals: Number of decimal places
        raise_on_error: Whether to raise exceptions or return NaN
        
    Returns:
        Calculated mean
    """
    compute_class = GroupedStatisticsCalculator()
    result = compute_class.compute(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        class_width=class_width,
        decimals=decimals
    )
    
    return result


def sta_freq_tally(
    data: list[int | float] | ndarray,
    class_width: int | float,
    start_from: int | float | None  = None,
    show_values: bool = False,
    include_cumfreq: bool = False,
    decimals: int = 4
) -> DataFrame:
    """
    Compute a grouped frequency distribution (frequency tally).

    This function groups one-dimensional numeric data into class intervals
    of fixed width, computes frequencies, and returns a structured result
    containing a formatted frequency table and summary statistics.

    Parameters
    ----------
    data : array-like of int or float
        Input data. Must be one-dimensional.
    class_width : int or float
        Width of each class interval.
    start_from : int or float, optional
        Lower bound of the first class. If ``None``, the first class starts at
        the floor of the minimum value rounded down to the nearest multiple
        of ``class_width``.
    show_values : bool, default False
        If ``True``, include a column listing individual values in each class.
        Values are shown only if the maximum class frequency does not exceed 50.
    include_cumfreq : bool, default False
        If ``True``, include a cumulative frequency column.
    decimals : int, default 4
        Number of decimal places used when formatting class limits and values.

    Returns
    -------
    result : ResultDict
        Dictionary-like result object with attribute access, containing:

        table : pandas.DataFrame
            Frequency table with columns:

            - ``Class`` : str
                Class interval labels.
            - ``Frequency`` : int
                Frequency per class.
            - ``Tally`` : str, optional
                Tally marks grouped in fives (omitted if max frequency > 50).
            - ``Values`` : str, optional
                Individual values per class (shown only if ``show_values=True``
                and max frequency ≤ 50).
            - ``Cum. Frequency`` : int, optional
                Cumulative frequency (included if ``include_cumfreq=True``).

        class_limits : ndarray
            Array of class interval labels.
        freq : ndarray
            Frequencies per class.
        cumfreq : ndarray
            Cumulative frequencies.
        col_names : pandas.Index
            Column names of the frequency table.
        stats : ResultDict
            Summary statistics with fields:

            - ``nrows`` : int
                Number of rows in the table.
            - ``ncols`` : int
                Number of columns in the table.
            - ``n`` : int
                Total number of observations.
            - ``min`` : float
                Minimum value.
            - ``max`` : float
                Maximum value.
            - ``range`` : float
                Data range.
            - ``mean`` : float
                Arithmetic mean.
            - ``var`` : float
                Variance.
            - ``std`` : float
                Standard deviation.

    Raises
    ------
    ValueError
        If ``data`` is not one-dimensional.
    TypeError
        If input values cannot be converted to numeric form.

    Warns
    -----
    FrequencyTallyWarning
        - If values below ``start_from`` are excluded, the excluded values
          are reported.
        - If the maximum class frequency exceeds 50, the ``Tally`` and
          ``Values`` columns are omitted for readability.

    Notes
    -----
    - Class intervals are left-closed and right-open.
    - The final class is dropped if its frequency is zero.
    - Tally marks are grouped in fives using visual separators.
    - Returned results use a ``ResultDict`` container, that supports 
      both key-based and attribute-based access to stored values..

    Examples
    --------
    >>> data = [
        38, 40, 54, 43, 43, 56, 46, 32, 37, 38, 52, 45, 45, 43, 38, 56, 46,
        26, 48, 38, 33, 40, 34, 36, 37, 29, 49, 43, 33, 52, 45, 40, 49, 44,
        41, 42, 46, 42, 40, 39, 36, 40, 32, 59, 52, 33, 39, 38, 48, 41
    ]
    >>> result = stm.sta_freq_tally(data, class_width=5, include_cumfreq=True)
    >>> result.table
         Class                  Tally  Frequency  Cum. Frequency
    1  25 - 29                     //          2               2
    2  30 - 34              ///// . /          6               8
    3  35 - 39      ///// . ///// . /         11              19
    4  40 - 44   ///// . ///// . ////         14              33
    5  45 - 49        ///// . ///// .         10              43
    6  50 - 54                   ////          4              47
    7  55 - 59                    ///          3              50
    
    >>> result.stats.mean
    41.92
    
    >>> result.stats.std
    7.143780511745863
    
    >>> result.cumfreq
    array([ 2,  8, 19, 33, 43, 47, 50], dtype=int64)
    """
    data_arr = asarray(data, dtype=float64)
    
    if data_arr.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
    
    # Vectorized calculations
    min_val = data_arr.min()
    max_val = data_arr.max()
    
    if start_from is None:
        start_val = floor(min_val / class_width) * class_width
    else:
        start_val = start_from
        
    below_mask = data_arr < start_val
    excluded_values = data_arr[below_mask]
    
    if len(excluded_values) > 0:
        data_arr = data_arr[data_arr >= start_val]
        excluded_values = numeric_format(excluded_values)
        excluded_str = str_data_join_contd(excluded_values)
        
        warnings.warn(
            f"{len(excluded_values)} value(s) below 'start_from' ({start_val}) "
            f"were excluded from the tally: {excluded_str}",
            category=FrequencyTallyWarning,
            stacklevel=2
        )

    end_val = ceil(max_val / class_width) * class_width + class_width
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
    
    if dframe["Frequency"].iat[-1] == 0:
        dframe = dframe.iloc[:-1]
    
    dframe.index = arange(1, len(dframe) + 1)
    freq_col = dframe["Frequency"]
    freq_cumsum = cumsum(freq_col)
    
    if include_cumfreq:
        dframe["Cum. Frequency"] = freq_cumsum
    
    data = asarray(data, dtype=float)
    nrows, ncols = dframe.shape
    
    return ResultDict(
        table=dframe,
        class_limits=dframe["Class"].values,
        freq=freq_col.values,
        cumfreq=freq_cumsum.values,
        columns=dframe.columns,
        stats = ResultDict(
            nrows=nrows,
            ncols=ncols,
            n=freq_col.sum(),
            min=data.min(),
            max=data.max(),
            range=data.max() - data.min(),
            mean=data.mean(),
            var=data.var(),
            std=data.std()
        )
    )