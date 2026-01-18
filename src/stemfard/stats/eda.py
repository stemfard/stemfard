from typing import Any, Literal
import warnings

from numpy import (
    arange, around, asarray, bincount, ceil, char, clip, cumsum, digitize,
    empty, float64, floor, mod, nan, ndarray
)
from numpy.typing import NDArray
from pandas import DataFrame, concat
from verifyparams import (
    verify_all_integers, verify_decimals, verify_int_or_float,
    verify_len_equal, verify_membership
)
from stemcore import arr_to_numeric, numeric_format, str_data_join_contd

from stemfard.core.utils_classes import FrequencyTallyWarning, ResultDict


_ALLOWED_FORMULAS = ["x-a", "x/w-a", "(x-a)/w"]
_ALLOWED_STATISTICS = ["mean", "std", "percentiles"]


FORMULAS_ARITHMETIC_MEAN = {
    "values_only": (
        f"x_{{1}}, \\: x_{{2}}, \\: x_{{3}}, \\: \\cdots, \\: x_{{n}}"
    ),
    "values_and_freq": (
        f"\\begin{{array}}{{|l|c|c|c|c|c|}} \\hline "
        f"\\mathrm{{Data}} & x_{{1}} & x_{{2}} & x_{{3}} & \\cdots & x_{{n}} \\\\ \\hline "
        f"\\mathrm{{Frequency}} & f_{{1}} & f_{{2}} & f_{{3}} & \\cdots & f_{{n}} \\\\ \\hline "
        f"\\end{{array}}"
    ),
    "arithmetic_mean": (
        f"\\displaystyle \\bar{{x}} = \\frac{{ \\sum x_{{i}}}}{{n}}"
    ),
    "arithmetic_mean_with_assumed_mean": (
        f"\\displaystyle \\bar{{x}} = A + \\frac{{ \\sum (x_{{i}} - A)}}{{n}}"
    ),
    "arithmetic_mean_with_freq": (
        f"\\displaystyle \\bar{{x}} "
        f"= \\frac{{ \\sum\\mathrm{{fx}}}}{{ \\sum\\mathrm{{f}}}}"
    ),
    "arithmetic_mean_with_freq_and_assumed_mean": (
        f"\\displaystyle \\bar{{x}} "
        f"= A + \\frac{{ \\sum\\mathrm{{ft}}}}{{ \\sum\\mathrm{{f}}}}"
    )
}


FORMULAS_ASSUMED_MEAN_TYPES: Literal["x-a", "x/w-a", "(x-a)/w"] = {
    "x-a": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\)",
        "Where \\( f \\) is the frequency and \\( t = x - A \\)."
    ],
    "x/w-a": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= \\left(A^{{*}} + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\right) "
        "\\times w \\)",
        f"Where \\( A^{{*}} \\) is the new assumed mean, \\( f \\) is the "
        "frequency, \\( t = x - A\\) and \\( w \\) is the "
        f"\\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ],
    "(x-a)/w": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\times w \\)",
        "Where \\( f \\) is the frequency, \\( t = x - A\\) and \\( w \\) "
        f"is the \\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ]
}


class GroupedStatisticsCalculator:

    def __init__(
        self,
        lower_limits: list[int | float] | NDArray[Any],
        upper_limits: list[int | float] | NDArray[Any],
        freq: list[int | float] | NDArray[Any],
        statistic: Literal["mean", "std", "percentiles"],
        decimals: int = 4,
        **kwargs
    ) -> None:
        """
        Initialize the calculator.
        """
        self.params = {}
        self.params_parsed = {}
        
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.freq = freq
        self.statistic = statistic
        self.decimals = decimals
        self.kwargs = kwargs
        
        # initialization
        self.class_labels = ""
        
        self._validate_common_params()
        
        self.params.update(
            {
                "lower_limits": lower_limits,
                "upper_limits": upper_limits,
                "freq": freq,
                "statistic": statistic,
                "decimals": decimals
            }
        )
        
        self.params_parsed.update(
            {
                "lower_limits": self.lower_limits,
                "upper_limits": self.upper_limits,
                "freq": self.freq,
                "statistic": self.statistic,
                "decimals": self.decimals
            }
        )
    
    
    def _validate_common_params(self) -> None:
        
        self.lower_limits = arr_to_numeric(
            data=self.lower_limits, param_name="lower_limits"
        )
        self.upper_limits = arr_to_numeric(
            data=self.upper_limits, param_name="upper_limits"
        )
        self.freq = arr_to_numeric(data=self.freq, dtype=None)
        
        verify_all_integers(value=self.freq, param_name="freq")
        
        # convert to integers after verification above
        self.freq = asarray(self.freq, dtype=int)
        
        verify_len_equal(
            self.lower_limits,
            self.upper_limits,
            self.freq,
            param_names=["lower_limits", "upper_limits", "freq"]
        )
        
        verify_membership(
            user_input=self.statistic,
            valid_items=_ALLOWED_STATISTICS,
            param_name="statistic"
        )
        
        self.decimals = verify_decimals(
            value=self.decimals, param_name="decimals"
        )
        
        self.class_labels = [
            f"{round(numeric_format(lower), self.decimals)} - "
            f"{round(numeric_format(upper), self.decimals)}"
            for lower, upper in zip(self.lower_limits, self.upper_limits)
        ]
        
        self.class_width = self.upper_limits[0] - self.lower_limits[0] + 1
        self.midpoints = self.lower_limits * 0.5 + self.upper_limits * 0.5
    
    
    def _validate_params_means(
        self,
        assumed_mean: int | float | None,
        assumed_mean_formula: Literal["mean", "std", "percentiles"]
    ) -> None:
        """Validate assumed mean parameters."""
        self.assumed_mean = verify_int_or_float(
            value=assumed_mean,
            allow_none=True,
            param_name="assumed_mean"
        )
        
        self.assumed_mean_formula = verify_membership(
            user_input=assumed_mean_formula,
            valid_items=_ALLOWED_FORMULAS,
            param_name="assumed_mean_formula"
        )

        self.params.update(
            {
                "assumed_mean": assumed_mean,
                "assumed_mean_formula": assumed_mean_formula
            }
        )
        self.params_parsed.update(
            {
                "assumed_mean": self.assumed_mean,
                "assumed_mean_formula": self.assumed_mean_formula
            }
        )
    
    
    def _validate_params_percentiles(
        self,
        percentiles,
        cumfreq_curve,
        x_values,
        **plot_kwargs
    ) -> None:
        
        self.percentiles = percentiles
        self.cumfreq_curve = cumfreq_curve
        self.x_values = x_values
        
        perc_dct = {"percentiles": percentiles}
        self.params.update(perc_dct)
        self.params_parsed.update(perc_dct)
        
        if cumfreq_curve:
            params_dct = {
                "x_values": x_values,
                "fig_width": plot_kwargs.get("fig_width", 6),
                "fig_height": plot_kwargs.get("fig_height", 4),
                "line_color": plot_kwargs.get("line_color", "blue"),
                "xaxis_orientation": plot_kwargs.get("xaxis_orientation", 0),
                "x_title": plot_kwargs.get("x_title", "Data")
            }
            self.params.update(params_dct)
            self.params_parsed.update(params_dct)
    
    
    def _apply_assumed_mean_formula(self) -> tuple[NDArray[float64], str]:
        """Apply assumed mean formula to midpoints."""
        if self.assumed_mean_formula == "x-a":
            t_values = self.midpoints - self.assumed_mean
            t_name = f"t = x - {round(self.assumed_mean, self.decimals)}"
            return t_values, t_name
        elif self.assumed_mean_formula == "x/w-a":
            t_values = self.midpoints / self.class_width - self.assumed_mean
            t_name = (
                f"t = x / {round(self.class_width, self.decimals)} - "
                f"{round(self.assumed_mean, self.decimals)}"
            )
            return t_values, t_name
        elif self.assumed_mean_formula == "(x-a)/w":
            t_values = (self.midpoints - self.assumed_mean) / self.class_width
            t_name = (
                f"t = (x - {round(self.assumed_mean, self.decimals)}) / "
                f"{round(self.class_width, self.decimals)}"
            )
            return t_values, t_name
    
    
    def _table_grouped_qtn(self) -> tuple[str, str, str, str]:
    
        df_series = {
            "Class": self.class_labels,
            "Frequency": around(self.freq, self.decimals)
        }
        
        table_qtn_df = DataFrame(data=df_series)
        table_qtn_df_rowise = table_qtn_df.T
        table_qtn_df_rowise.index = ["Class", "Frequency"]
        table_qtn_df_rowise.columns = range(1, table_qtn_df_rowise.shape[1] + 1)
        
        # ComputedDataModel for `data`
        obj_values_data = table_qtn_df.values.tolist()
        latex_data = dframe_to_mathjax_array(
            df=table_qtn_df,
            include_index=False,
            outer_border=True,
            inner_vlines=True
        )

        latex_data_rowise = dframe_to_mathjax_array(
            df=table_qtn_df_rowise,
            include_index=True,
            outer_border=True,
            inner_vlines=True
        )
        latex_data_rowise = latex_data_rowise\
            .replace("l|", "c|")\
            .replace("c|", "l|", 1)\
            .replace(f"\\mathrm{{Frequency}}", f"\\hline\\mathrm{{Frequency}}")\
            .replace("  & 1", "\\qquad i  & 1", 1)
        
        csv_data = result_to_csv(obj=table_qtn_df)
        
        return obj_values_data, latex_data, csv_data, latex_data_rowise
        
        
    def _prepare_out_dframe() -> DataFrame:
        
        NotImplemented
        
    
    def _prepare_out_dframe_latex() -> DataFrame:
        
        NotImplemented
        
        
    def _prepare_out_dframe_csv() -> DataFrame:
        
        NotImplemented
    
    
    def _calculate_mean(self, assumed_mean, assumed_mean_formula) -> ResultDict:
        """Calculate mean for grouped data."""
        self.assumed_mean = assumed_mean
        self.assumed_mean_formula = assumed_mean_formula
        
        self._validate_params_means(
            assumed_mean=assumed_mean,
            assumed_mean_formula=assumed_mean_formula
        )
        
        if self.assumed_mean is not None:
            if self.assumed_mean == -999:
                assumed_mean = (
                    0.5 * self.lower_limits[0] + 0.5 * self.upper_limits[-1]
                )
            else:
                self.assumed_mean = self._validate_assumed_mean(
                    assumed_mean=self.assumed_mean,
                    assumed_mean_formula=self.assumed_mean_formula
                )
            
            t_name, t_values = self._apply_assumed_mean_formula(
                midpoints=self.midpoints,
                class_width=self.class_width,
                assumed_mean=self.assumed_mean,
                assumed_mean_formula=self.assumed_mean_formula,
                decimals=self.decimals
            )

            total_freq = sum(self.freq)
            total_fx_or_ft = sum(t_values * self.freq)
            mean_value = total_fx_or_ft / total_freq
            
            dframe = DataFrame({
                "Class": self.class_labels,
                "Midpoint (x)": self.midpoints,
                t_name: t_values,
                "Frequency (f)": self.freq,
                "ft": self.midpoints * self.freq
            })
            
            # dframe = df_add_rows(
            #     df=dframe,
            #     rows=[["Total", nan, nan, total_freq, total_fx_or_ft]],
            #     row_names=["Total"]
            # )
            
            dframe = dframe.fillna("").round(self.decimals)
            nrows, ncols = dframe.shape
            mean_float = round(mean_value, self.decimals)
            
            return ResultDict(
                params=self.params,
                params_parsed=self.params_parsed,
                answer=mean_float,
                table=dframe,
                stats={
                    "nrows": nrows,
                    "ncols": ncols, 
                    "mean": mean_float,
                    "total_freq": total_freq,
                    "total_ft": total_fx_or_ft,
                }
            )
        else:
            total_freq = sum(self.freq)
            total_fx_or_ft = sum(self.midpoints * self.freq)
            mean_value = total_fx_or_ft / total_freq
            
            dframe = DataFrame({
                "Class": self.class_labels,
                "Midpoint (x)": around(self.midpoints, self.decimals),
                "Frequency (f)": self.freq,
                "fx": around(self.midpoints * self.freq, self.decimals)
            })
            
            # Add total row
            total_row = DataFrame({
                "Class": ["Total"],
                "Midpoint (x)": [nan],
                "Frequency (f)": [total_freq],
                "fx": [round(total_fx_or_ft, self.decimals)]
            })
            dframe = concat([dframe, total_row], ignore_index=True)
            nrows, ncols = dframe.shape
            mean_float = round(mean_value, self.decimals)
            
            return ResultDict(
                params=self.params,
                params_parsed=self.params_parsed,
                answer=round(mean_value, self.decimals),
                table=dframe,
                stats={
                    "nrows": nrows,
                    "ncols": ncols, 
                    "mean": mean_float,
                    "total_freq": total_freq,
                    "total_fx": total_fx_or_ft,
                }
            )
            
    
    def _calculate_mean_steps(self) -> list[str]:
        """Calculate variance for grouped data."""
        pass
    
    
    def _calculate_standard_deviation(self) -> ResultDict:
        """Calculate variance for grouped data."""
        pass
    
    
    def _calculate_standard_deviation_steps(self) -> list[str]:
        """Calculate variance for grouped data."""
        pass
    
    
    def _calculate_percentiles(self) -> ResultDict:
        """Calculate percentiles for grouped data."""
        pass
    
    
    def _calculate_percentiles_steps(self) -> list[str]:
        """Calculate percentiles for grouped data."""
        pass

    
    def compute(self, **kwargs) -> ResultDict:
        """
        Calculate grouped statistics with comprehensive error handling.
        """
        assumed_mean = kwargs.get("assumed_mean")
        assumed_mean_formula = kwargs.get("assumed_mean_formula", "(x-a)/w")
        
        if self.statistic == "mean":
            return self._calculate_mean(
                assumed_mean=assumed_mean,
                assumed_mean_formula=assumed_mean_formula
            )
        elif self.statistic == "std":
            return self._calculate_standard_deviation()
        elif self.statistic == "percentiles":
            return self._calculate_percentiles()
    
    
def sta_eda_grouped_mean(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    assumed_mean: int | float | None,
    assumed_mean_formula: Literal["mean", "std", "percentiles"] = "mean",
    decimals: int = 4,
) -> float:
    "Compute mean for grouped data"
    compute_class = GroupedStatisticsCalculator(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        statistic="mean",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    return compute_class.compute()


def sta_eda_grouped_mean_steps(
    lower_limits: list[int | float] | NDArray[float64],
    upper_limits: list[int | float] | NDArray[float64],
    freq: list[int | float] | NDArray[float64],
    assumed_mean: int | float | None,
    assumed_mean_formula: Literal["mean", "std", "percentiles"] = "mean",
    decimals: int = 4,
) -> float:
    "Compute mean for grouped data"
    compute_class = GroupedStatisticsCalculator(
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        freq=freq,
        statistic="mean",
        assumed_mean=assumed_mean,
        assumed_mean_formula=assumed_mean_formula,
        decimals=decimals
    )
    return compute_class.compute()


def sta_eda_grouped_std() -> None:
    pass


def sta_eda_grouped_std_steps() -> None:
    pass


def sta_eda_grouped_percentiles() -> None:
    pass


def sta_eda_grouped_percentiles_steps() -> None:
    pass


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
        columns : pandas.Index
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
                Data range (max - min).
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
    params = {}
    data_arr = arr_to_numeric(data, dtype=float)
    params.update(
        {
            "data": data,
            "data_parsed": data_arr
        }
    )
    
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
        params=params,
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