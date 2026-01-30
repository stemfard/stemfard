from __future__ import annotations

import warnings

from numpy import (
    arange, array, asarray, bincount, ceil, char, clip, cumsum, digitize, empty,
    float64, floor, median, mod, percentile, sqrt
)
from numpy.typing import NDArray
from scipy.stats import mode
from pandas import DataFrame, Series
from stemcore import arr_to_numeric, numeric_format, str_data_join_contd

from stemfard.core.utils_classes import FrequencyTallyWarning, rounded
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.models import FrequencyTallyResult, StatsDescriptives

# ================
# Helper functions
# ================

def _compute_bins(
    data: NDArray[float64],
    class_width: float,
    start_from: float | None
) -> NDArray[float64]:
    
    start = (
        floor(data.min() / class_width) * class_width
        if start_from is None else start_from
    )
    end = ceil(data.max() / class_width) * class_width + class_width
    
    return arange(start, end + class_width, class_width)


def _format_class_labels(bins: NDArray[float64], decimals: int) -> list[str]:
    
    labels: list[str] = []

    for left, right in zip(bins[:-1], bins[1:]):
        left_s = f"{left:.{decimals}f}".rstrip("0").rstrip(".")
        right_s = f"{right:.{decimals}f}".rstrip("0").rstrip(".")
        labels.append(f"{left_s} ≤ x < {right_s}")

    return labels


def _create_tally(count: int) -> str:
    
    if count <= 0:
        return ""
    groups, remainder = divmod(count, 5)
    parts = ["///// ."] * groups
    if remainder:
        parts.append("/" * remainder)
        
    return " ".join(parts)


def _format_values(values: NDArray[float64], decimals: int) -> str:
    if values.size == 0:
        return ""

    is_int = mod(values, 1) == 0
    result = empty(values.size, dtype=object)

    result[is_int] = values[is_int].astype(int).astype(str)

    floats = values[~is_int]
    if floats.size:
        formatted = char.mod(f"%.{decimals}f", floats)
        formatted = char.rstrip(formatted, "0")
        formatted = char.rstrip(formatted, ".")
        result[~is_int] = formatted

    return ", ".join(result)


# ==========
# Public API
# ==========

def sta_freq_tally(
    data: SequenceArrayLike,
    class_width: int | float,
    start_from: int | float | None = None,
    show_values: bool = False,
    include_cumfreq: bool = False,
    conf_level: float = 0.95,
    decimals: int = 4,
) -> FrequencyTallyResult:
    """
    Compute a grouped frequency distribution (frequency tally).

    This function groups one-dimensional numeric data into class
    intervals of fixed width, computes frequencies, and returns
    a structured result containing a formatted frequency table and
    summary statistics.

    Parameters
    ----------
    data : array-like of int or float
        Input data. Must be one-dimensional.
    class_width : int or float
        Width of each class interval.
    start_from : int or float, optional
        Lower bound of the first class. If ``None``, the first class
        starts at the floor of the minimum value rounded down to the 
        nearest multiple of ``class_width``.
    show_values : bool, default False
        If ``True``, include a column listing individual values in each
        class. Values are shown only if the maximum class frequency
        does not exceed 50.
    include_cumfreq : bool, default False
        If ``True``, include a cumulative frequency column.
    conf_level : int, default float
        Confidence interval.
    decimals : int, default 4
        Number of decimal places used when formatting class limits and
        values.

    Returns
    -------
    result : FrequencyTallyResult
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
      both key-based and attribute-based access to stored values.

    Examples
    --------
    >>> data = [
        38, 40, 54, 43, 43, 56, 46, 32, 37, 38, 52, 45, 45, 43, 38, 56, 46,
        26, 48, 38, 33, 40, 34, 36, 37, 29, 49, 43, 33, 52, 45, 40, 49, 44,
        41, 42, 46, 42, 40, 39, 36, 40, 32, 59, 52, 33, 39, 38, 48, 41
    ]
    >>> result = stm.sta_freq_tally(data, class_width=5, include_cumfreq=True)
    >>> result.table
             Class  Frequency                 Tally  Cum. Frequency
    1  25 ≤ x < 30          2                    //               2
    2  30 ≤ x < 35          6             ///// . /               8
    3  35 ≤ x < 40         11     ///// . ///// . /              19
    4  40 ≤ x < 45         14  ///// . ///// . ////              33
    5  45 ≤ x < 50         10       ///// . ///// .              43
    6  50 ≤ x < 55          4                  ////              47
    7  55 ≤ x < 60          3                   ///              50
    
    >>> result.stats.mean
    41.92
    
    >>> result.stats.std
    7.143780511745863
    
    >>> result.cumfreq
    array([ 2,  8, 19, 33, 43, 47, 50], dtype=int64)
    """
    from stemfard.stats.descriptives.others import stats_mean_ci
    
    # ---- Parse & validate input ----
    data_arr = arr_to_numeric(data, dtype=float)

    ndims = data_arr.ndim
    if ndims != 1:
        raise ValueError(f"Expected 'data' to be one-dimensional, got {ndims}")

    # ---- Exclude values below start ----
    bins = _compute_bins(data_arr, float(class_width), start_from)
    start_val = bins[0]

    below_mask = data_arr < start_val
    if below_mask.any():
        excluded = numeric_format(data_arr[below_mask])
        warnings.warn(
            f"{excluded.size} value(s) below start_from ({start_val}) were "
            f"excluded: {str_data_join_contd(excluded)}",
            category=FrequencyTallyWarning,
            stacklevel=2,
        )
        data_arr = data_arr[~below_mask]

    # ---- Frequency calculation ----
    indices = digitize(data_arr, bins, right=False) - 1
    indices = clip(indices, 0, len(bins) - 2)

    freq = bincount(indices, minlength=len(bins) - 1)

    # ---- Build table ----
    labels = _format_class_labels(bins, decimals)
    tallies = [_create_tally(int(f)) for f in freq]

    table_data: dict[str, list] = {
        "Class": labels,
        "Tally": tallies,
        "Frequency": freq.tolist()
    }

    if show_values and freq.max() <= 50:
        table_data["Values"] = [
            _format_values(data_arr[indices == i], decimals)
            for i in range(len(freq))
        ]

    df = DataFrame(table_data)

    if freq.max() > 50:
        warnings.warn(
            "Maximum frequency exceeds 50: 'Tally' and 'Values' columns "
            "omitted.",
            category=FrequencyTallyWarning,
            stacklevel=2,
        )
        df = df[["Class", "Frequency"]]

    # Drop trailing zero-frequency class
    if df["Frequency"].iat[-1] == 0:
        df = df.iloc[:-1]

    df.index = arange(1, df.shape[1] + 1)

    # ---- Cumulative frequency ----
    cumfreq = cumsum(df["Frequency"].to_numpy())

    if include_cumfreq:
        df["Cum. Frequency"] = cumfreq

    # ---- Stats ----
    full_data = asarray(data, dtype=float)
    total_freq = freq.sum()
    mean_ci = stats_mean_ci(
        data=full_data,
        conf_level=conf_level,
        steps_compute=False,
        steps_detailed=False,
        show_bg=False
    ).answer()
    n_sqrt = sqrt(full_data.size)
    p = (25, 50, 75)
    percentiles = percentile(a=full_data, q=p, method="linear")
    percentiles_arr = array([p, percentiles]).T
    iqr = percentiles[2] - percentiles[0]
    res_mode = mode(full_data, keepdims=False)
    data_mean = full_data.mean()
    data_std = full_data.std()
    data_kurt = Series(full_data).kurt()
    
    stats = StatsDescriptives(
        k=df.shape[0],
        n=total_freq,
        dfn=total_freq - 1,
        total=None,
        min=rounded(full_data.min(), decimals),
        max=rounded(full_data.max(), decimals),
        range=rounded(full_data.max() - full_data.min(), decimals),
        percentiles=rounded(percentiles_arr, decimals),
        p25=rounded(percentiles[0], decimals),
        p50=rounded(percentiles[1], decimals),
        p75=rounded(percentiles[2], decimals),
        iqr=rounded(iqr, decimals),
        iqd=rounded(iqr / 2, decimals),
        mode={
            "mode": rounded(res_mode.mode, decimals),
            "count": rounded(res_mode.count, decimals)
        },
        median=rounded(median(full_data), decimals),
        mean=rounded(data_mean, decimals),
        conf_level=rounded(conf_level, decimals),
        mean_ci={
            "lower": rounded(mean_ci.lower, decimals),
            "upper": rounded(mean_ci.upper, decimals)
        },
        var=rounded(full_data.var(ddof=1), decimals),
        std=rounded(data_std, decimals),
        stderror=rounded(full_data.std(ddof=1) / n_sqrt, decimals),
        sem=rounded(Series(full_data).sem(ddof=1), decimals),
        cv=rounded(data_std / data_mean, decimals),
        skew=rounded(Series(full_data).skew(), decimals),
        kurt={
            "kurt": rounded(data_kurt, decimals),
            "kurt + 3": rounded(data_kurt + 3, decimals)
        }
    )

    return FrequencyTallyResult(
        table=df,
        class_limits=df["Class"].to_numpy(),
        freq=df["Frequency"].to_numpy(),
        cumfreq=cumfreq,
        columns=df.columns.to_numpy(),
        stats=stats,
        params=data,
        params_parsed=data_arr
    )