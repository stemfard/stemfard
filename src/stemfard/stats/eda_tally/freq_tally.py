from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

from numpy import (
    arange, asarray, bincount, ceil, char, clip, cumsum, digitize, empty,
    float64, floor, int64, mod
)
from numpy.typing import NDArray
from pandas import DataFrame

from stemcore import arr_to_numeric, numeric_format, str_data_join_contd
from stemfard.core.utils_classes import FrequencyTallyWarning


# ===========
# Data models
# ===========

@dataclass(frozen=True)
class FrequencyStats:
    nrows: int
    ncols: int
    n: int
    min: float
    max: float
    range: float
    mean: float
    var: float
    std: float


@dataclass(frozen=True)
class FrequencyTallyResult:
    table: DataFrame
    class_limits: NDArray[float64]
    freq: NDArray[int64]
    cumfreq: NDArray[int64]
    columns: DataFrame.columns
    stats: FrequencyStats
    params: dict


# =========================
# Helper functions
# =========================

def _compute_bins(
    data: NDArray[float64],
    class_width: float,
    start_from: float | None
) -> NDArray[float64]:
    start = (
        floor(data.min() / class_width) * class_width
        if start_from is None
        else start_from
    )
    end = ceil(data.max() / class_width) * class_width + class_width
    return arange(start, end + class_width, class_width)


def _format_class_labels(
    bins: NDArray[float64],
    decimals: int
) -> list[str]:
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


def _format_values(
    values: NDArray[float64],
    decimals: int
) -> str:
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


# =========================
# Public API
# =========================

def sta_freq_tally(
    data: Iterable[int | float] | NDArray,
    class_width: int | float,
    start_from: int | float | None = None,
    show_values: bool = False,
    include_cumfreq: bool = False,
    decimals: int = 4,
) -> FrequencyTallyResult:
    """
    Compute a grouped frequency distribution (frequency tally).
    """

    # ---- Parse & validate input ----
    data_arr = arr_to_numeric(data, dtype=float)

    if data_arr.ndim != 1:
        raise ValueError("Data must be one-dimensional")

    params = {
        "data": data,
        "data_parsed": data_arr,
    }

    # ---- Exclude values below start ----
    bins = _compute_bins(data_arr, float(class_width), start_from)
    start_val = bins[0]

    below_mask = data_arr < start_val
    if below_mask.any():
        excluded = numeric_format(data_arr[below_mask])
        warnings.warn(
            f"{excluded.size} value(s) below start_from ({start_val}) were excluded: "
            f"{str_data_join_contd(excluded)}",
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
        "Frequency": freq.tolist(),
        "Tally": tallies,
    }

    if show_values and freq.max() <= 50:
        table_data["Values"] = [
            _format_values(data_arr[indices == i], decimals)
            for i in range(len(freq))
        ]

    df = DataFrame(table_data)

    if freq.max() > 50:
        warnings.warn(
            "Maximum frequency exceeds 50; 'Tally' and 'Values' columns omitted.",
            category=FrequencyTallyWarning,
            stacklevel=2,
        )
        df = df[["Class", "Frequency"]]

    # Drop trailing zero-frequency class
    if df["Frequency"].iat[-1] == 0:
        df = df.iloc[:-1]

    df.index = arange(1, len(df) + 1)

    # ---- Cumulative frequency ----
    cumfreq = cumsum(df["Frequency"].to_numpy())

    if include_cumfreq:
        df["Cum. Frequency"] = cumfreq

    # ---- Stats ----
    full_data = asarray(data, dtype=float)

    stats = FrequencyStats(
        nrows=df.shape[0],
        ncols=df.shape[1],
        n=int(df["Frequency"].sum()),
        min=float(full_data.min()),
        max=float(full_data.max()),
        range=float(full_data.max() - full_data.min()),
        mean=float(full_data.mean()),
        var=float(full_data.var()),
        std=float(full_data.std()),
    )

    return FrequencyTallyResult(
        table=df,
        class_limits=df["Class"].to_numpy(),
        freq=df["Frequency"].to_numpy(),
        cumfreq=cumfreq,
        columns=df.columns,
        stats=stats,
        params=params,
    )