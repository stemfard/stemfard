"""
Data models for grouped statistics calculation.

Immutability contract:
All classes in this module should be treated as immutable after construction.
Mutating attributes after initialization may lead to undefined behavior,
especially for cached properties.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal
from numpy import allclose, around, asarray, float64, int64
from numpy.typing import NDArray
from stemcore import arr_to_numeric, numeric_format
from verifyparams import (
    verify_all_integers, verify_boolean, verify_decimals,
    verify_elements_in_range, verify_int, verify_int_or_float,
    verify_len_equal, verify_membership, verify_numeric,
    verify_strictly_increasing, verify_string,
    verify_lower_lte_upper_arr
)

from stats.eda_grouped_data.formulas import (
    ALLOWED_FORMULAS, ALLOWED_STATISTICS, FIGURE_COLORS
)


# =========================================
# PARAMETER OBJECTS (INPUT VALIDATION ONLY)
# =========================================


@dataclass(slots=True)
class ParamsData:
    """
    Container for grouped data parameters.

    Instances should be treated as immutable after construction.
    """
    lower: list[int | float] | NDArray[float64]
    upper: list[int | float] | NDArray[float64]
    freq: list[int] | NDArray[int64]
    statistic: Literal["mean", "std", "percentiles"]
    decimals: int = 4
    
    def __post_init__(self):
        """Validate the grouped data after initialization."""
        self.lower = arr_to_numeric(
            data=self.lower, kind="array", param_name="lower"
        )
        verify_strictly_increasing(value=self.lower, param_name="lower")
        self.upper = arr_to_numeric(
            data=self.upper, kind="array", param_name="upper"
        )
        verify_strictly_increasing(value=self.upper, param_name="upper")
        verify_lower_lte_upper_arr(
            x=self.lower,
            y=self.upper,
            param_names=("lower", "upper")
        )
        _freq = arr_to_numeric(data=self.freq, kind="array", param_name="freq")
        verify_all_integers(value=_freq, param_name="freq")
        self.freq = asarray(_freq, dtype=int64)
        if self.freq.sum() <= 0:
            raise ValueError("Total frequency must be greater than zero")
        
        verify_len_equal(
            self.lower,
            self.upper,
            self.freq,
            param_names=["lower", "upper", "freq"]
        )
        
        verify_membership(
            value=self.statistic,
            valid_items=ALLOWED_STATISTICS,
            param_name="statistic"
        )
        
        verify_decimals(value=self.decimals)
    
    def __repr__(self) -> str:
        return (
            f"ParamsData(classes={len(self.lower)}, statistic={self.statistic})"
        )


@dataclass(slots=True)
class ParamsMeanStd:
    """
    Parameters for mean and standard deviation calculations.

    Instances should be treated as immutable after construction.
    """
    assumed_mean: int | float | Literal["auto"] | None = None
    formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None
    formula_std: Literal[1, 2, 3] = 1
    
    def __post_init__(self):
        """Validate mean parameters."""
        if self.assumed_mean is None:
            self.formula = None
        else:
            if self.assumed_mean != "auto":
                verify_int_or_float(
                    value=self.assumed_mean, param_name="assumed_mean"
                )
            
            if self.formula is None:
                self.formula = "(x-a)/w"
            
            verify_membership(
                value=self.formula,
                valid_items=ALLOWED_FORMULAS,
                param_name="formula"
            )
            
            verify_membership(
                value=self.formula_std,
                valid_items=[1, 2, 3],
                param_name="formula_std"
            )


@dataclass(slots=True)
class ParamsPercentiles:
    """
    Parameters for percentile calculations on grouped data.

    Instances should be treated as immutable after construction.
    """
    percentiles: int | float | list[int | float] | NDArray[int64 | float64]
    cumfreq_curve: bool = False
    x_values: int | float | list[int | float] | NDArray[int64 | float64] | None = None
    
    def __post_init__(self):
        """Validate percentile parameters."""
        self.percentiles = arr_to_numeric(
            data=self.percentiles, kind="array", param_name="percentiles"
        ).copy()
        
        self.cumfreq_curve = verify_boolean(
            value=self.cumfreq_curve, default=False
        )
        
        # Verify percentiles are between 0 and 100
        verify_elements_in_range(
            data=self.percentiles,
            lower=0,
            upper=100,
            param_name="percentiles"
        )
        
        if self.x_values is not None:
            self.x_values = arr_to_numeric(
                data=self.x_values, kind="array", param_name="x_values"
            ).copy()


@dataclass(slots=True)
class ParamsPlot:
    """
    Parameters controlling grouped-data plots.

    Instances should be treated as immutable after construction.
    """
    fig_width: int = 6
    fig_height: int = 4
    line_color: Literal[
        "blue", "black", "green", "red", "purple", "orange"
    ] = "blue"
    xaxis_orientation: int = 0
    x_label: str = "Data"
    y_label: str = "Cumulative Frequency"
    grid: bool = True
    
    def __post_init__(self):
        """Validate plot parameters."""
        verify_int(value=self.fig_width, param_name="fig_width")
        verify_int(value=self.fig_height, param_name="fig_height")
        
        verify_membership(
            value=self.line_color,
            valid_items=FIGURE_COLORS,
            param_name="line_color"
        )
        
        verify_numeric(
            value=self.xaxis_orientation,
            limits=[-180, 180],
            param_name="xaxis_orientation"
        )
        
        verify_string(value=self.x_label, param_name="x_label")
        verify_string(value=self.y_label, param_name="y_label")
        self.grid = verify_boolean(value=self.grid, default=True)


# ====================
# STRUCTURE + GEOMETRY
# ====================


@dataclass(slots=True)
class GroupedStructure:
    """
    Represents the geometric structure of grouped data.

    Instances should be treated as immutable after construction.
    """
    lower: NDArray[float64]
    upper: NDArray[float64]
    freq: NDArray[int64]
    decimals: int
    
    def __post_init__(self) -> None:
        if len(self.lower) < 2:
            raise ValueError("'lower' requires at least two elements")
        
        if len(self.upper) < 2:
            raise ValueError("'upper' requires at least two elements")

        if not allclose(self.lower[1:], self.upper[:-1]):
            raise ValueError("Class intervals must be contiguous")

        widths = self.upper - self.lower
        if not allclose(widths, widths[0]):
            raise ValueError("Class widths must be constant")
        
    @property
    def n(self) -> int:
        return len(self.lower)
    
    @cached_property
    def class_width(self) -> float:
        return float((self.upper - self.lower)[0])
    
    @property
    def class_width_rounded(self) -> float:
        return around(self.class_width, self.decimals)
    
    @cached_property
    def midpoints(self) -> NDArray[float64]:
        return (self.lower + self.upper) / 2.0
    
    @property
    def midpoints_rounded(self) -> NDArray[float64]:
        return around(self.midpoints, self.decimals)
    
    @cached_property
    def lower_bnds(self) -> NDArray[float64]:
        delta = (self.lower[1] - self.upper[0]) / 2.0
        return self.lower - delta
    
    @cached_property
    def upper_bnds(self) -> NDArray[float64]:
        delta = (self.lower[1] - self.upper[0]) / 2.0
        return self.lower + delta
    
    @cached_property
    def class_boundaries(self) -> list[str]:
        delta = (self.lower[1] - self.upper[0]) / 2.0
        lb = self.lower - delta
        ub = self.upper + delta

        return [
            f"{numeric_format(around(a, self.decimals))} - "
            f"{numeric_format(around(b, self.decimals))}"
            for a, b in zip(lb, ub)
        ]

    @cached_property
    def class_labels(self) -> list[str]:
        return [
            f"{numeric_format(around(a, self.decimals))} - "
            f"{numeric_format(around(b, self.decimals))}"
            for a, b in zip(self.lower, self.upper)
        ]

    @cached_property
    def total_freq(self) -> int:
        return int(self.freq.sum())
    
    @property
    def params(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return "params", "params_parsed"
    

# =========================================
# TRANSFORMATION LAYER (ASSUMED MEAN LOGIC)
# =========================================

@dataclass(slots=True)
class GroupedTransform:
    """
    Transformation layer for grouped data, including assumed-mean logic.

    Instances should be treated as immutable after construction.
    """
    structure: GroupedStructure
    params_mean: ParamsMeanStd
    
    @cached_property
    def assumed_mean(self) -> float | None:
        a = self.params_mean.assumed_mean
        if a is None:
            return None
        if a == "auto":
            return (self.structure.lower[0] + self.structure.upper[-1]) / 2.0
        return a
    
    @property
    def assumed_mean_rounded(self) -> float | None:
        if self.assumed_mean is None:
            return None
        return around(self.assumed_mean, self.structure.decimals)
    
    @property
    def is_calculate_assumed_mean(self) -> bool:
        """Check if assumed mean should be calculated automatically."""
        return self.params_mean.assumed_mean == "auto"
    
    @cached_property
    def assumed_mean_asterisk(self) -> float | None:
        """Calculate a* (assumed mean divided by class width)."""
        if self.assumed_mean is not None:
            return self.assumed_mean / self.structure.class_width
        return None
    
    @property
    def assumed_mean_asterisk_rounded(self) -> float | None:
        """Calculate rounded a*."""
        a_star = self.assumed_mean_asterisk
        if a_star is not None:
            return around(a_star, self.structure.decimals)
        return None
    
    @property
    def tname(self) -> str:
        """Get the column name for t values."""
        if self.assumed_mean_param is None:
            return "x"
        else:
            return "t"
    
    @cached_property
    def tvalues(self) -> NDArray[float64]:
        mid = self.structure.midpoints
        w = self.structure.class_width
        a = self.assumed_mean

        if a is None:
            return mid

        match self.params_mean.formula:
            case "x-a":
                return mid - a
            case "x/w-a":
                return mid / w - a / w
            case "(x-a)/w":
                return (mid - a) / w
            case _:
                raise RuntimeError("Invalid formula state")

    @property
    def tvalues_rounded(self) -> NDArray[float64]:
        return around(self.tvalues, self.structure.decimals)
    
    
# =============================
# STATISTICS ENGINE (PURE MATH)
# =============================

@dataclass(slots=True)
class GroupedStatistics:
    """
    Statistical engine for grouped data computations.

    Instances should be treated as immutable after construction.
    """
    structure: GroupedStructure
    transform: GroupedTransform
    
    @cached_property
    def fxt(self) -> NDArray[float64]:
        return self.transform.tvalues * self.structure.freq
    
    @property
    def fxt_rounded(self) -> NDArray[float64]:
        return around(self.fxt, self.structure.decimals)
    
    @cached_property
    def total_fxt(self) -> float:
        return float(self.fxt.sum())
    
    @property
    def total_fxt_rounded(self) -> NDArray[float64]:
        return around(self.total_fxt, self.structure.decimals)
    
    @cached_property
    def mean_of_t(self) -> float:
        return float(self.total_fxt / self.structure.total_freq)

    @property
    def mean_of_t_rounded(self) -> float:
        return around(self.mean_of_t, self.structure.decimals)
    
    @cached_property
    def mean_of_x(self) -> float:
        fx = self.structure.freq * self.structure.midpoints
        return fx.sum() / self.structure.total_freq

    @property
    def mean_of_x_rounded(self) -> float:
        return around(self.mean_of_x, self.structure.decimals)
    
    
@dataclass(slots=True)
class EDAGroupedDataParams:
    """
    Read-only facade over grouped data analysis.

    Instances should be treated as immutable after construction.

    Attributes
    ----------
    # ParamsData
    lower : NDArray[float64]
        Lower class limits.
    upper : NDArray[float64]
        Upper class limits.
    freq : NDArray[int64]
        Class frequencies.
    statistic : str
        Statistic type ("mean", "std", "percentiles").
    decimals : int
        Rounding precision for calculations.

    # ParamsMeanStd
    assumed_mean_param : int | float | "auto" | None
        Original input assumed mean.
    formula : str
        Formula selection ("x-a", "x/w-a", "(x-a)/w").
    formula_std : int
        Formula version for std calculation (1, 2, 3).

    # ParamsPercentiles
    percentiles_values : NDArray[float64]
        Percentiles requested.
    cumfreq_curve : bool
        Whether cumulative frequency curve is requested.
    x_values : NDArray[float64] | None
        Optional x-values for percentile calculation.

    # ParamsPlot
    fig_width : int
        Figure width in inches.
    fig_height : int
        Figure height in inches.
    line_color : str
        Line color for plots ("blue", "black", "green", "red", "purple", "orange").
    xaxis_orientation : int
        Degrees, from -180 to 180.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    grid : bool
        Whether to show grid.

    # Structure
    n : int
        Number of classes.
    class_width : float
        Width of each class.
    class_width_rounded : float
        Rounded class width.
    midpoints : NDArray[float64]
        Midpoints of classes.
    midpoints_rounded : NDArray[float64]
        Rounded midpoints.
    lower_bnds : NDArray[float64]
        Lower class boundaries.
    upper_bnds : NDArray[float64]
        Upper class boundaries.
    class_labels : list[str]
        "lower - upper" for each class.
    class_boundaries : list[str]
        "lower_bnd - upper_bnd" for each class.
    total_freq : int
        Sum of frequencies.

    # Transform
    assumed_mean : float | None
        Calculated assumed mean.
    assumed_mean_rounded : float | None
        Rounded assumed mean.
    is_calculate_assumed_mean : bool
        True if assumed_mean == "auto".
    assumed_mean_asterisk : float | None
        Assumed mean divided by class width (a*).
    assumed_mean_asteriks_rounded : float | None
        Rounded a*.
    tvalues : NDArray[float64]
        Transformed values according to formula.
    tvalues_rounded : NDArray[float64]
        Rounded t values.

    # Statistics
    fxt : NDArray[float64]
        t * freq.
    fxt_rounded : NDArray[float64]
        Rounded fxt.
    total_fxt : float
        Sum of fxt.
    total_fxt_rounded : float
        Rounded sum of fxt.
    mean_of_t : float
        Mean of t values.
    mean_of_t_rounded : float
        Rounded mean of t.
    mean_of_x : float
        Classical mean (Σfx / Σf).
    mean_x_rounded : float
        Rounded mean x.
    """
    data: ParamsData
    mean_std: ParamsMeanStd
    percentiles: ParamsPercentiles
    plot: ParamsPlot
    structure: GroupedStructure
    transform: GroupedTransform
    statistics: GroupedStatistics
    
    # ----------
    # ParamsData
    # ----------
    
    @property
    def lower(self) -> NDArray[float64]:
        return self.data.lower

    @property
    def upper(self) -> NDArray[float64]:
        return self.data.upper

    @property
    def freq(self) -> NDArray[int64]:
        return self.data.freq

    @property
    def statistic(self) -> str:
        return self.data.statistic
    
    @property
    def decimals(self) -> int:
        return self.data.decimals

    # -------------
    # ParamsMeanStd
    # -------------
    
    @property
    def assumed_mean_param(self):
        return self.mean_std.assumed_mean

    @property
    def formula(self):
        return self.mean_std.formula

    @property
    def formula_std(self):
        return self.mean_std.formula_std

    # -----------------
    # ParamsPercentiles
    # -----------------
    
    @property
    def percentiles_values(self):
        return self.percentiles.percentiles

    @property
    def cumfreq_curve(self):
        return self.percentiles.cumfreq_curve

    @property
    def x_values(self):
        return self.percentiles.x_values

    # ----------
    # ParamsPlot
    # ----------
    
    @property
    def fig_width(self):
        return self.plot.fig_width

    @property
    def fig_height(self):
        return self.plot.fig_height

    @property
    def line_color(self):
        return self.plot.line_color

    @property
    def xaxis_orientation(self):
        return self.plot.xaxis_orientation

    @property
    def x_label(self):
        return self.plot.x_label

    @property
    def y_label(self):
        return self.plot.y_label

    @property
    def grid(self):
        return self.plot.grid
    
    # ---------
    # Structure
    # --------- 
    
    @property
    def n(self) -> int:
        return self.structure.n
    
    @property
    def class_width(self) -> float:
        return self.structure.class_width

    @property
    def class_width_rounded(self) -> float:
        return self.structure.class_width_rounded
    
    @property
    def midpoints(self) -> NDArray[float64]:
        return self.structure.midpoints

    @property
    def midpoints_rounded(self) -> NDArray[float64]:
        return self.structure.midpoints_rounded
    
    @property
    def lower_bnds(self) -> NDArray[float64]:
        return self.structure.lower_bnds

    @property
    def upper_bnds(self) -> NDArray[float64]:
        return self.structure.upper_bnds

    @property
    def class_labels(self) -> list[str]:
        return self.structure.class_labels
    
    @property
    def class_boundaries(self) -> list[str]:
        return self.structure.class_boundaries
    
    @property
    def total_freq(self) -> int:
        return self.structure.total_freq
    
    # ---------
    # Transform
    # ---------
    
    @property
    def assumed_mean(self) -> float | None:
        return self.transform.assumed_mean

    @property
    def assumed_mean_rounded(self) -> float | None:
        return self.transform.assumed_mean_rounded

    @property
    def is_calculate_assumed_mean(self) -> bool:
        return self.transform.is_calculate_assumed_mean
    
    @property
    def assumed_mean_asterisk(self) -> float | None:
        return self.transform.assumed_mean_asterisk

    @property
    def assumed_mean_asteriks_rounded(self) -> float | None:
        return self.transform.assumed_mean_asterisk_rounded
    
    @property
    def tname(self) -> str:
        return self.transform.tname
    
    @property
    def tvalues(self) -> NDArray[float64]:
        return self.transform.tvalues

    @property
    def tvalues_rounded(self) -> NDArray[float64]:
        return self.transform.tvalues_rounded
    
    # ----------
    # Statistics
    # ----------
    
    @property
    def fxt(self) -> NDArray[float64]:
        return self.statistics.fxt

    @property
    def fxt_rounded(self) -> NDArray[float64]:
        return self.statistics.fxt_rounded
    
    @property
    def total_fxt(self) -> float:
        return self.statistics.total_fxt

    @property
    def total_fxt_rounded(self) -> float:
        return self.statistics.total_fxt_rounded

    @property
    def mean_of_t(self) -> float:
        return self.statistics.mean_of_t

    @property
    def mean_of_t_rounded(self) -> float:
        return self.statistics.mean_of_t_rounded

    @property
    def mean_of_x(self) -> float:
        return self.statistics.mean_of_x

    @property
    def mean_x_rounded(self) -> float:
        return self.statistics.mean_of_x_rounded