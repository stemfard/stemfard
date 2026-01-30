"""
Data models for grouped statistics calculations.

Immutability contract:
All classes in this module should be treated as immutable after
construction. Mutating attributes after initialization may lead to
undefined behavior, especially for cached properties.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import json
from typing import Any, Literal, Sequence

from numpy import allclose, around, float64, int64
from numpy.typing import NDArray
from stemcore import arr_to_numeric, numeric_format
from verifyparams import (
    verify_boolean, verify_decimals,
    verify_elements_in_range, verify_int, verify_int_or_float,
    verify_len_equal, verify_membership, verify_numeric, verify_string,
    verify_lower_lte_upper_arr
)

from stemfard.core.models import dataclass_to_dict, dict_to_dataclass
from stemfard.stats.stats_grouped_data.formulas import (
    ALLOWED_FORMULAS, ALLOWED_STATISTICS, FIGURE_COLORS
)


# =========================================
# PARAMETER OBJECTS (INPUT VALIDATION ONLY)
# =========================================


@dataclass(slots=True, frozen=True)
class ParamsData:
    """
    Container for grouped data parameters.

    Instances should be treated as immutable after construction.
    """
    lower: NDArray[float64] | Sequence[float]
    upper: NDArray[float64] | Sequence[float]
    freq: NDArray[int64] | Sequence[int]
    statistic: Literal["mean", "std", "percentiles"]
    decimals: int = 4
    
    snapshot: ParamsSnapshot = field(init=False)
    
    def __post_init__(self):
        """Validate the grouped data after initialization."""
        raw = {
            "lower": tuple(self.lower),
            "upper": tuple(self.upper),
            "freq": tuple(self.freq),
            "statistic": self.statistic,
            "decimals": self.decimals,
        }
        
        lower_arr: NDArray[float64] | Sequence[float] = arr_to_numeric(
            data=self.lower,
            param_name="lower",
            is_increasing=True
        )

        upper_arr: NDArray[float64] = arr_to_numeric(
            data=self.upper,
            is_increasing=True,
            param_name="upper"
        )
        
        verify_lower_lte_upper_arr(
            x=lower_arr,
            y=upper_arr,
            param_names=("lower", "upper")
        )
        
        freq_arr: NDArray[int64] = arr_to_numeric(
            data=self.freq,
            param_name="freq",
            all_integers=True,
            all_positive=True,
            allow_zero=True
        )
        
        verify_len_equal(
            lower_arr,
            upper_arr,
            freq_arr,
            param_names=["lower", "upper", "freq"]
        )
        
        verify_membership(
            value=self.statistic,
            valid_items=ALLOWED_STATISTICS,
            param_name="statistic"
        )
        
        verify_decimals(value=self.decimals)
        
        object.__setattr__(self, "lower", lower_arr)
        object.__setattr__(self, "upper", upper_arr)
        object.__setattr__(self, "freq", freq_arr)
        
        parsed = {
            "lower": lower_arr,
            "upper": upper_arr,
            "freq": freq_arr,
            "statistic": self.statistic,
            "decimals": self.decimals,
        }
        
        _params = ParamsSnapshot(raw=raw, parsed=parsed)
        object.__setattr__(self, "snapshot", _params)

    def __repr__(self) -> str:
        return (
            f"ParamsData(nclasses={len(self.lower)}, "
            f"total_freq={self.freq.sum()}, "
            f"statistic={self.statistic})"
        )


@dataclass(slots=True, frozen=True)
class ParamsMeanStd:
    """
    Parameters for mean and standard deviation calculations.

    Instances should be treated as immutable after construction.
    """
    assumed_mean: int | float | Literal["auto"] | None = None
    formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None
    var_formula: Literal[1, 2, 3] = 1
    
    snapshot: ParamsSnapshot = field(init=False)
    
    def __post_init__(self):
        """Validate mean parameters."""
        raw = {
            "assumed_mean": self._assumed_mean,
            "formula": self.formula,
            "var_formula": self.var_formula,
        }
        
        formula_str = self.formula if self._assumed_mean is not None else None
        
        if self._assumed_mean is not None and self._assumed_mean != "auto":
            verify_int_or_float(
                value=self._assumed_mean, param_name="assumed_mean"
            )
        
        if self.formula is not None:
            verify_membership(
                value=self.formula,
                valid_items=ALLOWED_FORMULAS,
                param_name="formula"
            )
        
        verify_membership(
            value=self.var_formula,
            valid_items=[1, 2, 3],
            param_name="var_formula"
        )
        
        object.__setattr__(self, "formula", formula_str)
        
        parsed = {
            "assumed_mean": self._assumed_mean,
            "formula": self.formula,
            "var_formula": self.var_formula,
        }
        
        _params = ParamsSnapshot(raw=raw, parsed=parsed)
        object.__setattr__(self, "snapshot", _params)
        
    def __repr__(self) -> str:
        return (
            f"ParamsMeanStd(assumed_mean={self._assumed_mean}, "
            f"formula={self.formula}, "
            f"var_formula={self.var_formula})"
        )


@dataclass(slots=True, frozen=True)
class ParamsPercentiles:
    f"""
    Parameters for percentile calculations on grouped data.

    Instances should be treated as immutable after construction.
    """
    percentiles: NDArray[float64] | Sequence[float]
    cumfreq_curve: bool = False
    x_values: NDArray[float64] | Sequence[float] | None = None
    
    snapshot: ParamsSnapshot = field(init=False)
    
    def __post_init__(self):
        """Validate percentile parameters."""
        raw = {
            "percentiles": (
                tuple(self.percentiles)
                if hasattr(self.percentiles, "__iter__")
                else self.percentiles            
            ),
            "cumfreq_curve": self.cumfreq_curve,
            "x_values": (
                tuple(self.x_values) if self.x_values is not None else None
            )
        }
        
        percentiles_arr = arr_to_numeric(
            data=self.percentiles, kind="array", param_name="percentiles"
        )
        verify_elements_in_range(
            data=percentiles_arr,
            lower=0,
            upper=100,
            param_name="percentiles"
        )
        
        cumfreq_curve_bool = verify_boolean(
            value=self.cumfreq_curve, default=False
        )
        
        x_values_arr = self.x_values
        if x_values_arr is not None:
            x_values_arr = arr_to_numeric(
                data=x_values_arr, kind="array", param_name="x_values"
            )
        
        object.__setattr__(self, "percentiles", percentiles_arr)
        object.__setattr__(self, "cumfreq_curve", cumfreq_curve_bool)
        object.__setattr__(self, "x_values", x_values_arr)
            
        parsed = {
            "percentiles": percentiles_arr,
            "cumfreq_curve": cumfreq_curve_bool,
            "x_values": x_values_arr,
        }
        
        _params =  ParamsSnapshot(raw=raw, parsed=parsed)
        object.__setattr__(self, "snapshot",_params)

    
    def __repr__(self) -> str:
        return (
            f"ParamsPercentiles(n={len(self.percentiles)}, "
            f"cumfreq_curve={self.cumfreq_curve})"
        )


@dataclass(slots=True, frozen=True)
class ParamsPlot:
    """
    Parameters controlling grouped-data plots.

    Instances should be treated as immutable after construction.
    """
    # raw
    fig_width: int = 6
    fig_height: int = 4
    line_color: Literal[
        "blue", "black", "green", "red", "purple", "orange"
    ] = "blue"
    xaxis_orientation: int = 0
    x_label: str = "Data"
    y_label: str = "Cumulative Frequency"
    grid: bool = True
    
    snapshot: ParamsSnapshot = field(init=False)
    
    def __post_init__(self):
        """Validate plot parameters."""
        raw = {
            "fig_width": self.fig_width,
            "fig_height": self.fig_height,
            "line_color": self.line_color,
            "xaxis_orientation": self.xaxis_orientation,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "grid": self.grid,
        }
        
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
        
        grid_bool = verify_boolean(value=self.grid, default=True)
        object.__setattr__(self, "grid", grid_bool)
        
        parsed = {
            "fig_width": self.fig_width,
            "fig_height": self.fig_height,
            "line_color": self.line_color,
            "xaxis_orientation": self.xaxis_orientation,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "grid": grid_bool,
        }
        
        _params = ParamsSnapshot(raw=raw, parsed=parsed)
        object.__setattr__(self, "snapshot", _params)
        
    def __repr__(self) -> str:
        return (
            f"ParamsPlot(fig={self.fig_width}x{self.fig_height}, "
            f"line_color={self.line_color}, "
            f"grid={self.grid})"
        )
        

#=================
# INPUT PARAMETERS
# ================

@dataclass(slots=True, frozen=True)
class ParamsSnapshot:
    """
    Captures user input and its verified/parsed/normalized form.
    """
    raw: dict[str, Any]
    parsed: dict[str, Any]
    
    def __repr__(self) -> str:
        return f"ParamsSnapshot(raw={self.raw}, parsed={self.parsed})"
    

# ====================
# STRUCTURE + GEOMETRY
# ====================


@dataclass(slots=True, frozen=True)
class GroupedStructure:
    """
    Represents the geometric structure of grouped data.

    Instances should be treated as immutable after construction.
    """
    lower: NDArray[float64]
    upper: NDArray[float64]
    freq: NDArray[int64]
    decimals: int
    params: ParamsSnapshot
    
    _delta: float = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        if len(self.lower) < 2:
            raise ValueError("'lower' requires at least two elements")
        
        if len(self.upper) < 2:
            raise ValueError("'upper' requires at least two elements")

        # if not allclose(self.lower[1:], self.upper[:-1]):
        #     raise ValueError("Class intervals must be contiguous")

        widths = self.upper - self.lower
        if not allclose(widths, widths[0]):
            raise ValueError("Class widths must be constant")
        
        # Precompute delta for class boundaries
        delta = (self.lower[1] - self.upper[0]) / 2.0
        object.__setattr__(self, "_delta", delta)
        
    @property
    def n(self) -> int:
        return len(self.lower)
    
    @cached_property
    def class_width(self) -> float:
        return float((self.upper - self.lower)[0])
    
    @cached_property
    def class_width_rounded(self) -> float:
        return around(self.class_width, self.decimals)
    
    @cached_property
    def midpoints(self) -> NDArray[float64]:
        return (self.lower + self.upper) / 2.0
    
    @cached_property
    def midpoints_rounded(self) -> NDArray[float64]:
        return around(self.midpoints, self.decimals)
    
    @cached_property
    def lower_bnds(self) -> NDArray[float64]:
        return self.lower - self._delta
    
    @cached_property
    def upper_bnds(self) -> NDArray[float64]:
        return self.lower + self._delta
    
    @cached_property
    def class_boundaries(self) -> list[str]:
        return [
            f"{numeric_format(around(a, self.decimals))} - "
            f"{numeric_format(around(b, self.decimals))}"
            for a, b in zip(self.lower_bnds, self.upper_bnds)
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
    def params_tuple(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.params.raw, self.params.parsed
    
    def __repr__(self) -> str:
        return (
            f"GroupedStructure(n={self.n}, "
            f"class_width={self.class_width_rounded}, "
            f"total_freq={self.total_freq})"
        )
    

# =========================================
# TRANSFORMATION LAYER (ASSUMED MEAN LOGIC)
# =========================================

@dataclass(slots=True, frozen=True)
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
    
    @cached_property
    def assumed_mean_rounded(self) -> float | None:
        if self._assumed_mean is None:
            return None
        return around(self._assumed_mean, self.structure.decimals)
    
    @property
    def is_calculate_assumed_mean(self) -> bool:
        """Check if assumed mean should be calculated automatically."""
        return self.params_mean.assumed_mean == "auto"
    
    @cached_property
    def assumed_mean_asterisk(self) -> float | None:
        """Calculate a* (assumed mean divided by class width)."""
        if self._assumed_mean is not None:
            return self._assumed_mean / self.structure.class_width
        return None
    
    @cached_property
    def assumed_mean_asterisk_rounded(self) -> float | None:
        """Calculate rounded a*."""
        a_star = self._assumed_mean_asterisk
        if a_star is not None:
            return around(a_star, self.structure.decimals)
        return None
    
    @property
    def tname(self) -> str:
        """Get the column name for t values."""
        if self._assumed_mean is None:
            return "x"
        else:
            return "t"
    
    @cached_property
    def tvalues(self) -> NDArray[float64]:
        """
        Compute transformed t-values based on the formula and assumed mean.
        """
        mid = self.structure.midpoints
        w = self.structure.class_width
        a = self._assumed_mean

        # If no assumed mean, t = midpoints
        if a is None:
            return mid

        # Precompute common intermediates -> avoids re-calculations
        x_minus_a = mid - a
        x_div_w = mid / w
        a_div_w = a / w
        x_minus_a_div_w = x_minus_a / w

        # Formula dispatch using dict
        formula_map = {
            "x-a": lambda: x_minus_a,
            "x/w-a": lambda: x_div_w - a_div_w,
            "(x-a)/w": lambda: x_minus_a_div_w,
        }

        formula_func = formula_map.get(self.params_mean.formula)
        if formula_func is None:
            raise RuntimeError(
                f"Invalid formula state: {self.params_mean.formula}"
            )

        return formula_func()

    @cached_property
    def tvalues_rounded(self) -> NDArray[float64]:
        return around(self.tvalues, self.structure.decimals)
    

# =============================
# STATISTICS ENGINE (PURE MATH)
# =============================

@dataclass(slots=True, frozen=True)
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
    
    @cached_property
    def fxt_rounded(self) -> NDArray[float64]:
        return around(self.fxt, self.structure.decimals)
    
    @cached_property
    def total_fxt(self) -> float:
        return float(self.fxt.sum())
    
    @cached_property
    def total_fxt_rounded(self) -> float:
        return around(self.total_fxt, self.structure.decimals)
    
    @cached_property
    def mean_of_t(self) -> float:
        return float(self.total_fxt / self.structure.total_freq)

    @cached_property
    def mean_of_t_rounded(self) -> float:
        return around(self.mean_of_t, self.structure.decimals)
    
    @cached_property
    def mean_of_x(self) -> float:
        fx = self.structure.freq * self.structure.midpoints
        return fx.sum() / self.structure.total_freq

    @cached_property
    def mean_of_x_rounded(self) -> float:
        return around(self.mean_of_x, self.structure.decimals)
    
    def __repr__(self) -> str:
        return (
            f"GroupedStatistics({self.structure.freq.sum()}, "
            f"mean_of_x={self.mean_of_x})"
        )


@dataclass(slots=True, frozen=True)
class EDAGroupedDataFacade:
    """
    Immutable container for grouped data parameters, with
    auto-initialized structure, transform, and statistics layers.
    """

    data: ParamsData
    mean_std: ParamsMeanStd
    percentiles: ParamsPercentiles
    plot: ParamsPlot
    structure: GroupedStructure
    transform: GroupedTransform
    statistics: GroupedStatistics
    
    # ------------------------
    # Dynamic serialization
    # ------------------------
    def to_dict(self) -> dict:
        """
        Convert the facade to a dict, recursively converting nested
        dataclasses. NumPy arrays are converted to lists for JSON
        compatibility.
        """
        return dataclass_to_dict({
            "data": self.data,
            "mean_std": self.mean_std,
            "percentiles": self.percentiles,
            "plot": self.plot
        })
        
    @classmethod
    def from_dict(cls, data: dict) -> EDAGroupedDataFacade:
        """
        Reconstruct the EDAGroupedDataFacade from a dictionary.
        Uses the factory `create` to ensure all derived layers are
        initialized.
        """
        # Convert dict -> dataclass objects
        data_obj = dict_to_dataclass(ParamsData, data["data"])
        mean_std_obj = dict_to_dataclass(ParamsMeanStd, data["mean_std"])
        percentiles_obj = dict_to_dataclass(
            ParamsPercentiles, data["percentiles"]
        )
        plot_obj = dict_to_dataclass(ParamsPlot, data["plot"])

        # Use the factory method to build full facade
        return cls.create(
            data=data_obj,
            mean_std=mean_std_obj,
            percentiles=percentiles_obj,
            plot=plot_obj
        )
        
    def to_json(self, **kwargs) -> str:
        """
        Serialize the facade to JSON. NumPy arrays become lists.
        kwargs are passed to json.dumps.
        """
        return json.dumps(self.to_dict(), indent=2, **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> EDAGroupedDataFacade:
        """
        Deserialize from JSON string to EDAGroupedDataFacade.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def create(
        cls,
        data: ParamsData,
        mean_std: ParamsMeanStd,
        percentiles: ParamsPercentiles,
        plot: ParamsPlot
    ) -> EDAGroupedDataFacade:
        """
        Factory method to build the full EDAGroupedDataFacade object.
        All derived attributes are fully initialized here.
        """
        
        structure = GroupedStructure(
            lower=data.lower,
            upper=data.upper,
            freq=data.freq,
            decimals=data.decimals,
            params=data.snapshot,
        )
        
        transform = GroupedTransform(
            structure=structure,
            params_mean=mean_std
        )
        
        statistics = GroupedStatistics(
            structure=structure,
            transform=transform
        )
        
        return cls(
            data=data,
            mean_std=mean_std,
            percentiles=percentiles,
            plot=plot,
            structure=structure,
            transform=transform,
            statistics=statistics
        )
        
    def __repr__(self) -> str:
        return (
            f"EDAGroupedDataFacade(nclasses={len(self.data.lower)}, "
            f"total_freq={self.data.freq.sum()}, "
            f"statistic={self.data.statistic})"
        )