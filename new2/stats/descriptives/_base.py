from dataclasses import dataclass
from functools import cached_property
from typing import Literal
import warnings

from numpy import (
    around, count_nonzero, float64, median, ndarray, percentile, prod, sort, sqrt
)
from numpy.typing import NDArray
from pandas import Series
from scipy import stats
from stemcore import arr_to_numeric
from verifyparams import verify_int_or_float

from stemfard.core._type_aliases import (
    ScalarSequenceArrayLike, SequenceArrayLike
)
from stemfard.stats.descriptives._parse_params import parse_descriptives


class MultimodalDataError(ValueError):
    """
    Raised when data are multimodal but a single mode is requested.
    """
    pass

class MultimodalSingleModeWarning(UserWarning):
    """
    Filter warnings can be done as follows.
    warnings.filterwarnings("ignore", category=MultimodalSingleModeWarning)
    """
    pass

class GeometricMeanDataError(ValueError):
    """Raised when data contains a zero."""
    pass


@dataclass(slots=True, frozen=True)
class StatsModeResult:
    mode: float
    count: int

    def _repr__(self) -> str:
        return f"StatsModeResult(mode={self.mode}, count={self.count})"
    

@dataclass(slots=True, frozen=True)
class StatsMultiModeResult:
    """Result of a multimodal computation."""
    modes: NDArray[float64]
    count: int

    @property
    def n_modes(self) -> int:
        """Return the number of modes."""
        return self.modes.size

    def _len__(self) -> int:
        """Return the number of modes."""
        return self.modes.size

    def _iter__(self):
        """Iterate over all modes."""
        return iter(self.modes)

    def _repr__(self) -> str:
        return f"StatsMultiModeResult(modes={self.modes}, count={self.count})"
    

@dataclass(slots=True, frozen=True)
class StatsRangeResult:
    min: float
    max: float
    range: float
    
    def _repr__(self) -> str:
        return (
            f"StatsRangeResult("
            f"min={self.min}, "
            f"max={self.max}, "
            f"range={self.range})"
        )

    
@dataclass(slots=True, frozen=True)
class StatsMeanConfidenceInterval:
    lower: float
    upper: float
    conf_level: float = 0.95

    def _repr__(self) -> str:
        return (
            f"StatsMeanConfidenceInterval("
            f"lower={self.lower}, "
            f"upper={self.upper}, "
            f"conf_level={self.conf_level})"
        )

    
@dataclass(slots=True, frozen=True)
class StatsKurtResult:
    kurt: float
    kurt_minus_3: float
    
    def _repr__(self) -> str:
        return (
            f"StatsKurtResult("
            f"kurt={self.kurt}, "
            f"kurt_minus_3={self.kurt_minus_3})"
        )
    

class BaseDescriptives:
    """Base class for descriptive statistics with cached properties."""

    def __init__(
        self,
        statistic: str = "mean",
        data: SequenceArrayLike | None = None,
        freq: SequenceArrayLike | None = None,
        assumed_mean: int | float | Literal["auto"] | None = None,
        conf_level: float = 0.95,
        var_formula: Literal[1, 2, 3] = 1,
        ddof: int = 1,
        p: int | float | None = None,
        quartiles_use_linear: bool = False,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        show_bg: bool = True,
        param_name: str = "x",
        decimals: int = 4
    ):
        parsed_params = parse_descriptives(
            statistic=statistic,
            data=data,
            freq=freq,
            assumed_mean=assumed_mean,
            conf_level=conf_level,
            var_formula=var_formula,
            ddof=ddof,
            p=p,
            quartiles_use_linear=quartiles_use_linear,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            show_bg=show_bg,
            param_name=param_name,
            decimals=decimals
        )
        
        self.is_calculate_amean = assumed_mean == "auto"
        
        self.statistic = parsed_params.statistic
        self._data = parsed_params.data
        self.freq = parsed_params.freq
        self.assumed_mean = parsed_params.assumed_mean
        self.conf_level = parsed_params.conf_level
        self.var_formula = parsed_params.var_formula
        self.ddof = parsed_params.ddof
        self.p = parsed_params.p
        self.quartiles_use_linear = parsed_params.quartiles_use_linear
        self.steps_compute = parsed_params.steps_compute
        self.steps_detailed = parsed_params.steps_detailed
        self.show_bg = parsed_params.show_bg
        self.param_name = parsed_params.param_name
        self.decimals = parsed_params.decimals
        
        self.params = parsed_params.params
        
    # ----- Common -----
    
    @property
    def data_rnd(self) -> NDArray[float64]:
        return around(self._data, self.decimals)

    @cached_property
    def data_sorted(self) -> NDArray[float64]:
        return sort(self._data)

    @cached_property
    def mean_dev(self) -> NDArray[float64]:
        return self._data - self.mean
    
    # ----- Location -----
    
    @cached_property
    def mean(self) -> float:
        return float(self._data.mean())
    
    @property
    def mean_rnd(self) -> float:
        return float(around(self.mean, self.decimals))
    
    @cached_property
    def median(self) -> float:
        return float(median(self._data))
    
    @cached_property
    def _get_modes(self) -> NDArray[float64]:
        return Series(self._data).mode().to_numpy()
    
    @cached_property
    def mode(self) -> StatsModeResult:
        """
        Compute the statistical mode.

        Notes
        -----
        - Returns a **single mode**.
        - Raises an error if the data are **multimodal**.
        - Use ``multimode`` to retrieve all modes for multimodal data.
        """
        count = self._get_modes.size
        if count > 1:
            raise MultimodalDataError(
                f"Data has multiple modes ({count}); use `stats_multimode()` "
                "to access all modes"
            )

        # Safe: exactly one mode exists
        result = stats.mode(self._data, keepdims=False)
        return StatsModeResult(
            mode=float(result.mode),
            count=int(result.count)
        )

    @cached_property
    def multimode(self) -> StatsMultiModeResult:
        """
        Compute all statistical modes.

        Notes
        -----
        - Uses ``scipy.stats.multimode``.
        - Returns all values with maximum frequency.
        """
        modes_arr = self._get_modes
        count = count_nonzero(self._data == modes_arr[0])
        
        if modes_arr.size == 1:
            warnings.warn(
                "You requested `stats_multimode()`, but your data contains "
                "a single mode, consider using `stats_mode()` instead",
                category=MultimodalSingleModeWarning,
                stacklevel=2
            )
        
        return StatsMultiModeResult(
            modes=modes_arr,
            count=count
        )
    
    @cached_property
    def harmonic(self) -> float:
        return float(self.n / (1 / self._data).sum())
    
    @cached_property
    def geometric(self) -> float:
        if (self._data <= 0).any():
            raise GeometricMeanDataError(
                "Geometric mean requires all values to be greater than zero."
            )
        return float(prod(self._data) ** (1 / self.n))
    
    # ----- Distribution -----
    
    @cached_property
    def min(self) -> float:
        return float(self._data.min())

    @cached_property
    def max(self) -> float:
        return float(self._data.max())
    
    @cached_property
    def range(self) -> StatsRangeResult:
        return StatsRangeResult(
            min=self.min,
            max=self.max,
            range=self.max - self.min
        )
    
    # no cached
    def var(self, ddof: int = 1) -> float:
        return float(self._data.var(ddof=ddof))
    
    # no cached
    def std(self, ddof: int = 1) -> float:
        return float(sqrt(self.var(ddof=ddof)))
    
    @cached_property
    def iqr(self) -> float:
        return float(self.p75 - self.p25)
    
    @cached_property
    def iqd(self) -> float:
        return float(self.iqr / 2)
    
    @cached_property
    def mean_abs_dev(self) -> float:
        return float(abs(self.mean_dev).mean())
    
    @property
    def mean_abs_dev_rnd(self) -> float:
        return float(around(self.mean_abs_dev, self.decimals))
    
    @cached_property
    def median_abs_dev(self) -> float:
        return float(median(abs(self._data - self.median)))
    
    @property
    def median_abs_dev_rnd(self) -> float:
        return float(around(self.median_abs_dev, self.decimals))
    
    # ----- Position -----
    
    # no cached
    def percentiles(self, p: ScalarSequenceArrayLike) -> float:
        if isinstance(p, (int, float)):
            p: int | float = verify_int_or_float(p, param_name="p")
        elif isinstance(p, (list, ndarray)):
            p: NDArray = arr_to_numeric(
                data=p,
                kind="array",
                param_name="data"
            )
            
        return percentile(a=self._data, q=p)
    
    @cached_property
    def p25(self) -> float:
        return float(percentile(a=self._data, q=25))
    
    @property
    def p25_rnd(self) -> float:
        return float(around(self.p25, self.decimals))
    
    @cached_property
    def p50(self) -> float:
        return float(percentile(a=self._data, q=50))
    
    @cached_property
    def p50_rnd(self) -> float:
        return float(around(self.p50, self.decimals))
    
    @cached_property
    def p75(self) -> float:
        return float(percentile(a=self._data, q=75))
    
    @cached_property
    def p75_rnd(self) -> float:
        return float(around(self.p75, self.decimals))

    
    # no cache
    def zscores(self, ddof=0) -> float:
        return self.mean_dev / self._data.std(ddof=ddof)
    
    # ----- Shape -----

    @cached_property
    def skew(self) -> float:
        _numer = (1 / self.n) * (self.mean_dev ** 3).sum()
        _denom = ((1 / self.n) * (self.mean_dev) ** 2).sum() ** (3 / 2)
        return float(_numer / _denom)
    
    @cached_property
    def skew_rnd(self) -> float:
        return float(around(self.skew, self.decimals))

    @cached_property
    def kurt(self) -> StatsKurtResult:
        _numer = (1 / self.n) * (self.mean_dev ** 4).sum()
        _denom = ((1 / self.n) * (self.mean_dev ** 2).sum()) ** 2 
        _kurtosis = float(_numer / _denom)
        return StatsKurtResult(
            kurt=_kurtosis,
            kurt_minus_3=_kurtosis - 3
        )
        
    # ----- Others -----
    
    @cached_property
    def n(self) -> int:
        return self._data.size
    
    @cached_property
    def total(self) -> float:
        return float(self._data.sum())
    
    @property
    def total_rnd(self) -> float:
        return float(around(self.total, self.decimals))
    
    
    
    # no cached
    def mean_ci(
        self,
        ddof: int = 1,
        conf_level: float = 0.95
    ) -> StatsMeanConfidenceInterval:
        "Mean confidence of interval"
        alpha = 1.0 - conf_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=self.n - 1)
        stderror = self._data.std(ddof=ddof) / sqrt(self.n)
        margin = t_critical * stderror
        
        return StatsMeanConfidenceInterval(
            lower=float(self.mean - margin),
            upper=float(self.mean + margin),
            conf_level=conf_level
        )
        
    # no cached
    def sem(self, ddof: int = 1) -> float:
        return float(self._data.std(ddof=ddof) / sqrt(self.n))
    
    # no cached
    def cv(self, ddof: int = 1) -> float:
        return float(self._data.std(ddof=ddof) / self.mean)
    
    
    # ----- Assumed mean calculations
   
    @property
    def assumed_mean_rnd(self) -> float | None:
        if self.assumed_mean is None:
            return None
        return float(around(self.assumed_mean, self.decimals))
    
    @property
    def tname(self) -> str:
        """Get the column name for t values."""
        return "x" if self.assumed_mean is None else "t"
    
    @cached_property
    def tvalues(self) -> NDArray[float64]:
        """
        Compute transformed t-values based on the formula and assumed mean.
        """
        if self.assumed_mean is None:
            return self._data
        return self._data - self.assumed_mean

    @property
    def tvalues_rnd(self) -> NDArray[float64]:
        return around(self.tvalues, self.decimals)
    
    @cached_property
    def total_tvalues(self) -> NDArray[float64]:
        return float(self.tvalues.sum())
    
    @property
    def total_tvalues_rnd(self) -> NDArray[float64]:
        return float(around(self.total_tvalues, self.decimals))
    
    @cached_property
    def mean_tvalues(self) -> NDArray[float64]:
        return float(self.tvalues.mean())
    
    @property
    def mean_tvalues_rnd(self) -> NDArray[float64]:
        return float(around(self.mean_tvalues, self.decimals))
    
    @cached_property
    def fxt(self) -> NDArray[float64]:
        if self.assumed_mean is None:
            return self.freq * self._data
        return self.freq * self.tvalues
    
    @cached_property
    def fxt_rnd(self) -> NDArray[float64]:
        if self.freq is None:
            return self.freq
        return around(self.fxt, self.decimals)
    
    @cached_property
    def total_fxt(self) -> float:
        return float(self.fxt.sum())
    
    @property
    def total_fxt_rnd(self) -> float:
        return float(around(self.total_fxt, self.decimals))
    
    @cached_property
    def mean_fxt(self) -> float:
        """Mean of ft -> sum(ft) / sum(f). DO NOT USE `self.fxt.mean()`"""
        return float(self.total_fxt / self.total_freq)
    
    @property
    def mean_fxt_rnd(self) -> float:
        return float(around(self.mean_fxt, self.decimals))
    
    @cached_property
    def total_freq(self) -> float:
        return self.freq.sum()