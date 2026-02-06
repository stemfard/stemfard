from dataclasses import dataclass

from numpy import sort
from stemcore import arr_to_numeric
from verifyparams import verify_decimals, verify_numeric, verify_numeric_arr


@dataclass(slots=True, frozen=True)
class StatsModeResult:
    mode: float
    count: int


class BaseIVPs:
    """Base class for descriptive statistics with cached properties."""

    def __init__(
        self,
        fexpr,
        time_span,
        decimals
    ):
        self.fexpr = fexpr
        self._time_span = verify_numeric_arr(
            time_span,
            n=2,
            all_positive=True,
            allow_zero=True,
            param_name="time_span"
        )
        self._t0, self._t1 = sort(self._time_span)
        decimals = verify_decimals(decimals)