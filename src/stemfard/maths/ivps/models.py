"""
Data models for initial value problems
"""

from dataclasses import dataclass
from typing import Callable

from numpy import ceil
from stemcore import arr_to_numeric
from sympy import Expr
from verifyparams import verify_boolean, verify_decimals, verify_int, verify_int_or_float, verify_lower_lte_upper, verify_numeric, verify_step_size


@dataclass(slots=True)
class IVPsModel:
    """Container for parameters."""
    method: str
    odeqtn: str
    exactsol: str | Expr | Callable | None
    eqtn_vars: tuple[str, str]
    derivs: tuple[str, ...] | None
    start_values_or_method: str
    time_span: tuple[float, float]
    y0: int | float = None
    stepsize: float | tuple[float, ...] | None
    nsteps: int | None
    tolerance: float
    compute_first_k: int | None
    decimals: int
    
    def __post_init__(self):
        """Validate the grouped data after initialization."""
        # method: str
        # odeqtn
        # exactsol
        # eqtn_vars
        # derivs
        if isinstance(str, self.start_values_or_method):
            start_method = self.start_values_or_method
            start_values = None
        else:
            start_method = None
            start_values = arr_to_numeric(
                data=self.start_values_or_method,
                kind="array",
                dtype=float,
                param_name="start_values_or_method"
            )
            
        # VERIFY_LENGTH
        a = verify_int_or_float(
            value=self.time_span[0],
            param_name="time_span[0]"
        )
        b = verify_int_or_float(
            value=self.time_span[1],
            param_name="time_span[1]"
        )
        self.time_span = (a, b)
        
        verify_lower_lte_upper(
            lower=a,
            upper=b,
            param_names=["time_span[0]", "time_span[1]"]
        )
        
        self.y0 = verify_int_or_float(value=self.y0, param_name="y0")
        self.stepsize = verify_int_or_float(
            value=self.stepsize, param_name="stepsize"
        )
        self.nsteps = verify_int(value=self.nsteps, param_name="nsteps")
        
        self.tolerance = verify_numeric(
            value=self.tolerance,
            limits=[0, 1],
            boundary="exclusive",
            param_name="tolerance"
        )
        
        if self.nsteps is None:
            m = ceil((b - a) / self.stepsize)
        else:
            m = self.nsteps
        
        self.compute_first_k = verify_numeric(
            value=self.compute_first_k,
            limits=[1, m],
            boundary="inclusive",
            param_name="compute_first_k"
        )
        
        self.decimals = verify_decimals(value=self.decimals)
        
    
    @property
    def n(self) -> int:
        """Number of classes."""
        return len(self.lower)
    
    def __repr__(self) -> str:
        return f"IVPsModel(eqtn=*)"