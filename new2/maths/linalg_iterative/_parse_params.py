from dataclasses import dataclass
from typing import Literal
import warnings

from numpy import identity, zeros_like
from sympy import Matrix
from verifyparams import (
    verify_array_or_matrix, verify_boolean, verify_decimals,
    verify_int_or_float, verify_linear_system, verify_membership,
    verify_numeric, verify_numeric_arr, verify_str_identifier
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import Array2DLike, SequenceArrayLike
from stemfard.core.warning_msgs import MultipleSpecifiedWarning


@dataclass(slots=True, frozen=True)
class ParseLinalgIterative:
    params: CoreParamsResult
    method: str
    steps_method: str
    A: Array2DLike
    b: SequenceArrayLike
    x0: SequenceArrayLike | None = None
    fvars: list[str] | None = None
    C: Array2DLike | Matrix | None = None
    relax_param: float | None = None
    atol: float | None = 1e-6
    rtol: float | None = None
    maxit: int = 50
    steps_compute: bool = True
    steps_detailed: bool = True
    steps_bg: bool = True
    param_names: tuple[str, str] = ("A", "b")
    decimals: int = 14
    
    def __repr__(self) -> str:
        return (f"ParseLinalgIterative(A={self.A.shape}, b={len(self.b)})")
    

def parse_linalg_iterative(
    method: str = "jacobi",
    steps_method: Literal["algebra", "matrix"] = "algebra",
    A: Array2DLike = ...,
    b: SequenceArrayLike = ...,
    x0: SequenceArrayLike | None = None,
    fvars: list[str] | None = None,
    C: Array2DLike | Matrix | None = None,
    relax_param: float | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = None,
    maxit: int = 50,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    param_names: tuple[str, str] = ("A", "b"),
    decimals: int = 14
) -> ParseLinalgIterative:
    
    raw_params = {}
    parsed_params = {}
    
    # method: str
    
    if method in ["conjugate", "gradient"]:
        method = "conjugate-gradient"
        
    if method in ["gauss", "seidel"]:
        method = "gauss-seidel"
        
    method = verify_membership(
        value=method,
        valid_items=["jacobi", "gauss-seidel", "sor", "conjugate-gradient"],
        param_name="method"
    )
    
    # steps_method: str
    
    steps_method = verify_membership(
        value=steps_method,
        valid_items=["algebra", "matrix"],
        param_name="steps_method"
    )
    
    # A: Array2DLike
    # b: SequenceArrayLike
    
    A, b = verify_linear_system(
        A, b, require_square=True, param_names=("A", "b")
    )
    
    nrows, ncols = A.shape
    
    # x0: SequenceArrayLike | None = None
    
    if x0 is None:
        x0 = zeros_like(b)
    else:
        x0 = verify_numeric_arr(x0, n=ncols, param_name="x0")
        x0 = x0.reshape(-1, 1)
    
    # fvars: list[str] | None = None
    
    if fvars is None:
        fvars = [f"x_{{{index + 1}}}" for index in range(ncols)]
    else:
        n_fvars = len(fvars)
        if n_fvars != ncols:
            raise ValueError(
                f"Expected 'fvars' to have {ncols} elements, got {n_fvars}"
            )

    # C: Array2DLike | Matrix | None = None
    
    if method == "conjugate-gradient":
        if C is None:
            C = identity(ncols)
        else:
            C = verify_array_or_matrix(
                C, nrows=nrows, ncols=ncols, is_square=True, param_name="C"
            )
    else:
        C = None
    
    # relax_param: float | None = None
    
    if method == "sor":
        if relax_param is not None:
            relax_param = verify_int_or_float(
                relax_param,
                is_positive=True,
                allow_zero=False,
                param_name="relax_param"
            )
    else:
        relax_param = None
    
    # atol: float | None = 1e-6
    # rtol: float | None = None
    
    if atol is None and rtol is None:
        raise ValueError("You must specify either 'atol' or 'rtol'")
        
    if atol is not None:
        atol = verify_numeric(
            atol, limits=[0, 1], boundary="exclusive", param_name="atol"
        )
        
    if rtol is not None:
        rtol = verify_numeric(
            rtol, limits=[0, 1], boundary="exclusive", param_name="rtol"
        )
    
    if atol and rtol:
        if atol <= rtol:
            rtol = None
            tol_used = "'rtol' was discarded and 'atol'"
        else:
            atol = None
            tol_used = "'atol' was discarded and 'rtol'"
        
        warnings.warn(
            "You specified both 'atol' and 'rtol'; "
            f"{tol_used} (the smaller one) used instead",
            category=MultipleSpecifiedWarning,
            stacklevel=3
        )
    
    
    # maxit: int = 50
    
    maxit = verify_numeric(
        maxit, limits=[1, 50], boundary="inclusive", param_name="maxit"
    )
    
    # steps_compute: bool = True
    # steps_detailed: bool = True
    # steps_bg: bool = True
    
    steps_compute = verify_boolean(steps_compute, default=True)
    steps_detailed = verify_boolean(steps_detailed, default=True)
    steps_bg = verify_boolean(steps_bg, default=True)
    
    # param_names: str = "x",
    
    try:
        A_name, b_name = param_names
    except (TypeError, ValueError):
        param_names = ("A", "b")
    
    A_name = verify_str_identifier(A_name)
    b_name = verify_str_identifier(b_name)
    param_names = (A_name, b_name)
    
    # decimals: int = 14
    
    decimals = verify_decimals(decimals, force_decimals=14)
       
    return ParseLinalgIterative(
        method=method,
        steps_method=steps_method,
        A=A,
        b=b,
        fvars=fvars,
        x0=x0,
        C=C,
        relax_param=relax_param,
        atol=atol,
        rtol=rtol,
        maxit=maxit,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        param_names=param_names,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )