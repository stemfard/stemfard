from dataclasses import dataclass

from sympy import Expr, Symbol
from verifyparams import (
    verify_array_or_matrix, verify_boolean, verify_decimals,
    verify_int_or_float, verify_linear_system, verify_membership,
    verify_numeric, verify_numeric_arr, verify_str_identifier
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike


@dataclass(slots=True, frozen=True)
class ParseLinalgMatrixArithmetics:
    params: CoreParamsResult
    method: str
    A: SequenceArrayLike
    B: SequenceArrayLike
    steps_compute: bool = True
    steps_detailed: bool = True
    show_bg: bool = True
    param_names: tuple[str, str] = ("A", "B")
    decimals: int = 14
    
    def __repr__(self) -> str:
        return (f"ParseLinalgMatrixArithmetics(A={self.A.shape})")
    

def parse_linalg_matrix_arithmetics(
    method: str = "add",
    A: SequenceArrayLike = ...,
    B: SequenceArrayLike | None = None,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 14
) -> ParseLinalgMatrixArithmetics:
    
    raw_params = {}
    parsed_params = {}
    
    # method: str
       
    method = verify_membership(
        value=method,
        valid_items=[
            "add", "subtract", "multiply", "divide", "raise",
            "add_scalar", "subtract_scalar", "multiply_scalar",
            "divide_scalar", "raise_scalar",
            "matmul", 
        ],
        param_name="method"
    )
    
    # steps_compute: bool = True
    # steps_detailed: bool = True
    # show_bg: bool = True
    
    steps_compute = verify_boolean(steps_compute, default=True)
    steps_detailed = verify_boolean(steps_detailed, default=True)
    show_bg = verify_boolean(show_bg, default=True)
    
    # param_names: str = "x",
    
    try:
        A_name, b_name = param_names
    except (TypeError, ValueError):
        param_names = ("A", "B")
    
    A_name = verify_str_identifier(A_name)
    b_name = verify_str_identifier(b_name)
    param_names = (A_name, b_name)
    
    # decimals: int = 14
    
    decimals = verify_decimals(decimals, force_decimals=14)
       
    return ParseLinalgMatrixArithmetics(
        method=method,
        A=A,
        B=B,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        show_bg=show_bg,
        param_names=param_names,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )