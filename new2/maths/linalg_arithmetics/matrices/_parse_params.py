from dataclasses import dataclass
from typing import Any

from sympy import Matrix
from verifyparams import (
    verify_array_or_matrix, verify_boolean, verify_decimals,
    verify_membership_iterable, verify_param_names
)

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core._parse_compute import parse_compute

VALID_MATRIX_ARITHMETICS = [
    "add", "subtract", "multiply", "divide", "raise", "matmul"
]


@dataclass(slots=True, frozen=True)
class MatrixArithmeticsParsedParams:
    A: Matrix
    B: Matrix
    compute_apis: list[str]
    param_names: tuple[str, str]
    steps_compute: bool
    steps_detailed: bool
    steps_bg: bool
    decimals: int
    params: CoreParamsResult
    

class MatrixArithmeticsParsedParameters:
    """Handles input parsing and validation"""
    
    @staticmethod
    def parse(
        A: SequenceArrayLike,
        B: SequenceArrayLike,
        compute_apis: list[str],
        param_names: tuple[str, str],
        steps_compute: bool,
        steps_detailed: bool,
        steps_bg: bool,
        decimals: int
    ) -> MatrixArithmeticsParsedParams:
        """Parse and validate input parameters"""
        
        parsed_params = parse_linalg_matrix_arithmetics(
            A=A,
            B=B,
            compute_apis=compute_apis,
            param_names=param_names,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            steps_bg=steps_bg,
            decimals=decimals
        )
        
        return parsed_params
    

def parse_linalg_matrix_arithmetics(
    A: SequenceArrayLike,
    B: SequenceArrayLike,
    compute_apis: list[str],
    param_names: tuple[str, str],
    steps_compute: bool,
    steps_detailed: bool,
    steps_bg: bool,
    decimals: int
) -> MatrixArithmeticsParsedParams:
    
    raw_params: dict[str, Any] = {}
    parsed_params: dict[str, Any] = {}
    
    raw_params.update({
        "A": A,
        "B": B,
        "compute_apis": compute_apis,
        "param_names": param_names,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "param_names": param_names,
        "decimals": decimals
    })
    
    A = verify_array_or_matrix(A=A, to_matrix=True, param_name="A")
    B = verify_array_or_matrix(A=B, to_matrix=True, param_name="B")
    
    compute_apis = parse_compute(
        param=compute_apis, valid=VALID_MATRIX_ARITHMETICS
    )
    compute_apis = verify_membership_iterable(
        value=compute_apis,
        valid_items=VALID_MATRIX_ARITHMETICS,
        param_name="compute_apis"
    )
    
    param_names = verify_param_names(param_names, n=2)
    steps_compute = verify_boolean(steps_compute, default=True)
    steps_detailed = verify_boolean(steps_detailed, default=True)
    steps_bg = verify_boolean(steps_bg, default=True)
    
    decimals = verify_decimals(decimals)
    
    parsed_params.update({
        "A": A,
        "B": B,
        "compute_apis": compute_apis,
        "param_names": param_names,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals
    })
       
    return MatrixArithmeticsParsedParams(
        A=A,
        B=B,
        compute_apis=compute_apis,
        param_names=param_names,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )