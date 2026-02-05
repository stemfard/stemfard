from typing import Any

from numpy import array, float64
from sympy import Matrix

from stemfard.maths.linalg_arithmetics.matrices._common import (
    align_matrices, elementwise_apply
)
from stemfard.core.rounding import round_dp


def matrix_arithmetics_ans(
    A: Matrix, B: Matrix, operation: str, decimals: int
) -> Any:

    A, B = align_matrices(A, B)
    
    if operation == "matmul":
        result = A @ B
    else:
        operations_map = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b,
            "raise": lambda a, b: a ** b,
        }
        
        result = elementwise_apply(A=A, B=B, func=operations_map[operation])
    
    if decimals >= 0:
        try:
            result = round_dp(array(result, dtype=float), decimals=decimals)
        except:
            result = result.evalf(decimals)
    
    return result.tolist()