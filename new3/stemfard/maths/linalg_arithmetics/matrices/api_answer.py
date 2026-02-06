from typing import Any

from sympy import Matrix

from stemfard.maths.linalg_arithmetics.matrices._common import elementwise_apply
from stemfard.core.serialization import serialize_result


def matrix_arithmetics_ans(
    A: Matrix,
    B: Matrix,
    operation: str,
    broadcast: bool = False,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = -1
) -> list[list[Any]]:
    """
    Perform matrix arithmetic operations on numeric or symbolic matrices.

    Parameters
    ----------
    A, B : np.ndarray or sympy.Matrix
        Input matrices.
    operation : str
        One of {"add", "subtract", "multiply", "divide", "raise", "matmul"}.
    broadcast : bool
        If True, allows elementwise broadcasting for matrices with
        compatible shapes.
    param_names : tuple[str, str]
        Names for error messages or display.
    decimals : int
        Number of decimal places for rounding numeric results.
        If -1, no rounding is applied. For symbolic matrices, evalf is
        used.

    Returns
    -------
    list[list[Any]]
        Resulting matrix as a Python nested list.
    """
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
        
        result = elementwise_apply(
            A=A,
            B=B,
            func=operations_map[operation],
            operation=operation,
            broadcast=broadcast,
            param_names=param_names
        )
    
    return serialize_result(result=result, decimals=decimals)