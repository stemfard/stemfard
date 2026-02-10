from typing import Any

from numpy import around, ndarray
from pandas import DataFrame
from sympy import Matrix, Rational, Expr, Symbol, Float, Integer


def serialize_scalar(val: Any, decimals: int = -1) -> str | int | float:
    # SymPy Float
    if isinstance(val, Float):
        return around(float(val), decimals) if decimals >= 0 else float(val)

    # SymPy Integer
    if isinstance(val, Integer):
        return int(val)

    # Native numbers
    if isinstance(val, (int, float)):
        return around(val, decimals) if decimals >= 0 else val
    
    # Rational and General SymPy expressions (pi, sqrt(5), sin(x), etc.)
    if isinstance(val, (Rational, Expr, Symbol)):
        return float(val.evalf(decimals)) if decimals >= 0 else str(val)

    # Strings
    if isinstance(val, str):
        return val

    return str(val)


def serialize_matrix(
    result: Matrix, decimals: int = -1
) -> list[list[str | int | float]]:
    """
    Convert a SymPy Matrix to a JSON-serializable nested list:
    - Rational numbers → exact string (e.g., "1/3")
    - SymPy symbolic expressions → string (e.g., "x + 1/2")
    - SymPy Float → Python float
    - SymPy Integer → Python int
    - Python int / float → kept as-is
    """
    return [
        [serialize_scalar(val, decimals=decimals) for val in row]
        for row in result.tolist()
    ]


def serialize_result(
    result: Any, decimals: int = -1
) -> str | int | float | list[list[str | int | float]]:
    """
    Prepare any result for JSON serialization.
    - SymPy Matrix: convert to nested list, exact rationals as strings
    - NumPy / Pandas arrays: nested list, optionally rounded
    - Scalars / strings: as-is
    """
    # SymPy Matrix
    if isinstance(result, Matrix):
        return serialize_matrix(result)

    # NumPy array
    if isinstance(result, ndarray):
        if decimals >= 0:
            try:
                result = around(result.astype(float), decimals)
            except (TypeError, ValueError, AttributeError):
                pass
        return result.tolist()

    # Pandas DataFrame
    if isinstance(result, DataFrame):
        if decimals >= 0:
            result = result.round(decimals)
        return result.to_numpy().tolist()

    # Scalars
    return serialize_scalar(val=result, decimals=decimals)