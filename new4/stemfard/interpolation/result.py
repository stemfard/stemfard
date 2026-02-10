from dataclasses import dataclass
from typing import Any

@dataclass
class InterpResult:
    table: Any          # e.g., pandas DataFrame
    polynomial: Any     # str or sympy.Expr
    value_at_x0: Any    # float or sympy.Float
    derivative: Any = None
    derivative_at_x0: Any = None
    plot: Any = None
