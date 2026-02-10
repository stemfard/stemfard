from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InterpRequest:
    x: List[float]
    y: List[float]
    x0: float
    method: str
    expr_variable: str = "x"

    # Optional / method-specific
    yprime: Optional[List[float]] = None
    poly_order: Optional[int] = None
    qs_constraint: float = 0.0
    sat_type: str = "ax/(x+b)"

    # Presentation / numeric control
    decimals: int = 12
    symbolic: bool = False