from typing import Any

from stemfard.core.rounding import fround
from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math
from stemfard.core.constants import StemConstants


def present_result(
    result: Any,
    decimals: int = -1,
    add_equal_sign: bool = True,
    n: int | None = 2,
    color: str = "red"
) -> str:
    """
    Final results for step by step procedure.
    """
    result_float = fround(x=result, decimals=decimals)
    add_equal_sign = f" = " if add_equal_sign else ""
    
    if color:
        result_float_latex = tex_to_latex(M=result_float)
        result_float = str_color_math(
            value=f"{add_equal_sign}{result_float_latex}"
        )
    
    if n:
        result_float = result_float + StemConstants.CHECKMARK * n
    
    return result_float