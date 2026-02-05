from typing import Literal

from stemfard.core._html import html_bg_level1
from stemfard.core._parse_compute import parse_compute
from stemfard.core._type_aliases import ScalarSequenceArrayLike, SequenceArrayLike
from stemfard.maths.linalg_arithmetics.matrices.api_steps import MatrixArithmetics
from stemfard.maths.linalg_arithmetics.matrices._parse_params import VALID_MATRIX_ARITHMETICS


def linalg_matrix_arithmetics(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    compute_apis: Literal[
        "add", "subtract", "multiply", "divide", "raise", "matmul"   
    ] = "add",
    param_names: tuple[str, str] = ("A", "B"),
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "A": A,
        "B": B,
        "compute_apis": compute_apis,
        "param_names": param_names,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    steps_mathjax: list[str] = []
    counter = 0
    compute_apis = parse_compute(
        param=compute_apis, valid=VALID_MATRIX_ARITHMETICS
    )
    
    result = MatrixArithmetics(**params)
    
    if not steps_compute:
        return "Return answer only, no steps"
    
    
    if "add" in compute_apis:
        counter += 1
        title = f"{counter}) Matrix addition"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_add_latex)
        
        
    if "subtract" in compute_apis:
        counter += 1
        title = f"{counter}) Matrix subtraction"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_subtract_latex)
        
        
    if "multiply" in compute_apis:
        counter += 1
        title = f"{counter}) Element-wise matrix multiplication"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_multiply_latex)
        
        
    if "divide" in compute_apis:
        counter += 1
        title = f"{counter}) Element-wise matrix division"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_divide_latex)
        
        
    if "raise" in compute_apis:
        counter += 1
        title = f"{counter}) Element-wise matrix power"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_raise_latex)
        
        
    if "matmul" in compute_apis:
        counter += 1
        title = f"{counter}) Matrix multiplication"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(result.linalg_matmul_latex)
        
    return steps_mathjax