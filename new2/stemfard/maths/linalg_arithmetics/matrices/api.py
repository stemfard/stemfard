from dataclasses import dataclass
from typing import Literal

from stemfard.core._html import html_bg_level1
from stemfard.core._type_aliases import ScalarSequenceArrayLike, SequenceArrayLike
from stemfard.maths.linalg_arithmetics.matrices.api_steps import MatrixArithmetics
from stemfard.maths.linalg_arithmetics.matrices._parse_params import parse_linalg_matrix_arithmetics
from stemfard.maths.linalg_arithmetics.matrices._docstring import (
    COMMON_DOCSTRING_PARAMS,
    inject_common_params
)


@dataclass(frozen=True, slots=True)
class BuildSteps:
    title: str
    steps: list[str]


@inject_common_params(COMMON_DOCSTRING_PARAMS)
def linalg_matrix_arithmetics(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    compute_apis: str | tuple[
        Literal["add", "subtract", "multiply", "divide", "raise", "matmul"], ...
    ] = "add",
    param_names: tuple[str, str] = ("A", "B"),
    result_name: str = "M",
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 12
) -> list[str] | MatrixArithmetics:
    """
    Perform matrix arithmetic computations.

    Parameters
    ----------
    A : SequenceArrayLike
        First matrix or array-like input.
    B : ScalarSequenceArrayLike
        Second matrix or array-like input.
    compute_apis : str | tuple of str
        Operation(s) to compute. Can be a single string (e.g., "add") or a tuple of
        operations. Allowed values: "add", "subtract", "multiply", "divide", "raise", "matmul".
    param_names : tuple[str, str], default=("A", "B")
        Names for the input parameters, used in computation display.
    {common_params}

    Returns
    -------
    list[str] | MatrixArithmetics
        - If `steps_compute` is True, returns a list of LaTeX/HTML steps for each requested operation.
        - If `steps_compute` is False, returns a `MatrixArithmetics` object with computed results.

    Examples
    --------
    >>> import numpy as np
    >>> from stemfard.maths.linalg_arithmetics.matrices import linalg_matrix_arithmetics
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> steps = linalg_matrix_arithmetics(A, B, compute_apis=("add", "multiply"))
    >>> result_obj = linalg_matrix_arithmetics(A, B, steps_compute=False)
    """    
    # Validation
    
    parsed = parse_linalg_matrix_arithmetics(
        A=A,
        B=B,
        compute_apis=compute_apis,
        param_names=param_names,
        result_name=result_name,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        decimals=decimals
    )
    
    # Instantiate computation object
    
    result = MatrixArithmetics(
        A=parsed.A,
        B=parsed.B,
        compute_apis=parsed.config.compute_apis,
        param_names=parsed.config.param_names,
        result_name=parsed.config.result_name,
        steps_compute=parsed.config.steps_compute,
        steps_detailed=parsed.config.steps_detailed,
        steps_bg=parsed.config.steps_bg,
        decimals=parsed.config.decimals
    )
    
    if not steps_compute:
        return result

    # Build step-by-step output
    
    steps_mathjax: list[str] = []

    compute_map = {
        "add": BuildSteps(
            title="Matrix addition",
            steps=result.linalg_add_latex
        ),
        "subtract": BuildSteps(
            title="Matrix subtraction",
            steps=result.linalg_subtract_latex
        ),
        "multiply": BuildSteps(
            title="Element-wise matrix multiplication",
            steps=result.linalg_multiply_latex
        ),
        "divide": BuildSteps(
            title="Element-wise matrix division",
            steps=result.linalg_divide_latex
        ),
        "raise": BuildSteps(
            title="Element-wise matrix power",
            steps=result.linalg_raise_latex
        ),
        "matmul": BuildSteps(
            title="Matrix multiplication",
            steps=result.linalg_matmul_latex
        )
    }
    
    for i, op in enumerate(parsed.config.compute_apis, 1):
        api = compute_map[op]
        steps_mathjax.append(html_bg_level1(title=f"{i}) {api.title}"))
        steps_mathjax.extend(api.steps)

    return steps_mathjax