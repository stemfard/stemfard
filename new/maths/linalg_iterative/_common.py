from dataclasses import dataclass
from numpy import around, asarray, float64
from numpy.typing import NDArray

from stemfard.core.arrays_highlight import highlight_array_vals_arr
from stemfard.core.constants import StemConstants


TOLERANCE = {
    "atol": (
        "\\( \\displaystyle"
        "\\quad || \\: x^{(k+1)} - x^{(k)} \\: || < \\varepsilon \\quad \\) "
    ),
    "rtol": (
        "\\( \\displaystyle\\quad "
        "\\frac{|| \\: x^{(k+1)} - x^{(k)} \\: ||}{|| \\: x^{(k + 1)} \\: "
        "||_{\\infty}} < \\varepsilon \\)"
    )
}


@dataclass(slots=True, frozen=True)
class LinalgSolveIterativeResult:
    arr: NDArray
    nrows: int
    solution: NDArray[float64]
    latex: str


def bg_matrices_dlu(abs_rel_tol: str) -> list[str]:
    
    steps_mathjax: list[str] = []
    
    steps_mathjax.append(
        "where \\( D, \\: L \\) and \\( U \\) are square "
        "matrices obtained from the coefficients matrix \\( A \\) as follows."
    )
    steps_mathjax.append("\\( \\quad D = \\) Diagonal matrix")
    steps_mathjax.append(
        "\\( \\quad L = -1 \\: \\times \\: \\) (Strictly lower triangular "
        "matrix of \\( A \\))."
    )
    steps_mathjax.append(
        "\\( \\quad U = -1 \\: \\times \\: \\) (Strictly upper triangular "
        "matrix of \\( A \\))."
    )
    steps_mathjax.append(
        "Iterations are performed until the following is satisfied."
    )
    steps_mathjax.append(TOLERANCE[abs_rel_tol])
    steps_mathjax.append(StemConstants.BORDER_HTML_BG)
    
    return steps_mathjax


def table_latex(
    arr: NDArray[float64], col_names: list[str], decimals: int = 12
) -> str:
    """Table of results for the iterative methods"""
    
    arr = around(arr, decimals)
    solution = arr[-1, :]
    nrows = arr.shape[0]
    index = asarray(["k"] + list(range(nrows))).reshape(-1, 1)
    latex = highlight_array_vals_arr(
        arr=arr,
        index=index,
        col_names=col_names,
        color_rows=nrows
    )
    
    return {
        "nrows": nrows,
        "solution": solution,
        "latex": latex.replace("\\left[", "").replace("\\right]", "")
    }