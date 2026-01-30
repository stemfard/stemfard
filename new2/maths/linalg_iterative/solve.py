from typing import Literal

from sympy import Matrix

from stemfard.core._type_aliases import Array2DLike, SequenceArrayLike
from stemfard.maths.linalg_iterative.jacobi import LinalgSolveJacobi
from stemfard.maths.linalg_iterative.conjugate_gradient import LinalgSolveConjugateGradient
from stemfard.maths.linalg_iterative.gauss_seidel import LinalgSolveGaussSeidel
from stemfard.maths.linalg_iterative.sor import LinalgSolveSOR


def linalg_jacobi(
    steps_method: Literal["algebra", "matrix"] = "algebra",
    A: Array2DLike = ...,
    b: SequenceArrayLike = ...,
    x0: SequenceArrayLike | None = None,
    fvars: list[str] | None = None,
    C: Array2DLike | Matrix | None = None,
    relax_param: float | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = None,
    maxit: int = 50,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "b"),
    decimals: int = 12
) -> LinalgSolveJacobi:
    """
    Jacobi iteration (algebra and matrix methods)
    
    Parameters
    ----------
    
    Returns
    -------
    
    Examples
    --------
    >>> import numpy as np
    >>> import stemfard as stm
    
    >>> A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
    >>> b = np.array([6, 25, -11, 15])
    >>> result = stm.linalg_jacobi(steps_method="algebra", A=A, b=b)
    >>> stm.prlatex(result.steps)
    """
    params = {
        "method": "jacobi",
        "steps_method": steps_method,
        "A": A,
        "b": b,
        "x0": x0,
        "fvars": fvars,
        "C": C,
        "relax_param": relax_param,
        "atol": atol,
        "rtol": rtol,
        "maxit": maxit,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    if steps_method == "algebra":
        return LinalgSolveJacobi(**params)._jacobi_algebra
    else:
        return LinalgSolveJacobi(**params)._jacobi_matrix
    
    
def linalg_gauss_seidel(
    steps_method: Literal["algebra", "matrix"] = "algebra",
    A: Array2DLike = ...,
    b: SequenceArrayLike = ...,
    x0: SequenceArrayLike | None = None,
    fvars: list[str] | None = None,
    C: Array2DLike | Matrix | None = None,
    relax_param: float | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = None,
    maxit: int = 50,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "b"),
    decimals: int = 12
) -> LinalgSolveGaussSeidel:
    
    params = {
        "method": "gauss-seidel",
        "steps_method": steps_method,
        "A": A,
        "b": b,
        "x0": x0,
        "fvars": fvars,
        "C": C,
        "relax_param": relax_param,
        "atol": atol,
        "rtol": rtol,
        "maxit": maxit,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    if steps_method == "algebra":
        return LinalgSolveGaussSeidel(**params)._gauss_seidel_algebra
    else:
        return LinalgSolveGaussSeidel(**params)._gauss_seidel_matrix
    

def linalg_sor(
    steps_method: Literal["algebra", "matrix"] = "algebra",
    A: Array2DLike = ...,
    b: SequenceArrayLike = ...,
    x0: SequenceArrayLike | None = None,
    fvars: list[str] | None = None,
    C: Array2DLike | Matrix | None = None,
    relax_param: float | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = None,
    maxit: int = 50,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "b"),
    decimals: int = 12
) -> LinalgSolveSOR:
    
    params = {
        "method": "sor",
        "steps_method": steps_method,
        "A": A,
        "b": b,
        "x0": x0,
        "fvars": fvars,
        "relax_param": relax_param,
        "atol": atol,
        "rtol": rtol,
        "maxit": maxit,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    if steps_method == "algebra":
        return LinalgSolveSOR(**params)._sor_algebra
    else:
        return LinalgSolveSOR(**params)._sor_matrix
    
    
def linalg_conjugate_gradient(
    A: Array2DLike = ...,
    b: SequenceArrayLike = ...,
    x0: SequenceArrayLike | None = None,
    fvars: list[str] | None = None,
    C: Array2DLike | Matrix | None = None,
    relax_param: float | None = None,
    atol: float | None = 1e-6,
    rtol: float | None = None,
    maxit: int = 50,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "b"),
    decimals: int = 12
) -> LinalgSolveConjugateGradient:
    
    params = {
        "method": "conjugate-gradient",
        "A": A,
        "b": b,
        "x0": x0,
        "fvars": fvars,
        "C": C,
        "relax_param": relax_param,
        "atol": atol,
        "rtol": rtol,
        "maxit": maxit,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return LinalgSolveConjugateGradient(**params)._conjugate_gradient_algebra