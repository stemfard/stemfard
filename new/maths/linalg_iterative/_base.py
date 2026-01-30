from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from numpy import (
    around, diag, float64, inf, matmul, tril, tril_indices_from, triu,
    triu_indices_from
)
from numpy.linalg import eigvals, inv, norm
from numpy.typing import NDArray
from sympy import Matrix

from stemfard.core._type_aliases import Array2DLike, SequenceArrayLike
from stemfard.maths.linalg_iterative._parse_params import parse_linalg_iterative
from stemfard.core.arrays_highlight import highlight_arrays_vals


@dataclass(slots=True, frozen=True)
class MatricesLDU:
    A: str
    L: str
    D: str
    U: str
    LU: str
    latex: list[str]


class BaseLinalgSolveIterative:
    """Base class for Iterative solutions of linear systems."""

    def __init__(
        self,
        method: str = "jacobi",
        steps_method: Literal["algebra", "matrix"] = "algebra",
        A: Array2DLike = ...,
        b: SequenceArrayLike = ...,
        fvars: list[str] | None = None,
        x0: SequenceArrayLike | None = None,
        C: Array2DLike | Matrix | None = ...,
        relax_param: float | None = None,
        atol: float | None = 1e-6,
        rtol: float | None = None,
        maxit: int = 10,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        show_bg: bool = True,
        param_names: tuple[str, str] = ("A", "b"),
        decimals: int = 14
    ):
        parsed_params = parse_linalg_iterative(
            method=method,
            steps_method=steps_method,
            A=A,
            b=b,
            fvars=fvars,
            x0=x0,
            C=C,
            relax_param=relax_param,
            atol=atol,
            rtol=rtol,
            maxit=maxit,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            show_bg=show_bg,
            param_names=param_names,
            decimals=decimals
        )
        
        self.method = parsed_params.method
        self.steps_method = parsed_params.steps_method
        self.A = parsed_params.A
        self.b = parsed_params.b
        self.fvars = parsed_params.fvars
        self.x0 = parsed_params.x0
        self.C = parsed_params.C
        self.relax_param = parsed_params.relax_param
        self.atol = parsed_params.atol
        self.rtol = parsed_params.rtol
        self.maxit = parsed_params.maxit
        self.steps_compute = parsed_params.steps_compute
        self.steps_detailed = parsed_params.steps_detailed
        self.show_bg = parsed_params.show_bg
        self.param_names = parsed_params.param_names
        self.decimals = parsed_params.decimals
        
        self.params = parsed_params.params
        
    @property
    def abs_rel_tol(self) -> str:    
        if self.atol is None:
            return "rtol"
        return "atol"
        
    @property
    def nrows(self) -> int:
        return self.A.shape[0]
    
    @property
    def ncols(self) -> int:
        return self.A.shape[1]
    
    @property
    def A_rnd(self) -> NDArray[float64]:
        return around(self.A, self.decimals)
    
    @property
    def b_rnd(self) -> NDArray[float64]:
        return around(self.b, self.decimals)
    
    @property
    def x0_rnd(self) -> NDArray[float64]:
        return around(self.x0, self.decimals)
    
    @property
    def C_rnd(self) -> NDArray[float64]:
        return around(self.C, self.decimals)
    
    @property
    def D(self) -> NDArray[float64]:
        return diag(diag(self.A))
    
    @property
    def D_rnd(self) -> NDArray[float64]:
        return around(self.D, self.decimals)
    
    @property
    def D_inv(self) -> NDArray[float64]:
        return inv(self.D)
    
    @property
    def D_inv_rnd(self) -> NDArray[float64]:
        return around(self.D_inv, self.decimals)
    
    @property
    def L(self) -> NDArray[float64]:
        return -tril(self.A, k=-1)
    
    @property
    def L_rnd(self) -> NDArray[float64]:
        return around(self.L, self.decimals)
    
    @property
    def U(self) -> NDArray[float64]:
        return -triu(self.A, k=1)
    
    @property
    def U_rnd(self) -> NDArray[float64]:
        return around(self.U, self.decimals)
    
    @property
    def L_plus_U(self) -> NDArray[float64]:
        return self.L + self.U
    
    @property
    def L_plus_U_rnd(self) -> NDArray[float64]:
        return around(self.L_plus_U, self.decimals)
    
    @property
    def xj(self) -> str:
        return [f"x{index}" for index in range(1, self.ncols + 1)]
    
    @property
    def xj_underscore(self) -> str:
        return [f"x_{index}" for index in range(1, self.ncols + 1)]
    
    @property
    def cj(self) -> NDArray[float64]:
        return matmul(self.D_inv, self.b)
    
    @property
    def cj_rnd(self) -> NDArray[float64]:
        return around(self.cj, self.decimals)
    
    @property
    def tj(self) -> NDArray[float64]:
        return matmul(self.D_inv, self.L_plus_U)
    
    @property
    def tj_rnd(self) -> NDArray[float64]:
        return around(self.tj, self.decimals)
    
    @property
    def tj_eigvals(self) -> NDArray[float64]:
        return eigvals(self.tj)
    
    @property
    def tj_eigvals_rnd(self) -> NDArray[float64]:
        return around(self.tj_eigvals, self.decimals)
    
    @cached_property
    def omega(self) -> float:
        return float(2 / (1 + (1 - (max(abs(self.tj_eigvals))) ** 2) ** (1/2)))
    
    @cached_property
    def omega_rnd(self) -> float:
        return float(around(self.omega, self.decimals))
    
    def kth_norm(self, x_new, x, abs_rel_tol):
        if abs_rel_tol == "atol":
            _kth_norm = norm(x=x_new - x, ord=inf)
        else:
            _kth_norm = (
                norm(x=x_new - x, ord=inf) / (norm(x=x_new, ord=inf) + 1e-32)
            )
        return _kth_norm
    
    @cached_property
    def matrices_dlu(self) -> MatricesLDU:
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append(
            f"The given system of \\( {self.nrows} \\) linear equations "
            "leads to the matrices \\( L, \\: D \\) and \\( U \\) given below"
        )
        
        row_idx, col_idx = tril_indices_from(self.A_rnd, k=-1)
        L_map_indices = {tup: "orange" for tup in tuple(zip(row_idx, col_idx))}
        
        row_idx, col_idx = triu_indices_from(self.A_rnd, k=1)
        U_map_indices = {tup: "blue" for tup in tuple(zip(row_idx, col_idx))}
        
        A_colored = highlight_arrays_vals(
            arr=self.A_rnd,
            color_diag="main",
            color_map_indices= L_map_indices | U_map_indices
        )
        D_colored = highlight_arrays_vals(
            arr=self.D_rnd,
            color_diag="main"
        )
        
        L_colored = highlight_arrays_vals(
            arr=self.L_rnd,
            color_map_indices=L_map_indices
        )
        
        U_colored = highlight_arrays_vals(
            arr=self.U_rnd,
            color_map_indices=U_map_indices
        )
        
        LU_colord = highlight_arrays_vals(
            arr=self.L_plus_U_rnd,
            color_map_indices= L_map_indices | U_map_indices
        )
        
        steps_mathjax.append(
            f"\\[ A = {A_colored} \\quad \\longrightarrow "
            f"\\quad L = {L_colored}, "
            f"\\quad D = {D_colored}, "
            f"\\quad U = {U_colored} \\]"
        )
        
        return MatricesLDU(
            A=A_colored,
            L=L_colored,
            D=D_colored,
            U=U_colored,
            LU=LU_colord,
            latex=steps_mathjax
        )