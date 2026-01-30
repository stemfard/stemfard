from functools import cached_property
from typing import Any

from numpy import around

from stemfard.core._type_aliases import Array2DLike, Array2DMatrixLike, SequenceArrayLike
from stemfard.maths.linalg_matrix_arithmetics._parse_params import parse_linalg_matrix_arithmetics


class BaseLinalgMatrixArithmetics:
    """Base class for Matrix arithmetics."""

    def __init__(
        self,
        method: str = "add",
        A: Array2DLike = ...,
        B: SequenceArrayLike = ...,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        show_bg: bool = True,
        param_names: tuple[str, str] = ("A", "B"),
        decimals: int = 14
    ):
        parsed_params = parse_linalg_matrix_arithmetics(
            method=method,
            A=A,
            B=B,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            show_bg=show_bg,
            param_names=param_names,
            decimals=decimals
        )
        
        self.prm_method: str = parsed_params.method
        self.prm_A: Array2DMatrixLike = parsed_params.A
        self.prm_B: Array2DMatrixLike = parsed_params.B
        self.prm_steps_compute: bool = parsed_params.steps_compute
        self.prm_steps_detailed: bool = parsed_params.steps_detailed
        self.prm_show_bg: bool = parsed_params.show_bg
        self.prm_param_names: tuple[str, str] = parsed_params.param_names
        self.prm_decimals: int = parsed_params.decimals
        
        self.msc_params: dict[str, Any] = parsed_params.params
    
    
    @property
    def bse_nrows(self) -> int:
        return self.prm_A.shape[0]
    
    @property
    def bse_ncols(self) -> int:
        return self.prm_A.shape[1]
    
    @property
    def bse_A_rnd(self) -> Array2DMatrixLike:
        return around(self.prm_A, self.prm_decimals)
    
    @property
    def bse_B_rnd(self) -> Array2DMatrixLike:
        return around(self.prm_B, self.prm_decimals)
    
    @cached_property
    def bse_AB(self) -> Array2DMatrixLike:
        return self.prm_A @ self.prm_B
    
    @property
    def bse_AB_rnd(self) -> Array2DMatrixLike:
        try:
            return around(self.bse_AB, self.prm_decimals)
        except (TypeError, ValueError):
            return self.bse_AB