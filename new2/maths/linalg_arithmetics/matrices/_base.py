from functools import cached_property

from numpy import asarray, prod, zeros
from numpy.typing import NDArray
from sympy import Matrix

from stemfard.maths.linalg_arithmetics.matrices._parse_params import MatrixArithmeticsParsedParameters
from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math
from stemfard.core.rounding import fround, round_dp
from stemfard.core.decimals import numeric_format
from stemfard.core.arrays_general import general_array


class BaseLinalgMatrixArithmetics:
    """Base class for Matrix arithmetics."""

    def __init__(self, **kwargs):
        
        parsed_params = MatrixArithmeticsParsedParameters.parse(**kwargs)
        
        self.A = parsed_params.A
        self.B = parsed_params.B
        self.compute_apis = parsed_params.compute_apis
        self.param_a = parsed_params.param_names[0]
        self.param_b = str_color_math(parsed_params.param_names[1])
        self.result_name = "M"
        self.steps_compute = parsed_params.steps_compute
        self.steps_detailed = parsed_params.steps_detailed
        self.steps_bg = parsed_params.steps_bg
        self.decimals = parsed_params.decimals
        self.params = parsed_params.params
        
        # derived
        self.nrows_a: int = self.A.shape[0]
        self.nrows = self.nrows_a
        self.ncols_a: int = self.A.shape[1]
        self.ncols = self.ncols_a
        self.nrows_b: int = self.B.shape[0]
        self.ncols_b: int = self.B.shape[1]
    
    
    @cached_property
    def A_rnd(self) -> Matrix:
        return fround(self.A, decimals=self.decimals)
    
    @cached_property
    def B_rnd(self) -> Matrix:
        return fround(self.B, decimals=self.decimals)
    
    @cached_property
    def matrices_a_and_b(self) -> str:
        a_latex = f"A = {general_array('a')}"
        b_latex = str_color_math(f"B = {general_array('b')}")
        return f"{a_latex} \\:, \\quad {b_latex}"
    
    @cached_property
    def zeros(self) -> NDArray:
        return zeros((self.nrows, self.ncols), dtype=object)
    
    @cached_property
    def a_latex(self) -> str:
        A = self.A
        if self.decimals > -1:
            A = fround(self.A, decimals=self.decimals)
        return tex_to_latex(A)
    
    @cached_property
    def b_latex(self) -> str:
        B = self.B
        if self.decimals > -1:
            B = fround(self.B, decimals=self.decimals)
        return str_color_math(tex_to_latex(B))
    
    @cached_property
    def floats_in_b(self) -> int:
        try:
            M = numeric_format(asarray(self.B).flatten())
            ones_zeros_count = len(M[(M == -1) | (M == 0) | (M == 1)])
        except (ValueError, TypeError):
            ones_zeros_count = 0
        
        float_count = tex_to_latex(
            round_dp(self.B, decimals=self.decimals)
        ).count(".")
        
        return ones_zeros_count + float_count
        
    @cached_property
    def non_floats_in_b(self) -> int:
        return prod(self.B.shape) - self.floats_in_b