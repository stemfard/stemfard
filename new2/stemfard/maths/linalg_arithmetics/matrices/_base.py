from functools import cached_property

from numpy import asarray, prod, zeros
from numpy.typing import NDArray
from sympy import Matrix

from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math
from stemfard.core.rounding import fround, round_dp
from stemfard.core.decimals import numeric_format
from stemfard.core.arrays_general import general_array
from stemfard.maths.linalg_arithmetics.matrices._parse_params import (
    parse_linalg_matrix_arithmetics,
)


class BaseLinalgMatrixArithmetics:
    """Base class for matrix arithmetic operations."""

    def __init__(self, **kwargs):
        parsed = parse_linalg_matrix_arithmetics(**kwargs)

        # core objects
        self.A: Matrix = parsed.A
        self.B: Matrix = parsed.B
        self.config = parsed.config
        self.params = parsed.params

        # derived dimensions
        self.nrows_a = self.A.rows
        self.ncols_a = self.A.cols
        self.nrows_b = self.B.rows
        self.ncols_b = self.B.cols

        # common aliases
        self.nrows = self.nrows_a
        self.ncols = self.ncols_a

    # ------------------ Config shortcuts ------------------ #

    @property
    def compute_apis(self) -> tuple[str, ...]:
        return self.config.compute_apis

    @property
    def param_names(self) -> tuple[str, ...] | None:
        return self.config.param_names

    @property
    def result_name(self) -> str:
        return self.config.result_name

    @property
    def steps_compute(self) -> str:
        return self.config.steps_compute

    @property
    def steps_detailed(self) -> str:
        return self.config.steps_detailed

    @property
    def steps_bg(self) -> str:
        return self.config.steps_bg

    @property
    def decimals(self) -> int:
        return self.config.decimals

    # ------------------ Rounded matrices ------------------ #

    @cached_property
    def A_rnd(self) -> Matrix:
        return fround(self.A, decimals=self.decimals)

    @cached_property
    def B_rnd(self) -> Matrix:
        return fround(self.B, decimals=self.decimals)

    # ------------------ Presentation ------------------ #

    @property
    def _param_names(self) -> tuple[str, str]:
        if self.param_names is None:
            return "A", "B"
        return self.param_names
    
    @property
    def param_a(self) -> str:
        return self._param_names[0]

    @property
    def param_b(self) -> str:
        return str_color_math(self._param_names[1])

    @cached_property
    def matrices_a_and_b(self) -> str:
        a = f"{self.param_a} = {general_array('a')}"
        b = str_color_math(f"{self.param_b} = {general_array('b')}")
        return f"{a} \\:, \\quad {b}"

    @cached_property
    def a_latex(self) -> str:
        return tex_to_latex(self.A_rnd if self.decimals > -1 else self.A)

    @cached_property
    def b_latex(self) -> str:
        latex = tex_to_latex(self.B_rnd if self.decimals > -1 else self.B)
        return str_color_math(latex)

    # ------------------ Numeric analysis ------------------ #

    @cached_property
    def zeros_arr(self) -> NDArray:
        return zeros((self.nrows, self.ncols), dtype=object)
        
    @cached_property
    def floats_in_b(self) -> int:
        try:
            flat = numeric_format(asarray(self.B).flatten())
            ones_zeros = len(flat[(flat == -1) | (flat == 0) | (flat == 1)])
        except (ValueError, TypeError):
            ones_zeros = 0

        float_count = tex_to_latex(
            round_dp(self.B, decimals=self.decimals)
        ).count(".")

        return ones_zeros + float_count

    @cached_property
    def non_floats_in_b(self) -> int:
        return prod(self.B.shape) - self.floats_in_b