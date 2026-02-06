# NOTE: COMPLETED

from functools import cached_property

from numpy import asarray, zeros
from numpy.typing import NDArray
from sympy import Matrix

from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math
from stemfard.core.rounding import fround
from stemfard.core.arrays_general import general_array


class BaseLinalgMatrixArithmetics:
    """Base class for matrix arithmetic operations."""

    def __init__(
        self,
        A: Matrix,
        B: Matrix,
        broadcast: bool = False,
        operations: tuple[str, ...] = ("add", ),
        param_names: tuple[str, str] = ("A", "B"),
        result_name: str = "ans",
        steps_compute: bool = True,
        steps_detailed: bool = False,
        steps_bg: bool = False,
        decimals: int = -1
    ):
        # user inputs
        self.A: Matrix = A
        self.B: Matrix = B
        self.broadcast: bool = broadcast
        self.operations: tuple[str, ...] = operations
        self.param_names: tuple[str, str] = param_names
        self.result_name: str = result_name
        self.steps_compute: bool = steps_compute
        self.steps_detailed: bool = steps_detailed
        self.steps_bg: bool = steps_bg
        self.decimals: int = decimals

        # derived dimensions
        self.nrows_a = self.A.rows
        self.ncols_a = self.A.cols
        self.nrows_b = self.B.rows
        self.ncols_b = self.B.cols

        # common aliases
        self.nrows = self.nrows_a
        self.ncols = self.ncols_a

    # Rounded matrices

    @cached_property
    def A_rnd(self) -> Matrix:
        """Rounded version of A according to decimals."""
        return fround(self.A, decimals=self.decimals)

    @cached_property
    def B_rnd(self) -> Matrix:
        """Rounded version of B according to decimals."""
        return fround(self.B, decimals=self.decimals)

    # Presentation

    @property
    def param_a(self) -> str:
        """Parameter name for matrix A."""
        return self.param_names[0]

    @property
    def param_b(self) -> str:
        """
        Colored parameter name for matrix B for LaTeX presentation.
        Useful when highlighting matrix names in equations or outputs.
        """
        return str_color_math(self.param_names[1])

    @cached_property
    def a_latex(self) -> str:
        """LaTeX representation of rounded or original matrix A."""
        return tex_to_latex(self.A_rnd if self.decimals > -1 else self.A)
    
    @cached_property
    def _b_latex_raw(self) -> str:
        """
        LaTeX representation of B, rounded if decimals > -1.
        This is cached to avoid recomputation for b_latex and numeric
        analysis.
        """
        matrix_to_use = self.B_rnd if self.decimals > -1 else self.B
        return tex_to_latex(matrix_to_use)

    @cached_property
    def b_latex(self) -> str:
        """
        LaTeX representation of rounded or original matrix B, colored.
        Uses cached raw LaTeX.
        """
        return str_color_math(self._b_latex_raw)

    @cached_property
    def matrices_a_and_b(self) -> str:
        """LaTeX-formatted representation of matrices A and B."""
        a = f"{self.param_a} = {general_array('a')}"
        b = str_color_math(f"{self.param_b} = {general_array('b')}")
        return f"{a} \\:, \\quad {b}"

    # Numeric analysis

    @cached_property
    def zeros_array(self) -> NDArray:
        """
        NumPy array of zeros with the same shape as matrix A.
        
        Returns
        -------
        NDArray
            Array of zeros matching the dimensions of matrix A.
        """
        return zeros(self.A.shape, dtype=object)
    
    @cached_property
    def b_numeric_info(self) -> tuple[int, int, int]:
        """
        Numeric analysis of matrix B (SymPy Matrix), compatible with
        symbolic elements.

        Returns
        -------
        zeros_count : int
            Number of elements exactly equal to zero.
        float_count : int
            Number of elements that appear as floats (have decimal point), 
            counted from LaTeX representation.
        non_float_count : int
            Number of elements that are not floats.
            
        Note
        ----
        `float_count` is computed from LaTeX representation, so may
        include symbolic expressions with decimals. `non_float_count` is
        computed as total elements minus `float_count`.
        """
        # Flatten the matrix and filter numeric zeros safely
        flat_arr = asarray(self.B).flatten()
        zeros_count = sum(1 for x in flat_arr if x.is_number and x == 0)

        # Count decimals using cached LaTeX
        float_count = self._b_latex_raw.count(".")
        non_float_count = self.B.rows * self.B.cols - float_count

        return zeros_count, float_count, non_float_count
    
    @property
    def reciprocal_action(self) -> str | None:
        """
        Generates LaTeX instruction for reciprocal operation.
        Only applicable if decimals == -1.
        """
        zeros_count, floats_count, non_float_count = self.b_numeric_info

        if non_float_count == 0 or self.decimals != -1:
            return None

        # Determine descriptive message
        if zeros_count == 0:
            msg = "non-decimal"
        elif floats_count == 0:
            msg = "non-zero"
        else:
            msg = "non-zero and non-decimal"

        s = "" if non_float_count == 1 else "s"
        return (
            f"Take the reciprocal of the \\( {non_float_count} \\) {msg} "
            f"colored element{s} in the above matrix then change the sign "
            "from \\( \\div \\) to \\( \\times \\) as shown below."
        )