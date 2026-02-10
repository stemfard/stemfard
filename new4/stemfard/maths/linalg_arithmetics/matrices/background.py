from dataclasses import dataclass
from functools import cached_property
from stemfard.maths.linalg_arithmetics.matrices._base import BaseLinalgMatrixArithmetics
from stemfard.core._strings import str_color_math
from stemfard.core.arrays_general import general_array_ab, generate_a_times_b
from stemfard.maths.linalg_arithmetics.matrices.utils import MatrixOperations


@dataclass(frozen=True)
class OperationSpec:
    verb: str
    symbol: str
    
    
OPERATION_MAP: dict[MatrixOperations, OperationSpec] = {
    "add": OperationSpec("adding", "+"),
    "subtract": OperationSpec("subtracting", "-"),
    "multiply": OperationSpec("multiplying", "\\times"),
    "divide": OperationSpec("dividing", "\\div"),
    "raise": OperationSpec("raising", "^")
}

class MatrixArithmeticsBackground(BaseLinalgMatrixArithmetics):
    """
    Provides MathJax step-by-step explanations for matrix arithmetic
    operations, including element-wise operations (add, subtract,
    multiply, divide, raise) and matrix multiplication.
    """

    def _introduction_elementwise(
        self, operation: MatrixOperations
    ) -> list[str]:
        """
        Returns a list of MathJax steps for an element-wise matrix
        operation.
        """
        steps_mathjax: list[str] = []

        B = str_color_math("B")
        b_ij = str_color_math("b_{ij}")

        steps_mathjax.append(
            "Let \\( A \\) and \\( B \\) be two \\( m \\times n \\) "
            "matrices as given below."
        )
        steps_mathjax.append(f"\\[ {self.matrices_a_and_b} \\]")

        spec = OPERATION_MAP[operation]

        # Operation-specific natural language
        if operation == "subtract":
            desc = (
                f"subtracting the elements of matrix \\( {B} \\) from "
                "the corresponding elements of matrix \\( A \\)"
            )
            symbol = spec.symbol

        elif operation == "divide":
            desc = (
                f"dividing the elements of matrix \\( A \\) by the "
                f"corresponding elements of matrix \\( {B} \\)"
            )
            symbol = spec.symbol

        elif operation == "raise":
            desc = (
                f"raising the elements of matrix \\( A \\) to the "
                f"corresponding elements of matrix \\( {B} \\)"
            )
            symbol = "\\:^"

        else:
            desc = (
                f"{spec.verb} the corresponding elements of matrices "
                f"\\( A \\) and \\( {B} \\)"
            )
            symbol = spec.symbol

        intro = (
            f"The matrix \\( C \\), obtained by {desc}, is denoted by "
            f"\\( C = A {symbol} {B} \\). For each element \\( a_{{ij}} \\) "
            f"of matrix \\( A \\) and each element \\( {b_ij} \\) of matrix "
            f"\\( {B} \\), the corresponding element \\( c_{{ij}} \\) in the "
            f"resulting matrix \\( C \\) is calculated as "
            f"\\( c_{{ij}} = a_{{ij}} {symbol} {b_ij} \\)."
        )

        steps_mathjax.append(intro)
        
        return steps_mathjax
    
    
    @property
    def b_reciprocal(self) -> str:
        """
        Returns the element-wise reciprocal of B as a LaTeX string.
        """
        arr = general_array_ab(A="a", B=f"\\frac{{1}}{{b}}", operation="\\div")
        return arr.replace("\\div", "\\times")

    
    def bg_elementwise(self, operation: MatrixOperations) -> list[str]:
        """
        Returns MathJax steps for any element-wise operation
        (add, subtract, multiply, divide, raise).
        """
        steps_mathjax: list[str] = []
        
        try:
            spec = OPERATION_MAP[operation]
        except KeyError as e:
            raise ValueError(
                f"Expected 'operation' to be one of: "
                f"{', '.join(OPERATION_MAP)}; got {operation}"
            ) from e

        steps_mathjax = self._introduction_elementwise(operation)
        symbol = spec.symbol
        
        temp_step = (
            f"C = A {symbol} {self.b_latex} = "
            f"{general_array_ab(operation=symbol)}"
        )

        if operation == "divide":
            temp_step = f"{temp_step} \\equiv {self.b_reciprocal}"
        
        steps_mathjax.append(f"\\[ {temp_step} \\]")

        return steps_mathjax

    
    @cached_property
    def bg_matmul(self) -> list[str]:
        """
        Returns MathJax steps for matrix multiplication (A · B).
        """
        steps_mathjax: list[str] = []

        dims_m = str_color_math("m")
        dims_p = str_color_math("p", color="red")
        dims_n = str_color_math("n")
        dims_ab = f"{{{dims_m} \\times {dims_n}}}"

        a_dims_latex = f"A_{{{dims_m} \\times {dims_p}}}"
        b_dims_latex = f"B_{{{dims_p} \\times {dims_n}}}"

        steps_mathjax.append(
            "Let \\( A \\) and \\( B \\) be two matrices as given below."
        )
        steps_mathjax.append(f"\\[ {self.matrices_a_and_b} \\]")

        steps_mathjax.append(
            f"The matrix product \\( {a_dims_latex} \\cdot {b_dims_latex} \\) "
            f"is an \\( {dims_ab} \\) matrix \\( C \\) whose entries are "
            "given by:"
        )
        steps_mathjax.append(
            f"\\( \\quad c_{{ij}} = \\sum\\limits_{{k = 1}}^{{p}} "
            f"a_{{ik}} \\: {str_color_math(f'b_{{kj}}')} "
            f"= a_{{i1}} \\: {str_color_math(f'b_{{1j}}')} + "
            f"a_{{i2}} \\: {str_color_math(f'b_{{2j}}')} + "
            f"\\cdots + a_{{ip}} \\: {str_color_math(f'b_{{pj}}')} \\)"
        )
        steps_mathjax.append("That is,")
        steps_mathjax.append(f"\\( \\quad {generate_a_times_b()} \\)")

        return steps_mathjax