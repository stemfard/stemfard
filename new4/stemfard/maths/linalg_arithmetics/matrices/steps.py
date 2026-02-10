from dataclasses import dataclass
from typing import Any

from numpy import zeros

from stemfard.core._latex import tex_to_latex, tex_to_latex_arr
from stemfard.maths.linalg_arithmetics.matrices.background import MatrixArithmeticsBackground
from stemfard.core.constants import StemConstants
from stemfard.core._strings import str_color_math, str_remove_tzeros
from stemfard.core.rounding import fround
from stemfard.core.io._io import present_checked_answer
from stemfard.core._is_dtypes import is_negative
from stemfard.maths.linalg_arithmetics.matrices.answer import matrix_arithmetics_ans
from stemfard.core.signs import abs_value, pm_sign
from stemfard.core._html import html_bg_level2
from stemfard.maths.linalg_arithmetics.matrices.utils import (
    MATRIX_OPERATIONS,
    MATRIX_OPS_ELEMENTWISE
)
from stemfard.core.models import AnswerStepsMathjax
from stemfard.core.errors.general import OperationError
from stemfard.maths.linalg_arithmetics.matrices.question import question_str
from stemfard.maths.linalg_arithmetics.matrices.syntax import generate_syntax
from stemfard.core.io.serialize import serialize_matrix


@dataclass
class FloatsCount:
    float_n: int
    non_float_n: int


class MatrixArithmetics(MatrixArithmeticsBackground):
    """Matrix arithmetics with unified step-by-step LaTeX generation."""
    
    def answer(self, operation: str) -> list[list[Any]]:
        return matrix_arithmetics_ans(
            A=self.A,
            B=self.B,
            broadcast=self.broadcast,
            operation=operation,
            decimals=self.decimals
        )


    def _html_title(self, title: str) -> str:
        """Return HTML for a step title."""
        return html_bg_level2(title=title)


    def _user_matrices_latex(self, operation: str) -> list[str]:
        """
        LaTeX description of the input matrices for a given operation.
        """
        steps_mathjax: list[str] = []
        
        steps_mathjax = [
            f"For the matrices \\( {self.param_a} \\) and "
            f"\\( {self.param_b_latex} \\) given below."
        ]
        steps_mathjax.append(
            f"\\( \\quad {self.param_a} = {self.a_latex} \\:, "
            f"\\quad {self.param_b_latex} \\: {str_color_math('=')} \\: "
            f"{self.b_latex} \\)",
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)

        if operation == "raise":
            steps_mathjax.append(
                f"The result, \\( {self.param_a}\\:^{{{self.param_b_latex}}} \\), "
                "is obtained by raising each element of "
                f"\\( {self.param_a} \\) to the corresponding element of "
                f"\\( {self.param_b_latex} \\)."
            )
        elif operation != "matmul":
            op_map = {
                "add": (
                    "adding",
                    "sum",
                    f"{self.param_a} + {self.param_b_latex}"
                ),
                "subtract": (
                    "subtracting",
                    "difference",
                    f"{self.param_a} - {self.param_b_latex}"
                ),
                "multiply": (
                    "multiplying",
                    "element-wise product",
                    f"{self.param_a} \\times {self.param_b_latex}"
                ),
                "divide": (
                    "dividing",
                    "element-wise quotient",
                    f"{self.param_a} \\div {self.param_b_latex}"
                )
            }
            verb, noun, formula = op_map[operation]
            steps_mathjax.append(
                f"The {noun}, \\( {formula} \\), is obtained by {verb} "
                "corresponding elements."
            )

        return steps_mathjax


    def _elementwise_latex(self, operation: str) -> list[str]:
        """Generates LaTeX steps for element-wise operations."""
        steps_mathjax: list[str] = []

        # Optional background steps
        if self.steps_bg:
            steps_mathjax.extend(self.bg_elementwise(operation=operation))
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)

        # User-provided matrices
        steps_mathjax.extend(self._user_matrices_latex(operation))
        
        decimals = self.decimals
        A, B = (self.A_rnd, self.B_rnd) if decimals > -1 else (self.A, self.B)
        api_answer = self.answer(operation)
        result_rnd = fround(api_answer, decimals=decimals)

        op_symbol = {
            "add": "+",
            "subtract": "-",
            "multiply": "\\times",
            "divide": "\\div",
            "raise": "\\:^"
        }[operation]

        # Initialize step arrays
        result_step_1 = self.zeros_array.copy()
        result_step_2 = (
            self.zeros_array.copy()
            if operation in {"divide", "raise"} else None
        )
        result_step_3 = (
            self.zeros_array.copy() if operation == "raise" else None
        )

        # Precompute standard steps for all elements
        for i in range(self.nrows):
            for j in range(self.ncols):
                a_ij = A[i, j]
                b_ij = B[i, j]
                a_ij_latex = tex_to_latex(a_ij, decimals)
                b_ij_latex_color = str_color_math(tex_to_latex(b_ij, decimals))
                if is_negative(b_ij):
                    b_ij_latex_color = (
                        str_color_math(f"\\left({b_ij_latex_color}\\right)")
                    )
                result_step_1[i, j] = (
                    f"{a_ij_latex} {op_symbol} {b_ij_latex_color}"
                )

        # Operation-specific detailed steps
        if operation == "divide":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    a_ij = A[i, j]
                    b_ij = B[i, j]
                    if b_ij != 0 and "." not in str_remove_tzeros(b_ij):
                        B_recip = tex_to_latex(1 / b_ij, decimals)
                        if is_negative(b_ij):
                            B_recip = f"\\left({B_recip}\\right)"
                        a_ij_latex = tex_to_latex(a_ij, decimals)
                        result_step_2[i, j] = (
                            f"{a_ij_latex} \\times {str_color_math(B_recip)}"
                        )

        elif operation == "raise":
            for i in range(self.nrows):
                for j in range(self.ncols):
                    a_ij = A[i, j]
                    b_ij = B[i, j]
                    a_ij_latex = tex_to_latex(a_ij, decimals)
                    b_ij_latex_color = (
                        str_color_math(tex_to_latex(b_ij, decimals))
                    )
                    if is_negative(b_ij):
                        b_ij_latex_color = (
                            str_color_math(f"\\left({b_ij_latex_color}\\right)")
                        )

                    # Only handle negative powers safely
                    if b_ij < 0 and a_ij != 0:
                        neg_power_step = f"-1 \\times {b_ij_latex_color}"
                        result_step_2[i, j] = neg_power_step

                        abs_power = tex_to_latex(abs(B[i, j]), decimals)
                        reciprocal_step = (
                            f"\\left({tex_to_latex(1 / a_ij, decimals)}"
                            f"\\right)^{str_color_math(abs_power)}"
                        )
                        result_step_3[i, j] = (
                            f"{neg_power_step} \\Rightarrow {reciprocal_step}"
                        )

        # Build LaTeX output
        steps_mathjax.append(
            f"\\( {self.result_name} "
            f"= {self.param_a} {op_symbol} {self.param_b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} {op_symbol} {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )

        if result_step_2 is not None and self.reciprocal_action is not None:
            steps_mathjax.append(self.reciprocal_action)
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(result_step_2)} \\)"
            )

        if result_step_3 is not None:
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(result_step_3)} \\)"
            )

        # Final rounded result
        steps_mathjax.append(
            f"\\( \\quad {present_checked_answer(result_rnd)} \\)"
        )

        return steps_mathjax

    
    def _matmul_latex(self) -> list[str]:
        """Generates LaTeX steps for matrix multiplication."""
        steps_mathjax = []

        if self.steps_bg:
            steps_mathjax.extend(self.bg_matmul)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
           
        steps_mathjax.extend(self._user_matrices_latex("matmul"))
         
        steps_mathjax.append(
            f"The product \\( {self.result_name} "
            f"= {self.param_a} \\cdot {self.param_b_latex} \\) is evaluated as "
            "follows."
        )

        steps_mathjax.append(
            self._html_title(
                "STEP 1: Determine the dimensions the two matrices"
            )
        )
        
        steps_mathjax.append(f"\\( \\text{{Matrix }} {self.param_a} \\)")
        steps_mathjax.append(
            f"\\( \\quad\\text{{Rows }}: {self.nrows_a} \\:, "
            f"\\quad \\text{{Columns }}: {self.ncols_a} \\)"
        )
        
        steps_mathjax.append(f"\\( \\text{{Matrix }} {self.param_b_latex} \\)")
        steps_mathjax.append(
            f"\\( \\quad\\text{{Rows }}: {self.nrows_b} \\:, "
            f"\\quad \\text{{Columns: }} {self.ncols_b} \\)"
        )
        
        steps_mathjax.append(
            self._html_title(
                "STEP 2: Understand the dimensions of the product matrix"
            )
        )
        
        steps_mathjax.append(
            f"The product matrix \\( {self.result_name} \\) will be a "
            f"\\( {self.nrows_a} \\) by \\( {self.ncols_b} \\) matrix. That "
            "is, it will be a matrix whose dimensions is the number of "
            f"\\( \\textit{{rows}} \\) of matrix \\( {self.param_a} \\) by "
            f"the number of \\( \\textit{{columns}} \\) of matrix "
            f"\\( {self.param_b_latex} \\)."
        )

        steps_mathjax.append(self._html_title("STEP 3: Multiply the matrices"))
        steps_mathjax.append(
            f"\\( {self.result_name} "
            f"= {self.param_a} \\cdot {self.param_b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} \\cdot {self.b_latex} \\)"
        )

        # Compute product
        A, B = self.A, self.B
        nrows_a, ncols_a = A.shape
        _, ncols_b = B.shape
        AB_expr = zeros((nrows_a, ncols_b), dtype=object)
        AB_calc = zeros((nrows_a, ncols_b), dtype=object)
        decimals = self.decimals
        api_answer = self.answer("matmul")
        result_rnd = fround(api_answer, decimals=decimals)

        for i in range(nrows_a):
            for j in range(ncols_b):
                terms_expr, terms_calc = [], []
                for k in range(ncols_a):
                    Aik = tex_to_latex(A[i, k], decimals)
                    Bik = tex_to_latex(B[k, j], decimals)
                    if is_negative(A[i, k]):
                        Aik = f"\\left({Aik}\\right)"
                    if is_negative(B[k, j]):
                        Bik = f"\\left({Bik}\\right)"
                    terms_expr.append(f"{Aik} \\times {str_color_math(Bik)}")
                    prod = fround(A[i, k] * B[k, j], decimals)
                    terms_calc.append(
                        tex_to_latex(prod) 
                        if k == 0 else 
                        pm_sign(prod) + tex_to_latex(abs_value(prod, False))
                    )

                AB_expr[i, j] = " + ".join(terms_expr)
                AB_calc[i, j] = " + ".join(terms_calc)

        steps_mathjax.append(f"\\( \\quad = {tex_to_latex_arr(AB_expr)} \\)")
        steps_mathjax.append(f"\\( \\quad = {tex_to_latex_arr(AB_calc)} \\)")
        steps_mathjax.append(f"\\( \\quad {present_checked_answer(result_rnd)} \\)")

        return steps_mathjax


    # Unified property for all operations
    def steps_latex(self, operation: str) -> AnswerStepsMathjax:
        
        if operation not in MATRIX_OPERATIONS:
            raise OperationError(
                operation=operation,
                valid_operations=MATRIX_OPERATIONS
            )
        
        question = question_str(
            a_latex=self.a_latex,
            b_latex=self.b_latex,
            operation=self.operations,
            param_names=(self.param_a, self.param_b_latex)
        )
        
        syntax = generate_syntax(
            A=serialize_matrix(self.A),
            B=serialize_matrix(self.B),
            operations=self.operations,
            param_names=(self.param_a, self.param_b)
        )
        
        api_answer = self.answer(operation)
        
        if operation in MATRIX_OPS_ELEMENTWISE:
            return AnswerStepsMathjax(
                question=question,
                answer=api_answer,
                steps=self._elementwise_latex(operation),
                syntax=syntax
            )
            
        else: # operation == "matmul":
            return AnswerStepsMathjax(
                question=question,
                answer=api_answer,
                steps=self._matmul_latex(),
                syntax=syntax
            )
        