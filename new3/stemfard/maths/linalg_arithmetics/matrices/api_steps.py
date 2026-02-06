from dataclasses import dataclass

from numpy import arange, zeros
from sympy import latex

from stemfard.core._latex import tex_to_latex, tex_to_latex_arr
from stemfard.maths.linalg_arithmetics.matrices.api_bg import MatrixArithmeticsBackground
from stemfard.core.constants import StemConstants
from stemfard.core._strings import str_color_math, str_remove_tzeros
from stemfard.core.rounding import fround
from stemfard.core._io import present_result
from stemfard.core._is_dtypes import is_negative
from stemfard.maths.linalg_arithmetics.matrices.api_answer import matrix_arithmetics_ans
from stemfard.core._html import html_bg_level2
from stemfard.core.signs import abs_value, pm_sign


@dataclass
class FloatsCount:
    float_n: int
    non_float_n: int


class MatrixArithmetics(MatrixArithmeticsBackground):
    """Matrix arithmetics"""
    
    def answer(self, operation):
        answer = matrix_arithmetics_ans(
            A=self.A,
            B=self.B,
            broadcast=self.broadcast,
            operation=operation,
            decimals=self.decimals
        )
        return answer
    
    
    def user_matrices(self, operation: str) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append(
            f"For the matrices \\( {self.param_a} \\) and "
            f"\\( {self.param_b} \\) given below."
        )
        steps_mathjax.append(
            f"\\( \\quad {self.param_a} = {self.a_latex} \\:, "
            f"\\quad {self.param_b} \\: {str_color_math('=')} \\: "
            f"{self.b_latex} \\)"
        )
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        if operation == "raise":
            steps_mathjax.append(
                f"The result, \\( {self.param_a}\\:^{{{self.param_b}}} \\), is "
                f"obtained by raising each individual element of matrix "
                f"\\( {self.param_a} \\) to the corresponding element of "
                f"matrix \\( {self.param_b} \\) as follows."
            )
        else:
            _map = {
                "add": (
                    "adding", 
                    "sum",
                    f"{self.param_a} + {self.param_b}"
                ),
                "subtract": (
                    "subtracting",
                    "difference",
                    f"{self.param_a} - {self.param_b}"
                ),
                "multiply": (
                    "multiplying",
                    "element-wise product",
                    f"{self.param_a} \\times {self.param_b}"
                ),
                "divide": (
                    "dividing",
                    "element-wise quotient",
                    f"{self.param_a} \\div {self.param_b}"
                ),
            }
            
            mapped = _map[operation]
            steps_mathjax.append(
                f"The {mapped[1]}, \\( {mapped[2]} \\), is obtained "
                f"by {mapped[0]} the corresponding elements of matrices "
                f"\\( {self.param_a} \\) and \\( {self.param_b} \\) as "
                "follows."
            )
        
        return steps_mathjax
    
    
    @property
    def linalg_add_latex(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_linalg_add)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
        
        steps_mathjax.extend(self.user_matrices(operation="add"))
        
        decimals = self.decimals
        
        if decimals > -1:
            A, B = self.A_rnd, self.B_rnd
        else:
            A, B = self.A, self.B
            
        result = self.answer(operation="add")
        result_rnd = fround(result, decimals=decimals)
                
        result_step_1 = self.zeros_array
        
        for row in range(self.nrows):
            for col in range(self.nrows):
                A_ij = A[row, col]
                B_ij = B[row, col]
                A_ij_latex = tex_to_latex(A_ij, decimals=decimals)
                B_ij_latex = str_color_math(
                    tex_to_latex(B_ij, decimals=decimals)
                )
                
                if is_negative(B_ij):
                    B_ij_latex = str_color_math(f"\\left({B_ij_latex}\\right)")
                
                result_step_1[row, col] = f"{A_ij_latex} + {B_ij_latex}"

        steps_mathjax.append(
            f"\\( {self.result_name} = {self.param_a} + {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} + {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )
        
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
                    
        return steps_mathjax


    @property
    def linalg_subtract_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_linalg_subtract)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
            
        steps_mathjax.extend(self.user_matrices(operation="subtract"))
        
        decimals = self.decimals
        
        if decimals > -1:
            A, B = self.A_rnd, self.B_rnd
        else:
            A, B = self.A, self.B
            
        result = self.answer(operation="subtract")
        result_rnd = fround(result, decimals=decimals)
                
        result_step_1 = self.zeros_array
        
        for row in range(self.nrows):
            for col in range(self.nrows):
                A_ij = A[row, col]
                B_ij = B[row, col]
                A_ij_latex = tex_to_latex(A_ij, decimals=decimals)
                B_ij_latex = str_color_math(
                    tex_to_latex(B_ij, decimals=decimals)
                )
                
                if is_negative(B_ij):
                    B_ij_latex = str_color_math(f"\\left({B_ij_latex}\\right)")
                
                result_step_1[row, col] = f"{A_ij_latex} - {B_ij_latex}"

        steps_mathjax.append(
            f"\\( {self.result_name} = {self.param_a} - {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} - {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )
        
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
        
        return steps_mathjax

    
    @property
    def linalg_multiply_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_linalg_multiply)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
            
        steps_mathjax.extend(self.user_matrices(operation="multiply"))
        
        decimals = self.decimals
        
        if decimals > -1:
            A, B = self.A_rnd, self.B_rnd
        else:
            A, B = self.A, self.B
            
        result = self.answer(operation="multiply")
        result_rnd = fround(result, decimals=decimals)
                
        result_step_1 = self.zeros_array
        
        for row in range(self.nrows):
            for col in range(self.nrows):
                A_ij = A[row, col]
                B_ij = B[row, col]
                A_ij_latex = tex_to_latex(A_ij, decimals=decimals)
                B_ij_latex = str_color_math(
                    tex_to_latex(B_ij, decimals=decimals)
                )
                
                if is_negative(B_ij):
                    B_ij_latex = str_color_math(f"\\left({B_ij_latex}\\right)")
                
                result_step_1[row, col] = f"{A_ij_latex} \\times {B_ij_latex}"

        steps_mathjax.append(
            f"\\( {self.result_name} "
            f"= {self.param_a} \\times {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} \\times {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )
        
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
        
        return steps_mathjax

    
    @property
    def linalg_divide_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_linalg_divide)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
            
        steps_mathjax.extend(self.user_matrices(operation="divide"))
                
        decimals = self.decimals
        
        if decimals > -1:
            A, B = self.A_rnd, self.B_rnd
        else:
            A, B = self.A, self.B
        
        result = self.answer(operation="divide")
        result_rnd = fround(result, decimals=decimals)
                
        result_step_1 = self.zeros_array
        result_step_2 = result_step_1.copy()
        
        for row in range(self.nrows):
            for col in range(self.nrows):
                A_ij = A[row, col]
                B_ij = B[row, col]
                A_ij_latex = tex_to_latex(A_ij, decimals=decimals)
                B_ij_latex = str_color_math(
                    tex_to_latex(B_ij, decimals=decimals)
                )
                
                if is_negative(B_ij):
                    B_ij_latex = str_color_math(f"\\left({B_ij_latex}\\right)")    
                result_step_1[row, col] = f"{A_ij_latex} \\div {B_ij_latex}"
                
                if (B_ij == 0 or "." in str_remove_tzeros(B_ij)):
                    result_step_2[row, col] = (f"{A_ij_latex} \\div {B_ij_latex}")
                else:
                    B_ij_recp = tex_to_latex(1 / B_ij, decimals=decimals)
                    B_ij_reciprocal_latex = str_color_math(B_ij_recp)
                    if is_negative(B_ij):
                        B_ij_reciprocal_latex = (
                            f"\\left({B_ij_reciprocal_latex}\\right)"
                        )
                    result_step_2[row, col] = (
                        f"{A_ij_latex} \\times {B_ij_reciprocal_latex}"
                    )

        steps_mathjax.append(
            f"\\( {self.result_name} = {self.param_a} \\div {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} \\div {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )
        
        if self.reciprocal_action is not None:
            steps_mathjax.append(self.reciprocal_action)
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(result_step_2)} \\)"
            )
            
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
        
        return steps_mathjax

    
    @property
    def linalg_raise_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_linalg_raise)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
            
        steps_mathjax.extend(self.user_matrices(operation="raise"))
        
        decimals = self.decimals
        
        if decimals > -1:
            A, B = self.A_rnd, self.B_rnd
        else:
            A, B = self.A, self.B
            
        result = self.answer(operation="raise")
        result_rnd = fround(result, decimals=decimals)
                
        result_step_1 = self.zeros_array
        
        for row in range(self.nrows):
            for col in range(self.nrows):
                A_ij = A[row, col]
                B_ij = B[row, col]
                A_ij_latex = tex_to_latex(A_ij, decimals=decimals)
                B_ij_latex = str_color_math(
                    tex_to_latex(B_ij, decimals=decimals)
                )
                
                if is_negative(B_ij):
                    B_ij_latex = str_color_math(f"\\left({B_ij_latex}\\right)")
                
                result_step_1[row, col] = f"{A_ij_latex} \\:^ {B_ij_latex}"

        steps_mathjax.append(
            f"\\( {self.result_name} = {self.param_a} \\:^ {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} \\:^ {self.b_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {tex_to_latex_arr(result_step_1)} \\)"
        )
        
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
        
        return steps_mathjax
    
    
    @property
    def linalg_matmul_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        if self.steps_bg:
            steps_mathjax.extend(self.bg_matmul)
            steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
        
        steps_mathjax.append(
            f"For the matrices \\( {self.param_a} \\) and "
            f"\\( {self.param_b} \\) given below."
        )
        steps_mathjax.append(
            f"\\( \\quad {self.param_a} = {self.a_latex} \\:, "
            f"\\quad {self.param_b} \\: {str_color_math('=')} \\: "
            f"{self.b_latex} \\)"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            f"The product \\( {self.result_name} "
            f"= {self.param_a} \\cdot {self.param_b} \\) is evaluated as "
            "follows."
        )
        
        result = self.answer(operation="matmul")
        result_rnd = fround(result, decimals=self.decimals)
        
        title = (
            "STEP 1: Determine the dimensions of each of the above matrices"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(f"\\( \\text{{Matrix }} {self.param_a} \\)")
        steps_mathjax.append(
            f"\\( \\quad\\text{{Rows }}: {self.nrows_a} \\:, "
            f"\\quad \\text{{Columns }}: {self.ncols_a} \\)"
        )
        
        steps_mathjax.append(f"\\( \\text{{Matrix }} {self.param_b} \\)")
        steps_mathjax.append(
            f"\\( \\quad\\text{{Rows }}: {self.nrows_b} \\:, "
            f"\\quad \\text{{Columns: }} {self.ncols_b} \\)"
        )
        
        title = (
            "STEP 2: Understand the dimensions of the product matrix"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The product matrix \\( {self.result_name} \\) will be a "
            f"\\( {self.nrows_a} \\) by \\( {self.ncols_b} \\) matrix. That "
            "is, it will be a matrix whose dimensions is the number of "
            f"\\( \\textit{{rows}} \\) of matrix \\( {self.param_a} \\) by "
            f"the number of \\( \\textit{{columns}} \\) of matrix "
            f"\\( {self.param_b} \\)."
        )
        
        title = "STEP 3: Multiply the matrices"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( {self.result_name} "
            f"= {self.param_a} \\cdot {self.param_b} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.a_latex} \\cdot {self.b_latex} \\)"
        )
        
        A = self.A
        B = self.B
        nrows_a, ncols_a = A.shape
        nrows_b, ncols_b = B.shape
        AB = zeros((nrows_a, ncols_b), dtype=object)
        AB_mul = AB.copy()
        decimals = self.decimals
        
        for row in range(nrows_a):
            for col in range(ncols_b):
                # initialize a string that will store values at each 
                # iteration for steps 1 and 2
                row_index_step1 = [""] * nrows_b
                row_index_step2 = row_index_step1.copy()
                for k in arange(ncols_a):
                    B_kj_latex = latex(fround(B[k, col], decimals))
                    if is_negative(B[k, col]):
                        B_kj_latex = (f"\\left({B_kj_latex}\\right)")
                    A_ik_latex = latex(fround(A[row, k], decimals))
                    if is_negative(A[row, k]):
                        A_ik_latex = (f"\\left({A_ik_latex}\\right)")
                    row_index_step1[k] = (
                        f"{A_ik_latex} \\times {str_color_math(B_kj_latex)}"
                    )
                    # if it is the first value of the column, just give 
                    # the value as it is (i.e. do not include the + or - 
                    # sign of the pm_sign() function)
                    AB_ij = fround(A[row, k] * B[k, col], decimals)
                    if k == 0:
                        row_index_step2[k] = latex(AB_ij)
                    else:
                        row_index_step2[k] = (
                            pm_sign(AB_ij) + latex(abs_value(AB_ij, tolatex=False))
                        )
                # replace each ith and jth index by the results above 
                # joined together
                ab1_ij = " + ".join("".join(line) for line in row_index_step1)
                ab2_ij = " + ".join("".join(line) for line in row_index_step2)
                AB[row, col] = ab1_ij
                AB_mul[row, col] = ab2_ij

        steps_mathjax.append(f"\\( \\quad = {tex_to_latex_arr(AB)} \\)")
        steps_mathjax.append(f"\\( \\quad = {tex_to_latex_arr(AB_mul)} \\)") 
        steps_mathjax.append(f"\\( \\quad {present_result(result_rnd)} \\)")
        
        return str_remove_tzeros(steps_mathjax)