from functools import cached_property
from typing import Literal

from numpy import around, asarray, dot, float64, matmul, vstack, zeros, zeros_like
from numpy.typing import NDArray
from sympy import Matrix

from stemfard.core._html import html_bg_level2
from stemfard.core._strings import str_color_math, str_remove_tzeros
from stemfard.core._type_aliases import Array2DLike, SequenceArrayLike
from stemfard.core.constants import StemConstants
from stemfard.maths.linalg_iterative._base import (
    BaseLinalgSolveIterative, MatricesLDU
)
from stemfard.maths.linalg_iterative._common import (
    LinalgSolveIterativeResult, bg_matrices_dlu, table_latex
)
from stemfard.core._latex import tex_to_latex_arr
from stemfard.core.models import AnswerStepsResult


def bg_jacobi_algebra() -> list[str]:
    
    steps_mathjax: list[str] = []
    
    steps_mathjax.append(
        "In algebra form, the Jacobi iterative scheme is given by "
        f"\\( \\text{{Equation 1}} \\) below."
    )
    
    eqtn = (
        f"x_{{i}}^{{(k + 1)}} = \\frac{{1}}{{a_{{ii}}}}\\Bigg[b_{{i}} "
        f"- \\sum\\limits_{{j=1 \\:, \\: j \\ne i}}^{{n}}\\Big(a_{{ij}} \\: "
        f"x_{{j}}^{{(k)}}\\Big)\\Bigg] \\qquad \\cdots \\qquad (1)"
    )
    
    steps_mathjax.append(f"\\( \\displaystyle\\quad {str_color_math(eqtn)} \\)")
    steps_mathjax.append(
        "for \\( i = 1, \\: 2, \\: \\cdots, \\: n \\) and "
        "\\( k = 0, \\: 1, \\: \\cdots, \\: n - 1 \\)"
    )
    
    steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
        
    return steps_mathjax


def bg_jacobi_matrix(
    abs_rel_tol: Literal["atol", "rtol"] = "atol"
) -> list[str]:
    
    steps_mathjax: list[str] = []
    
    steps_mathjax.append(
        "In matrix form, the Jacobi iterative scheme is given by Equation "
        f"\\( {str_color_math(value='(1)')} \\)."
    )
    steps_temp = (
        f"x^{{(k + 1)}} = T_{{j}} \\: x^{{(k)}} + c_{{j}} "
        "\\qquad \\cdots \\qquad (1)"
    )
    steps_mathjax.append(
        f"\\( \\qquad {str_color_math(value=steps_temp)} \\)"
    )
    steps_mathjax.append(
        "for \\( k = 0, \\: 1, \\: 2, \\: \\cdots, \\: n-1 \\)"
    )
    steps_mathjax.append("where")
    steps_mathjax.append(
        f"\\( \\quad T_{{j}} = D^{{-1}}(L + U)  \\quad \\) and "
        f"\\(\\quad c_{{j}} = D^{{-1}} b \\)"
    )
    
    steps_mathjax.extend(bg_matrices_dlu(abs_rel_tol=abs_rel_tol))
    
    return steps_mathjax


class LinalgSolveJacobi(BaseLinalgSolveIterative):
    "Jacobi iteration steps (algebra and matrix)"
    
    def calc_jacobi(self, abs_rel_tol) -> NDArray[float64]:
        X = []
        x = self.x0
        tolerance = self.atol if abs_rel_tol == "atol" else self.rtol
        
        for _ in range(self.maxit):
            X.append(x.flatten().tolist())
            x_new = zeros_like(x)
            
            for i in range(self.nrows):
                s1 = dot(self.A[i, :i], x[:i])
                s2 = dot(self.A[i, i + 1:], x[i + 1:])
                x_new[i] = (self.b[i] - s1 - s2) / self.A[i, i]

            kth_norm = self.kth_norm(x_new=x_new, x=x, abs_rel_tol=abs_rel_tol)
            
            if kth_norm < tolerance:
                break
            
            x = x_new.copy()
            
        return vstack((asarray(X), x.reshape(1, -1)))
    
    
    @cached_property
    def _table_latex(self) -> LinalgSolveIterativeResult:
        arr = self.calc_jacobi(abs_rel_tol=self.abs_rel_tol)
        result = table_latex(
            arr=arr, col_names=self.fvars, decimals=self.decimals
        )
        
        return LinalgSolveIterativeResult(
            arr=arr,
            nrows=result["nrows"],
            solution=result["solution"],
            latex=result["latex"]
        )
        
    
    @cached_property
    def _table_conclusion(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        table_results = self._table_latex
        
        steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
        
        steps_mathjax.append(
            "The approximations obtained in all the above iterations are "
            "presented in the table below."
        )
        
        steps_mathjax.append(f"\\( {table_results.latex} \\)")
        
        steps_mathjax.append(
            "The values in the last row of the table above gives the final"
            " approximated solution. These are extracted and presented below."
        )
        
        for index in range(self.nrows):
            steps_mathjax.append(
                f"\\( x_{{{index + 1}}} "
                f"= {table_results.solution[index]} \\)"
            )
        
        return {
            "latex": [
                StemConstants.BORDER_HTML_BLUE_WIDTH_2,
                f"\\( {table_results.latex} \\)"
            ],
            "steps": steps_mathjax
        }
    
    
    @cached_property
    def _jacobi_algebra(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        table_jacobi = self._table_latex
        
        x = zeros((self.ncols, 1))
        x_rnd = x.copy()
        x_new = x.copy()
        
        if self.steps_bg:
            temp_steps = bg_jacobi_algebra()
            steps_mathjax.extend(temp_steps)
        
        if self.steps_compute:
            steps_mathjax.append(
                f"The Jacobi representation of the given system of "
                f"\\( {str_color_math(self.nrows)} \\) linear equations is "
                f"given by Equations \\( {str_color_math('1.1')} \\) "
                f"through \\( {str_color_math(f'1.{self.nrows}')} \\) "
                "presented below."
            )
            
            steps_temp: list[str] = []
            
            for i in range(self.nrows):
                steps_temp.append(
                    f"\\( \\displaystyle\\quad x_{{i + 1}}^{{(k + 1)}} "
                    f"= \\frac{{1}}{{{self.A_rnd[i, i]}}} \\: "
                    f"\\Big[{self.beta_rnd[i, 0]} "
                )
                
                for j in range(self.ncols):
                    if i != j:
                        steps_temp.append(
                            f"+ {self.A_rnd[i, j]} \\: x_{{j + 1}}^{{(k)}} "
                        )
                
                steps_temp.append(
                    "\\Big] \\qquad \\cdots \\qquad "
                    f"{str_color_math(f'1.{i + 1}')} \\)____"
                )
                
            steps_mathjax.extend("".join(steps_temp).split("____"))
            steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
            
            steps_mathjax.append(
                "The iterations are performed using the above Equations as is "
                "illustrated below."
            )
            steps_mathjax.append(
                f"\\( \\color{{green}}{{\\textbf{{Remarks:}}}} \\) The values "
                f"of \\( b_{{i}} \\) and \\( a_{{ij}} \\: (i, \\: j = 1, \\: "
                f"\\cdots, {self.nrows}) \\) in Equations "
                f"\\( {str_color_math('(1.j.k)')} \\) in the steps that "
                "appear below are:"
            )
            steps_mathjax.append(
                f"\\( \\quad {str_color_math(f'b_{{i}}')} \\longleftarrow \\) "
                f"Element in the \\( i^{{th}} \\) position of the vector of "
                "constant (RHS)."
            )
            steps_mathjax.append(
                f"\\( \\quad {str_color_math(f'a_{{ij}}')} \\longleftarrow \\) "
                f"Element in the \\( i^{{th}} \\) row and \\( j^{{th}} \\) "
                "column of the coefficients matrix."
            )
            
            for k in range(table_jacobi.nrows - 1):
                steps_temp = html_bg_level2(title=f"Iteration {k + 1}")
                steps_mathjax.append(steps_temp)
                
                s1 = [f"x_{{{sub + 1}}}^{{({k})}}" for sub in range(self.ncols)]
                s2 = [
                    f"x_{{{sub + 1}}}^{{({k + 1})}}"
                    for sub in range(self.ncols)
                ]
                if k == 0:
                    steps_mathjax.append(
                        f"This iteration uses the given initial values of "
                        f"\\( {', '.join(s1)} \\) to approximate the "
                        f"values \\( {', '.join(s2)} \\)."
                    )
                else:
                    steps_mathjax.append(
                        f"This iteration uses the values of "
                        f"\\( {', '.join(s1)} \\) calculated in iteration "
                        f"{k} above to approximate the values "
                        f"\\( {', '.join(s2)} \\)."
                    )
                
                for i in range(self.nrows):
                    steps_mathjax.append(
                        f"Let \\( k = {k} \\) in Equation "
                        f"\\( {str_color_math(f'(1.{i + 1})')} \\) to "
                        "get Equation "
                        f"\\( {str_color_math(f'(1.{i + 1}.{k + 1})')} \\) "
                        "below."
                    )
                    
                    steps_temp: list[str] = []
                    
                    steps_temp.append(
                        f"\\(\\displaystyle x_{{{i + 1}}}^{{({k + 1})}} "
                        f"= \\frac{{1}}{{a_{{{i + 1}{{{i + 1}}}}}}}"
                        f"\\Big[b_{{{i + 1}}}")
                    
                    # line 1
                    
                    for j in range(self.ncols):
                        if i != j:
                            steps_temp.append(
                                f" - a_{{{i + 1}{j + 1}}} \\: "
                                f"{str_color_math(f'x_{{{j + 1}}}^{{({k})}}')}"
                            )
    
                    steps_temp.append(
                        f"\\Big] \\qquad \\cdots \\qquad "
                        f"{str_color_math(f'(1.{i + 1}.{k + 1})')} \\)"
                    )
                    steps_mathjax.append("".join(steps_temp))
                        
                    # line 2
                    
                    steps_temp: list[str] = []
                    
                    steps_temp.append(
                        f"\\( \\displaystyle\\quad "
                        f"= \\frac{{1}}{{{self.A_rnd[i, i]}}} "
                        f"\\Big[{self.beta_rnd[i, 0]}"
                    )
                    
                    for j in range(self.ncols):
                        if i != j:
                            steps_temp.append(
                                f"+ {self.A_rnd[i, j]} \\: "
                                f"\\left( {str_color_math(x_rnd[j, 0])} "
                                "\\right)"
                            )
                            
                    steps_temp.append("\\Big] \\)")
                    steps_mathjax.append("".join(steps_temp))
                    
                    # line 3
                    
                    total = 0
                    
                    steps_temp: list[str] = []
                    
                    steps_temp.append(
                        f"\\( \\displaystyle\\quad "
                        f"= \\frac{{1}}{{{self.A_rnd[i, i]}}} "
                        f"\\: \\Big[{self.beta_rnd[i, 0]}"
                    )
                    
                    for j in range(self.ncols):
                        if i != j:
                            Ax = self.A[i, j] * x[j, 0]
                            total += Ax
                            steps_temp.append(
                                f" + {float(around(Ax, self.decimals))}"
                            )
                    steps_temp.append("\\Big] \\)")
                    steps_mathjax.append("".join(steps_temp))
                    
                    # line 4 - 5
                    
                    bi_minus_total_rnd = float(
                        around((self.b[i, 0] - total), self.decimals)
                    )
                    steps_mathjax.append(
                        f"\\( \\displaystyle\\quad "
                        f"= \\frac{{{bi_minus_total_rnd}}}{{{self.A_rnd[i, i]}}} \\)"
                    )
                    
                    result = 1 / self.A[i, i] * (self.b[i, 0] - total)
                    steps_mathjax.append(
                        f"\\( \\quad "
                        f"= {float(around(result, self.decimals))} \\)"
                    )

                    steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
                    
                    x_new[i, 0] = result
                     
                x = x_new.copy()
                x_rnd = around(x, self.decimals)
                
                steps_mathjax.append(
                    "A summary of approximations from iteration "
                    f"\\( {k + 1} \\) is as follows."
                )
                
                for index in range(self.nrows):
                    steps_mathjax.append(
                        f"\\( x_{{{index + 1}}} = {x_rnd[index, 0]} \\)"
                    )
                
                steps_mathjax.append(
                    "These approximations will be used in iteration "
                    f"\\( {k + 2} \\) below."
                )
        
        if self.steps_compute:
            steps_mathjax.extend(self._table_conclusion["steps"])
        else:
            steps_mathjax.extend(self._table_conclusion["latex"])
        
        return AnswerStepsResult(
            answer=table_jacobi.arr,
            steps=[str_remove_tzeros(step) for step in steps_mathjax]
        )
    
    
    @cached_property
    def _jacobi_matrix(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        table_jacobi = self._table_latex
        matrices_ldu: MatricesLDU = self.matrices_dlu
        
        if self.steps_bg:
            temp_steps = bg_jacobi_matrix(abs_rel_tol=self.abs_rel_tol)
            steps_mathjax.extend(temp_steps)
            
        if self.steps_compute:
            steps_mathjax.extend(matrices_ldu.latex)
            
            steps_mathjax.append(
                f"With the above matrices, the values of \\( T_{{j}} \\) and "
                f"\\( c_{{j}} \\) in Equation \\( {str_color_math('(1)')} \\) "
                "are calculated as follows."
            )
            
            steps_mathjax.append(
                f"\\( { str_color_math('T_{{j}} = D^{{-1}}(L + U)') } \\)"
            )
            
            l_latex = (
                matrices_ldu.L.replace("left[", "left(").replace("right]", "right)")
            )
            u_latex = (
                matrices_ldu.U.replace("left[", "left(").replace("right]", "right)")
            )
            
            steps_mathjax.append(
                "\\( \\quad "
                f"= {matrices_ldu.D}^{{\\:-1}} \\left[{l_latex} "
                f"+ {u_latex} \\right] \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(self.D_inv_rnd)} "
                f"{matrices_ldu.LU} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(self.tj_rnd)} \\)"
            )
            
            steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
            
            steps_mathjax.append(
                f"\\( {str_color_math(f'c_{{j}} = D^{{-1}}~b')} \\)"
            )
            steps_mathjax.append(
                "\\( \\quad "
                f"= {matrices_ldu.D}^{{\\:-1}} {tex_to_latex_arr(self.b)} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = {tex_to_latex_arr(self.cj_rnd)} \\)"
            )
            
            steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
            
            steps_mathjax.append(
                f"Substitute the above values of \\( T_{{j}} \\) and "
                f"\\( c_{{j}} \\) into  Equation "
                f"\\( {str_color_math('(1)')} \\) as follows."
            )
            steps_mathjax.append(
                f"\\( x^{{(k + 1)}} = T_{{j}} \\: "
                f"{str_color_math('x^{(k)}')} + c_{{j}} \\qquad \\cdots "
                f"\\qquad {str_color_math('(2)')} \\)"
            )
            
            tj_latex = tex_to_latex_arr(self.tj_rnd, brackets="[]")
            xj_latex = tex_to_latex_arr(self.xj_underscore, brackets="[]")
            cj_latex = tex_to_latex_arr(self.cj_rnd, brackets="[]")
            
            steps_mathjax.append(
                f"\\( \\quad = {tj_latex} \\: "
                f"{str_color_math(xj_latex)} + {cj_latex} \\)"
            )
            steps_mathjax.append(
                f"Iterations are performed using Equation "
                f"\\( {str_color_math('(2)')} \\) above. These iterations "
                "are presented below."
            )
            
            xj_k = self.x0
            xj_k_rnd = around(xj_k, self.decimals)
                      
            for k in range(table_jacobi.nrows - 1):
                
                steps_temp = html_bg_level2(title=f"Iteration {k + 1}")
                steps_mathjax.append(steps_temp)
                
                if k == 0:
                    steps_mathjax.append(
                        f"This iteration uses the vector of initial guesses "
                        f"\\( {str_color_math(f'x^{{({k})}}')} \\) "
                        "to approximate the values of "
                        f"\\( {str_color_math(f'x^{{({k + 1})}}')} \\)."
                    )
                else:
                    steps_mathjax.append(
                        f"This iteration uses the result "
                        f"\\( {str_color_math(f'x^{{({k})}}')} \\) "
                        f"obtained in iteration {k} above to approximate the "
                        f"values of "
                        f"\\( {str_color_math(f'x^{{({k + 1})}}')} \\)."
                    )
                
                steps_mathjax.append(
                    f"Let \\( k = {k} \\) in Equation "
                    f"\\( {str_color_math('(2)')} \\) to get Equation "
                    f"\\( {str_color_math(f'(2.{k + 1})')} \\) below."
                )
                steps_mathjax.append(
                    f"\\( \\quad x^{{({k + 1})}} "
                    f"= T_{{j}} \\: {str_color_math(f'x^{{({k})}}')} "
                    f"+ c_{{j}} \\qquad \\cdots \\qquad "
                    f"{str_color_math(f'2.{k + 1}')} \\)"
                )
                
                if k == 0:
                    steps_mathjax.append(
                        f"where \\( {str_color_math('x^{(0)}')} \\) is the "
                        "vector of initial guesses."
                    )
                else:
                    steps_mathjax.append(
                        f"where \\( {str_color_math(f'x^{{({k})}}')} \\) "
                        "is the approximated solution from iteration "
                        f"\\( {k} \\) above."
                    )
                
                xj_k_latex = str_color_math(tex_to_latex_arr(arr=xj_k_rnd))
                 
                steps_mathjax.append(
                    f"\\( x^{{({k + 1})}} = {tj_latex} {xj_k_latex} "
                    f"+ {cj_latex} \\)"
                )
                
                tj_xj = matmul(self.tj, xj_k)
                steps_mathjax.append(
                    f"\\( \\quad "
                    f"= {tex_to_latex_arr(around(tj_xj, self.decimals))} "
                    f"+ {cj_latex} \\)"
                )
                
                xj_k = tj_xj + self.cj # Tj * xj^(k) + cj
                xj_k_rnd = around(xj_k, self.decimals)
                steps_mathjax.append(
                    f"\\( \\quad = {tex_to_latex_arr(xj_k_rnd)} \\)"
                )
                           
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
                  
                steps_mathjax.append(
                    "From the above results,  summary of approximations from iteration "
                    f"\\( {k + 1} \\) is as follows "
                    "(i.e. they are obtained from the above result)."
                )
                
                for index in range(self.nrows):
                    steps_mathjax.append(
                        f"\\( x_{{{index + 1}}} = {xj_k_rnd[index, 0]} \\)"
                    )
                
                steps_mathjax.append(
                    "These approximations will be used in iteration "
                    f"\\( {k + 2} \\) below."
                )
        
        if self.steps_compute:
            steps_mathjax.extend(self._table_conclusion["steps"])
        else:
            steps_mathjax.extend(self._table_conclusion["latex"])
        
        return AnswerStepsResult(
            answer=table_jacobi.arr,
            steps=steps_mathjax
        )