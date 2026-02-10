from functools import cached_property
from typing import Literal

from numpy import asarray, dot, float64, vstack, zeros_like
from numpy.typing import NDArray

from stemfard.maths.linalg_iterative._base import BaseLinalgSolveIterative
from stemfard.core.models import AnswerStepsResult


def bg_gauss_seidel_algebra(abs_rel_tol: Literal["atol", "rtol"] = "atol") -> list[str]:
    
    steps_mathjax: list[str] = []
    
    return steps_mathjax
    

def bg_gauss_seidel_matrix(abs_rel_tol: Literal["atol", "rtol"] = "atol") -> list[str]:
    
    steps_mathjax: list[str] = []
    
    return steps_mathjax


class LinalgSolveGaussSeidel(BaseLinalgSolveIterative):
    "Gauss-Seidel iteration steps (algebra and matrix)"
    
    @cached_property
    def calc_gauss_seidel(self, abs_rel_tol) -> NDArray[float64]:
        X = []
        x = self.x0
        x_new = zeros_like(x)
        tolerance = self.atol if abs_rel_tol == "atol" else self.rtol
        
        for _ in range(1, self.maxit+1):
            X.append(x.tolist())
            
            for i in range(self.nrows):
                s1 = dot(self.A[i, :i], x_new[:i])
                s2 = dot(self.A[i, i + 1:], x[i + 1:])
                x_new[i] = (self.b[i] - s1 - s2) / self.A[i, i]
                
            kth_norm = self.kth_norm(x_new=x_new, x=x, abs_rel_tol=abs_rel_tol)
        
            if kth_norm < tolerance:
                break
            
            x = x_new.copy()
        
        return vstack((asarray(X), x.reshape(1, -1)))
        
    
    @cached_property
    def _gauss_seidel_algebra(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        return AnswerStepsResult(
            answer=[99, 99],
            steps=steps_mathjax
        )
    
    
    @cached_property
    def _gauss_seidel_matrix(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        return AnswerStepsResult(
            answer=[99, 99],
            steps=steps_mathjax
        )