from functools import cached_property
from typing import Literal

from numpy import asarray, dot, float64, vstack
from numpy.typing import NDArray

from stemfard.maths.linalg_iterative._base import BaseLinalgSolveIterative
from stemfard.core.models import AnswerStepsResult


def bg_sor_algebra(abs_rel_tol: Literal["atol", "rtol"] = "atol") -> list[str]:
    
    steps_mathjax: list[str] = []
    
    return steps_mathjax
    

def bg_sor_matrix(abs_rel_tol: Literal["atol", "rtol"] = "atol") -> list[str]:
    
    steps_mathjax: list[str] = []
    
    return steps_mathjax


class LinalgSolveSOR(BaseLinalgSolveIterative):
    "Successive-over Relaxation iteration steps (algebra and matrix)"
    
    @cached_property
    def calc_sor(self, abs_rel_tol) -> NDArray[float64]:
        X = []
        x = self.x0
        w = self.omega
        tolerance = self.atol if abs_rel_tol == "atol" else self.rtol
        
        for k in range(self.maxit):
            X.append(x.tolist())
            x_new  = x.copy()
            
            for i in range(self.nrows):
                s1 = dot(self.A[i, :i], x[:i])
                s2 = dot(self.A[i, i + 1:], x_new[i + 1:])
                x[i] = (
                    x[i] * (1 - w) + (w / self.A[i, i]) * (self.b[i] - s1 - s2)
                )
            
            kth_norm = self.kth_norm(x_new=x_new, x=x, abs_rel_tol=abs_rel_tol)
        
            if kth_norm < tolerance:
                break
            
            x = x_new.copy()
        
        return vstack((asarray(X), x.reshape(1, -1)))
    
    
    @cached_property
    def _sor_algebra(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        return AnswerStepsResult(
            answer=[99, 99],
            steps=steps_mathjax
        )
    
    
    @cached_property
    def _sor_matrix(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        return AnswerStepsResult(
            answer=[99, 99],
            steps=steps_mathjax
        )