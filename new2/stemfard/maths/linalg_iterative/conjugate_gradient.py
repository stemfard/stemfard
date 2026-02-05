from functools import cached_property
from typing import Literal

from numpy import asarray, dot, float64, inf, vstack
from numpy.linalg import inv, norm
from numpy.typing import NDArray

from stemfard.maths.linalg_iterative._base import BaseLinalgSolveIterative
from stemfard.core.models import AnswerStepsResult


def bg_conjugate_gradient(
    abs_rel_tol: Literal["atol", "rtol"] = "atol"    
) -> list[str]:
    
    steps_mathjax: list[str] = []
    
    return steps_mathjax


class LinalgSolveConjugateGradient(BaseLinalgSolveIterative):
    "Conjugate gradient matrix steps"
    
    @cached_property
    def calc_conjugate_gradient(self, abs_rel_tol) -> NDArray[float64]:
        X = []
        x = self.x0
        # begin computations
        C_inv = inv(self.C)
        r = self.b - dot(self.A, x)
        w = dot(C_inv, r)
        v = dot(C_inv.T, w)
        alpha = dot(w, w)
        # step 3
        k = 1
        
        tolernce = self.atol if self.atol else self.rtol
        
        while k <= self.maxit:
            X.append(x.flatten().tolist())
            # step 4
            if norm(v, inf) < tolernce:
                break
            # step 5
            u = dot(self.A, v)
            t = alpha / dot(v, u)
            x = x + t * v
            r = r - t * u
            w = dot(C_inv, r)
            beta = dot(w, w)
            # step 6
            if abs(beta) < tolernce:
                if norm(r, inf) < tolernce:
                    break
            # step 7
            s = beta/alpha
            v = dot(C_inv.T, w) + s * v
            alpha = beta
            k = k + 1
        
        return vstack((asarray(X), x.reshape(1, -1)))
    
    
    @cached_property
    def _conjugate_gradient_algebra(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        return AnswerStepsResult(
            answer=[99, 99],
            steps=steps_mathjax
        )