from collections.abc import Callable
from sympy import Matrix


def align_matrices(A, B) -> tuple[Matrix, Matrix]:
    
    A = Matrix(A)
    B = Matrix(B)
    
    # If A is 1x1 and B is larger, broadcast A
    if A.shape == (1, 1) and B.shape != (1, 1):
        A = Matrix.ones(*B.shape) * A[0, 0]
    
    # If B is 1x1 and A is larger, broadcast B
    elif B.shape == (1, 1) and A.shape != (1, 1):
        B = Matrix.ones(*A.shape) * B[0, 0]
    
    return A, B


def elementwise_apply(
    A: Matrix, func: Callable, B: Matrix | None = None
) -> Matrix:
    """
    Apply a function elementwise to one or two SymPy matrices.

    Parameters
    ----------
    A : sympy.Matrix
        First matrix.
    func : Callable
        Function of one or two arguments.
    B : sympy.Matrix, optional
        Second matrix. If given, must have the same shape as A.

    Returns
    -------
    sympy.Matrix
        New matrix with func applied elementwise.
    """
    rows, cols = A.shape

    if B is None:
        return Matrix(rows, cols, lambda i, j: func(A[i, j]))
    else:
        if A.shape != B.shape:
            raise ValueError(
                "Expected both matrices to have same shape, "
                f"got {A.shape} vs {B.shape}."
            )
        return Matrix(rows, cols, lambda i, j: func(A[i, j], B[i, j]))