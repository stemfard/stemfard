# NOTE: COMPLETED

from collections.abc import Callable
from typing import Any

from sympy import Matrix, ones

from stemfard.core.errors.dimensions import (
    MatrixCompatibilityError, ShapeMismatchError
)
from stemfard.core.errors.general import OperationError


# ----------------------------
# Human-readable operation map
# ----------------------------
MAP_OPERATION = {
    "add": "Matrix addition",
    "subtract": "Matrix subtraction",
    "multiply": "Element-wise matrix multiplication",
    "divide": "Element-wise matrix division",
    "raise": "Element-wise matrix power",
    "matmul": "Matrix multiplication"
}

VALID_MATRIX_ARITHMETICS = tuple(MAP_OPERATION.keys())
ELEMENTWISE_OPS = ("add", "subtract", "multiply", "divide", "raise")

# --------------------------
# Optional broadcasting & expansion
# --------------------------
def broadcast_and_expand_matrices(
    A: Matrix,
    B: Matrix,
    operation: str,
    param_names: tuple[str, str]
) -> tuple[Matrix, Matrix]:
    """
    Validate and broadcast two matrices for elementwise operations.
    Expands 1 x 1 matrices to match the other matrix if needed.

    Parameters
    ----------
    A : sympy.Matrix
    B : sympy.Matrix
    operation : str
        Elementwise operation (add, subtract, multiply, divide, raise)
    param_names : tuple[str, str]
        Names of the parameters for error messages

    Returns
    -------
    tuple[Matrix, Matrix]
        Broadcasted matrices ready for elementwise computation

    Raises
    ------
    ShapeMismatchError
        If the matrices cannot be broadcasted to compatible shapes.
    ValueError
        If the operation is not an allowed elementwise operation.
    
    Notes
    -----
    This function only supports elementwise operations. Matrix
    multiplication (matmul) is not supported and should be handled
    separately.
    """
    if operation not in ELEMENTWISE_OPS:
        raise OperationError(
            operation=operation,
            valid_operations=ELEMENTWISE_OPS
        )
    
    # Determine broadcasted shape
    rows_a, cols_a = A.shape
    rows_b, cols_b = B.shape

    rows = (
        rows_a if rows_a == rows_b or rows_b == 1
        else rows_b if rows_a == 1
        else None
    )
    cols = (
        cols_a if cols_a == cols_b or cols_b == 1
        else cols_b if cols_a == 1
        else None
    )

    if rows is None or cols is None:
        raise ShapeMismatchError(
            operation=MAP_OPERATION[operation],
            param_names=param_names,
            shape_a=A.shape,
            shape_b=B.shape
        )

    # Expand 1×1 matrices to target shape
    def _expand(matrix: Matrix) -> Matrix:
        if matrix.shape == (1, 1) and (rows, cols) != (1, 1):
            return ones(rows, cols) * matrix[0, 0]
        return matrix
    
    return _expand(A), _expand(B)

# -------------------------------------
# Strict shape validation (user option)
# -------------------------------------
def check_matrix_shapes(
    A: Matrix,
    B: Matrix,
    operations: tuple[str, ...],
    param_names: tuple[str, str]
) -> None:
    """
    Validate that matrices A and B have compatible shapes for the
    given operations. This is strict validation, no broadcasting.

    Raises
    ------
    ShapeMismatchError
        If elementwise operations cannot be performed due to shape
        mismatch
    MatrixCompatibilityError
        If matmul shapes are incompatible
    ValueError
        If an unknown operation is supplied
    """
    for op in operations:
        if op in ELEMENTWISE_OPS:
            if A.shape != B.shape:
                raise ShapeMismatchError(
                    operation=MAP_OPERATION[op],
                    param_names=param_names,
                    shape_a=A.shape,
                    shape_b=B.shape
                )
        elif op == "matmul":
            if A.shape[1] != B.shape[0]:
                raise MatrixCompatibilityError(
                    param_names=param_names,
                    shape_a=A.shape,
                    shape_b=B.shape
                )
        else:
            raise OperationError(
                operation=op,
                valid_operations=VALID_MATRIX_ARITHMETICS
            )

# --------------------------------
# Elementwise function application
# --------------------------------
def elementwise_apply(
    A: Matrix,
    B: Matrix | None = None,
    *,
    func: Callable[..., Any],
    operation: str | None = None,
    broadcast: bool = False,
    param_names: tuple[str, str] = ("A", "B")
) -> Matrix:
    """
    Apply a function elementwise to one or two SymPy matrices.

    Parameters
    ----------
    A : sympy.Matrix
        First matrix.
    B : sympy.Matrix, optional
        Second matrix. Must have the same shape as A if
        `broadcast=False`.
    func : Callable
        Function of one (if B is None) or two arguments (if B is given).
    operation : str, optional
        Name of the elementwise operation (e.g. 'add', 'multiply').
        Required if `broadcast=True`.
    broadcast : bool
        If True, allows broadcasting / 1x1 expansion.
    param_names : tuple[str, str]
        Names of matrices for error messages.

    Returns
    -------
    sympy.Matrix
        New matrix with `func` applied elementwise.

    Raises
    ------
    ShapeMismatchError
        If shapes are incompatible and broadcasting is disabled.
    OperationError
        If broadcasting is requested without a valid elementwise 
        operation.

    Notes
    -----
    This function only supports elementwise operations. For matrix
    multiplication (matmul), use dedicated functions.
    """
    # --------------------------
    # Internal helper
    # --------------------------
    def _apply_func(a: Matrix, b: Matrix | None = None) -> Matrix:
        rows, cols = a.shape # use `a` (broadcasted) not `A` (original)
        if b is None:
            return Matrix(rows, cols, lambda i, j: func(a[i, j]))
        return Matrix(rows, cols, lambda i, j: func(a[i, j], b[i, j]))
    
    # Single matrix case, i.e. only A is provided
    if B is None:
        return _apply_func(a=A)
    
    # Two matrices case, i.. both A and B are provided
    if broadcast:
        if operation is None or operation not in ELEMENTWISE_OPS:
            msg = "Broadcasting requires an explicit valid elementwise operation"
            raise OperationError(
                operation=operation,
                valid_operations=ELEMENTWISE_OPS,
                message=msg
            )
        A_broadcasted, B_broadcasted = broadcast_and_expand_matrices(
            A, B, operation=operation, param_names=param_names
        )
    elif A.shape != B.shape:
        raise ShapeMismatchError(
            operation=operation or "Element-wise operation",
            param_names=param_names,
            shape_a=A.shape,
            shape_b=B.shape
        )
    else:
        A_broadcasted, B_broadcasted = A, B

    return _apply_func(A_broadcasted, B_broadcasted)