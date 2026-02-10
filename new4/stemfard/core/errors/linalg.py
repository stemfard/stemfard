class SingularMatrixError(ValueError):
    """
    Raised when a matrix is singular (determinant = 0) and cannot be
    invertedor used in an operation requiring non-singular matrices.
    """
    def __init__(self, operation: str, param_name: str):
        message = (
            f"{operation} expects matrix {param_name!r} to be "
            f"non-singular (i.e. det != 0), got a singular matrix"
        )
        super().__init__(message)