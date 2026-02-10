class ShapeMismatchError(ValueError):
    """
    Raised when matrix shapes do not match for element-wise operations.
    """
    def __init__(
        self,
        operation: str,
        param_names: tuple[str, str],
        shape_a: tuple[int, int],
        shape_b: tuple[int, int]
    ):
        param_name_a, param_name_b = param_names
        message = (
            f"{operation} requires matrices {param_name_a!r} and "
            f"{param_name_b!r} to have the same shape, "
            f"got {param_name_a}={shape_a} vs {param_name_b}={shape_b}"
        )
        super().__init__(message)
        
        
class MatrixCompatibilityError(ValueError):
    """
    Raised when matrices cannot be multiplied via matmul due to shape
    mismatch.
    """
    def __init__(
        self,
        param_names: tuple[str, str],
        shape_a: tuple[int, int],
        shape_b: tuple[int, int]
    ):
        param_name_a, param_name_b = param_names
        message = (
            f"Matrix multiplication requires columns of {param_name_a!r} "
            f"to equal rows of {param_name_b!r}, got {param_name_a}={shape_a} "
            f"vs {param_name_b}={shape_b}"
        )
        super().__init__(message)
        

class NonSquareMatrixError(ValueError):
    """
    Raised when an operation requires a square matrix, but a non-square
    matrix is provided.
    """
    def __init__(self, param_name: str, shape: tuple[int, int]):
        message = (
            f"Expected {param_name!r} to be square, got shape "
            f"{param_name}={shape}"
        )
        super().__init__(message)


class LengthError(ValueError):
    """
    Raised when a sequence length is invalid.
    """
    def __init__(self, param_name: str, length: int, expected: int):
        message = (
            f"Expected {param_name!r} to have {expected} elements, "
            f"got {length}"
        )
        super().__init__(message)

    
class LengthMismatchError(ValueError):
    """
    Raised when multiple sequences do not have matching lengths.
    """
    def __init__(self, param_names: tuple[str, ...], lengths: tuple[int, ...]):
        if len(param_names) != len(lengths):
            raise ValueError(
                "'param_names' and 'lengths' must have the same number of "
                "elements"
            )

        # Build the length info e.g. ['x(n=5)', 'y(n=7)', 'z(n=5)']
        param_lengths = [
            f"{name}(n={length})" for name, length in zip(param_names, lengths)
        ]
        message = (
            f"Expected {', '.join(param_names)} to have the same number of "
            f"elements; got {', '.join(param_lengths)}"
        )
        super().__init__(message)