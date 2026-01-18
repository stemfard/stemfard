from typing import Any

from numpy import prod
from sympy import SympifyError


ERR_SYMPIFY_ERRORS = (
    SyntaxError,
    NameError,
    TypeError,
    ValueError,
    AttributeError,
    SympifyError
)


class ParameterError(Exception):
    """
    Base class for parameter-related errors.
    """
    pass


class ParamTypeError(TypeError, ParameterError):
    """
    Raised when a parameter has an invalid type.
    """
    def __init__(self, *, value: Any, expected: str, param_name: str) -> None:
        message = (
            f"Expected {param_name!r} to be {expected}, "
            f"got {type(value).__name__}"
        )
        super().__init__(message)
        
        
class ParamValueError(ValueError, ParameterError):
    """
    Raised when a parameter has an invalid value.
    """
    def __init__(self, *, value: Any, expected: str, param_name: str) -> None:
        message = (
            f"Expected {param_name!r} to be {expected}, got {value}"
        )
        super().__init__(message)


class ParamLengthError(ValueError, ParameterError):
    """
    Raised when a parameter has an invalid length.
    """
    def __init__(self, *, value: Any, expected: str, param_name: str) -> None:
        n = prod(value.shape) if hasattr(value, "shape") else len(value)
        message = (
            f"Expected {param_name!r} to have {expected}, got {n}"
        )
        super().__init__(message)
        
        
class ParamShapeError(ValueError, ParameterError):
    """
    Raised when a parameter has an invalid shape.
    """
    def __init__(self, *, value: Any, expected: str, param_name: str) -> None:
        shape = value.shape if hasattr(value, "shape") else len(value)
        message = (
            f"Expected {param_name!r} to have {expected}, got {shape}"
        )
        super().__init__(message)
        
        
class ParamAttributeError(ValueError, AttributeError):
    """
    Raised when a parameter has an invalid number of dimensions.
    """
    def __init__(
        self,
        *,
        obj: str,
        param_name: str,
        msg: str | None = None
    ) -> None:
        message = f"{obj!r} object has no attribute {param_name!r}"
        if msg:
            message += f": {msg}"
        super().__init__(message)