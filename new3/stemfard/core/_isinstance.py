from typing import Any

from sympy import Expr, Symbol, flatten, sympify

from stemfard.core.errors.errors import SYMPIFY_ERRORS


def get_library_origin(obj: Any) -> str | None:
    """
    Detects the originating library of an object by inspecting its 
    type's module path.
    
    Returns:
        'numpy', 'sympy', 'pandas' if matched; otherwise None.
    """
    module = type(obj).__module__
    for lib in ("numpy", "sympy", "pandas"):
        if module.startswith(lib):
            return lib
    return None


def is_instance_from_library(obj: Any, library: str) -> bool:
    """
    Checks whether the object or any of its ancestor types originate 
    from a given library.

    Args:
        obj: The object to check.
        library: The root module name (e.g., 'numpy', 'sympy', 'pandas').

    Returns:
        True if object's class or ancestors are from the given library.
    """
    return any(type_.__module__.startswith(library) for type_ in type(obj).__mro__)


def is_numpy_instance(obj: Any) -> bool:
    """Checks if the object is a NumPy type."""
    return is_instance_from_library(obj=obj, library="numpy")


def is_sympy_instance(obj: Any) -> bool:
    """Checks if the object is a SymPy type."""
    return is_instance_from_library(obj=obj, library="sympy")


def is_pandas_instance(obj: Any) -> bool:
    """Checks if the object is a pandas DataFrame or Series."""
    return is_instance_from_library(obj=obj, library="pandas")


def is_system_of_expressions(expr_list: list[str | Expr]) -> bool:
    
    try:
        if not isinstance(expr_list, list):
            expr_list = list(expr_list)
        expr_list = flatten(expr_list)
    except TypeError:
        return False

    for expr_str in expr_list:
        try:
            expr = sympify(expr_str)
        except SYMPIFY_ERRORS:
            return False  # failed to parse

        # All free symbols must be sympy.Symbols and there must be at least one
        try:
            syms = expr.free_symbols
            if len(syms) == 0 or not all(isinstance(s, Symbol) for s in syms):
                return False
        except AttributeError:
            return False

    return True