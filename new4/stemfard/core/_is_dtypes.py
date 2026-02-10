from typing import Any

from numpy import array, issubdtype, number, prod
from sympy import Expr, Matrix, SympifyError, flatten, sympify
from sympy.core.relational import Relational

from verifyparams.core.errors import SYMPIFY_ERRORS


def is_maths_function(obj: Any) -> bool:
    """
    Check if object is already a usable mathematical function.
    """
    if not callable(obj):
        return False
    
    # Reject classes (they're constructors, not functions)
    # numpy ufuncs like np.sin are instances, not the ufunc class
    if isinstance(obj, type):
        return False
    
    return True


def is_symexpr(obj: Any) -> bool:
    """
    Check if an object is a symbolic expression (contains variables).
    
    A symbolic expression is defined as having at least one free symbol
    (variable) that is not a defined constant.
    
    Parameters
    ----------
    obj : Any
        Object to check. Can be:
        - sympy.Expr object
        - String representation of symbolic expression
        - Any other object (will return `False` if not symbolic)
    
    Returns
    -------
    bool
        `True` if the object represents a symbolic expression with 
        variables, `False` otherwise.
    
    Examples
    --------
    >>> import sympy as sym
    >>> x, y = sym.symbols('x y')
    
    >>> is_symexpr(sym.pi/4)      # Constant expression
    False
    
    >>> is_symexpr(sym.pi/x)      # Contains variable x
    True
    
    >>> is_symexpr(x**2 + x*y - 5)  # Multiple variables
    True
    
    >>> is_symexpr('x**2 + y')   # String input
    True
    
    >>> is_symexpr('3.14')       # Numeric string
    False
    
    >>> is_symexpr(42)           # Plain number
    False
    
    >>> is_symexpr([x, y])       # List (not expression)
    False
    """
    if isinstance(obj, Expr):
        return len(obj.free_symbols) > 0
    
    if isinstance(obj, str):
        try:
            expr = sympify(obj)
            if isinstance(expr, Expr):
                return len(expr.free_symbols) > 0
            else:
                return False
        except SYMPIFY_ERRORS:
            return False

    return False


def is_any_element_negative(M: list[int | float]) -> bool:
    """
    Check if any of the elements in the array are negative, including 
    symbolic expressions or string.

    Parameters
    ----------
    array_ : array_like
        The array-like object to be checked.

    Returns
    -------
    bool
        Returns True if any element in the array is negative, 
        otherwise returns False.

    Raises
    ------
    Exception
        If the input array cannot be converted to a numpy array.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.is_any_element_negative([1, 2, 3])
    False
    >>> stm.is_any_element_negative([-1, 2, 3])
    True
    >>> stm.is_any_element_negative([['b', 3, 8], [3, 1, 9]])
    False
    >>> stm.is_any_element_negative([[4, 'a', 9], ['-g', 8, 5]])
    True
    """
    try:
        M = array(M)
    except (TypeError, ValueError) as e:
        raise ValueError(str(e)) from e
    if issubdtype(M.dtype, number): # all values are numeric
        return any(M < 0)
    else: # contains symbols
        return any([is_negative(value) for value in flatten(M)])


def is_negative(value: int | float | str | Expr) -> bool:
    return str(value).strip().startswith("-")
    

def is_negative_detailed(value: int | float | str | Expr) -> bool:
    """
    Safely checks whether a value is negative.
    Works for numeric, string, symbolic expressions, and relational objects.
    Never raises an exception.
    """
    try:
        expr = sympify(value)
    except SympifyError:
        return str(value).strip().startswith("-")

    try:
        # Known numeric or symbolic negativity
        if hasattr(expr, "is_negative") and expr.is_negative is not None:
            return bool(expr.is_negative)

        # Relational expressions
        if isinstance(expr, Relational):
            try:
                lhs_val = float(expr.lhs)
                rhs_val = float(expr.rhs)
                return (lhs_val - rhs_val) < 0
            except Exception:
                return False

        # Numeric fallback
        try:
            return float(expr) < 0
        except Exception:
            return str(expr).strip().startswith("-")
    except Exception:
        return False


def is_numeric(string: str) -> bool:
    """
    Check if a string represents a purely numeric value
    (i.e., no free symbols or symbolic expressions).

    Parameters
    ----------
    string : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string represents a numeric value 
        (int or float), False otherwise.
    """
    from stemlab.core.symbolic import sym_sympify
    try:
        val = sym_sympify(string, is_expr=True)
        return val.is_number and not val.free_symbols
    except (SympifyError, ValueError, ArithmeticError, TypeError):
        return False
    
    
def contains_symbols(obj: Any) -> bool:
    """
    Check if a scalar or matrix contains any symbolic variables.
    Works with SymPy objects and plain Python/numpy data.
    """
    try:
        return bool(Matrix(obj).free_symbols)
    except (ValueError, TypeError, AttributeError, SympifyError):
        return bool(getattr(obj, 'free_symbols', set()))
    
    
def is_all_symbolic(obj: Any) -> bool:
    try:
        Matrix(obj)
    except (ValueError, TypeError, AttributeError):
        return False
    is_symbol_list = [contains_symbols(b_ij) for b_ij in flatten(obj)]
    return sum(is_symbol_list) == prod(obj.shape)