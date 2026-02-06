from numpy import sign
from stemcore import is_symexpr
from sympy import Abs, SympifyError, latex, sympify

from stemfard.core.rounding import fround


def pm_sign(value) -> str:
    """
    Return '+' if value is non-negative, '-' if negative.
    Works for numeric, string, and SymPy expressions.
    """
    try:
        if sign(float(value)) == -1:
            return "-"
        else:
            return "+" # value >= 0
    except (TypeError, ValueError):
        try:
            expr = sympify(value)
            if is_symexpr(expr):
                return "-" if str(expr).startswith('-') else "+"
            return "-" if expr.is_negative else "+"
        except (TypeError, ValueError, AttributeError, SympifyError):
            return '-' if str(value).strip().startswith('-') else '+'
        
        
def abs_value(
    value: int | float, tolatex: bool = True, decimals: int = -1    
) -> str:
    """
    Returns the absolute value of an object.

    Parameters
    ----------
    value : {int, float}
        The numerical value for which the absolute value needs to be 
        determined.

    Returns
    -------
    value_cleaned : str
        A string representing the absolute value of the input value in 
        LaTeX format.

    Raises
    ------
    Exception
        If an error occurs during the calculation or conversion to 
        LaTeX.

    Examples
    --------
    >>> stm.abs_value(5)
    '5'
    >>> stm.abs_value(-3.14)
    '3.14'
    >>> stm.abs_value(0)
    '0'
    """
    try:
        value = sympify(value)
        value = fround(x=value, decimals=decimals)
        if is_symexpr(value):
            # lstrip() - remove leading `-` just incase it is there
            result = latex(value) if tolatex else str(value)
            result = result.lstrip("-").strip(" ")
        else:
            result = latex(Abs(value)) if tolatex else Abs(value)
    except (ValueError, TypeError, AttributeError) as e:
        raise type(e)(str(e)) from e
    
    return result
        
