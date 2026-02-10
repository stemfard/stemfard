import math
from typing import Any
from decimal import Decimal, ROUND_HALF_UP

from pandas import DataFrame, Series, set_option
from sympy import (
    FiniteSet, Matrix, Expr, Number, NumberSymbol, Rational, Integer, Float,
    sympify, SympifyError
)
from numpy import around, float64, round, asarray
from numpy.typing import NDArray

from stemlab.core.symbolic import is_symexpr
from stemlab.core.arraylike import is_iterable


def float_dict(result: dict, decimals: int = -1) -> dict:
    """
    Convert numeric elements of a dictionary to floats, with 
    optional rounding.

    Parameters
    ----------
    result : dict
        The dictionary containing values to be converted to floats.
    decimals : int, optional (default=-1)
        Number of decimal points for rounding. If -1 (default), 
        no rounding will be performed.

    Returns
    -------
    result : dict
        A dictionary with numeric elements rounded to the specified 
        number of decimals.

    Examples
    --------
    >>> stm.float_dict({'a': 1.234, 'b': 2.345}, decimals=2)
    {'a': 1.23, 'b': 2.34}
    >>> stm.float_dict({'a': 'text', 'b': 2 * 3.14}, decimals=2)
    {'a': 'text', 'b': 6.28}
    >>> stm.float_dict({'value1': 'text', 'value2': 11/13}, decimals=4)
    {'value1': 'text', 'value2': 0.8462}
    """
    def convert_value(value, decimals):
        try:
            # Try to convert to float and round if needed
            if isinstance(value, (int, float)):
                return round(
                    float(value), decimals
                ) if decimals != -1 else float(value)
            # Check if the value can be evaluated as a numeric expression
            elif isinstance(value, Expr):
                return value.evalf(
                    decimals
                ) if decimals != -1 else value.evalf()
            return value
        except (ValueError, TypeError):
            return value

    return {
        key: convert_value(value, decimals)
        for key, value in result.items()
    }


def round_half_up(x: float, decimals: int = -1):
    """
    Rounds a floating-point number to a specified number of decimal places 
    using the ROUND_HALF_UP method from the Decimal module.

    Parameters
    ----------
    x : float
        The number to be rounded.
    decimals : int, optional (default=-1)
        The number of decimal places to round to.

    Returns
    -------
    number_float : float
        The rounded number as a float.

    Examples
    --------
    >>> import stemlab as stm
    >>> stm.round_half_up(0.045, decimals=2)
    0.05

    >>> stm.round_half_up(1.234567, decimals=3)
    1.235

    >>> stm.round_half_up(5.678, decimals=0)
    6.0

    >>> stm.round_half_up(3.14159, decimals=-1)
    3.14159
    """
    if decimals == -1:
        return x
    number = Decimal(str(x))
    zeros_str = '0' * (decimals - 1)
    number_float = number.quantize(
        Decimal(f'0.{zeros_str}1'), rounding=ROUND_HALF_UP
    )
    return float(number_float)


def dframe_round(dframe, decimals):
    """
    Applies different rounding to columns in a DataFrame based 
    on their data types.
    
    Parameters
    ----------
    dframe: pandas.DataFrame 
        The DataFrame to with values to be rounded off.
    decimals: int
        Number of decimal points to use for truncation and rounding.
        
    Notes
    -----
    - For columns containing strings, rounds numeric values in the 
    string to the specified decimal points.
    - For numeric columns, rounds the values to the specified decimal points.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with values rounded off.
    """
    df = dframe.copy() # this is important to avoid overwriting original dframe
    def round_number(x, decimals):
        """
        Truncate objects which might contain both numbers and strings.
        """
        try:
            # Convert to float, truncate, and convert back to string with 
            # fixed decimal places
            rounded = round(float(x) * 10 ** decimals) / 10 ** decimals
            return f'{rounded:.{decimals}f}'
        except (ValueError, TypeError):
            return x

    # Identify string and numeric columns
    string_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Apply truncation to string columns
    df[string_columns] = df[string_columns].map(
        lambda x: round_number(x, decimals=decimals)
    )

    # Round numeric columns
    df[numeric_columns] = df[numeric_columns].round(decimals)

    return df


def round_to_nearest_half_up(
    x: float,
    to_nearest_5: bool = False,
    is_delta=True
) -> float:
    if to_nearest_5:
        x = math.floor(x * 2 + 0.5) / 2    
    else:
        x = around(x, decimals=1)
        if is_delta and x % 1 != 0 and int(round(x * 10)) % 2 != 0:
            x = around(x + 0.1, decimals=1)
        
    return x

   
def fround(
    x: int | float | list | Expr,
    decimals: int = -1,
    to_Matrix=True
) -> int | float | list | Expr | NDArray | Matrix:
    """
    Round the elements of an array or a sympy expression to the 
    specified number of decimal points.

    Parameters
    ----------
    x : {float, array_like, sym.Expr}
        The array or expression to round.
    decimals : int, optional (default=-1)
        Number of decimal points (or significant figures) to round the 
        results to. Default -1 means no rounding.
    to_Matrix : bool, optional (default = False)
        If `True`, result will be converted to a sympy matrix.

    Returns
    -------
    result : {int, float, list, Expr}
        The value of `x` rounded to `decimals`.
        
    Notes
    -----
    The result will be converted to a sympy object where applicable. 
    That is, sympy.Matrix, sympy.Expr, sympy.Int, sympy.Float, etc

    Examples
    --------
    >>> import sympy as sym
    >>> import pandas as pd
    >>> import stemlab as stm
    >>> stm.fround(x=[1.234, 2.345, 3.456], decimals=2)
    Matrix([[1.23000000000000], [2.34000000000000], [3.46000000000000]])
    >>> x, y = sym.symbols('x y')
    >>> matrix_result = sym.Matrix([[1.234, x], [y, 3.456]])
    >>> stm.fround(x=matrix_result, decimals=3) # sig. figures
    Matrix([[1.23, x], [y, 3.46]])
    >>> df_result = pd.DataFrame({'A': [1.234, 2.345], 'B': [3.456, 4.567]})
    >>> stm.fround(x=df_result, decimals=2)
          A     B
    0  1.23  3.46
    1  2.35  4.57
    """

    if decimals == -1 or not isinstance(decimals, int):
        return x
    
    try:
        return round(float(x), decimals)
    except (TypeError, ValueError):
        pass
    
    def get_numeric(x: Any) -> float:
        """
        Returns float if value is an array or list of a single element.
        
        Parameters
        ----------
        x : Any
            The result to be rounded off.
        
        Returns
        -------
        x : float
            The numeric value as a float if the result has only one 
            element.
        """
        try:
            x = Matrix(x)
            if hasattr(x, 'shape') and (x.shape == (1, 1) or len(x) == 1):
                x = x[0]
        except:
            pass
        
        return x
    
    if isinstance(x, str):
        try:
            x = sympify(x)
        except (TypeError, ValueError, AttributeError, SympifyError):
            pass
    
    set_option('display.precision', decimals)
    
    if isinstance(x, (DataFrame, Series)):
        return dframe_round(dframe=x, decimals=decimals)

    if is_symexpr(x):
        try:
            x = x.evalf(decimals)
        except AttributeError:
            pass
    
    if isinstance(x, (set, FiniteSet)):
        x = list(x)
    
    if is_iterable(x) and not isinstance(x, dict):
        try:
            x = Matrix(x)
            try:
                N = asarray(x.evalf(decimals + 10), dtype=float64)
                x = asarray(Matrix(round(N, decimals)), dtype=float64)
            except Exception: # symbolic matrix
                try:
                    x = x.evalf(decimals)
                except Exception:
                    pass
        except Exception:
            return x

    if isinstance(x, dict) or (isinstance(x, str) and ":" in x):
        try:
            x = float_dict(x, decimals=decimals)
        except Exception:
            pass
    
    numeric_vals = (int, float, Integer, Float, Rational, Number, NumberSymbol)
    if isinstance(x, numeric_vals):
        try:
            x = round_half_up(float(x), decimals)
        except Exception:
            pass

    if to_Matrix:
        try:
            x = Matrix(x)
        except (TypeError, ValueError, SympifyError):
            pass

    x = get_numeric(x) # return a scalar if the array has a single element
    
    return x


def round_dp(
    x: int | float | list | Expr, force:bool = True, decimals: int = -1
) -> int | float | list | Expr | NDArray | Matrix:
    
    if decimals != -1:
        x = fround(x=x, decimals=decimals, to_Matrix=False)
        
    if not hasattr(x, "tolist") and not is_symexpr(x):
        try:
            if isinstance(x, Integer):
                x = int(x)
            elif isinstance(x, (Float, Rational)):
                x = float(x)
        except:
            pass
    
    return x