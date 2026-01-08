from typing import Any, Literal
from collections.abc import Sequence

from numpy import array, asarray, ndarray
from pandas import DataFrame, Series


def to_numeric(
    data: Sequence[Any],
    kind: Literal["array", "series", "dataframe"] = "array",
    dtype: Any = float,
    param_name: str = "data"
) -> ndarray | Series | DataFrame:
    """
    Conversion of input to numeric ndarray, Pandas Series, or DataFrame.

    Parameters
    ----------
    data : sequence
        Input data. Can contain numbers or numeric strings.
        1D for 'array' or 'series', 2D for 'dataframe'.
        Generators are supported for 'array' or 'series'.
    kind : {'array', 'series', 'dataframe'}, default 'array'
        Type of object to return.
    dtype : type, default float
        Desired numeric type.
    param_name : str, default='data'
        Name to be used in error messages.

    Returns
    -------
    ndarray, pd.Series, or pd.DataFrame
        Numeric representation of input.

    Raises
    ------
    TypeError
        If `data` is a string/bytes instead of a sequence.
    ValueError
        If any element cannot be converted to numeric.
    """
    if isinstance(data, (str, bytes)):
        raise TypeError(f"Expected a sequence, got {type(data).__name__}")

    # ----------- 1D Series -----------
    if kind == "series":
        try:
            return Series(data, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert {param_name!r} to Series with dtype "
                f"{dtype}"    
            ) from e

    # ----------- 2D DataFrame -----------
    elif kind == "dataframe":
        try:
            return DataFrame(data, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert {param_name!r} to DataFrame with dtype "
                f"{dtype}"
            ) from e

    # ----------- ndarray (1D or 2D) -----------
    elif kind == "array":
        try:
            return asarray(data, dtype=dtype)
        except (ValueError, TypeError) as e:
            try:
                return array(
                    a=[asarray(row, dtype=dtype) for row in data],
                    dtype=dtype
                )
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot convert {param_name!r} to a 1D array with dtype "
                    f"{dtype}"  
                ) from e