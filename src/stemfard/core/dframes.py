from numpy.typing import NDArray
from pandas import DataFrame, concat
from typing import Sequence, Any

from stemfard.core.errors import ParamTypeError, ParamValueError


def df_add_rows(
    df: DataFrame,
    rows: Sequence[Sequence[Any]] | NDArray,
    row_names: Sequence[str] | None = None,
    *,
    ignore_index: bool = False,
    param_name: str = "data"
) -> DataFrame:
    """
    Append one or more rows to a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The original DataFrame to which new rows will be added.
    rows : sequence of sequences or numpy.typing.NDArray
        Row data to append. Each row must contain the same number of
        elements as `df.columns`.
    row_names : sequence of str or None, optional
        Index labels for the new rows. If provided, its length must
        match the number of rows in `rows`.
    ignore_index : bool, default False
        If True, the resulting DataFrame will have a new integer index.
        If False, the index labels in `row_names` are preserved.
    param_name : str,
        Parameter name to be used in error messages.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the original data and the appended
        rows.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If the shape of `rows` does not match `df.columns`, or if
        `row_names` has an invalid length.

    Notes
    -----
    This function never mutates the input DataFrame `df`.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    >>> rows = [[5, 6], [7, 8]]
    >>> df_add_rows(df, rows, row_names=["r3", "r4"])
        A  B
    r3  5  6
    r4  7  8
    """
    if not isinstance(df, DataFrame):
        raise ParamTypeError(
            value=df,
            expected="a DataFrame",
            param_name=param_name
        )

    # --- Handle empty input safely ---
    if rows is None or len(rows) == 0:
        return df.copy()

    if row_names is not None and len(row_names) != len(rows):
        raise ParamValueError(
            value=row_names,
            expected=len(rows),
            param_name=param_name
        )

    new_rows = DataFrame(
        data=rows,
        columns=df.columns,
        index=row_names
    )

    return concat([df, new_rows], ignore_index=ignore_index)