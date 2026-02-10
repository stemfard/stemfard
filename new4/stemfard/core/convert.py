from typing import Any
from numpy import array, around, float64
from pandas import DataFrame, Series, isna, to_numeric
from pandas.api.types import is_numeric_dtype
from stemcore import is_symexpr
from sympy import (
    Float, Integer, Matrix, Rational, Symbol, flatten, sympify, Expr
)

from stemfard.core._strings import str_remove_tzeros
     

def df_convert_to_float(df: DataFrame) -> DataFrame:
    """
    Convert numeric, numeric-looking, or numeric-with-NaN columns to float.
    If a column contains non-numeric values (like letters), leave it unchanged.
    """ 
    df = df.copy()
    for col in df.columns:
        try:
            # First attempt: force strict conversion
            df[col] = to_numeric(df[col], errors="raise").astype(float)
        except (ValueError, TypeError, AttributeError):
            try:
                # Second attempt: allow NaN coercion (for None/NaN/missing values)
                converted = to_numeric(df[col], errors="coerce")
                # If all non-numeric entries became NaN → column is safe to cast
                if converted.notna().any() and converted.isna().sum() <= df[col].isna().sum():
                    df[col] = converted.astype(float)
            except Exception:
                pass
    return df


def to_nested_list(data: list[Any]) -> list[list[Any]]:
    if data and not isinstance(data[0], list):
        data = array([data]).T.tolist()
    return data


def to_dataframe(
    data: Any,
    row_names: list[str] | None = None,
    col_names: list[str] | None = None
) -> DataFrame:
    """
    Safely convert a list-of-lists into a pandas DataFrame with
    optional row/column names. Performs validation before conversion.
    """
    
    # its a list/dictionary of dictionary so create dataframe directly
    if ":" in str(data) and "{" in str(data):
        df = DataFrame(data)
        return df
    
    try:
        data = to_nested_list(data)
    except Exception:
        pass
    
    if hasattr(data, "tolist"):
        data = data.tolist()
    
    if not isinstance(data, (list, tuple)):
        raise TypeError(
            "DataFrame values must be a list or tuple with at least 2 elements"
        )

    if len(data) == 0:
        raise ValueError("Cannot create DataFrame from empty values")

    # Validate that all rows are of equal length
    try:
        row_lengths = {len(r) for r in data}
    except (TypeError):
        data = [data]
        row_lengths = {len(r) for r in data}
    if len(row_lengths) > 1:
        raise ValueError("Jagged rows not allowed in DataFrame input")
    
    error_types = (
        ValueError, TypeError, AttributeError, MemoryError, OverflowError,
        NotImplementedError
    )
    try:
        df = DataFrame(data=data, index=row_names, columns=col_names)
        try:
            return df_convert_to_float(df)
        except:
            return df
    except error_types as e:
        raise type(e)(f"Unable to create DataFrame: {e}") from e


def df_round_safe(df: DataFrame, decimals: int = 14) -> DataFrame:
    decimals = 14 if decimals == -1 else decimals
    for col in df.columns:
        try:
            converted = to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted.round(decimals).astype(float)
        except Exception:
            pass
    return df


def dframe_to_latex(
    df: DataFrame,
    include_index=True,
    environment: str = "tabular",  # "array" or "tabular"
    header_color: str = "gray",
    first_col_align: str = "l",  # Default without | for tabular
    outer_border: bool = False,
    inner_hlines: bool = False,
    inner_vlines: bool = False,
    indent: str = "\t",  # Default to tabs, can be "    " for spaces
    remove_math: bool = False,
    is_round_safe: str = False
) -> str:
    """
    Convert a pandas DataFrame to LaTeX array or tabular
    environment.
    
    This function generates LaTeX code for either MathJax-compatible
    array environments or standard LaTeX tabular environments with
    various formatting options including borders, colored headers,
    and customizable alignment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert to LaTeX format. Must not be empty.
    environment : {'tabular', 'array'}, default 'tabular'
        LaTeX environment to use:
        - 'array': MathJax-compatible array environment 
          (uses \\textrm or \\mathrm for text)
        - 'tabular': Standard LaTeX tabular environment
    header_color : str, default='gray'
        Color for the header row. Requires \\usepackage[table]{xcolor}
        in LaTeX preamble for tabular environment. Use None for no 
        header coloring.
    first_col_align : str, default='l'
        Alignment for the first column (typically index). Common values:
        - 'l': left-aligned
        - 'r': right-aligned  
        - 'c': center-aligned
        - For arrays with inner_vlines: 'l|', 'r|', etc.
    outer_border : bool, default=False
        If True, adds vertical lines on both sides of the table.
    inner_hlines : bool, default=False
        If True, adds horizontal lines between all rows.
    inner_vlines : bool, default=False
        If True, adds vertical lines between all columns.
    indent : str, default '\\t'
        Indentation string for LaTeX code formatting. Common values:
        - '\\t': tab indentation
        - '    ': 4-space indentation
        - '  ': 2-space indentation
    
    Returns
    -------
    str
        LaTeX code for the specified environment containing the
        DataFrame data.
    
    Raises
    ------
    ValueError
        If DataFrame is empty or environment is not 'array' or 
        'tabular'.
    
    Notes
    -----
    - For MathJax rendering (web-based), use environment='array'
    - For standard LaTeX documents, use environment='tabular' 
    - The function automatically determines whether to include the 
      index based on its data type and name presence
    - Numeric columns are right-aligned, text columns are left-aligned
    - Special LaTeX characters are automatically escaped
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['X', 'Y'])
    
    Basic tabular environment:
    >>> print(dframe_to_latex(df))
    \\begin{tabular}{lrr}
    \\hline
    \\rowcolor{gray}  & A & B \\\\
    \\hline
    X & 1 & 3 \\\\
    Y & 2 & 4 \\\\
    \\hline
    \\end{tabular}
    
    MathJax array with borders:
    >>> print(dframe_to_latex(df, environment='array', outer_border=True, inner_vlines=True))
    \\begin{array}{|l|r|r|}
        \\hline
        \\rowcolor{gray} \\textrm{} & \\textrm{A} & \\textrm{B} \\\\
        \\hline
        \\textrm{X} & 1 & 3 \\\\
        \\textrm{Y} & 2 & 4 \\\\
        \\hline
    \\end{array}
    
    Custom indentation and no header color:
    >>> print(dframe_to_latex(df, header_color=None, indent='    '))
    \\begin{tabular}{lrr}
        \\hline
        & A & B \\\\
        \\hline
        X & 1 & 3 \\\\
        Y & 2 & 4 \\\\
        \\hline
    \\end{tabular}
    """
    if not isinstance(df, DataFrame):
        df = DataFrame(df)
        
    if df.empty:
        return f"\\textbf{{DataFrame is empty}}"

    if is_round_safe:
        df = df_round_safe(df=df)
    
    # Validate environment
    if environment not in ("array", "tabular"):
        raise ValueError("environment must be 'array' or 'tabular'")
    
    # Handle first column alignment based on environment
    if inner_vlines and len(first_col_align) > 1:
        first_col_align = first_col_align[0]  # avoid say r|| for array
    
    # Build column alignment
    col_format = []
    if include_index:
        col_format.append(first_col_align)
    for col in df.columns:
        col_format.append("r" if is_numeric_dtype(df[col]) else "l")
    
    # Add inner vertical lines
    if inner_vlines:
        col_format = "|".join(col_format)
    else:
        col_format = "".join(col_format)
    
    # Add outer borders
    if outer_border:
        col_format = "|" + col_format + "|"
    
    # Prepare data including index
    if include_index:
        data = [[df.index.name or ""] + list(df.columns)]
        data += [[idx] + list(row) for idx, row in zip(df.index, df.values)]
    else:
        data = [list(df.columns)] + df.values.tolist()
        
    # Helper to format each cell based on environment
    def to_latex_value(v):
        if isna(v):
            return ""
        
        try:
            float(sympify(v))
            return str(v)
        except:
            # Escape LaTeX special characters
            v = str(v)
            # special_chars = {
            #     '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            #     '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', 
            #     '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
            # }
            # for char, escape in special_chars.items():
            #     v = v.replace(char, escape)
            
            # Array uses \textrm{}, tabular doesn't need it
            if environment == "array" and remove_math:
                v = v.replace(' ', '\\:')
                return f"\\mathrm{{{v}}}"
            else:
                return v
    
    # Build LaTeX table
    latex_str = f"\\begin{{{environment}}}{{{col_format}}}\n"
    latex_str += f"{indent}\\hline\n"

    for i, row in enumerate(data):
        line_continuation = " \\\\" if environment == "array" else r" \\"
        row_str = " & ".join(to_latex_value(v) for v in row) + line_continuation
        
        # Apply header color
        if i == 0 and header_color:
            if environment == "tabular":
                row_color = f"\\rowcolor{{{header_color}}}"
            else:
                row_color = ""
            row_str = f"{indent}{row_color} {row_str}"
            pass
        else:
            row_str = f"{indent}{row_str}"
        
        latex_str += row_str + "\n"

        # Add horizontal lines
        if i == 0:
            latex_str += f"{indent}\\hline\n"
        elif inner_hlines and i < len(data) - 1:
            latex_str += f"{indent}\\hline\n"
    
    # Final horizontal line
    latex_str += f"{indent}\\hline\n"
    latex_str += f"\\end{{{environment}}}"
    
    return str_remove_tzeros(latex_str)


def dframe_to_array(
    df: DataFrame,
    include_index: bool = True,
    header_color: str = "gray",
    first_col_align: str = "l|",
    outer_border: bool = False,
    inner_hlines: bool = False,
    inner_vlines: bool = False,
    is_round_safe: bool = False
) -> str:
    """Convert DataFrame to MathJax array environment."""
    return dframe_to_latex(
        df=df,
        include_index=include_index,
        environment="array",
        header_color=header_color,
        first_col_align=first_col_align,
        outer_border=outer_border,
        inner_hlines=inner_hlines,
        inner_vlines=inner_vlines,
        indent="\t",
        is_round_safe=is_round_safe
    )


def df_to_latex_tabular(
    df: DataFrame,
    include_index: bool = True,
    header_color: str = "gray",
    first_col_align: str = "l",
    outer_border: bool = False,
    inner_hlines: bool = False,
    inner_vlines: bool = False,
    is_round_safe: bool = False
) -> str:
    """Convert DataFrame to LaTeX tabular environment."""
    return dframe_to_latex(
        df=df,
        include_index=include_index,
        environment="tabular",
        header_color=header_color,
        first_col_align=first_col_align,
        outer_border=outer_border,
        inner_hlines=inner_hlines,
        inner_vlines=inner_vlines,
        indent="\t",
        is_round_safe=is_round_safe
    )
    

def result_int_float_or_str(k: Any):
    return k if isinstance(k, (int, float, bool)) else str(k)


def result_to_string(obj: Any, decimals: int = -1) -> str:
    """
    Convert any object to string safely:
    - Numeric arrays/lists are rounded if decimals >= 0.
    - SymPy matrices, symbolic, or relational objects are handled
        gracefully.
    - Everything is returned as a string.
    """
    try:
        # Try numeric conversion
        arr = array(obj)
        if decimals >= 0:
            arr = around(arr, decimals=decimals)
        return str_remove_tzeros(str(arr.tolist()))
    except Exception:
        # Fallback: attempt SymPy or generic conversion
        try:
            mat = Matrix(obj)
            if decimals >= 0:
                # Attempt numeric evaluation
                arr = array(
                    mat.evalf(decimals + 10),
                    dtype=float64
                )
                arr = around(arr, decimals=decimals)
                return str_remove_tzeros(str(arr.tolist()))
            return str_remove_tzeros(str(mat.tolist()))
        except Exception:
            # Last resort: fallback to string
            try:
                return str_remove_tzeros(str(obj.tolist()))
            except AttributeError:
                return str_remove_tzeros(str(obj))


def df_add_space(df: DataFrame) -> DataFrame:
    nrows, ncols = df.shape
    for row in range(nrows):
        for col in range(ncols):
            A_ij = str(df.iloc[row, col])
            if "/" in A_ij or A_ij.endswith("*I"):
                # add space to avoid Excel converting such values to date
                df.iloc[row, col] = f" {A_ij.replace('  ', ' ')}"
    return df


def result_to_csv(
    obj: Any,
    is_system: bool = False,
    index: bool = False,
    header: bool = False    
) -> str:
    
    types_ = (str, int, float, Integer, Float, Symbol, Rational)
    obj_str = str(obj)
    
    if isinstance(obj, types_) or is_symexpr(obj) or "=" in obj_str or ">" in obj_str or "<" in obj_str or "!=" in obj_str:
        obj = DataFrame(data=[str(obj)])
        result = obj.to_csv(header=header, index=index)
        return str_remove_tzeros(result)
    
    if isinstance(obj, (DataFrame, Series)):
        result = obj.to_csv(header=True, index=index)
    elif isinstance(obj, Expr):
        result = ", ".join(map(str, flatten(obj)))
    else:
        if hasattr(obj, "tolist"):
            try:
                obj = DataFrame(data=obj.tolist())
            except ValueError:
                obj = DataFrame(data=[obj.tolist()])
            result = obj.to_csv(header=header, index=index)
        else:
            try:
                obj = DataFrame(data=obj)
            except (ValueError, TypeError, AttributeError):
                try:
                    obj = DataFrame(data=[obj])
                except (ValueError, TypeError, AttributeError):
                    obj = DataFrame(data=["Export to CSV failed"])
            result = obj.to_csv(header=header, index=index)
             
    return str_remove_tzeros(result.replace(",", ", "))