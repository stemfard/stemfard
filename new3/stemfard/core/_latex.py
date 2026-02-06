from typing import Iterable, List, Literal

from numpy import (
    array, asarray, insert, matrix, ndarray, vstack, hstack, round
)
from pandas import DataFrame
from sympy import (
    MatrixBase, SympifyError, latex, Matrix, flatten, sympify
)

from stemlab.core.datatypes import (ListArrayLike, NumpyArray, is_function)
from stemlab.core.symbolic import is_symexpr
from stemfard.core._strings import str_data_join, str_remove_tzeros
from stemfard.core.convert import dframe_to_array
from stemfard.core._type_aliases import SequenceArrayLike


def tex_array_to_latex(
    M: SequenceArrayLike,
    align: Literal['left', 'center', 'right'] = "right",
    brackets: Literal['[', '(', '|', ''] = '['
) -> str:
    """
    Convert a list or 2D array to LaTeX array format.

    Parameters
    ----------
    M : ArrayLike
        Input array (1D or 2D) to be converted to LaTeX.
    align : {'left', 'center', 'right'}, optional (default='right')
        Alignment of columns in the LaTeX array.
    brackets : {'[', '(', '|', ''}, optional (default='[')
        Type of brackets surrounding the LaTeX array.

    Returns
    -------
    str
        LaTeX representation of the array.

    Raises
    ------
    ValueError
        If input is not a list or cannot be converted to a matrix.

    Examples
    --------
    >>> u = [4, 5, 6]
    >>> tex_array_to_latex(u)
    '\\left[\\begin{array}{rrr}4 & 5 & 6\\end{array}\\right]'

    >>> v = [[4, 5, 6], [1, 2, 3]]
    >>> tex_array_to_latex(v, align='center', brackets='(')
    '\\left(\\begin{array}{ccc}4 & 5 & 6\\\\1 & 2 & 3\\end{array}\\right)'

    >>> w = [[1, 2], [3, 4]]
    >>> tex_array_to_latex(w, brackets='|')
    '\\left|\\begin{array}{rr}1 & 2\\\\3 & 4\\end{array}\\right|'
    """
    try:
        M = matrix(M)
        ncols = M.shape[1]
        M = M.tolist()
        delimiter = " \\\\ "
        latex_matrix = f"\\begin{{array}}{{{align[0] * ncols}}} {delimiter.join([' & '.join(map(str, line)) for line in M])}\\end{{array}}"
    except Exception as e:
        raise e
    bracket_symbols = {
        "[": ("\\left[", "\\right]"),
        "(": ("\\left(", "\\right)"),
        "|": ("\\left|", "\\right| "),
        "": ("", ""),
    }
    left, right = bracket_symbols.get(brackets, ("", ""))
    latex_syntax = f"{left}{latex_matrix}{right}"

    return str_remove_tzeros(latex_syntax)


def tex_list_to_latex(
    lst: List,
    align: Literal["left", "right", "center"] = "right"
) -> str:
    """
    Converts a list-like object to LaTeX array format. Useful when values 
    cannot be converted to mathematical expressions using `sym.sympify()` 
    from SymPy.

    Parameters
    ----------
    lst : list or iterable
        Object containing values to be converted to LaTeX.
    align : {"left", "right", "center"}, optional (default="right")
        Alignment of the values in the array.

    Returns
    -------
    str
        LaTeX syntax for the array.

    Raises
    ------
    ValueError
        If input is empty or alignment is invalid.

    Examples
    --------
    >>> u = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
    >>> tex_list_to_latex(u)
    '\\begin{array}{rrrr}Quarter 1 & Quarter 2 & Quarter 3 & Quarter 4\\end{array}'

    >>> v = [1, 2, 3, 4]
    >>> tex_list_to_latex(v, align='center')
    '\\begin{array}{cccc}1 & 2 & 3 & 4\\end{array}'
    """
    if not lst:
        raise ValueError("Input list cannot be empty")
    
    if align not in ("left", "right", "center"):
        align = "right"

    # Flatten the list and ensure it's 1D
    flattened = []
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            flattened.extend(item)
        else:
            flattened.append(item)

    if not flattened:
        raise ValueError("No valid elements found after flattening")

    align_char = {'left': 'l', 'center': 'c', 'right': 'r'}[align]

    latex_syntax = (
        f'\\begin{{array}}{{{align_char * len(flattened)}}} '
        f'{" & ".join(map(str, flattened))}'
        f' \\end{{array}}'
    )

    return latex_syntax


def tex_matrix_to_latex_eqtns(
    A: NumpyArray | Matrix,
    b: NumpyArray | Matrix,
    displaystyle: bool = True,
    hspace: int = 0,
    vspace: int = 7,
    inline: bool = False
) -> str:
    """
    Convert matrix to linear equations.

    A : {list, tuple, Series, NDarray}
        Coefficients matrix.
    b : {list, tuple, Series, NDarray}
        Constants matrix.
    hspace : int, optional (default=0)
        Horizontal space before the equations.
    vspace : int, optional (default=7)
        Vertical space between rows.
    inline : bool, optional (default=True)
        If `True`, then use '$' (i.e. equations within text), 
        otherwise use '$$' (i.e. equations on new line).

    Returns
    -------
    Axb_latex : str
        A string containing the Latex syntax for an linear equations.

    Examples
    --------
    >>> A = [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1],
    ... [0, 3, -1, 8]]
    >>> b = [6, 25, -11, 15]
    >>> M = stm.tex_matrix_to_latex_eqtns(A, b)
    >>> print(M)
    $$
    \displaystyle
    \\begin{aligned} 
        10 x_{1} - x_{2} + 2 x_{3} &= 6 \\[7pt] 
        - x_{1} + 11 x_{2} - x_{3} + 3 x_{4} &= 25 \\[7pt] 
        2 x_{1} - x_{2} + 10 x_{3} - x_{4} &= -11 \\[7pt] 
        3 x_{2} - x_{3} + 8 x_{4} &= 15 
    \end{aligned} 
    $$
    """
    from stemlab.core.symbolic import sym_sympify
    
    A = Matrix(A)
    A_nrows = A.shape[0]
    b = Matrix(b)
    b_nrows, b_ncols = b.shape
    if b_ncols != 1:
        b = b.T
    if A_nrows != b_nrows:
        raise ValueError("Lengths do not match")
        
    Ax = []
    # create LHS Ax
    for row in range(A.shape[0]):
        terms_joined = " + ".join(
            [f"{value} * x{k + 1}" for k, value in enumerate(A[row, :])]
        )
        Ax.append(sym_sympify(terms_joined, is_expr=False))
    Ax = Matrix(Ax)
    # join LHS (Ax) to RHS (b) to form Ax = b
    Axb = []
    for row in range(A.shape[0]):
        Axb.append([f"{latex(Ax[row, 0])} &= {latex(b[row, 0])}"])

    displaystyle = "\\displaystyle" if displaystyle else ""
    dollar = "$" if inline else "$$"
    hspace = "" if hspace == 0 else "\\hspace{" + str(hspace) + "cm}"

    delimiter = f" \\\\[{str(vspace)}pt] \n\t"
    Axb = delimiter.join(flatten(Axb))
    Axb_latex = (
        dollar + "\n" + hspace
        + displaystyle + "\n\\begin{aligned} \n\t"
        + Axb + " \n\\end{aligned} \n" + dollar
    )

    return Axb_latex


def tex_table_to_latex(
    data: ListArrayLike,
    row_names: list | None = None,
    col_names: list | None = None,
    row_title: str = '',
    caption: str = 'Table',
    first_row_bottom_border: bool = False,
    last_row_top_border: bool = False,
    decimals: int = 8
):
    """
    Convert a table of statistics (as an array) to a Latex array
    syntax.

    Parameters
    ----------
    data : array_like
        An array_like object with the statistics.
    row_names : array_like
        Row names.
    col_names : array_like
        Column names.
    row_title : str, optional (default='')
        The title / heading of the rows.
    caption : str, optional (default='Table')
        Table caption.
    first_row_bottom_border : bool, optional (default=False)
        If `True`, a bottom border will be added on first row
    last_row_top_border : bool, optional (default=False)
        If `True`, a top border will be added on last row
    decimals : int, optional (default=8)
        Number of decimal points or significant figures for symbolic
        expressions.

    Returns
    -------
    html_latex : str
        String containing the html and Latex syntax.

    Examples
    --------
    import numpy as np
    import stemlab as stm

    >>> M = np.array(
        [[2318, 4276, 1664, 1279],
        [3431, 1246, 3558, 4503],
        [2282, 2299, 3956, 5467],
        [1855, 2805, 5212, 5704],
        [3093, 3576, 5137, 5024]]
    )
    >>> stm.tex_table_to_latex(
        data=M, col_names=['Qtr 1', 'Qtr 2', 'Qtr 3', 'Qtr 4']
    )
    ['<span style="color:blue;"><strong>Table</strong></span><br />$\\begin{array}{l|rrrr} \\hline 1 & \\text{Qtr 1} & \\text{Qtr 2} & \\text{Qtr 3} & \\text{Qtr 4} \\\\ 2 & 2318 & 4276 & 1664 & 1279 \\\\ 3 & 3431 & 1246 & 3558 & 4503 \\\\ 4 & 2282 & 2299 & 3956 & 5467 \\\\ 5 & 1855 & 2805 & 5212 & 5704 \\\\ 6 & 3093 & 3576 & 5137 & 5024 \\\\ \\hline \\end{array}$']
    """
    from stemlab.core.decimals import fround

    html_latex = []
    try:
        data = fround(data, decimals)
    except:
        pass
    
    if col_names is not None:
        col_names = [f'\\mathrm{{{col_name}}}' for col_name in col_names]
        last_column_name = col_names[-1].title()
        if ('Total' in last_column_name or 'Sum' in last_column_name
            or first_row_bottom_border is True):
            col_names[-1] = f'{col_names[-1]} \\hline'
        
        if col_names is not None:
            data = vstack([col_names, data])
            
    if isinstance(row_title, str):
        row_title = [row_title]
    row_title[0] = f'\\mathrm{{{row_title[0]}}}'
    
    if row_names is not None:
        row_names = array(
            [[f'\\mathrm{{{row_name}}}' for row_name in row_names]]
        ).T
        row_names = insert(arr=row_names, obj=0, values=row_title, axis=0)
        row_names = row_names.astype('<U250')
        last_row_name = row_names[-1][0].title()
        if ('Total' in last_row_name or 'Sum' in last_row_name
            or first_row_bottom_border is True):
            row_names[-1][0] = f' \\hline {row_names[-1][0]}'

        if row_names is not None:
            data = hstack([row_names, data])
    
    results_latex = f'${tex_array_to_latex(data, brackets="")}$'
    results_latex = results_latex\
        .replace('\\hline \\\\', '\\\\ \\hline', 1)\
        .replace('{r', '{l|')\
        .replace("r}", "r} \\hline")\
        .replace("hline} \\\\", "} \\\\ \\hline ")\
        .replace(f"\\end{{array}}", f" \\\\ \\hline \\end{{array}}")\
        .replace('nan', '')\
        .replace('\\hline &', '&')\
        .replace(' (', '~(')
    
    if caption:
        caption_number = f'\\color{{blue}}{{\\text{{Table No: }}}}'
        caption = f'{caption_number}\\text{{{caption}}}'
        caption = f"${caption}$<br />" if caption else ""
    else:
        caption = ""
    html_latex = f'{caption}{str_remove_tzeros(results_latex)}'

    return html_latex


def tex_to_latex_arr(
    arr, brackets: Literal["[]", "()", "||"] | None = "[]"
) -> str:
    
    try:
        arr = asarray(arr, dtype="<U2500")
    except (TypeError, ValueError, AttributeError) as e:
        raise ValueError(str(e)) from e
        
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        
    col_align = "r" * arr.shape[0]
    arr_joined = " \\\\ ".join([" & ".join(row) for row in arr])
    
    arr = f"\\begin{{array}}{{{col_align}}} {arr_joined} \\end{{array}}"
    
    # --- Wrap brackets ---
    if brackets:
        if len(brackets) != 2:
            raise ValueError(
                "Expected 'brackets' to be a pair of characters "
                f"e.g., '[]', '()', '{{}}' or '||'; got {brackets}"
            )
            
        valid = ("[]", "()", "||", None)
        if brackets not in valid:
            raise ValueError(
                f"Expected 'brackets' to be one of: {str_data_join(valid)}, "
                f"got {brackets}"
            )
            
        return f"\\left{brackets[0]}{arr}\\right{brackets[1]}"
    
    return arr


def tex_to_latex(
    M: SequenceArrayLike,
    align: Literal['left', 'right'] = 'right',
    brackets: Literal['[', '(', '|', ''] = '[',
    scalar_bracket: bool = False,
    raise_exception: bool = False,
    decimals:int = -1
):
    """
    ```
    Converts an array_like object to latex.

    Parameters
    ----------
    M : array_like
        array-like object containing the values of the array. The 
        values of the array must be converted to
    align : str, optional (default='right')
        Alignment of the values of the array.
    brackets : {'[', '(', '|', ''}, optional (default='[')
        Brackets to be used;  
        ==========================================================  
        brackets                       Description   
        ==========================================================  
        '[' .......................... Use square brackets []  
        '(' .......................... Use parenthesis brackets ()  
        '|' .......................... Use | e.g. for determinants  
        '' ........................... No brackets / parenthesis  
        ==========================================================  
    scalar_bracket : bool, optional (default=False)
        If `True`, scalars will be enclosed within `brackets`. 
    raise_exception : bool, optional (default=False)
        If `True`, an Exception will be raised if a value cannot be 
        converted, otherwise, the value will be returned as it was 
        entered.
    decimals : int, default=-1
        Number of decimal points.
    
    Returns
    -------
    latex_syntax : str
        A string containing the Latex syntax for the entered array.
    
    Examples
    --------
    >>> u = [4, 5, 6]
    >>> stm.tex_to_latex(M=u)
    \\left[\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right]
    >>> v = [[4, 5, 6]]
    >>> stm.tex_to_latex(M=v, brackets='(')
    \\left(\\begin{array}{rrr} 4 & 3 & 9\\end{array}\\right)
    >>> w = [[4, 5, 6]]
    >>> stm.tex_to_latex(M=w, brackets='')
    \\begin{array}{rrr} 4 & 3 & 9\\end{array}
    >>> P = [[4, 5, 6], [3, 7, 9]]
    >>> stm.tex_to_latex(M=P, align='center', brackets='[')
    \\left[\\begin{array}{ccc} 4 & 5 & 6 \\\\ 3 & 7 & 9\\end{array}\\right]
    ```
    """
    if is_function(M):
        raise ValueError(f"'M' cannot be a callable function.")
    
    if isinstance(M, DataFrame):
        try:
            M = M.round(decimals) if decimals >= 0 else M
        except (ValueError, TypeError, AttributeError):
            pass
        
        return dframe_to_array(
            df=M,
            outer_border=True
        )
    
    if isinstance(M, str):
        try:
            result = latex(sympify(M))
        except Exception:
            result = M
            
        return result
    
    if is_symexpr(M): # symbolic expression
        try:
            M = M.evalf(decimals) if decimals >= 0 else M
        except (TypeError, ValueError, AttributeError):
            pass
        result = latex(sympify(M))
    
    else: # if not symbolic
        try:
            if isinstance(M, (MatrixBase, ndarray)):
                M = M.tolist()
            elif isinstance(M, DataFrame) or (":" in str(M) and "{"):
                M = DataFrame(M)
                result = dframe_to_array(
                    df=M,
                    include_index=True,
                    header_color="gray",
                    first_col_align="l|",
                    outer_border=False,
                    inner_hlines=False,
                    inner_vlines=False
                )
                return result
                
            if isinstance(M, (tuple, list, ndarray)):
                try: # numpy
                    M = array(M)
                    if decimals >= 0:
                        M = M.astype(float).round(decimals)
                    M = Matrix(M)
                except: # sympy
                    M = Matrix(M)
                    if decimals >= 0:
                        M = M.evalf(decimals)
                ncols = M.shape[1]
                # convert to latex
                cols_align = align[0] * ncols
                result_latex = (
                    latex(M, mat_delim=brackets)
                    .replace(f"begin{{matrix}}", f"begin{{array}}{{{cols_align}}} ")
                    .replace(f"end{{matrix}}", f"end{{array}}")
                )
                if "{cc" in result_latex:  # large matrices
                    # replaces the first `ncols` occurances of `c` 
                    # with `r`, this avoids replacing any other `c` 
                    # that could be a valid element of the array
                    result = result_latex.replace("c" * ncols, "r" * ncols, ncols)
                else:
                    result = result_latex
            else:
                # if string, integer, float or other sympy numerical values
                try:
                    M = round(x=M, decimals=decimals)
                except (ValueError, TypeError):
                    pass
                if scalar_bracket:
                    result = f'\\left({latex(sympify(M))}\\right)'
                else:
                    result = latex(sympify(M))
        except (TypeError, ValueError, AttributeError) as e:
            if raise_exception:
                raise type(e)(str(e)) from e
            else:
                # values that cannot be converted to latex e.g html, latex
                # will be returned the way they were entered
                return str(M)
    
    try:
        ncols = Matrix(M).shape[1]
        cols_align = align[0] * ncols
        result = result\
        .replace(f"begin{{matrix}}", f"begin{{array}}{{{cols_align}}} ")\
        .replace(f"end{{matrix}}", f"end{{array}}")
    except (TypeError, ValueError, AttributeError, SympifyError):
        pass
    
    latex_syntax = str_remove_tzeros(result)\
        .replace("\\\\", " \\\\ ")\
        .replace("\\end", " \\end")
    
    return latex_syntax