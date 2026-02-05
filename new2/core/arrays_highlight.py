from typing import Any, Literal, Sequence

from numpy import array, asarray, full, hstack, isin, nan, ndarray, vstack
from stemcore import str_data_join
from stemlab.core.htmlatex import tex_array_to_latex

from stemfard.core._strings import str_caption, str_color_math, str_remove_tzeros
from stemfard.core._type_aliases import IntegerSequenceArrayLike, SequenceArrayLike
from stemfard.core._enumerate import ColorCSS


def one_d_array_stack(
    data: SequenceArrayLike,
    ncols: int = 10,
    color_vals: Any = None,
    color_idx: IntegerSequenceArrayLike | None = None,
    first_only: bool = False
) -> str:
    """
    Convert a 1D array/list/tuple into a LaTeX matrix with specified
    number of columns. Supports highlighting by value or by index.

    Parameters
    ----------
    data : list | tuple | NDArray
        Input one-dimensional data.
    ncols : int, default=10
        Number of columns in the output matrix.
    color_vals : Any or list[Any], optional
        Value(s) to be highlighted.
    color_idx : int | (row, col) | list of these, optional
        Index/indices to highlight. Flat indices refer to the original
        1D array before reshaping.
    first_only : bool, default=False
        If True, highlight only the first matching value or index.

    Returns
    -------
    str
        LaTeX matrix string.
    """
    A = array(data, dtype=object).flatten()

    # Pad with NaNs to make divisible by ncols
    remainder = len(A) % ncols
    if remainder:
        A = hstack((A, full(ncols - remainder, nan, dtype=object)))

    nrows = len(A) // ncols
    A = A.reshape(nrows, ncols).astype(object)

    # ---- Highlight by index (highest priority) ----
    if color_idx is not None:
        if not isinstance(color_idx, (list, tuple, set)):
            color_idx = [color_idx]

        if first_only:
            color_idx = [next(iter(color_idx))]

        for idx in color_idx:
            if isinstance(idx, tuple):
                r, c = idx
            else:
                r, c = divmod(idx, ncols)

            if 0 <= r < nrows and 0 <= c < ncols and A[r, c] is not nan:
                A[r, c] = str_color_math(A[r, c], add_box=True)

    # ---- Highlight by value (fallback) ----
    elif color_vals is not None:
        if not isinstance(color_vals, (list, tuple, set, ndarray)):
            color_vals = [color_vals]

        mask = isin(A, color_vals)

        if first_only and mask.any():
            r, c = next(zip(*mask.nonzero()))
            A[r, c] = str_color_math(A[r, c], add_box=True)
        else:
            A[mask] = [str_color_math(x, add_box=True) for x in A[mask]]

    result = tex_array_to_latex(A, brackets="").replace("nan", "")
    
    return str_remove_tzeros(result)


def highlight_arrays_vals(
    arr: SequenceArrayLike,
    align: str = "r",
    brackets: Literal["[]", "()", "||"] | None = "[]",
    inline: bool = False,  # True -> use \smallmatrix
    # Coloring options
    color_indices: Sequence[tuple[int, int]] | None = None,
    color_rows: Sequence[int] | None = None,
    color_cols: Sequence[int] | None = None,
    color_diag: Literal["main", "minor"] | None = None,
    diag_offset: int | list[int] = 0,
    color: str = ColorCSS.COLORDEFAULT.value,
    color_map_indices: dict[tuple[int, int], str] | None = None,
    color_map_rows: dict[int, str] | None = None,
    color_map_cols: dict[int, str] | None = None,
    color_map_diag: dict[str, str] |None = None,
    color_map_diag_offsets: dict[int | tuple[str, int], str] | None = None,
    # Boxing options
    box_indices: Sequence[tuple[int, int]] = None,
    box_rows: Sequence[int] | None = None,
    box_cols: Sequence[int] | None = None,
) -> str:
    """
    Apply LaTeX styling to an array.

    Supports:
    - Coloring entries, rows, columns, diagonals with specific colors
    - Boxing entries, rows, columns

    Parameters
    ----------
    matrix : SequenceArrayLike
        Existing array-like object
    align : str
        LaTeX column alignment ('l', 'c', 'r').
    
    Coloring/Boxing options
    -----------------------
    color_indices, box_indices : list[(i,j)]
        Zero-based entries to color/box.
    color_rows, box_rows : list[int]
        Zero-based rows to color/box.
    color_cols, box_cols : list[int]
        Zero-based columns to color/box.
    color_diag : {"main","minor"}
        Color main or minor diagonal.
    diag_offset : int or list[int]
        Diagonal offset k (can be ±) or list of offsets e.g. [0, 1, -1]
    color : str
        Default LaTeX color name.
    color_map_indices : dict
        Dictionary mapping specific indices to colors: {(i,j): "color"}
    color_map_rows : dict  
        Dictionary mapping rows to colors: {row_i: "color"}
    color_map_cols : dict
        Dictionary mapping columns to colors: {col_j: "color"}
    color_map_diag : dict
        Dictionary mapping diagonals to colors: {"main": "color", "minor": "color"}
    color_map_diag_offsets : dict
        Dictionary mapping diagonal offsets to colors: 
        - {offset: "color"} for main diagonals
        - {(diag_type, offset): "color"} for specific diagonal types
        Examples: 
        {0: "red", 1: "blue", -1: "green"} 
        {("main", 0): "red", ("main", 1): "blue", ("minor", 0): "green"}

    Returns
    -------
    LaTeX string
    
    Examples
    --------
    >>> A = Matrix([[5, 6, -8], [2, 4, 1], [-3, 2, 7]])

    >>> # Color specific elements (zero-based indices)
    >>> latex_str = highlight_arrays_vals(A, color_indices=[(0,0), (1,1), (2,2)], color="red")
    >>> display(Math(latex_str))

    >>> # Color entire rows (zero-based indices)
    >>> latex_str = highlight_arrays_vals(A, color_rows=[0, 2], color="green")
    >>> display(Math(latex_str))

    >>> # Color entire columns (zero-based indices)
    >>> latex_str = highlight_arrays_vals(A, color_cols=[1], color="orange")
    >>> display(Math(latex_str))

    >>> # Color main diagonal
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", color="purple")
    >>> display(Math(latex_str))

    >>> # Color main diagonal with offset
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", diag_offset=1, color="brown")
    >>> display(Math(latex_str))

    >>> # Color minor diagonal
    >>> latex_str = highlight_arrays_vals(A, color_diag="minor", color="teal")
    >>> display(Math(latex_str))

    >>> # Boxing Examples:
    >>> # Box specific elements
    >>> latex_str = highlight_arrays_vals(A, box_indices=[(0,1), (1,0), (2,2)])
    >>> display(Math(latex_str))

    >>> # Box entire rows
    >>> latex_str = highlight_arrays_vals(A, box_rows=[1])
    >>> display(Math(latex_str))

    >>> # Box entire columns
    >>> latex_str = highlight_arrays_vals(A, box_cols=[0, 2])
    >>> display(Math(latex_str))

    >>> # Combined Styling Examples:
    >>> # Color diagonal and box specific elements
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", box_indices=[(0,1), (1,2)], color="blue")
    >>> display(Math(latex_str))

    >>> # Color rows and box columns
    >>> latex_str = highlight_arrays_vals(A, color_rows=[0, 1], box_cols=[2], color="red")
    >>> display(Math(latex_str))

    >>> # Color and box the same elements
    >>> latex_str = highlight_arrays_vals(A, color_indices=[(0,0), (2,2)], box_indices=[(0,0), (2,2)], color="green")
    >>> display(Math(latex_str))

    >>> # Larger Matrix Example:
    >>> B = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    >>> # Color main diagonal and box corners
    >>> latex_str = highlight_arrays_vals(B, color_diag="main", box_indices=[(0,0), (0,3), (3,0), (3,3)], color="magenta")
    >>> display(Math(latex_str))

    >>> # Non-Square Matrix Example:
    >>> C = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> latex_str = highlight_arrays_vals(C, color_rows=[0], box_cols=[1], color="cyan")
    >>> display(Math(latex_str))

    >>> # Matrix Operations with Styling:
    >>> A = Matrix([[5, 6, -8], [2, 4, 1], [-3, 2, 7]])

    >>> # Create augmented matrix and style it
    >>> augmented = A.row_join(Matrix([1, 0, 1]))
    >>> latex_str = highlight_arrays_vals(augmented, color_cols=[3], box_rows=[1], color="darkblue")
    >>> display(Math(latex_str))

    >>> # Different Alignment:
    >>> latex_str = highlight_arrays_vals(A, align='c', color_diag="main")
    >>> display(Math(latex_str))

    >>> # Color main diagonal with specific color
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "orange"})
    >>> display(Math(latex_str))

    >>> # Color minor diagonal with specific color
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"minor": "red"})
    >>> display(Math(latex_str))

    >>> # Color both diagonals with different colors
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "blue", "minor": "green"})
    >>> display(Math(latex_str))

    >>> # Color main diagonal with offset
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "purple"}, diag_offset=1)
    >>> display(Math(latex_str))

    >>> # Color specific elements with different colors
    >>> latex_str = highlight_arrays_vals(A, color_map_indices={(0,0): "red", (1,1): "green", (2,2): "blue"})
    >>> display(Math(latex_str))

    >>> # Color rows with different colors (using ("row", i) keys)
    >>> latex_str = highlight_arrays_vals(A, color_map_rows={("row", 0): "red", ("row", 2): "blue"})
    >>> display(Math(latex_str))

    >>> # Color columns with different colors (using ("col", j) keys)
    >>> latex_str = highlight_arrays_vals(A, color_map_cols={("col", 1): "orange", ("col", 2): "purple"})
    >>> display(Math(latex_str))

    >>> # Color diagonals with specific colors
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "red", "minor": "blue"})
    >>> display(Math(latex_str))

    >>> # Mixed coloring: rows, columns, and individual elements
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={
    ...     ("row", 0): "red",     # entire row 0
    ...     ("col", 1): "green",   # entire column 1  
    ...     (2,2): "blue",         # element (2,2)
    ...     "main": "orange"       # main diagonal
    ... })
    >>> display(Math(latex_str))

    >>> # Color main diagonal with specific color
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "orange"})
    >>> display(Math(latex_str))

    >>> # Color minor diagonal with specific color  
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"minor": "red"})
    >>> display(Math(latex_str))

    >>> # Color both diagonals with different colors
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "blue", "minor": "green"})
    >>> display(Math(latex_str))

    >>> # Color main diagonal with offset
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "purple"}, diag_offset=1)
    >>> display(Math(latex_str))

    >>> # Color main diagonal with multiple offsets
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", diag_offset=[0, 1], color="red")
    >>> display(Math(latex_str))

    >>> # Color main diagonal with negative offsets
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", diag_offset=[0, -1], color="blue")
    >>> display(Math(latex_str))

    >>> # Color multiple diagonals using color_map_diag with offsets
    >>> latex_str = highlight_arrays_vals(A, color_map_diag={"main": "orange"}, diag_offset=[0, 1, -1])
    >>> display(Math(latex_str))

    >>> # Mixed traditional and color_map with different offsets
    >>> latex_str = highlight_arrays_vals(A, 
        color_diag="main", 
        diag_offset=[0, 1], 
        color="red",
        color_map_diag={"minor": "blue"}
    )
    >>> display(Math(latex_str))

    # Single offset still works (backward compatibility)
    >>> latex_str = highlight_arrays_vals(A, color_diag="main", diag_offset=1, color="purple")
    >>> display(Math(latex_str))

    >>> # Color main diagonals with different colors for each offset
    >>> latex_str = highlight_arrays_vals(A, 
        color_map_diag_offsets={0: "red", 1: "blue", -1: "green"}
    )
    >>> display(Math(latex_str))

    >>> # Your original example - now working with different colors for each offset
    >>> latex_str = highlight_arrays_vals(A, 
        color_map_diag_offsets={0: "orange", 1: "blue", -1: "green"}
    )
    >>> display(Math(latex_str))

    >>> # Mixed main and minor diagonals
    >>> latex_str = highlight_arrays_vals(A,
        color_map_diag_offsets={
            ("main", 0): "red", 
            ("main", 1): "blue", 
            ("minor", 0): "green"
        }
    )
    >>> display(Math(latex_str))

    >>> latex_str = highlight_arrays_vals(
        A, 
        color_map_diag_offsets={
            ("main", 0): "brown",
            ("main", -1): "blue",
            ("main", 1): "red"
        }
    )
    >>> display(Math(latex_str))

    >>> latex_str = highlight_arrays_vals(
        A,
        color_map_diag_offsets={
            ("minor", 0): "brown",
            ("minor", -1): "blue",
            ("minor", 1): "red"
        }
    )
    >>> display(Math(latex_str))
    """
    # --- Convert input to SymPy Matrix ---
    A = arr
    if not isinstance(A, ndarray):
        try:
            A = array(A, dtype=object)
        except (TypeError, ValueError, AttributeError) as e:
            raise TypeError(
                f"Cannot convert input to an array: {type(A).__name__}"
            ) from e
            
    if A.ndim == 1:
        A = A.reshape(-1, 1)

    m, n = A.shape

    # --- Normalize sets and dicts ---
    color_indices = set(color_indices or [])
    if isinstance(color_rows, int):
        color_rows = [color_rows]
    color_rows = set(color_rows or [])
    if isinstance(color_rows, int):
        color_cols = [color_cols]
    color_cols = set(color_cols or [])
    box_indices = set(box_indices or [])
    box_rows = set(box_rows or [])
    box_cols = set(box_cols or [])

    color_map_indices = color_map_indices or {}
    color_map_rows = color_map_rows or {}
    color_map_cols = color_map_cols or {}
    color_map_diag = color_map_diag or {}
    color_map_diag_offsets = color_map_diag_offsets or {}

    diag_offsets = (
        [diag_offset] if isinstance(diag_offset, int) else diag_offset
    )

    # --- Helper: Determine color ---
    def get_color(i: int, j: int) -> str | None:
        if (i, j) in color_map_indices:
            return color_map_indices[(i, j)]
        if i in color_map_rows:
            return color_map_rows[i]
        if j in color_map_cols:
            return color_map_cols[j]
        for key, col in color_map_diag_offsets.items():
            if isinstance(key, tuple):
                diag_type, offset = key
                if diag_type == "main" and j - i == offset:
                    return col
                elif diag_type == "minor" and i + j == (n - 1) + offset:
                    return col
            elif isinstance(key, int):
                if j - i == key:
                    return col
        if "main" in color_map_diag:
            for off in diag_offsets:
                if j - i == off:
                    return color_map_diag["main"]
        if "minor" in color_map_diag:
            for off in diag_offsets:
                if i + j == (n - 1) + off:
                    return color_map_diag["minor"]
        if (i, j) in color_indices or i in color_rows or j in color_cols:
            return color
        if color_diag == "main" and any(j - i == off for off in diag_offsets):
            return color
        
        is_any = any(i + j == (n - 1) + off for off in diag_offsets)
        if color_diag == "minor" and is_any:
            return color
        return None

    # --- Helper: Apply color and box ---
    def style_entry(i: int, j: int, val: Any) -> str:
        # s = latex(val)
        s = str(val)
        c = get_color(i, j)
        boxed = (i, j) in box_indices or i in box_rows or j in box_cols
        if boxed:
            s = f"\\boxed{{{s}}}"
        if c:
            s = f"{{\\color{{{c}}}{{{s}}}}}"
        return s

    # --- Generate rows ---
    rows = [
        " & ".join(style_entry(i, j, A[i, j]) for j in range(n))
        for i in range(m)
    ]
    body = " \\\\ ".join(rows)

    # --- Array type ---
    if inline:
        array_str = f"\\begin{{smallmatrix}} {body} \\end{{smallmatrix}}"
    else:
        array_str = f"\\begin{{array}}{{{align * n}}} {body} \\end{{array}}"

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
        if brackets:
            array_str = f"\\left{brackets[0]}{array_str}\\right{brackets[1]}"
    
    return array_str.replace("& nan ", "& ")


def highlight_array_vals_arr(
    arr: SequenceArrayLike,
    index: list[str] | str | None = None,
    col_names: list[str] | str | None = None,
    heading: str = "",
    last_row: str | None = "",
    cap_label: str = "Table",
    cap_number: int | str = ...,
    cap_title: str = ...,
    align: str = "r",
    brackets: Literal["[]", "()", "||"] | None = "[]",
    inline: bool = False,  # True -> use \smallmatrix
    # Coloring options
    color_indices: Sequence[tuple[int, int]] | None = None,
    color_rows: Sequence[int] | None = None,
    color_cols: Sequence[int] | None = None,
    color_diag: Literal["main", "minor"] | None = None,
    diag_offset: int | list[int] = 0,
    color: str = ColorCSS.COLORDEFAULT.value,
    color_map_indices: dict[tuple[int, int], str] | None = None,
    color_map_rows: dict[int, str] | None = None,
    color_map_cols: dict[int, str] | None = None,
    color_map_diag: dict[str, str] |None = None,
    color_map_diag_offsets: dict[int | tuple[str, int], str] | None = None,
    # Boxing options
    box_indices: Sequence[tuple[int, int]] = None,
    box_rows: Sequence[int] | None = None,
    box_cols: Sequence[int] | None = None
    
):
    
    nrows, ncols = asarray(arr).shape
    
    if col_names is not None:
        if isinstance(col_names, str) and col_names == "auto":
            col_names = [f"C_{idx + 1}" for idx in range(ncols)]
        arr = vstack((col_names, arr))
        
    if index is not None:
        if isinstance(index, str) and index == "auto":
            if last_row is None:
                index = [heading] + list(range(1, nrows + 1))
            else:
                index = [heading] + list(range(1, nrows)) + [last_row]
        index = asarray(index).flatten().reshape(-1, 1)
        arr = hstack((index, arr))
        
    arr = highlight_arrays_vals(
        arr=arr,
        align=align,
        brackets=brackets,
        inline=inline,
        color_indices=color_indices,
        color_rows=color_rows,
        color_cols=color_cols,
        color_diag=color_diag,
        diag_offset=diag_offset,
        color=color,
        color_map_indices=color_map_indices,
        color_map_rows=color_map_rows,
        color_map_cols=color_map_cols,
        color_map_diag=color_map_diag,
        color_map_diag_offsets=color_map_diag_offsets,
        box_indices=box_indices,
        box_rows=box_rows,
        box_cols=box_cols
    )
    
    arr = (
        arr
        .replace("{rr", "{r|r", 1)
        .replace("r}", "r} \\hline ", 1)
        .replace("\\ ", "\\ \\hline ", 1)
        .replace("\\end", "\\\\ \\hline\\end", 1)
    )
    
    caption = str_caption(label=cap_label, num=cap_number, caption=cap_title)
    
    return (
        f"<div style='margin-top:15px;'></div>{caption} "
        f"\\( \\displaystyle {arr} \\)"
    )