from typing import Literal

from stemcore import str_data_join

from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math


def general_array(
    element: str ='a', 
    nrows: int | None = None, 
    ncols: int | None = None, 
    rows: str = "m", 
    cols: str = "n", 
    is_square: bool = False,
    is_transpose: bool = False,
    operation: Literal['+', '-', '\\times', '\\div', '^'] | None = None,
    constant: str = 'c'
) -> str:
    """
    Generate a general m x n array.
    """
    from sympy import zeros # should be here
    
    if is_square and nrows == ncols:
        rows = cols = "n"

    A = element
    
    if nrows is None:
        if is_transpose:
            result = (
                "\\left["
                f"\\begin{{array}}{{cccc}} "
                f"{A}_{{11}} & {A}_{{21}} & \\cdots & {A}_{{{rows}1}} \\\\ "
                f"{A}_{{12}} & {A}_{{22}} & \\cdots & {A}_{{{rows}2}} \\\\ "
                f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
                f"{A}_{{1{cols}}} & {A}_{{2{cols}}} & \\cdots & {A}_{{{rows}{cols}}} "
                f"\\end{{array}} "
                "\\right]"
            )
        else:
            result = (
                "\\left["
                f"\\begin{{array}}{{cccc}} "
                f"{A}_{{11}} & {A}_{{12}} & \\cdots & {A}_{{1{cols}}} \\\\ "
                f"{A}_{{21}} & {A}_{{22}} & \\cdots & {A}_{{2{cols}}} \\\\ "
                f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
                f"{A}_{{{rows}1}} & {A}_{{{rows}2}} & \\cdots & {A}_{{{rows}{cols}}} "
                f"\\end{{array}} "
                "\\right]"
            )
            
        if operation is not None:
            operations = ['+', '-', '\\times', '\\div', '^']
            if operation not in operations:
                raise ValueError(
                    "Expected 'operation to be one of "
                    f"{str_data_join(operations)}, got {operation}"
                )
            
            constant = str_color_math(constant)
            
            if operation == '\\times':
                result = result.replace(f'{A}_', f'{constant}\\:\\dot {A}_')
            else:
                result = result.replace(' &', f'{operation}{constant} &')
                result = result.replace(f'dots{operation}{constant} &', 'dots &')
                result = result.replace('\\:', f'{operation}{constant}')
    else:
        ncols = nrows if ncols is None else ncols
        M = zeros(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                if is_transpose:
                    M[row, col] = f'{element}{col + 1}{row + 1}'
                else:
                    M[row, col] = f'{element}{row + 1}{col + 1}'
        result = tex_to_latex(M)
    
    return result.replace("^", "\\: ^")


def general_array_ab(
    A: str = 'a', 
    B: str = 'b', 
    rows: str = "m", 
    cols: str = "n", 
    is_square = False,
    operation: Literal["+", "-", "\\times", "\\div", "^"] = "+", 
    color: bool = True
) -> str:
    """
    Perform an arithmetic operation on two m x n matrices and color 
    elements of the second matrix.
    """
    if is_square:
        rows = cols = "n"
        
    B_ij = [
        "",
        f"{B}_{{11}}",
        f"{B}_{{12}}",
        f"{B}_{{1{cols}}}",
        f"{B}_{{21}}",
        f"{B}_{{22}}",
        f"{B}_{{2{cols}}}",
        f"{B}_{{{rows}1}}",
        f"{B}_{{{rows}2}}",
        f"{B}_{{{rows}{cols}}}",
    ]
    if color:
        B_ij = [str_color_math(value=B_ij[i]) for i in range(len(B_ij))]
    
    cdots = "\\cdots"
    B11 = B_ij[1]
    B12 = B_ij[2]
    B1n = B_ij[3]
    B21 = B_ij[4]
    B22 = B_ij[5]
    B2n = B_ij[6]
    Bm1 = B_ij[7]
    Bm2 = B_ij[8]
    Bmn = B_ij[9]
    
    if operation == "^": operation = f"\\:{operation}"

    result = (
        f"\\left["
        f"\\begin{{array}}{{cccc}} "
        f"{A}_{{11}} {operation} {B11} & "
        f"{A}_{{12}} {operation} {B12} & "
        f"{cdots} & {A}_{{1{cols}}} {operation} {B1n} \\\\ "
        f"{A}_{{21}} {operation} {B21} & "
        f"{A}_{{22}} {operation} {B22} & "
        f"{cdots} & {A}_{{2{cols}}} {operation} {B2n} \\\\ "
        f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
        f"{A}_{{{rows}1}} {operation} {Bm1} & "
        f"{A}_{{{rows}2}} {operation} {Bm2} & "
        f"{cdots} & "
        f"{A}_{{{rows}{cols}}} {operation} {Bmn} "
        f"\\end{{array}} \\right]"
    )
        
    return result


def general_array_ak(
    A: str = 'a', 
    k: str = 'k', 
    rows: str = "m", 
    cols: str = "n", 
    is_square = False,
    operation: str = '+',
    color: bool = True
) -> str:
    """
    Perform an arithmetic operation on two m x n matrices and color 
    elements of the second matrix.
    """    
    if is_square:
        rows = cols = "n"

    if color:
        k = str_color_math('k')
    
    if operation == "^":
        operation = "\\:^"
        
    if operation == "\\times":
        result = (
            f"\\left["
            f"\\begin{{array}}{{cccc}} "
            f"{k} {operation} {A}_{{11}} & "
            f"{k} {operation} {A}_{{12}} & "
            f"\\cdots & "
            f"{k} {operation} {A}_{{1{cols}}} \\\\ "
            f"{k} {operation} {A}_{{21}} & "
            f"{k} {operation} {A}_{{22}} & "
            f"\\cdots & "
            f"{k} {operation} {A}_{{2{cols}}} \\\\ "
            f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
            f"{k} {operation} {A}_{{{rows}1}} & "
            f"{k} {operation} {A}_{{{rows}2}} & "
            f"\\cdots & "
            f"{k} {operation} {A}_{{{rows}{cols}}} "
            f"\\end{{array}} \\right]"
        )
    else:
        result = (
            f"\\left["
            f"\\begin{{array}}{{cccc}} "
            f"{A}_{{11}} {operation} {k} & "
            f"{A}_{{12}} {operation} {k} & "
            f"\\cdots & "
            f"{A}_{{1{cols}}} {operation} {k} \\\\ "
            f"{A}_{{21}} {operation} {k} & "
            f"{A}_{{22}} {operation} {k} & "
            f"\\cdots & "
            f"{A}_{{2{cols}}} {operation} {k} \\\\ "
            f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
            f"{A}_{{{rows}1}} {operation} {k} & "
            f"{A}_{{{rows}2}} {operation} {k} & "
            f"\\cdots & {A}_{{{rows}{cols}}} {operation} {k} "
            f"\\end{{array}} \\right]"
        )
    
    return result


def generate_a_times_b():
    arr_latex = (
        f"\\begin{{aligned}} C =&\, "
            f"\\left["
                f"\\begin{{array}}{{cccc}} "
                    f"a_{{11}} \\, {str_color_math('b_{{11}}')} + "
                    f"a_{{12}} \\, {str_color_math('b_{{21}}')} + \\cdots + "
                    f"a_{{1n}} \\, {str_color_math('b_{{m1}}')} & "
                    f"a_{{11}} \\, {str_color_math('b_{{12}}')} + "
                    f"a_{{12}} \\, {str_color_math('b_{{22}}')} + \\cdots + "
                    f"a_{{1n}} \\, {str_color_math('b_{{m2}}')} & \\cdots & "
                    f"a_{{11}} \\, {str_color_math('b_{{1n}}')} + "
                    f"a_{{12}} \\, {str_color_math('b_{{2n}}')} + \\cdots + "
                    f"a_{{1n}} \\, {str_color_math('b_{{mn}}')} \\\\ "
                    f"a_{{21}} \\, {str_color_math('b_{{11}}')} + "
                    f"a_{{22}} \\, {str_color_math('b_{{21}}')} + \\cdots + "
                    f"a_{{2n}} \\, {str_color_math('b_{{m1}}')} & "
                    f"a_{{21}} \\, {str_color_math('b_{{12}}')} + "
                    f"a_{{22}} \\, {str_color_math('b_{{22}}')} + \\cdots + "
                    f"a_{{2n}} \\, {str_color_math('b_{{m2}}')} & \\cdots & "
                    f"a_{{21}} \\, {str_color_math('b_{{1n}}')} + "
                    f"a_{{22}} \\, {str_color_math('b_{{2n}}')} + \\cdots + "
                    f"a_{{2n}} \\, {str_color_math('b_{{mn}}')} \\\\ "
                    f"\\vdots & \\vdots & \\ddots & \\vdots \\\\ "
                    f"a_{{m1}} \\, {str_color_math('b_{{11}}')} + "
                    f"a_{{m2}} \\, {str_color_math('b_{{21}}')} + \\cdots + "
                    f"a_{{mn}} \\, {str_color_math('b_{{m1}}')} & "
                    f"a_{{m1}} \\, {str_color_math('b_{{12}}')} + "
                    f"a_{{m2}} \\, {str_color_math('b_{{22}}')} + \\cdots + "
                    f"a_{{mn}} \\, {str_color_math('b_{{m2}}')} & \\cdots & "
                    f"a_{{m1}} \\, {str_color_math('b_{{1n}}')} + "
                    f"a_{{m2}} \\, {str_color_math('b_{{2n}}')} + \\cdots + "
                    f"a_{{mn}} \\, {str_color_math('b_{{mn}}')} "
                f"\\end{{array}} "
            f"\\right] "
        f"\\end{{aligned}}"
    )
    
    return arr_latex