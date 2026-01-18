from numpy import array, full, hstack, nan, ndarray
from sympy import flatten
from stemlab.core.htmlatex import tex_to_latex


def one_d_array_stack(data: list | tuple | ndarray, ncols: int = 10):
    
    A = array(flatten(data))
    k = len(A) % ncols
    if k != 0:
        B = full(ncols - len(A) % ncols, nan) 
        A = hstack((A, B))
    nrows = int(len(A) / ncols)
    A = A.reshape(nrows, ncols)
    
    return tex_to_latex(A, brackets="").replace(f"\\text{{NaN}}", "")